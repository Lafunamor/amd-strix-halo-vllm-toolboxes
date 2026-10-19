# SPDX-License-Identifier: Apache-2.0
"""Opt-in hybrid W8A8-FP8 linear kernel for gfx1151 (Strix Halo).

gfx1151 has no native FP8 matrix cores, so no stock ROCm FP8 kernel accepts it
and per-tensor/per-channel FP8 checkpoints fail to load. With
VLLM_STRIX_FP8_TRITON=1 this kernel is registered ahead of the stock ROCm list
and serves those layers two ways:

* one-row calls (single-stream decode) run @leonyurko's FP8 dequant-GEMV from
  leonyurko/vllm-fp8-strix-halo-kernel-support (fp8_triton.py). It reads the
  FP8 weight, so it moves half the bytes of a BF16 GEMV;
* every other call (prefill, batched decode) runs the original BF16
  activation against a BF16 copy of the weight through vLLM's ROCm unquantized
  GEMM. The kernel's own multi-row GEMM is 2-3x slower than BF16 here.

The BF16 copy is built in process_weights_after_loading, before vLLM measures
memory, so the KV-cache sizing sees it. Weight memory is FP8 + BF16, about
1.5x the BF16 checkpoint.

Both paths sit inside one torch custom op so torch.compile and CUDA graphs see
a single opaque call. Do not serve with --enforce-eager: eager mode makes the
stock torch kernels pad activations to 17 rows; this kernel requests no
padding, but compiled mode is still what measured 1.7x BF16 decode.

The fp8_triton module is not part of this repository (PolyForm Noncommercial
1.0.0). Images fetch it at a pinned commit; VLLM_STRIX_FP8_KERNELS_DIR points
at the checkout when it is not already importable.
"""

import os
import sys

import torch

from vllm.platforms import current_platform

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)

_ENABLED = os.environ.get("VLLM_STRIX_FP8_TRITON") == "1"
_DEFAULT_KERNELS_DIR = os.path.join(sys.prefix, "share", "strix-fp8")
_DEQUANT_ROWS = 4096


def _import_fp8_gemm():
    try:
        from fp8_triton import fp8_gemm
    except ImportError:
        kernels_dir = os.environ.get(
            "VLLM_STRIX_FP8_KERNELS_DIR", _DEFAULT_KERNELS_DIR
        )
        if kernels_dir not in sys.path:
            sys.path.append(kernels_dir)
        from fp8_triton import fp8_gemm
    return fp8_gemm


@torch.library.custom_op("strix::fp8_hybrid_linear", mutates_args=())
def _fp8_hybrid_linear(
    x: torch.Tensor,
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bf16: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if x.shape[0] == 1:
        # fp8_gemm writes its GEMV result into a buffer it reuses per N.
        out = _import_fp8_gemm()(
            x_q.contiguous(), weight, x_scale, weight_scale, out_dtype, bias
        )
        return out.clone() if out_dtype == torch.float32 else out

    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

    out = rocm_unquantized_gemm_impl(x.to(weight_bf16.dtype), weight_bf16, bias)
    return out.to(out_dtype)


@_fp8_hybrid_linear.register_fake
def _(x, x_q, x_scale, weight, weight_scale, weight_bf16, bias, out_dtype):
    return x.new_empty((x.shape[0], weight.shape[1]), dtype=out_dtype)


class StrixFP8HybridScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not _ENABLED:
            return False, "requires VLLM_STRIX_FP8_TRITON=1."
        if not current_platform.is_rocm():
            return False, "requires ROCm."
        from vllm.platforms.rocm import on_gfx1x

        if not on_gfx1x():
            return False, "requires a gfx1x GPU."
        try:
            _import_fp8_gemm()
        except ImportError:
            return False, (
                "requires fp8_triton from leonyurko/vllm-fp8-strix-halo-kernel-"
                "support (set VLLM_STRIX_FP8_KERNELS_DIR)."
            )
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.activation_quant_key.scale.group_shape.is_per_group():
            return False, "does not support block-scaled activations."
        if c.weight_quant_key.scale.group_shape.is_per_group():
            return False, "does not support block-scaled weights."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w, w_s, _, _ = self._get_layer_params(layer)
        # The scheme stores the FP8 weight as a (K, N) view of an (N, K) tensor.
        weight_nk = w.t()
        scale = w_s.float().reshape(-1)
        weight_bf16 = torch.empty(
            weight_nk.shape, dtype=torch.bfloat16, device=weight_nk.device
        )
        for start in range(0, weight_nk.shape[0], _DEQUANT_ROWS):
            stop = start + _DEQUANT_ROWS
            rows = weight_nk[start:stop].float()
            rows *= scale if scale.numel() == 1 else scale[start:stop, None]
            weight_bf16[start:stop] = rows
        layer.weight_strix_bf16 = torch.nn.Parameter(weight_bf16, requires_grad=False)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w, w_s, x_s, x_s_ub = self._get_layer_params(layer)
        x_2d = x.view(-1, x.shape[-1])
        out_dtype = self.config.out_dtype or x.dtype
        # Cheap next to either GEMM; quantizing unconditionally keeps the row
        # count out of the traced graph.
        x_q, x_scale = self.quant_fp8(x_2d, x_s, x_s_ub)
        out = torch.ops.strix.fp8_hybrid_linear(
            x_2d, x_q, x_scale, w, w_s, layer.weight_strix_bf16, bias, out_dtype
        )
        return out.view(*x.shape[:-1], w.shape[1])

    def apply_scaled_mm(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("apply_weights dispatches directly.")


def register() -> None:
    """Put the kernel ahead of the stock ROCm FP8 kernels when opted in."""
    if not _ENABLED or not current_platform.is_rocm():
        return
    from vllm.model_executor.kernels.linear import _POSSIBLE_FP8_KERNELS
    from vllm.platforms import PlatformEnum

    kernels = _POSSIBLE_FP8_KERNELS.setdefault(PlatformEnum.ROCM, [])
    if StrixFP8HybridScaledMMLinearKernel not in kernels:
        kernels.insert(0, StrixFP8HybridScaledMMLinearKernel)
