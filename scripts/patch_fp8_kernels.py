import sys
from pathlib import Path

# FP8 (W8A8) Strix Halo kernel registration for vLLM (EXPERIMENTAL, opt-in).
# Kernels: https://github.com/leonyurko/vllm-fp8-strix-halo-kernel-support
#
# gfx1151 has no native FP8 tensor support, so every stock ROCm FP8 scaled-mm
# kernel rejects it and per-tensor/per-channel FP8 checkpoints fail to load
# ("Failed to find a kernel that can implement the ScaledMM linear layer").
# The previous version of this script swapped torch._scaled_mm call sites, but
# the kernels owning those sites never pass the gfx1151 support check, so it
# never ran.
#
# This registers StrixFP8HybridScaledMMLinearKernel (gfx1x_fp8_hybrid.py, copied
# into vllm/model_executor/kernels/linear/scaled_mm/ by the Dockerfile) at the
# head of the ROCm FP8 kernel list. It only does so when VLLM_STRIX_FP8_TRITON=1
# is set at serve time, so the default path stays upstream.
#
# Run from the vLLM source root. No-ops if already applied; fails the build if
# the anchor is gone so an inert integration cannot ship silently.

MARKER = "# PATCHED: gfx1x Strix FP8 hybrid kernel"
REGISTRATION = f'''

{MARKER} (opt-in: VLLM_STRIX_FP8_TRITON=1)
from vllm.model_executor.kernels.linear.scaled_mm.gfx1x_fp8_hybrid import (  # noqa: E402
    register as _register_strix_fp8_hybrid,
)

_register_strix_fp8_hybrid()
'''


def patch_fp8():
    print("Registering the Strix Halo FP8 hybrid kernel (opt-in via VLLM_STRIX_FP8_TRITON)...")
    kernel = Path("vllm/model_executor/kernels/linear/scaled_mm/gfx1x_fp8_hybrid.py")
    if not kernel.exists():
        sys.exit(f"ERROR: {kernel} missing; copy scripts/gfx1x_fp8_hybrid.py first.")
    p = Path("vllm/model_executor/kernels/linear/__init__.py")
    txt = p.read_text()
    if MARKER in txt:
        print(" -> FP8 patch: already applied; skipping")
        return
    if "_POSSIBLE_FP8_KERNELS" not in txt:
        sys.exit(f"ERROR: _POSSIBLE_FP8_KERNELS not found in {p}; vLLM layout changed.")
    p.write_text(txt.rstrip("\n") + "\n" + REGISTRATION)
    print(f" -> FP8 patch: registration appended to {p}")


if __name__ == "__main__":
    patch_fp8()
