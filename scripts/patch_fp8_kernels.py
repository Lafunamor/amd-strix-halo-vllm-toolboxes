import sys
from pathlib import Path

# FP8 (W8A8) Strix Halo kernel routing for vLLM.
# Kernels: https://github.com/leonyurko/vllm-fp8-strix-halo-kernel-support
# (the kernel modules — fp8_triton.py etc. — are placed on PYTHONPATH, e.g. /opt/fp8).
#
# gfx1151 has no native FP8 tensor support, so vLLM's compressed-tensors W8A8-FP8
# path falls back to a slow generic GEMM. This routes the scaled-mm call through
# leonyurko's fused FP8->bf16 Triton dequant-GEMM (fp8_triton.fp8_gemm).
#
# Deliberately a SEPARATE patch file (NOT patch_strix.py) so the FP8 work stays
# independent of the is_integrated memory PR (#66) and can be merged in any order.
# Surgical call-swap (not a file overlay): preserves vLLM's current scale-handling
# in apply_scaled_mm, only redirecting the GEMM call. No-ops if already applied.

SCALED_MM_FB = '''

def _scaled_mm_fb(A, B, *, out_dtype, scale_a, scale_b, bias=None):
    # gfx1151: fused FP8->bf16 Triton dequant GEMM (weights stay FP8 in HBM).
    # From leonyurko/vllm-fp8-strix-halo-kernel-support; fp8_triton on PYTHONPATH.
    # Falls back to a bf16 matmul + manual dequant if the kernel is unavailable.
    try:
        from fp8_triton import fp8_gemm
        return fp8_gemm(A.contiguous(), B, scale_a, scale_b, out_dtype, bias)
    except Exception:
        import torch as _t
        o = (A.to(_t.bfloat16) @ B.to(_t.bfloat16)).to(_t.float32)
        sa = scale_a.to(_t.float32).reshape(-1)
        sb = scale_b.to(_t.float32).reshape(-1)
        o = o * (sa if sa.numel() == 1 else sa.view(-1, 1))
        o = o * (sb if sb.numel() == 1 else sb.view(1, -1))
        if bias is not None:
            o = o + bias.to(_t.float32)
        return o.to(out_dtype)
'''


def patch_fp8():
    print("Applying Strix Halo FP8 Triton kernel routing to vLLM...")
    p = Path('vllm/model_executor/kernels/linear/scaled_mm/pytorch.py')
    if not p.exists():
        print(" -> FP8 patch: scaled_mm/pytorch.py not found; skipping (vLLM layout changed?)")
        return
    txt = p.read_text()
    if '_scaled_mm_fb' in txt:
        print(" -> FP8 patch: already applied; skipping")
        return
    n = txt.count('output = torch._scaled_mm(')
    if n == 0:
        print(" -> FP8 patch: no 'output = torch._scaled_mm(' call sites found; skipping")
        return
    # 1) redirect the apply_scaled_mm GEMM calls to the Triton kernel
    txt = txt.replace('output = torch._scaled_mm(', 'output = _scaled_mm_fb(')
    # 2) inject the routing function at module scope (called at runtime)
    txt = txt + SCALED_MM_FB
    p.write_text(txt)
    print(f" -> FP8 patch: routed {n} scaled_mm call site(s) to fp8_triton + injected _scaled_mm_fb")
    print("Successfully patched vLLM for Strix Halo FP8 kernels.")


if __name__ == '__main__':
    patch_fp8()
