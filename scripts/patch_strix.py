import sys
import re
import glob
import site
from pathlib import Path

def patch_vllm():
    print("Applying Strix Halo patches to vLLM (ai-notes modernization)...")

    # Patch 1: vllm/platforms/__init__.py (Device detection bypass for APUs + amdsmi import prepend)
    p_init = Path('vllm/platforms/__init__.py')
    if p_init.exists():
        txt = p_init.read_text()
        
        # 1A. Prepend amdsmi import
        if "try:\n    import amdsmi" not in txt[:300]:
            header = ("# PATCHED: amdsmi import order (must be before torch)\n"
                      "try:\n"
                      "    import amdsmi  # noqa: F401\n"
                      "except ImportError:\n"
                      "    pass\n\n")
            txt = header + txt

        # 1B. Force is_rocm=True bypassing empty processor handles on APUs
        txt = txt.replace('is_rocm = False\n    logger.debug("Checking if ROCm platform is available.")', 'is_rocm = True\n    logger.debug("Checking if ROCm platform is available.")')
        # Just to catch variations
        txt = txt.replace('if len(amdsmi.amdsmi_get_processor_handles()) > 0:', 'if True:')
        
        p_init.write_text(txt)
        print(" -> Patched vllm/platforms/__init__.py (Device detection & amdsmi import prepended)")

    # Patch 1.5: rocm.py (Force device type to gfx1151 for APUs)
    p_rocm_plat = Path('vllm/platforms/rocm.py')
    if p_rocm_plat.exists():
        txt = p_rocm_plat.read_text()
        if 'def _get_gcn_arch() -> str:\n    return "gfx1151"' not in txt:
            txt = txt.replace('def _get_gcn_arch() -> str:', 'def _get_gcn_arch() -> str:\n    return "gfx1151"\n\ndef _old_get_gcn_arch() -> str:')
            txt = re.sub(r'device_type = .*', 'device_type = "rocm"', txt)
            if 'def get_device_name(self' in txt:
                txt = re.sub(r'def get_device_name\(self, device_id: int = 0\) -> str:.*?return [^\n]+', 'def get_device_name(self, device_id: int = 0) -> str:\n        return "AMD-gfx1151"', txt, flags=re.DOTALL)
            p_rocm_plat.write_text(txt)
            print(" -> Patched vllm/platforms/rocm.py (Forced gfx1151 arch detection)")

    # Patch 2: _aiter_ops.py (Enable AITER on gfx1x, disable FP8 linear)
    p_aiter = Path('vllm/_aiter_ops.py')
    if p_aiter.exists():
        txt = p_aiter.read_text()
        # Extend is_aiter_found_and_supported
        if "or current_platform.on_gfx1x()" not in txt:
            txt = txt.replace("current_platform.on_gfx9()", "(current_platform.on_gfx9() or current_platform.on_gfx1x())")
        # Disable FP8 linear
        if "is_linear_fp8_enabled" in txt:
            txt = re.sub(
                r'(def is_linear_fp8_enabled.*?:\n\s+return) (.*?)\n', 
                r'\1 \2 and not current_platform.on_gfx1x()\n', 
                txt, count=1, flags=re.DOTALL
            )
        p_aiter.write_text(txt)
        print(" -> Patched vllm/_aiter_ops.py (gfx1x support & FP8 linear disabled)")

    # Patch 3: rocm_aiter_fa.py
    p_fa = Path('vllm/v1/attention/backends/rocm_aiter_fa.py')
    if p_fa.exists():
        txt = p_fa.read_text()
        if "or current_platform.on_gfx1x()" not in txt:
            txt = txt.replace("current_platform.on_gfx9()", "(current_platform.on_gfx9() or current_platform.on_gfx1x())")
        p_fa.write_text(txt)
        print(" -> Patched vllm/v1/attention/backends/rocm_aiter_fa.py (gfx1x support)")

    # Patch 4: Skip encoder cache profiling (MIOpen hangs on gfx1151)
    gpu_runner_files = glob.glob('vllm/**/gpu_model_runner.py', recursive=True)
    for f in gpu_runner_files:
        p = Path(f)
        txt = p.read_text()
        if '_get_mm_dummy_batch' in txt and '#PATCHED' not in txt:
            lines = txt.split('\n')
            in_block = False
            patched_lines = []
            for line in lines:
                if '_get_mm_dummy_batch' in line and 'batched_dummy_mm_inputs' in line:
                    in_block = True
                if in_block:
                    patched_lines.append('#PATCHED# ' + line)
                    if 'encoder_cache[' in line:  # Broader catch
                        in_block = False
                else:
                    patched_lines.append(line)
            p.write_text('\n'.join(patched_lines))
            print(f" -> Patched encoder profiling in {f}")

    # Patch 5: custom_ops RMSNorm block on gfx1x (Full CUDA Graph capture)
    p_rocm = Path('vllm/platforms/rocm.py')
    if p_rocm.exists():
        txt = p_rocm.read_text()
        if "is_aiter_found_and_supported() and not self.on_gfx1x():" not in txt:
            txt = txt.replace(
                "if is_aiter_found_and_supported():\n            custom_ops.append(\"+rms_norm\")",
                "if is_aiter_found_and_supported() and not self.on_gfx1x():\n            custom_ops.append(\"+rms_norm\")"
            )
        p_rocm.write_text(txt)
        print(" -> Patched vllm/platforms/rocm.py (custom_ops rms_norm bypassed on gfx1x)")

    # Patch 6: Triton backend AttrsDescriptor repr
    for sp in site.getsitepackages():
        triton_compiler = Path(sp) / "triton/backends/compiler.py"
        if triton_compiler.exists():
            txt = triton_compiler.read_text()
            if "def __repr__(self):" not in txt:
                txt = txt.replace(
                    "def to_dict(self):", 
                    "def __repr__(self):\n        return f'AttrsDescriptor.from_dict({self.to_dict()!r})'\n\n    def to_dict(self):"
                )
                triton_compiler.write_text(txt)
                print(f" -> Patched {triton_compiler} (AttrsDescriptor repr)")

    # Patch 7: aiter JIT path fix — aiter builds .so files into ~/.aiter/jit/
    # but importlib.import_module("aiter.jit.<module>") only looks in the
    # installed package directory. Fix by adding the JIT cache to __path__.
    for sp in site.getsitepackages():
        aiter_jit_init = Path(sp) / "aiter/jit/__init__.py"
        if aiter_jit_init.exists():
            txt = aiter_jit_init.read_text()
            if "# PATCHED: JIT cache path" not in txt:
                jit_path_fix = '''
# PATCHED: JIT cache path for Strix Halo
# aiter's JIT compiles .so modules into ~/.aiter/jit/ but importlib looks
# in the installed package directory. Add the JIT cache to __path__.
import os as _os
_jit_cache = _os.path.join(_os.path.expanduser("~"), ".aiter", "jit")
if _os.path.isdir(_jit_cache) and _jit_cache not in __path__:
    __path__.append(_jit_cache)
'''
                txt += jit_path_fix
                aiter_jit_init.write_text(txt)
                print(f" -> Patched {aiter_jit_init} (JIT cache added to __path__)")

    # Patch 8: flash_attn_interface.py — make aiter import soft as safety net.
    # If aiter JIT fails for any reason, flash_attn should still load (TRITON_ATTN works).
    # ROCM_ATTN will also work when aiter JIT succeeds (patch 7 fixes the path).
    for sp in site.getsitepackages():
        for fa_egg in Path(sp).glob("flash_attn*.egg"):
            fa_iface = fa_egg / "flash_attn/flash_attn_interface.py"
            if fa_iface.exists():
                txt = fa_iface.read_text()
                hard_import = "from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2 as flash_attn_gpu"
                if hard_import in txt and "except (ImportError" not in txt:
                    soft_import = (
                        "try:\n"
                        "    from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2 as flash_attn_gpu\n"
                        "except (ImportError, KeyError, ModuleNotFoundError):\n"
                        "    flash_attn_gpu = None"
                    )
                    txt = txt.replace(hard_import, soft_import)
                    fa_iface.write_text(txt)
                    print(f" -> Patched {fa_iface} (aiter import made resilient)")
        # Also check non-egg installs
        fa_iface = Path(sp) / "flash_attn/flash_attn_interface.py"
        if fa_iface.exists():
            txt = fa_iface.read_text()
            hard_import = "from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2 as flash_attn_gpu"
            if hard_import in txt and "except (ImportError" not in txt:
                soft_import = (
                    "try:\n"
                    "    from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2 as flash_attn_gpu\n"
                    "except (ImportError, KeyError, ModuleNotFoundError):\n"
                    "    flash_attn_gpu = None"
                )
                txt = txt.replace(hard_import, soft_import)
                fa_iface.write_text(txt)
                print(f" -> Patched {fa_iface} (aiter import made resilient)")

    print("Successfully patched vLLM/Environment for Strix Halo.")

if __name__ == "__main__":
    patch_vllm()
