import sys
import re
import glob
from pathlib import Path
from unittest.mock import MagicMock

def patch_vllm():
    print("Applying Strix Halo patches to vLLM...")

    # Patch 1: vllm/platforms/__init__.py
    p_init = Path('vllm/platforms/__init__.py')
    if p_init.exists():
        txt = p_init.read_text()
        txt = txt.replace('import amdsmi', '# import amdsmi')
        txt = re.sub(r'is_rocm = .*', 'is_rocm = True', txt)
        txt = re.sub(r'if len\(amdsmi\.amdsmi_get_processor_handles\(\)\) > 0:', 'if True:', txt)
        txt = txt.replace('amdsmi.amdsmi_init()', 'pass')
        txt = txt.replace('amdsmi.amdsmi_shut_down()', 'pass')
        p_init.write_text(txt)
        print(" -> Patched vllm/platforms/__init__.py")

    # Patch 2: vllm/platforms/rocm.py
    p_rocm = Path('vllm/platforms/rocm.py')
    if p_rocm.exists():
        txt = p_rocm.read_text()
        header = 'import sys\nfrom unittest.mock import MagicMock\nsys.modules["amdsmi"] = MagicMock()\n'
        txt = header + txt
        txt = re.sub(r'device_type = .*', 'device_type = "rocm"', txt)
        txt = re.sub(r'device_name = .*', 'device_name = "gfx1151"', txt)
        txt += '\n    def get_device_name(self, device_id: int = 0) -> str:\n        return "AMD-gfx1151"\n'
        p_rocm.write_text(txt)
        print(" -> Patched vllm/platforms/rocm.py")

    # Patch 3: CUDA/HIP Macro injections for PyTorch Nightly Compatibility
    macro_def = """
#ifndef C10_HIP_CHECK
#define C10_HIP_CHECK(error) do { if (error != hipSuccess) { abort(); } } while(0)
#endif
#ifndef C10_CUDA_CHECK
#define C10_CUDA_CHECK(error) do { if (error != cudaSuccess) { abort(); } } while(0)
#endif
#define getCurrentHIPStreamMasqueradingAsCUDA getCurrentCUDAStream
"""
    # Apply to all .cu and .hip files in csrc
    csrc_files = glob.glob('csrc/**/*.cu', recursive=True) + glob.glob('csrc/**/*.hip', recursive=True)
    patched_csrc_count = 0
    for f in csrc_files:
        p_f = Path(f)
        if p_f.exists():
            txt = p_f.read_text()
            # Only prepend if not already patched to avoid duplicate macros
            if "getCurrentHIPStreamMasqueradingAsCUDA getCurrentCUDAStream" not in txt:
                p_f.write_text(macro_def + '\n' + txt)
                patched_csrc_count += 1
    
    print(f" -> Patched {patched_csrc_count} C/C++ source files with missing macros.")
    print("Successfully patched vLLM for Strix Halo.")

if __name__ == "__main__":
    patch_vllm()
