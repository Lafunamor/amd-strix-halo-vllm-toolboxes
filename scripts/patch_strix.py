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
        txt = txt.replace('def _get_gcn_arch() -> str:', 'def _get_gcn_arch() -> str:\n    return "gfx1151"\n\ndef _old_get_gcn_arch() -> str:')
        txt = re.sub(r'device_type = .*', 'device_type = "rocm"', txt)
        txt = re.sub(r'device_name = .*', 'device_name = "gfx1151"', txt)
        txt += '\n    def get_device_name(self, device_id: int = 0) -> str:\n        return "AMD-gfx1151"\n'
        p_rocm.write_text(txt)
        print(" -> Patched vllm/platforms/rocm.py")

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
                    if 'encoder_cache[f"tmp_{i}"]' in line:
                        in_block = False
                else:
                    patched_lines.append(line)
            p.write_text('\n'.join(patched_lines))
            print(f" -> Patched encoder profiling in {f}")

    print("Successfully patched vLLM for Strix Halo.")

if __name__ == "__main__":
    patch_vllm()
