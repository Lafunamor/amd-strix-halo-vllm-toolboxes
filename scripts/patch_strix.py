import sys
import re
import glob
from pathlib import Path
from unittest.mock import MagicMock

def patch_vllm():
    print("Applying Strix Halo patches to vLLM...")

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
