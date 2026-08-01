"""Workaround for an amdsmi bug: VRAM total is the BIOS carveout on APUs.

On AMD APUs (e.g. gfx1151 / Strix Halo) amdsmi reports the small dedicated BIOS VRAM
carveout as the VRAM total, instead of the memory the GPU actually addresses:

    amdsmi VRAM total   ->     536,870,912   (0.50 GiB)   <- BIOS carveout
    amdsmi GTT total    -> 118,111,600,640   (110.00 GiB) <- unified pool
    KFD mem_banks/0     -> 118,111,600,640   (110.00 GiB) <- the driver's truth
    HIP totalGlobalMem  -> 118,111,600,640   (110.00 GiB)

sysfs `mem_info_vram_total` only exposes the carveout, and
`rsmi_dev_memory_total_get()` only consults KFD when that read fails or returns 0 --
on an APU it "succeeds" with the carveout, so KFD is never asked.

Consumers that size a memory budget from the VRAM total therefore see 0.5 GiB on a
110 GiB device. vLLM's ROCm platform does exactly this in
`RocmPlatform.get_device_total_memory()`.

The bug is in amdsmi, not in its consumers, so patch it here rather than in vLLM.
This mirrors the fix proposed upstream (prefer the KFD/GTT total when it exceeds the
sysfs VRAM total):

    https://github.com/ROCm/rocm-systems/pull/8419   (APU case proposed in a comment)
    https://github.com/ROCm/rocm-systems/issues/8476 (APUs aren't identifiable via amdsmi)

Delete this script once the fix ships in a ROCm release. It is idempotent and a no-op
if amdsmi already reports the correct total.
"""

import importlib.util
import os
import sys

# The workaround inserted just before `return total.value`. Shape-independent.
_WORKAROUND = """
    # --- Strix Halo / APU workaround (see scripts/patch_amdsmi.py) --------------
    # On APUs the VRAM total is only the small BIOS carveout, while the GPU
    # addresses the unified pool that the GTT total (and KFD mem_banks) reports.
    # Prefer the larger of the two, mirroring the fix proposed upstream in
    # ROCm/rocm-systems#8419. Remove once that ships in a ROCm release.
    if mem_type == AmdSmiMemoryType.VRAM:
        try:
            _gtt = ctypes.c_uint64()
            if (
                amdsmi_wrapper.amdsmi_get_gpu_memory_total(
                    processor_handle, AmdSmiMemoryType.GTT, ctypes.byref(_gtt)
                )
                == 0
                and _gtt.value > total.value
            ):
                return _gtt.value
        except Exception:  # never let the workaround break a working query
            pass
    # --- end workaround --------------------------------------------------------

    return total.value"""

# amdsmi_get_gpu_memory_total ends with a `_check_res(...)` on the C call then
# `return total.value`. The `_check_res(...)` block is formatted differently across
# ROCm releases (multi-line pre-7.14, single-line in 7.14+), so support both.
_CHECK_RES_MULTILINE = """    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_memory_total(
            processor_handle, mem_type, ctypes.byref(total))
    )
"""
_CHECK_RES_SINGLELINE = """    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_memory_total(processor_handle, mem_type, ctypes.byref(total))
    )
"""

# (anchor, replacement) pairs, tried in order.
PATCHES = [
    (block + "\n    return total.value", block + _WORKAROUND)
    for block in (_CHECK_RES_MULTILINE, _CHECK_RES_SINGLELINE)
]

MARKER = "Strix Halo / APU workaround"


def main() -> int:
    spec = importlib.util.find_spec("amdsmi")
    if spec is None or not spec.origin:
        print(" -> amdsmi not installed; nothing to patch", file=sys.stderr)
        return 1

    path = os.path.join(os.path.dirname(spec.origin), "amdsmi_interface.py")
    txt = open(path).read()

    if MARKER in txt:
        print(" -> amdsmi already patched (idempotent no-op)")
        return 0

    for anchor, replacement in PATCHES:
        if anchor in txt:
            open(path, "w").write(txt.replace(anchor, replacement, 1))
            print(" -> Patched amdsmi_get_gpu_memory_total (APU: prefer GTT over VRAM carveout)")
            return 0

    # No known shape matched: amdsmi changed again, or the upstream fix has landed.
    # Fail loudly rather than silently ship an image that OOMs on APUs at runtime.
    print(
        " -> amdsmi_get_gpu_memory_total does not match any known shape; "
        "the VRAM-carveout workaround was NOT applied. If ROCm/rocm-systems#8419 has "
        "landed (VRAM total now reports the full pool) this script is obsolete -- verify "
        "and drop it; otherwise the anchor needs updating for the new amdsmi shape.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
