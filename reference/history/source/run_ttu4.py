#!/usr/bin/env python3
"""Phase-4 runner: everything TTU-visible lives in a low-4GB VMM mapping
(paper-faithful: "requesting a base address in the low 4 GB" because TTU
dereferences some references as 32-bit words).

Layout at VA 0x4000000 (16 MB, one cuMemCreate + cuMemMap):
    +0x000000  AS image (patched 0xD0 self-pointer = 0x4000000)
    +0x002000  scratch (result buffer candidate)

Usage: python3 run_ttu4.py <cubin> <as_dump.bin> <hit|miss> <base|node>
"""
import ctypes, struct, sys
from pathlib import Path

cuda = ctypes.CDLL("libcuda.so.1")
for fn, args in [
    ("cuInit", [ctypes.c_int]),
    ("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int]),
    ("cuCtxCreate_v2", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]),
    ("cuModuleLoadData", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]),
    ("cuModuleGetFunction", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]),
    ("cuMemCreate", [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint64]),
    ("cuMemAddressReserve", [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t,
                             ctypes.c_size_t, ctypes.c_uint64, ctypes.c_uint64]),
    ("cuMemMap", [ctypes.c_uint64, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint64, ctypes.c_uint64]),
    ("cuMemSetAccess", [ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]),
    ("cuMemcpyHtoD_v2", [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]),
    ("cuMemcpyDtoH_v2", [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]),
    ("cuLaunchKernel", [ctypes.c_void_p] + [ctypes.c_uint]*6 + [ctypes.c_uint,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]),
    ("cuCtxSynchronize", []),
    ("cuGetErrorString", [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]),
]:
    getattr(cuda, fn).argtypes = args
    getattr(cuda, fn).restype = ctypes.c_int

def err_str(rc):
    s = ctypes.c_char_p(); cuda.cuGetErrorString(rc, ctypes.byref(s))
    return (s.value or b"?").decode()

def check(rc, what):
    if rc != 0: print(f"FAIL {what}: rc={rc} ({err_str(rc)})"); sys.exit(2)

class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]

class AllocFlags(ctypes.Structure):
    _fields_ = [("compressionType", ctypes.c_ubyte), ("gpuDirectRDMACapable", ctypes.c_ubyte),
                ("usage", ctypes.c_ushort), ("reserved", ctypes.c_ubyte * 4)]

class CUmemAllocationProp(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("requestedHandleTypes", ctypes.c_int),
                ("location", CUmemLocation), ("win32HandleMetaData", ctypes.c_void_p),
                ("allocFlags", AllocFlags)]

class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", CUmemLocation), ("flags", ctypes.c_int)]

RAYS = {"hit":  (0.25, 0.25, -5.0, 0.0, 0.0, 1.0),
        "miss": (50.0, 2.0,  -5.0, 0.0, 0.0, 1.0)}
def f32(x): return struct.unpack('<I', struct.pack('<f', x))[0]

BASE_VA   = 0x4000000
TOTAL     = 16 * 1024 * 1024
AS_OFF    = 0x0
SCR_OFF   = 0x2000

def main():
    _, cubin_path, as_path, rayname, rootmode = sys.argv
    cubin = Path(cubin_path).read_bytes()
    asimg = bytearray(Path(as_path).read_bytes())
    ox, oy, oz, dx, dy, dz = RAYS[rayname]

    cuda.cuInit(0)
    dev = ctypes.c_int(); check(cuda.cuDeviceGet(ctypes.byref(dev), 0), "dev")
    ctx = ctypes.c_void_p(); check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "ctx")

    prop = CUmemAllocationProp()
    prop.type = 1                    # CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = 0    # NONE
    prop.location.type = 1           # CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = 0             # device 0 (CUDA_VISIBLE_DEVICES remaps)
    handle = ctypes.c_uint64()
    check(cuda.cuMemCreate(ctypes.byref(handle), TOTAL, ctypes.byref(prop), 0), "cuMemCreate")

    va = ctypes.c_uint64()
    check(cuda.cuMemAddressReserve(ctypes.byref(va), TOTAL, 0, BASE_VA, 0), "reserve")
    print(f"reserved VA = 0x{va.value:x} (requested 0x{BASE_VA:x})")
    check(cuda.cuMemMap(va.value, TOTAL, 0, handle.value, 0), "map")
    adesc = CUmemAccessDesc(prop.location, 3)  # READ|WRITE
    check(cuda.cuMemSetAccess(va.value, TOTAL, ctypes.byref(adesc), 1), "setaccess")

    base = va.value
    struct.pack_into('<Q', asimg, 0xD0, base)          # AS self-pointer -> base
    check(cuda.cuMemcpyHtoD_v2(base + AS_OFF, bytes(asimg), len(asimg)), "upAS")
    check(cuda.cuMemcpyHtoD_v2(base + SCR_OFF, bytes(8192), 8192), "zero scratch")

    if rootmode == "base":
        root = base
    elif rootmode == "node":
        root = base + 0x180
    else:
        root = base + int(rootmode, 0)
    cfg = struct.pack('<5Q', root,
                      (f32(oy) << 32) | f32(ox),
                      base + SCR_OFF,               # [0x000]-slot payload via ozz pair
                      (f32(dy) << 32) | f32(dx),
                      (f32(dz) << 32) | f32(dz))
    cfg_va = ctypes.c_uint64(); check(cuda.cuMemAlloc_v2(ctypes.byref(cfg_va), len(cfg)), "cfg")
    check(cuda.cuMemcpyHtoD_v2(cfg_va.value, cfg, len(cfg)), "upcfg")
    out = ctypes.create_string_buffer(64); out2 = ctypes.create_string_buffer(64)
    out_va = ctypes.c_uint64(); check(cuda.cuMemAlloc_v2(ctypes.byref(out_va), 64), "out")
    out2_va = ctypes.c_uint64(); check(cuda.cuMemAlloc_v2(ctypes.byref(out2_va), 64), "out2")

    mod = ctypes.c_void_p(); check(cuda.cuModuleLoadData(ctypes.byref(mod), cubin), "mod")
    fn = ctypes.c_void_p(); check(cuda.cuModuleGetFunction(ctypes.byref(fn), mod, b"ttu_probe2"), "fn")
    a0, a1, a2 = (ctypes.c_uint64(out_va.value), ctypes.c_uint64(cfg_va.value), ctypes.c_uint64(out2_va.value))
    params = (ctypes.c_void_p*3)(ctypes.cast(ctypes.byref(a0), ctypes.c_void_p),
                                 ctypes.cast(ctypes.byref(a1), ctypes.c_void_p),
                                 ctypes.cast(ctypes.byref(a2), ctypes.c_void_p))
    rc = cuda.cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, None, params, None)
    if rc: print(f"launch fail: {err_str(rc)}"); return 2
    rc = cuda.cuCtxSynchronize()
    if rc: print(f"FAULT rc={rc} ({err_str(rc)})"); return 1

    check(cuda.cuMemcpyDtoH_v2(out, out_va.value, 64), "dtoh1")
    check(cuda.cuMemcpyDtoH_v2(out2, out2_va.value, 64), "dtoh2")
    q0 = struct.unpack('<Q', out[:8])[0]
    q1 = struct.unpack('<Q', out2[:8])[0]
    print(f"[{rayname} root={rootmode}] q0=0x{q0:016x} q1=0x{q1:016x}")

    # dump whole VMM region, find any writes outside the AS image
    whole = ctypes.create_string_buffer(TOTAL)
    check(cuda.cuMemcpyDtoH_v2(whole, base, TOTAL), "dtohVMM")
    w = whole.raw
    Path("vmm_dump.bin").write_bytes(w)
    changes = []
    for off in range(0, TOTAL, 8):
        if off < len(asimg) and off >= AS_OFF and off < AS_OFF + len(asimg):
            continue  # skip AS image region (expected content)
        v = w[off:off+8]
        if v != bytes(8):
            changes.append((hex(off), v.hex()))
    print(f"non-AS nonzero qwords ({len(changes)}): {changes[:12] if changes else 'NONE'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
