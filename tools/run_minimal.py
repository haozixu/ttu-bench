#!/usr/bin/env python3
"""Fresh-context SM86 TTU functional probe; no OptiX or driver kernel launch."""
import argparse
import ctypes
import hashlib
import json
import math
import struct
from cuda_driver import check, cuda, sections
from minimal import launch
from sm86_fixture import AS_SHA256, FIXTURES, load_group, operands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['hit', 'miss'], nargs='?', default='hit')
    parser.add_argument('--ox', type=float)
    parser.add_argument('--oy', type=float)
    parser.add_argument('--oz', type=float, default=-5.0)
    parser.add_argument('--dz', type=float, default=1.0)
    parser.add_argument('--tmax', type=float, default=100.0)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--artifact')
    parser.add_argument('--check', action='store_true', help='compare against analytic single-triangle intersection')
    parser.add_argument('--full-group', action='store_true', help='unmodified 20-instruction driver group')
    args = parser.parse_args()
    if args.dz == 0 or args.repeat < 1:
        parser.error('--dz must be nonzero and --repeat must be positive')
    ox = args.ox if args.ox is not None else (0.25 if args.mode == 'hit' else 50.0)
    oy = args.oy if args.oy is not None else (0.25 if args.mode == 'hit' else 2.0)
    check(cuda.cuInit(0), 'init')
    device = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(device), 0), 'device')
    context = ctypes.c_void_p()
    check(cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device), 'context')
    as_data = (FIXTURES / 'optix_as.bin').read_bytes()
    if hashlib.sha256(as_data).hexdigest() != AS_SHA256:
        raise ValueError('AS differs from the validated single-triangle fixture')
    as_image = bytearray(as_data)
    def alloc(size):
        address = ctypes.c_uint64()
        check(cuda.cuMemAlloc_v2(ctypes.byref(address), size), 'alloc')
        return address.value
    base = alloc(len(as_image))
    struct.pack_into('<Q', as_image, 0xd0, base)
    check(cuda.cuMemcpyHtoD_v2(base, bytes(as_image), len(as_image)), 'AS upload')
    output = alloc(512)
    original = (FIXTURES / 'jit_16.elf').read_bytes()
    group = load_group(full=args.full_group)
    global_section = next(candidate for candidate in sections(original)
                          if candidate['name'] == '.nv.global.init')
    dummy = alloc(4096)
    dummy_data = original[global_section["off"]:global_section["off"] + 128]
    check(cuda.cuMemcpyHtoD_v2(dummy, dummy_data, len(dummy_data)), "dummy upload")
    values = operands(base, ox, oy, args.oz, args.dz, args.tmax, dummy)
    expected_t = (3.0 - 3.0 * ox - 1.5 * oy - args.oz) / args.dz
    expected_hit = ox >= 0 and oy >= 0 and ox + oy / 2 <= 1 and 0 <= expected_t <= args.tmax
    results = []
    for _ in range(args.repeat):
        result = launch(values, output, group, args.artifact)
        if args.check:
            if expected_hit:
                measured = -struct.unpack('<f', struct.pack('<I', result[73]))[0]
                if result[72] == 0xffffffff or not math.isclose(measured, expected_t, rel_tol=1e-6, abs_tol=1e-6):
                    raise RuntimeError(f'expected hit t={expected_t}, got {result[72:74]}')
            else:
                if result[72:74] != (0xffffffff, 0xffffffff):
                    raise RuntimeError(f'expected miss, got {result[72:74]}')
        results.append([f'{result[register]:08x}' for register in (72, 73, 32, 33, 40, 41)])
    print(json.dumps(dict(mode=args.mode, origin=[ox, oy, args.oz], direction=[0, 0, args.dz],
                         tmax=args.tmax, as_base=hex(base), registers=['R72','R73','R32','R33','R40','R41'],
                         results=results, expected_hit=expected_hit, expected_t=expected_t if expected_hit else None, checked=args.check,
                         t_negative=struct.unpack('<f', struct.pack('<I', result[73]))[0] if result[72] != 0xffffffff else None,
                         ttu_instructions=len(group)//16,
                         as_sha256=hashlib.sha256(as_data).hexdigest(),
                         group_sha256=hashlib.sha256(group).hexdigest()), allow_nan=False))


if __name__ == '__main__':
    main()
