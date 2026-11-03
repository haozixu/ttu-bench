#!/usr/bin/env python3
"""Bounded host-controlled diagnostics of IAS return slots and resume candidates.

Policy modes are experimental hypotheses, not validated general continuation.
Every request is a new kernel and OPEN/CLOSE session; no renderer is implied.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

from cuda_driver import check, cuda, sections
from instance_protocol import decode_instance_event, transform_ray
from optix_scene import Scene
from parent_resume import RawStateProbe
from run_instance_probe import read_device
from run_scene import geometry, reference, world_geometry
from sm86_fixture import FIXTURES, operands, triangle_result

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=2, help='static instances of the same one-triangle GAS')
    parser.add_argument('--spacing', type=float, default=4)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--resume', choices=('none', 'copy', 'promote', 'pop'), default='none')
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--ray', nargs=8, type=float, default=(.25, .25, -5, 0, 0, 0, 1, 100))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.count <= 8 or not 1 <= args.steps <= 8:
        parser.error('count and steps must be in 1..8')
    triangles, _ = geometry('single', 1)
    instances = [dict(id=17 + 24 * index,
                      transform=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, args.spacing * index])
                 for index in range(args.count)]
    if args.reverse:
        instances.reverse()
    world, identities = world_geometry(triangles, instances)
    expected = reference(world, args.ray, ray_flags=1, identities=identities)
    check(cuda.cuInit(0), 'init')
    device = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(device), 0), 'device')
    context = ctypes.c_void_p()
    check(cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device), 'context')
    scene = Scene(triangles, instances=instances)
    probe = RawStateProbe()
    image = (FIXTURES / 'jit_16.elf').read_bytes()
    static = next(section for section in sections(image) if section['name'] == '.nv.global.init')
    dummy = ctypes.c_uint64()
    check(cuda.cuMemAlloc_v2(ctypes.byref(dummy), 4096), 'dummy allocate')
    check(cuda.cuMemcpyHtoD_v2(dummy.value, image[static['off']:static['off'] + 128], 128), 'dummy upload')
    ias = read_device(scene.info.base, scene.info.size)
    gas = read_device(scene.info.gas_base, scene.info.gas_size)
    roots = [struct.unpack_from('<Q', data, 0x20)[0] for data in (ias, gas)]

    def inputs(ray, base, root):
        return operands(base, *ray[:3], ray[6], ray[7], dummy.value, dx=ray[4], dy=ray[5],
                        tmin=ray[3], ray_flags=1, root_offset=root)

    state = inputs(args.ray, scene.info.base, roots[0])
    records = []
    fatal_error = None
    pending = []
    for step in range(args.steps if args.resume != 'none' else 1):
        record = dict(step=step, input=[f'{value:08x}' for value in state[:76]])
        try:
            raw = probe.query(state)
            record['raw'] = [f'{value:08x}' for value in raw]
            if raw[33] >> 27 != 0x16:
                record['result'] = triangle_result(raw)
                records.append(record)
                break
            event = decode_instance_event(raw, scene.info.base, ias)
            local_ray = transform_ray(args.ray, event['inverse_transform'])
            child = probe.query(inputs(local_ray, scene.info.gas_base, roots[1]))
            record.update(event=event, local_ray=local_ray, child_raw=[f'{value:08x}' for value in child],
                          child_result=triangle_result(child))
            records.append(record)
            state = inputs(args.ray, scene.info.base, roots[0])
            if args.resume == 'copy':
                for start in (34, 50, 54, 56, 58, 60):
                    state[start:start + 2] = raw[start:start + 2]
            elif args.resume == 'promote':
                state[50:52] = raw[54:56]
                state[34:36] = raw[56:58]
                state[54:62] = [0] * 8
                if not state[51]:
                    break
            elif args.resume == 'pop':
                count = (raw[55] >> 25) & 7
                if count > 3:
                    raise ValueError('return stack exceeds the three explicit entry slots')
                for entry in reversed(range(count)):
                    pending.append((raw[54], (raw[55] & ~(7 << 25)) | (1 << 25),
                                    raw[56 + 2 * entry], raw[57 + 2 * entry]))
                if not pending:
                    break
                low, high, stack_low, stack_high = pending.pop()
                state[50:52] = low, high
                state[34:36] = stack_low, stack_high
                state[54:62] = [0] * 8
        except RuntimeError as error:
            fatal_error = str(error)
            record['error'] = fatal_error
            if not records or records[-1] is not record:
                records.append(record)
            break
    oracle = [] if fatal_error else scene.trace([args.ray], ray_flags=1)
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(), contract=__doc__,
                  args=vars(args) | {'output': str(args.output)}, instances=instances, triangles=triangles,
                  expected=expected, optix=oracle, records=records, fatal_error=fatal_error,
                  ias_base=scene.info.base, ias_hex=ias.hex(), gas_base=scene.info.gas_base,
                  ias_root_offset=roots[0], gas_root_offset=roots[1], build=probe.build_info,
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  source_sha256={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                 for name in ('run_parent_probe.py', 'parent_resume.py', 'device_instance.py',
                                              'instance_template.cu', 'instance_protocol.py', 'sm86_fixture.py',
                                              'run_scene.py', 'optix_scene.py', 'scene_bridge.cpp', 'ttu_isa.py')},
                  gpu_inventory=subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version', '--format=csv'],
                                               capture_output=True, text=True, check=True).stdout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(mode=args.resume, count=args.count, fatal_error=fatal_error,
                          events=[record.get('event', {}).get('instance_id_candidate') for record in records],
                          child=[record.get('child_result', record.get('result')) for record in records])))
    if not fatal_error:
        probe.close()
        scene.close()
        check(cuda.cuMemFree_v2(dummy.value), 'dummy free')
        check(cuda.cuCtxDestroy_v2(context), 'context destroy')
    if fatal_error:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
