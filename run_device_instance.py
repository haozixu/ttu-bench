#!/usr/bin/env python3
"""Check device-side instance dispatch and an independent child query (no IAS resume)."""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess

from cuda_driver import check, cuda, sections
from device_instance import DeviceProbe, Params
from instance_protocol import decode_instance_event, transform_ray
from optix_scene import Scene
from run_instance_probe import matches_triangle, read_device
from run_scene import geometry, general_rays, instances_for, reference, world_geometry
from sm86_fixture import FIXTURES, triangle_result

HERE = Path(__file__).resolve().parent
STATUS = {1: 'ias_miss', 2: 'unexpected_event', 3: 'invalid_event_address',
          4: 'invalid_child_root', 5: 'independent_child_complete', 6: 'invalid_instance_metadata'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', choices=('identity', 'translate', 'rotate', 'scale'), default='identity')
    parser.add_argument('--scene', choices=('single', 'separated', 'overlap', 'grid'), default='single')
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--instance-id', type=lambda value: int(value, 0), default=17)
    parser.add_argument('--ray-index', type=int, help='isolate one ray; default runs the general suite')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.count < 1 or args.repeat < 1 or not 0 <= args.instance_id < 1 << 24:
        parser.error('count/repeat must be positive and instance-id must fit in 24 bits')
    triangles, _ = geometry(args.scene, args.count)
    instances = instances_for(args.instance)
    instances[0]['id'] = args.instance_id
    world, identities = world_geometry(triangles, instances)
    rays = general_rays(world)
    if args.ray_index is not None:
        if not 0 <= args.ray_index < len(rays):
            parser.error('ray-index outside suite')
        indices = [args.ray_index]
    else:
        indices = list(range(len(rays)))
    targets = [reference(world, ray, ray_flags=1, identities=identities) for ray in rays]
    check(cuda.cuInit(0), 'init')
    device = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(device), 0), 'device')
    context = ctypes.c_void_p()
    check(cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device), 'context')
    scene = Scene(triangles, instances=instances)
    probe = DeviceProbe()
    image = (FIXTURES / 'jit_16.elf').read_bytes()
    static = next(section for section in sections(image) if section['name'] == '.nv.global.init')
    dummy = ctypes.c_uint64()
    check(cuda.cuMemAlloc_v2(ctypes.byref(dummy), 4096), 'dummy allocate')
    check(cuda.cuMemcpyHtoD_v2(dummy.value, image[static['off']:static['off'] + 128], 128), 'dummy upload')
    ias = read_device(scene.info.base, scene.info.size)
    gas = read_device(scene.info.gas_base, scene.info.gas_size)
    roots = [struct.unpack_from('<Q', data, 0x20)[0] for data in (ias, gas)]
    for root, data in zip(roots, (ias, gas)):
        if root % 128 or root + 128 > len(data):
            raise ValueError('empirical AS root outside aligned range')
    root = scene.info.base + roots[0]
    params = Params(ias_base=scene.info.base, ias_size=scene.info.size,
                    dummy=dummy.value, child_base=scene.info.gas_base, child_size=scene.info.gas_size,
                    root_low=(root >> 7) & 0xffffffff, root_high=(root >> 39) | 0x02484000,
                    dummy_low=(dummy.value >> 6) & 0xffffffff,
                    dummy_high=(dummy.value >> 38) | 0x02402800)
    records, failures = [], []
    fatal_error = None
    for index in indices:
        for trial in range(args.repeat):
            ray, target = rays[index], targets[index]
            params.ray[:] = ray
            record = dict(ray_index=index, trial=trial, expected=target)
            try:
                result = probe.launch(params)
                initial, child = tuple(result.initial), tuple(result.child)
                record.update(status=result.status, status_name=STATUS.get(result.status, 'unknown'),
                              requests=result.requests, parent_slot_nonzero_raw=result.parent_slot_nonzero,
                              initial=[f'{value:08x}' for value in initial],
                              child=[f'{value:08x}' for value in child], event_address=result.event,
                              child_root=result.child_root, instance_id_raw=result.instance_id)
                if result.status == 1:
                    passed = (result.requests == 1 and target['primitive'] is None
                              and triangle_result(initial)['kind'] == 'miss')
                elif result.status == 5:
                    event = decode_instance_event(initial, scene.info.base, ias)
                    local_ray = transform_ray(ray, event['inverse_transform'])
                    decoded = triangle_result(child)
                    passed = (result.requests == 2 and result.instance_id == args.instance_id
                              and result.event == event['record_address']
                              and result.child_root == scene.info.gas_base + roots[1]
                              and all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)
                                      for a, b in zip(result.local_ray, local_ray))
                              and (decoded['kind'] == 'miss' and target['primitive'] is None
                                   or target['primitive'] is not None and matches_triangle(decoded, target)
                                   and result.instance_id == target['instance_id']))
                    record.update(event=event, local_ray=list(result.local_ray), child_result=decoded)
                else:
                    passed = False
                record['passed'] = passed
            except RuntimeError as error:
                fatal_error = str(error)
                record.update(passed=False, error=fatal_error)
            except ValueError as error:
                record.update(passed=False, error=str(error))
            records.append(record)
            if not record['passed']:
                failures.append(f'ray{index} trial{trial}: {record.get("error", "result mismatch")}')
            if fatal_error:
                break
        if fatal_error:
            break
    oracle = [] if fatal_error else scene.trace([rays[index] for index in indices], ray_flags=1)
    for index, actual in zip(indices, oracle):
        target = targets[index]
        passed = all(actual[key] == target[key] for key in ('primitive', 'instance_id', 'front_face'))
        if target['primitive'] is not None:
            passed = (passed and math.isclose(actual['t'], target['t'], rel_tol=1e-5, abs_tol=1e-5)
                      and all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)
                              for a, b in zip(actual['barycentrics'], target['barycentrics'])))
        if not passed:
            failures.append(f'CPU/OptiX ray{index}: mismatch')
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                  contract='single static instance; device dispatch/transform/independent child query; NOT parent resume',
                  parent_resume_implemented=False, instance_mode=args.instance, scene=args.scene,
                  triangles=triangles, instances=instances, rays=rays, ray_indices=indices,
                  cpu=targets, optix=oracle, records=records, failures=failures, fatal_error=fatal_error,
                  build=probe.build_info, ias_base=scene.info.base, gas_base=scene.info.gas_base,
                  ias_root_offset=roots[0], gas_root_offset=roots[1], ias_hex=ias.hex(),
                  gas_sha256=hashlib.sha256(gas).hexdigest(),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  source_sha256={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                 for name in ('run_device_instance.py', 'device_instance.py', 'instance_template.cu',
                                              'instance_protocol.py', 'run_instance_probe.py', 'run_scene.py',
                                              'sm86_fixture.py', 'cuda_driver.py', 'ttu_isa.py', 'optix_scene.py',
                                              'scene_bridge.cpp', 'scene_oracle.cu')},
                  gpu_inventory=subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version', '--format=csv'],
                                               capture_output=True, text=True, check=True).stdout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(instance=args.instance, scene=args.scene, cases=len(records),
                          ttu_requests=sum(record.get('requests', 0) for record in records),
                          failures=failures[:8], failure_count=len(failures), fatal_error=fatal_error)))
    if not fatal_error:
        probe.close()
        scene.close()
        check(cuda.cuMemFree_v2(dummy.value), 'dummy free')
        check(cuda.cuCtxDestroy_v2(context), 'context destroy')
    if args.check and failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
