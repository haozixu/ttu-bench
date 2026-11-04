#!/usr/bin/env python3
"""Validate one static-instance event, then query its child independently.

The two launches do not preserve or resume an IAS traversal session.
Results must not be treated as a general IAS implementation.
"""
from project_paths import hash_sources
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
from instance_protocol import decode_instance_event, transform_ray
from minimal import launch
from optix_scene import Scene
from run_scene import geometry, general_rays, instances_for, reference, world_geometry
from sm86_fixture import FIXTURES, load_group, operands, triangle_result




def read_device(address, size):
    data = ctypes.create_string_buffer(size)
    check(cuda.cuMemcpyDtoH_v2(data, address, size), 'instance readback')
    return data.raw


def matches_triangle(actual, expected):
    if actual['kind'] != 'triangle' or actual['primitive'] != expected['primitive']:
        return False
    return (actual['front_face'] == expected['front_face']
            and math.isclose(actual['t'], expected['t'], rel_tol=1e-5, abs_tol=1e-5)
            and all(math.isclose(first, second, rel_tol=1e-5, abs_tol=1e-5)
                    for first, second in zip(actual['barycentrics'], expected['barycentrics'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', choices=('identity', 'translate', 'rotate', 'scale'), default='identity')
    parser.add_argument('--scene', choices=('single', 'separated', 'overlap', 'grid'), default='single')
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--instance-id', type=lambda value: int(value, 0), default=17)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('count must be positive')
    if not 0 <= args.instance_id < 1 << 24:
        parser.error('instance-id must fit in 24 bits')
    triangles, _ = geometry(args.scene, args.count)
    instances = instances_for(args.instance)
    instances[0]['id'] = args.instance_id
    world, identities = world_geometry(triangles, instances)
    rays = general_rays(world)
    targets = [reference(world, ray, ray_flags=1, identities=identities) for ray in rays]
    check(cuda.cuInit(0), 'init')
    device = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(device), 0), 'device')
    context = ctypes.c_void_p()
    check(cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device), 'context')
    scene = Scene(triangles, instances=instances)
    allocations = []

    def allocate(size, data=None):
        address = ctypes.c_uint64()
        check(cuda.cuMemAlloc_v2(ctypes.byref(address), size), 'allocation')
        allocations.append(address.value)
        if data is not None:
            check(cuda.cuMemcpyHtoD_v2(address.value, data, len(data)), 'upload')
        return address.value

    original = (FIXTURES / 'jit_16.elf').read_bytes()
    static = next(section for section in sections(original) if section['name'] == '.nv.global.init')
    dummy = allocate(4096, original[static['off']:static['off'] + 128])
    output = allocate(512)
    ias_data = read_device(scene.info.base, scene.info.size)
    gas_data = read_device(scene.info.gas_base, scene.info.gas_size)
    ias_root = struct.unpack_from('<Q', ias_data, 0x20)[0]
    gas_root = struct.unpack_from('<Q', gas_data, 0x20)[0]
    for offset, data in ((ias_root, ias_data), (gas_root, gas_data)):
        if offset % 128 or offset + 128 > len(data):
            raise ValueError('empirical root offset outside aligned AS range')
    records, failures = [], []
    fatal_error = None
    for group_name in ('expanded', 'reduced'):
        group = load_group(expanded=group_name == 'expanded')
        for index, (ray, target) in enumerate(zip(rays, targets)):
            values = operands(scene.info.base, ray[0], ray[1], ray[2], ray[6], ray[7], dummy,
                              dx=ray[4], dy=ray[5], tmin=ray[3], ray_flags=1, root_offset=ias_root)
            record = dict(group=group_name, ray_index=index, ray=list(ray), expected=target,
                          group_sha256=hashlib.sha256(group).hexdigest())
            try:
                raw = launch(values, output, group)
                record['ias_registers'] = [f'{value:08x}' for value in raw[:76]]
                if raw[33] >> 27 != 0x16:
                    decoded = triangle_result(raw)
                    record.update(event=None, ias_result=decoded, independent_child_query=False,
                                  passed=target['primitive'] is None and decoded['kind'] == 'miss')
                else:
                    event = decode_instance_event(raw, scene.info.base, ias_data)
                    record['event'] = event
                    if event['instance_id_candidate'] != args.instance_id:
                        raise ValueError('decoded instance ID does not match build input')
                    inverse, forward = event['inverse_transform'], instances[0]['transform']
                    composition = [sum(inverse[4 * axis + component] * forward[4 * component + column]
                                       for component in range(3))
                                   + (inverse[4 * axis + 3] if column == 3 else 0)
                                   for axis in range(3) for column in range(4)]
                    if not all(math.isclose(value, float(index // 4 == index % 4), abs_tol=1e-5)
                               for index, value in enumerate(composition)):
                        raise ValueError('decoded matrix is not the inverse of build input')
                    if event['child_root_address'] != scene.info.gas_base + gas_root:
                        raise ValueError('decoded child root does not match builder GAS root')
                    if event['child_handle_candidate'] != scene.info.gas_handle:
                        raise ValueError('decoded child handle does not match builder GAS handle')
                    local_ray = transform_ray(ray, event['inverse_transform'])
                    local_expected = reference(triangles, local_ray, ray_flags=1)
                    if local_expected['primitive'] != target['primitive']:
                        raise ValueError('decoded inverse transform changes expected primitive')
                    child_values = operands(scene.info.gas_base, local_ray[0], local_ray[1], local_ray[2],
                                            local_ray[6], local_ray[7], dummy, dx=local_ray[4], dy=local_ray[5],
                                            tmin=local_ray[3], ray_flags=1,
                                            root_offset=event['child_root_address'] - scene.info.gas_base)
                    child_raw = launch(child_values, output, group)
                    decoded = triangle_result(child_raw)
                    passed = (decoded['kind'] == 'miss' and target['primitive'] is None
                              or target['primitive'] is not None and matches_triangle(decoded, target)
                              and event['instance_id_candidate'] == target['instance_id'])
                    record.update(local_ray=list(local_ray), local_cpu=local_expected,
                                  child_registers=[f'{value:08x}' for value in child_raw[:76]],
                                  child_result=decoded, independent_child_query=True, passed=passed)
            except RuntimeError as error:
                fatal_error = str(error)
                record.update(passed=False, error=fatal_error)
            except ValueError as error:
                record.update(passed=False, error=str(error))
            records.append(record)
            if not record['passed']:
                failures.append(f'{group_name} ray{index}: {record.get("error", "result mismatch")}')
            if fatal_error:
                break
        if fatal_error:
            break
    oracle = [] if fatal_error else scene.trace(rays, ray_flags=1)
    for index, (target, actual) in enumerate(zip(targets, oracle)):
        matched = (target['primitive'] == actual['primitive'] and target['instance_id'] == actual['instance_id']
                   and target['front_face'] == actual['front_face'])
        if target['primitive'] is not None:
            matched = (matched and math.isclose(target['t'], actual['t'], rel_tol=1e-5, abs_tol=1e-5)
                       and all(math.isclose(first, second, rel_tol=1e-5, abs_tol=1e-5)
                               for first, second in zip(target['barycentrics'], actual['barycentrics'])))
        if not matched:
            failures.append(f'CPU/OptiX ray{index}: mismatch')
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                  contract='single static instance; host decodes IAS event and starts independent child GAS query; NOT resume',
                  raw_ttu_ias_continuation_implemented=False, instance_mode=args.instance, scene=args.scene,
                  instance_id=args.instance_id,
                  triangles=triangles, instances=instances, rays=rays, cpu=targets, optix=oracle,
                  as_base=scene.info.base, gas_base=scene.info.gas_base,
                  ias_root_offset=ias_root, gas_root_offset=gas_root,
                  ias_hex=ias_data.hex(), gas_sha256=hashlib.sha256(gas_data).hexdigest(),
                  records=records, failures=failures, fatal_error=fatal_error,
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  source_sha256=hash_sources(('run_instance_probe.py', 'instance_protocol.py', 'run_scene.py', 'cuda_driver.py',
                                              'sm86_fixture.py', 'minimal.py', 'ttu_isa.py', 'cubin_builder.py',
                                              'optix_scene.py', 'scene_bridge.cpp', 'scene_oracle.cu')),
                  gpu_inventory=subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version', '--format=csv'],
                                               capture_output=True, text=True, check=True).stdout)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(instance=args.instance, scene=args.scene, triangles=len(triangles),
                          cases=len(records), independent_child_queries=sum(record.get('independent_child_query', False)
                                                                           for record in records),
                          failures=failures[:8], failure_count=len(failures), fatal_error=fatal_error)))
    if not fatal_error:
        scene.close()
        for address in allocations:
            check(cuda.cuMemFree_v2(address), 'free')
        check(cuda.cuCtxDestroy_v2(context), 'context destroy')
    if args.check and failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
