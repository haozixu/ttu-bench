#!/usr/bin/env python3
"""Validate device-side parent resubmission for a restricted flat 1-3 instance IAS."""
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
from device_instance import Params
from instance_protocol import decode_instance_event, transform_ray
from optix_scene import Scene
from parent_resume import ParentProbe, ParentParams
from run_instance_probe import read_device, matches_triangle
from run_scene import geometry, general_rays, reference, world_geometry
from sm86_fixture import FIXTURES, triangle_result

SOURCES = ('run_parent_resume.py', 'parent_resume.py', 'parent_template.cu', 'device_instance.py',
           'instance_template.cu', 'instance_protocol.py', 'run_instance_probe.py', 'run_scene.py',
           'sm86_fixture.py', 'ttu_isa.py', 'cuda_driver.py', 'optix_scene.py', 'scene_bridge.cpp', 'scene_oracle.cu')


def matches(actual, target):
    if any(actual[key] != target[key] for key in ('primitive', 'instance_id', 'front_face')):
        return False
    return target['primitive'] is None or (
        math.isclose(actual['t'], target['t'], rel_tol=1e-5, abs_tol=1e-5)
        and all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)
                for a, b in zip(actual['barycentrics'], target['barycentrics'])))


def main():
    cases_path = FIXTURES / 'phase17-parent-cases.json'
    cases = {case['name']: case for case in json.loads(cases_path.read_text())['scenarios']}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=tuple(cases), default='chain2')
    parser.add_argument('--scene', choices=('single', 'separated', 'overlap', 'grid'), default='single')
    parser.add_argument('--count', type=int, default=64, help='triangle count for grid')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--ray-index', type=int)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--stop-after-first', action='store_true', help='negative control: omit pending-parent resubmission')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 1 or args.count < 1:
        parser.error('repeat and count must be positive')
    case = cases[args.scenario]
    if 'rays' in case and args.scene != 'single':
        parser.error('counterexample rays are defined for single-triangle GAS')
    triangles, _ = geometry(args.scene, args.count)
    instances = case['instances']
    world, identities = world_geometry(triangles, instances)
    rays = case.get('rays') or general_rays(world)
    indices = list(range(len(rays))) if args.ray_index is None else [args.ray_index]
    if any(not 0 <= index < len(rays) for index in indices):
        parser.error('ray-index outside suite')
    targets = [reference(world, ray, ray_flags=1, identities=identities) for ray in rays]
    check(cuda.cuInit(0), 'init')
    device = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(device), 0), 'device')
    context = ctypes.c_void_p()
    check(cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device), 'context')
    scene = Scene(triangles, instances=instances)
    probe = ParentProbe()
    image = (FIXTURES / 'jit_16.elf').read_bytes()
    static = next(section for section in sections(image) if section['name'] == '.nv.global.init')
    dummy = ctypes.c_uint64()
    check(cuda.cuMemAlloc_v2(ctypes.byref(dummy), 4096), 'dummy allocate')
    check(cuda.cuMemcpyHtoD_v2(dummy.value, image[static['off']:static['off'] + 128], 128), 'dummy upload')
    ias, gas = (read_device(base, size) for base, size in
                ((scene.info.base, scene.info.size), (scene.info.gas_base, scene.info.gas_size)))
    roots = [struct.unpack_from('<Q', data, 0x20)[0] for data in (ias, gas)]
    if any(root % 128 or root + 128 > len(data) for root, data in zip(roots, (ias, gas))):
        raise ValueError('AS root outside aligned range')
    root = scene.info.base + roots[0]
    params = Params(ias_base=scene.info.base, ias_size=scene.info.size, dummy=dummy.value,
                    child_base=scene.info.gas_base, child_size=scene.info.gas_size,
                    root_low=(root >> 7) & 0xffffffff, root_high=(root >> 39) | 0x02484000,
                    dummy_low=(dummy.value >> 6) & 0xffffffff, dummy_high=(dummy.value >> 38) | 0x02402800)
    control = ParentParams(base=params, stop_after_first=args.stop_after_first)
    records, failures = [], []
    fatal_error = None
    for index in indices:
        for trial in range(args.repeat):
            control.base.ray[:] = rays[index]
            record = dict(ray_index=index, trial=trial, expected=targets[index])
            try:
                result = probe.launch(control)
                actual = dict(primitive=None, t=None, barycentrics=None, instance_id=None, front_face=None)
                if result.primitive != 0xffffffff:
                    actual = dict(primitive=result.primitive, t=result.t, barycentrics=[result.u, result.v],
                                  instance_id=result.instance_id, front_face=bool(result.front_face))
                passed = result.status == 1 and result.pending_count == 0 and matches(actual, targets[index])
                traces = []
                for trace in result.trace[:result.steps]:
                    parent, child = tuple(trace.parent), tuple(trace.child)
                    decoded = triangle_result(child) if trace.kind == 2 else None
                    entry = dict(kind=trace.kind, instance_id=trace.instance_id, index_word=trace.index_word,
                                 pending_after=trace.pending_after, event_address=trace.event,
                                 child_root=trace.child_root, input=[f'{value:08x}' for value in trace.input],
                                 parent=[f'{value:08x}' for value in parent],
                                 child=[f'{value:08x}' for value in child], local_ray=list(trace.local_ray),
                                 child_result=decoded)
                    if trace.kind == 2:
                        event = decode_instance_event(parent, scene.info.base, ias)
                        local = transform_ray(rays[index], event['inverse_transform'])
                        local = (*local[:7], trace.local_ray[7])
                        expected_child = reference(triangles, local, ray_flags=1)
                        child_ok = (decoded['kind'] == 'miss' and expected_child['primitive'] is None
                                    or expected_child['primitive'] is not None and matches_triangle(decoded, expected_child))
                        entry.update(event=event, expected_child=expected_child, child_matches=child_ok)
                        passed = (passed and child_ok and trace.instance_id == event['instance_id_candidate']
                                  and trace.index_word == event['instance_index_word']
                                  and trace.event == event['record_address']
                                  and trace.child_root == scene.info.gas_base + roots[1]
                                  and all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)
                                          for a, b in zip(trace.local_ray[:7], local[:7])))
                    traces.append(entry)
                if 'required_first_instance' in case:
                    first = traces[0]
                    passed = (passed and first['kind'] == 2
                              and first['instance_id'] == case['required_first_instance']
                              and first['child_result']['kind'] == case['required_first_child']
                              and len(traces) > 1)
                    if case['name'] == 'first-farther-later-nearer':
                        passed = passed and actual['t'] is not None and first['child_result']['t'] > actual['t'] + 1e-4
                record.update(status=result.status, requests=result.requests, steps=result.steps,
                              maximum_pending=result.maximum_pending, pending_at_end=result.pending_count,
                              actual=actual, traces=traces, passed=passed)
            except RuntimeError as error:
                fatal_error = str(error)
                record.update(passed=False, error=fatal_error)
            except ValueError as error:
                record.update(passed=False, error=str(error))
            records.append(record)
            if not record['passed']:
                failures.append(f'ray{index} trial{trial}: {record.get("error", "status/result mismatch")}')
            if fatal_error:
                break
        if fatal_error:
            break
    oracle = [] if fatal_error else scene.trace([rays[index] for index in indices], ray_flags=1)
    for index, actual in zip(indices, oracle):
        if not matches(actual, targets[index]):
            failures.append(f'CPU/OptiX ray{index}: mismatch')
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(), scenario=case, scene=args.scene,
                  contract='Restricted device-side parent resubmission: flat IAS, 1-3 positive-determinant static instances, shared triangle GAS, relative entry type0x21; no performance measurement',
                  stop_after_first_control=args.stop_after_first,
                  triangles=triangles, instances=instances, rays=rays, ray_indices=indices, cpu=targets, optix=oracle,
                  records=records, failures=failures, fatal_error=fatal_error, build=probe.build_info,
                  ias_base=scene.info.base, ias_hex=ias.hex(), gas_base=scene.info.gas_base,
                  ias_root_offset=roots[0], gas_root_offset=roots[1], gas_sha256=hashlib.sha256(gas).hexdigest(),
                  case_fixture_sha256=hashlib.sha256(cases_path.read_bytes()).hexdigest(),
                  source_sha256=hash_sources(SOURCES),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  gpu_inventory=subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version', '--format=csv'],
                                               capture_output=True, text=True, check=True).stdout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(scenario=args.scenario, scene=args.scene, cases=len(records),
                          requests=sum(record.get('requests', 0) for record in records),
                          failures=failures[:6], failure_count=len(failures), fatal_error=fatal_error)))
    if not fatal_error:
        probe.close()
        scene.close()
        check(cuda.cuMemFree_v2(dummy.value), 'dummy free')
        check(cuda.cuCtxDestroy_v2(context), 'context destroy')
    if args.check and failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
