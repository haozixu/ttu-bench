#!/usr/bin/env python3
"""Build live triangle GAS, issue raw TTU, compare CPU and OptiX.

Root packing uses an empirical header offset and fixture tags,
not a universal OptiX handle decoder.
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
from minimal import launch
from optix_scene import Scene
from sm86_fixture import FIXTURES, load_group, operands, triangle_result

BASE_TRIANGLE = [(1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0)]


def translated(x=0, y=0, z=0):
    return [[vx + x, vy + y, vz + z] for vx, vy, vz in BASE_TRIANGLE]


def geometry(name, count):
    if name == 'single':
        triangles = [translated()]
        targets = [(0.25, 0.25)]
    elif name == 'separated':
        triangles = [translated(), translated(x=3)]
        targets = [(0.25, 0.25), (3.25, 0.25)]
    elif name in ('overlap', 'overlap-reversed'):
        triangles = [translated(), translated(z=4)]
        if name == 'overlap-reversed':
            triangles.reverse()
        targets = [(0.25, 0.25)]
    else:
        width = math.ceil(math.sqrt(count))
        triangles = [translated(x=3 * (index % width), y=3 * (index // width))
                     for index in range(count)]
        sampled = sorted({0, count // 2, count - 1})
        targets = [(3 * (index % width) + 0.25, 3 * (index // width) + 0.25) for index in sampled]
    rays = [(x, y, -5, 0, 0, 0, 1, 100) for x, y in targets]
    rays += [(-10, -10, -5, 0, 0, 0, 1, 100), (*rays[0][:7], 6),
             (0.125, 0.75, -5, 0, 0, 0, 1, 100),
             (0.5, 0.125, -5, 0, 0, 0, 1, 100)]
    return triangles, rays


def general_rays(triangles):
    rays = []
    indices = sorted({0, len(triangles) // 2, len(triangles) - 1})
    directions = [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0),
                  (0, 1, 0), (0, -1, 0), (0.25, -0.5, 1),
                  (-0.5, 0.25, -1), (0.5, 1, -0.25), (0, 0, 2),
                  (0, 0, -0.5)]
    for index in indices:
        vertices = triangles[index]
        for bary_u, bary_v in ((0.125, 0.625), (0.375, 0.5)):
            target = [(1-bary_u-bary_v) * vertices[0][axis]
                      + bary_u * vertices[1][axis] + bary_v * vertices[2][axis]
                      for axis in range(3)]
            for direction in directions:
                origin = [position - 5 * delta for position, delta in zip(target, direction)]
                rays.append((*origin, 0, *direction, 100))
                rays.append((*origin, 5.25, *direction, 100))
    rays += [(0.25, 0.25, -5, 8, 0, 0, 1, 100),
             (0.25, 0.25, -5, 12, 0, 0, 1, 100),
             (0.25, 0.25, -5, 0, 0, 0, 1, 6),
             (-10, -10, -5, 0, 0, 0, 1, 100)]
    return rays


def instances_for(mode):
    identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    translated = [1, 0, 0, 4, 0, 1, 0, -2, 0, 0, 1, 3]
    rotated = [0, -1, 0, 4, 1, 0, 0, -2, 0, 0, 1, 3]
    scaled = [2, 0, 0, 4, 0, 0.5, 0, -2, 0, 0, 3, 3]
    transforms = {'none': [], 'identity': [identity], 'translate': [translated],
                  'rotate': [rotated], 'scale': [scaled], 'pair': [identity, translated]}[mode]
    return [dict(id=17 + index * 24, transform=transform) for index, transform in enumerate(transforms)]


def world_geometry(triangles, instances):
    if not instances:
        return triangles, [(index, None) for index in range(len(triangles))]
    world, identities = [], []
    for instance in instances:
        matrix = instance['transform']
        for index, triangle in enumerate(triangles):
            world.append([[sum(matrix[4 * axis + component] * vertex[component] for component in range(3))
                           + matrix[4 * axis + 3] for axis in range(3)] for vertex in triangle])
            identities.append((index, instance['id']))
    return world, identities


def reference(triangles, ray, *, ray_flags=0, identities=None):
    def sub(a, b): return [x - y for x, y in zip(a, b)]
    def dot(a, b): return sum(x * y for x, y in zip(a, b))
    def cross(a, b): return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    origin, direction = ray[:3], ray[4:7]
    closest = dict(primitive=None, t=None, barycentrics=None, front_face=None, instance_id=None)
    best = ray[7]
    for index, (a, b, c) in enumerate(triangles):
        e1, e2 = sub(b, a), sub(c, a)
        p = cross(direction, e2)
        determinant = dot(e1, p)
        if abs(determinant) < 1e-12:
            continue
        if ray_flags & 0x10 and determinant < 0 or ray_flags & 0x20 and determinant > 0:
            continue
        displacement = sub(origin, a)
        u = dot(displacement, p) / determinant
        q = cross(displacement, e1)
        v = dot(direction, q) / determinant
        t = dot(e2, q) / determinant
        if 0 <= u <= 1 and 0 <= v and u + v <= 1 and ray[3] <= t <= best:
            primitive, instance_id = identities[index] if identities is not None else (index, None)
            closest = dict(primitive=primitive, t=t, barycentrics=[u, v],
                           front_face=determinant > 0, instance_id=instance_id)
            best = t
    return closest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', choices=('single', 'separated', 'overlap', 'overlap-reversed', 'grid'))
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--group', choices=('full', 'expanded', 'reduced', 'both'), default='reduced',
                        help='both compares specialized expanded(19) and reduced(13); full(20) is diagnostic')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--oracle-only', action='store_true')
    parser.add_argument('--as-dump', type=Path)
    parser.add_argument('--root-offset', type=lambda value: int(value, 0),
                        help='override empirical root offset from AS header+0x20')
    parser.add_argument('--geometry-opaque', action='store_true', help='build with DISABLE_ANYHIT geometry flag')
    parser.add_argument('--ray-suite', choices=('legacy', 'general'), default='legacy')
    parser.add_argument('--instances', choices=('none', 'identity', 'translate', 'rotate', 'scale', 'pair'), default='none')
    parser.add_argument('--reverse-winding', action='store_true')
    parser.add_argument('--ray-flags', type=lambda value: int(value, 0), default=1,
                        help='OptiX flags: 0, 1 (disable anyhit), 0x10/0x20 (cull back/front), or combine with 1')
    parser.add_argument('--ray', nargs=8, type=float, metavar=('OX', 'OY', 'OZ', 'TMIN', 'DX', 'DY', 'DZ', 'TMAX'))
    parser.add_argument('--ray-index', type=int, help='run one ray in a fresh process to isolate state changes')
    args = parser.parse_args()
    if args.count < 1 or args.repeat < 1:
        parser.error('count and repeat must be positive')
    if args.ray_flags < 0 or args.ray_flags & ~0x31 or args.ray_flags & 0x30 == 0x30:
        parser.error('unsupported or mutually exclusive ray flags')
    triangles, rays = geometry(args.scene, args.count)
    if args.reverse_winding:
        triangles = [[triangle[0], triangle[2], triangle[1]] for triangle in triangles]
    instances = instances_for(args.instances)
    world, identities = world_geometry(triangles, instances)
    if args.ray_suite == 'general':
        rays = general_rays(world)
    if args.ray is not None:
        if not all(math.isfinite(value) for value in args.ray) or not 0 <= args.ray[3] < args.ray[7]:
            parser.error('ray requires finite values and 0 <= tmin < tmax')
        if not any(args.ray[4:7]):
            parser.error('ray direction must be nonzero')
        rays = [tuple(args.ray)]
    ray_indices = list(range(len(rays)))
    if args.ray_index is not None:
        if not 0 <= args.ray_index < len(rays):
            parser.error('ray-index is outside this scene')
        ray_indices = [args.ray_index]
    expected = [reference(world, ray, ray_flags=args.ray_flags, identities=identities) for ray in rays]
    check(cuda.cuInit(0), 'init')
    device = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(device), 0), 'device')
    context = ctypes.c_void_p()
    check(cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device), 'context')
    scene = Scene(triangles, geometry_flags=1 if args.geometry_opaque else 0, instances=instances)

    def allocate(size, data=None):
        pointer = ctypes.c_uint64()
        check(cuda.cuMemAlloc_v2(ctypes.byref(pointer), size), 'allocation')
        if data is not None:
            check(cuda.cuMemcpyHtoD_v2(pointer.value, data, len(data)), 'upload')
        return pointer.value

    original = (FIXTURES / 'jit_16.elf').read_bytes()
    static = next(section for section in sections(original) if section['name'] == '.nv.global.init')
    dummy = allocate(4096, original[static['off']:static['off'] + 128])
    output = allocate(512)
    as_data = ctypes.create_string_buffer(scene.info.size)
    check(cuda.cuMemcpyDtoH_v2(as_data, scene.info.base, scene.info.size), 'AS readback')
    if args.as_dump:
        args.as_dump.write_bytes(as_data.raw)
    root_offset = args.root_offset if args.root_offset is not None else struct.unpack_from('<Q', as_data.raw, 0x20)[0]
    if root_offset % 128 or not 0 <= root_offset <= scene.info.size - 128:
        raise ValueError('root offset must point to an aligned block inside the AS')
    groups = () if args.oracle_only else (('expanded', 'reduced') if args.group == 'both' else (args.group,))
    records = []
    failures = []
    fatal_error = None
    for group_name in groups:
        group = load_group(full=group_name == 'full', expanded=group_name == 'expanded')
        for index in ray_indices:
            ray = rays[index]
            for trial in range(args.repeat):
                values = operands(scene.info.base, ray[0], ray[1], ray[2], ray[6], ray[7], dummy,
                                  dx=ray[4], dy=ray[5], tmin=ray[3], ray_flags=args.ray_flags, root_offset=root_offset)
                try:
                    raw = launch(values, output, group)
                except RuntimeError as error:
                    fatal_error = dict(group=group_name, ray_index=index, trial=trial, error=str(error))
                    failures.append(str(error))
                    break
                decoded = triangle_result(raw)
                hit, t = decoded['hit'], decoded['t']
                target = expected[index]
                matches = hit == (target['primitive'] is not None)
                if hit and target['t'] is not None:
                    matches = matches and math.isfinite(t) and math.isclose(t, target['t'], rel_tol=1e-5, abs_tol=1e-5)
                primitive_matches = decoded['primitive'] == target['primitive'] and decoded['known_tag']
                face_matches = decoded['front_face'] == target['front_face']
                bary_matches = not hit
                if hit and target['barycentrics'] is not None:
                    bary_matches = all(math.isfinite(a) and math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)
                                       for a, b in zip(decoded['barycentrics'], target['barycentrics']))
                if not matches:
                    failures.append(f'{group_name} ray{index} trial{trial}: hit/t mismatch')
                if not primitive_matches or not bary_matches or not face_matches:
                    failures.append(f'{group_name} ray{index} trial{trial}: primitive/barycentrics/face mismatch')
                records.append(dict(group=group_name, group_sha256=hashlib.sha256(group).hexdigest(),
                                    ttu_instructions=len(group) // 16,
                                    ray_index=index, trial=trial, hit=hit,
                                    t=t if t is None or math.isfinite(t) else None,
                                    primitive_candidate=decoded['primitive'],
                                    result_tag=hex(raw[72] >> 28),
                                    result_kind=decoded['kind'], status_class=decoded['status_class'],
                                    known_triangle_tag=decoded['known_tag'],
                                    front_face_candidate=decoded['front_face'], front_face_matches=face_matches,
                                    t_sign_bit=decoded['t_sign_bit'],
                                    barycentrics_candidate=[x if math.isfinite(x) else None for x in decoded['barycentrics']]
                                        if decoded['barycentrics'] is not None else None,
                                    primitive_matches=primitive_matches, barycentrics_matches=bary_matches,
                                    hit_t_matches=matches, registers=[f'{value:08x}' for value in raw[:76]]))
            if fatal_error:
                break
        if fatal_error:
            break
    oracle = [] if fatal_error else scene.trace(rays, ray_flags=args.ray_flags)
    if instances and groups:
        failures.append('instance return/resume protocol is not implemented; GAS decoder is diagnostic only')
    for index, (cpu, optix) in enumerate(zip(expected, oracle)):
        matches = (cpu['primitive'] == optix['primitive'] and cpu['front_face'] == optix['front_face']
                   and cpu['instance_id'] == optix['instance_id'])
        if cpu['t'] is not None and optix['t'] is not None:
            matches = matches and math.isclose(cpu['t'], optix['t'], rel_tol=1e-5, abs_tol=1e-5)
            matches = matches and all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)
                                      for a, b in zip(cpu['barycentrics'], optix['barycentrics']))
        if not matches:
            failures.append(f'CPU/OptiX ray{index} mismatch')
    source_hashes = hash_sources(('run_scene.py', 'scene_bridge.cpp', 'scene_oracle.cu', 'optix_scene.py',
                                  'minimal.py', 'cubin_builder.py', 'ttu_isa.py', 'sm86_fixture.py'))
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(), scene=args.scene,
                  triangles=triangles, rays=rays, cpu=expected, optix=oracle, records=records,
                  instances=instances, world_triangles=world,
                  as_base=hex(scene.info.base), as_size=scene.info.size, handle=hex(scene.info.handle),
                  gas_base=hex(scene.info.gas_base), gas_size=scene.info.gas_size, gas_handle=hex(scene.info.gas_handle),
                  as_sha256=hashlib.sha256(as_data.raw).hexdigest(),
                  root_offset=root_offset, root_assumption='AS base + header[0x20] or explicit override; original fixture tags',
                  ttu_before_optix_trace=bool(groups),
                  ttu_primitive_decoded=bool(records) and all(record['known_triangle_tag'] for record in records),
                  decoder_scope='empirical 8/a-tags, low28 primitive, abs(R73) t, [abs(R32),R33] bary, R32 sign face; triangle GAS only',
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), ray_indices=ray_indices,
                  dummy_base=hex(dummy),
                  source_sha256=source_hashes, failures=failures,
                  fatal_error=fatal_error, oracle_only=args.oracle_only,
                  geometry_flags=1 if args.geometry_opaque else 0,
                  ray_suite=args.ray_suite, reverse_winding=args.reverse_winding,
                  ray_flags=args.ray_flags,
                  ttu_instance_id_decoded=False, ray_reference_contract='closest hit',
                  opaque_triangle_request=bool(args.ray_flags & 1 or args.geometry_opaque),
                  gpu_inventory=subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version', '--format=csv'],
                                               capture_output=True, text=True, check=True).stdout)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(scene=args.scene, triangles=len(triangles), rays=len(rays),
                          launches=len(records), as_size=scene.info.size,
                          failures=failures[:8], failure_count=len(failures),
                          ttu_primitive_decoded=report['ttu_primitive_decoded'])))
    scene.close()
    if args.check and failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
