#!/usr/bin/env python3
"""One-warp functional/timing probe; reports instrumented SM-clock intervals."""
from project_paths import hash_sources
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess

from cuda_driver import check, cuda, sections
from sm86_fixture import AS_SHA256, FIXTURES, operands, triangle_result
from warp_timing import TimingProbe, build



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=('none', 'nop', 'mf0', 'reduced', 'expanded'), default='reduced')
    parser.add_argument('--lanes', type=int, choices=(1, 32), default=1)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--launches', type=int, default=12)
    parser.add_argument('--alu', type=int, default=0)
    parser.add_argument('--placement', choices=('before-wait', 'after-wait'), default='before-wait')
    parser.add_argument('--open-gap', type=int, help='clock after OPEN, then 0..128 FADDs before MF; fused body unchanged')
    parser.add_argument('--ray', choices=('hit', 'miss', 'short'), default='hit')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, help='local cubin with process-specific VAs; not replayable')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if not 2 <= args.samples <= 8 or not 2 <= args.launches or not 0 <= args.alu <= 512:
        parser.error('samples 2..8, launches >=2 and alu 0..512 required')
    if args.open_gap is not None and (args.group not in ('reduced', 'expanded') or not 0 <= args.open_gap <= 128):
        parser.error('--open-gap requires reduced/expanded and 0..128')
    check(cuda.cuInit(0), 'init')
    device, context = ctypes.c_int(), ctypes.c_void_p()
    check(cuda.cuDeviceGet(ctypes.byref(device), 0), 'device')
    check(cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device), 'context')
    allocations = []

    def allocate(data, size=None):
        pointer = ctypes.c_uint64()
        check(cuda.cuMemAlloc_v2(ctypes.byref(pointer), size or len(data)), 'allocate')
        allocations.append(pointer.value)
        if data:
            check(cuda.cuMemcpyHtoD_v2(pointer.value, data, len(data)), 'upload')
        return pointer.value

    as_data = (FIXTURES / 'optix_as.bin').read_bytes()
    assert hashlib.sha256(as_data).hexdigest() == AS_SHA256
    base = allocate(b'', len(as_data))
    relocated = bytearray(as_data)
    struct.pack_into('<Q', relocated, 0xd0, base)
    check(cuda.cuMemcpyHtoD_v2(base, bytes(relocated), len(relocated)), 'AS upload')
    jit = (FIXTURES / 'jit_16.elf').read_bytes()
    section = next(row for row in sections(jit) if row['name'] == '.nv.global.init')
    dummy = allocate(jit[section['off']:section['off'] + 128], 4096)
    ox, oy = (50., 2.) if args.ray == 'miss' else (.25, .25)
    tmax = 6. if args.ray == 'short' else 100.
    values = operands(base, ox, oy, -5., 1., tmax, dummy, ray_flags=1)
    image, build_info = build(values, group=args.group, samples=args.samples, alu=args.alu,
                             placement=args.placement, open_gap=args.open_gap)
    if args.artifact:
        args.artifact.parent.mkdir(parents=True, exist_ok=True)
        args.artifact.write_bytes(image)
    probe = TimingProbe(image, lanes=args.lanes, samples=args.samples)
    failures, records = [], []
    before = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,clocks.sm,temperature.gpu,utilization.gpu',
                             '--format=csv'], capture_output=True, text=True, check=True).stdout
    fatal_error = None
    for launch in range(args.launches):
        try:
            lanes = probe.launch()
        except RuntimeError as error:
            fatal_error = str(error)
            break
        for lane, samples in enumerate(lanes):
            for sample, raw in enumerate(samples):
                clocks = [raw[r] | raw[r+1] << 32 for r in (80, 82, 84, 86)]
                accumulator = struct.unpack('<f', struct.pack('<I', raw[88]))[0]
                result = triangle_result(raw) if args.group in ('reduced', 'expanded') else None
                passed = (raw[78] == lane and accumulator == args.alu + (args.open_gap or 0)
                          and all(a <= b for a, b in zip(clocks, clocks[1:])))
                after_open = raw[90] | raw[91] << 32 if args.open_gap is not None else None
                if after_open is not None:
                    passed = passed and clocks[0] <= after_open <= clocks[1]
                if result is not None:
                    passed = passed and (result['kind'] == 'miss' if args.ray != 'hit' else
                              result['kind'] == 'triangle' and result['primitive'] == 0
                              and math.isclose(result['t'], 6.875, abs_tol=1e-6)
                              and result['barycentrics'] == [.125, .625] and result['front_face'] is False)
                if not passed:
                    failures.append(dict(launch=launch, lane=lane, sample=sample))
                records.append(dict(launch=launch, lane=lane, sample=sample, clocks=clocks,
                                    group_interval=clocks[1]-clocks[0], prewait_interval=clocks[2]-clocks[1],
                                    postwait_interval=clocks[3]-clocks[2], total=clocks[3]-clocks[0],
                                    accumulator=accumulator, result=result, passed=passed,
                                    after_open=after_open,
                                    open_issue_interval=after_open-clocks[0] if after_open is not None else None,
                                    raw=[f'{value:08x}' for value in raw]))
    # Lanes share warp scheduling: use only lane0, exclude first launch/sample.
    steady = [r for r in records if r['lane'] == 0 and r['launch'] > 0 and r['sample'] > 0]
    summary = {key: dict(min=min(r[key] for r in steady), median=statistics.median(r[key] for r in steady),
                         max=max(r[key] for r in steady))
               for key in ('group_interval', 'prewait_interval', 'postwait_interval', 'total')} if steady else {}
    if steady and args.open_gap is not None:
        summary['open_issue_interval'] = dict(min=min(r['open_issue_interval'] for r in steady),
                                              median=statistics.median(r['open_issue_interval'] for r in steady),
                                              max=max(r['open_issue_interval'] for r in steady))
    sources = ('run_warp_timing.py', 'warp_timing.py', 'timing_template.cu', 'cubin_builder.py',
               'sm86_fixture.py', 'ttu_isa.py', 'cuda_driver.py')
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(), args=vars(args) | {'output': str(args.output),
                   'artifact': str(args.artifact) if args.artifact else None},
                  contract='1 CTA/warp; identical per-lane rays, disjoint snapshots; restored inputs each sample; clocks and ALU outside MACROFUSE; no pure RT latency claim',
                  clock_definition=['before OPEN/MF0/empty', 'after complete group without explicit wait',
                                    'after wait-all and optional before-wait ALU', 'after optional after-wait ALU'],
                  summary_lane0_steady=summary, steady_samples=len(steady), records=records, failures=failures,
                  fatal_error=fatal_error, build=build_info, ray=[ox,oy,-5,0,0,0,1,tmax],
                  as_sha256=AS_SHA256, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), inventory_before=before,
                  compiler_version=subprocess.run(['nvcc', '--version'],capture_output=True,text=True,check=True).stdout,
                  source_sha256=hash_sources(sources))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(group=args.group, lanes=args.lanes, alu=args.alu, placement=args.placement,
                          records=len(records), failures=len(failures), fatal_error=fatal_error, summary=summary)))
    if not fatal_error:
        probe.close()
        for pointer in allocations:
            check(cuda.cuMemFree_v2(pointer), 'free')
        check(cuda.cuCtxDestroy_v2(context), 'context destroy')
    if args.check and (failures or fatal_error):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
