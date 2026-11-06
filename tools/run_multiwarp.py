#!/usr/bin/env python3
"""Measure bounded multiwarp request throughput in one CTA on one SM."""
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
import time

from cuda_driver import check, cuda, sections
from multiwarp import MultiwarpProbe, RETURN_REGISTERS, build
from project_paths import hash_sources
from sm86_fixture import AS_SHA256, FIXTURES, f32, operands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warps', type=int, choices=(1, 2, 4, 8, 16), default=1)
    parser.add_argument('--group', choices=('none', 'reduced', 'expanded'), default='reduced')
    parser.add_argument('--iterations', type=int, default=128)
    parser.add_argument('--launches', type=int, default=12)
    parser.add_argument('--warmup-seconds', type=float, default=.2)
    parser.add_argument('--checking', choices=('each', 'final'), default='each')
    parser.add_argument('--ray', choices=('hit', 'miss', 'short'), default='hit')
    parser.add_argument('--wait-lead', type=int, default=2, help='0..8 NOPs between LD and scoreboard wait; calibration only')
    parser.add_argument('--consumer-delay', type=int, default=15, help='0..15 scheduled NOP cycles after wait; calibration only')
    parser.add_argument('--restore', choices=('returned', 'all'), default='returned')
    parser.add_argument('--ld-stall', type=int, choices=range(1, 16))
    parser.add_argument('--return-barrier', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--wait-mask', type=lambda value: int(value, 0), default=63)
    parser.add_argument('--wait-site', choices=('clock', 'nop', 'consumer'), default='clock')
    parser.add_argument('--variant', choices=('normal', 'before-open', 'open-gap', 'split9', 'split10', 'after-close', 'after-wait'), default='normal')
    parser.add_argument('--gap', type=int, default=0, help='dependent FADDs on the selected TTU warps')
    parser.add_argument('--gap-warps', type=int, default=16, help='only warp IDs below this value execute the gap')
    parser.add_argument('--trace', action='store_true', help='single request timestamps outside fused bodies')
    parser.add_argument('--ttu-warps', type=int, help='number of contiguous TTU-role warps; others execute FADD')
    parser.add_argument('--role-start', type=int, default=0)
    parser.add_argument('--compute-count', type=int, default=64)
    parser.add_argument('--compute-iterations', type=int)
    parser.add_argument('--peer-delay', type=int, default=0, help='one-time FADD delay on TTU warp IDs >= gap-warps')
    parser.add_argument('--compute-chains', type=int, choices=(1, 4), default=1)
    parser.add_argument('--return-loads', type=int, choices=range(1, 8), help='expanded group: retain N original return loads; see return-mode')
    parser.add_argument('--return-pad', action='store_true', help='replace omitted return loads with two-cycle NOPs outside MF')
    parser.add_argument('--return-mode', choices=('prefix', 'sink', 'repeat'), default='prefix',
                        help='prefix truncates; sink/repeat retain seven LDs with unused destinations RZ or repeated first LD')
    parser.add_argument('--close-gap', type=int, default=0, help='FADDs around final LD.CLOSE on warp IDs below gap-warps')
    parser.add_argument('--close-gap-site', choices=('before-close', 'after-close'), default='before-close')
    parser.add_argument('--window-cycles', type=int, default=0, help='common device-clock measurement duration; 0 uses fixed iteration counts')
    parser.add_argument('--ttu-mask', type=lambda value: int(value, 0))
    parser.add_argument('--compute-mask', type=lambda value: int(value, 0))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifacts', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.iterations <= 4096 or args.launches < 3:
        parser.error('iterations 1..4096 and launches >=3 required')
    if not math.isfinite(args.warmup_seconds) or not 0 <= args.warmup_seconds <= 10:
        parser.error('warmup-seconds must be finite and in 0..10')
    if not 0 <= args.wait_lead <= 8 or not 0 <= args.consumer_delay <= 15:
        parser.error('wait-lead 0..8 and consumer-delay 0..15 required')
    if not 0 <= args.wait_mask <= 63 or (args.wait_site == 'consumer' and args.checking != 'each'):
        parser.error('wait-mask 0..63; consumer wait requires checking each')
    args.ttu_warps = args.warps if args.ttu_warps is None else args.ttu_warps
    args.compute_iterations = args.iterations if args.compute_iterations is None else args.compute_iterations
    if not 0 <= args.ttu_warps <= args.warps or not 0 <= args.role_start <= args.warps - args.ttu_warps:
        parser.error('TTU role range must fit within the CTA')
    if not 0 <= args.gap <= 4096 or not 0 <= args.gap_warps <= 16 or not 1 <= args.compute_count <= 1024 or not 1 <= args.compute_iterations <= 4096:
        parser.error('invalid gap or compute workload')
    if args.trace and (args.iterations != 1 or args.wait_site == 'consumer'):
        parser.error('trace requires iterations=1 and clock/nop wait')
    if args.group == 'none' and args.variant not in ('normal', 'before-open', 'after-close', 'after-wait'):
        parser.error('protocol gap requires TTU group')
    if not 0 <= args.peer_delay <= 1024:
        parser.error('peer-delay must be 0..1024')
    if args.return_loads is not None and args.group != 'expanded':
        parser.error('return-loads requires group expanded')
    if args.return_pad and args.return_loads is None:
        parser.error('return-pad requires return-loads')
    if (args.ttu_mask is None) != (args.compute_mask is None) or (args.ttu_mask is not None and
            (args.ttu_mask < 0 or args.compute_mask < 0 or (args.ttu_mask | args.compute_mask) >= 1 << args.warps or args.ttu_mask & args.compute_mask)):
        parser.error('role masks must be paired, disjoint and fit the CTA')
    if args.ttu_mask is not None:
        args.ttu_warps = args.ttu_mask.bit_count()
    if args.window_cycles and (not 10000 <= args.window_cycles <= 1000000 or args.trace or args.checking != 'each' or args.peer_delay):
        parser.error('window requires 10000..1000000 cycles, checking each, no trace/peer delay')
    if args.return_mode != 'prefix' and args.return_loads is None:
        parser.error('return-mode requires return-loads')
    if not 0 <= args.close_gap <= 1024 or (args.close_gap and (args.return_mode == 'prefix' or args.trace)):
        parser.error('close-gap 0..1024 requires sink/repeat mode and no trace')
    inventory = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,clocks.sm,utilization.gpu',
                                '--format=csv'], capture_output=True, text=True, check=True).stdout
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

    original = (FIXTURES / 'optix_as.bin').read_bytes()
    if hashlib.sha256(original).hexdigest() != AS_SHA256:
        raise ValueError('AS fixture hash mismatch')
    base = allocate(b'', len(original))
    relocated = bytearray(original)
    struct.pack_into('<Q', relocated, 0xd0, base)
    check(cuda.cuMemcpyHtoD_v2(base, bytes(relocated), len(relocated)), 'upload relocated AS')
    jit = (FIXTURES / 'jit_16.elf').read_bytes()
    section = next(row for row in sections(jit) if row['name'] == '.nv.global.init')
    dummy = allocate(jit[section['off']:section['off'] + 128], 4096)
    ox, oy = (50., 2.) if args.ray == 'miss' else (.25, .25)
    tmax = 6. if args.ray == 'short' else 100.
    values = operands(base, ox, oy, -5., 1., tmax, dummy, ray_flags=1)
    expected = (0xa0000000, f32(6.875), f32(-.125), f32(.625)) if args.ray == 'hit' else (0xffffffff,) * 4
    image, info = build(values, expected, group=args.group, iterations=args.iterations, checking=args.checking,
                       wait_lead=args.wait_lead, consumer_delay=args.consumer_delay, restore_mode=args.restore,
                       ld_stall=args.ld_stall, return_barrier=args.return_barrier, wait_mask=args.wait_mask, wait_site=args.wait_site,
                       variant=args.variant, gap=args.gap, gap_warps=args.gap_warps, trace=args.trace,
                       ttu_warps=args.ttu_warps, role_start=args.role_start, compute_count=args.compute_count,
                       compute_iterations=args.compute_iterations, peer_delay=args.peer_delay,
                       compute_chains=args.compute_chains, return_loads=args.return_loads, return_pad=args.return_pad,
                       window_cycles=args.window_cycles, ttu_mask=args.ttu_mask, compute_mask=args.compute_mask,
                       return_mode=args.return_mode, close_gap=args.close_gap, close_gap_site=args.close_gap_site)
    probe = MultiwarpProbe(image, warps=args.warps, shared_bytes=info['dynamic_shared_bytes'])
    launches, failures, raw_launches = [], [], []
    verified_requests = 0

    def analyze(raw, guards, launch):
        nonlocal verified_requests
        warp_rows = []
        smids = set()
        for thread, words in enumerate(raw):
            warp = thread // 32
            ttu_role = bool(args.ttu_mask & (1 << warp)) if args.ttu_mask is not None else args.role_start <= warp < args.role_start + args.ttu_warps
            compute_role = bool(args.compute_mask & (1 << warp)) if args.compute_mask is not None else not ttu_role
            active = ttu_role or compute_role
            iterations = words[86] if args.window_cycles and active else args.iterations if ttu_role else args.compute_iterations
            returned = tuple(words[register] for register in RETURN_REGISTERS)
            start, stop = (words[80] | words[81] << 32), (words[82] | words[83] << 32)
            smids.update((words[79], words[84]))
            valid = words[78] == thread and words[79] == words[84] and start < stop and words[88] == 0
            valid = valid and (0 < words[4] <= words[86] <= words[4] + 1 if args.window_cycles and active else words[86] == (0 if active or args.window_cycles else args.iterations))
            gap_count = args.gap * iterations if ttu_role and warp < args.gap_warps and args.variant != 'normal' else 0
            gap_count += args.close_gap * iterations if ttu_role and warp < args.gap_warps else 0
            valid = valid and words[85] == f32(gap_count)
            if ttu_role:
                valid = valid and returned == expected
                if args.group != 'none':
                    verified_requests += iterations
            elif compute_role:
                for chain, register in enumerate((87, 89, 90, 91)[:args.compute_chains]):
                    operations = (args.compute_count + args.compute_chains - 1 - chain) // args.compute_chains
                    valid = valid and words[register] == f32(operations * iterations)
            if ttu_role:
                valid = valid and words[87] == f32(args.peer_delay if warp >= args.gap_warps else 0)
            if thread % 32:
                valid = valid and (words[80:84] == raw[thread - thread % 32][80:84])
                if args.window_cycles:
                    leader = raw[thread - thread % 32]
                    valid = valid and words[4:11] == leader[4:11] and words[86] == leader[86]
                    valid = valid and words[11] >> 8 == leader[11] >> 8
            else:
                row = dict(warp=warp, role='ttu' if ttu_role else 'fadd' if compute_role else 'idle', smid=words[79],
                           virtual_warp=(words[11] >> 8) & 127, virtid_raw=words[11], start=start, stop=stop, cycles=stop-start)
                if args.window_cycles and active:
                    epoch = (start & ~0xffffffff) | words[10]
                    if epoch > start:
                        epoch -= 1 << 32
                    begin, end = epoch + 4096, epoch + 4096 + args.window_cycles
                    first, last = words[6] | words[7] << 32, words[8] | words[9] << 32
                    valid = valid and begin <= first < end <= last <= stop
                    row['window'] = dict(begin=begin, end=end, first=first, last=last,
                                         completed=words[4], total=words[86], late_start_cycles=first-begin)
                if args.trace and ttu_role:
                    row['markers'] = {name: words[register] | words[register + 1] << 32
                                      for name, register in (('after_open', 4), ('before_gap', 6), ('after_gap', 8),
                                                             ('after_close', 18), ('after_wait', 20))}
                    markers = row['markers']
                    valid = valid and start < markers['after_close'] < markers['after_wait'] < stop
                    if args.group != 'none':
                        valid = valid and start < markers['after_open'] < markers['after_close']
                    if gap_count:
                        valid = valid and start < markers['before_gap'] < markers['after_gap'] < stop
                warp_rows.append(row)
            if not valid:
                failures.append(dict(launch=launch, thread=thread, error_bits=words[88], counter=words[86],
                                     returned=[f'{word:08x}' for word in returned], gap_bits=f'{words[85]:08x}',
                                     compute_bits=f'{words[87]:08x}'))
        if not guards or len(smids) != 1:
            failures.append(dict(launch=launch, guards=guards, smids=sorted(smids)))
        span = max(row['stop'] for row in warp_rows) - min(row['start'] for row in warp_rows)
        roles = {}
        for role in ('ttu', 'fadd'):
            selected = [row for row in warp_rows if row['role'] == role]
            if selected:
                if args.window_cycles:
                    start, stop = selected[0]['window']['begin'], selected[0]['window']['end']
                    valid_window = all(row['window']['begin'] == start and row['window']['end'] == stop for row in selected)
                    if not valid_window:
                        failures.append(dict(launch=launch, role=role, mismatched_window=True))
                    operations = sum(row['window']['completed'] for row in selected) * 32 * (1 if role == 'ttu' else args.compute_count)
                else:
                    start, stop = min(row['start'] for row in selected), max(row['stop'] for row in selected)
                    operations = len(selected) * 32 * (args.iterations if role == 'ttu' else args.compute_iterations * args.compute_count)
                roles[role] = dict(start=start, stop=stop, span_cycles=stop-start,
                                   operations_per_1000_cycles=1000 * operations / (stop-start))
        if len(roles) == 2:
            overlap = max(0, min(row['stop'] for row in roles.values()) - max(row['start'] for row in roles.values()))
            for row in roles.values():
                row['overlap_fraction'] = overlap / row['span_cycles']
        ttu_iterations = (sum(row['window']['total'] for row in warp_rows if row['role'] == 'ttu')
                          if args.window_cycles else args.ttu_warps * args.iterations)
        return dict(launch=launch, span_cycles=span, warps=warp_rows, roles=roles,
                    common_interval_cycles=max(0, min(row['stop'] for row in warp_rows) - max(row['start'] for row in warp_rows)),
                    warp_iterations_per_cycle=ttu_iterations / span,
                    lane_iterations_per_cycle=ttu_iterations * 32 / span)

    fatal_error = None
    warmup_start = time.monotonic()
    warmup_launches = 0
    try:
        while time.monotonic() - warmup_start < args.warmup_seconds:
            analyze(*probe.launch(), f'warmup-{warmup_launches}')
            warmup_launches += 1
            if failures:
                break
        warmup_elapsed = time.monotonic() - warmup_start
        for launch in range(args.launches):
            raw, guards = probe.launch()
            if args.artifacts:
                raw_launches.append(raw)
            launches.append(analyze(raw, guards, launch))
    except RuntimeError as error:
        fatal_error = str(error)
        warmup_elapsed = time.monotonic() - warmup_start
    steady = launches[2:]
    summary = {key: dict(min=min(row[key] for row in steady), median=statistics.median(row[key] for row in steady),
                        max=max(row[key] for row in steady))
               for key in ('span_cycles', 'warp_iterations_per_cycle', 'lane_iterations_per_cycle')} if steady else {}
    for role in ('ttu', 'fadd'):
        rates = [row['roles'][role]['operations_per_1000_cycles'] for row in steady if role in row['roles']]
        if rates:
            summary[role + '_operations_per_1000_cycles'] = dict(min=min(rates), median=statistics.median(rates), max=max(rates))
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(), config={key: value for key, value in vars(args).items() if key not in ('output', 'artifacts')},
                  contract='One CTA, full warps, contiguous roles or disjoint explicit role masks with idle controls. One serial request per TTU lane; hot fixed opaque GAS, identical rays. Window mode uses a shared clock epoch +4096 and a common fixed duration; role rates count completed iterations timestamped before the deadline, excluding the final crossing iteration. All issued iterations are checked. Clock/check/loop overhead is included. CTA rates include start guard and crossing iterations. Fixed-count mode uses each role span. VIRTID is a virtual warp identifier, not a direct SMSP register. None uses four result MOVs. Skip first two launches; clocks unlocked.',
                  checking=args.checking, expected=[f'{word:08x}' for word in expected], build=info, resources=probe.attributes,
                  as_sha256=AS_SHA256, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), inventory_before=inventory,
                  warmup=dict(launches=warmup_launches, elapsed_seconds=warmup_elapsed, checked=True),
                  verified_lane_requests=verified_requests if args.checking == 'each' and not failures and not fatal_error else None,
                  summary=summary, launches=launches, failures=failures, fatal_error=fatal_error,
                  source_sha256=hash_sources(('run_multiwarp.py', 'multiwarp.py', 'multiwarp_template.cu', 'cubin_builder.py',
                                             'mixed_timing.py', 'warp_timing.py', 'ttu_isa.py', 'sm86_fixture.py', 'cuda_driver.py')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if args.artifacts:
        args.output.with_suffix('.cubin').write_bytes(image)
        args.output.with_suffix('.raw.json').write_text(json.dumps(raw_launches) + '\n')
    print(json.dumps(dict(config=report['config'], failures=len(failures), fatal_error=fatal_error, summary=summary)))
    if not fatal_error:
        probe.close()
        for pointer in allocations:
            check(cuda.cuMemFree_v2(pointer), 'free')
        check(cuda.cuCtxDestroy_v2(context), 'destroy context')
    if failures or fatal_error:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
