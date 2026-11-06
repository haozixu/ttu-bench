#!/usr/bin/env python3
"""Compact sync, split-group and TTU/FADD role comparisons; discard per-case reports."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile

from project_paths import ROOT


def cases(suite):
    if suite in ('mapping', 'window', 'return-controls'):
        common = dict(window_cycles=300000, ld_stall=2, wait_lead=0, consumer_delay=0,
                      wait_mask=2, wait_site='consumer', compute_chains=4, compute_count=256)
        if suite == 'mapping':
            for first in range(4):
                for second in range(4, 8):
                    yield dict(common, warps=8, ttu_mask=0, compute_mask=(1 << first) | (1 << second)), True
            for warp in range(8):
                yield dict(common, warps=8, ttu_mask=0, compute_mask=1 << warp), True
            for residue in range(4):
                yield dict(common, warps=16, ttu_mask=0, compute_mask=0x1111 << residue), True
                yield dict(common, warps=16, ttu_mask=0, compute_mask=15 << (residue * 4)), True
        elif suite == 'window':
            for warps, ttu_mask, compute_mask in ((8, 1, 16), (8, 1, 2), (8, 17, 34),
                                                   (8, 15, 240), (16, 0x5555, 0xaaaa),
                                                   (16, 0x1111, 0x2222), (16, 0x0101, 0x1010),
                                                   (16, 0x0055, 0x5500), (16, 0x0055, 0x00aa)):
                for selected_ttu, selected_compute in ((ttu_mask, 0), (0, compute_mask), (ttu_mask, compute_mask)):
                    yield dict(common, warps=warps, ttu_mask=selected_ttu, compute_mask=selected_compute), True
            yield dict(common, warps=16, ttu_mask=0xffff, compute_mask=0), True
        else:
            for warps in (4, 16):
                for mode in ('prefix', 'sink', 'repeat'):
                    for loads in (1, 2, 4, 7):
                        yield dict(common, warps=warps, group='expanded', restore='all',
                                   return_loads=loads, return_mode=mode), True
            for loads in (1, 7):
                for gap in (1, 64, 256):
                    for site in ('before-close', 'after-close'):
                        yield dict(common, warps=16, group='expanded', restore='all', return_loads=loads,
                                   return_mode='sink', close_gap=gap, close_gap_site=site, gap_warps=1), True
    elif suite == 'sync':
        for site in ('clock', 'nop', 'consumer'):
            for stall in (1, 2, 4):
                yield dict(warps=8, iterations=1, wait_lead=0, consumer_delay=0,
                           wait_site=site, ld_stall=stall, wait_mask=2), stall != 1
        for barrier, mask in ((1, 0), (1, 4), (2, 4), (2, 2), (4, 16), (4, 2)):
            yield dict(warps=8, iterations=1, wait_lead=0, consumer_delay=0,
                       ld_stall=4, return_barrier=barrier, wait_mask=mask), mask == 1 << barrier
        for group in ('reduced', 'expanded'):
            for ray in ('hit', 'miss', 'short'):
                for warps in (1, 4, 16):
                    yield dict(group=group, ray=ray, warps=warps, iterations=128, wait_lead=0,
                               consumer_delay=0, ld_stall=2, wait_mask=2, wait_site='consumer'), True
    elif suite == 'split':
        for warps in (2, 4, 8, 16):
            for variant in ('before-open', 'open-gap', 'split9', 'split10', 'after-close', 'after-wait'):
                yield dict(warps=warps, iterations=1, variant=variant, gap=2048,
                           gap_warps=1, trace=True), True
                if variant in ('before-open', 'open-gap', 'split9', 'split10'):
                    yield dict(warps=warps, iterations=1, variant=variant, gap=2048,
                               gap_warps=1, trace=True, peer_delay=256), True
    elif suite == 'scaling':
        for group in ('reduced', 'expanded', 'none'):
            for warps in (1, 2, 4, 8, 16):
                for restore in ('returned', 'all'):
                    yield dict(group=group, warps=warps, iterations=512, restore=restore,
                               ld_stall=2, wait_lead=0, consumer_delay=0, wait_mask=2,
                               wait_site='consumer'), True
        for variant in ('before-open', 'split9', 'split10', 'after-wait'):
            for gap in (0, 16, 64, 256):
                yield dict(warps=8, iterations=128, variant=variant, gap=gap,
                           ld_stall=2, wait_lead=0, consumer_delay=0, wait_mask=2,
                           wait_site='consumer', restore='all'), True
    elif suite == 'mixed':
        for warps in (4, 8, 16):
            compute_iterations = max(768, warps * 96)
            for ttu_warps in (0, warps // 2, warps):
                for role_start in (range(warps - ttu_warps + 1) if ttu_warps == warps // 2 else (0,)):
                    for group in ('reduced', 'none'):
                        yield dict(warps=warps, ttu_warps=ttu_warps, role_start=role_start,
                                   group=group, iterations=256, compute_iterations=compute_iterations, compute_count=64,
                                   ld_stall=2, wait_lead=0, consumer_delay=0,
                                   wait_mask=2, wait_site='consumer'), True
            for role_start in (0, warps // 2):
                for baseline in ('ttu', 'fadd'):
                    yield dict(warps=warps, ttu_warps=warps // 2, role_start=role_start,
                               group='reduced' if baseline == 'ttu' else 'none',
                               iterations=256 if baseline == 'ttu' else 1,
                               compute_iterations=1 if baseline == 'ttu' else compute_iterations,
                               compute_count=1 if baseline == 'ttu' else 64,
                               ld_stall=2, wait_lead=0, consumer_delay=0, wait_mask=2,
                               wait_site='consumer'), True
    elif suite == 'issue':
        for warps in (4, 8, 16):
            for role_start in (0, warps // 2):
                for baseline in ('mixed', 'ttu', 'fadd', 'none'):
                    yield dict(warps=warps, ttu_warps=warps // 2, role_start=role_start,
                               group='reduced' if baseline in ('mixed', 'ttu') else 'none',
                               iterations=1 if baseline == 'fadd' else 256,
                               compute_iterations=1 if baseline == 'ttu' else max(1024, warps * 128),
                               compute_count=1 if baseline == 'ttu' else 64, compute_chains=4,
                               ld_stall=2, wait_lead=0, consumer_delay=0, wait_mask=2,
                               wait_site='consumer'), True
    elif suite == 'returns':
        for warps in (1, 4, 8, 16):
            for return_loads in range(1, 8):
                yield dict(warps=warps, group='expanded', return_loads=return_loads,
                           restore='all', iterations=256, ld_stall=2, wait_lead=0,
                           consumer_delay=0, wait_mask=2, wait_site='consumer'), True
        for warps in (8, 16):
            for return_loads in range(1, 8):
                yield dict(warps=warps, group='expanded', return_loads=return_loads, return_pad=True,
                           restore='all', iterations=256, ld_stall=2, wait_lead=0,
                           consumer_delay=0, wait_mask=2, wait_site='consumer'), True


def compact(report):
    result = dict(summary=report['summary'], resources=report['resources'], build=report['build'],
                  source_sha256=report['source_sha256'], verified_lane_requests=report['verified_lane_requests'],
                  failures=len(report['failures']), fatal_error=report['fatal_error'])
    if report['failures']:
        result['failure_sample'] = report['failures'][:2]
    if report['config']['trace']:
        traces = []
        for launch in report['launches'][2:]:
            holder = launch['warps'][0]
            begin, end = holder['markers']['before_gap'], holder['markers']['after_gap']
            peers = launch['warps'][1:]
            traces.append(dict(gap_cycles=end-begin,
                               peer_open_inside=sum(begin < row['markers']['after_open'] < end for row in peers),
                               peer_close_inside=sum(begin < row['markers']['after_close'] < end for row in peers),
                               peer_wait_inside=sum(begin < row['markers']['after_wait'] < end for row in peers),
                               peers=[dict(warp=row['warp'], start=row['start']-begin,
                                           **{name: value-begin for name, value in row['markers'].items()
                                              if name not in ('before_gap', 'after_gap')}) for row in peers]))
        result['traces'] = traces
    if report['launches'][2:]:
        result['warp_cycles_median'] = [statistics.median(launch['warps'][warp]['cycles']
                                       for launch in report['launches'][2:])
                                       for warp in range(report['config']['warps'])]
        result['role_overlap_fraction_median'] = {
            role: statistics.median(launch['roles'][role]['overlap_fraction'] for launch in report['launches'][2:])
            for role in ('ttu', 'fadd') if 'overlap_fraction' in report['launches'][2]['roles'].get(role, {})}
        result['warp_virtual_ids'] = sorted({tuple(row['virtual_warp'] for row in launch['warps'])
                                             for launch in report['launches'][2:]})
        if report['config']['window_cycles']:
            result['window_warps'] = [dict(warp=warp, role=report['launches'][2]['warps'][warp]['role'],
                                          completed_median=statistics.median(launch['warps'][warp].get('window', {}).get('completed', 0)
                                                                            for launch in report['launches'][2:]))
                                      for warp in range(report['config']['warps'])]
            result['max_late_start_cycles'] = max(row['window']['late_start_cycles']
                                                  for launch in report['launches'][2:] for row in launch['warps'] if 'window' in row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('sync', 'split', 'scaling', 'mixed', 'issue', 'returns',
                                          'mapping', 'window', 'return-controls'), required=True)
    parser.add_argument('--gpus', default='6')
    parser.add_argument('--launches', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--seed', type=int, help='shuffle independent cases reproducibly')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.launches < 3 or args.repeats < 1:
        parser.error('launches >=3 and repeats >=1 required')
    rows = []
    environments = {}
    passed = True
    selected_cases = list(cases(args.suite)) * args.repeats
    if args.seed is not None:
        random.Random(args.seed).shuffle(selected_cases)
    with tempfile.TemporaryDirectory(prefix='ttu-interactions-') as directory:
        for gpu in args.gpus.split(','):
            for index, (config, expected_pass) in enumerate(selected_cases):
                output = Path(directory) / 'case.json'
                output.unlink(missing_ok=True)
                command = [sys.executable, str(ROOT / 'bench.py'), 'run_multiwarp',
                           '--launches', str(args.launches), '--warmup-seconds', '.2', '--output', str(output)]
                for key, value in config.items():
                    command.append('--' + key.replace('_', '-'))
                    if value is not True:
                        command.append(str(value))
                try:
                    process = subprocess.run(command, cwd=ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu},
                                             capture_output=True, text=True, timeout=30)
                except subprocess.TimeoutExpired:
                    rows.append(dict(gpu=gpu, config=config, expected_pass=expected_pass, valid=False, timeout=True))
                    passed = False
                    break
                report = json.loads(output.read_text()) if output.exists() else None
                if report is not None:
                    environments.setdefault(gpu, report['inventory_before'])
                valid = (report is not None and report['fatal_error'] is None and
                         (process.returncode == 0 and not report['failures'] if expected_pass else
                          process.returncode == 1 and bool(report['failures'])))
                if valid and not expected_pass:
                    valid = all(failure.get('error_bits', 0) != 0 and failure.get('counter') == 0 and
                                failure.get('returned') == report['expected'] and
                                failure.get('gap_bits') == '00000000' and failure.get('compute_bits') == '00000000'
                                for failure in report['failures'])
                if valid and config.get('trace') and config.get('peer_delay'):
                    for launch in report['launches'][2:]:
                        holder = launch['warps'][0]['markers']
                        begin, end = holder['before_gap'], holder['after_gap']
                        valid &= all(begin < peer['markers']['after_open'] < peer['markers']['after_close'] <
                                     peer['markers']['after_wait'] < end for peer in launch['warps'][1:])
                row = dict(gpu=gpu, config=config, expected_pass=expected_pass, valid=valid,
                           exit=process.returncode, result=compact(report) if report else None)
                if not valid:
                    row['stderr'] = process.stderr[-1000:]
                rows.append(row)
                passed &= valid
                print(json.dumps(dict(gpu=gpu, case=index, config=config, valid=valid,
                                      failures=len(report['failures']) if report else None,
                                      summary=report['summary'] if report else None)), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(),
                                          suite=args.suite, repeats=args.repeats, seed=args.seed,
                                          passed=passed, environments=environments,
                                          rows=rows), indent=2) + '\n')
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
