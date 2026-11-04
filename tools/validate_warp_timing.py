#!/usr/bin/env python3
"""One-warp calibration, valid TTU groups, and independent-ALU overlap matrix."""
from project_paths import bench_command
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', nargs='+', default=['5'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = []
    for lanes in (1, 32):
        for group in ('none', 'nop', 'mf0', 'reduced', 'expanded'):
            cases.append((lanes, group, 'hit', 0, 'before-wait', None))
        for group in ('none', 'reduced', 'expanded'):
            for alu in (16, 32, 64, 128):
                for placement in ('before-wait', 'after-wait'):
                    cases.append((lanes, group, 'hit', alu, placement, None))
        for group in ('reduced', 'expanded'):
            for ray in ('miss', 'short'):
                cases.append((lanes, group, ray, 0, 'before-wait', None))
            for gap in (0, 8, 16, 32, 64):
                cases.append((lanes, group, 'hit', 0, 'before-wait', gap))
    rows = []
    for gpu in args.gpus:
        for lanes, group, ray, alu, placement, gap in cases:
            label = f'gpu{gpu}-l{lanes}-{group}-{ray}-a{alu}-{placement}' + (f'-open{gap}' if gap is not None else '')
            output = args.output.with_name(f'{args.output.stem}-{label}.json')
            command = [*bench_command('run_warp_timing'), '--lanes', str(lanes), '--group', group,
                       '--ray', ray, '--alu', str(alu), '--placement', placement, '--samples', '4', '--launches', '12',
                       '--check', '--output', str(output)]
            if gap is not None:
                command += ['--open-gap', str(gap)]
            process = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
                                     capture_output=True, text=True, timeout=25)
            if process.returncode:
                raise RuntimeError(f'{label} failed; stopping matrix: {process.stdout}\n{process.stderr}')
            data = output.read_bytes()
            report = json.loads(data)
            if report['fatal_error'] or report['failures'] or len(report['records']) != lanes*4*12:
                raise RuntimeError(f'{label}: incomplete report')
            row = dict(gpu=gpu, lanes=lanes, group=group, ray=ray, alu=alu, placement=placement,
                       open_gap=gap,
                       report=output.name, sha256=hashlib.sha256(data).hexdigest(),
                       launches=12, warp_requests=48 if group in ('reduced','expanded') else 0,
                       lane_requests=lanes*48 if group in ('reduced','expanded') else 0,
                       lane_checks=len(report['records']), summary=report['summary_lane0_steady'])
            rows.append(row)
            print(json.dumps(dict(label=label, total=row['summary']['total']['median'],
                                  group=row['summary']['group_interval']['median'])), flush=True)
    summary = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                   contract='Single CTA/warp, 1 or 32 active lanes; preserved valid TTU groups; all lane results checked; clocks/ALU outside fused body; no preemption or inter-warp claim',
                   cases=rows, totals={key:sum(row[key] for row in rows)
                                     for key in ('launches','warp_requests','lane_requests','lane_checks')},
                   validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary['totals']))


if __name__ == '__main__':
    main()
