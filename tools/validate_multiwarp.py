#!/usr/bin/env python3
"""Compact multiwarp baseline; retain one summary and discard routine case files."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile

from project_paths import bench_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', nargs='+', default=['1'])
    parser.add_argument('--warps', nargs='+', type=int, choices=(1, 2, 4, 8, 16), default=[1, 2, 4, 8, 16])
    parser.add_argument('--iterations', nargs='+', type=int, default=[32, 128, 512])
    parser.add_argument('--launches', type=int, default=12)
    parser.add_argument('--checking', choices=('each', 'final'), default='each')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.launches < 3 or any(not 1 <= iterations <= 4096 for iterations in args.iterations):
        parser.error('launches >=3 and iterations 1..4096 required')
    cases, environments = [], {}
    with tempfile.TemporaryDirectory(prefix='ttu-multiwarp-matrix-') as directory:
        path = Path(directory) / 'case.json'
        for gpu in args.gpus:
            for iterations in args.iterations:
                for warps in args.warps:
                    for group in ('none', 'reduced', 'expanded'):
                        command = bench_command('run_multiwarp') + ['--warps', str(warps), '--iterations', str(iterations),
                            '--group', group, '--launches', str(args.launches), '--checking', args.checking, '--output', str(path)]
                        result = subprocess.run(command, env=os.environ | {'CUDA_VISIBLE_DEVICES': str(gpu)},
                                                capture_output=True, text=True, timeout=30)
                        if result.returncode:
                            raise RuntimeError(f'GPU{gpu}, {warps} warps, {iterations} iterations, {group}: {result.stderr or result.stdout}')
                        report = json.loads(path.read_text())
                        environments[str(gpu)] = dict(cuda_visible_devices=report['cuda_visible_devices'],
                                                     inventory_before=report['inventory_before'])
                        row = dict(gpu=str(gpu), config=report['config'], resources=report['resources'],
                                   summary=report['summary'], failures=len(report['failures']), build=report['build'],
                                   smids=sorted({warp['smid'] for launch in report['launches'] for warp in launch['warps']}),
                                   verified_lane_requests=report['verified_lane_requests'],
                                   warmup=report['warmup'],
                                   source_sha256=report['source_sha256'])
                        cases.append(row)
                        print(json.dumps(dict(gpu=gpu, group=group, warps=warps, iterations=iterations,
                                              cycles=report['summary']['span_cycles']['median'])), flush=True)
                        path.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(), cases=cases, environments=environments,
        contract='All lanes checked each request by default; one CTA per launch, no device saturation claim. Raw case files discarded.',
        passed=True), indent=2) + '\n')


if __name__ == '__main__':
    main()
