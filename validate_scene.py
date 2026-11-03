#!/usr/bin/env python3
"""Run the limited live-GAS matrix; one fresh process/context per scene/GPU."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', nargs='+', default=['7'])
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--ray-suite', choices=('legacy', 'general'), default='legacy')
    parser.add_argument('--geometry-opaque', action='store_true')
    parser.add_argument('--reverse-winding', action='store_true')
    parser.add_argument('--ray-flags', default='1')
    parser.add_argument('--output', type=Path, default=HERE / 'results/local-scenes.json')
    args = parser.parse_args()
    if args.repeat < 1 or args.timeout < 1:
        parser.error('repeat and timeout must be positive')
    cases = [('single', None), ('separated', None), ('overlap', None),
             ('overlap-reversed', None)] + [('grid', n) for n in (16, 64, 257, 1024)]
    directory = args.output.parent / (args.output.stem + '-cases')
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for gpu in args.gpus:
        for scene, count in cases:
            destination = directory / f'gpu{gpu}-{scene}{count or ""}.json'
            command = [sys.executable, str(HERE / 'run_scene.py'), scene, '--group', 'both',
                       '--repeat', str(args.repeat), '--check', '--output', str(destination),
                       '--ray-suite', args.ray_suite, '--ray-flags', args.ray_flags]
            if args.geometry_opaque:
                command.append('--geometry-opaque')
            if args.reverse_winding:
                command.append('--reverse-winding')
            if count:
                command += ['--count', str(count)]
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}
            for name in ('MINIMAL_NOTTU', 'MINIMAL_ZERO'):
                env.pop(name, None)
            try:
                run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=args.timeout)
                report = json.loads(destination.read_text()) if destination.exists() else {}
                passed = run.returncode == 0 and bool(report.get('records')) and not report.get('failures')
                records.append(dict(gpu=gpu, scene=scene, count=count, command=command,
                    returncode=run.returncode, passed=passed, report=str(destination.resolve()),
                    launches=len(report.get('records', [])), stdout=run.stdout, stderr=run.stderr))
            except subprocess.TimeoutExpired:
                records.append(dict(gpu=gpu, scene=scene, count=count, command=command,
                                    passed=False, timeout=True, launches=0))
            print(f'GPU{gpu} {scene} {count or ""}: {"PASS" if records[-1]["passed"] else "FAIL"}', flush=True)
            if not records[-1]['passed']:
                break
        if not records[-1]['passed']:
            break
    result = dict(timestamp=datetime.now(timezone.utc).isoformat(), cases=records,
                  passed=all(case['passed'] for case in records),
                  launches=sum(case['launches'] for case in records))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=result['passed'], cases=len(records), launches=result['launches'])))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
