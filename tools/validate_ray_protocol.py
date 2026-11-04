#!/usr/bin/env python3
"""Validate opaque/culling requests and instance oracles in fresh processes."""
from project_paths import ROOT as HERE, bench_command
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', nargs='+', default=['0', '5'])
    parser.add_argument('--repeat', type=int, default=2)
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--output', type=Path, default=HERE / 'results/local-ray-protocol.json')
    args = parser.parse_args()
    if args.repeat < 1 or args.timeout < 1:
        parser.error('repeat and timeout must be positive')
    cases = [dict(name=f'flags-{flags:x}-geometry-{geometry}-winding-{winding}',
                  scene='grid', flags=flags, geometry=geometry, winding=winding, instances='none')
             for flags, geometry, winding in [(0, 1, 0), (1, 0, 0), (1, 1, 1),
                 (0x10, 1, 0), (0x20, 1, 1), (0x11, 0, 0), (0x21, 0, 0),
                 (0x11, 0, 1), (0x21, 0, 1)]]
    cases += [dict(name=f'instance-{mode}-oracle', scene='single', flags=1,
                   geometry=0, winding=0, instances=mode)
              for mode in ('identity', 'translate', 'rotate', 'scale', 'pair')]
    directory = args.output.parent / (args.output.stem + '-cases')
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for gpu in args.gpus:
        for case in cases:
            destination = directory / f'gpu{gpu}-{case["name"]}.json'
            command = [*bench_command('run_scene'), case['scene'], '--count', '64',
                       '--group', 'both', '--ray-suite', 'general', '--ray-flags', str(case['flags']),
                       '--instances', case['instances'], '--repeat', str(args.repeat),
                       '--check', '--output', str(destination)]
            if case['geometry']:
                command.append('--geometry-opaque')
            if case['winding']:
                command.append('--reverse-winding')
            if case['instances'] != 'none':
                command.append('--oracle-only')
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}
            for name in ('MINIMAL_NOTTU', 'MINIMAL_ZERO'):
                env.pop(name, None)
            try:
                run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=args.timeout)
                report = json.loads(destination.read_text()) if destination.exists() else {}
                passed = run.returncode == 0 and bool(report.get('optix')) and not report.get('failures')
                if case['instances'] == 'none':
                    passed = passed and bool(report.get('records'))
                records.append(dict(gpu=gpu, name=case['name'], passed=passed, command=command,
                                    report=str(destination.resolve()), launches=len(report.get('records', [])),
                                    oracle_rays=len(report.get('optix', [])), returncode=run.returncode,
                                    stdout=run.stdout, stderr=run.stderr))
            except subprocess.TimeoutExpired:
                records.append(dict(gpu=gpu, name=case['name'], passed=False, timeout=True,
                                    command=command, launches=0, oracle_rays=0))
            print(f'GPU{gpu} {case["name"]}: {"PASS" if records[-1]["passed"] else "FAIL"}', flush=True)
            if not records[-1]['passed']:
                break
        if not records[-1]['passed']:
            break
    result = dict(timestamp=datetime.now(timezone.utc).isoformat(), cases=records,
                  passed=all(record['passed'] for record in records),
                  launches=sum(record['launches'] for record in records),
                  oracle_rays=sum(record['oracle_rays'] for record in records))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=result['passed'], cases=len(records), launches=result['launches'])))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
