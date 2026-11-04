#!/usr/bin/env python3
"""Reproduce flat-IAS parent resubmission and first-child-only negative controls."""
from project_paths import ROOT as HERE, bench_command
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
    parser.add_argument('--gpus', nargs='+', default=['2', '6'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = [(case['name'], 'single') for case in
             json.loads((HERE / 'fixtures/phase17-parent-cases.json').read_text())['scenarios']]
    cases += [('chain3', 'grid'), ('scaled-triple', 'grid'), ('rotated-pair', 'separated')]
    rows = []
    for gpu in args.gpus:
        for scenario, scene in cases:
            path = args.output.with_name(f'{args.output.stem}-gpu{gpu}-{scenario}-{scene}.json')
            command = [*bench_command('run_parent_resume'), '--scenario', scenario,
                       '--scene', scene, '--count', '64', '--repeat', '2', '--check', '--output', str(path)]
            result = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
                                    timeout=40, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(f'{command}: {result.stdout}\n{result.stderr}')
            report = json.loads(path.read_text())
            if report['failures'] or report['fatal_error'] or not all(record['passed'] for record in report['records']):
                raise RuntimeError(f'incomplete or failed report: {path}')
            rows.append(dict(gpu=gpu, scenario=scenario, scene=scene, report=path.name,
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             launches=len(report['records']), requests=sum(record['requests'] for record in report['records']),
                             resumed_parent_requests=sum(record['steps'] - 1 for record in report['records'])))
            print(result.stdout.strip(), flush=True)
    controls = []
    for scenario in ('first-miss-later-hit', 'first-farther-later-nearer'):
        path = args.output.with_name(f'{args.output.stem}-control-{scenario}.json')
        command = [*bench_command('run_parent_resume'), '--scenario', scenario,
                   '--stop-after-first', '--check', '--output', str(path)]
        result = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus[0]),
                                timeout=40, text=True, capture_output=True)
        if result.returncode != 1:
            raise RuntimeError(f'negative control did not fail --check: {result.stdout}\n{result.stderr}')
        report = json.loads(path.read_text())
        record = report['records'][0]
        if (report['fatal_error'] or record['status'] != 9 or record['actual'] == record['expected']
                or record['steps'] != 1 or not record['pending_at_end']):
            raise RuntimeError(f'negative control does not demonstrate missing pending work: {path}')
        controls.append(dict(report=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             expected_returncode=1, actual_returncode=result.returncode, passed=True,
                             actual=record['actual'], expected=record['expected']))
    summary = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                   contract='Restricted flat 1-3 instance IAS, positive transforms, one shared triangle GAS; pending hardware references saved and resubmitted by device code',
                   cases=rows, negative_controls=controls,
                   validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   totals={key: sum(row[key] for row in rows)
                           for key in ('launches', 'requests', 'resumed_parent_requests')})
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['totals']))


if __name__ == '__main__':
    main()
