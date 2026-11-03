#!/usr/bin/env python3
"""Reproduce the single-instance device-dispatch matrix in fresh processes."""
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
    parser.add_argument('--gpus', nargs='+', default=['1', '2'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = [('identity', 'single', 16, 17, 3), ('translate', 'single', 16, 41, 2),
             ('rotate', 'single', 16, 0, 2), ('scale', 'single', 16, 0xffffff, 2),
             ('identity', 'grid', 64, 41, 2), ('scale', 'grid', 64, 17, 2),
             ('identity', 'overlap', 16, 17, 2), ('rotate', 'separated', 16, 41, 2)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (mode, scene, count, instance_id, repeat) in enumerate(cases):
        gpu = args.gpus[(index // 4) % len(args.gpus)]
        path = args.output.with_name(f'{args.output.stem.removesuffix("-matrix")}-{mode}-{scene}.json')
        command = [sys.executable, str(Path(__file__).with_name('run_device_instance.py')),
                   '--instance', mode, '--scene', scene, '--count', str(count), '--instance-id', str(instance_id),
                   '--repeat', str(repeat), '--check', '--output', str(path)]
        subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu), timeout=40, check=True)
        report = json.loads(path.read_text())
        records = report['records']
        if report['failures'] or report['fatal_error'] or not all(record['passed'] for record in records):
            raise RuntimeError(f'incomplete or failed report: {path}')
        rows.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         instance=mode, scene=scene, triangles=len(report['triangles']),
                         instance_id=instance_id, gpu=gpu, launches=len(records),
                         requests=sum(record['requests'] for record in records),
                         ias_misses=sum(record['status'] == 1 for record in records),
                         child_misses=sum(record.get('child_result', {}).get('kind') == 'miss' for record in records)))
    summary = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                   contract='single static instance; device-side independent child query; NOT parent resume; no timing',
                   validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=rows,
                   totals={key: sum(row[key] for row in rows)
                           for key in ('launches', 'requests', 'ias_misses', 'child_misses')})
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['totals']))


if __name__ == '__main__':
    main()
