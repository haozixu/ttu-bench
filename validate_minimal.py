#!/usr/bin/env python3
"""Functional regression matrix. Each case gets a fresh process/CUDA context."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', nargs='+', default=['7'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = [
        ['hit'], ['miss'], ['hit', '--oz', '-10'], ['hit', '--dz', '2'],
        ['hit', '--ox', '0.5'], ['hit', '--tmax', '6'],
        ['hit', '--full-group'], ['miss', '--full-group'],
    ]
    records = []
    for gpu in args.gpus:
        for case in cases:
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            for name in ('MINIMAL_ZERO', 'MINIMAL_NOTTU'):
                environment.pop(name, None)
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('run_minimal.py')),
                                     *case, '--repeat', '5', '--check'], env=environment,
                                    text=True, capture_output=True, timeout=20)
            if result.returncode:
                raise RuntimeError(f'GPU {gpu} case {case}: {result.stdout}\n{result.stderr}')
            record = json.loads(result.stdout)
            record.update(gpu=gpu, case=case)
            records.append(record)
            print(f'GPU {gpu}: {" ".join(case)} PASS', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
    print(f'{len(records)} cases, {len(records)*5} launches passed; {args.output}')


if __name__ == '__main__':
    main()
