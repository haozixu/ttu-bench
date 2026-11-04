#!/usr/bin/env python3
"""Run renderer checks in fresh processes with bounded GPU work."""
import argparse
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

from project_paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='5')
    parser.add_argument('--native-build', choices=('monolithic', 'linked'), default='monolithic')
    parser.add_argument('--native-opt', type=int, choices=(0, 1, 2, 3))
    parser.add_argument('--mirror-floor',action='store_true')
    parser.add_argument('--artifacts', action='store_true', help='retain per-case reports and raw/build artifacts')
    parser.add_argument('--output', type=Path, default=ROOT/'results/local/renderer-validation')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output directory must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [(1,1,1,32),(31,1,2,32),(32,1,2,32),(33,1,2,32),
             (37,29,4,128),(64,48,8,128),(19,17,16,64),(17,19,3,256)]
    results = []
    for index, (width,height,grid,block) in enumerate(cases):
        destination = args.output/f'case-{index:02}'
        command = [sys.executable, str(ROOT/'bench.py'), 'render_demo', '--width',str(width),
                   '--height',str(height),'--grid',str(grid),'--block',str(block),'--check-only',
                   '--native-build',args.native_build,'--output',str(destination)]
        if args.native_opt is not None:
            command += ['--native-opt',str(args.native_opt)]
        if args.mirror_floor:
            command.append('--mirror-floor')
        if args.artifacts:
            command.append('--artifacts')
        try:
            result = subprocess.run(command, env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu),
                                    capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired as error:
            results.append(dict(command=command, passed=False, error='timeout120s',
                                stdout=(error.stdout or b'').decode(errors='replace'),
                                stderr=(error.stderr or b'').decode(errors='replace')))
            (args.output/'matrix.json').write_text(json.dumps(dict(cases=results,passed=False),indent=2)+'\n')
            raise SystemExit(1)
        report = json.loads((destination/'report.json').read_text()) if (destination/'report.json').exists() else {}
        case = dict(command=command,returncode=result.returncode,passed=result.returncode==0 and report.get('passed',False),
                    stdout=result.stdout,stderr=result.stderr,report=str(destination/'report.json'),
                    report_sha256=hashlib.sha256((destination/'report.json').read_bytes()).hexdigest()
                    if (destination/'report.json').exists() else None)
        results.append(case)
        if not args.artifacts:
            case.pop('report')
            case.pop('report_sha256')
            case.pop('stdout')
            case['failures'] = report.get('failures', [])
            case['fatal_error'] = report.get('fatal_error')
            case['source_sha256'] = report.get('source_sha256', {})
            shutil.rmtree(destination, ignore_errors=True)
        (args.output/'matrix.json').write_text(json.dumps(dict(cases=results,passed=all(row['passed'] for row in results)),indent=2)+'\n')
        print(json.dumps(dict(case=index,passed=case['passed'])),flush=True)
        if not case['passed']:
            print(result.stdout,result.stderr)
            raise SystemExit(1)


if __name__ == '__main__':
    main()
