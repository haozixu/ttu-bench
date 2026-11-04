#!/usr/bin/env python3
"""Run the four pinned Bunny resolutions, preserving failures and raw outputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from prepare_bunny import bunny_path
from project_paths import ROOT, hash_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',default='5')
    parser.add_argument('--width',type=int,default=256)
    parser.add_argument('--height',type=int,default=192)
    parser.add_argument('--samples',type=int,default=5)
    parser.add_argument('--iterations',type=int,default=20)
    parser.add_argument('--levels',nargs='+',choices=('res4','res3','res2','full'),
                        default=['res4','res3','res2','full'])
    parser.add_argument('--reverse',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--mirror-floor',action='store_true')
    parser.add_argument('--artifacts', action='store_true', help='retain per-case reports and raw/build artifacts')
    parser.add_argument('--output',type=Path,default=ROOT/'results/local/renderer-bunny-matrix')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output directory must be empty')
    for level in args.levels:
        bunny_path(level)
    args.output.mkdir(parents=True,exist_ok=True)
    levels = list(dict.fromkeys(args.levels))
    if args.reverse:
        levels.reverse()
    matrix = dict(timestamp=datetime.now(timezone.utc).isoformat(),gpu=args.gpu,cases=[],passed=False,
        contract='Four source LODs, same camera/rays/normalization; exact coincident faces explicitly deduplicated. Geometric barycentric validation, exact primitive/face. Each process continues only after validation. All failures retained; no automatic image masking.')
    for level in levels:
        destination = args.output/level
        command = [sys.executable,str(ROOT/'bench.py'),'render_demo','--bunny',level,
            '--mesh-deduplicate','--validation','geometric','--width',str(args.width),'--height',str(args.height),
            '--samples',str(args.samples),'--iterations',str(args.iterations),'--output',str(destination)]
        if args.check_only:
            command.append('--check-only')
        if args.mirror_floor:
            command.append('--mirror-floor')
        if args.artifacts:
            command.append('--artifacts')
        try:
            result = subprocess.run(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu),
                                    capture_output=True,text=True,timeout=180)
            path = destination/'report.json'
            report = json.loads(path.read_bytes()) if path.exists() else {}
            case = dict(level=level,command=command,returncode=result.returncode,stdout=result.stdout,
                stderr=result.stderr,report=str(path),
                report_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
                passed=result.returncode==0 and report.get('passed',False),
                timings=report.get('timings',{}),failures=report.get('failures',[]))
            if not args.artifacts:
                case.pop('report')
                case.pop('report_sha256')
                case.pop('stdout')
                case['fatal_error'] = report.get('fatal_error')
                case['source_sha256'] = report.get('source_sha256', {})
                shutil.rmtree(destination, ignore_errors=True)
        except subprocess.TimeoutExpired as error:
            case = dict(level=level,command=command,passed=False,error='timeout180s',
                stdout=(error.stdout or b'').decode(errors='replace'),
                stderr=(error.stderr or b'').decode(errors='replace'))
        matrix['cases'].append(case)
        matrix['passed'] = len(matrix['cases'])==len(levels) and all(case['passed'] for case in matrix['cases'])
        matrix['source_sha256'] = hash_sources(['examples/renderer/validate_meshes.py'])
        (args.output/'matrix.json').write_text(json.dumps(matrix,indent=2)+'\n')
        print(json.dumps(dict(level=level,passed=case['passed'])),flush=True)
    if not matrix['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
