#!/usr/bin/env python3
"""Check MF-removal controls: fresh process, expected 715, no timeout masking."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',type=int,nargs='+',default=[5,1])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    cases=[]
    for gpu in args.gpus:
        for lanes in (1,32):
            for group in ('reduced','expanded'):
                for mode in ('nop','zero'):
                    for repeat in range(3):
                        path=args.output.parent/f'{args.output.stem}-gpu{gpu}-l{lanes}-{group}-{mode}-r{repeat}.json'
                        command=['python3','run_macrofuse_control.py','--mode',mode,'--group',group,'--lanes',str(lanes),
                                 '--launches','3','--check','--output',str(path)]
                        try:
                            p=subprocess.run(command,env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu)},capture_output=True,text=True,timeout=25)
                            r=json.loads(path.read_bytes()) if path.exists() else {}
                            passed=(p.returncode==1 and 'CUDA error 715' in (r.get('fatal_error') or '') and not r.get('records'))
                            case=dict(gpu=gpu,lanes=lanes,group=group,mode=mode,repeat=repeat,file=str(path),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
                                returncode=p.returncode,fatal_error=r.get('fatal_error'),passed=passed,stderr=p.stderr)
                        except subprocess.TimeoutExpired:
                            case=dict(gpu=gpu,lanes=lanes,group=group,mode=mode,repeat=repeat,passed=False,error='timeout25s')
                        cases.append(case)
                        args.output.write_text(json.dumps(dict(cases=cases,passed=all(c['passed'] for c in cases)),indent=2)+'\n')
                        if not passed:
                            print(json.dumps(case),flush=True)
                            raise SystemExit(1)
        print(f'GPU{gpu}: expected 715 controls passed',flush=True)
    print(json.dumps(dict(cases=len(cases),expected_faults=len(cases))))


if __name__=='__main__': main()
