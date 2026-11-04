#!/usr/bin/env python3
"""Bounded Phase19 matrix. Each case gets its own process/context and timeout."""
from project_paths import bench_command
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def configurations():
    rows=[]
    def add(**kw):
        row=dict(group='reduced',lanes=32,kind='none',count=0,placement='before-wait',variant='normal',gap=0,ray='hit') | kw
        if row not in rows: rows.append(row)
    for lanes in (1,32):
        for group in ('none','reduced','expanded'): add(lanes=lanes,group=group)
    for kind,counts in [('fadd',(64,)),('hmma',(1,8,32)),('global',(1,2,4)),('shared',(1,16,32))]:
        for count in counts:
            for group in ('none','reduced'):
                for placement in ('before-wait','after-wait'):
                    add(kind=kind,count=count,group=group,placement=placement)
    for lanes in (1,32):
        for variant in ('prefix-mf0','open-mf0'): add(lanes=lanes,variant=variant)
        for variant in ('split9','split10'):
            for gap in (0,16,64): add(lanes=lanes,variant=variant,gap=gap)
        for gap in (16,64): add(lanes=lanes,variant='open-gap',gap=gap)
    for variant in ('split9','split10'): add(group='expanded',variant=variant,gap=16)
    for kind,count in [('hmma',8),('global',2),('shared',16)]:
        for placement in ('before-wait','after-wait'):
            add(group='expanded',kind=kind,count=count,placement=placement)
    # Miss controls also exercise split groups and distinct scoreboard paths.
    for ray in ('miss','short'): add(variant='split10',gap=16,kind='hmma',count=8,ray=ray)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',nargs='+',type=int,default=[5,1])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    cases=[]
    for gpu in args.gpus:
        for index,row in enumerate(configurations()):
            path=args.output.parent/f'{args.output.stem}-gpu{gpu}-{index:03}.json'
            command=bench_command('run_mixed_timing')
            for key,value in row.items(): command.extend(['--'+key.replace('_','-'),str(value)])
            command.extend(['--check','--output',str(path)])
            try:
                process=subprocess.run(command,env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu)},
                    capture_output=True,text=True,timeout=25)
                report=json.loads(path.read_bytes()) if path.exists() else {}
                passed=process.returncode==0 and not report.get('failures') and not report.get('fatal_error')
                case=dict(gpu=gpu,config=row,file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
                    returncode=process.returncode,stdout=process.stdout,stderr=process.stderr,passed=passed,
                    records=len(report.get('records',[])),summary=report.get('summary_lane0_steady'))
            except subprocess.TimeoutExpired:
                case=dict(gpu=gpu,config=row,file=str(path),passed=False,error='timeout25s')
            cases.append(case)
            args.output.write_text(json.dumps(dict(cases=cases,passed=all(c['passed'] for c in cases)),indent=2)+'\n')
            if not case['passed']:
                print(json.dumps(case),flush=True)
                raise SystemExit(1)
            if index%10==0: print(f'gpu{gpu}: {index+1}/{len(configurations())} passed',flush=True)
    totals=dict(cases=len(cases),launches=len(cases)*12,lane_checks=sum(c['records'] for c in cases),
        warp_ttu_requests=sum(48 for c in cases if c['config']['group']!='none'),
        lane_ray_requests=sum(48*c['config']['lanes'] for c in cases if c['config']['group']!='none'))
    args.output.write_text(json.dumps(dict(cases=cases,passed=True,totals=totals),indent=2)+'\n')
    print(json.dumps(totals))


if __name__=='__main__': main()
