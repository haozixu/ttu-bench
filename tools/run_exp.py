#!/usr/bin/env python3
"""EoSS instruction behavior reproduction; each experiment uses a child process.

Reconstructed runner, not a restoration of the deleted source. No BVH is
supplied: successful execution is a protocol observation, not a traversal oracle.
"""
from project_paths import ROOT as HERE, hash_sources, bench_command
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from build_and_patch import DEFAULT_EXPERIMENTS, EXPERIMENTS, build_experiment, sequence
from cuda_driver import check, cuda
from minimal import launch_image



def child(name, threads, wait_mask):
    check(cuda.cuInit(0), 'init')
    device = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(device), 0), 'device')
    context = ctypes.c_void_p()
    check(cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device), 'context')
    output = ctypes.c_uint64()
    check(cuda.cuMemAlloc_v2(ctypes.byref(output), 512), 'output allocation')
    image = build_experiment(name, output.value, wait_mask=wait_mask)
    result = dict(experiment=name, threads=threads, wait_mask=wait_mask, ttu_sha256=hashlib.sha256(sequence(name)).hexdigest(),
                  cubin_sha256=hashlib.sha256(image).hexdigest(), output_address=hex(output.value))
    try:
        values = launch_image(image, output.value, threads=threads)
        result.update(status='ok', registers=[f'{value:08x}' for value in values[:76]])
    except RuntimeError as error:
        result.update(status='cuda_error', error=str(error), post_error_sync=int(cuda.cuCtxSynchronize()))
    print(json.dumps(result))
    return 0 if result['status'] == 'ok' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=(*EXPERIMENTS, 'all'), nargs='?', default='all')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--threads', type=int, choices=(1, 32), default=1)
    parser.add_argument('--timeout', type=float, default=20)
    parser.add_argument('--wait-mask', type=lambda value: int(value, 0), default=0,
                        help='scoreboard wait before dump; default 0 for standalone instruction behavior')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeat < 1 or args.timeout <= 0 or not 0 <= args.wait_mask < 64:
        parser.error('repeat/timeout must be positive; wait-mask must fit in six bits')
    if args.child:
        if args.experiment == 'all':
            parser.error('child requires a single experiment')
        return child(args.experiment, args.threads, args.wait_mask)
    selected = DEFAULT_EXPERIMENTS if args.experiment == 'all' else (args.experiment,)
    records = []
    for name in selected:
        for trial in range(args.repeat):
            command = [*bench_command('run_exp'), name, '--child', '--threads', str(args.threads),
                       '--wait-mask', str(args.wait_mask)]
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, cwd=HERE)
                try:
                    record = json.loads(completed.stdout)
                except json.JSONDecodeError:
                    record = dict(experiment=name, status='harness_error', stdout=completed.stdout)
                record.update(returncode=completed.returncode, stderr=completed.stderr, trial=trial)
            except subprocess.TimeoutExpired as error:
                record = dict(experiment=name, status='timeout', trial=trial,
                              stdout=(error.stdout or b'').decode(errors='replace'),
                              stderr=(error.stderr or b'').decode(errors='replace'))
            expected_status = 'cuda_error' if name == 'exp2_open2' else 'ok'
            record['expected_status'] = expected_status
            record['matches_expected'] = record['status'] == expected_status
            if name == 'exp2_open2':
                record['matches_expected'] = (record['matches_expected'] and
                                              'CUDA error 715 ' in record.get('error', '') and
                                              record.get('post_error_sync') == 715)
            records.append(record)
            print(json.dumps({key: value for key, value in record.items() if key != 'registers'}), flush=True)
            if record['status'] == 'timeout':
                break
        if records[-1]['status'] == 'timeout':
            break
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version', '--format=csv'],
                         capture_output=True, text=True, check=True).stdout
    hashes = hash_sources(('run_exp.py', 'build_and_patch.py', 'ttu_isa.py', 'eoss_helpers.py', 'cuda_driver.py',
                           'cubin_builder.py', 'minimal.py', 'minimal_template.cu'))
    eoss_data = HERE.parent / 'research/community/eoss-framework/data/encodings/ttu_sm120.json'
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(), harness='reconstructed-eoss-v2',
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu_inventory=gpu,
                  threads=args.threads, wait_mask=args.wait_mask, source_sha256=hashes,
                  eoss_ttu_data_sha256=hashlib.sha256(eoss_data.read_bytes()).hexdigest(), records=records)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    return 0 if all(record['matches_expected'] for record in records) else 1


if __name__ == '__main__':
    sys.exit(main())
