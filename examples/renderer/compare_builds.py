#!/usr/bin/env python3
"""Compare guarded monolithic and separately compiled native TTU containers."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

from device_instance import PAIRS, patch_compiled
from project_paths import ROOT, hash_sources
from sm86_fixture import load_group
from ttu_isa import decode, unpack


def check_patch(image, symbol, metadata, separate):
    metadata = metadata | {name: {int(key): value for key, value in metadata[name].items()}
                           for name in ('register_map', 'input_register_map')}
    patched, actual = patch_compiled(image, symbol, separate_maps=separate)
    if actual != metadata:
        raise ValueError('patch metadata differs')
    start = actual['patch_file_offset']
    length = 21*16
    if image[:start] != patched[:start] or image[start+length:] != patched[start+length:]:
        raise ValueError('patch changed bytes outside marker region')
    for original, instruction in zip(unpack(load_group(expanded=True)), unpack(patched[start+16:start+length-16])):
        before, after = decode(original), decode(instruction)
        fields = ('rb', 'register2') if before['name'] == 'TTUST' else (
            ('rd', 'register2') if before['name'] == 'TTULD' else ())
        mapping = actual['input_register_map'] if before['name'] == 'TTUST' else actual['register_map']
        expected = before.copy()
        expected.update({field: mapping[before[field]] for field in fields if before[field] != 255})
        if after != expected:
            raise ValueError('patched operands differ from map')
    mutations = [(0, 24, actual['input_register_map'][PAIRS[1]]),
                 (0, 16, actual['register_map'][PAIRS[1]]),
                 (18, 24, actual['register_map'][PAIRS[1]]),
                 (2, 24, 242), (17, 24, 254)]
    checks = []
    for index, shift, register in mutations:
        changed = bytearray(image)
        offset = start+index*16
        word = struct.unpack_from('<Q', changed, offset)[0]
        word = word & ~(255 << shift) | register << shift
        struct.pack_into('<Q', changed, offset, word)
        try:
            patch_compiled(changed, symbol, separate_maps=separate)
        except ValueError as error:
            checks.append(dict(marker=index, field_shift=shift, register=register, rejected=str(error)))
        else:
            raise ValueError('corrupted marker unexpectedly accepted')
    return dict(outside_bytes_preserved=True, operands_verified=True, corruptions_rejected=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='5')
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=192)
    parser.add_argument('--grid', type=int, default=4)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--artifacts', action='store_true', help='retain per-case reports and raw/build artifacts')
    parser.add_argument('--output', type=Path, default=ROOT/'results/local/renderer-build-comparison')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output directory must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    configurations = [(mode, level) for mode in ('monolithic','linked') for level in range(4)]
    if args.reverse:
        configurations.reverse()
    matrix = dict(timestamp=datetime.now(timezone.utc).isoformat(), gpu=args.gpu, cases=[], passed=False,
                  contract='Local exploratory optimization comparison. Rejected marker layouts never launch. Separate processes, unlocked clocks; small timing differences do not prove improvement.')
    builds = []
    try:
        for mode, level in configurations:
            destination = args.output/f'{mode}-o{level}'
            command = [sys.executable, str(ROOT/'bench.py'), 'render_demo', '--native-build',mode,
                       '--native-opt',str(level),'--width',str(args.width),'--height',str(args.height),
                       '--grid',str(args.grid),'--samples',str(args.samples),'--iterations',str(args.iterations),
                       '--output',str(destination),'--artifacts']
            result = subprocess.run(command, env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu),
                                    capture_output=True, text=True, timeout=120)
            report_path = destination/'report.json'
            report = json.loads(report_path.read_bytes()) if report_path.exists() else {}
            if report.get('build_directory'):
                builds.append(Path(report['build_directory']))
            case = dict(mode=mode,level=level,command=command,returncode=result.returncode,
                        stdout=result.stdout,stderr=result.stderr,report=str(report_path),
                        report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest() if report else None)
            if result.returncode and report.get('fatal_error') == 'expected 21 contiguous DADD marker instructions':
                case.update(status='rejected-before-launch', passed=True)
            elif result.returncode == 0 and report.get('passed'):
                case.update(status='validated', passed=True, timings=report['timings'],
                            resources=report['build']['native_linked_resources'])
                build = Path(report['build']['directory'])
                original = (build/'native-original.cubin').read_bytes()
                case['patch_checks'] = check_patch(original, report['build']['patch']['patch_symbol'],
                                                   report['build']['patch'], mode == 'linked')
            else:
                case.update(status='failed',passed=False,error=report.get('fatal_error'))
            matrix['cases'].append(case)
            if not args.artifacts:
                case.pop('report')
                case.pop('report_sha256')
                case.pop('stdout')
                case['source_sha256'] = report.get('source_sha256', {})
            print(json.dumps({key:case[key] for key in ('mode','level','status','passed')}),flush=True)
            if not case['passed']:
                raise ValueError('build or validation failed')
        matrix['passed'] = any(case['status'] == 'validated' for case in matrix['cases'])
    except Exception as error:
        matrix['fatal_error'] = str(error)
    finally:
        if not args.artifacts:
            for build in builds:
                shutil.rmtree(build)
            for destination in args.output.iterdir():
                if destination.is_dir():
                    shutil.rmtree(destination)
        matrix['source_sha256'] = hash_sources(['examples/renderer/compare_builds.py'])
        (args.output/'matrix.json').write_text(json.dumps(matrix,indent=2)+'\n')
    if not matrix['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
