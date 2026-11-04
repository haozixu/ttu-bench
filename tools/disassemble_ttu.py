#!/usr/bin/env python3
"""Expose ordinary SASS around TTU slots by replacing TTU with NOP for analysis.

Outputs are not executable replacements for the original code.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile

from cuda_driver import sections
from ttu_isa import decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--section', default='.text.traversal_mttu')
    parser.add_argument('--range', dest='ranges', action='append', default=[], metavar='START:END',
                        help='hex byte offsets; repeatable; defaults to entire section')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    data = args.source.read_bytes()
    section = next(section for section in sections(data) if section['name'] == args.section)
    original = data[section['off']:section['off'] + section['size']]
    ordinary = bytearray(original)
    removed = []
    for offset in range(0, len(ordinary), 16):
        instruction = original[offset:offset + 16]
        try:
            decoded = decode(instruction)
        except ValueError:
            continue
        removed.append(dict(offset=hex(offset), bytes=instruction.hex(), decoded=decoded))
        ordinary[offset:offset + 16] = struct.pack('<QQ', 0x7918, 0x000fe20000000000)
    ranges = [tuple(int(value, 16) for value in pair.split(':')) for pair in args.ranges]
    if not ranges:
        ranges = [(0, len(ordinary))]
    if any(len(pair) != 2 or not 0 <= pair[0] < pair[1] <= len(ordinary) for pair in ranges):
        parser.error('range must lie inside the section')
    with tempfile.TemporaryDirectory(prefix='ttu-disasm-') as directory:
        binary = Path(directory) / 'ordinary.bin'
        binary.write_bytes(ordinary)
        command = ['nvdisasm', '-b', 'SM86', '-ndf', '-hex', str(binary)]
        disassembly = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    selected, active = [], False
    for line in disassembly.splitlines():
        match = re.search(r'/\*([0-9a-f]+)\*/', line)
        if match:
            active = any(start <= int(match[1], 16) < end for start, end in ranges)
        if active:
            selected.append(line)
    args.output.write_text('TTU slots replaced with NOP for disassembly only; original offsets retained. Not runnable.\n'
                           + '\n'.join(selected) + '\n')
    report = dict(source=str(args.source), source_sha256=hashlib.sha256(data).hexdigest(),
                  section=args.section, section_bytes=len(original),
                  excerpt=str(args.output), excerpt_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                  ranges=[[hex(start), hex(end)] for start, end in ranges],
                  command=['nvdisasm', '-b', 'SM86', '-ndf', '-hex', 'ordinary.bin'],
                  nvdisasm_version=subprocess.run(['nvdisasm', '--version'], capture_output=True, text=True, check=True).stdout,
                  tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  ttu_removed=removed, use='Static evidence only; not runnable traversal code')
    args.manifest.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(ttu_slots_removed=len(removed), excerpt_instructions=len(selected) // 2)))


if __name__ == '__main__':
    main()
