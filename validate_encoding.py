#!/usr/bin/env python3
"""CPU checks against driver bytes, EoSS golden data, and cleanup hashes."""
import hashlib
import json
from pathlib import Path

from eoss_helpers import canonical_sm120, control, patch_slots
from sm86_fixture import load_group
from ttu_isa import decode, field, inst, set_field, ttucctl, ttugo, ttuld, ttumf, ttuopen, ttust, unpack

HERE = Path(__file__).resolve().parent


def main():
    builders = {'TTUOPEN': ttuopen, 'TTUMACROFUSE': ttumf, 'TTUST': ttust,
                'TTUGO': ttugo, 'TTULD': ttuld, 'TTUCCTL': ttucctl}
    for instruction in unpack(load_group(full=True)):
        decoded = decode(instruction)
        name = decoded.pop('name')
        if name == 'TTUST':
            decoded['rb2'] = decoded.pop('register2')
        elif name == 'TTULD':
            decoded['rd2'] = decoded.pop('register2')
        decoded['upper'] = int.from_bytes(instruction[8:], 'little')
        if builders[name](**decoded) != instruction:
            raise RuntimeError(f'round trip failed: {name}')
    reduced = load_group()
    if len(reduced) != 13 * 16 or decode(unpack(reduced)[1])['count'] != 11:
        raise RuntimeError('reduced-group size or fuse count changed')
    expanded = unpack(load_group(expanded=True))
    if len(expanded) != 19 or decode(expanded[1])['count'] != 17:
        raise RuntimeError('expanded-group size or fuse count changed')
    if any(decode(instruction)['predicate'] != 7 for instruction in expanded):
        raise RuntimeError('expanded group must have a statically selected path')
    original_loads = [x for x in unpack(load_group(full=True)) if decode(x)['name'] == 'TTULD']
    if [x for x in expanded if decode(x)['name'] == 'TTULD'] != original_loads:
        raise RuntimeError('expanded group must preserve every original return slot')
    if ttuld(72, True, rd2=32) != inst(0x00030000004873d2, 0x000e6200000e0420):
        raise RuntimeError('SM86 LD operands or CLOSE changed')
    canonical = unpack(canonical_sm120())
    if len(canonical) != 10 or field(canonical[-1], 'close') != 0:
        raise RuntimeError('EoSS golden sequence changed; review CLOSE distinction')
    if canonical[-1] != inst(0x73d2, 0x000fe20000000100):
        raise RuntimeError('EoSS raw close-labelled encoding changed')
    changed = control.write_control(ttuld(72, True, rd2=32), wt_bar=5, wr_bar=0)
    if decode(changed) != decode(ttuld(72, True, rd2=32)):
        raise RuntimeError('EoSS scheduler edit changed operand fields')
    if patch_slots(bytes(32), 16, changed)[16:] != changed:
        raise RuntimeError('EoSS REPLACE_FULL did not preserve all bits')
    for name, value in [('count', 32), ('rb', 256), ('predicate', 16), ('address', -1)]:
        try:
            set_field(ttuopen(), name, value)
        except ValueError:
            pass
        else:
            raise RuntimeError('out-of-range field was accepted')
    manifest = json.loads((HERE / 'cleanup-2026-11-02.json').read_text())
    restored = []
    for name in ('build_ttu3.py', 'probe3.cu', 'run_ttu4.py'):
        expected = next(entry for entry in manifest['deleted_text'] if entry['path'] == name)
        data = (HERE / 'reference/history/source' / name).read_bytes()
        if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
            raise RuntimeError(f'recovered source differs from deletion manifest: {name}')
        restored.append(name)
    print(json.dumps(dict(status='pass', driver_roundtrips=20, reduced_instructions=13, expanded_instructions=19,
                          recovered_exact=restored, eoss_close_bit=72, sm86_close_bit=74)))


if __name__ == '__main__':
    main()
