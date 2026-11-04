"""Single-triangle SM86 mttu fixture knowledge; separate from general ISA."""
from project_paths import ROOT as HERE
import hashlib
import math
import struct
from pathlib import Path

from cuda_driver import sections
from ttu_isa import decode, fused_group, set_field, unpack

FIXTURES = HERE / 'fixtures'
AS_SHA256 = '715c99ef42c84709f9d0f91b6b458e253895981dec2ea4fa16a51cf56c7ab269'
GROUP_SHA256 = 'f8ff2e20f9711dcaff44c9af412c1e4f79d2f00ccfe56aceb4795e6584f50a02'
REDUCED_SHA256 = 'ced3108f52be2c8fd4fb811b988e5e71f81a4ccf70d8a6df0aaa9d2af66e6bbd'


def load_group(full=False, expanded=False):
    original = (FIXTURES / 'jit_16.elf').read_bytes()
    section = next(candidate for candidate in sections(original)
                   if candidate['name'] == '.text.traversal_mttu')
    group = original[section['off'] + 0xe70:section['off'] + 0xfb0]
    if hashlib.sha256(group).hexdigest() != GROUP_SHA256:
        raise ValueError('TTU group differs from the validated SM86 fixture')
    if full:
        return group
    if expanded:
        # Specialize the original P4=false path, retaining every return slot.
        # The original predicated group remains available as a diagnostic.
        instructions = unpack(group)
        body = [set_field(instruction, 'predicate', 7)
                for instruction in instructions[2:]
                if decode(instruction)['predicate'] != 4]
        return fused_group(body, opening=instructions[0],
                           macro_upper=int.from_bytes(instructions[1][8:], 'little'))
    instructions = unpack(group)
    body = []
    for instruction in instructions[2:]:
        decoded = decode(instruction)
        if decoded['name'] == 'TTUST':
            if decoded['address'] == 0x200:
                continue
            instruction = set_field(instruction, 'predicate', 7)
        elif decoded['name'] == 'TTULD':
            if decoded['address'] != 0x300:
                continue
            instruction = set_field(instruction, 'close', 1)
        body.append(instruction)
    group = fused_group(body, opening=instructions[0],
                        macro_upper=int.from_bytes(instructions[1][8:], 'little'))
    if hashlib.sha256(group).hexdigest() != REDUCED_SHA256:
        raise ValueError('reduced group differs from the validated SM86 fixture')
    return group


def f32(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def operands(base, ox, oy, oz, dz, tmax, dummy, *, dx=0, dy=0, tmin=0, ray_flags=0, root_offset=0x300):
    if ray_flags & ~0x31 or ray_flags & 0x30 == 0x30:
        raise ValueError('supported ray flags: NONE, DISABLE_ANYHIT, and one face-culling flag')
    values = [0] * 128
    config52 = 0xff030207 | (ray_flags & 0x30)
    config53 = 0x00400408 | ((ray_flags & 1) << 5)
    assignments = {
        2: f32(dx), 3: f32(dy),
        12: f32(oz), 13: f32(tmin), 14: f32(dx), 15: f32(dy),
        16: f32(dz), 17: f32(tmax),
        24: f32(ox), 25: f32(oy), 32: f32(dz), 33: f32(tmax),
        36: f32(oz), 37: f32(tmin),
        40: config52 + 0x200, 41: (config53 & 0x5543b) | 0x700000, 42: 0x00ff0000,
        50: (dummy >> 6) & 0xffffffff, 51: (dummy >> 38) | 0x02402800,
        52: config52, 53: config53,
        54: ((base + root_offset) >> 7) & 0xffffffff,
        55: ((base + root_offset) >> 39) | 0x02484000,
        66: f32(ox), 67: f32(oy), 72: 0x00ff0000,
    }
    for register, value in assignments.items():
        values[register] = value
    return values


def triangle_result(raw):
    """Empirical decoder for the current triangle GAS experiments.

    Only IDs exercised by the scene matrix are validated. The low-28-bit
    mask and high-nibble tags (8/a observed) are hypotheses, not a complete status ABI.
    R32's sign tracks front/back face; R73's sign tracks nonopaque/opaque
    requests in the tested subset. Other events must not be decoded as triangles.
    """
    def fp(register):
        return struct.unpack('<f', struct.pack('<I', raw[register]))[0]
    if raw[72:74] == (0xffffffff, 0xffffffff):
        return dict(hit=False, primitive=None, t=None, barycentrics=None, known_tag=True,
                    front_face=None, t_sign_bit=None, kind='miss', status_class=None)
    valid_attributes = all(math.isfinite(value) and 0 <= value <= 1
                           for value in (abs(fp(32)), fp(33)))
    if raw[72] >> 28 not in (0x8, 0xa) or not valid_attributes or not math.isfinite(fp(73)):
        return dict(hit=None, primitive=None, t=None, barycentrics=None, known_tag=False,
                    front_face=None, t_sign_bit=None, kind='other', status_class=raw[33] >> 27)
    return dict(hit=True, primitive=raw[72] & 0x0fffffff, t=abs(fp(73)),
                barycentrics=[abs(fp(32)), fp(33)], known_tag=raw[72] >> 28 in (0x8, 0xa),
                front_face=not bool(raw[32] >> 31), t_sign_bit=raw[73] >> 31,
                kind='triangle', status_class=None)
