"""SM86 TTU fields and byte builders, grounded in Rosetta and jit_16.

Builders return 16-byte little-endian instructions. The upper word contains
operands as well as scheduling. Raw field edits preserve every other bit.
These layouts do not assert that an arbitrary protocol or operand is valid.
"""
import struct

OPCODES = {
    'TTUOPEN': 0x3d0, 'TTUMACROFUSE': 0x9d4, 'TTUST': 0x3d1,
    'TTUGO': 0x3d3, 'TTULD': 0x3d2, 'TTUCCTL': 0x3d5,
}
FIELDS = {
    'predicate': (12, 4), 'address': (40, 16), 'count': (40, 5),
    'rb': (32, 8), 'rd': (16, 8), 'register2': (64, 8),
    'close': (74, 1), 'ivall': (80, 1),
}


def inst(word, upper):
    return struct.pack('<QQ', word, upper)


def unpack(sequence):
    if len(sequence) % 16:
        raise ValueError('SASS sequence length must be a multiple of 16')
    return [sequence[offset:offset + 16] for offset in range(0, len(sequence), 16)]


def field(instruction, name):
    if len(instruction) != 16:
        raise ValueError('one SASS instruction must have 16 bytes')
    shift, width = FIELDS[name]
    return (int.from_bytes(instruction, 'little') >> shift) & ((1 << width) - 1)


def set_field(instruction, name, value):
    field(instruction, name)
    shift, width = FIELDS[name]
    if not isinstance(value, int) or not 0 <= value < 1 << width:
        raise ValueError(f'{name} must fit in {width} bits')
    encoded = int.from_bytes(instruction, 'little')
    mask = ((1 << width) - 1) << shift
    return ((encoded & ~mask) | value << shift).to_bytes(16, 'little')


def decode(instruction):
    field(instruction, 'predicate')
    encoded = int.from_bytes(instruction, 'little')
    opcode = (encoded & 0xfff) | ((encoded >> 91) & 1) << 12
    name = next((name for name, value in OPCODES.items() if value == opcode), None)
    if name is None:
        raise ValueError(f'not a supported TTU instruction: {opcode:#x}')
    names = ['predicate']
    if name == 'TTUMACROFUSE':
        names += ['count']
    elif name == 'TTUST':
        names += ['address', 'rb', 'register2']
    elif name == 'TTULD':
        names += ['address', 'rd', 'register2', 'close']
    elif name == 'TTUCCTL':
        names += ['ivall']
    return dict(name=name, **{key: field(instruction, key) for key in names})


def _build(name, upper, predicate, **fields):
    instruction = inst(OPCODES[name] | 7 << 12, upper)
    for key, value in dict(predicate=predicate, **fields).items():
        instruction = set_field(instruction, key, value)
    return instruction


def ttuopen(*, predicate=7, upper=0x000fec0000000100):
    return _build('TTUOPEN', upper, predicate)


def ttumf(count, *, predicate=7, upper=0x000fe80000000000):
    return _build('TTUMACROFUSE', upper, predicate, count=count)


def ttust(address, rb, rb2=255, *, predicate=7, upper=0x0001e20000000000):
    """Write two explicit adjacent register pairs: rb:rb+1, rb2:rb2+1."""
    return _build('TTUST', upper, predicate, address=address, rb=rb, register2=rb2)


def ttugo(*, predicate=7, upper=0x000fe20000000000):
    return _build('TTUGO', upper, predicate)


def ttuld(rd, close=False, *, address=0x300, rd2=255, predicate=7,
          upper=0x000e6200000e0000):
    """SM86 mttu layout; CLOSE is absolute bit 74, not the EoSS canned bit 72."""
    return _build('TTULD', upper, predicate, address=address, rd=rd,
                  register2=rd2, close=int(close))


def ttuclose(*, predicate=7, upper=0x000e6400000e0000):
    """Rosetta TTUCLOSE alias: TTULD.CLOSE[0x300] RZ,RZ; not runtime-tested."""
    return ttuld(255, True, address=0x300, rd2=255, predicate=predicate, upper=upper)


def ttucctl(ivall=False, *, predicate=7, upper=0x000fe20000000000):
    return _build('TTUCCTL', upper, predicate, ivall=int(ivall))


def fused_group(body, *, opening=None, macro_upper=0x000fe80000000000):
    """Compute the lexical fuse count; runtime protocol validity is separate."""
    body = list(body)
    if not 1 <= len(body) <= 31:
        raise ValueError('SM86 fuse count must be between 1 and 31')
    for instruction in body:
        if decode(instruction)['name'] in ('TTUOPEN', 'TTUMACROFUSE'):
            raise ValueError('OPEN/MACROFUSE cannot be inside this fused body')
    return (ttuopen() if opening is None else opening) + ttumf(len(body), upper=macro_upper) + b''.join(body)


def input_registers(sequence):
    registers = set()
    for instruction in unpack(sequence):
        decoded = decode(instruction)
        if decoded['name'] == 'TTUST':
            for start in (decoded['rb'], decoded['register2']):
                if start != 255:
                    registers.update((start, start + 1))
    return sorted(registers)
