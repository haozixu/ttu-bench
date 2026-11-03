"""Reusable SM86 CUDA ELF container patching and ordinary SASS helpers."""
import struct
import subprocess
from pathlib import Path

from cuda_driver import sections

NOP = (0x7918, 0x000fc00000000000)
MOV_CONTROL = 0x000fe40000000f00
CUDA_ENTRY = (0x0000460000047ab9, 0x000fc80000000a00)
EXIT = (0x794d, 0x000fea0003800000)


def compile_template(source):
    source = Path(source)
    template = source.with_suffix('.cubin')
    if not template.exists() or template.stat().st_mtime < source.stat().st_mtime:
        subprocess.run(['nvcc', '-arch=sm_86', '-cubin', str(source), '-o', str(template)], check=True)
    return template.read_bytes()


def mov_immediate(register, value, *, wait_mask=0):
    if not 0 <= register <= 254 or not 0 <= value <= 0xffffffff or not 0 <= wait_mask < 64:
        raise ValueError('invalid MOV register, immediate or wait mask')
    return (0x7802 | register << 16 | value << 32, MOV_CONTROL | wait_mask << 52)


def dump_registers(output_address, *, wait_mask=0x3f):
    """Write R0..R75 using R76:R77; caller selects the required scoreboards."""
    instructions = [mov_immediate(76, output_address & 0xffffffff, wait_mask=wait_mask),
                    mov_immediate(77, output_address >> 32)]
    for register in range(0, 76, 4):
        instructions.append((0x7986 | 76 << 24 | register << 32 | (register * 4) << 40,
                             0x004fe8000c101d04))
    return instructions


def patch_kernel(template, kernel, instructions, *, registers=80, replace=None):
    """Replace a single-entry template and its register/EXIT metadata.

replace(image, file_offset, instruction_bytes) can supply an EoSS backend.
Templates with multiple entry kernels are deliberately rejected.
"""
    image = bytearray(template)
    instructions = list(instructions)
    table = sections(image)
    text_sections = [section for section in table if section['name'].startswith('.text.')]
    if len(text_sections) != 1 or text_sections[0]['name'] != '.text.' + kernel:
        raise ValueError('expected one CUDA entry kernel with the requested name')
    section = text_sections[0]
    if not instructions or instructions[-1] != EXIT:
        raise ValueError('last instruction must be the sole EXIT')
    if sum(instruction == EXIT for instruction in instructions) != 1:
        raise ValueError('only one EXIT is supported')
    if len(instructions) * 16 > section['size'] or not 1 <= registers <= 255:
        raise ValueError('instruction sequence or register count exceeds template limits')
    info = struct.unpack_from('<I', image, section['shdr'] + 44)[0]
    struct.pack_into('<I', image, section['shdr'] + 44, (info & 0xffffff) | registers << 24)
    for index in range(section['size'] // 16):
        instruction = instructions[index] if index < len(instructions) else NOP
        offset = section['off'] + index * 16
        encoded = struct.pack('<QQ', *instruction)
        if replace is None:
            image[offset:offset + 16] = encoded
        else:
            replace(image, offset, encoded)
    exit_updates = register_updates = 0
    for section in table:
        if section['name'] == '.nv.info.' + kernel:
            for index in range(section['size'] - 7):
                offset = section['off'] + index
                if image[offset:offset + 4] == b'\x04\x1c\x04\x00':
                    struct.pack_into('<I', image, offset + 4, (len(instructions) - 1) * 16)
                    exit_updates += 1
        elif section['name'] == '.nv.info':
            for index in range(section['size'] - 11):
                offset = section['off'] + index
                if image[offset:offset + 4] == b'\x04\x2f\x08\x00':
                    struct.pack_into('<I', image, offset + 8, registers)
                    register_updates += 1
    if exit_updates != 1 or register_updates != 1:
        raise ValueError('expected exactly one EXIT-offset and one register-count metadata entry')
    return image
