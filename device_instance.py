"""Patch a compiler-managed CUDA function with a remapped expanded TTU group.

Compiler code handles events; this functional harness is not a timing kernel.
"""
import ctypes
import hashlib
from pathlib import Path
import struct
import subprocess
import tempfile

from cuda_driver import check, cuda, sections
from sm86_fixture import load_group
from ttu_isa import decode, set_field, unpack


HERE = Path(__file__).resolve().parent
PAIRS = (42, 52, 12, 66, 16, 14, 72, 40, 36, 24, 32, 2, 34, 50, 56, 54, 60, 58)


class Params(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ('ias_base', 'ias_size', 'dummy', 'child_base', 'child_size')] + [
        (name, ctypes.c_uint32) for name in ('root_low', 'root_high', 'dummy_low', 'dummy_high')] + [
        ('ray', ctypes.c_float * 8)]


class Result(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in ('status', 'requests', 'instance_id', 'parent_slot_nonzero')] + [
        ('initial', ctypes.c_uint32 * 76), ('child', ctypes.c_uint32 * 76),
        ('local_ray', ctypes.c_float * 8), ('event', ctypes.c_uint64), ('child_root', ctypes.c_uint64)]


def issue_source():
    lines = ['extern "C" __device__ __noinline__ void ttu_issue(const volatile uint64_t* input, volatile uint64_t* output) {']
    lines += [f'    uint64_t value{index} = input[{index}];' for index in range(18)]
    lines.append('    asm volatile(')
    for index in range(18):
        immediate = 0x3ff0000000000000 + (index << 32)
        lines.append(f'        "add.f64 %{index}, %{index}, 0d{immediate:016X};\\n"')
    for index in range(3):
        immediate = 0x4000000000000000 + (index << 32)
        lines.append(f'        "add.f64 %0, %0, 0d{immediate:016X};\\n"')
    lines.append('        : ' + ', '.join(f'"+l"(value{index})' for index in range(18)) + ');')
    lines += [f'    output[{index}] = value{index};' for index in range(18)]
    return '\n'.join(lines + ['}'])


def build():
    source = (HERE / 'instance_template.cu').read_text() + '\n' + issue_source() + '\n'
    with tempfile.TemporaryDirectory(prefix='ttu-instance-') as directory:
        path = Path(directory) / 'instance.cu'
        cubin = path.with_suffix('.cubin')
        path.write_text(source)
        command = ['nvcc', '-arch=sm_86', '-cubin', '-Xptxas', '-O0', str(path), '-o', str(cubin)]
        compiled = subprocess.run(command, capture_output=True, text=True, check=True)
        image = bytearray(cubin.read_bytes())
    markers = []
    for section in sections(image):
        if not section['name'].startswith('.text.'):
            continue
        for offset in range(section['off'], section['off'] + section['size'], 16):
            word, upper = struct.unpack_from('<QQ', image, offset)
            if word & 0xfff == 0x429:
                markers.append((offset, word, upper))
    if len(markers) != 21 or any(markers[index][0] != markers[0][0] + 16 * index for index in range(21)):
        raise ValueError('expected 21 contiguous DADD marker instructions')
    register_map = {}
    for index, original in enumerate(PAIRS):
        _, word, _ = markers[index]
        if word >> 32 != 0x3ff00000 + index or (word >> 16 & 255) != (word >> 24 & 255):
            raise ValueError('DADD marker operand or immediate changed')
        register_map[original] = word >> 16 & 255
    registers = [register + delta for register in register_map.values() for delta in (0, 1)]
    if len(set(registers)) != 36 or any(register >= 255 for register in registers):
        raise ValueError('compiler assigned overlapping or invalid register pairs')
    for index in range(3):
        if markers[18 + index][1] >> 32 != 0x40000000 + index:
            raise ValueError('trailing DADD marker changed')
    mapped = []
    for instruction in unpack(load_group(expanded=True)):
        decoded = decode(instruction)
        names = ('rb', 'register2') if decoded['name'] == 'TTUST' else (
            ('rd', 'register2') if decoded['name'] == 'TTULD' else ())
        for name in names:
            if decoded[name] != 255:
                instruction = set_field(instruction, name, register_map[decoded[name]])
        mapped.append(instruction)
    wait = struct.pack('<QQ', 0x7918, 0x000fe20000000000 | 0x3f << 52)
    replacement = wait + b''.join(mapped) + wait
    start = markers[0][0]
    image[start:start + len(replacement)] = replacement
    return bytes(image), dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                             cubin_sha256=hashlib.sha256(image).hexdigest(), register_map=register_map,
                             patch_file_offset=start, group_sha256=hashlib.sha256(b''.join(mapped)).hexdigest(),
                             compiler_output=compiled.stderr,
                             compiler_version=subprocess.run(['nvcc', '--version'], capture_output=True,
                                                             text=True, check=True).stdout)


class DeviceProbe:
    def __init__(self):
        if ctypes.sizeof(Params) != 88 or ctypes.sizeof(Result) != 672:
            raise ValueError('unexpected host/device structure layout')
        self.image, self.build_info = build()
        self.module = ctypes.c_void_p()
        check(cuda.cuModuleLoadData(ctypes.byref(self.module), self.image), 'instance module')
        self.function = ctypes.c_void_p()
        check(cuda.cuModuleGetFunction(ctypes.byref(self.function), self.module, b'instance_probe'), 'instance kernel')
        self.input = ctypes.c_uint64()
        self.output = ctypes.c_uint64()
        check(cuda.cuMemAlloc_v2(ctypes.byref(self.input), ctypes.sizeof(Params)), 'params allocate')
        check(cuda.cuMemAlloc_v2(ctypes.byref(self.output), ctypes.sizeof(Result)), 'result allocate')

    def launch(self, params):
        check(cuda.cuMemcpyHtoD_v2(self.input.value, ctypes.byref(params), ctypes.sizeof(params)), 'params upload')
        zero = bytes(ctypes.sizeof(Result))
        check(cuda.cuMemcpyHtoD_v2(self.output.value, zero, len(zero)), 'result zero')
        parameters = (ctypes.c_void_p * 2)(ctypes.addressof(self.input), ctypes.addressof(self.output))
        check(cuda.cuLaunchKernel(self.function, 1, 1, 1, 1, 1, 1, 0, None, parameters, None), 'instance launch')
        check(cuda.cuCtxSynchronize(), 'instance sync')
        result = Result()
        check(cuda.cuMemcpyDtoH_v2(ctypes.byref(result), self.output.value, ctypes.sizeof(result)), 'result readback')
        return result

    def close(self):
        check(cuda.cuMemFree_v2(self.input), 'params free')
        check(cuda.cuMemFree_v2(self.output), 'result free')
        check(cuda.cuModuleUnload(self.module), 'instance unload')
