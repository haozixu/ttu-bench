"""Build a CUDA-entry TTU probe from validated driver operands, without its
megakernel, traversal software, callbacks, or local/shared-memory accesses.
"""
import ctypes
import os
import struct
from pathlib import Path

from cuda_driver import check, cuda
from cubin_builder import CUDA_ENTRY, EXIT, MOV_CONTROL, compile_template, mov_immediate, dump_registers, patch_kernel
from ttu_isa import decode, input_registers, unpack

HERE = Path(__file__).resolve().parent


def build(registers, output_address, group):
    template = compile_template(HERE / 'minimal_template.cu')
    registers = list(registers)
    for register in os.environ.get("MINIMAL_ZERO", "").split(","):
        if register:
            registers[int(register)] = 0
    instructions = [CUDA_ENTRY]
    for register in input_registers(group):
        value = registers[register]
        instructions.append(mov_immediate(register, value))
    if any(decode(instruction)['predicate'] != 7 for instruction in unpack(group)):
        instructions.append((0x7802 | 74 << 16, MOV_CONTROL))
        instructions.append((0x7804 | 74 << 24 | 0x1f << 32, 0x000fc80000000000))
    if os.environ.get('MINIMAL_NOTTU') != '1':
        instructions.extend(struct.iter_unpack('<QQ', group))
    output = dump_registers(output_address)
    if os.environ.get('MINIMAL_NOTTU') == '1':
        output[0] = mov_immediate(76, output_address & 0xffffffff)
    instructions.extend(output)
    instructions.append(EXIT)
    return patch_kernel(template, 'ttu_minimal', instructions)


def launch_image(image, output_address, *, kernel='ttu_minimal', threads=1, artifact=None):
    """Execute a patched container; caller owns the context and allocations."""
    if artifact:
        Path(artifact).write_bytes(image)
    module = ctypes.c_void_p()
    check(cuda.cuModuleLoadData(ctypes.byref(module), bytes(image)), 'minimal module')
    function = ctypes.c_void_p()
    check(cuda.cuModuleGetFunction(ctypes.byref(function), module, kernel.encode()), 'minimal function')
    argument = ctypes.c_uint64(output_address)
    parameters = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.byref(argument), ctypes.c_void_p))
    check(cuda.cuLaunchKernel(function, 1, 1, 1, threads, 1, 1, 0, None, parameters, None), 'minimal launch')
    check(cuda.cuCtxSynchronize(), 'minimal sync')
    result = ctypes.create_string_buffer(512)
    check(cuda.cuMemcpyDtoH_v2(result, output_address, 512), 'minimal results')
    check(cuda.cuModuleUnload(module), "minimal unload")
    return struct.unpack('<128I', result.raw)


def launch(registers, output_address, group, artifact=None):
    return launch_image(build(registers, output_address, group), output_address, artifact=artifact)
