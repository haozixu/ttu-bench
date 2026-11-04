"""Bounded, unrolled SM86 timing transport: one warp, per-lane snapshots.

TTU groups are byte-identical to the validated fixture groups. Clocks and ALU
work remain outside MACROFUSE. Each sample restores all TTUST inputs and waits
before the next sample. This measures an instrumented request, not pure RT time.
"""
from project_paths import KERNELS
import ctypes
import hashlib
from pathlib import Path
import struct

from cubin_builder import CUDA_ENTRY, EXIT, compile_template, mov_immediate, patch_kernel
from cuda_driver import check, cuda
from sm86_fixture import load_group
from ttu_isa import input_registers, ttumf

STRIDE = 512
REGISTERS = 96
SNAPSHOT_REGISTERS = 92


def clock(register, wait=0):
    # nvcc 12.1: CS2R R<even>, SR_CLOCKLO (64-bit). Four-cycle issue delay.
    return (0x7805 | register << 16, 0x000fc80000015000 | wait << 52)


def arithmetic(count):
    # FADD R88, R88, 1.0; four-cycle dependency spacing.
    return [(0x7421 | 88 << 16 | 88 << 24 | 0x3f800000 << 32,
             0x000fc80000000000)] * count


def build(values, *, group='reduced', samples=4, alu=0, placement='before-wait', open_gap=None):
    if group not in ('none', 'nop', 'mf0', 'reduced', 'expanded'):
        raise ValueError('unknown group')
    if not 1 <= samples <= 8 or not 0 <= alu <= 512:
        raise ValueError('samples must be 1..8 and alu 0..512')
    if placement not in ('before-wait', 'after-wait'):
        raise ValueError('unknown placement')
    if open_gap is not None and (group not in ('reduced', 'expanded') or not 0 <= open_gap <= 128):
        raise ValueError('OPEN gap requires a TTU group and 0..128 FADDs')
    ttu = load_group(expanded=group == 'expanded') if group in ('reduced', 'expanded') else b''
    body = ttu if ttu else (ttumf(0) if group == 'mf0' else
                           struct.pack('<QQ', 0x7918, 0x000fe80000000000) if group == 'nop' else b'')
    sequence = [CUDA_ENTRY,
                (0x7919 | 78 << 16, 0x000e220000000000),  # S2R R78, SR_LANEID; writes SB0
                mov_immediate(79, samples * STRIDE),
                # IMAD.WIDE.U32 R76, R78, R79, c[0][0x160]; waits for SB0.
                (0x0000580000007625 | 76 << 16 | 78 << 24, 0x001fca00078e004f)]
    offsets = []
    for sample in range(samples):
        sequence.append((0x7918, 0x03ffe20000000000))  # wait all before input restoration
        sequence.extend(mov_immediate(register, 0) for register in range(76))
        sequence.extend(mov_immediate(register, values[register]) for register in input_registers(ttu))
        sequence.append(mov_immediate(88, 0))
        sequence.append(clock(80, wait=0x3f))
        offsets.append(dict(sample=sample, group_offset=len(sequence) * 16, group_bytes=len(body),
                            inserted_bytes=(open_gap + 1) * 16 if open_gap is not None else 0))
        words = list(struct.iter_unpack('<QQ', body))
        if open_gap is not None:
            # OPEN is outside the fused body. Preserve MF count/body exactly.
            sequence.append(words[0])
            sequence.append(clock(90))
            sequence.extend(arithmetic(open_gap))
            sequence.extend(words[1:])
        else:
            sequence.extend(words)
        sequence.append(clock(82))
        if placement == 'before-wait':
            sequence.extend(arithmetic(alu))
        sequence.append(clock(84, wait=0x3f))
        if placement == 'after-wait':
            sequence.extend(arithmetic(alu))
        sequence.append(clock(86))
        # Keep padding above the highest 128-bit store source (R88..R91).
        for register in range(0, SNAPSHOT_REGISTERS, 4):
            displacement = sample * STRIDE + register * 4
            sequence.append((0x7986 | 76 << 24 | register << 32 | displacement << 40,
                             0x0001e8000c101d04))  # STG read dependency barrier0
            # A store may consume its source registers after issue. Protect
            # snapshots from the next sample overwriting clock/result slots.
            sequence.append((0x7918, 0x001fe20000000000))
    sequence.append(EXIT)
    template = compile_template(KERNELS / 'timing_template.cu')
    image = patch_kernel(template, 'ttu_timing', sequence, registers=REGISTERS)
    info = dict(cubin_sha256=hashlib.sha256(image).hexdigest(),
                group_sha256=hashlib.sha256(body).hexdigest(), offsets=offsets,
                instruction_count=len(sequence), registers=REGISTERS,
                timing_registers=[80, 82, 84, 86], accumulator_register=88,
                open_gap=open_gap, after_open_clock_register=90 if open_gap is not None else None,
                note='All samples restore inputs; clocks/ALU outside fused group; 1 CTA, 1..32 active lanes')
    return bytes(image), info


class TimingProbe:
    def __init__(self, image, *, lanes, samples):
        if not 1 <= lanes <= 32:
            raise ValueError('one warp only')
        self.lanes, self.samples = lanes, samples
        self.size = lanes * samples * STRIDE
        self.output = ctypes.c_uint64()
        self.module, self.function = ctypes.c_void_p(), ctypes.c_void_p()
        check(cuda.cuMemAlloc_v2(ctypes.byref(self.output), self.size), 'timing output allocate')
        check(cuda.cuModuleLoadData(ctypes.byref(self.module), image), 'timing module')
        check(cuda.cuModuleGetFunction(ctypes.byref(self.function), self.module, b'ttu_timing'), 'timing function')

    def launch(self):
        zero = bytes(self.size)
        check(cuda.cuMemcpyHtoD_v2(self.output.value, zero, self.size), 'timing clear')
        parameters = (ctypes.c_void_p * 1)(ctypes.addressof(self.output))
        check(cuda.cuLaunchKernel(self.function, 1, 1, 1, self.lanes, 1, 1, 0, None, parameters, None), 'timing launch')
        check(cuda.cuCtxSynchronize(), 'timing synchronize')
        output = ctypes.create_string_buffer(self.size)
        check(cuda.cuMemcpyDtoH_v2(output, self.output.value, self.size), 'timing readback')
        return [[struct.unpack_from('<92I', output.raw, (lane * self.samples + sample) * STRIDE)
                 for sample in range(self.samples)] for lane in range(self.lanes)]

    def close(self):
        check(cuda.cuModuleUnload(self.module), 'timing unload')
        check(cuda.cuMemFree_v2(self.output), 'timing output free')
