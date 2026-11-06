"""One-CTA SM86 TTU/FADD probes with checked fixed-count or common-window loops.

In common-window mode, R4 counts completions; R5 is elapsed-clock scratch;
R6:R7/R8:R9 mark first work/last completion; R10 holds the shared clock epoch;
R11 records VIRTID. R76:R77 address output, R78 is thread ID, R79/R84 are SM IDs,
R80:R83 delimit role execution, R85 checks gap work, R86 counts iterations,
R87/R89:R91 hold compute chains, and R88 accumulates TTU result errors.
R92:R93 are clock scratch. The fixture's TTU inputs do not overlap R4:R11.
"""
import ctypes
import hashlib
import struct

from cubin_builder import CUDA_ENTRY, EXIT, compile_template, mov_immediate, patch_kernel
from cuda_driver import check, cuda
from mixed_timing import control, nop
from project_paths import KERNELS
from sm86_fixture import load_group
from ttu_isa import input_registers, unpack, ttumf, set_field, fused_group
from warp_timing import clock, arithmetic


STRIDE = 384
REGISTERS = 96
RETURN_REGISTERS = (72, 73, 32, 33)


def xor_immediate(destination, source, value):
    return control(0x7812 | destination << 16 | source << 24 | value << 32, 0x078e3cff)


def or_register(destination, source):
    return control(0x7212 | destination << 16 | destination << 24 | source << 32, 0x078efcff)


def branch(source, target, predicate=7):
    displacement = (target - source - 16) & ((1 << 50) - 1)
    return (0x947 | predicate << 12 | (displacement & 0xffffffff) << 32,
            0x000fea0003800000 | displacement >> 32)


def build(values, expected, *, group='reduced', iterations=128, checking='each', wait_lead=2, consumer_delay=15,
          restore_mode='returned', ld_stall=None, return_barrier=1, wait_mask=63, wait_site='clock',
          variant='normal', gap=0, gap_warps=16, trace=False,
          ttu_warps=16, role_start=0, compute_count=64, compute_iterations=None, peer_delay=0,
          compute_chains=1, return_loads=None, return_pad=False, window_cycles=0,
          ttu_mask=None, compute_mask=None, return_mode='prefix', close_gap=0, close_gap_site='before-close'):
    compute_iterations = iterations if compute_iterations is None else compute_iterations
    if group not in ('none', 'reduced', 'expanded') or checking not in ('each', 'final'):
        raise ValueError('invalid group or checking mode')
    if not 1 <= iterations <= 4096 or len(expected) != 4:
        raise ValueError('iterations must be 1..4096 and expected must contain four words')
    if not 0 <= wait_lead <= 8 or not 0 <= consumer_delay <= 15:
        raise ValueError('wait_lead must be 0..8 and consumer_delay 0..15')
    if restore_mode not in ('returned', 'all'):
        raise ValueError('restore_mode must be returned or all')
    if ld_stall is not None and not 1 <= ld_stall <= 15:
        raise ValueError('ld_stall must be 1..15')
    if return_barrier not in range(1, 6) or not 0 <= wait_mask <= 63 or wait_site not in ('clock', 'nop', 'consumer'):
        raise ValueError('invalid return barrier, wait mask or wait site')
    if wait_site == 'consumer' and checking != 'each':
        raise ValueError('consumer wait requires per-request checking')
    if variant not in ('normal', 'before-open', 'open-gap', 'split9', 'split10', 'after-close', 'after-wait'):
        raise ValueError('invalid gap variant')
    if not 0 <= gap <= 4096 or not 0 <= gap_warps <= 16 or not 0 <= ttu_warps <= 16:
        raise ValueError('gap 0..4096 and warp counts 0..16 required')
    if not 0 <= role_start <= 15 or role_start + ttu_warps > 16 or not 1 <= compute_count <= 1024:
        raise ValueError('invalid role range or compute count')
    if trace and iterations != 1:
        raise ValueError('trace requires iterations=1')
    if not 1 <= compute_iterations <= 4096 or (trace and wait_site == 'consumer'):
        raise ValueError('invalid compute iterations or trace with consumer wait')
    if not 0 <= peer_delay <= 1024:
        raise ValueError('peer_delay must be 0..1024')
    if compute_chains not in (1, 4):
        raise ValueError('compute_chains must be 1 or 4')
    if return_loads is not None and (group != 'expanded' or not 1 <= return_loads <= 7):
        raise ValueError('return_loads requires expanded group and 1..7')
    if return_pad and return_loads is None:
        raise ValueError('return_pad requires return_loads')
    if window_cycles and (not 10000 <= window_cycles <= 1000000 or trace or checking != 'each' or peer_delay):
        raise ValueError('window requires 10000..1000000 cycles, checking each, no trace or peer delay')
    if return_mode not in ('prefix', 'sink', 'repeat') or (return_mode != 'prefix' and return_loads is None):
        raise ValueError('invalid return_mode')
    if not 0 <= close_gap <= 1024 or (close_gap and (return_mode == 'prefix' or trace)):
        raise ValueError('close_gap 0..1024 requires a fixed seven-load return mode and no trace')
    if close_gap_site not in ('before-close', 'after-close'):
        raise ValueError('invalid close gap site')
    if (ttu_mask is None) != (compute_mask is None) or (ttu_mask is not None and
            (not 0 <= ttu_mask <= 0xffff or not 0 <= compute_mask <= 0xffff or ttu_mask & compute_mask)):
        raise ValueError('role masks must both be provided, disjoint and 16-bit')
    if group == 'none' and variant not in ('normal', 'before-open', 'after-close', 'after-wait'):
        raise ValueError('protocol gap requires TTU group')
    raw = load_group(expanded=group == 'expanded')
    if return_loads is not None:
        instructions = unpack(raw)
        payload = instructions[2:12 + return_loads] if return_mode == 'prefix' else instructions[2:]
        if return_mode == 'sink':
            for index in range(10 + return_loads, len(payload)):
                payload[index] = set_field(set_field(payload[index], 'rd', 255), 'register2', 255)
        elif return_mode == 'repeat':
            for index in range(10 + return_loads, len(payload)):
                close = index == len(payload) - 1
                payload[index] = set_field(payload[10], 'close', int(close))
        payload[-1] = set_field(payload[-1], 'close', 1)
        raw = fused_group(payload, opening=instructions[0],
                          macro_upper=int.from_bytes(instructions[1][8:], 'little'))
    body = list(struct.iter_unpack('<QQ', raw)) if group != 'none' else []
    for index, (low, upper) in enumerate(body):
        if low & 0xfff == 0x3d2:
            upper = (upper & ~(7 << 46)) | return_barrier << 46
            if ld_stall is not None:
                upper = (upper & ~(15 << 41)) | ld_stall << 41
            body[index] = low, upper
    inputs = input_registers(raw)
    overwritten = set()
    for instruction in unpack(raw):
        encoded = int.from_bytes(instruction, 'little')
        if encoded & 0xfff == 0x3d2:
            for start in ((encoded >> 16) & 255, (encoded >> 64) & 255):
                if start != 255:
                    overwritten.update((start, start + 1))
    restore = inputs if restore_mode == 'all' else sorted(overwritten.intersection(inputs))
    sequence = [CUDA_ENTRY,
        control(0x7919 | 78 << 16, 0x2100, wr=0, stall=1),
        control(0x7919 | 79 << 16, 0x4300, wr=1, stall=1),
        mov_immediate(90, STRIDE),
        control(0x0000580000007625 | 76 << 16 | 78 << 24, 0x078e005a, wait=1, stall=8)]
    sequence.extend(mov_immediate(register, 0) for register in range(76))
    sequence.extend(mov_immediate(register, values[register]) for register in inputs)
    sequence.extend([mov_immediate(86, 0 if window_cycles else iterations), mov_immediate(88, 0), mov_immediate(87, 0),
                     mov_immediate(85, 0), nop(63),
                     control(0x7919 | 11 << 16, 0x300, wr=2, stall=1), nop(4)])

    def compare_thread(threshold):
        return control(0x780c | 78 << 24 | threshold << 32, 0x03f06070, stall=13)

    def elapsed_clock(register):
        """Subtract epoch R10 into R5 with unsigned 32-bit wraparound."""
        return control(0x7824 | 5 << 16 | register << 24 | 1 << 32, 0x078e0a0a, stall=8)

    def compare_elapsed(threshold):
        return control(0x780c | 5 << 24 | threshold << 32, 0x03f06070, stall=13)

    def window_clock(register):
        """Use conservative clock-to-integer spacing, not a latency minimum."""
        low, upper = clock(register)
        return control(low, upper, stall=15)

    if window_cycles:
        sequence.extend([mov_immediate(4, 0), compare_thread(1), window_clock(92),
                         control(0x388 | 8 << 12 | 255 << 24 | 92 << 32, 0x800, rd=3), nop(8),
                         (0x7b1d, 0x000fec0000000000),
                         control(0x7984 | 10 << 16 | 255 << 24, 0x800, wr=3, stall=1), nop(8), clock(80)])
    else:
        sequence.extend([(0x7b1d, 0x000fec0000000000), clock(80)])

    dispatch = []
    idle_dispatch = []
    ttu_dispatch = []
    if ttu_mask is not None:
        for warp in range(16):
            sequence.append(compare_thread((warp + 1) * 32))
            selected = ttu_dispatch if ttu_mask & (1 << warp) else dispatch if compute_mask & (1 << warp) else idle_dispatch
            selected.append((len(sequence), 8))
            sequence.append((0, 0))
    elif role_start:
        sequence.append(compare_thread(role_start * 32))
        dispatch.append((len(sequence), 8))
        sequence.append((0, 0))
    if ttu_mask is None:
        sequence.append(compare_thread((role_start + ttu_warps) * 32))
        dispatch.append((len(sequence), 0))
        sequence.append((0, 0))

    def start_window():
        if window_cycles:
            offset = len(sequence) * 16
            sequence.extend([window_clock(92), elapsed_clock(92), compare_elapsed(4096)])
            sequence.append(branch(len(sequence) * 16, offset, 8))
            sequence.append(clock(6))

    for offset, predicate in ttu_dispatch:
        sequence[offset] = branch(offset * 16, len(sequence) * 16, predicate)
    start_window()
    if peer_delay:
        sequence.append(compare_thread(gap_warps * 32))
        skip_delay = len(sequence)
        sequence.append((0, 0))
        sequence.extend((low & ~(255 << 16 | 255 << 24) | 87 << 16 | 87 << 24, upper)
                        for low, upper in arithmetic(peer_delay))
        sequence[skip_delay] = branch(skip_delay * 16, len(sequence) * 16, 8)
    loop_offset = len(sequence) * 16
    sequence.extend(mov_immediate(register, values[register]) for register in restore)

    def append_gap():
        if not gap:
            return
        sequence.append(compare_thread(gap_warps * 32))
        skip = len(sequence)
        sequence.append((0, 0))
        if trace:
            sequence.append(clock(6))
        sequence.extend((low & ~(255 << 16 | 255 << 24) | 85 << 16 | 85 << 24, upper)
                        for low, upper in arithmetic(gap))
        if trace:
            sequence.append(clock(8))
        sequence[skip] = branch(skip * 16, len(sequence) * 16, 0)

    def append_close_gap():
        sequence.append(compare_thread(gap_warps * 32))
        skip = len(sequence)
        sequence.append((0, 0))
        sequence.extend((low & ~(255 << 16 | 255 << 24) | 85 << 16 | 85 << 24, upper)
                        for low, upper in arithmetic(close_gap))
        sequence[skip] = branch(skip * 16, len(sequence) * 16, 0)

    if variant == 'before-open':
        append_gap()
    group_offset = len(sequence) * 16
    if body:
        sequence.append(body[0])
        if trace:
            sequence.append(clock(4))
        if variant == 'open-gap':
            append_gap()
        if variant in ('split9', 'split10') or close_gap:
            split = len(body) - 3 if close_gap else 9 if variant == 'split9' else 10
            payload = body[2:]
            sequence.append(struct.unpack('<QQ', ttumf(split)))
            sequence.extend(payload[:split])
            append_gap()
            if close_gap and close_gap_site == 'before-close':
                append_close_gap()
            sequence.append(struct.unpack('<QQ', ttumf(len(payload) - split)))
            sequence.extend(payload[split:])
            if close_gap and close_gap_site == 'after-close':
                append_close_gap()
        else:
            sequence.extend(body[1:])
    if group == 'none':
        sequence.extend(mov_immediate(register, value) for register, value in zip(RETURN_REGISTERS, expected))
    if return_pad:
        sequence.extend([nop()] * (7 - return_loads))
    if trace:
        sequence.append(clock(18))
    if variant == 'after-close':
        append_gap()
    sequence.extend([nop()] * wait_lead)
    if wait_site == 'clock':
        sequence.append(clock(92, wait=wait_mask))
    elif wait_site == 'nop':
        sequence.append(nop(wait_mask))
    if consumer_delay:
        sequence.append(control(0x7918, 0, stall=consumer_delay))
    if trace:
        sequence.append(clock(20))
    if variant == 'after-wait' and wait_site != 'consumer':
        append_gap()

    def append_checks():
        for index, (register, value) in enumerate(zip(RETURN_REGISTERS, expected)):
            instruction = xor_immediate(89, register, value)
            if index == 0 and wait_site == 'consumer':
                instruction = instruction[0], instruction[1] | wait_mask << 52
            sequence.extend([instruction, or_register(88, 89)])

    if checking == 'each':
        append_checks()
    if variant == 'after-wait' and wait_site == 'consumer':
        append_gap()

    def loop_control():
        if window_cycles:
            sequence.extend([control(0x7810 | 86 << 16 | 86 << 24 | 1 << 32, 0x07ffe0ff, stall=8),
                             window_clock(8), elapsed_clock(8), compare_elapsed(4096 + window_cycles),
                             control(0x810 | 4 << 16 | 4 << 24 | 8 << 12 | 1 << 32, 0x07ffe0ff, stall=8)])
        else:
            sequence.extend([control(0x7810 | 86 << 16 | 86 << 24 | 0xffffffff << 32, 0x07ffe0ff, stall=8),
                             control(0x720c | 86 << 24 | 255 << 32, 0x03f05270, stall=13)])

    loop_control()
    branch_offset = len(sequence) * 16
    sequence.append(branch(branch_offset, loop_offset, 8 if window_cycles else 0))
    if checking == 'final':
        append_checks()
    join_branch = len(sequence)
    sequence.append((0, 0))
    compute_offset = len(sequence) * 16
    for offset, predicate in dispatch:
        sequence[offset] = branch(offset * 16, compute_offset, predicate)
    sequence.append(mov_immediate(86, 0 if window_cycles else compute_iterations))
    sequence.extend(mov_immediate(register, 0) for register in (87, 89, 90, 91))
    start_window()
    compute_loop = len(sequence) * 16
    for index in range(compute_count):
        register = (87, 89, 90, 91)[index % compute_chains]
        sequence.append(control(0x7421 | register << 16 | register << 24 | 0x3f800000 << 32,
                                0, stall=4 if compute_chains == 1 else 1))
    loop_control()
    sequence.append(branch(len(sequence) * 16, compute_loop, 8 if window_cycles else 0))
    sequence[join_branch] = branch(join_branch * 16, len(sequence) * 16)
    for offset, predicate in idle_dispatch:
        sequence[offset] = branch(offset * 16, len(sequence) * 16, predicate)
    sequence.extend([clock(82, wait=63),
                     control(0x7919 | 84 << 16, 0x4300, wr=0, stall=1),
                     nop(63), (0x7b1d, 0x000fec0000000000)])
    for register in range(0, 92, 4):
        sequence.extend([(0x7986 | 76 << 24 | register << 32 | register * 4 << 40,
                          0x0001e8000c101d04), nop(1)])
    sequence.append(EXIT)
    image = bytes(patch_kernel(compile_template(KERNELS / 'multiwarp_template.cu'),
                              'ttu_multiwarp', sequence, registers=REGISTERS))
    info = dict(cubin_sha256=hashlib.sha256(image).hexdigest(),
                group_sha256=hashlib.sha256(raw).hexdigest() if body else None,
                registers=REGISTERS, static_shared_bytes=0, local_bytes=0,
                instruction_count=len(sequence), loop_offset=loop_offset, group_offset=group_offset,
                branch_offset=branch_offset, restored_registers=restore,
                checking=checking, iterations=iterations,
                wait_lead_nops=wait_lead, consumer_delay=consumer_delay,
                restore_mode=restore_mode,
                ld_stall=ld_stall, return_barrier=return_barrier, wait_mask=wait_mask, wait_site=wait_site,
                variant=variant, gap=gap, gap_warps=gap_warps, trace=trace,
                ttu_warps=ttu_warps, role_start=role_start, compute_count=compute_count,
                compute_iterations=compute_iterations,
                peer_delay=peer_delay,
                compute_chains=compute_chains,
                return_loads=return_loads,
                return_pad=return_pad,
                window_cycles=window_cycles, window_guard_cycles=4096, dynamic_shared_bytes=4 if window_cycles else 0,
                ttu_mask=ttu_mask, compute_mask=compute_mask, return_mode=return_mode, close_gap=close_gap,
                close_gap_site=close_gap_site,
                issued_group_sha256=hashlib.sha256(b''.join(struct.pack('<QQ', *instruction) for instruction in body)).hexdigest(),
                timed_loop_instructions=(branch_offset + 16 - loop_offset) // 16)
    return image, info


class MultiwarpProbe:
    def __init__(self, image, *, warps, shared_bytes=0):
        if warps not in (1, 2, 4, 8, 16):
            raise ValueError('warps must be 1, 2, 4, 8 or 16')
        self.threads = warps * 32
        self.shared_bytes = shared_bytes
        self.size = (self.threads + 2) * STRIDE
        self.output = ctypes.c_uint64()
        self.module, self.function = ctypes.c_void_p(), ctypes.c_void_p()
        check(cuda.cuMemAlloc_v2(ctypes.byref(self.output), self.size), 'multiwarp allocate')
        check(cuda.cuModuleLoadData(ctypes.byref(self.module), image), 'multiwarp module')
        check(cuda.cuModuleGetFunction(ctypes.byref(self.function), self.module, b'ttu_multiwarp'), 'multiwarp function')
        cuda.cuFuncGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]
        self.attributes = {}
        for name, attribute in (('max_threads', 0), ('shared_bytes', 1), ('local_bytes', 3), ('registers', 4)):
            value = ctypes.c_int()
            check(cuda.cuFuncGetAttribute(ctypes.byref(value), attribute, self.function), 'multiwarp attribute')
            self.attributes[name] = value.value
        cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        blocks = ctypes.c_int()
        check(cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(blocks), self.function, self.threads, shared_bytes), 'multiwarp occupancy')
        self.attributes['max_resident_blocks'] = blocks.value

    def launch(self):
        sentinel = b'\xa5' * self.size
        check(cuda.cuMemcpyHtoD_v2(self.output.value, sentinel, self.size), 'multiwarp clear')
        start = ctypes.c_uint64(self.output.value + STRIDE)
        parameters = (ctypes.c_void_p * 1)(ctypes.addressof(start))
        check(cuda.cuLaunchKernel(self.function, 1, 1, 1, self.threads, 1, 1, self.shared_bytes, None, parameters, None), 'multiwarp launch')
        check(cuda.cuCtxSynchronize(), 'multiwarp synchronize')
        output = ctypes.create_string_buffer(self.size)
        check(cuda.cuMemcpyDtoH_v2(output, self.output.value, self.size), 'multiwarp readback')
        guards = output.raw[:STRIDE] == sentinel[:STRIDE] and output.raw[-STRIDE:] == sentinel[-STRIDE:]
        records = [struct.unpack_from('<96I', output.raw, (thread + 1) * STRIDE) for thread in range(self.threads)]
        return records, guards

    def close(self):
        check(cuda.cuModuleUnload(self.module), 'multiwarp unload')
        check(cuda.cuMemFree_v2(self.output), 'multiwarp free')
