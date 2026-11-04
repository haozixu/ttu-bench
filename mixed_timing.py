"""Phase19: explicit scoreboards, HMMA/LDG/LDS workloads, bounded MF variants.

Phase18 modules remain untouched so their recorded source hashes stay valid.
TTU result uses SB1, memory SB2, HMMA SB5; clocks C wait only on SB1.
"""
import ctypes
import hashlib
from pathlib import Path
import struct
from cubin_builder import CUDA_ENTRY, EXIT, compile_template, mov_immediate, patch_kernel
from cuda_driver import check, cuda
from sm86_fixture import load_group
from ttu_isa import input_registers, ttumf, unpack
from warp_timing import clock, arithmetic

HERE = Path(__file__).resolve().parent
STRIDE = 512


def control(low, upper, *, wait=0, wr=7, rd=7, stall=4):
    upper &= ~(((1 << 17)-1) << 41)
    upper |= stall << 41 | wr << 46 | rd << 49 | wait << 52
    return low, upper


def nop(wait=0):
    return control(0x7918, 0, wait=wait, stall=2)


def work(kind, count):
    if kind == 'fadd':
        return arithmetic(count)
    if kind == 'hmma':
        # HMMA.16816.F32 R96, R100, R100, R96. All f16 inputs are 1.
        # SB5 alone did not make immediate FADD consumers safe in pilots.
        # Use >=32 scheduled cycles per MMA, then consume all four results.
        return [control(0x723c | 96 << 16 | 100 << 24 | 100 << 32,
                        0x1800 | 96, wait=32, wr=5, stall=15),
                control(0x7918,0,stall=15),control(0x7918,0,stall=2)] * count + [nop(32)] + [
                    (0x7421 | (116+i)<<16 | (96+i)<<24,0x000fc80000000000) for i in range(4)]
    if kind == 'global':
        # LDG.E.64.STRONG.GPU R112,[R112.64], dependent 128-byte-spaced ring.
        return [control(0x7981 | 112 << 16 | 112 << 24 | 4 << 32,
                        0x0c1efb00, wait=4, wr=2, stall=1)] * count + [nop(4)] + [
                    control(0x7202 | r<<16 | (r-2)<<32,0xf00) for r in (114,115)]
    if kind == 'shared':
        # LDS R108,[R108], per-lane two-node ring, conflict-free across banks.
        return [control(0x7984 | 108 << 16 | 108 << 24,
                        0x0800, wait=4, wr=2, stall=1)] * count + [nop(4),
                    control(0x7202 | 109<<16 | 108<<32,0xf00)]
    if kind == 'none':
        return []
    raise ValueError(kind)


def build(values, pointer, *, kind='none', count=0, placement='before-wait',
          variant='normal', group='reduced', gap=0, samples=4):
    if kind not in ('none','fadd','hmma','global','shared') or not 0 <= count <= 128:
        raise ValueError('invalid workload')
    if variant not in ('normal','prefix-mf0','open-mf0','split9','split10','open-gap'):
        raise ValueError('invalid variant')
    if placement not in ('before-wait','after-wait') or not 0 <= gap <= 64 or not 2 <= samples <= 4:
        raise ValueError('invalid timing parameters')
    if group not in ('none','reduced','expanded') or (group == 'none' and variant != 'normal'):
        raise ValueError('invalid group')
    raw = load_group(expanded=group == 'expanded') if group != 'none' else b''
    body = list(struct.iter_unpack('<QQ',raw))
    if body:
        if variant == 'prefix-mf0': body = list(struct.iter_unpack('<QQ',ttumf(0))) + body
        if variant == 'open-mf0': body.insert(1, struct.unpack('<QQ',ttumf(0)))
        if variant == 'open-gap': body[1:1] = arithmetic(gap)
        if variant in ('split9','split10'):
            split = 9 if variant == 'split9' else 10
            payload=body[2:]
            body=[body[0],struct.unpack('<QQ',ttumf(split))] + payload[:split] + (
                [clock(90)] if split == 10 else []) + arithmetic(gap) + [
                struct.unpack('<QQ',ttumf(len(payload)-split))] + payload[split:]
    sequence=[CUDA_ENTRY,(0x7919 | 78<<16,0x000e220000000000),mov_immediate(79,samples*STRIDE),
        (0x0000580000007625 | 76<<16 | 78<<24,0x001fca00078e004f)]
    offsets=[]
    for sample in range(samples):
        sequence.append(nop(63))
        sequence.extend(mov_immediate(r,0) for r in list(range(76))+list(range(80,124)))
        sequence.extend(mov_immediate(r,values[r]) for r in input_registers(raw))
        sequence.extend(mov_immediate(r,0x3c003c00) for r in range(100,104))
        sequence.extend([mov_immediate(112,pointer & 0xffffffff),mov_immediate(113,pointer>>32)])
        # Per-lane shared ring: lane*4 <-> lane*4+128. Setup outside clocks.
        sequence.extend([control(0x7824 | 108<<16 | 78<<24 | 4<<32,0x078e00ff),
            # Keep setup forwarding conservative; tighter IMAD addend spacing
            # returned stale R110 in the pilot. No setup is inside clocks.
            mov_immediate(110,128),control(0x7918,0,stall=8),
            control(0x7824 | 110<<16 | 78<<24 | 4<<32,0x078e006e,stall=8),
            control(0x7388 | 78<<24 | 110<<32,0x4800,rd=3),nop(8),
            control(0x7388 | 78<<24 | 108<<32 | 128<<40,0x4800,rd=3),nop(8),nop(63)])
        sequence.append(clock(80))
        offsets.append(dict(sample=sample,group_offset=len(sequence)*16,group_instructions=len(body)))
        sequence.extend(body)
        sequence.append(clock(82))
        if placement=='before-wait': sequence.extend(work(kind,count))
        sequence.append(clock(84,wait=2))  # TTU only. Work has its own completion checks.
        # Immediate raw copies prove C actually waited, before later stores
        # could otherwise hide a too-early timestamp. Outside all fused bodies.
        sequence.extend(control(0x7202 | dst<<16 | src<<32,0xf00)
                        for dst,src in zip(range(120,124),(72,73,32,33)))
        if placement=='after-wait': sequence.extend(work(kind,count))
        sequence.append(clock(86))
        for r in range(0,124,4):
            sequence.extend([(0x7986 | 76<<24 | r<<32 | (sample*STRIDE+r*4)<<40,
                              0x0001e8000c101d04),nop(1)])
    sequence.append(EXIT)
    image=bytes(patch_kernel(compile_template(HERE/'timing_template.cu'),'ttu_timing',sequence,registers=128))
    info=dict(cubin_sha256=hashlib.sha256(image).hexdigest(),ttu_group_sha256=hashlib.sha256(raw).hexdigest(),
        instruction_count=len(sequence),registers=128,dynamic_shared_bytes=256,offsets=offsets,
        scoreboard=dict(ttu=1,memory=2,shared_setup=3,hmma=5,snapshot=0),
        register_map=dict(hmma_accumulator=[96,97,98,99],hmma_inputs=[100,101,102,103],shared=108,global_pointer=[112,113]))
    return image,info


class MixedProbe:
    def __init__(self,image,*,lanes=32,samples=4):
        self.lanes,self.samples=lanes,samples
        self.size=lanes*samples*STRIDE
        self.output=ctypes.c_uint64()
        self.module,self.function=ctypes.c_void_p(),ctypes.c_void_p()
        check(cuda.cuMemAlloc_v2(ctypes.byref(self.output),self.size),'output allocate')
        check(cuda.cuModuleLoadData(ctypes.byref(self.module),image),'module')
        check(cuda.cuModuleGetFunction(ctypes.byref(self.function),self.module,b'ttu_timing'),'function')

    def launch(self):
        parameters=(ctypes.c_void_p*1)(ctypes.addressof(self.output))
        check(cuda.cuLaunchKernel(self.function,1,1,1,self.lanes,1,1,256,None,parameters,None),'launch')
        check(cuda.cuCtxSynchronize(),'synchronize')
        output=ctypes.create_string_buffer(self.size)
        check(cuda.cuMemcpyDtoH_v2(output,self.output.value,self.size),'readback')
        return [[struct.unpack_from('<124I',output.raw,(lane*self.samples+s)*STRIDE)
                 for s in range(self.samples)] for lane in range(self.lanes)]

    def close(self):
        check(cuda.cuModuleUnload(self.module),'unload')
        check(cuda.cuMemFree_v2(self.output),'output free')
