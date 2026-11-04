#!/usr/bin/env python3
"""Reconstructed Phase-1 EoSS experiments; original source was not recovered.

The new harness initializes R0..R75 and dumps registers after the sequence.
It is not byte-identical to the deleted FADD-chain harness.
"""
from project_paths import KERNELS
import argparse
import struct
from pathlib import Path

from cubin_builder import CUDA_ENTRY, EXIT, compile_template, dump_registers, mov_immediate, patch_kernel
from eoss_helpers import canonical_sm120, replace_full
from ttu_isa import inst, ttucctl, ttugo, ttuld, ttumf, ttuopen, ttust

EXPERIMENTS = ('probe', 'exp1_open1', 'exp2_open2', 'exp3_mf0', 'exp4_cctl',
               'exp4_cctl_ivall', 'exp4_cctl_wr0', 'exp4_cctl_ivall_wr0', 'exp5_group')
DEFAULT_EXPERIMENTS = ('probe', 'exp1_open1', 'exp2_open2', 'exp3_mf0', 'exp4_cctl',
                       'exp4_cctl_ivall', 'exp5_group')


def sequence(name):
    opening = ttuopen(upper=0x000fec0000000000)
    if name == 'probe':
        return b''
    if name == 'exp1_open1':
        return opening
    if name == 'exp2_open2':
        return opening * 2
    if name == 'exp3_mf0':
        return ttumf(0, upper=0x000fc80000000000)
    if name.startswith('exp4_cctl'):
        upper = 0x000fe20000000000
        if name.endswith('wr0'):
            upper &= ~(7 << 46)
        return ttucctl(ivall='ivall' in name, upper=upper)
    if name == 'exp5_group':
        return canonical_sm120()
    raise ValueError(f'unknown experiment: {name}')


def build_experiment(name, output_address, *, wait_mask=0):
    group = sequence(name)
    instructions = [CUDA_ENTRY]
    instructions.extend(mov_immediate(register, 0) for register in range(76))
    instructions.extend(struct.iter_unpack('<QQ', group))
    instructions.extend(dump_registers(output_address, wait_mask=wait_mask))
    instructions.append(EXIT)
    template = compile_template(KERNELS / 'minimal_template.cu')
    return patch_kernel(template, 'ttu_minimal', instructions, replace=replace_full)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=EXPERIMENTS)
    parser.add_argument('--output-address', type=lambda value: int(value, 0), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-mask', type=lambda value: int(value, 0), default=0)
    args = parser.parse_args()
    args.output.write_bytes(build_experiment(args.experiment, args.output_address, wait_mask=args.wait_mask))


if __name__ == '__main__':
    main()
