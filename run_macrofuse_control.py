#!/usr/bin/env python3
"""Targeted MF necessity controls; keep all ST/GO/LD.CLOSE and scheduling.

Run each variant in a fresh process with an external timeout. The MF instruction
is replaced by a matched NOP or MF(0), without deleting OPEN or CLOSE.
"""
import argparse
import hashlib
from pathlib import Path
import struct
import sys
from cuda_driver import sections
from ttu_isa import set_field
import run_mixed_timing as runner


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode',choices=('nop','zero'),required=True)
    args,rest=parser.parse_known_args()
    original=runner.build
    def build(*positional,**kwargs):
        if kwargs['variant']!='normal' or kwargs['group']=='none' or kwargs['gap']:
            raise ValueError('control requires a normal TTU group with no gap')
        image,info=original(*positional,**kwargs)
        image=bytearray(image)
        section=next(s for s in sections(image) if s['name']=='.text.ttu_timing')
        replacements=[]
        for sample in info['offsets']:
            offset=section['off']+sample['group_offset']+16
            before=bytes(image[offset:offset+16])
            if args.mode=='zero': after=set_field(before,'count',0)
            else:
                upper=struct.unpack_from('<Q',before,8)[0] & (((1<<23)-1)<<41)
                after=struct.pack('<QQ',0x7918,upper)
            image[offset:offset+16]=after
            replacements.append(dict(text_offset=sample['group_offset']+16,before=before.hex(),after=after.hex()))
        info.update(cubin_sha256=hashlib.sha256(image).hexdigest(),
            macrofuse_control=dict(mode=args.mode,replacements=replacements,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                note='Matched NOP or MF(0) replaces MF; all ST/GO/LD.CLOSE unchanged'))
        return bytes(image),info
    runner.build=build
    sys.argv=[sys.argv[0]]+rest
    runner.main()


if __name__=='__main__': main()
