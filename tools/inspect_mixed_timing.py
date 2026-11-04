#!/usr/bin/env python3
"""Record ordinary SASS and TTU fuse boundaries from an actually tested cubin.

The sanitized binary is disassembly input only; it must never be executed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from cuda_driver import sections
from ttu_isa import decode


def inspect(report_path, destination):
    report=json.loads(report_path.read_text())
    source=Path(report['args']['artifact'])
    image=source.read_bytes()
    if hashlib.sha256(image).hexdigest()!=report['build']['cubin_sha256']:
        raise ValueError('cubin differs from tested report')
    section=next(s for s in sections(image) if s['name']=='.text.ttu_timing')
    first=report['build']['offsets'][0]['group_offset']
    next_sample=report['build']['offsets'][1]['group_offset']
    code=bytearray(image[section['off']:section['off']+next_sample])
    removed=[]
    for offset in range(0,len(code),16):
        instruction=bytes(code[offset:offset+16])
        try: decoded=decode(instruction)
        except ValueError: continue
        if decoded['name']=='TTUMACROFUSE':
            count=decoded['count']
            for pos in range(offset+16,offset+16*(count+1),16):
                following=decode(bytes(code[pos:pos+16]))
                if following['name'] not in ('TTUST','TTUGO','TTULD','TTUCCTL','TTUMACROFUSE'):
                    raise ValueError('unexpected instruction in fused body')
        removed.append(dict(offset=hex(offset),bytes=instruction.hex(),decoded=decoded))
        # Same control fields; replace only opcode/operands for disassembly.
    for row in removed:
        offset=int(row['offset'],16)
        upper=struct.unpack_from('<Q',code,offset+8)[0] & (((1<<23)-1)<<41)
        struct.pack_into('<QQ',code,offset,0x7918,upper)
    start=first-16*16
    end=next(offset+16 for offset in range(first,len(code),16)
             if struct.unpack_from('<Q',code,offset)[0] & 0xfff == 0x986)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        binary=Path(temporary)/'ordinary.bin'
        binary.write_bytes(code[start:end])
        command=['nvdisasm','-b','SM86','-ndf','-hex',str(binary)]
        disassembly=subprocess.run(command,capture_output=True,text=True,check=True).stdout
    destination.with_suffix('.sass').write_text('// TTU replaced by NOP only for disassembly; never execute.\n'+disassembly)
    metadata=dict(report=str(report_path),report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        cubin_source=str(source),cubin_sha256=hashlib.sha256(image).hexdigest(),
        section='.text.ttu_timing',excerpt_start=hex(start),excerpt_end=hex(end),
        excerpt_sha256=hashlib.sha256(destination.with_suffix('.sass').read_bytes()).hexdigest(),
        tool_sha256=hashlib.sha256(Path(shutil.which('nvdisasm')).read_bytes()).hexdigest(),ttu_removed=removed,
        instruction='NOP replacement is analysis-only; runtime used original TTU bytes')
    destination.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    inspect(args.report,args.destination)
