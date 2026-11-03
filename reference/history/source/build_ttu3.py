#!/usr/bin/env python3
"""Build probe3 cubin and the single phase-3 traversal cubin (e7).

Sequence (fused group of 8): ST[0x100](root) ST[0x040](origin.xy)
ST[0x050](dir part 1) ST[0x080](dir part 2) GO LD1 LD2 CLOSE.
TTUST stores 64 bits from the implicit pair Rb:Rb+1 (verified empirically:
Rb at encoding bits [39:32]; ctrl byte 0 must stay 0).
TTULD reads 64 bits into the implicit pair Rd:Rd+1 (Rd at bits [39:32]).
Result qword 0 -> out[0], qword 1 -> out2[0] (thread 0; the FADD pollution
term is 10*a which vanishes at tid 0).
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, "/home/hzx/gpu-engine/research/community")
from eoss.binary import BinaryPatcher, elf as eoss_elf      # noqa: E402
from eoss.binary.cuobjdump import parse_sass_instructions, disassemble  # noqa: E402
from eoss.enc import control as ctrl                        # noqa: E402

NOP = bytes.fromhex("18790000000000000000000000080000")

def inst(word, c): return word.to_bytes(8, "little") + c.to_bytes(8, "little")
def ttuopen():              return inst(0x00000000000073D0, 0x000FEC0000000000)
def ttumf(n):               return inst((n << 40) | 0x79D4, 0x000FC80000000000)
def ttust(addr, rb):        return inst(((addr & 0xFFFF) << 40) | ((rb & 0xFF) << 32) | 0x73D1,
                                        0x000FE20000000000)
def ttugo():                return inst(0x00000000000073D3, 0x000FE20000000000)
def ttuld(rd, close=False): return inst(((rd & 0xFF) << 32) | 0x73D2,
                                        0x000FE20000000100 if close else 0x000FE20000000000)

NSLOTS = 12


def main():
    cubin = HERE / "probe3.cubin"
    subprocess.run(["nvcc", "-arch=sm_86", "-cubin", str(HERE / "probe3.cu"),
                    "-o", str(cubin)], check=True)
    sass = disassemble(str(cubin))
    insns = parse_sass_instructions(sass, kernel_name="ttu_probe2")
    print(f"ttu_probe2: {len(insns)} instructions")

    ldg64 = [i for i in insns if i.text.startswith("LDG.E.64")]
    assert len(ldg64) == 5, f"expected 5 cfg loads, got {len(ldg64)}"
    dests = [int(re.search(r"LDG\.E\.64(?:\.\w+)*\s+(R\d+)", i.text).group(1)[1:]) for i in ldg64]
    root_lo, oxy_lo, ozz_lo, dxy_lo, dzz_lo = dests
    print(f"cfg pairs: root=R{root_lo}+1 oxy=R{oxy_lo}+1 ozz=R{ozz_lo}+1 dxy=R{dxy_lo}+1 dzz=R{dzz_lo}+1")

    stgs = [i for i in insns if i.text.startswith("STG")]
    assert len(stgs) == 2
    srcs = [int(re.search(r",\s*(R\d+)\s*;?$", i.text).group(1)[1:]) for i in stgs]
    print(f"STG srcs: out<-R{srcs[0]}  out2<-R{srcs[1]}")

    last_ldg = ldg64[-1].offset
    region = [i for i in insns if i.offset > last_ldg][:NSLOTS]
    assert len(region) == NSLOTS
    sec_off = eoss_elf.find_cuda_text_sections(cubin.read_bytes())[0][0]

    import os
    variant = os.environ.get("LDVAR", "w1")
    r0, r1 = srcs
    def ld(addr, rd, close=False, style="39"):
        if style == "39":
            w = ((addr & 0xFFFF) << 40) | ((rd & 0xFF) << 32) | 0x73D2
        else:  # "23"
            w = ((addr & 0xFFFF) << 40) | ((rd & 0xFF) << 16) | 0x73D2
        c = 0x000FE20000000100 if close else 0x000FE20000000000
        return inst(w, c)
    if variant == "w1":    # addr=0x300, dest@39:32
        seq = [ttuopen(), ttumf(8),
               ttust(0x100, root_lo), ttust(0x040, oxy_lo),
               ttust(0x050, dxy_lo), ttust(0x080, dzz_lo),
               ttugo(), ld(0x300, r0), ld(0x300, r1), ld(0x300, r1, close=True)]
    elif variant == "w2":  # addr=0, dest@23:16 (retest clean)
        seq = [ttuopen(), ttumf(8),
               ttust(0x100, root_lo), ttust(0x040, oxy_lo),
               ttust(0x050, dxy_lo), ttust(0x080, dzz_lo),
               ttugo(), ld(0, r0, style="23"), ld(0, r1, style="23"), ld(0, r1, style="23", close=True)]
    elif variant == "w3":  # addr=0x300, dest@23:16
        seq = [ttuopen(), ttumf(8),
               ttust(0x100, root_lo), ttust(0x040, oxy_lo),
               ttust(0x050, dxy_lo), ttust(0x080, dzz_lo),
               ttugo(), ld(0x300, r0, style="23"), ld(0x300, r1, style="23"), ld(0x300, r1, style="23", close=True)]
    elif variant == "w4":  # addr=0x300, dest at ctrl byte0 (bits 71:64)
        def ldc(addr, rd, close=False):
            c = (0x000FE20000000100 if close else 0x000FE20000000000) | (rd & 0xFF)
            return inst(((addr & 0xFFFF) << 40) | 0x73D2, c)
        seq = [ttuopen(), ttumf(8),
               ttust(0x100, root_lo), ttust(0x040, oxy_lo),
               ttust(0x050, dxy_lo), ttust(0x080, dzz_lo),
               ttugo(), ldc(0x300, r0), ldc(0x300, r1), ldc(0x300, r1, close=True)]
    elif variant == "x1":  # result ptr candidate at ST[0x000] (R9:R10 = ozz pair? no - use dzz pair)
        # store scratch pointer (runtime) via ozz pair; ray cfg via others
        seq = [ttuopen(), ttumf(8),
               ttust(0x100, root_lo), ttust(0x040, oxy_lo),
               ttust(0x050, dxy_lo), ttust(0x080, dzz_lo),
               ttugo(), ld(0x300, r0), ld(0x300, r1), ld(0x300, r1, close=True)]
    elif variant in ("b_mf", "c_stroot", "d_ld", "e_go"):
        # golden frame, swap ONE word at a time with driver version
        if variant == "b_mf":
            grp = [ttust(0x000,0), ttust(0x040,0), ttust(0x050,0),
                   ttust(0x080,0), ttugo(), ttuld(0), ttuld(0), ttuld(0, close=True)]
            seq = [ttuopen(), inst((8 << 40) | 0x79d4, 0x000fe80000000000)] + grp
        elif variant == "c_stroot":
            grp = [inst(0x00000000000083d1 | (root_lo << 32), 0x0005e200000000ff),
                   ttust(0x040,0), ttust(0x050,0),
                   ttust(0x080,0), ttugo(), ttuld(0), ttuld(0), ttuld(0, close=True)]
            seq = [ttuopen(), inst((8 << 40) | 0x79d4, 0x000fc80000000000)] + grp
        elif variant == "d_ld":
            import os
            rdv = int(os.environ.get("RD", "8"))
            rd2v = int(os.environ.get("RD2", str(rdv+2)))
            grp = [ttust(0x000,0), ttust(0x040,0), ttust(0x050,0),
                   ttust(0x080,0), ttugo(),
                   inst(0x0003400000000000 | (rdv << 16) | 0x73d2, 0x000ee200000e0000 | rd2v),
                   ttuld(0), ttuld(0, close=True)]
            seq = [ttuopen(), inst((8 << 40) | 0x79d4, 0x000fc80000000000)] + grp
        else:  # e_go sanity
            grp = [ttust(0x000,0), ttust(0x040,0), ttust(0x050,0),
                   ttust(0x080,0), ttugo(), ttuld(0), ttuld(0), ttuld(0, close=True)]
            seq = [ttuopen(), inst((8 << 40) | 0x79d4, 0x000fc80000000000)] + grp
    elif variant in ("f_hit", "f_miss", "fb_hit", "fb_miss"):
        # driver-style: 64-bit root ST (byte0=0xff), LD rd=8/12 (4-aligned), STG patched to R8/R12
        rootmode_ = "base" if variant.startswith("f_") else "node"
        ld1 = inst(0x0003400000000000 | (8 << 16) | 0x73d2, 0x000ee200000e0000 | 10)
        ld2 = inst(0x0003500000000000 | (12 << 16) | 0x73d2, 0x000ee200000e0000 | 14)
        st_root = inst(0x00000000000083d1 | (root_lo << 32), 0x0005e200000000ff)
        grp = [st_root, ttust(0x040, oxy_lo), ttust(0x050, dxy_lo), ttust(0x080, dzz_lo),
               ttugo(), ld1, ld2, ttuld(0, close=True)]
        seq = [ttuopen(), inst((8 << 40) | 0x79d4, 0x000fc80000000000)] + grp
    elif variant == "g":
        # driver 6-group verbatim; cfg[0] = pointer to OUR crafted descriptor (64-bit)
        ld1 = inst(0x0003400000000000 | (8 << 16) | 0x73d2, 0x000ee200000e0000 | 10)
        ld2 = inst(0x0003500000000000 | (12 << 16) | 0x73d2, 0x000ee200000e0000 | 14)
        ld3 = inst(0x0003600000000000 | (8 << 16) | 0x73d2, 0x000ee400000e04ff)
        st_root = inst(0x00000000000083d1 | (root_lo << 32), 0x0005e200000000ff)
        st_zero = inst(0x000280ff000073d1, 0x0005e20000000002)
        grp = [st_root, st_zero, ttugo(), ld1, ld2, ld3]
        seq = [ttuopen(), inst((6 << 40) | 0x79d4, 0x000fe80000000000)] + grp
    elif variant in ("d1", "d2"):
        # Driver-template replica with even LD dests (R14/R18) + STG src patch.
        st_root = inst(0x00000000000083d1 | (root_lo << 32), 0x0005e200000000ff)   # 64-bit root?
        if variant == "d2":
            st_root = inst(0x00000000000073d1 | (root_lo << 32), 0x0005e20000000000)  # 32-bit root
        st_zero = inst(0x000280ff000073d1, 0x0005e20000000002)
        go      = inst(0x00000000000073d3, 0x000fe20000000000)
        ld1     = inst(0x0003400000000000 | (14 << 16) | 0x73d2, 0x000ee200000e0000 | 16)
        ld2     = inst(0x0003500000000000 | (18 << 16) | 0x73d2, 0x000ee200000e0000 | 20)
        ld3     = inst(0x0003600000000000 | (14 << 16) | 0x73d2, 0x000ee400000e04ff)
        grp = [st_root, st_zero, go, ld1, ld2, ld3]
        seq = [ttuopen(), inst((len(grp) << 40) | 0x79d4, 0x000fe80000000000)] + grp
    elif variant in ("z1", "z2", "z3"):
        # z1: add ST[0x048] = ozz (origin z?), 5ST+GO+2LD+CLOSE = MF(9)
        # z2: like y1 but LD addr 0x308 for second LD
        # z3: like y1 but origin xy packed (ox,oy) swapped
        if variant == "z1":
            grp = [ttust(0x100, root_lo), ttust(0x040, oxy_lo), ttust(0x048, ozz_lo),
                   ttust(0x050, dxy_lo), ttust(0x080, dzz_lo)]
        elif variant == "z2":
            grp = [ttust(0x100, root_lo), ttust(0x040, oxy_lo),
                   ttust(0x050, dxy_lo), ttust(0x080, dzz_lo)]
        else:
            grp = [ttust(0x100, root_lo), ttust(0x040, oxy_lo),
                   ttust(0x050, dxy_lo), ttust(0x080, dzz_lo)]
        grp.append(ttugo())
        if variant == "z2":
            grp += [ld(0x300, r0), ld(0x308, r1), ld(0x300, r1, close=True)]
        else:
            grp += [ld(0x300, r0), ld(0x300, r1), ld(0x300, r1, close=True)]
        seq = [ttuopen(), ttumf(len(grp))] + grp
    elif variant in ("y1", "y2", "y3", "y4"):
        # 5 STs incl. ST[0x000] = scratch ptr (ozz pair); y4 drops GO
        has_go = variant != "y4"
        grp = [ttust(0x100, root_lo), ttust(0x040, oxy_lo),
               ttust(0x000, ozz_lo), ttust(0x050, dxy_lo), ttust(0x080, dzz_lo)]
        if has_go: grp.append(ttugo())
        grp += [ld(0x300, r0), ld(0x300, r1), ld(0x300, r1, close=True)]
        seq = [ttuopen(), ttumf(len(grp))] + grp
        if len(seq) > NSLOTS:  # need 11 slots: extend region in caller? just assert
            pass
    else:
        raise SystemExit("bad LDVAR")
    assert len(seq) <= NSLOTS, f"{variant}: {len(seq)} > {NSLOTS}"

    out = HERE / f"e7_{variant}.cubin"
    p = BinaryPatcher.from_bytes(cubin.read_bytes(), source=str(cubin))
    for i, b16 in enumerate(seq):
        p.patch_at(sec_off + region[i].offset, b16, ctrl.PatchPolicy.REPLACE_FULL)
    p.save(str(out))
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
