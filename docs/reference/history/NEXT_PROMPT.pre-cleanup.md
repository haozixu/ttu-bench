# NEXT PROMPT — Phase 12(主线:外核上下文真值 → 修复组输入 → t 翻转)

```
读 /home/hzx/gpu-engine/ttu-bench/HANDOFF.md:先读头部"卡点",再读 Phase 11
全段(读出通道教训链 §2 必读 —— 三条内存邮箱路线已判死,别重走),§1(编码
真值)按需查。Phase 10 §7(.X8 窗口)只在需要时查。INDEX.md §0 = 回归命令。

主线状态一句话:TTU 组 @0xe70 的返回值已用"寄存器中继"通道直接测得 =
R72:R73 = 0xFFFFFFFF_FFFFFFFF(64 位 -1),hit/miss 相同 —— 组在输入/上下文
层被引擎拒绝;静态输入(mask/oz/tmin/dz/tmax/根头路径)逐项核验完好;
缺口收敛为外核上下文寄存器 R56:57 / R60:61(ST[0x180]/0x190,mk+traversal
全程零写入 = 纯外核 driver 调度核所设)与 R40(min-t 累加器初值)。

先验证(3 条,全部应复现;GPU 先 nvidia-smi 挑空闲,CUDA_VISIBLE_DEVICES 隔离):
  python3 run_ttu4.py e7_w1.cubin optix_as.bin hit base          # q0=0x7fffffff
  MK_LOCAL=1 MK_RGR21=4 python3 run_mk.py hit 12                 # 干净退出 out=0
  MK_MAILBOX=1 MK_RGST=0x3f MK_LOCAL=1 MK_RGR21=4 \
    python3 run_mk.py hit 12                                     # out=ffffffff(t 中继)

本轮实验(按序;每发独立进程 + timeout;fault 毒化 context):

P1 e7 框架协议重放(推荐先做,自包含):e7_w1 = 自研核 = 全寄存器可控 +
   读出已验证。把 mttu 组 19 条 TTU 指令(jit_16 .text.traversal_mttu
   0xe70-0xfa0 的槽字节逐字 dump —— TTU 指令不在 tmtu_dis.txt 文本里,是
   反汇编空洞,字节在二进制里)重放进 e7 的 patch 区,寄存器按 e7 分配重绑
   (ST/LD 槽语义不变:0x0000=cfg(mask<<16,frac) 0x040=(oz,tmin)
   0x050=(dz,tmax) 0x080/0x0c0/0x0d0=复制 0x100/0x200=0 0x180=R56:57
   0x190=R60:61 → LD[0x300]→t)。
   判读:e7 里组能否产出哨兵/有限 t?
   - 若能 → 扫 R56:57/R60:61/R40 候选值(0,0 / shared 基址 / AS 指针 /
     0x7f800000=+inf …),哪个让 -1 翻转 = 外核上下文语义答案;
   - 若 e7 里也恒 -1 → -1 与上下文无关,转查组输入的更基本层
     (cfg 字/GO 前置条件/MF 结构),用 e7 逐项增删对比。
   工具:build_ttu5/6.py 槽位可变框架 + run_ttu5/6.py;组字节 dump 用
   run_mk.py 的 sections()。

P2 in-flight 补丁真管线(信息量最大,依赖 watch 基建):watch_capture.cpp
   (SIGSEGV 页保护 + handler 内改写,Phase 8 已建)+ patch_capture.cpp
   (双副本签名定位)。在 optixPipelineCreate 的 JIT 镜像写入流中,对
   traversal 0x1020/0x1030 与 raygen1 出店数据寄存器做中继补丁
   (R63←R72;RGST=R63)→ 真管线跑通后 out = 真 t;再中继 R56/R60 =
   直接读出外核上下文真值。注意 jit_*.elf dump 自宿主堆,无磁盘缓存可补
   (已核实 GLCache .bin=32B 非模块);上传发生在 optixPipelineCreate 内部,
   创建后补宿主镜像无效(Phase 8 判死),必须写入流中截。
   ⚠ 中继寄存器选择要保守:真管线里 R62/R63 被 walk 消费(0x12b0/0x3480 读),
   可能改变真管线行为 —— 若真管线崩,先退化为只读不改(R56/R60 本来就活到
   raygen1,直接 RGST 读,无需中继!)—— P2 的最小形态 = 只补 raygen1 出店
   RGST=R56/R60,零行为改动,纯观测。

P3 顺手账(如果 P1/P2 需要更多组内可见性):0x1020/0x1030 中继源寄存器可
   随意换(0x48=R72 0x49=R73 0x28=R40 0x00=R2 0x2a=R42…),一发一个值;
   ORIG 断言保护槽字节,别动 0xe70-0xfa0。

E3 记录链修复(在组产出真实 t/哨兵之后才有意义):un-skip yield 块(去掉
   mk+0x5b0 的 BRA)+ funptr 表 HtoD 补 14/15 槽防越界 + 把 traversal 完成
   码接到 CH(11)/miss(10)。CH 读 global [R2](R2=R3-8)→ 追 R3 赋值点。
E4 收官:out = 6.875 / 0xdead0000 → 按 INDEX §3 造多引擎争抢负载。

禁止(防跑偏):
- 不重走三条内存邮箱路线(X8-LDS 读侧 / c-bank 全局 STG / raygen1 本地
  LDL)—— 全部判死,证据链在 Phase 11 §2
- 不动 TTU 组 0xe70-0xfa0(P1 的重放是"拷贝到 e7",不是改 jit_16)
- 不做 .X8 imm 语义考古(0xfc-716 矛盾挂账 HANDOFF §7,主线不需要)
- 寄存器读数先过对照表(R40=-1 mk写 / R21=-7 raygen1写 / R106=payload /
  R43·R19=漂移垃圾 / R22=mk部分覆写(低24位0x200600固定,顶字节漂移)),
  再下结论
- 不删 jit_*.elf / optix_as.bin / heap_*.bin / xtab_*.bin / e7_w1.cubin /
  tmtu_dis.txt / _text_*.bin
- 编码新指令前先查 Phase 9 §5 / Phase 10 §1 / Phase 11 §2.5 的真值表;
  715 = 词形或未用字节非 0,700 = 地址,716 = 对齐(本地寻址从 raygen1
  全灭 —— 无 local 窗,别再试)

完成后:HANDOFF 加 Phase 12 段 + notes 追记 + README 状态段 + 本文件更新
为 Phase 13 任务书。
```
