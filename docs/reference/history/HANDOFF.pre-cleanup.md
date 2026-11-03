# HANDOFF — TTU/RT Core 逆向项目(给下一个 session)

## 目标
在 8×RTX 3090(SM86, driver 535.309.01)上绕过 OptiX,用裸 CUDA kernel 驱动 NVIDIA
隐藏的 TTU(RT Core)SASS 指令,最终造出自研 TTU 微基准 → 用于多引擎(RT Core/CUDA
core/NVDEC/NVENC)资源争抢研究。

## 当前状态(截至本 HANDOFF)
**已完成:**
1. TTU 8 类指令在 SM86 上首次公开执行;协议 = OPEN → MF(n) → 融合组 → (无 CLOSE)
2. 编码规则实测:寄存器 @ 编码字 bits[39:32];TTULD 目标寄存器必须 4-对齐
   (8/12 ✓,5/13/14 → illegal instruction);ctrl byte0 ∈ {0,0xff,0x02} 合法
3. 535 旧协议解码(逐字,来自 jit_08.elf 的 traversal_utrav MF(6) 组):
   `ST[0x0000]←R18(描述符指针), ST[0x0280]←RZ, GO, LD[0x340/350/360]→结果`
   与 EoSS 论文(新驱动≥570/SM120,8×TTUST 配置+root@[0x100])结构性不同
4. OptiX 8 最小管线跑通(HIT primID=0);20 个 JIT cubin 已 dump(jit_*.elf)
5. 描述符 oracle:ST[0x0000]=自构指针 → 引擎解引用(700 fault = 跟着走了)
6. **(Phase 9)启动模块全解码**:jit_16/19 = 已链接管线(raygen+megakernel+mttu);
   TTU 输入块 = 寄存器现算,ray 走 shared,ST[0x0000]=配置字;megakernel 可直接
   cuModuleLoadData 发射(run_mk.py)—— 详见 Phase 9

7. **(Phase 10)编码真值全集 + 全流程接通**:SM86 访存/c-bank 指令编码逐字段
   验证;Phase 9 三个补丁 bug 修复(prologue IMAD 字段错位=旧 715 主因、probe 缺
   R1、NOTTU 非法 NOP);**LMEMHIOFF=0xfff9c0(loader 一直建着 local 窗,Phase 9
   相反结论系 probe bug)**;raygen0→traversal→raygen1 首次接通;TTU 组确认执行;
   .X8 双 shared 窗口发现 —— 详见 Phase 10 + §7

8. **(Phase 11)组结果直读 + 寄存器中继通道**:组后牺牲槽 MOV R62/R63←R72:73 +
   MK_RGST = 已验证的组内寄存器读出;测得组返回 64 位 -1(拒绝码,双 ray 同);
   大量读出机制真值(X8-LDS 读侧从未工作 / raygen1 无 local 窗 / 寄存器文件顶
   R106 / STL b3 必须 0 / store 不 fault load fault)—— 详见 Phase 11

**卡点(Phase 11 更新:组被引擎拒绝,缺口 = 外核上下文寄存器真值):**
- ~~.128 spill 全局化~~ **判死**(§3);~~LMEMHIOFF 修复~~ **不需要**(窗口本来就有)
- ~~"组结果正确性未知"~~ **已测**(Phase 11 寄存器中继):TTU 组 @0xe70 返回
  **t = R72:R73 = 0xFFFFFFFF_FFFFFFFF(64 位 -1),hit/miss 相同** —— 引擎在
  输入/上下文层直接拒绝,连 miss 哨兵(0x7fffffff,e7 组实测可得)都不写
- 静态输入(mask/oz/tmin/dz/tmax/根头路径)逐项核验完好;**唯一无对照的输入 =
  R56:57 / R60:61(ST[0x180]/0x190 外核上下文,mk+traversal 全程零写入)+
  R40(min-t 初值)** —— 直接发射缺 driver 调度核的寄存器现场
- 现状:全流程干净退出但 out 恒 0;R3=0(无记录指针),CH/miss(槽 10/11)
  从未被调用 —— **E3 记录链在组产出真实 t/哨兵之前无意义**
- 跑通后:hit 期望 f32 6.875 / miss 0xdead0000(经 raygen1 写 out)→ 微基准收官
- 注意:interposition 全部无效(libnvoptix 经 cuGetExportTable 私有绑定驱动符号)
- 注意:fault(700/715/716)毒化 CUDA context,任何失败实验都要独立进程 + timeout

## 关键文件(ttu-bench/)
- `probe.cu/probe2.cu/probe3.cu` — 宿主 kernel(直线 FADD 链,patch 区)
- `build_and_patch.py / build_ttu2.py / build_ttu3.py` — SASS patcher
  (eoss-framework 二进制 API + REPLACE_FULL;LDVAR 环境变量选变体)
- `run_exp.py / run_ttu2.py / run_ttu3.py / run_ttu4.py` — ctypes 驱动 API runner
- `run_mk.py` — **(Phase 9)megakernel 直接发射器**(jit_16 + coreParams 合成,见 Phase 9 §4)
- `tmtu_dis.txt / mk_simple_dis.txt / jit_09_travttu_dis.txt / jit_09_dispatcher_dis.txt`
  — 关键反汇编语料(Phase 9 分析入口)
- `optix_min.cu / optix_min_host.cpp` — OptiX 8 最小管线 + AS dump + 进程内 probe
  (argv: ptx [cubin] [layout_idx];fault 毒化 → 每布局一进程)
- `hook_load.c/.so` — interposition 钩子(证实全部被绕过,留档)
- `jit_*.elf` — 20 个 JIT cubin(traversal_ttu_2lh/_mttu/_dmm/_utrav 家族;
  **jit_16/jit_19 = 已链接启动模块,jit_01=CE memcpy 核,jit_12=BVH 构建,jit_17=mk REL 版**)
- `as_dump_[0-2].bin` — Vulkan AS;`optix_as.bin` — OptiX AS(格式对比未做)
- `e7_*.cubin` — 各阶段实验 cubin
- 编码真源:`research/community/eoss-framework`(ttu_sm120.json + JSON/SM86
  Rosetta JSON + tests/test_ttu_data.py golden 序列;import 需 symlink eoss→eoss-framework)

## 已知事实速查
- TTU opcode 基值:OPEN=0x3D0 ST=0x3D1 LD=0x3D2 GO=0x3D3 MF=0x9D4 CCTL=0x3D5
  (跨代一致,SM86=SM120);编码字低 16 位 = 0x7<op 低 3 位>|pred
- 16B/条 = 8B word(LE) + 8B ctrl(LE);TTU addr 字段 @ word bits[55:40]
- R18 = 遍历核参数中的描述符指针;真 OptiX traversable = 我们 cuMemAlloc 缓冲+0xc
  (Phase 9 精化:mttu 管线里 R18:19 = root AS 指针,经 raygen 从 coreParams[0x100] 装入)
- CUDA/Vulkan 页表不共享;cuMemAddressReserve 忽略请求地址(535);
  NCU@SM86 无 rtcore__rays.sum
- 编译:probe 用 nvcc -arch=sm_86 -cubin;host 用 g++ -I OptiX-8.0.0/include
  -I ~/cuda-12.1/include -lcuda -ldl;SDK 在 ~/gpu-engine/OptiX-8.0.0
- GPU 选择:先 nvidia-smi 查空闲(3/4/5/7 常空闲);CUDA_VISIBLE_DEVICES 隔离
- 资料索引:research/INDEX.md;逆向结论:research/notes/ttu-reversing-implications.md
- 下一 session 入口:NEXT_PROMPT.md(Phase 12 任务书)

---

# Phase 8 记录(指令页探针 → 协议重建,2026-11-01 session)

## 执行摘要
按 HANDOFF 四步走了 1-3,发现"定位模块 GPU VA 后 patch"的路线在 535 上**结构性不可达**
(证据链见下),但探针过程把 535 TTU 协议逆向到了前所未有的深度:
**两级 TTU 融合组完整解码 + LD 结果通道打通(t=0x7fffffff=miss 哨兵可读!)+ 状态指针
= 驱动 per-thread 共享内存初始化**。下一步 = 补齐 state/root/ray 格式(路径已明确)。

## Step 1(宿主结构扫描)✓ 完成
- 驱动私有堆 = **0x205c00000,8MB+256B,DtoH 可读**,跨进程基址稳定
- launch config 双缓冲 @ heap+0x3b00 / +0x5400(每次发射填一个):
  +0x30/38/40 = drg/dms/dhg(SBT 指针)、+0x100/0x108 = 用户 params {gas, out} 拷贝、
  +0x50 = 0x7f…221e000(驱动内部)、自引用链 0x205c03e00/0x55c05400 等
- gas 句柄 = dOut+0xc 再次确认;**发射期间堆无 mid-flight 变化**(引擎不写可见堆)

## Step 2(定位 JIT 模块 GPU VA)✗ 结构性不可达(全部角度已穷尽)
1. DtoH 全扫(驱动堆 160MB 窗 + 用户窗口 ±1GB):只有 8MB 堆 + 用户分配,**无模块文本**
2. cuPointerGetAttribute 对驱动内部分配:mt=DEVICE 但 GetAddressRange 谎报;0x7f6647000000
   类 x20 引用值 = PGAttr rc=1(不在 UVA 表)
3. 内核探针(pinned 渐进 + fault 跳页,drive_kprobe.py/kprobe3):0x5200000000 全窗
   2MB/4KB 步长逐页 fault;模块 slab(dverts+0x24000000 类,delta 每进程漂移)页 0-9 fault
4. interposition:libnvoptix **零 cu* 动态导入,只有 dlopen/dlsym** → hook dlsym 成功
   (hook_dlsym.so:wrap 了 cuGetExportTable/ModuleLoad*),拿到 9 张私有 export table
   (xtab_*.bin 已存)——但模块上传**不走任何 cuModuleLoad*/cuLink* 路径**
   (运行时 inline-detour 公开 stub 的真实现,零调用)
5. 上传机制 = ioctl/BAR 直写(strace 无 >4KB write/ioctl);56MB nvidiactl staging 窗
   (0x200400000)只有静态 SBT 头拷贝,发射数据不经过
6. 宿主 JIT 镜像 patch(create 后、launch 前)无效 = **上传发生在 optixPipelineCreate 内部**;
   pipeline#2 复用 JIT 缓存(SIGSEGV-watch 观测到 1 次 cache 读,不重新上传)
7. jit_09 的 .nv.constant2(c[0x3] 函数表)宿主副本 = 未解析偏移对,重定位在 GPU 侧做
8. libnvidia-rtcore.so(119MB,内嵌 NVVM)= 遍历 JIT 的家,SASS 运行时生成,无静态资源可改

## Step 3(in-vivo patch)→ 转向,副产品 = 协议大突破
- patch_capture.cpp:宿主镜像双副本签名定位 ✓(2 hits,上下文验证)、5 种变体编码器 ✓
- watch_capture.cpp:SIGSEGV 页保护 watch 技术可用(fault rip = libc memcpy /
  rtcore+0x3d4575 结构遍历)
- **协议解码新发现(从驱动真组逐字)**:
  * 顶层组 @ .text.utrav_trav_ttu+0xbb0:OPEN, MF(10), ST[0x0000]←R50, ST[0x040]←R48,
    ST[0x050]←R42, ST[0x100]←R2(0x23d1), ST[0x110]←R22(0x23d1), ST[0x200]←RZ(0xa3d1),
    GO, LD[0x300]→R50, LD[0x380]→R22, LD[0x390]→R36 —— 535 的顶层协议比 triangle_fetch
    的 MF(6) 组丰富得多
  * **新 opcode 变体**:0x83d1(ST[0x0000] 用)、0x23d1、0xa3d1(低 16 位高半字节 ≠ 0x7!
    旧"0x7<op>|pred"规则不完整);ctrl byte0 ∈ {0,0xff,0x02,0x18,0x20,0x32,0x40,0x46,…}
    (随槽位变化,逐槽拷贝真值最稳)
  * **MF 组必须连续**:组内 NOP → 715;MF 计数 ≠ 实际 TTU 指令数 → 715/watchdog
  * **TTU 序列后必须保护捕获寄存器**:编译器的地址重算指令(MOV R6,c[0x0][0x170] 等)
    若在 TTU 序列后执行会覆盖 LD 目标寄存器 —— probe5 的做法 = 序列 + 重发地址重算 + NOP 填充
  * LD 结果通道可用:**LD[0x300]→Rd 读回 t 值,0x7fffffff = miss 哨兵**(e7 框架实测)
  * R50 的来源(utrav_trav_ttu 反汇编,TTU 槽 NOP 后可读,/tmp/tt_dis.txt):
    `@!P0 LDS.64 R50, [R3]`,R3 = tid*0x40 + (R0*8 & 0x7f8) —— **状态指针 = 驱动写进
    sharedStack 共享内存的 per-thread 值**(每线程 0x40B 槽),R52/R54/R56 同源(保存/恢复槽)
  * R2 = R52 或 0(@!P2 条件),R48 入参 bit0x10 = 模式标志
- compute-sanitizer = TTU 调试神器:直接给出 fault 地址 + SASS PC(锁定 0x7f9300000008
  = (常规分配 hi32<<32)|8 的可疑写,源于 probe5 自身的寄存器踩踏,修好后剩纯协议问题)
  —— 注意(Phase 9):仅对 nvcc 编译的 probe 核有效;对 cuModuleLoadData 的驱动私有模块
  **无设备侧输出**,改用 cuda-gdb(batch 模式,异常 PC 映射函数+offset)

## Step 4(真描述符跑 probe)≈ 60%
- probe3+run_ttu4+e7_w1 **至今干净可复现**(刚验证):root 扫遍 base/0x180/0x280/0x300/
  0x380/0x400/0xc 全部 t=0x7fffffff(miss)
- **发现 e7-w1 的 ray 打包从没送进 oz**:build_ttu3 的 w1 只 ST 了 (oy,ox)/(dy,dx)/(dz,dz),
  cfg[2](oz 槽)从未被 ST —— origin.z/dir.z 语义 = 引擎内残留 = 全 miss 的最可能原因!
- state 块 = 0x5200000000 窗(配置里 x18 精确引用,4GB 对齐 = 引擎寻址(基|偏移)假设的来源,
  但 run_ttu4 用非对齐 root 也 clean → 引擎用完整 64 位指针,0x52 = 单纯的 VMM 工作区),
  DtoH/内核均不可读 → 状态初值格式只能从驱动 DMA 或静态反汇编拿
- optix_as.bin 全图只有 0xD0 一个绝对指针(已修),AS 位置无关 ✓

## 下一步(优先级序)—— **⚠ 已被 Phase 9 取代:状态栈架构答案见 Phase 9 §2**
1. **★ 状态块格式 = 唯一硬缺口**(已用实验封死其他假设):
   - build_ttu6.py + run_ttu6.py = 槽位映射可变的 e7 框架;ray 打包 8 种 × root 2 种
     全部 t=0x7fffffff → **打包/根指针不是缺的关键**
   - 引擎没有合法的 per-thread 状态栈(ST[0x0000]←R50,驱动经 sharedStack 写入)就
     不启动遍历,t 恒 = inf。R50 = LDS.64 [tid*0x40 + (R0*8 & 0x7f8)](shared)或
     LDL [R47+0x8+…] 本地内存
   - 路线 A(静态):追踪驱动入口核把什么写进 sharedStack —— 反汇编 jit_09 的
     traversal_utrav 分发器找 STS(写共享)指令 + 其寄存器来源 → c[0x0] 参数 = launch
     config 里的字段 → 复刻初始栈帧
   - 路线 B(动态):发射 DMA staging 抓取:heap mid-flight diff 的 +0x5588-0x5660 区有
     5 个 host 指针(0x55ba…)= DMA 描述符候选;读 host 侧这些指针的内容
   - 路线 C:0x5200000000 窗口 = 引擎工作区(VMM),启动后页在引擎页表里但 CPU/内核
     不可读;确认 kprobe3 4KB 细扫全 fault(已做)
2. R48/R42/R2/R22 静态追踪(/tmp/tt_dis.txt,TTU 已 NOP 化,可读)
3. 拿到状态格式后:复刻初始栈帧 + ST[0x0000]=副本 + 正确 ray 打包 → t 翻转 → 微基准完成

## 本轮新增工程资产(全部可复用)
- tools/sec_elf.py + tools/rebuild_cubin.py:任意 JIT ELF 段 → nvdisasm 可读 cubin
- scan_dev.cpp/scan2.cpp(设备内存内容扫描)、scan_maps.cpp(/dev/nvidia 映射扫描)
- kprobe.cu/2/3 + kprobe*_host + drive_kprobe.py:fault 容忍内核 VA 探针(pinned 渐进提交)
- hook_dlsym.so(dlsym 层 hook,wrap cuGetExportTable 成功)、detour.so(运行时 stub
  inline-detour 框架,含 trampoline rip-fix;9 张 export table 全表项透明包装)
- watch_capture.cpp(SIGSEGV 页保护 watch + 处理器内 patch)
- patch_capture.cpp(宿主镜像 patcher,5 变体)、heapdiff.cpp、stage_scan.cpp
- probe5.cu + build_ttu5.py + run_ttu5.py(修正协议 probe,结构问题已修:region 全换 +
  地址重算重发 + MF 只数 TTU;TTU5_E7/CLASSIC/LDSET/DROP/ROOTOFF/STATE0/STATECFG 变体)
- xtab_*.bin(9 张驱动私有 export table dump)、heap_mid/heap_post.bin、linkin/linkout

## 新坑速查
- cuMemcpyDtoH 对未映射 VA = 干净 rc=1,**不毒化**(可放心扫)
- 内核读未映射 VA = 700 毒化(必须进程隔离 + pinned 渐进)
- MF 组内 NOP = 715;MF 计数错 = 715/看门狗;组 = OPEN..最后一条 TTU,尾部 NOP 不算
- compute-sanitizer 跑 TTU 程序:719(unspecified launch failure)= 上层 700 的包装
- cuMemAddressReserve 请求地址仍被忽略(535);但 VMM 8GB 预留内可 4GB 对齐放置
- 驱动堆基址 0x205c00000 跨进程稳定,可直接硬编码扫描

---

# Phase 9 记录(链接模块全解码 + megakernel 直接发射,2026-11-01 session 2)

## 执行摘要
**状态栈问题在架构层已破解**。发现并解码了驱动 535 的完整启动模块 jit_16/jit_19
(= 用户程序 + megakernel + traversal_mttu + sharedStack 的**已链接** cubin):
TTU 输入块在**寄存器里现算**(不是 DMA 写的全局栈!)+ ray 打包在 **shared memory**
(每线程 8B 槽 ×4 阵列)+ megakernel = 16 位状态机 + 函数表。run_mk.py 已能直接
cuModuleLoadData 这个驱动私有模块、打补丁 coreParams、发射 megakernel_simple 并
干净运行(状态机全流程跑通,只剩 .128 spill 的全局化编码矩阵)。

## 1. 模块地图(jit_16 = jit_19 ±32B,同一管线的两个实例)
- `.text.__raygen__rg_0xc7cea55af3e8b61c_ss_0` (512B):**光打包器**。全部常量:
  R10/R11=0.25f, R12=-5.0f(oz!), R13=0(tmin), R14/R15=0(dx,dy), R16=1.0f(dz),
  R17=100.0f(tmax), **R18:19 = c[0x3][0x100] = ROOT**, R20=0, **R38=0xff01**
  (bits[8:15]=mask=0xff, bit0=?), R44=0, R106=0(payload), R21=-1, R23=(R21+1)|(c[0x3][0x36]<<16),
  然后 RET —— **光在寄存器里传给 traversal,不过内存!**
- `__raygen__ss_1` (256B):**输出写回**。R21=-7(→megakernel EXIT);
  `STG.E.STRONG.SM [R22*4 + c[0x3][0x108]], R106`(R22 &= c[0x3][0x70]=CTA id 掩码)
  → **coreParams[0x108] = out 缓冲, [0x100] = root** —— 与 heap config 的 +0x100/+0x108 {gas,out} 一一对应!
- `__miss__`:R21=-3, R106=0xdead0000(魔数!)。`__closesthit__`:R106 = 共享[R40&~7] 读回
  (R41 bit23 时)否则 (R40&0x3fffffff)|0x10000000;R21=-3。
- `megakernel_simple/_sync/_recon` (2048B ×3):**状态机调度器**(见 §3)。
- `traversal_mttu` (0x8880):真正遍历,**TTU 组 @0xe70**。`triangle_fetch_mttu` (0x600):
  TTU 组 @0x470(OPEN,MF,ST[0x000]←R42:43变体c3d1,ST,GO,LD×3)。
- stateFunPtrs(.nv.global.init+0x80,112B=14槽):[0-3]=rtcoreNop*,[4]=traversal_mttu,
  [6]=triangle_fetch_mttu,[10]=miss,[11]=closesthit,[12]=raygen0,[13]=raygen1;
  [5][7][8][9] 无重定位=驱动运行时填(cuModuleLoadData 后=0,已用 slot0=nop 填补)。

## 2. **状态栈答案(mttu 家族,= 真实管线)**
traversal_mttu 序言(0x0-0xe60):解析 root 头(word0 类型位)+ 量化网格(RCP 路径,
R63=t-scale)→ **shared 初始化 @0x940-0xa90**:
```
R27 = tid | ((tid*8)&0x1f)   ← 32 线程的置换槽位
STS.64 [R27.X8+0x000] ← (F2I(R63), 0)     ← 状态槽(slot0)
STS.64 [R27.X8+0x100] ← R10:R11 = (ox,oy)  ← ray.a
STS.64 [R27.X8+0x200] ← R12:R14 = (oz,dx)  ← ray.b
STS.64 [R27.X8+0x300] ← R15:R16 = (dy,dz)  ← ray.c
```
**TTU 输入块 = 寄存器现算 @0xe70 组**:
```
OPEN, MF(18),
ST[0x0000]←R42:43 = ((R38<<8)&0xff0000, R63-R40) = (mask<<16, time_frac) ← 配置字,非指针!
ST[0x0040]←R12:13=(oz,tmin)  ST[0x0050]←R16:17=(dz,tmax)   ← 0x040/0x050
ST[0x0080]←R72:73=(mask<<16,frac) = [0x0000] 的复制        ← 0x080
ST[0x00c0]←(oz,tmin)  ST[0x00d0]←(dz,tmax)                 ← 0x0c0/0x0d0 = 0x040/0x050 复制
ST[0x0100]←R34:35=(0,0)  ST[0x0200]←R34:35=(0,0)           ← 0x100/0x200 = 零!
ST[0x0180]←R56:57  ST[0x190]←R60:61                        ← 外核上下文(未解码!)
GO, LD[0x300]→R72:73(t!)  LD[0x320]→R40:41  LD[0x330]→R2:3
LD[0x380]→R50:51  LD[0x3b0]→R42:43(新状态)  LD[0x3c0]→R54:55  LD[0x3d0]→R58:59
```
**结论:① ST[0x0000] 不是指针 = (mask<<16, timefrac) 配置;② ray 走 shared
(引擎可能经 R56:57 = 外核给的 shared 泛型地址读取 — 待验证);③ 0x100/0x200 = 零,
root 不在 ST 槽里 —— 之前 e7_w1 的"ST[0x100]=root"布局假设全错!**
LD.E.STRONG.SM [R68+4](R68=R72=LD[0x300] 结果)证明**引擎自身可读写 shared**。

## 3. megakernel 状态机(mk_simple 0x800 全解码)
- 序言:LMEM 溢出检查(c[0x0][0x28] vs coreParams[0x28]),网格边界(c[0x3][0x00-0x24]),
  R21 = *(S16*)coreParams[0x38](状态变量,驱动持久化)。
- 循环 .L_x_6:R21∈{-6,-3}→R21=R23&0xffff(R23=raygen 留下的下一状态);
  否则 funptr = *(u64*)(c[0x3][0x58] + R21*8) → CALL(**R4 = mk+0x440 = 重定位的返回地址**,
  被调函数 RET.ABS.NODEC R4 回跳!)。
- raygen 返回 -1 → 0x450 分支 → root/mask 检查 → CALL traversal(**traversal 可重入:
  返回 -1 = "没走完,再调",状态栈在其 shared/局部帧里!**)。
- traversal 返回 ≥-5 → yield 路径(0x5a0-0x680,R40/R41=命中记录,RET 回外核)——
  **mk 是被驱动外核(不在 dump 里)反复调用的切片!**
- nextTbl(c[0x3][0x40]/[0x48])仅当 R38 bit19=1 时用;我们的 0xff01 → 恒走表 A[R20]。

## 4. run_mk.py(新工程资产,直接发射 megakernel)
两阶段 load(先 load→GetGlobal(stateFunPtrs)→unload→重打补丁→再 load,VA 稳定已验证)+
补丁:shared size 0→0x1000(section header sh_size,loader 认!);模块级 .nv.info 的
19 个 MIN/MAX_STACK(00081104/00081204, func, 0)全改 0x400(LOCAL=0 会 700);
mk+0x680 RET→BRA;+0x5b0 yield 块→跳过;+0x3d0 R21&=0xd 钳制;
spill 全局化:R47:R48 ← c64[0x48]、R8:R9 ← c64[0x40](各两条 IMAD.MOV c-bank 读,
coreParams[0x40]=gstack2、[0x48]=gstack+0x18,深度步长 0x8→0x10 保 16B 对齐);
coreParams[0x6c]=0;状态槽 5/7/8/9 填 nop;mk+0x380 R9=0 → NOP(保 R8 高半)。
**开关**:MK_GLOBALIZE=1(默认,实验态:.64 已转全局,715)/ =0(全部 spill NOP ——
干净退出但 traversal 异常返回 -7,out 恒 0);MK_SPLIT128=1 时 .128 也转。
**已验证:模块加载 ✓ 重定位应用 ✓ VA 稳定 ✓ 状态机运行 ✓ raygen→traversal 全流程 ✓**
**剩余 715/716:.128 spill 的全局化 ctrl-size 编码(见 §6)。**

## 5. SASS 手工编码规则(本次全部实测,以后直接用)
- **IMAD.MOV.U32 Rd, RZ, RZ, c[b][off]**:word = 0x7624|(Rd<<16)|(0xff<<24)|(off/4<<32)|(b<<46)
  —— byte2=dst, byte3=A(0xff=RZ), **byte4=off/4, byte5=bank<<6**;C=RZ 在 ctrl 尾字节 0xff。
  同构:reg 源版本 C 在 ctrl 尾字节=Rs(word byte4=0xff)。
- **MOV Rd, c[0x0][off]** = 0x7a02|(Rd<<16)|(off/4<<32)(bank0 隐含)。
- **S2R Rd, SR_LMEMHIOFF** = 0x7919|(Rd<<16), ctrl 0x000e220000003700。**可以读**
  (probe 实测 = 0xfff9c0);唯一坑:Rd 必须在该 kernel 已分配的寄存器数内
  (probe3 = 18 个,R44 超界 → 715)。S2UR(0x79c3)+ IMAD.MOV Rd←URx 也实测可通。
- **S2UR UR4, SR_LMEMHIOFF** = 0x79c3(原 mk@0x10);IMAD.MOV Rd←URx = 0x7e24|(Rd<<16)|(0xff<<24)|(URx<<32)。
- **BRA**:word=(s32(target-pc-0x10)<<32)|0x7947(无条件)/0x0947(@P)/0x8947(@!P),ctrl 0x000fea0003800000。
- **谓词位**:word byte1 bits[6:4]=谓词号(0b111=无),bit7=取反。LOP3 的 LUT 在 ctrl byte6。
- **c[0x3] 读**:ULDC = byte4=off/4, byte5=bank<<6(0xc0=bank3);IMAD.MOV 同 byte4/5。
- **LDG.E.CONSTANT=(0x81,0x73) 系**:ctrl byte1=0x89/0x8b/0x8d = 32/64/128;
  STL/LDL=(0x87,0x73)/(0x83,0x79),ctrl byte1=0x0a/0x0c=64/128 —— **两类 ctrl 尺寸编码不同!**
- **NOP** = word 0x7918, ctrl 0x000fc00000000000(0x1879 在驱动模块里=715!)。
- eoss Rosetta(JSON/SM86):ast/ald = 通用 ST.E/LD.E,opcode=bits[91]+[11:0],size=ctrl bits[75:74]。

## 6. 下一步(优先级)
1. **★ .128 spill 全局化编码矩阵**(run_mk.py 已可一键迭代):候选
   STG.E.128 = (byte0,byte1,ctrl-b1) ∈ {(0x88,0x79,0x1d),(0x87,0x79,0x0c),(0x86,0x79,0x1d),...},
   LDG.E.128 ∈ {(0x81,0x73,0x8d),(0x82,0x73,0x8d),...}。每变体独立进程+timeout;
   715=编码非法,716=对齐(基址 16 对齐需保持,深度步长已 0x8→0x10 加宽),
   对齐后出 700 = 布局错位。**用 e7 方法:每 launch 测一个编码。**
2. 或:.128 保持局部+修 LMEMHIOFF:loader 对驱动格式模块不给 local 窗口
   (LMEMHIOFF=0,nvcc 模块=0xfff9c0) —— 差异在每 kernel .nv.info 格式,
   试把 loctest2 式 per-kernel 条目注入 megakernel 的 .nv.info。
3. megakernel 跑通后:out[0] 应 = f32 6.875(hit)/0xdead0000(miss)→ 收官微基准。
4. R56:57/R60:61(ST[0x180]/0x190 源)= 外核上下文,遍历结果回读时观察。
5. utrav 家族(jit_09,e7_w1 用)同理解码:traversal_utrav 的 ST[0x0000]←R50
   = LDS sharedStack 槽 → 初始值 = 外核写入,格式应与 mttu 的配置字同构。

## 7. 本轮新增资产
- run_mk.py(megakernel 直接发射,全部补丁参数化,MK_* 环境开关;内含
  patch_probe_lmem = SR_LMEMHIOFF/c[0x0][0x28] 探针补丁,替代独立的 probe3_lmem cubin)
- tmtu_dis.txt / mk_simple_dis.txt(mttu 反汇编全文,TTU 组 @0xe70 已标注)
- _text_megakernel_simple.bin / _text_traversal_mttu.bin / _text_rg_ss_0.bin
  (jit_16 关键节原始码;再生:直接 ELF parse,见 run_mk.py 的 sections())
- cuda-gdb 工作流:batch -x cmds --args python3 -u script;异常 PC 映射函数+offset
- compute-sanitizer 对 cuModuleLoadData 的驱动私有模块**不报设备侧信息**(改用 cuda-gdb)

---

# Phase 10 记录(编码真值全集 + 补丁纠错 + 流程插桩 harness,2026-11-02 session)

## 执行摘要
用 nvcc 参考核 + 模块内部指令双向验证,拿到了 **SM86 全套访存/c-bank 指令编码真值**;
据此发现 **Phase 9 的三个补丁 bug**(prologue IMAD 字段错位 / probe 缺 R1 / NOTTU 用了
非法 NOP 0x1879)——旧"715 之谜"的主因。证明 **.128 spill 全局化结构性不可行**
(88 个位点用 12 种基址寄存器,R8:R9 在真管线里根本不是 64 位对)。转向流程本身:
建立 megakernel 插桩 harness,**首次接通 raygen0 → traversal → raygen1 全流程**
(MK_LOCAL 态:原始局部指令零改动,干净退出),**证明 TTU 组在执行**
(NOTTU 差分:NOP→700,真组→干净)。剩余缺口收敛为一条:**traversal 走完但未产出
命中记录(R3=0)**——下一步 = 追 walk 循环的记录产出路径。

## 1. SM86 编码真值(全部经 GPU 实测/模块内已执行指令验证)
访存指令 16B = word(LE) + ctrl(LE):
- **word 低16**:STG.E=0x7986,LDG.E=0x7981,LDG.E.CONSTANT=0x7381,STL=0x7387,
  LDL=0x7983。word byte1 bits[6:4]=谓词号(7=无),bit7=取反(对访存指令同样成立,
  已由 @!P0/@!P2 位点验证)。
- **字段**:b2=dst(载)/0(存);b3=地址对基址 Rb(全局=[Rb:Rb+1],局部=[Rb]);
  b4=数据reg(存)/0x04(普通载)/0x00(.CONSTANT 载);**imm24 = word bits[63:40]**
  (b5-b7,LE,字节单位,负偏移如 -0x10 = f0 ff ff)。
- **ctrl**:模板可整字拷贝。nvcc STG .32/.64/.128 = 0x000fe2000c101904 /
  0x0045e8000c101b04 / 0x004fe8000c101d04;LDG .64/.128 = 0x000ea2000c1e1b00 /
  0x000ea2000c1e1d00。ctrl-b1 尺寸位:local 08/0a/0c,global 19/1b/1d(=local|0x11),
  driver .CONSTANT 89/8b/8d。ctrl-b0:STG=0x04,LDG/STL/LDL=0x00;nvcc global 的
  ctrl b3=0x0c,driver global=0x00,local=0x00(nvcc/driver 两套都合法,勿混搭)。
- **uniform 偏移寻址已被本模块证明**:traversal @0x19b0
  `LDG.E.64.CONSTANT R2, [R2.64+UR4]`(b4=0x04=UR4!)——地址 = (Rb+1:Rb)+(URb+1:URb)
  两个 64 位相加;UR 对经 ULDC.64 装载后永不被标量代码破坏。
- **c[0x3] 读**(b5=off/4,b6=0xc0=bank3,对 ULDC/IMAD/IADD3/LEA 全族一致):
  `ULDC URx, c[3][off]` = 0x7ab9|(URx<<16)|(off/4<<40)|0xc0<<48,ctrl 0x000fe20000000800
  (.64 版 ctrl b2=0x0a,见 @0x1960);`MOV Rd, c[b][off]` = 0x7a02|…(仅 bank0!
  带 bank3 会 715);IMAD 读 c[0x3] 用 @0x1930 模板 word 0x00c01300ff217624 + ctrl
  0x000fca00000e06ff —— **注意该模板是 IMAD.X(带进位),纯加载需消进位或改用 ULDC**。
- `IMAD.MOV Rd, RZ, RZ, imm8` = 0x7424|dst<<16|imm8<<32(imm8 符号扩展,0xff=-1),
  ctrl 0x000fe200078e00ff;`MOV Rd, Rs` = 0x7202|dst<<16|Rs<<32;
  `MOV Rd, imm8` = 0x7802|…;RET(R4 链接) = 0x7950|(R4<<16)=0x0000000000047950。
- **合法 NOP = 0x7918 + ctrl 0x000fc00000000000;0x1879 在驱动模块 = 715!**

## 2. Phase 9 补丁纠错(已全部修入 run_mk.py)
1. **prologue IMAD.MOV 字段错位**:旧字 0x0000c012ff007624 实际解码 =
   `IMAD.MOV R18 ← c[0x0][0x300]`(dst/off 互换 + bank 位丢失)——R47:R48 从未
   被装入,R8:R9 同;这是旧 715 的首要嫌疑。旧 §5 公式 "byte4=off/4, byte5=bank<<6"
   对 IMAD/ULDC 均错(实际 b5=off/4, b6=bank<<6)。
2. **patch_probe_lmem 缺 R1**:STG [R0:R1] 用 R0=R41 对,但 R1 没装高 32 位 → 700。
3. **MK_NOTTU 的 NOP 字 = 0x1879(非法!)**——已改 0x7918;修复后 NOTTU 差分实验
   才成为可能(并立刻证明 TTU 组在执行)。
4. **.nv.info attr id 纠正**:0x1104=FRAME_SIZE(LOCAL attr 来源!)、0x1204=MIN_STACK、
   0x2304=MAX_STACK、0x2f04=REGCOUNT(nvcc 模块级四件套;cuobjdump -elf 实名核对)。
   jit_16 无 MAX_STACK 条目;module 级 0x2c01(fmt0)为驱动专有条目。

## 3. 全局化路线判死(证据链)
- 88 个 STL/LDL 位点基址寄存器 = {R0,R2,R3,R6,R7,R8,R14,R27,R32,R35,R41,R47}
  共 12 种;多数在栈帧反转循环里被 32 位算术循环携带(R6-=0x40 / R3+=0x40)。
- 全局寻址 = [Rb:Rb+1] 对;Rb+1 在位点处是垃圾(R4=RET 链接寄存器不可预载;
  mk @0x380 原指令 = `IMAD.MOV R9, RZ, RZ, RZ` 即 **R9=0**,R8:R9 真管线里不是对)。
- R48 被 @0x420(SHF.R.U64)/@0x750 覆写,R47 唯一稳定但无配对。
- uniform-offset 形式 [Rb.64+URpair] 可绕高半问题,但 Rb+1 垃圾仍进和 → 死。
- **结论:让 .128 保持局部(真 local 窗口)或换裁剪版 traversal 是仅剩路线。**
- **★ LMEMHIOFF 真值已测出 = 0xfff9c0(与 nvcc 模块逐位一致!)** —— 经
  `MK_RG106=lmem`(S2R R106, SR_LMEMHIOFF = 0x7919|0x6a<<16, ctrl
  0x000e220000003700,raygen0 payload 通道)实测。**Phase 9 "loader 对驱动格式
  模块不设 local 窗口" 的结论是错的**(测量 probe 本身有 bug)。局部指令原样
  可用,全局化路线彻底封存;route B = 已成事实的默认状态。

## 4. 流程插桩 harness(run_mk.py 新开关,全部已验证)
- **MK_LOCAL=1**:跳过 tmtu_lmem/mk_r8/globalize 三补丁,原始局部指令全保留。
- **MK_RGR21=n**:raygen0 @0x120 `IMAD.MOV R21,0xff(-1)` → n。**n=4 = 把流程接进
  traversal**(旧默认 -1&0xd=13 直跳 raygen1,traversal 从未运行——旧实验全在此
  打转!)。
- **MK_RGST=reg**:raygen1 的 out-store 数据 reg R106→reg。已验证通道:R40/R72/R46/
  R63/R24/R3/R2/R21。注意 **mk @0x4d0/0x540 会把 R40 写成 -1**(post-call 区),
  R40 通道读不到 traversal 的记录;R21 通道只读得到 -7(raygen1 先改再存)。
- **MK_RG106=imm:N|lmem**:raygen0 @0x110 `MOV R106,RZ` → 标记/S2R LMEMHIOFF。
  imm 通道已验证(out=0x5a:**raygen0→raygen1→out 链路全通**)。
- **MK_RG1CH=1**:raygen1 头部替换为 __closesthit__ 的 7 条 payload 指令(逐字拷贝:
  P0=R41&0x800000;@P0 R2=R3-8、R106=LDG.E.?????.CONSTANT [R2];
  @!P0 R106=(R40&0x3fffffff)|0x10000000)+ 原_store/RET 重排。已验证:@P0 LDG
  真执行(垃圾 R2 → 716),插桩通道本身可用。
- **MK_CLMP=hex**:@0x3d0 夹具掩码可调(默认 d)。
- **MK_PROBE_LMEM**:probe 已修(R0:R1 = c64[0x38] 经 MOV.c 形式 + 自包含 EXIT
  @0xd0)但仍 700 —— c-bank 读模板(@0x1930)是 IMAD.X 带进位,probe 需换 ULDC
  形式;下一轮先跑 MK_RG106=lmem(零 c-bank 依赖)拿 LMEMHIOFF。

## 5. 流程知识(本轮动态插桩结论)
- mk 入口:R21 = *(S16*)coreParams[0x38](=state0=12)→ funptr[12]=raygen0。
- raygen0:打包 ray 到寄存器(R10/R11=0.25,R12=-5,R13=0,R14/R15=0,R16=1,R17=100,
  R18:19=root+0xc,R38=0xff01,R44=0,R106=0),R21 出口 = -1(@0x120,imm8)。
- traversal 入口:@0x70 `LDG.E.128.CONSTANT R64,[R2]`(R2=R18&~0x3f)**覆写
  R64:R67 = AS 头 128bit**——我们的头 = (0x00100003, 1, 0x980, 0)。tag=R18&0x3f=0xc
  ≠0 → @0x130 跳过 0x140-0x250(类型解码死区)→ 0x270;
  **P0=(R66≠0)&&(R67≠0) 读到的是头 qword2=(0x980,0) → R67=0 → P0=0 →
  @0x2e0 @!P0 BRA .L_x_5** —— 整个 tag 解码/量化(R63/R46 标志)被跳过
  (面包屑自洽:R46=0x6000 初值,R63=0)。→ .L_x_5(0x930)→ shared 射线打包
  (0xa40-0xa90 STS ✓)→ @0xab0 walk setup(R24=ox=0.25 ✓ 面包屑)→ walk。
- **TTU 组(@0xe70)确认执行**:MK_NOTTU=1(合法 NOP)→ 下游 700;真组 → 干净。
  t = LD[0x300]→R72:73 在组内产生,但 raygen1 时 R72=-1(寄存器被后续复用)。
- traversal 返回码 C:C&0xd=13 → raygen1(实测全流程干净退出,out=R106=0)。
  C&0xf∈{13,15}(C=-3 或 -1)。**mask 0xd 永远到不了 slot 10(miss)/11(CH)**。
- traversal 遗留寄存器(实测):R3=0,R2=-1,R41=0xff000000 —— **没有记录指针**。
- optix_as.bin 根头 word0 = 0x00100003(bit19=0,bit22=0,bit20=1)。

## 6. 下一步(优先级序)—— LMEMHIOFF=0xfff9c0 后全面更新
1. **★ 追 walk 循环(0xc60-0x8590)的记录产出**:traversal 走完但 R3=0(无记录
   指针)、R40=-1(mk 自己写的,别用它做面包屑)、R41=0xff000000。TTU 组确认
   执行,t 在组内产生(LD[0x300]→R72:73)但寄存器随后被复用。**记录的正体在
   shared memory**(CH 读 shared[R40&~7] —— 但 CH 的实际指令 = LDG.CONSTANT [R2],
   R2=R3-8)。用已验证的面包屑通道(R46/R63/R24/R3 经 MK_RGST)在 walk 内定位
   记录产生/丢失点。注意:局部位点现在确认可用(有窗!),0x1c80-0x2a00 的
   栈帧反转循环会真执行 —— 行为可能与无窗假设下的推演不同。
2. **yield 块(0x5a0-0x680)un-skip 实验**:去掉 mk+0x5b0 的 BRA,让 yield 路径
   原样跑(其 RET 已被 ret→BRA 补丁改为循环回 .L_x_3)。yield 块含 nextTbl 查询
   + R40/R41 消费 —— 可能正是 R3/记录指针的赋值点。配合 MK_CLMP 试 mask 0xf
   (funptr 表 14/15 槽越界问题:两阶段 load 后 HtoD 补写表尾即可!)
3. **CH/miss 槽接入**:funptr 表 [10]/[11] = miss/CH(重定位已指向真实函数)。
   记录一旦产出,用 MK_CLMP 或表改写把 traversal 完成码接到 10/11,再由
   CH/miss 置 R106 → raygen1 写 out = 期望 6.875(hit)/0xdead0000(miss)。
4. AS 格式对比(若 walk 仍不出记录):optix_as.bin 头 = (0x00100003, 1, 0x980, 0),
   头 qword2 高 32 位 = 0 触发 @0x2e0 null-descriptor 分支(真管线同样走这条!
   —— 已用面包屑证明与真路径一致,所以不是当前缺口);as_dump_*.bin(Vulkan)
   /heap_mid.bin(真管线堆)对比,jit_12 构建核再生内部格式。
5. 编码矩阵收官留档:patch_globalize_local(v2)带真值模板 + MK_GLB_SCOPE 开关,
   连同 §3 判死证据链保留;除非特殊需要,不再投入。

## 7. Phase 10 续:.X8 双 shared 窗口的发现(2026-11-02 深夜追加)
**第二轮插桩(shared dump 攻势)的完整结论 —— 本轮最大发现:**

- **cb1 bit7 (0x80) = 第二 shared 窗口选择器**。plain(cb1=0x08/0x0a)= CTA 静态
  shared(0x1000);**.X8(cb1=0x88/0x8a)= 驱动的 "sharedStack" 窗口** —— walk 的
  ray/state 数组(0xa40-0xa90 的 STS)全写在 .X8 窗口,plain-LDS 永远读不到
  (这就是最初 sweep 全零的原因)。
- **.X8 寻址 = 单寄存器 ×8 + 原始字节 imm**:[Rb.X8+imm] → 字节偏移 Rb*8+imm。
  证据链:v14 700(R22=VA>>3 → ×8=VA → 窗口外);v16 回环成功(R2=0x40 → 0x200);
  0xfc → 716(.64 需 8 对齐,0xfc mod 8 = 4!);v23 [R2=0 + imm] 干净执行。
- **.X8 只有 .64 宽度合法**:32-bit .X8(cb1=0x88)= 静默无效(v6/v8 的教训);
  .64 数据/目标对 = 奇偶必须偶对齐(R9:R10、R21:R22 → 715;R2:R3、R22:R23 ✓)。
- **X8 访问需要先有一次 X8-STS**("激活"):无 STS 直接 LDS = 700(v17)或静默
  (v18);STS 后同窗 LDS = 完全正常(v16 out=0x40 ✓)。
- **walk 遗留基址对 R27=R28=0**(MK_RGST=0x1b/0x1c 实测,hit/miss 同)—— 但
  v23 以 (0,0) 基址 + imm 扫 0x0-0x500 = 全零,与 walk 在 0xa40 写入 0.25 矛盾
  —— 未解:要么 walk 的 R27:R28 在 STS 时刻 ≠ raygen1 时刻,要么 .X8 窗口在
  walk 结束后被驱动语义复位。**下轮第一件事**。
- run_mk.py 的 MK_RT 现为 v23 形态(X8 窗口扫描探针,MK_RT=imm);MK_SRD =
  plain-LDS 探针;MK_SHMEM 可调 shared 大小;MK_OUT64 全 64B out 转储。
- **编码补遗**:`MOV R2, R22` 寄存器形式 = 0x7202|dst<<16|src<<32(实测 ✓);
  SHF.R.U64 的高半源寄存器在 **ctrl byte0**(@0x420 实测);IMAD.WIDE R22 =
  R22*R23+c64(R23=4 乘数来自被挤掉的 @0x10 —— R22=0 时无害)。

### 下轮计划(更新)
1. **X8 窗口可见性之谜**:在 walk 内部(0xa90 之后、0x1bf0 之前找死槽/安全点)
   插 X8-STS 面包屑,确认 walk 写入的 X8 地址;对比 raygen1 时刻同址读数。
2. 若 X8 窗口 = per-CTA 大 shared(驱动 loader 分配):MK_SHMEM 加大到 0x8000
   无效(func attr 变了但内容仍零)→ 窗口大小不由 sh_size 控制,由 .X8 语义
   自带;试 .X8-LDS 直接扫 (R27*8 + 0x100*8?) 的更大范围(imm ≤ 0xffff,8B 粒度)。
3. 记录/产出链:CH 的 LDG.CONSTANT [R2](R2=R3-8)读的是 **global**——记录
   最终在 global;追 R3 的赋值点(0x86b0 LDG [R6] 读 funptr 表项?/yield 块)。

---

# Phase 11 记录(组结果直读 + 寄存器中继通道:组输入路径断裂证实,2026-11-02 session 2)

## 执行摘要
E1(分化扫描)→ 无任何 ray 依赖分化;E2 走完一条曲折但决定性的路:**建成"寄存器
中继"读出通道(组后牺牲槽 MOV R62/R63←R72:73 + MK_RGST 直读),测得 TTU 组
@0xe70 的返回 t = R72:R73 = 0xFFFFFFFF_FFFFFFFF(64 位全 -1),hit/miss 完全相同
—— 引擎连 miss 哨兵(0x7fffffff,e7_w1 可得)都不写,在输入/上下文层直接拒绝**。
静态输入(mask/oz/tmin/dz/tmax)全部完好;头号嫌疑 = **直接发射缺外核(driver
调度核)的寄存器上下文:R56:57 / R60:61(ST[0x180]/0x190 "外核上下文",mk 和
traversal 全程零写入!)与 R40(min-t 累加器初值)**。E3(记录链)在组产出真实
t/哨兵之前无意义,Phase 12 = 先修组输入。

## 1. E1 分化扫描(完成,判读:输入路径退化)
- 8 寄存器(R43/R19/R15/R52/R54/R33/R20/R37)× hit/miss,全部无 ray 依赖:
  R15/R20/R37/R54=0,R52/R33=0xffffffff(确定性);**R43/R19 = 逐 run 漂移垃圾
  (0x00007fXX,同 ray 三连跑全不同)→ 对照表新增:R43/R19 不是 walk 输出**。
- 结论:结果端无可读分化 → 与 E2 的 t=-1 互证:组输入层就断了。

## 2. E2 读出攻坚(约 25 发的完整教训链 — 全部记档,防重复踩坑)
按时间序的失败与结论:
1. **X8 邮箱(原计划)→ 读侧从未工作**:v23 MK_RT 形态连"自环"都读不回
   (activate-STS 写 0x200 再读 0x200 = 0)。两种 LDS.64 .X8 ctrl
   (LDG 拼的 0x000ea80000008a00 / 原生 LDS 模板改的 0x0007220000008a00)皆败。
   **§7 的"v23 sweep 全零之谜"一部分 = 读侧自身坏了**,窗口是否复位仍未定。
   模块内不存在任何原生 LDS .X8(只有 4 条 STS.64 .X8)—— 没有可拷的真值模板。
2. **全局直写邮箱(4 槽 0x1000-0x1030)**:raygen1 原生对
   (MOV R23,4 @+0x10 → IMAD.WIDE R22:R23 = R22*4+c64[0x108] @+0x40 →
   STG.E [R22:R23], R106 @+0x50,nvdisasm 已核)逐字拷入 traversal。
   - C=c[0x3][0x108] → **716**(读出乱值 → VA0 或错位;traversal 的 c-bank
     上下文 ≠ raygen1 的 —— 同一编码在 raygen1 完好);
   - C=c[0x3][0x58]/[0x48](traversal 原生读过的槽)→ 干净但**写入点在全部
     自有缓冲(含 16MB AS 窗,MK_SCAN 扫描)之外** —— c-bank 在 traversal 里
     读到的不是我们写的 .nv.constant3 值(疑 per-function bank 映射;未解)。
3. **raygen1 机制对照(MK_RGSTG)→ 一次乌龙 + 一个确证**:+0x80 槽在 RET(+0x60)
   之后 = 死填充,首测无效;改占 +0x60/+0x70 后 **out = R2:R3 = ffffffff00000000
   = Phase 10 记载的 raygen1 遗留值,机制全链验证通过**。
4. **本地内存中继 → raygen1 无 local 窗**:LDL 在 raygen1 里全偏移 700
   (0x0/0x40/0x100/0x200/0x300/0x3f8/0xfffac0);traversal 侧 STL 同地址"干净"
   = **store 不查地址(假干净)**。本地地址模型修正:合法域 = [LMEMHIOFF, 0)
   负偏移域的推测也不成立(raygen1 全灭)—— raygen1 声明帧 = 0,无 local 可用。
5. **715 编码坑(全部实测,补进 §1 真值表)**:
   - STL/LDL 布局:b2=基址(局部=单寄存器),b3=恒 0(非 0 = 715!),b4=数据,
     b5-b7=imm24。与 STG 的 b3=基址不同!
   - **RZ(0xff)作 local 基址 = 715**;用已知零寄存器(R63/R3)。
   - raygen1 寄存器文件顶 = **R106**:LDL.64 dst 对 R106:R107 = 715
     (R107 未分配 —— out-store 推定为 **.32 c-carry 形式**,ctrl cb1=0xa9,
     只碰 R106;out[4:8] 恒 0 = 未写而非 R107=0)。R70 等中间号安全。
   - 操作数换寄存器:base/data 换 = 合法(MK_RGST 同款);**dst 换成
     文件外 = 715**。imm 数值(0x10/0x600)与 ctrl reuse 字节 = 无罪。
6. **★ 成功形态 = 寄存器中继**:0x1020/0x1030(组后 P0 块,运行时必执行、
   原 @!P0 死代码)→ `MOV R62, R72` / `MOV R63, R72`(MOV reg 形式,零寻址),
   raygen1 侧 MK_RGST=0x3e/0x3f 直读。R63 在 0x44e0 会被重写但试验显示存活;
   R62 存活到 0x6230。**收官态:0x1030 源已改 R73(0x49)= R63←R73**(§4;
   R72:R73 双双 = -1,读哪个都是 ffffffff)。

## 3. E2 判读(主线状态更新)
- **t = 0xFFFFFFFF_FFFFFFFF(64 位 -1),hit = miss。** 语义 = 引擎拒绝/中止,
  不是"无命中"(哨兵 0x7fffffff 由 e7_w1 旧协议组实测可得,引擎会写哨兵!)。
- 组的静态输入逐项核验完好:R38=0xff01(→R42=0xff0000 mask)、oz/tmin/dz/tmax
  (R12/13/16/17)在 walk 内只读、R34:35=0、根头路径与真管线一致(Phase 10 §5)。
- **唯一在 we-vs-real 之间无对照的输入:R40(min-t 累加器,初值 = 外核遗留)、
  R56:57 / R60:61(ST[0x180]/0x190,mk+traversal 全程零写入 = 纯外核上下文)。**
  直接发射时它们 = 0/垃圾;真管线里由 driver 调度核设置。疑引擎因上下文无效
  拒绝整组 → 返回 -1。
- R22@raygen1 基线 = 0x71200600(低 24 位 = mk post-call 模式,顶字节漂移);
  R22 中继被 mk 部分覆写 → **R22 不是干净中继通道,R62/R63 是**。

## 4. 本轮新增工程资产(run_mk.py)
- `MK_MAILBOX=1`:组后中继(0x1000-0x1030 四槽;现为 R62←R72、R63←R73;
  ORIG 字节断言保护);=2/=3:对应槽 NOP 二分档。
- `MK_MAILBOXREG`:原 X8/全局邮箱时代的捕获寄存器选择(现为中继模式的注释)。
- `MK_RGSTG=reg`:raygen1 +0x60 第二条 out-store(RET 移 +0x70)—— 机制对照
  通道,可读任意 raygen1 时寄存器到 out[0:4]。
- `MK_RGLDL=off`:raygen1 本地 LDL 读回(仅对有 local 窗的函数有意义,留档)。
- `MK_SCAN=1`:全自有缓冲(含 16MB AS 窗)模式扫描(R24 探针值)。
- raygen0/raygen1 布局已完整测绘:**RET 前无空闲槽**(NOP 全在 RET 后)。

## 5. Phase 12 方向(优先级)
1. **★ 修复组输入 = 找到外核上下文真值**。三条路线:
   a. **in-flight 补丁真管线**(最有信息量):watch_capture.cpp 的 SIGSEGV 页保护
      + handler 内改写 = Phase 8 已建基础设施;在 optixPipelineCreate 的
      JIT 镜像写入流中打中继补丁(R63←R72 + RGST=R63),真管线跑通后 out = 真 t;
      再加 R56/R60 中继 = **直接读出外核上下文真值**。jit_*.elf dump 自宿主堆,
      无磁盘缓存可补(已核实),只能在写入流中截。
   b. **e7 框架协议重放**:e7_w1 是自研核 = 全寄存器可控 + 读出已验证;
      把 mttu 组 19 条(0xe70-0xfa0 槽字节从 jit_16 dump)重放到 e7,
      扫 R56:57/R60:61/R40 候选值,哪个让 t ≠ -1 = 答案。
      注意 e7 无 .X8 窗 —— ray 需走 plain shared(引擎 generic 读)或
      ST 槽全量携带。
   c. libnvoptix 静态分析调度核的 CALL 现场寄存器准备(119MB 无符号,最贵)。
2. 若 a/b 拿到真值 → 回 run_mk:外核上下文初始化(需要槽 —— raygen0 RET 前无
   空位;候选 = 置换一条 @!P2 恢复?P2=0 恒成立(tmax=100 无符号 < 0xF7FFFFFF)
   不可用;或把初始化放进 mk 的空隙 —— 待查)→ t 翻转 → E3 记录链 → E4 收官。
3. 顺手账:R62/R63 中继可随时改源寄存器(0x48/0x49 = R72/R73),任何"组后寄存器"
   现在都可观测 —— 组内其它 LD(R40:41 / R2:3 / R42:43)同理可测,若需要。
