# TTU SASS microbenchmark — 最简 RT Core 指令复现(SM86 / RTX 3090)

**结果:在 GA102 上成功绕过 OptiX,用裸 CUDA kernel 执行了 NVIDIA 隐藏的 TTU(RT Core)SASS 指令,并复现了论文的关键硬件语义。**

## 方法(4 步,全部本机可复现)

1. **编码来源**:eoss-framework artifact(ASPLOS'27 EoSS 论文的官方开源框架)
   - `data/encodings/ttu_sm120.json`:opcode 基值 + golden 融合组序列
   - `JSON/SM86/json_86_with_conditions.json`:NVIDIA Rosetta String 中 SM86 的 TTU 位域定义
     (opcode = bit91+bits[11:0],TTUMACROFUSE.Sb = bits[44:40],TTUST.Rb = bits[39:32],
      TTULD.Rd = bits[23:16],SR_TTU_TICKET_INFO = SR133)
   - 三方交叉验证:opcode 基值(0x3D0-0x3D5)与论文 Figure 4 的 raw hex(…73d0/…73d1/…79d4)吻合
2. **编译宿主**:`probe.cu` — 直线型 16×FADD 链,`nvcc -arch=sm_86 -cubin`
3. **SASS patch**:定位 `.text.ttu_probe` 文件偏移,把中段连续指令用 16 字节编码
   (8B word + 8B control,LE)替换,`PatchPolicy.REPLACE_FULL` 保真控制字
   - 验证:cuobjdump 显示 TTU 为空白助记符 + raw hex,但接受二进制(论文 Fig.4 现象复现)
4. **运行**:ctypes 驱动 API,单 warp(会话状态 per-warp),空闲 GPU + timeout

## 实验结果(SM86, 535.309.01)

| 实验 | 注入序列 | 结果 | 与论文(SM120)对照 |
|---|---|---|---|
| probe | 无(基线) | OK | — |
| exp1_open1 | TTUOPEN | **OK,无错误** | "completes without error" ✓ |
| exp2_open2 | TTUOPEN, TTUOPEN | **FAULT rc=715 illegal instruction**(可复现 ×3) | "faults upon a second invocation → per-warp session state" ✓ |
| exp3_mf0 | TTUMACROFUSE(0) | OK | ✓ |
| exp4_cctl | TTUCCTL(IVALL=0) | OK | ✓ |
| exp5_group | 完整融合组 OPEN→MF(8)→4×ST→GO→2×LD→LD.CLOSE | **OK 端到端** | 论文需真 BVH;我们 garbage 配置下也完成(说明 TTU 对无效根容错或未解引用) |

> exp2 的条件性 fault 是硬件语义的判别性证据:nop 不可能"因会话已激活"而 fault。
> exp2 fault 后 context 被毒化(后续操作持续 715),属 CUDA 正常行为,GPU 无 Xid 残留。

## 局限与下一阶段

- NCU 在 SM86 上**不暴露** `rtcore__rays.sum`(SM120-only)→ 计数器级验证不可行;
  只有 stall 类指标区分 "non-RTCORE" MIO pipe
- exp5 的遍历没有有效 BVH → 尚未证明 ray 真的被处理。下一步:
  1. 按论文格式造最小 BVH(子指针=绝对虚拟地址,需 cuMemCreate 分配在低 4GB VA)
  2. 用 TTUST[0x100] 设根节点、[0x40]/[0x50] 设 ray,GO 后 TTULD[0x300] 读回
     (Rd=bits[23:16] 捕获结果寄存器,存到 global memory 验证 hit/miss + t)
  3. 命中/未命中随 ray 方向变化 = 端到端功能证明
- 已知安全边界:TTUMACROFUSE 非零 count 错配 → watchdog(~6s,可恢复);
  漏发 TTULD.CLOSE → 永久挂起(勿试,除非 GPU 可重置)

## 文件

- `probe.cu` — 宿主 kernel(直线 FADD 链)
- `build_and_patch.py` — 编译 + patch 生成 exp1-5
- `run_exp.py` — ctypes 驱动 API runner(单 warp,含 fault 恢复探测)
- 依赖:`research/community/eoss-framework`(已建 `eoss` 符号链接供 import)

---

# Phase 2-4 记录(真 BVH + 遍历配置逆向,进行中)

## 已确立的事实(SM86 实测)

### 真 NVIDIA AS 格式(vk_bvh_dump.c,专有 Vulkan 驱动构建 2-tri BLAS)
- BLAS 2688B;头部 0x180B(0x48 处 = 0x180 = 节点流偏移;0x60 处 = 图元数)
- 节点流 +0x180:24-bit 量化 box 边界,哨兵 0x47E87FFF(量化 +inf)
- 顶点重编码副本在 ~0x480
- **全 AS 只有一个地址依赖字段:u64 @0xD0 = AS base 自指针**(三个不同 VA 的 BLAS diff 只差这 1 字节)→ 节点全是相对引用,重定位 = 拷贝 + 改 0xD0
- 专有驱动的 Vulkan AS 天然分配在低 4GB(0x40e1000)

### TTU 指令编码修正(实测)
- TTUST:源寄存器 Rb @ 编码字 bits[39:32],**64-bit 存储 = 隐式对 Rb:Rb+1**;TTU 地址 @ bits[55:40]
- TTULD:目标寄存器 @ bits[39:32](Rosetta JSON 的 "Rd 23_16" 是错的——置非零 → illegal instruction)
- **ctrl byte0(编码字后的第一个控制字节)必须为 0**(JSON 的 Rc/Rd2@71_64 按 byte0 写入 → fault)
- LD addr=0x300(论文口径):LD 自身尝试真实内存访问;LD addr=0:干净但无写入
- MF 融合计数 = 组内指令数(复核)

### 行为
- GO + LD[0x300] 且 config[0x000]=0 → **illegal memory access (700)**:遍历消费 [0x000]
- ST[0x000]=合法指针 → 700 消失(遍历正常运行),但结果尚未出现在 scratch/寄存器
- 无 GO 时 LD[0x300] 干净 → 700 确实来自 GO 遍历的访存

### 环境限制
- CUDA 与 Vulkan 页表**不共享**(同进程跨 API 直读 Vulkan VA → CUDA_ERROR_INVALID_VALUE)
- cuMemAddressReserve 在 535 驱动上**忽略请求地址**(低 4GB VA 不可预留)→ 论文的 low-4GB 技巧不能直接复刻
- NCU 无 rtcore__rays.sum(SM120 only);仅有 non-RTCORE stall 指标

## 下一步(按优先级)
1. **in-vivo 配置拦截**(EoSS 的方法,最可靠):拿 OptiX SDK 头文件 → 跑真 optixTriangle → 用本工程 patcher 改 JIT cubin 里的 TTUST 操作数,把 4 个配置 qword 存到 global memory 读出 → 得到 100% 正确的完整配置模板 → 照抄后替换 root/ray 字段
2. CUDA-GL interop:GL 缓冲区可能落在低 4GB 且 UVA 同址 → 提供低 4GB 的 CUDA 可见内存(放 AS 拷贝)
3. 继续盲测配置语义:[0x000] 的指针/描述符结构、[0x048] 合法性、origin z / dir z 的归属、cull mask/flags 的编码

## 文件
- `vk_bvh_dump.c` / `as_dump_[0-2].bin` — Vulkan BLAS 构建 + dump(+ CUDA 页表探测)
- `probe3.cu` / `build_ttu3.py` / `run_ttu3.py` / `run_ttu4.py` — phase 2-4 实验框架(LDVAR 环境变量切变体)
- `run_ttu2.py` / `e6_*` `s1_golden` `v1-v7` — 编码二分过程(已固化为结论,留档)

---

# Phase 5 记录(OptiX in-vivo + 真 535 协议解码)

## 里程碑
1. **最小 OptiX 8 管线全通**:`optix_min.cu` + `optix_min_host.cpp`,1-tri GAS + raygen/miss/CH,
   `optixLaunch` 成功,`out[0]=0x10000000 (HIT, primID=0)` —— 真硬件光追在本进程跑通
2. **JIT 模块捕获**:堆内存扫描(签名 = opcode 字节 + ctrl 约束)dump 出 20 个 CUDA cubin,
   找到驱动内部遍历核:`traversal_ttu_2lh / _2lh_tri / _dmm / _mttu / _utrav / utrav_trav_bottom_level_instance`
   (LD_PRELOAD 钩 cuModuleLoadData 不适用——驱动内部调用绕过 PLT interposition)

## 535 驱动真协议(从 traversal_utrav 的 MF(6) 融合组逐字解码)
```
OPEN                      word=0x00000000000073d0 ctrl=0x000fec0000000000
MF(6)                     word=0x00000600000079d4 ctrl=0x000fe80000000000   ← ctrl 与我们模板不同(0xe8)
  ST[0x0000] ← R18        word=0x00000012000083d1 ctrl=0x0005e200000000ff   ← pred 半字节=0x8;byte0=0xff
  ST[0x0280] ← RZ         word=0x000280ff000073d1 ctrl=0x0005e20000000002
  GO                      word=0x00000000000073d3 ctrl=0x000fe20000000000
  LD[0x340] → R8,  Rd2=10 word=0x00034000000873d2 ctrl=0x000ee200000e000a
  LD[0x350] → R12, Rd2=14 word=0x00035000000c73d2 ctrl=0x000ee200000e000e
  LD[0x360] → R16, Rd2=ff word=0x00036000001073d2 ctrl=0x000ee400000e04ff
(无 CLOSE —— 会话留给 libcuda 上下文保存路径收尾,SR133 机制)
```
**与 EoSS SM120 协议的结构性差异**:没有 8×TTUST 配置序列;config/光线不走 TTUST!

## 新发现的硬件规则(实测)
- **TTULD 目标寄存器必须 4-对齐**:rd=8/12 ✓;rd=5/13/14 → illegal instruction(无条件、与 rd2 无关)
- LD 按 (Rd:?, Rd2:?) 写 16 字节(addr 步长 0x10 吻合)
- ctrl byte0 ∈ {0, 0xff, 0x02} 合法(驱动在用);byte0=5/6 非法
- LD addr=0x340/0x350/0x360 为结果块地址;LD addr=0 → 完全无操作

## 行为结论
- 用驱动模板 + LD(8/12) 执行**完全干净**,R8/R12 被 LD 写 0(LD 确实在写)
- hit/miss 尚无区分 → **ST[0x0000] 存的不是 AS 根,而是指向设备侧"光线+配置描述符"的指针**
  (证据:此前的 700 fault = 描述符指针为 0 时 GO 的访存;ST000=合法指针 → 700 消失)
- 描述符内容不在主机内存(已扫描 rw 段,GAS 值/origin 浮点对均无)→ 在设备侧参数区

## 下一轮(明确路径)
in-vivo 指令页探针(EoSS 同款):真 OptiX trace 运行中,遍历核的参数 = R18 值(描述符指针)。
JIT cubin 在 GPU 上的 VA 可从 OptixPipeline/OptiX 内部结构挖出;然后同 context 发一个探针 kernel
读该 VA 处的参数与描述符内存 → 得到完整描述符格式 → 用我们自己的 probe 复刻。
备选:hook cuLaunchKernelEx 失败(驱动内部调用绕过 interposition),需走指令页路线。

## 文件
- `optix_min.cu` / `optix_min_host.cpp` — OptiX 8 最小管线 + 内存扫描/描述符搜索
- `hook_load.c` / `.so` — LD_PRELOAD 钩子(LoadData ✓ 有效、LaunchKernel ✗ 被绕过)
- `jit_*.elf` — dump 的 20 个 JIT cubin(含 5+ 驱动遍历核)
- `jit_08.sass` 等 — 反汇编存档

## Phase 6(协议版本澄清 + 描述符 oracle 建立)

- **版本对应**:EoSS 协议 = 新驱动(≥570)/SM120;我们的 = 旧驱动 535/SM86。两者都是"真"协议,
  535 的内存描述符协议此前无人记录(新增 delta)
- 535 协议中 ST[0x0000] 存的是**设备侧描述符指针**(光线+配置+root 的结构体),TTU 通过 SM MMU
  自行读取。低 4GB 约束疑似描述符内部引用的格式约束,而非指针本身(64-bit 指针存储 byte0=0xff 可行)
- **cuGetProcAddress interposition 被绕过**:libnvoptix 经 cuGetExportTable 私有通道绑定驱动符号,
  cuModuleLoadData/cuLaunchKernel(Ex) 钩子全部不触发(经验记录)
- OptiX traversable handle = 我们自己的 cuMemAlloc 缓冲 + 0xc —— **OptiX 生成的 AS 数据就在我们
  分配的 buffer 里,可直接从主机 dump**(OptiX AS 格式 vs Vulkan AS 格式对比待做)
- 描述符 oracle 建立:g 变体(ST[0x0000]=自构描述符 64-bit 指针)→ 700 illegal memory access =
  引擎确实解引用了我们的指针。剩余 = 描述符布局搜索(有 oracle:700/clean/结果差异)

## 文件更新
- `hook_load.c` — cuGetProcAddress/LoadData/launch 钩子(模块钩子无效,留档)
- `build_ttu3.py` LDVAR=g / `run_ttu3.py` — 描述符 oracle 实验

## Phase 7(描述符布局扫描 + 瓶颈定位)

- `optix_min_host.cpp` 重写为干净版本:OptiX 管线 + OptiX AS dump(`optix_as.bin`, 2432B,
  与 `as_dump_0.bin` 的 Vulkan AS 格式对比待做)+ 进程内 TTU probe(布局 oracle)
- **布局扫描结果:8 种布局全部 700**(L1/L2/L3/L4/L6/L7/L1b/L8,root 分别试了
  traversable handle 和 AS base)
- 主导假设:引擎只读指针的低 32 位(R18 单寄存器语义),我们 64-bit desc_va 被截断 →
  无论布局如何都 700。验证方法 = 拿到真描述符指针(R18 运行时值)
- interposition 路线全部关闭(libnvoptix 经 cuGetExportTable 私有绑定)
- **剩余唯一可靠路径 = EoSS 指令页技术**:
  1. 从 OptixPipeline/OptiXModule 宿主结构挖出 JIT 遍历模块的 GPU VA
     (扫描 pipeline handle 周边宿主内存的设备指针,经 cuPointerGetAttribute 验证,
     探针 kernel 读页内容匹配 TTU opcode 签名)
  2. 同 context 探针 kernel 读该 VA → 定位遍历核的参数区
  3. in-vivo patch 遍历核的 `ST[0x0000]←R18` 为普通 STG,把 R18(真描述符指针)写进
     我们可控的 buffer → 读出真描述符内存 → 完整格式
  4. 或者直接:用真描述符指针跑我们的 probe(改 ray 字段)→ hit/miss 翻转 = 完成

## 本轮新增经验(编译/流程坑,全部已踩平)
- cuModuleLoadData 传的是映像内容不是路径
- CUlaunchConfig 字段名 gridDimX(非 gridSizeX);cuGetProcAddress_v2 在 12.1 = 5 参
  (cuuint64_t flags + CUdriverProcAddressQueryResult*);cuda.h 的 #define 需 #undef 才能
  同时定义 4 参/5 参两个钩子
- fault 毒化 context → 布局扫描必须每布局独立进程

---

# Phase 8 记录(指令页探针 → 协议大突破,2026-11-01)

指令页探针四步走了 1-3:"定位模块 GPU VA 后 patch"在 535 上结构性不可达
(DtoH 全扫/内核探针/interposition/宿主镜像 patch 全部穷尽,证据链见 HANDOFF Phase 8),
但副产品把 535 协议逆向到前所未有的深度:

- 驱动私有堆 = 0x205c00000(8MB,DtoH 可读);launch config 双缓冲
- 顶层 TTU 组逐字解码(OPEN, MF(10), ST×6, GO, LD×3)+ 新 opcode 变体(0x83d1/0x23d1/0xa3d1)
- 硬件规则:MF 组必须连续(组内 NOP=715)、MF 计数错=715/watchdog、
  TTULD 目标寄存器 4-对齐、TTU 序列后必须重发地址重算
- **LD 结果通道打通**:LD[0x300] 读回 t,0x7fffffff = miss 哨兵(e7_w1 工作对)
- 状态指针 = 驱动写进 sharedStack 共享内存的 per-thread 值(当时理解,Phase 9 修正)

文件:`optix_min_host.cpp / kprobe*.cu / hook_dlsym.c / detour.c / watch_capture.cpp /
patch_capture.cpp / heapdiff.cpp / probe5.cu + build_ttu5.py / build_ttu6.py` 等
(全部清单与负结果见 INDEX §2)。

---

# Phase 9 记录(启动模块全解码 + megakernel 直接发射,2026-11-01)

## 核心答案:状态栈不是 DMA 全局缓冲

在 dump 的 20 个 JIT cubin 里找到**已链接的完整启动模块 jit_16/jit_19**
(用户 raygen/miss/CH + megakernel×3 + traversal_mttu + sharedStack 定义),
配合 coreParams 常量组与 stateFunPtrs 函数表,把 535 的真实启动路径全部解码:

1. **光在寄存器里**:raygen(512B,全常量)返回 R10-R20
   (origin/dir/tmin/tmax + root=coreParams[0x100] + mask=0xff01),megakernel
   直接 CALL traversal —— ray 不过内存。
2. **TTU 输入块 = 遍历序言寄存器现算**(traversal_mttu @0xe70,MF(18) 逐字):
   `ST[0x0000]=(mask<<16, time_frac)` 是配置字**不是指针**;`0x040=(oz,tmin)`;
   `0x050=(dz,tmax)`;`0x100/0x200=0`(root 不在 ST 槽!Phase 8 前的布局假设作废);
   ray 走 shared memory(每线程 8B 槽 ×4 阵列,置换槽位)。
3. **megakernel = 16 位状态机**:funptr 表在 global memory(stateFunPtrs,14 槽),
   raygen→-1→traversal(可重入,-1=未完成)→miss/CH→raygen1 写 out→-7→EXIT;
   ≥-5 = yield 回驱动外核(mk 是被反复调用的切片)。
4. **coreParams = launch config 的 GPU 侧镜像**:[0x100]=root、[0x108]=out ——
   与 heap_mid 的 config +0x100/+0x108 {gas,out} 一一对应(Phase 8 观察的闭合)。

## megakernel 直接发射(run_mk.py)

驱动私有 cubin 可以直接 `cuModuleLoadData`:重定位(.nv.global.init 函数表)自动应用,
shared 大小认 section header;local 窗口(Phase 9 以为要靠 MIN/MAX_STACK 改出 ——
Phase 10 修正:本来就有,LMEMHIOFF=0xfff9c0)。
配套:两阶段 load(VA 稳定)、coreParams 合成(状态变量/函数表/root/out/栈基)、
全套 SASS 手工补丁(状态机返回地址、yield 跳过、状态号钳制、spill 全局化)。
**状态机全流程已跑通**(raygen→traversal→干净退出)。当时的"剩余 = .128 spill
全局化"判断已被 Phase 10 推翻(全局化判死、local 窗本来就有,无需全局化);
当前真实卡点见下方 Phase 10/11 段落。

## Phase 10 状态(2026-11-02)与主线卡点

- raygen0→traversal→raygen1 **全流程首次接通**(MK_LOCAL=1 MK_RGR21=4,干净退出)
- 编码真值全集(§1)+ Phase 9 三 bug 修复;LMEMHIOFF=0xfff9c0(local 窗本来就有)
- .128 全局化判死;.X8 双 shared 窗口发现(HANDOFF §7)
- **主线卡点(Phase 11 要解的)**:traversal 对 hit/miss 无任何可观测分化、
  R3=0(无记录指针)、CH/miss 槽从未被调用 → out 恒 0。
  ⚠ "TTU 组在执行"只证明结果被消费,不证明结果正确(NOTTU 差分的局限)。
  下一步 = HANDOFF/NEXT_PROMPT Phase 11 的 E1(分化扫描)→ E2(TTU 结果直读)
  → E3(记录链修复)→ E4(out 出 6.875/0xdead0000)

## Phase 11 状态(2026-11-02)与主线卡点(更新)

- **组结果直读完成**:寄存器中继通道(组后 MOV R62/R63←R72:73 + MK_RGST)建成,
  测得 TTU 组返回 **t = 0xFFFFFFFF_FFFFFFFF(64 位 -1),hit/miss 相同** ——
  引擎在输入/上下文层拒绝,连 miss 哨兵都不写(e7 旧协议组可得哨兵,判据可信)。
- 静态输入(mask/oz/tmin/dz/tmax)逐项核验完好;**缺口 = 外核上下文寄存器
  R56:57/R60:61(ST[0x180]/0x190,mk/traversal 全程零写入)+ R40 初值** ——
  直接发射缺 driver 调度核的寄存器现场。
- 大量读出机制真值入档:X8-LDS 读侧从未工作(§7 全零之谜部分破解)、
  raygen1 无 local 窗/寄存器文件顶 R106/.32 c-carry out-store、STL b3 必须 0、
  RZ 不能作 local 基址、store 不 fault 而 load fault(见 HANDOFF Phase 11)。
- **主线卡点(Phase 12 要解的)**:外核上下文真值获取(in-flight 补丁真管线 /
  e7 框架协议重放 / libnvoptix 静态分析三选一)→ 组输入修复 → t 翻转 →
  E3 记录链(yield/CH/miss)→ E4(out = 6.875/0xdead0000)。
- 工具:MK_MAILBOX(组后中继)/ MK_RGSTG(raygen1 第二 out-store)/
  MK_RGLDL / MK_SCAN(run_mk.py;详 HANDOFF Phase 11 §4)。

## 意义(对微基准与争抢研究)

- TTU 微基准**不需要复刻驱动栈**:寄存器 ABI + shared 布局 + 输入块布局即可(全部已解码)
- funptr 表在 global memory 可运行时改写 → 状态函数可替换为任意代码片段,
  TTU/CUDA-core 混合负载可以完全绕过 OptiX 构造
- 驱动的 JIT 产物本身是可重放资产(无需 OptiX 运行时)

## 文件(Phase 8-9 新增)

- `run_mk.py` — megakernel 直接发射器(MK_* 开关;HANDOFF Phase 9 §4)
- `tmtu_dis.txt` / `mk_simple_dis.txt` / `_text_*.bin` — mttu/megakernel 反汇编语料
- `build_ttu5.py / build_ttu6.py / probe5.cu / run_ttu5.py / run_ttu6.py` — Phase 8 协议 probe
- 完整协议/编码规则 = `HANDOFF.md` Phase 9 §5;工具索引 = `INDEX.md`
