# INDEX — ttu-bench 工具与资产目录(Phase 11 后)

实验叙事读 `README.md`(Phase 1-9)与 `HANDOFF.md`(Phase 8-11 详情 + 下一步;
**最新 = Phase 11**,Phase 8/9/10 的"下一步"已被取代)。
本文件 = 工具脚本索引 + 数据资产清单 + 再生方法。

## 0. 立即可复现的运行(均已验证)

```
# 工作对:TTU 序列执行 + LD 结果捕获(端到端干净)
CUDA_VISIBLE_DEVICES=<空闲GPU> python3 run_ttu4.py e7_w1.cubin optix_as.bin hit base
# 期望: [hit root=base] q0=0x000000007fffffff (t=+inf=miss哨兵) q1=0x3f800000

# (Phase 10)megakernel 全流程:raygen0 -> traversal -> raygen1(原始局部指令,干净)
CUDA_VISIBLE_DEVICES=<空闲GPU> MK_LOCAL=1 MK_RGR21=4 python3 run_mk.py hit 12

# (Phase 11)TTU 组结果直读(寄存器中继):组返回 t = R72:R73 = -1(拒绝码,双 ray 同)
CUDA_VISIBLE_DEVICES=<空闲GPU> MK_MAILBOX=1 MK_RGST=0x3f MK_LOCAL=1 MK_RGR21=4 python3 run_mk.py hit 12
# 期望: out[0..8]=ffffffff00000000(= R73 经 R63 中继;0xffffffff = 引擎拒绝码)
# 插桩读寄存器(经 raygen1 out 通道):MK_RGST=0x2e(R46)/0x3f(R63)/0x18(R24)...
# raygen0 payload 标记/读 LMEMHIOFF:MK_RG106=imm:0x5a / MK_RG106=lmem
# raygen1 变 closesthit 记录读取:MK_RG1CH=1;分发夹具掩码:MK_CLMP=d|4|f
# TTU 组 NOP 对分(合法 NOP):MK_NOTTU=1 —— 证明组在执行的差分实验
# 详见 HANDOFF Phase 10(编码真值全集 §1、补丁纠错 §2、流程知识 §5、下一步 §6)
```

## 1. 核心资产(不可再生,勿删)

| 文件 | 内容 |
|---|---|
| `jit_00..20.elf` | 20 个驱动 JIT cubin(dump 自宿主堆)。**jit_16/jit_19 = 已链接启动模块**(raygen+miss+CH+megakernel×3+traversal_mttu+sharedStack,两者差 32B);jit_09/jit_11 = utrav 遍历家族(顶层组 `.text.utrav_trav_ttu+0xbb0`,未链接);jit_07/08 = mttu 家族 REL 版;jit_01 = CE memcpy 核;jit_12 = BVH 构建核;jit_13 = 用户程序 REL;jit_17 = megakernel REL 版 |
| `jit_09_travttu_dis.txt` | utrav_trav_ttu 全量反汇编(TTU 槽已 NOP 化)——utrav 家族入口 |
| `jit_09_dispatcher_dis.txt` | traversal_utrav 反汇编(注意:c[0x0][0x0] 字样 = type-0x40 重定位位,真实 = c[0x3][coreParams+N]) |
| `tmtu_dis.txt` | **traversal_mttu 全量反汇编(34KB,Phase 9 主语料;TTU 组 @0xe70 已逐字解码)** |
| `mk_simple_dis.txt` | megakernel_simple 全反汇编(状态机调度器) |
| `_text_megakernel_simple.bin` / `_text_traversal_mttu.bin` / `_text_rg_ss_0.bin` | jit_16 关键节原始码(补丁脚本的输入;再生 = 直接 ELF parse,见 run_mk.py 的 sections()) |
| `optix_as.bin` | OptiX 构建的 1-tri AS(2432B,位置无关,唯一绝对指针 @0xD0=base 自指针,用前必须 patch) |
| `as_dump_0/1/2.bin`, `as_dump.bin` | Vulkan 专有驱动 AS(Phase 2 格式对照) |
| `heap_mid.bin` / `heap_post.bin` | 驱动 8MB 堆(基址 0x205c00000)mid-flight/post-launch 快照:launch config 结构(+0x3b00 双缓冲)、SBT/params 拷贝、DMA 描述符候选区(+0x5588 的 5 个 host 指针) |
| `xtab_*.bin` ×9 | 驱动私有 export table 全量 dump(GUID 命名;槽位→符号映射待做) |
| `e7_w1.cubin` / `probe3.cubin` | 已验证工作对的二进制(probe3.cubin 兼作 rebuild_cubin.py 模板) |

## 2. 工具脚本(按用途)

### 编码/SASS 构建
| 脚本 | 用途 | 再生 |
|---|---|---|
| `build_and_patch.py` | Phase 1:exp1-5(OPEN/MF/CCTL/融合组验证) | `python3 build_and_patch.py`(产物 exp*.cubin) |
| `build_ttu2.py` | Phase 2:编码二分(v1-v7, e6 系列) | LDVAR 环境变量 |
| `build_ttu3.py` | Phase 2-4:e7 系列 probe(工作对 w1 的来源) | `LDVAR=w1 python3 build_ttu3.py` → e7_w1.cubin |
| `build_ttu5.py` | Phase 8:修正 535 协议 probe(顶层组 13 槽 + 地址重算重发)。变体:TTU5_E7/CLASSIC/LDSET/DROP/ROOTOFF/STATE0/STATECFG | `TTU5_TAG=x … python3 build_ttu5.py` |
| `build_ttu6.py` | Phase 8:ray 槽位映射扫描器(TTU6_MAP="hi,lo:hi,lo:hi,lo") | `TTU6_TAG=x TTU6_MAP=… python3 build_ttu6.py` |
| `tools/sec_elf.py` | 从任意 JIT ELF 抽 `.text.*` 段 → 裸代码 | `python3 tools/sec_elf.py jit_09.elf <段名> out.bin`(注:对 EM_CUDA 的 off 解析有偏差,关键节用 run_mk.py 里的 sections() 直读更稳) |
| `tools/rebuild_cubin.py` | 用 probe3.cubin 当模板 + 外部代码 → nvdisasm 可读 cubin | `python3 tools/rebuild_cubin.py probe3.cubin code.bin out.cubin` |

### 运行器(ctypes 驱动 API;每变体独立进程 + timeout!)
| 脚本 | 配对 |
|---|---|
| `run_exp.py` | exp1-5(Phase 1) |
| `run_ttu2.py` / `run_ttu3.py` | 编码二分 / 描述符 oracle(LDVAR) |
| `run_ttu4.py` | **工作对**;VMM 16MB + AS 自指针 patch;`rootmode` = base/node/任意偏移。注意:运行会写 `vmm_dump.bin`(可删) |
| `run_ttu5.py` | probe5(VMM 8GB 预留 + 4GB 对齐窗实验) |
| `run_ttu6.py` | probe6(槽位映射扫描) |
| `run_mk.py` | **(Phase 9/10)megakernel 直接发射**:cuModuleLoadData(jit_16)+ coreParams 合成 + 两阶段 load + 全套 SASS 补丁。**流程/插桩开关(Phase 10)**:`MK_LOCAL=1`(原始局部指令,主工作态)、`MK_RGR21=4`(接通 traversal)、`MK_RGST=reg`(raygen1 转储任意寄存器)、`MK_RG106=imm:N\|lmem`(raygen0 payload/读 LMEMHIOFF)、`MK_RG1CH=1`(raygen1 变 CH 记录读取)、`MK_CLMP`(分发夹具掩码)、`MK_SRD`(plain-LDS 探针)、`MK_RT`(X8 窗口探针)、`MK_SHMEM`/`MK_OUT64`;补丁态开关:`MK_GLOBALIZE`+`MK_GLB_SCOPE`(已判死留档)、`MK_NOTTU=1`、`MK_PROBE_LMEM`、`MK_ZEROROOT`;(Phase 11)`MK_MAILBOX=1`(组后寄存器中继 R62←R72/R63←R73)、`MK_RGSTG`(raygen1 第二 out-store)、`MK_RGLDL`(本地 LDL 读回,raygen1 无 local 窗留档)、`MK_SCAN`。主工作态 = `MK_LOCAL=1 MK_RGR21=4`(干净退出) |

### probe 内核
| 文件 | 说明 |
|---|---|
| `probe.cu / probe2.cu` | Phase 1-2 宿主核(直线 FADD) |
| `probe3.cu` | Phase 3+:cfg 5×64bit 载入 + 不定分支保活 → LD 结果从 store 源寄存器流出(**工作对内核**) |
| `probe5.cu` | Phase 8:6×64bit cfg,配 build_ttu5 的 13 槽协议序列 |

### OptiX 真管线
| 文件 | 说明 |
|---|---|
| `optix_min.cu` / `optix_min.ptx` | 1-tri GAS + raygen/miss/CH(产生 optix_as.bin;jit_16 里的同名函数 = 它们的 535 JIT 产物) |
| `optix_min_host.cpp` | 管线 + 进程内 TTU probe(Phase 5-7 布局 oracle) |

### 扫描/侦察(Phase 8)
| 文件 | 用途 | 关键结论 |
|---|---|---|
| `scan_dev.cpp` | PGAttr 驱动堆发现 + 内容 needle 扫 | 堆=0x205c00000;0x7f6647000000 类引用非 UVA |
| `scan2.cpp` | DtoH 无畏全扫(无效 VA = 干净 rc=1) | 模块文本不在任何 DtoH 可达区 |
| `scan_maps.cpp` | /dev/nvidia* 宿主映射扫描 | 56MB nvidiactl staging 窗只有静态 SBT 拷贝 |
| `kprobe.cu/_host.cpp` | 内核侧 VA 探针 v1(全有全无) | 2MB 全跨度探针太粗暴 → fault |
| `kprobe2.cu/_host.cpp + drive_kprobe.py` | v2:pinned 渐进 + 跳页 | 引擎工作区/slab 结构性不可读 |
| `kprobe3.cu/_host.cpp` | v3:全数据走 pinned + 进程内候选推导 | 0x52 窗逐页 fault 终判 |
| `heapdiff.cpp` | 驱动堆 pre/mid/post 三点 diff | config 双缓冲;mid-flight 无变化 |
| `rela_scan.cpp` | 宿主镜像 constant2/rela 解析检查 | 重定位在 GPU 侧做 |
| `stage_scan.cpp` | 发射期间扫 staging 窗 | 只有静态内容,发射数据不经过 |
| `test_dtoh.cpp` | DtoH 对无效 VA 行为标定 | **rc=1 干净不毒化**(可放心扫) |
| `patch_capture.cpp` | 宿主 JIT 镜像定位 + 5 变体 patch + launch | patch 在 create 后无效(上传在 create 内) |
| `watch_capture.cpp` | SIGSEGV 页保护 watch(fault rip/栈回溯) | JIT 缓存跨 pipeline 复用确认 |
| `vk_bvh_dump.c` | Phase 2:Vulkan AS 构建/dump(留档) | D7 证据 |

### interposition/hook(Phase 8)
| 文件 | 用途 | 状态 |
|---|---|---|
| `hook_dlsym.c` | **hook dlsym/dlopen**(libnvoptix 零 cu* 导入,只经 dlsym 绑驱动)→ wrap cuGetExportTable/ModuleLoad*,dump 9 张 xtab | 有效,但模块上传不走这些路径 |
| `detour.c` | 运行时 inline-detour 框架:公开 stub 解析(`ff 25 disp32` 间接跳转)→ trampoline(rip-fix)→ 内实现劫持;export table 全表项透明 asm thunk(压参/恢复/helper/转发) | 框架可用;证明 ModuleLoad*/Link* 零调用 |
| `hook_load.c` | Phase 5 LD_PRELOAD 钩子(留档负结果) | cu* 内部调用绕过 PLT |

### 已删除(结论已记录,勿再生成)
- `stage_*.bin`(2.2GB staging 快照:仅静态 SBT 拷贝 @+0xe88c0)
- `devscan_*.bin`(每进程 ASLR 的堆快照;结构结论在 HANDOFF/heap_mid+post)
- `vmm_dump.bin`(run_ttu4 运行副产物,会再生成)
- `exp*/e6*/e7*(除w1)/v1-v7/s1/s2v7/movref/probe{,2}.cubin`:Phase 1-4 编码二分产物,全部可由 build_and_patch/build_ttu2/build_ttu3 再生
- `probe5_m_*.cubin/probe6_*`:变体扫描产物,TTU5_*/TTU6_* 环境变量可再生
- `probe5.cubin`:build_ttu5 再生(TTU5_TAG)
- `tmtu.bin`(=_text_traversal_mttu.bin 重复)、`_nv_*.bin`(constant0/3/global.init=全零模板、shared=0B,均可由 jit_16 再生;coreParams 结论在 HANDOFF Phase 9 §6)
- `x_*.dis`(miss/CH/rtcoreNop/triangle_fetch 小函数反汇编:结论在 HANDOFF Phase 9 §1,再生 = sec_elf+rebuild+nvdisasm)
- `hook_export.*`:PLT hook cuGetExportTable 未触发(libnvoptix 走 dlsym),被 hook_dlsym 取代
- 编译二进制(全部有源码)、日志(detour/hook_dlsym/launch_params:结论已提取)

## 3. 编译命令备忘
```bash
# probe 内核
nvcc -arch=sm_86 -cubin probe3.cu -o probe3.cubin
# host 工具
g++ -O1 -o X X.cpp -I ~/gpu-engine/OptiX-8.0.0/include -I ~/cuda-12.1/include -lcuda -ldl
# 纯 CUDA 工具(无 OptiX)
g++ -O1 -o X X.cpp -I ~/cuda-12.1/include -lcuda
# hook/so
gcc -shared -fPIC -o hook_dlsym.so hook_dlsym.c -ldl -I ~/cuda-12.1/include
gcc -shared -fPIC -o detour.so detour.c -I ~/cuda-12.1/include
# eoss-framework import 需要符号链接(已存在)
# /home/hzx/gpu-engine/research/community/eoss -> eoss-framework
```

## 4. 实验纪律(每次跑之前看一眼)
1. 先 `nvidia-smi` 挑空闲 GPU(5/7 常空闲),`CUDA_VISIBLE_DEVICES` 隔离
2. 任何可能 700/715/716 的实验:独立进程 + `timeout`;fault 毒化 context
3. probe3/probe5 的 cubin 用后即弃可再生;**不要删 jit_*.elf / optix_as.bin / heap_*.bin / xtab_*.bin / e7_w1.cubin / tmtu_dis.txt / _text_*.bin**
4. 遍历发散(内核死循环)不用管:驱动 watchdog ~6s 自动恢复;**永远不要发缺 CLOSE 的挂死序列给独占 GPU**
5. fault 定位:**cuda-gdb batch**(compute-sanitizer 对 cuModuleLoadData 的驱动私有模块无设备侧输出):
   `cuda-gdb -batch -x cmds --args python3 -u script.py`,异常 PC − stateFunPtrs 槽 VA = 函数+偏移

## 5. Phase 9 关键事实(速查;详情 = HANDOFF Phase 9)
- **状态栈架构答案**:TTU 输入块 = 遍历序言寄存器现算;`ST[0x0000]=(mask<<16,time_frac)` 配置字
  (非指针!);ray 走 shared(`tid|((tid*8)&0x1f)` 置换槽 ×4 阵列 @+0/{0x100,0x200,0x300});
  0x100/0x200=0(root 不在 ST 槽);0x180/0x190=外核上下文(未解码)—— e7_w1 的
  "ST[0x100]=root" 布局假设作废
- **megakernel 状态机**:R21=*(S16*)coreParams[0x38];funptr=c[0x3][0x58]+R21*8;
  CALL 的返回地址 = R4 = mk+0x440(重定位);raygen→-1→traversal(可重入,-1=未完成);
  ≥-5 = yield 回外核;raygen1 写 out(coreParams[0x108])并置 -7 → EXIT
- **coreParams**(.nv.constant3 前 256B):[0x38]=状态变量ptr [0x40]/[0x48]=nextTblA/B
  (run_mk 复用为全局栈基,因 yield 路径被跳过)[0x58]=stateFunPtrs [0x6c]=spill 基址
  (真驱动=0,lmioff 相对)[0x100]=root [0x108]=out [0x28]=LMEM 对
- **stateFunPtrs** 14 槽:[0-3]=rtcoreNop* [4]=traversal_mttu [6]=triangle_fetch_mttu
  [10]=miss [11]=closesthit [12]/[13]=raygen0/1;5/7/8/9 驱动运行时填(load 后=0,须补)
- **loader 行为**:cuModuleLoadData 自动应用 .nv.global.init 重定位 ✓;shared 大小认
  section header sh_size ✓;~~local 窗口对驱动格式模块不设(LMEMHIOFF=0)~~
  **[Phase 10 修正]** LMEMHIOFF=0xfff9c0,local 窗本来就有(traversal spill 循环真跑);
  **[Phase 11 补充]** raygen1 的 LDL 全地址 700 = 它自己的 per-function 帧为 0,
  本地窗可用性按函数而定;STL 对非法地址不 fault(store 假干净)
- **SASS 手工编码规则全集** = HANDOFF Phase 9 §5 + **Phase 10 §1 修正**(Phase 9 的
  "IMAD.MOV c-bank byte4=off/4 byte5=bank<<6" 系错读,**实际 b5=off/4、b6=bank<<6**;
  BRA 相对 pc+0x10;谓词 = word byte1 bits[6:4]+bit7;NOP = 0x7918 +
  ctrl 0x000fc00000000000)+ **Phase 11 §2.5 补遗**(STL/LDL b2=基址 b3=恒 0
  b4=数据;RZ 不可作 local 基址;寄存器文件按函数裁剪,raygen1 顶 = R106)
- ~~**待解**:.128 spill 全局化 ctrl-size 矩阵~~ **[Phase 10 判死 + 编码已全解]**:
  全局化结构性不可行(12 种基址寄存器);真值 = HANDOFF Phase 10 §1;.X8 双 shared
  窗口 = Phase 10 §7;LMEMHIOFF=0xfff9c0(窗口本来就有)
