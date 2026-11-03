# Phase 12：独立 CUDA TTU 的 hit/miss 已跑通

日期：2026-11-02。设备：RTX 3090 / GA102 / SM86，driver 535.309.01，CUDA 12.1。

**当前主入口是 `run_minimal.py`。它在全新 CUDA context 中自行分配 AS、dummy root 和输出缓冲，只发射自研 CUDA kernel；不创建 OptiX pipeline，不调用原始 megakernel，不需要驱动私有调度上下文。**

这是功能正确性里程碑，尚无 RT/CUDA 争抢或延迟测量结果；13 条是当前已验证的精简 TTU 序列，不声称全局最短。研究问题与已有工作的边界见 [研究空白审视](../../research/notes/rt-cuda-research-gap.md)。

## 复现

先用 `nvidia-smi` 选择空闲卡。在 `ttu-bench/` 项目根目录运行：

```bash
CUDA_VISIBLE_DEVICES=7 timeout 20s python3 run_minimal.py hit --check --repeat 5
CUDA_VISIBLE_DEVICES=7 timeout 20s python3 run_minimal.py miss --check --repeat 5
CUDA_VISIBLE_DEVICES=7 timeout 20s python3 run_minimal.py hit --oz -10 --check
CUDA_VISIBLE_DEVICES=7 timeout 20s python3 run_minimal.py hit --dz 2 --check
CUDA_VISIBLE_DEVICES=7 timeout 20s python3 run_minimal.py hit --tmax 6 --check
```

输出为 JSON；`--check` 根据原始三角形做独立的解析求交，失败会非零退出。`mode` 只选择默认 x/y，最终结果由全部 ray 参数决定，例如 `hit --tmax 6` 应得到 miss。

完整验证：

```bash
python3 validate_minimal.py --gpus 5 7 --output results/local-functional.json
```

每个 case 独立进程 / 新 context；每 case 发射 5 次（每次加载/卸载自研 module），覆盖精简组和原始 20 指令组。已在 GPU 5、7 各完成 8 case，共 **16 case / 80 launch，全通过**。原始结果、GPU UUID、资源报告现收在 `results/phase12/`；曾保存的含进程VA的cubin已删除，见文末历史说明。GPU kernel 的资源为 **80 registers/thread、0 local、0 shared、0 stack**，单 active lane。

另有两个反例对照：`MINIMAL_NOTTU=1` 跳过 TTU、`MINIMAL_ZERO=50,51` 清零 dummy 指针，两者均无法通过基础 hit 的解析 oracle，记录在 `results/phase12/phase12-negative-controls.json`。这些变量仅用于诊断；验证脚本会清除它们。

保存的 cubin 含当次分配的 VA，只供编码/资源核验，不能在新进程直接重放；每次运行会重新 patch 地址。`--repeat` 验证多个独立 launch，尚不验证同一 kernel 内的重复 TTU 循环。输出转储 R0–R75 中未初始化且未被 TTULD 写入的寄存器没有解释价值。

## 几何与判据

三角形顶点来自生成参考 `reference/optix/build_fixture.cpp`：`(1,0,0), (0,2,0), (0,0,3)`；其平面为 `3x+1.5y+z=3`。

| 输入 | 预期 t | 实测 R73（FP32 的 -t） |
|---|---:|---|
| o=(.25,.25,-5), d=(0,0,1) | 6.875 | `c0dc0000` |
| o=(.25,.25,-10), d=(0,0,1) | 11.875 | `c13e0000` |
| o=(.25,.25,-5), d=(0,0,2) | 3.4375 | `c05c0000` |
| o=(.5,.25,-5), d=(0,0,1) | 6.125 | `c0c40000` |
| o=(50,2,-5), d=(0,0,1) | miss | R72=R73=`ffffffff` |
| 基础 hit ray，但 tmax=6 | miss | R72=R73=`ffffffff` |

基础 hit 的 R72=`a0000000`，R32=`be000000`、R33=`3f200000`。R72 全部状态位、R32/R33 的完整编码语义尚未解完，不应将 R72 整字误认成 primitive ID 或普通 FP32 t。这里只把 `-FP32(R73)` 作为该协议 / 正 z 方向 ray 族的已验证距离判据。尚未测三角形边界、负方向、任意 dx/dy、实例层次。

## 实现与实际依赖

- `minimal_template.cu` 仅提供合法 CUDA ELF 容器；其 FADD 链在加载前被完全替换，不执行。
- `minimal.py` 根据 TTUST 的显式源寄存器对，只初始化所需 GPR；载入 CUDA 全局访存 descriptor，发射 TTU 组，等待 scoreboard 后写回结果，EXIT。GPU 端没有软件 traversal、callback、local/shared 访问。
- `run_minimal.py` 从保存的 `fixtures/jit_16.elf` 中取 **0xe70–0xfa0 的 20×16B 原始协议**，从 `.nv.global.init` 取 dummyRootNode 的前 128B；这是静态数据来源，不会加载/执行该驱动 module。另读取已保存的 `fixtures/optix_as.bin`，不在运行时构建 AS。
- `cuda_driver.py` 提供当前使用的 Driver API 与 ELF 读取；清理后不再从已删除的旧 megakernel runner 导入任何代码。
- AS 用普通 `cuMemAlloc` 分配在高 VA，修正 `AS+0xd0` 自指针；本例不需要低 4GB VA。只验证了此固定单三角形 AS，不外推所有生产 AS 可同样重定位。
- **dummyRootNode 不可省略**：原 module 的此对象包含单位变换。直接清零输入 R50:R51 会把 hit 变成 miss；只清 R50 可能 fault。现在从原始静态数据复制后自行分配和编码。
- 原 AS SHA256：`715c99ef42c84709f9d0f91b6b458e253895981dec2ea4fa16a51cf56c7ab269`。

两个已验证的地址编码（不外推到任意节点类型）：

```text
R50 = uint32(dummyVA >> 6)
R51 = uint32(dummyVA >> 38) | 0x02402800
R54 = uint32((ASVA + 0x300) >> 7)
R55 = uint32((ASVA + 0x300) >> 39) | 0x02484000
```

Phase13纠正：此前R50额外OR8来自固定现场中的地址位，不应固化为tag；原JIT的0x7f0–0x840为右移/搬运。去掉后固定fixture16case/80launch仍通过，见 `results/phase13/phase13-baseline-regression.json`。生产AS与多三角形的新增能力及限制见 [PHASE13.md](PHASE13.md)。

## 精简组：13 条 TTU 指令

```text
TTUOPEN
TTUMACROFUSE(11)
TTUST [0x000] <- R42:R43, R52:R53
TTUST [0x040] <- R12:R13, R66:R67
TTUST [0x050] <- R16:R17, R14:R15
TTUST [0x080] <- R72:R73, R40:R41
TTUST [0x0c0] <- R36:R37, R24:R25
TTUST [0x0d0] <- R32:R33, R2:R3
TTUST [0x100] <- R34:R35, R50:R51
TTUST [0x180] <- R56:R57, R54:R55
TTUST [0x190] <- R60:R61, R58:R59
TTUGO
TTULD.CLOSE [0x300] -> R72:R73, R32:R33
```

相对于原组：删除 P4 分支的 ST[0x200]，将其余条件 ST 改为无条件；只读 [0x300]，将 CLOSE 移到此 LD，删其余 6 个 LD；同步修改 MF 计数。保留 `--full-group` 作为原始 20 指令对照，P4=false。默认精简组不需要设置 predicate。

原始 group SHA256：`f8ff2e20f9711dcaff44c9af412c1e4f79d2f00ccfe56aceb4795e6584f50a02`。
精简 group SHA256：`ced3108f52be2c8fd4fb811b988e5e71f81a4ccf70d8a6df0aaa9d2af66e6bbd`。

## 对 Phase 1–11 结论和旧待办的纠正

| 旧说法/待办 | 新证据与处理 |
|---|---|
| TTULD 的 Rd 在 bits39:32，Rosetta 的 bits23:16 错了 | **旧结论错误**。有效原组的 Rd 就在 bits23:16；另一显式目的寄存器在绝对 bits71:64。实际 LD 目的含 R50、R54、R58、R2，直接否定“必须 4 对齐” |
| ctrl byte0 必须 0 | **旧结论错误**。这是 Rc / 第二目的等字段的一部分；本次原组大量非零值有效 |
| TTUST 只传一个隐式相邻对 | 不完整。此组的 ST 是两个显式寄存器对，Rb 在 bits39:32，Rc 在 bits71:64；每对内部相邻 |
| -1 是拒绝码，miss 必为 0x7fffffff | **旧推断无依据**。此有效 mttu 协议的 miss 就是 R72:R73 全 -1；旧 e7 不是可靠通用 oracle |
| hit/miss 输入不同却返回相同 | 旧 megakernel runner 解包 mode 却没改 raygen 常量，实际在跑同一条 ray；当前独立 runner 直接合成输入并检查几何结果 |
| MK_LOCAL=1 保留全部 local 指令 | 旧代码仍调用 `patch_globalize_local(..., on=False)`，导致 **88 个 local spill 槽被 NOP**；此路径已被独立 kernel 替代并删除 |
| R56/R60 从未被写，是外核上下文 | **静态事实错误**。例如 R60 在 traversal+0xc50 被赋值；此例允许这些输入为零，完整独立 kernel 已证明无需外核现场 |
| 组后 MOV 中继足以读出 TTULD | 缺少正确的 scoreboard 等待，也可能被后续软件覆盖；改用直接 snapshot + EXIT 读 GPR |
| 先做 in-flight OptiX 注入、修完整 yield/callback，再推进 TTU | 对当前最小 ISA 目标已无必要；保留旧资产，不沿此路线继续耗时 |
| 独立 kernel 的 715 说明 TTU 不可脱离驱动 | no-TTU 对照也曾 fault，属于宿主指令/写回夹具；早装 descriptor、调整写回寄存器对后通过。未单独归因每个修正的硬件机制，不把它写成 ISA 规则 |

定位这些问题时曾在组前/组后覆盖软件路径，等待 scoreboard，直接转储 GPR 并 EXIT；最终验证进一步移到全新 context，消除先前 kernel 留下状态的混淆。恢复真实 local 指令后的完整 megakernel 仍会在后续 +0x1bf0 local load fault，因此不作为可用基线。清理时已删除 `run_mk.py` 及其 snapshot/中继开关；方法和根因记录在此，不保留失效的复现命令。

## 下一步优先级

1. 分离 TTULD 输出与 TTUST 输入，去掉重复初始化，构造依赖正确的重复 TTU 操作；设备内计时排除输入准备、global 输出和 launch。不能把目前 Python 运行时间解释为 RT latency。
2. 将目前固定 GPR 编号紧凑重映射，降低 80-register 驻留开销；加入整 warp 的 lane/ray 语义验证，再控制 RT warp 数。单 active lane 不代表 full-warp 吞吐。
3. 一个 CTA 内不同 warp 做 TTU 与 FFMA/INT/S2R/shared/LDG aggressor，核验同 SM；先得热小 AS 下的局部干扰矩阵，再扩大 BVH 工作集区分下游访存。
4. 扩展 AS / ray family，依据已经补读的 Ray-by-Ray 与 RTSpMSpM 全文界定增量；提出可否证的资源模型，而不是仅报告 slowdown。

2026-11-02 清理后：当前 fixture 在 `fixtures/`，其他原始 JIT 与 AS 格式参考在 `reference/`。旧 heap/xtab/VMM 快照、错误路线脚本和临时产物已删除，决策与原始 hash 见 [CLEANUP.md](CLEANUP.md)。原验证结果保留；清理后的回归另存 `results/maintenance/cleanup-functional.json`。保存的含进程 VA 的 cubin 已删除，需要时通过 `--artifact` 重新生成，仅供检查。
