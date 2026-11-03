# EoSS 复现、SM86 差异与可复用组件

更新：2026-11-02。协议行为、架构/驱动差异和失败实验本身具有工程价值，不以能否形成 research contribution 决定是否保留。先前把它们与错误推断一起删除是过度清理，现已纠正。

## 恢复范围与来源

旧 README/INDEX/HANDOFF/NEXT_PROMPT 从清理前会话输出恢复到 `reference/history/`。`build_ttu3.py`、`probe3.cu`、`run_ttu4.py` 的历史源码与 `cleanup-2026-11-02.json` 中删除前的大小和 SHA256 一致。历史文档包含错误结论，放在历史目录保留原文，不作为当前运行说明。

原 `build_and_patch.py`、`run_exp.py` 的完整源码未找到。当前同名文件是**重建版**，保留 exp1–5 实验名；改用显式寄存器初始化和读回容器，不能视为旧 FADD 链 harness 的逐字节复原。

## 两条可以独立使用的实验线

```bash
CUDA_VISIBLE_DEVICES=7 python3 run_exp.py all --repeat 3 --output results/local-eoss.json
CUDA_VISIBLE_DEVICES=7 python3 run_exp.py exp2_open2 --repeat 3
CUDA_VISIBLE_DEVICES=7 python3 run_minimal.py hit --check --repeat 5
python3 validate_encoding.py
```

先用 nvidia-smi 选空闲 GPU。`run_exp.py` 自动为每个变体和 trial 创建独立子进程/context，20 秒超时；超时后停止整组，保留错误和 stdout/stderr。不会自动重置 GPU。`all` 包含 probe、exp1–5 和 IVALL 对照；可能挂起的 scoreboard-write 变体只能显式选择，不在默认组内。错误 count、缺 CLOSE 的挂起扫描没有加入默认实验。

`run_exp.py` 默认一个 active lane，`--threads 32` 可选择完整 warp，但该模式所有 lane 共用 dump 地址，只适合看执行/fault 状态，不能解释为逐 lane 结果。默认 `--wait-mask 0` 用于独立指令行为；`run_minimal.py` 保持已验证的结果等待。复现一个 fault 必须记录 TTU 编码及后续 wait 条件，不能仅比较助记符。

## 历史与新测结果

历史结果来自恢复的 `docs/reference/history/README.pre-cleanup.md`；缺少最初 raw 日志。新结果使用 GPU7/SM86、driver535.309.01，每项三个新 context，记录在 `results/eoss/eoss-reconstructed-sm86-nowait.json`：

最终入口检查另存 `results/eoss/eoss-basic-final.json`：7变体×3次共21项均符合预期，包括双OPEN的715和context毒化。该报告增加了预期状态检查；正常指令发生意外CUDA错误时，runner也会非零退出。

| 实验 | 历史 SM86 记录 | 重建 harness 新测 | EoSS §6 的 SM120 报告 |
|---|---|---|---|
| probe | OK | 3/3 OK | 对照 |
| exp1_open1 | OK | 3/3 OK | 单次 OPEN 成功 |
| exp2_open2 | rc715，重复3次；context 毒化 | 3/3 rc715，之后 sync 仍715 | 第二次 OPEN fault |
| exp3_mf0 | OK | 3/3 OK | count0 成功 |
| exp4_cctl，IVALL=0 | OK | 3/3 OK | 默认 CCTL 成功 |
| exp4_cctl_ivall，IVALL=1 | 无独立历史记录 | 3/3 OK | IVALL 默认配置成功 |
| exp5_group，原 artifact 的 MF8/4ST/GO/3LD | garbage 输入下 OK | 3/3 OK，零 GPR 输入，无 AS | 完整协议结合 rtcore__rays.sum 验证1条 ray |

最后一项只能说明这些字节在该输入/harness 下完成，不能证明它关闭了 SM86 会话、发起真实 traversal 或与 SM120 执行相同操作。真正 hit/miss 的证据来自另一条带 AS 和解析几何 oracle 的功能线。

两个显式探索变体留下了新负结果：

- `results/eoss/eoss-reconstructed-sm86-lane1.json` 保存重建 v1：CCTL IVALL=0、write scoreboard0，后续 wait_mask=0x3f，在20秒超时；未继续后面的变体。此前18次基础试验均完成，包括预期的双OPEN fault。超时记录没有指令级 fault PC，不能区分 CCTL、本次 wait 或其他同步位置。
- `results/eoss/eoss-cctl-ivall-wr0-nowait.json` 保存重建 v2：GPU5，IVALL=1、write scoreboard0、后续 wait_mask=0，首个 trial 仍20秒超时，停止重试。论文报告 SM120 的 active scheduling scoreboard write 导致 fault。这是值得保留的环境差异，但尚无同字节 SM120 对照，不能宣布架构机制不同。

两次超时后 nvidia-smi 均显示设备空闲、无残留 compute process，没有执行 GPU reset。超时进程被终止不等于已证明硬件 watchdog 语义。没有为补齐表格反复发送这些挂起变体。

## 可复用代码与知识

| 模块 | 接口与边界 |
|---|---|
| `ttu_isa.py` | `inst/ttuopen/ttumf/ttust/ttugo/ttuld/ttuclose/ttucctl` 返回16字节；`decode/field/set_field` 保留未修改位；`input_registers/fused_group` 管理显式寄存器对和count。TTUCLOSE 是 Rosetta 的 LD.CLOSE/RZ 别名，仅做编码支持，未单独实测 |
| `eoss_helpers.py` | 原样读取 SM120 canonical 数据；调用 EoSS `BinaryPatcher` 与 `PatchPolicy.REPLACE_FULL`；EoSS 调度字段读写可从 `control` 使用 |
| `cubin_builder.py` | 编译合法单入口 CUDA ELF，生成 MOV/寄存器dump，替换 SASS 和寄存器/EXIT 元数据；支持 EoSS patch 回调；当前元数据处理仅支持自己的模板 |
| `minimal.py` | 从 TTUST 推导输入并生成 standalone probe；`launch_image` 接受已构造 image，供两条实验线复用 |
| `sm86_fixture.py` | 经 hash 校验的原20条/精简13条组、单三角形地址打包和 ray 寄存器模板；这些是 fixture 知识，不假装通用 ISA |

例如构造或修改指令，不需要复制 minimal 的位运算：

```python
from ttu_isa import ttuld, ttust, fused_group, decode
from eoss_helpers import control

store = ttust(0x40, 12, 66)
load = ttuld(72, close=True, address=0x300, rd2=32)
load = control.write_control(load, wt_bar=0)
print(decode(load))
```

这只是编码示例；任意 ST/LD 拼成一组不保证协议有效。`fused_group` 仅检查词法 count，不能检查 AS、输入语义或硬件会话状态。指令模板中的 scheduling 默认值是本地工作组的起点，更换依赖和寄存器后必须重新核验。

## 应保留的 SM86 / SM120 delta

| 观察 | 当前证据 | 可作什么解释 |
|---|---|---|
| 两代TTUST/TTULD均有bits71:64第二寄存器字段；SM120内部键名为Ra_URc | 完整ARGUMENTS与FORMAT对照；8个TTU类的opcode、按alias归一化的字段布局及condition表达式均一致 | 纠正前版“SM120没有第二寄存器字段”的误读；它只是键名变化，不构成已证ISA差异。静态一致不保证内部硬件、时序或全部运行行为相同 |
| 535镜像内 mttu MF18、10ST、7LD 与 EoSS 描述的组不同；同镜像还有 utrav MF6 家族 | 保存的 JIT 字节及旧记录；mttu20/13条均通过 oracle | 驱动生成协议/遍历家族的差异；不能把整个535或SM86叫作“描述符协议”，也不能仅归因于架构 |
| SM86 上未见论文使用的 rtcore__rays.sum | 本机工具可见性记录 | 环境/工具差异；不据此断言所有 Ampere 不可测 |
| 本地mttu fixture支持普通cuMemAlloc高VA；论文重放选择低4GB的VMM区域 | 本地hit/miss及地址打包；EoSS §6描述低4GB重放 | 有用的重放路径差异；不能据单fixture推断任意AS/SM120对高VA的支持情况 |
| IVALL+scoreboard write 的 timeout 与论文 fault 不同 | 上述新负结果与明确 harness 条件 | 候选行为差异，尚待最小同字节/同步条件对照 |
| EoSS canned 的 CLOSE-labelled 字是 bit72；SM86工作组为bit74 | artifact原始数据与工作组；**两代 Rosetta 的 close 均为bit74** | artifact与编码定义不一致，不能算已证明的架构差异；不静默改写上游数据 |

Rosetta 两代的 MACROFUSE.Sb 都是 bits44:40，5位；旧代码用8位掩码不代表硬件支持8位count。`validate_encoding.py` 校验20条原指令 round trip、13条组 hash、字段边界、EoSS调度改写不伤操作数，以及已恢复源码的原hash。

完整静态对照含源文件hash，保存在 `results/eoss/ttu-sm86-sm120-layout.json`。两代OPEN都有DUAL位（bit72），两代LD都有CLOSE位（bit74）。目前没有从这些编码资料发现明显的TTU接口代际断裂。更复杂AS、软件续遍历、驱动路径及探索路线见 [AS_ROADMAP.md](AS_ROADMAP.md)。

“Rd必须4对齐”“control byte0必须零”“ST000一定是描述符指针”“-1说明缺外核现场”已被有效组或功能反例推翻，应撤回解释，同时保留导致错误判断的实验条件，避免下次重新踩坑。
