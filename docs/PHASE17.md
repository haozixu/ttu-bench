# Phase 17：受限的父 IAS 续遍历

日期：2026-11-03。RTX 3090 / SM86，driver 535.309.01，CUDA 12.1，OptiX 8。承接 [Phase16](PHASE16.md)。本阶段是功能验证，没有性能测量。

一个 CUDA kernel 现在可以保存硬件返回的父级待处理引用，独立查询当前 child，再恢复这些引用继续查 IAS。**已验证平坦 IAS、1–3 个静态正行列式变换的 instance、共用一份 triangle GAS、ray flags1（DISABLE_ANYHIT）、geometry flags0，以及返回引用类型0x21。** 这比单 child 查询多了一段真正使用硬件返回状态的续遍历，但不是完整 OptiX/JIT 遍历状态机，也不是任意三实例布局的保证。

## 两个反例为什么重要

仅仅查出第一个 instance 的 child，不足以完成 IAS 最近命中。正式矩阵包含两个固定反例，并记录实际事件顺序：

| 场景 | 第一个 child | 完整结果 | 停在第一个 child 的负对照 |
|---|---|---|---|
| first-miss-later-hit | instance41，miss | instance17，primitive0，t=5 | 错误地报告 miss |
| first-farther-later-nearer | instance41，t=5 | instance17，primitive0，t≈4.128019 | 错误地保留 instance41、t=5 |

两个负对照确实得到错误的几何结果，且仍有 pending work；不只是人为设置失败状态。完整循环还通过交换 builder 输入顺序的变体。遍历顺序来自 raw 事件，不能由 builder 输入顺序推断。

## 修正：instance ID 不能从 E 附近猜

Phase15/16 的单实例实验曾把 `E-0x0c` 低24位当 instance ID、`E-0x20` 当 child handle。这两个相邻偏移**撤回为通用解码规则**。双实例中，当前事件确实对应 instance17，`E-0x0c` 却是41。单实例正确只是布局碰巧相邻；历史报告及其 hash 原样保留。

与 JIT 对照后，当前解码为：

```text
E = R72 | ((R73 & 0x1ffff) << 32)
index = uint32[E+4]                       # 1-based
table_offset = uint64[IAS_base+0x30]
count = uint32[IAS_base+0x70]
M = IAS_base + table_offset + (index-1)*0x80
instance_id = uint32[M+0x74] & 0xffffff
child_handle = uint64[M+0x60]
```

`E+0x10` 仍是逆矩阵，与索引元数据 `M+0x30` 的12个 float 交叉核对；`uint64[E+8]` 低49位仍对上 child root。host decoder 检查索引、边界及矩阵一致性；device kernel 检查索引、分配边界和 child root。该布局依然是本机已测 builder/driver 的经验协议。

静态证据：原 JIT 的0x1030读取 `[E+4]`，0x1150减1；0x4360读取 traversable base+0x30，0x4460–0x4480使用0x80步长。见 [instance片段](../reference/jit/jit16-instance-ordinary.sass)。

## 怎样恢复父级

19条组返回的 R54:R55 提供观察到的续遍历基址/配置；`(R55>>25)&7` 对应本阶段观察到的显式 pending 数量。条目位于 R56:R57、R58:R59、R60:R61。当前只接受 `((entry_high>>17)&0x1ff)==0x21` 的相对引用类型。

在查询 child **之前**，普通 CUDA 代码把这些条目保存在软件栈。每个条目保存：

```text
base_low  = R54
base_high = (R55 & ~(7<<25)) | (1<<25)
entry_low, entry_high = 对应的返回寄存器对
```

完成 child 后，保留当前最近 t，重新准备世界坐标 ray，将 tmax 缩到最近 t。弹出一个待处理引用，构造下一请求：

```text
R50:R51 = base_low:base_high
R34:R35 = entry_low:entry_high
R54..R61 = 0
```

然后重新 OPEN/GO/LD.CLOSE，得到下一个 instance 事件或 miss；可能产生的更多 pending 引用继续入栈。不是保持同一次 OPEN，也不是遍历 builder 提供的所有 instance 独立求交取 min。每个 child 查询仍是独立的 GAS 请求，父级进度由返回引用连接。

直接照搬整组返回状态、或者只搬一个条目却不保存其余条目，会丢失后续工作。[run_parent_probe.py](../run_parent_probe.py) 保留 none/copy/promote/pop 的诊断入口；其中 copy/promote 是待否证的历史假设，不是可用的通用算法。它的 pop 也是 host 诊断策略，不承担正式最近命中判定。

JIT 对照新增 [parent普通指令片段](../reference/jit/jit16-parent-ordinary.sass) 和 [来源manifest](../reference/jit/jit16-parent-disassembly.json)：0x2b20–0x30f0处理返回描述符，0x3200–0x3740、0x40a0–0x41d0保存引用，0x7e40–0x84a0弹栈恢复，0x8500–0x8580返回请求准备。本实现只提取已验证的子集，没有逐字移植全部 shared/local 栈分支。反汇编文件的 TTU 槽已换 NOP，仅供静态分析，不可运行。

## 实现与复现

| 文件 | 用途 |
|---|---|
| [parent_resume.py](../parent_resume.py) | raw state transport、parent kernel ABI及编译/launch封装 |
| [parent_template.cu](../parent_template.cu) | 单kernel的事件分派、child查询、pending软件栈及恢复循环 |
| [run_parent_resume.py](../run_parent_resume.py) | 每次最终结果及每个child与CPU/OptiX对照，保存输入/父/子完整raw状态 |
| [validate_parent_resume.py](../validate_parent_resume.py) | 15场景×两GPU，加两个 first-child-only 负对照；每项独立进程/context，40秒timeout |
| [phase17-parent-cases.json](../fixtures/phase17-parent-cases.json) | 显式变换、反例ray及交换输入顺序的场景定义 |

共享 [device_instance.py](../device_instance.py) 增加可选kernel名、源码及参数/结果类型；marker只在目标kernel的text section内查找，避免编译器为不同入口复制 device callee 后误计数。默认单instance接口保留。

```bash
CUDA_VISIBLE_DEVICES=6 timeout 40s python3 run_parent_resume.py --scenario first-miss-later-hit --check --output results/local/parent-miss-hit.json
CUDA_VISIBLE_DEVICES=6 timeout 40s python3 run_parent_resume.py --scenario first-farther-later-nearer --check --output results/local/parent-nearer.json
python3 validate_parent_resume.py --gpus 2 6 --output results/local/parent-matrix.json
```

先检查GPU空闲情况，index仅为本次运行选择。负对照加 `--stop-after-first --check`，预期退出码1。发布结果通过manifest解析，不能直接按矩阵原始子报告名在当前目录读取。

## 验证结果

[正式矩阵](../results/phase17/final-parent-matrix.json)：GPU2和GPU6各15项，每条ray重复2次，共 **30项、4,768次kernel launch、11,492次TTU请求，其中1,484次为恢复父级后的请求**，全部通过。各GPU为2,384次launch、5,746次请求、742次恢复请求。两张卡的UUID、driver、源码/fixture/group hash保存在原报告中。

包含1/2/3 instance、交换输入顺序、旋转、非均匀缩放，三实例共享grid64 GAS、旋转双实例共享separated GAS。最终比对 hit/miss、instance/primitive ID、t、barycentrics和front face；每个事件还核对索引/矩阵与缩短 tmax 后的 child CPU oracle。正式矩阵观察到的软件pending最大为2项；这也不是硬件栈容量上限。

共享代码改动后，[单instance回归](../results/phase17/single-instance-regression.json) 的8项、1,344次launch、2,615次请求全通过；[固定fixture回归](../results/phase17/fixed-fixture-regression.json) 的8项、40次launch（含原20条组）通过；编码round-trip及恢复源码hash检查通过。

[资源记录](../results/phase17/parent-resource.json)：parent_probe为88 registers/thread、848字节stack、0显式shared；结果结构7856字节，保存逐步raw trace。`LOCAL:0` 不表示没有stack搬运。只有一个active thread，编译使用 `-Xptxas -O0`。它是可读功能容器，不能据此推算TTU延迟、最小寄存器占用或内部遍历栈大小。

## 保留的边界与负观察

- 四实例的 [host诊断](../results/phase17/pop-four.json) 首次返回的三个条目类型为0x21、0x21、**0x24**，第三个引用重新提交后直接返回triangle，而不是0x16实例事件；当前kernel不实现其语义，并拒绝instance数大于3。3不是硬件容量上限，也不代表增加计数限制就能支持任意IAS。
- [等距重叠样本](../results/phase17/device-chain2-overlap.json.gz) 有51项检查失败：29条裸TTU记录未通过、22项CPU/OptiX比较不匹配；它们不是51条不同ray。29条裸TTU失败记录的t均与CPU相同，冲突在命中身份。两份GAS三角形经instance变换后完全重合，CPU的 `<=` 更新选择后一个身份，而当前缩短 tmax 的 child 查询未必返回相等端点；OptiX也未必选同一身份。观察到的冲突涉及并列命中身份/边界语义，尚未系统解决。保留原始失败，不放宽oracle把它改成通过，也不将它计入正式正矩阵。
- 软件栈16项、父请求最多8步；不支持的引用类型、返回类型、边界或步数都会显式返回错误。没有证明任意AS都能在这些界限内完成。
- 嵌套IAS、不同child GAS、负行列式变换、motion、用户any-hit/custom primitive、一般溢出路径及full warp仍未实现。flags0较远候选反例也仍未解决。

下一功能任务是从四实例最小反例辨认新引用类型与直接triangle返回的坐标/instance归属，再逐项扩展；另行隔离等距/tmax边界。测量线可以先用已稳定的opaque GAS推进正确设备内重复和计时，无需等待完整OptiX支持。
