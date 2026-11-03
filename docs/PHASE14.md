# Phase 14：一般 ray、flags 与 instance 返回边界

日期：2026-11-03。RTX 3090 / SM86，driver 535.309.01，CUDA 12.1，OptiX 8 SDK。承接 [Phase13](PHASE13.md)，未测性能。

后续 [Phase15](PHASE15.md) 已解码静态instance记录中的逆变换、ID和child引用，并验证host处理后的独立child查询；仍未完成父级状态resume。另补4096三角形GAS测试。本文保留Phase14当时的实验范围。

## 13 / 19 / 20 条组究竟差在哪里

下表的 count 是 MF 后的**词法指令数**，不是实测微操作数或硬件遍历步数。

| 组 | TTUOPEN / MF | ST | GO | LD | MF count | 功能证据 |
|---|---|---:|---:|---:|---:|---|
| full 20 | 原 JIT 字节不改 | 10，含互斥 P4 / !P4 提交 | 1 | 7，最后 CLOSE | 18 | 固定单三角形 fixture 通过；实时 GAS 有可重复 CUDA700 |
| expanded 19 | 保留原 OPEN，重算 MF | 固定 P4=false 后9条，改成 PT | 1 | 保留原7条，最后 CLOSE | 17 | Phase13 与本阶段 opaque 路径均通过 |
| reduced 13 | 同 expanded 的提交路径 | 9 | 1 | 只读[0x300]，在此 CLOSE | 11 | 与19条组同矩阵通过 |

20→19 同时改变条件执行、MF count、后续指令位置和普通输入/predicate准备；不是只少做一次有效 ST。P4=false 时原20条中本就有一个不应执行的 ST。当前结果**不能单独认定 predicate、MF、寄存器初始化或调度中的哪一项是故障根因**。新测 `DISABLE_ANYHIT` 请求仍不能修复20条组，见 [full 对照](../results/phase14/phase14-full-opaque-request.json)。固定 fixture 的20条组仍通过，故也不能宣称20条协议本身无效。

19→13 只减少6个返回槽的读取、重算MF并提前CLOSE。已经验证的triangle子集，其 hit/miss、ID、t、barycentrics与front-face信息都能从[0x300]一次LD取得。**这不说明其他6槽无意义**：19条组保留续遍历和其他事件的诊断信息。IAS已经出现非triangle事件，不能继续套用triangle decoder。

13不是理论最短；指令更少也没有证明更快。两组仍各有一次GO；指令数不能代替内部遍历、访存或RT工作量。当前推荐13条作为已验证子集的功能基线，19条作完整返回对照，20条保留为原字节诊断入口。

## ray 输入：从单轴扩展到一般方向

对照原JIT 0xb30–0xe60 的搬运和TTUST源寄存器，补齐：

| 字段 | 对应输入寄存器 |
|---|---|
| origin x/y/z | R66/R67/R12；另一份R24/R25/R36 |
| direction x/y/z | R14/R15/R16；另一份R2/R3/R32 |
| tmin | R13 / R37 |
| tmax | R17 / R33 |

接口是 `sm86_fixture.operands(..., dx=..., dy=..., tmin=..., ray_flags=...)`；固定fixture的旧调用仍有效。有限flags映射依据JIT的R52/R53和R40/R41构造：face-culling进入R52的bits4/5，DISABLE_ANYHIT进入R53的bit5，R40/R41从同一配置派生。不是直接把OptiX枚举原样写入任意槽。

一般ray套件包含±X/±Y/±Z、斜方向、非单位方向11种，两个非边界barycentric目标，对第一/中间/最后primitive采样；含tmin=5.25过滤、空间miss与tmax截断。重叠双三角形还含tmin=8跳过近处、命中远处，以及tmin=12全部排除。CPU按参数t计算，不把t当欧氏距离。

## 返回解码：此前的两个符号假设过窄

| 输出 | 当前triangle子集的经验解码 | 交叉验证 |
|---|---|---|
| R72低位 | `R72 & 0x0fffffff` 对上primitive | 实际覆盖ID至1023，不能反推完整位宽保证 |
| R72高位 | 观察到8/a两种tag | 真实语义仍未解释 |
| R73 | `abs(FP32(R73))` 对上非负参数t | opaque请求/几何命中为正；非opaque返回候选命中为负，完整状态语义待解 |
| R32 | `abs(FP32(R32))` 对上barycentric u | sign=0对上front face，sign=1对上back face；CPU determinant与OptiX `optixIsFrontFaceHit`独立核对 |
| R33 | FP32对上barycentric v | 同CPU/OptiX |
| miss | R72=R73=ffffffff | 与未知事件分开 |

此前固定负号的 `-R73` / `-R32` 只适用于原+Z、非opaque/back-face案例。[初始一般ray负记录](../results/phase14/phase14-general-initial.json) 保留了R32符号造成的误判；[Phase13 flag=1记录](../results/phase13/phase13-opaque-flag-negative.json) 中primitive与barycentrics已经正确，实际t为正，旧decoder错误翻成负数。**撤回“flag=1不支持/结果本身不正确”的推断**；原记录不改写。

`triangle_result`现在区分triangle、miss与other。未知状态不能因为R72低位碰巧像ID就强行解码；拒绝非有限/越界属性仍只是有限分类规则，不是完整事件ABI。测试避开边和顶点；signed zero、退化与其他极端另做。

## 重要反例：一次GO并不总是完整closest-hit

[精简复现](../results/phase14/phase14-nonopaque-candidate.json)：16三角形网格，ray=(.25,5.25,1.875)+(0,-1,0)t，geometry flags=0、ray flags=0。CPU与OptiX最近命中primitive4、t=2；13/19条组都稳定返回primitive0、t=5，barycentrics与front-face对该triangle是正确的。

同一一般ray矩阵，设置 `OPTIX_RAY_FLAG_DISABLE_ANYHIT`，或将geometry设为 `OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT`，即可与closest-hit oracle匹配。这**与nonopaque候选返回、软件accept/ignore后需要续遍历的解释一致**；尚未实现accept/ignore/resume，不能把这项解释当作已验证的完整状态机。没有any-hit shader并不保证裸提交路径自动具有OptiX完整closest-hit语义。

`run_scene.py`现在默认 `--ray-flags 1`，OptiX与TTU使用同样的请求flags。`--ray-flags 0`仍可诊断；CLI明确保存flags，并以closest-hit oracle暴露上述反例，不会把合法候选默认为成功。Phase13默认raw flags0但oracle disable-anyhit=1，原+Z矩阵恰好匹配；它没有覆盖这个多命中反例，因此不能据其通过声称通用closest-hit。

## 正式结果

```bash
python3 validate_scene.py --gpus 2 6 --ray-suite general --repeat 3 --output results/local-general.json
python3 validate_ray_protocol.py --gpus 0 5 --output results/local-ray-protocol.json
CUDA_VISIBLE_DEVICES=5 timeout 30s python3 run_scene.py overlap --ray-suite general --ray-flags 1 --group both --check
CUDA_VISIBLE_DEVICES=5 timeout 30s python3 run_scene.py grid --count 64 --geometry-opaque --ray-flags 0x10 --ray-suite general --group both --check
```

| 记录 | 结果 |
|---|---|
| [phase14-general-v2.json](../results/phase14/phase14-general-v2.json) | GPU2/6，8场景各跑13/19条组：16 cases、10,416 launches通过。默认ray flags1；geometry flags0；网格最多1024三角形 |
| [phase14-ray-protocol-v2.json](../results/phase14/phase14-ray-protocol-v2.json) | GPU0/5，9组geometry/ray flags/绕序组合：18个TTU cases、9,792 launches通过；另10个instance oracle-only cases通过，共28cases |
| [phase14-cross-group-check.json](../results/phase14/phase14-cross-group-check.json) | 上述共20,208次TTU launch；13/19条组的R72、R73、R32、R33在10,104对ray/trial中逐位相同；每条也分别通过CPU/OptiX容差与front-face比较 |
| [phase14-fixture-regression.json](../results/phase14/phase14-fixture-regression.json) | GPU3固定fixture：8 cases、40 launches通过，包括原20条组 |

ray flags覆盖NONE、DISABLE_ANYHIT、CULL_BACK_FACING_TRIANGLES、CULL_FRONT_FACING_TRIANGLES及其适用组合；geometry flags覆盖0/1，含反转triangle绕序。两种剔除不同时启用；enforce-anyhit、first-hit终止、primitive类别剔除等未实现，CLI拒绝未验证位。性能、full warp和设备内重复仍未测。

`phase14-*-v2`为最终源码矩阵；其他不带v2/initial/sign-check的JSON保留早期探索、错误decoder或中间代码版本。每份保存自身源码hash，不能跨版本混称。

## 静态instance：builder/oracle已通，裸TTU尚未完成

`Scene(..., instances=...)`现在支持静态IAS引用GAS并持有依赖。instance ID为17，双instance第二个为41。恒等、平移、90度旋转、非均匀正缩放、双instance的CPU世界坐标triangle与OptiX结果全部匹配；这是上述10个**oracle-only** cases，不能算TTU instance功能成功。

```bash
CUDA_VISIBLE_DEVICES=5 timeout 30s python3 run_scene.py single --instances scale --ray-suite general --oracle-only --check
CUDA_VISIBLE_DEVICES=5 timeout 30s python3 run_scene.py single --instances identity --group expanded --ray .25 .25 -5 0 0 0 1 100 --check
```

第二条当前预期检查失败，它是未实现续遍历的诊断入口。[正式事件记录](../results/phase14/phase14-ias-event-final.json) 和 [当次IAS字节](../results/phase14/phase14-ias-identity-final.bin) 配套：

- 原固定`base+0x300`对IAS不成立；本例header+0x20的64位偏移为0x380，GAS样本为0x300。runner改为读这个经验字段，并提供 `--root-offset` 覆盖；检查对齐/范围，但仍不声称通用handle解码。
- 正确root下命中instance包围盒，R72:R73作为64位值恰好等于IAS base+0x200；与镜像中的instance记录对应。R33=b0000000，`R33>>27=0x16`，不是triangle的barycentric。
- 原JIT的0xfd0–0x1000也先从R33取高5位并与0x16比较，之后按事件分支。此处只作对照，不把这个类号命名成已完全确定的硬件ABI。
- 13条组足以暴露这个事件，但19条组额外返回了其他状态。尚未解码instance ID、变换/child引用/continuation，也未验证第二次TTU提交；`--check`会明确失败，`ttu_instance_id_decoded`为false。

原JIT普通SASS相关片段与来源hash保存在 [jit16-protocol-ordinary.sass](../reference/jit/jit16-protocol-ordinary.sass) / [manifest](../reference/jit/jit16-protocol-disassembly.json)。为让nvdisasm继续，提取时把TTU槽换成NOP；这只用于读普通指令，**不是可运行的替代traversal**。

下一步优先沿0x16分支读instance记录与child root的构造，先完成一个identity instance的返回/重提交流程并验证ID/t/face，再扩大变换。nonopaque triangle的accept/ignore/resume是另一项协议任务。两者都需要明确终止/续遍历状态，不能靠多发几次GO试到相同t就算通过。
