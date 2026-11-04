# 文件索引

更新：2026-11-03。入口是 [README.md](../README.md)，初次阅读先看 [TTU 协议入门](TTU_PROTOCOL_GUIDE.md)。当前协议边界见 [PHASE14.md](PHASE14.md)，基础协议见 [PHASE12.md](PHASE12.md)，独立指令行为、差异与恢复记录见 [EOSS_REPRO.md](EOSS_REPRO.md)。

当前主线为 [Phase19](PHASE19.md) MACROFUSE行为、同warp Tensor/global/shared重叠；复杂IAS协议逆向暂挂，Phase17证据和待解路径保留。

2026-11-04的后续工程顺序见 [MEASUREMENT_ROADMAP](MEASUREMENT_ROADMAP.md)：先小renderer补不同lane和三后端比较，再回多warp。这是计划，不是新增实验结果。

入口README位于 `ttu-bench/` 根目录，详细文档集中在 `docs/`。下表源码、数据和结果的行内路径以 `ttu-bench/` 为基准；所有复现命令也从该目录执行。文档点击链接已按新位置更新。

功能基线 [Phase17](PHASE17.md)：保存并重新提交硬件返回引用，已测平坦1–3静态instance的父IAS续遍历通过两GPU的30项矩阵；仍限共享GAS及0x21引用类型。单instance相邻ID/handle规则已撤回，改用事件索引查IAS表。[Phase15](PHASE15.md)保留记录解码、host参考实现及4096三角形GAS单GO测试。

## 可执行代码

| 文件 | 用途与依赖 |
|---|---|
| `ttu_isa.py` | TTU 族16字节编码、字段读取/修改、融合组count及输入寄存器分析；标准库 |
| `eoss_helpers.py` | 本地 EoSS 数据与 BinaryPatcher/control 适配，REPLACE_FULL 保留操作数和调度位 |
| `cubin_builder.py` | 普通 SASS / 单入口 CUDA ELF patch 和元数据更新；cuda_driver |
| `sm86_fixture.py` | 原20条/精简13条组、hash、root/ray输入；fixture知识与通用ISA分离 |
| `build_and_patch.py` | 重建 EoSS exp1–5，使用共享构建器和 EoSS 完整替换；非原源码恢复 |
| `run_exp.py` | EoSS复现；每trial新进程，记录fault/context毒化/超时与hash；非几何oracle |
| `validate_encoding.py` | 原20指令round trip、精简hash、字段边界、EoSS保真patch、历史源码hash检查 |
| `run_minimal.py` | 主入口；读取 fixtures，分配 CUDA 内存，合成输入，精简/保留原始 TTU 组，提供解析求交检查 |
| `minimal.py` | 组装功能probe、发射已构建image和读回；共用ttu_isa/cubin_builder/cuda_driver |
| `minimal_template.cu` | nvcc 编译容器；源中的 FADD 链不会执行 |
| `cuda_driver.py` | 当前代码使用的 Driver API 函数签名与 ELF section 解析；没有 megakernel 或 OptiX 初始化 |
| `validate_minimal.py` | 8 个场景 × 指定 GPU × 每场景 5 次 launch；每场景独立进程，新 context，20 秒超时 |
| `optix_scene.py` / `scene_bridge.cpp` | 在调用者当前CUDA context内构建/持有triangle GAS并提供OptiX oracle；首次编译需OptiX8 SDK/nvcc/g++ |
| `scene_oracle.cu` | 最小OptiX raygen/closest-hit/miss，返回primitive/t/barycentrics |
| `run_scene.py` | 生产GAS功能入口，CPU+OptiX+裸TTU三方比较；默认13条组，both比较19/13条组，原20条组单列诊断 |
| `validate_scene.py` | 单/双三角形及16/64/257/1024网格回归；每case新进程，保存raw寄存器与源码/AS/group hash |
| `validate_ray_protocol.py` | geometry/ray flags、绕序的TTU回归；另测5种静态IAS的CPU/OptiX oracle，明确不算TTU实例成功 |
| `instance_protocol.py` | 0x16静态instance事件记录的有边界解码、逆变换ray；经验布局，非完整续遍历ABI |
| `run_instance_probe.py` | 一个静态instance的真实IAS事件、host解码及独立child GAS查询；验证矩阵/ID/child root，明确不是父级遍历resume |
| `device_instance.py` | 18个寄存器对的编译器分配映射、19条TTU组patch、host/device结构及CUDA launch；功能容器，非计时核 |
| `instance_template.cu` | 设备端IAS事件分派、记录边界检查、逆变换及独立child查询；不恢复父级 |
| `run_device_instance.py` | 单静态instance设备功能入口，保存两个请求的raw返回并核对CPU/OptiX及host事件解码 |
| `validate_device_instance.py` | 8项device-instance回归，逐项新进程/context、40秒timeout、报告hash |
| `parent_resume.py` / `parent_template.cu` | raw state transport与受限父IAS设备续遍历，显式pending软件栈；单active thread，非计时核 |
| `run_parent_resume.py` / `validate_parent_resume.py` | 最终及逐事件CPU/OptiX检查；30项正矩阵与两个first-child-only负对照 |
| `run_parent_probe.py` | host控制raw状态的none/copy/promote/pop诊断；不是通用算法或正式closest-hit入口 |
| `warp_timing.py` / `timing_template.cu` | 单warp raw SASS测量构建及逐lane快照，96寄存器，无stack/shared；保持融合组body不变 |
| `run_warp_timing.py` / `validate_warp_timing.py` | clock64四窗口、1/32 lanes、ALU位置、OPEN-gap和MF0/NOP校准；正式172项矩阵 |
| `disassemble_ttu.py` | 保存源hash/原TTU字节，将TTU换NOP后反汇编普通指令片段；输出不可运行 |
| `mixed_timing.py` / `run_mixed_timing.py` | Phase19：128寄存器、256B dynamic shared、独立scoreboard、即时消费者及HMMA保守padding |
| `validate_mixed_timing.py` | 152项单warp Tensor/global/shared/FADD及拆组对照，两GPU；独立process/timeout |
| `run_macrofuse_control.py` / `validate_macrofuse_control.py` | 匹配NOP/MF0替换正常MF的48项预期715负对照，保留ST/GO/LD.CLOSE |
| `inspect_mixed_timing.py` | 从实测cubin记录SASS、来源hash及融合范围，TTU换NOP的输出只供分析 |
| `bundle_results.py` / `publish_results.py` | 原字节合并及新阶段ZIP发布，矩阵引用hash验证 |
| `validate_result_store.py` | 旧alias、ZIP损坏拒绝、恢复越界及不同内容覆盖拒绝 |
| `result_store.py` | 按manifest读取原始/新路径的结果、校验压缩及原始SHA256、恢复原始目录结构；标准库 |

`minimal_template.cubin` 和 `__pycache__/` 是自动生成的缓存。功能入口不需要 EoSS；EoSS复现/编码验证入口使用已有 `research/community/eoss -> eoss-framework` 符号链接。

`scene_bridge.so` / `scene_oracle.ptx` 同样是生成缓存。`run_scene.py` 使用OptiX builder/oracle，`run_minimal.py` 的固定fixture入口仍不需要OptiX。

## 固定输入

| 文件 | 当前使用方式 |
|---|---|
| `fixtures/optix_as.bin` | 2432B 单三角形 AS；每次运行修正 +0xd0 自指针；runner 校验原文件 SHA256 |
| `fixtures/jit_16.elf` | 取 traversal_mttu+0xe70 的 320B TTU 组，以及 .nv.global.init 前 128B 的 dummy root；整个镜像不被加载执行 |
| `fixtures/phase17-parent-cases.json` | 静态instance变换、反例ray及输入顺序变体；报告保存文件hash |
| `fixtures/manifest.json` | 原始文件大小、SHA256、获取背景与已验证用途 |

固定输入并非通用 AS builder。更换几何或驱动镜像，需要重新验证布局、root 打包和功能 oracle，不能仅移除 SHA256 检查。

## 实验证据

结果已按 `results/phase12/` 至 `results/phase19/`、`results/eoss/`、`results/maintenance/` 分类。Phase17起单ZIP保存逐项报告，较早阶段保留gzip；内部路径/command及hash仍为原始记录，不改写历史结果。用 [manifest](../results/manifest.json) 或 `result_store.load_result()`解析旧路径。中间版本保存在被ignore的 `results/archive/`，clone不含这些本地归档；正式结论依赖的矩阵、逐case数据和关键反例均在发布集。见 [RESULTS.md](RESULTS.md)。

最新 [Phase19](PHASE19.md)：[v3矩阵](../results/phase19/final-v3-mixed-matrix.json)152项、194784次逐lane检查；[48项MF替换负对照](../results/phase19/final-mf-controls.json)均预期715。211份原报告/矩阵/摘要集中成5个物理文件，207个成员在records.zip。提供Tensor/global/shared真实SASS及立即消费检查；HMMA精确依赖仍待解读。

此前 [Phase18](PHASE18.md)：[单warp v2矩阵](../results/phase18/final-v2-warp-matrix.json)及[records.zip](../results/phase18/records.zip)中的172份case和3份校准/旧失败，记录2064次launch、6144次warp级请求和136224次逐lane检查；同warp异步重叠、OPEN-gap、MF0与匹配NOP校准均通过。资源、sanitizer、快照源寄存器竞争的旧失败证据一并入manifest。

此前 [Phase17](PHASE17.md)：[parent矩阵](../results/phase17/final-parent-matrix.json)保存30项、4,768次launch、11,492次请求及两个负对照，逐case与负对照共41份在[records.zip](../results/phase17/records.zip)；阶段共7个物理文件。另有单instance/fixture回归、资源、代表事件、四实例与等距边界记录；全部加入manifest。

此前 [Phase16](PHASE16.md)：[device矩阵](../results/phase16/phase16-device-matrix.json)及同目录下`phase16-device-{instance}-{scene}.json.gz`保存8项、1344次kernel launch/2615次TTU请求，全通过；[代表性事件](../results/phase16/example-events.json)为方便阅读的派生样例。`phase16-device-resource-usage.txt`为单独构建的资源报告；`phase16-fixture-regression.json`为8case/40launch固定fixture回归。设备端独立child查询不是父级resume。

此前 [Phase15](PHASE15.md)：[instance矩阵](../results/phase15/phase15-instance-matrix.json)包含6场景、928次IAS查询和874次独立child查询，全通过；[4096三角形GAS](../results/phase15/phase15-grid4096.json.gz)为136条ray×13/19组，共272次launch通过。实例host处理与独立查询不计作裸TTU续遍历成功。

此前 [Phase14](PHASE14.md)：`results/phase14/phase14-general-v2.json`及对应cases目录保存16case/10416launch；`results/phase14/phase14-ray-protocol-v2.json`保存18个flags/winding TTU case、9792launch，另10个IAS oracle-only case。`results/phase14/phase14-cross-group-check.json`记录10104对主要返回逐位相同。`phase14-*-v2`是最终源码矩阵；早期initial/sign-check等记录按自身源码hash理解。

负结果：`phase14-nonopaque-candidate.json`为flags0较远候选反例；`phase14-ias-event-final.json`与`phase14-ias-identity-final.bin`配套，保存root0x380下的instance相关事件；`phase14-full-opaque-request.json`表明ray flag1仍不能修复20条组。`phase14-fixture-regression.json`为8case/40launch固定fixture回归。

| 文件 | 含义 |
|---|---|
| `results/phase12/phase12-functional.json` | 原始功能回归：GPU 5/7，16 case，80 launch；有 ray 参数、期望结果、实际寄存器值和 group hash |
| `results/phase12/phase12-negative-controls.json` | 跳过 TTU、清零 dummy 指针两种变体都不能通过基础 hit 检查 |
| `results/phase12/phase12-gpus.csv` | 原验证设备的 index、UUID、架构和驱动 |
| `results/phase12/phase12-resource-usage.txt` | 80 registers，0 local/shared/stack 的 cuobjdump 报告 |
| `results/phase12/phase12-artifact.json` | 单次基础 hit 的运行记录，不是可直接重放的 cubin |
| `results/maintenance/cleanup-functional.json` | 清理依赖和迁移 fixtures 后的同一功能回归 |
| `results/maintenance/helpers-functional.json` | helpers提取后GPU5，8场景40次launch通过 |
| `results/eoss/eoss-reconstructed-sm86-nowait.json` | 重建v2的EoSS基础复现；GPU7，7变体各3次，含预期双OPEN715 |
| `results/eoss/eoss-basic-final.json` | 最终入口21项预期状态检查全部通过，保存源码hash、TTU hash、驱动/设备信息 |
| `results/eoss/ttu-sm86-sm120-layout.json` | 两代8个TTU类完整位域与opcode、按alias归一化比较、condition表达式对照及源文件hash；仅静态证据 |
| `results/phase13/phase13-scenes.json` / `phase13-scenes-cases/` | GPU5/7生产GAS矩阵，16case/980launch；19条expanded与13条reduced均核对hit/ID/t/barycentrics |
| `results/phase13/phase13-baseline-regression.json` | R50地址打包修正后的固定fixture回归，16case/80launch，含原20条组 |
| `results/phase13/phase13-full-predicated-negative.json` | 实时双三角形GAS上原20条组在3次成功后CUDA700，根因未定 |
| `results/phase13/phase13-opaque-flag-negative.json` | geometry flag=1，OptiX/CPU正确而当前13条TTU返回t不匹配 |
| `results/phase13/phase13-decoder-a-only-negative.json` | 初版decoder错误拒绝257三角形场景的8-tag；原始寄存器保留，hit/ID/t/bary实际匹配 |
| `results/eoss/eoss-reconstructed-sm86-lane1.json` | 重建v1的18次基础结果和CCTL写scoreboard后wait-all的超时；明确不是完整矩阵 |
| `results/eoss/eoss-cctl-ivall-wr0-nowait.json` | GPU5，IVALL+写scoreboard0、无后续wait，首项超时后停止 |

`--artifact PATH` 生成的 cubin 含当次进程 VA，只能核验编码/资源，不能在另一进程直接重放。已删除此前保存的这种 cubin；需要时由 runner 重新生成。

## 参考资产

Phase17新增 `reference/jit/jit16-parent-ordinary.sass` / `jit16-parent-disassembly.json`，补齐返回引用、软件栈保存/恢复及重新提交分支。仅供分析，不可运行。

新增`reference/jit/jit16-protocol-ordinary.sass`与`jit16-protocol-disassembly.json`：ray/flags构造和返回分支的普通SASS片段、源hash与移除TTU的manifest。仅供反汇编分析，不可运行。

| 路径 | 保留理由 |
|---|---|
| [历史记录说明](reference/history/README.md) | 旧Markdown原样保存在 `docs/reference/history/`；恢复源码仍在 `reference/history/source/`。含旧错误解释，只作历史证据 |
| `reference/jit/jit_*.elf` | 其余 20 个原始驱动 JIT 镜像，供不同遍历路径与编码比较；当前 runner 不使用 |
| `reference/optix/optix_min.cu` | 保存 fixture 的 OptiX device 程序参考 |
| `reference/optix/build_fixture.cpp` | 构建同一几何的 OptiX 管线与 AS；已移除错误描述符猜测和无效扫描分支 |
| `reference/vulkan/vk_bvh_dump.c`、`as_dump_0/1/2.bin` | Vulkan 两三角形 BLAS 生成参考及不同 VA 的原始镜像，仅供格式比较 |
| [参考资产说明](reference/README.md) | 来源、用途边界和生成参考的用法；资产仍在 `reference/` |

参考生成器不保证重建出逐字节相同的镜像；尤其 JIT 原始快照不能由目前的 AS 生成器直接再生，因此保留原始数据。

## 文档

[TTU_PROTOCOL_GUIDE.md](TTU_PROTOCOL_GUIDE.md) 从ray、BVH、GAS/IAS和instance讲到flags、barycentrics及裸TTU请求/返回协议；区分公开概念与本机实验结论。

文档迁移前路径与SHA256保存在 [document-migration.json](document-migration.json)；四份 `.pre-cleanup.md` 快照逐字节保留，因此快照里的旧链接和旧结论也保持原状。

[HANDOFF.md](HANDOFF.md) 跟踪后续任务，[NEXT_PROMPT.md](NEXT_PROMPT.md) 作短入口。[CLEANUP.md](CLEANUP.md) 记录原清理及其纠正；`cleanup-2026-11-02.json` 保留当时删除清单，不改写成当前文件清单。恢复身份、API和差异证据集中在 EOSS_REPRO。

[AS_ROADMAP.md](AS_ROADMAP.md) 评估代际差异来源、复杂AS/完整OptiX语义的路线与成本，并复核四份未恢复旧文件。当前以微架构探索/工具/blog为合理成果。

[PHASE13.md](PHASE13.md) 记录生产GAS进展；较早schema的`scene-*.json`已收在本地 `results/archive/exploratory/`，不代替正式矩阵。NVIDIA专利、软件CWBVH及AMD硬件格式的适用边界见 [公开AS资料核查](../../research/notes/nvidia-bvh-public-records.md)。
