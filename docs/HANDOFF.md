# 当前交接：SM86单instance设备端路径已通，父级恢复待解

README位于 `ttu-bench/` 根目录，详细文档集中在 `docs/`；命令从 `ttu-bench/` 执行。概念与协议边界先读 [TTU 协议入门](TTU_PROTOCOL_GUIDE.md)。旧文档快照在 `docs/reference/history/`，恢复源码仍在 `reference/history/source/`。结果已按阶段归档，大型raw日志为无损gzip；中间版本在本地被ignore的archive，原字节均保留。路径/hash映射和读取方法见 [RESULTS.md](RESULTS.md)，不要直接按旧matrix路径读取新目录。

最新先看 [Phase16](PHASE16.md)：`instance_template.cu`已在一个kernel内实现IAS请求→0x16事件分派→逆变换→独立child查询。8项矩阵、1344次kernel launch、2615次TTU请求全通过，含541次instance事件后child miss。`device_instance.py`把编译器分配的18个寄存器对映射到19条expanded组；`run_device_instance.py`保存raw返回并做CPU/OptiX及host事件解码对照。`validate_device_instance.py --gpus 1 2 --output results/local-device-matrix.json`可复现。这个功能容器有288字节stack和搬运，不能作为最小计时核；尚未恢复父级IAS状态。

[Phase15](PHASE15.md)保留逆矩阵、ID、child引用的字段证据和1802次host参考查询。普通JIT矩阵运算在0x47d0–0x4970，child root构造在0x60d0–0x64b0，证据在reference/jit/jit16-instance-*。不要重复从零猜字段；下一步是父级保存/恢复和多instance反例。另：4096三角形GAS的272次launch已通过，1024只是原最大测试规模。

更新：2026-11-02。先读 [README.md](../README.md) 和 [PHASE12.md](PHASE12.md)，用 `run_minimal.py hit/miss --check` 验证环境。当前功能基线不依赖 OptiX 启动、驱动调度现场、shared/local 栈或 EoSS Python 包。

前阶段 [Phase14](PHASE14.md)：一般方向/tmin/opaque/culling flags共20208次TTU launch通过，13/19条组主要返回逐位一致。`run_scene`默认ray flags1；flags0的nonopaque多命中可返回较远候选，不能视为完整closest-hit。R73/R32符号承载额外信息，t/u取幅值；geometry flag1旧负结果已纠正。IAS事件与ID/变换字段已有Phase15/16推进，父级续遍历仍未实现。

前阶段 [Phase13](PHASE13.md)：GPU5/7上单/双三角形与最大1024网格，共16case/980launch通过。`both`比较19条expanded与13条reduced；19条保留所有LD但静态消除P4分支。原20条组在实时AS上有CUDA700，固定fixture回归仍通过，不能当作通用可靠控制组。原flag1负记录已有上述解码纠正。

公开线索：[NVIDIA complet/node专利](../../research/notes/nvidia-bvh-public-records.md)，两份PDF已下载。近期功能任务为父级IAS恢复；nonopaque triangle的accept/ignore另线处理。测量顺序仍按下文。无需先独立构造全部BVH才能推进。

EoSS 指令行为复现与 SM86/SM120 delta 是另一条需要保留的实验线，见 [EOSS_REPRO.md](EOSS_REPRO.md)。`run_exp.py` / `build_and_patch.py` 已重建，TTU helpers、EoSS适配、CUDA ELF patch、fixture输入已拆为共用组件。原20指令round trip及8场景40次功能回归通过。旧交接/README在 `docs/reference/history/`，部分原源码在 `reference/history/source/`，不能把旧解释直接当作已证事实。

两代TTU静态编码比此前判断更接近：SM120第二寄存器字段仍在bits71:64，只是内部键名换成Ra_URc；8个类的opcode、归一化字段布局、condition表达式均一致。不同驱动遍历家族的组结构和计数器可见性仍值得记录；CCTL scoreboard超时尚待归因。EoSS canned bit72与两代Rosetta bit74不一致，不能叫作架构差异。详见EOSS_REPRO与新增 [AS_ROADMAP.md](AS_ROADMAP.md)。

## 已证明与尚未证明

| 事实 | 证据 / 限定 |
|---|---|
| 新 CUDA context 直接执行真实 TTU traversal | 基础 hit、空间 miss、改变 oz / dz / ox / tmax 均通过解析求交 |
| 可用 13 条 TTU 取代保存的 20 条原组 | 保留 9 ST、1 GO、1 LD.CLOSE；修正 MF count；两种组均通过功能回归 |
| dummy root 必须有效 | R50:R51 清零对照由 hit 变 miss；目前自行分配并编码，不是外核寄存器现场 |
| 该 fixture 可用高 CUDA VA | 普通 cuMemAlloc + AS 自指针修正，无低 4GB VMM 需求 |
| 80 registers，0 local/shared/stack | cuobjdump 报告；只是当前自研 kernel 的公开资源声明 |
| 单 active lane 的功能正确 | 不代表 full-warp 的性能或 RT buffer 容量 |
| 尚无争抢/延迟结果 | 80 次功能 launch 不等于吞吐实验；没有同 kernel 循环或设备内计时 |

硬件可能仍有内部状态/缓冲；自研 kernel 没有显式 shared 分配，不能推出 RT 不使用内部存储，也不能直接否定 OptiX 的 shared 开销观测。

## 必须保留的协议事实

- TTULD 第一个目的寄存器字段为绝对 bits23:16；第二个在 bits71:64。此组允许 R2、R50、R54、R58，旧“必须 4 对齐”错误。
- TTUST 有两个显式源寄存器对，起始寄存器分别在 bits39:32 和 bits71:64；不要忽略第二对。
- triangle hit观察到8/a两种tag，低位对上primitive；`[abs(FP32(R32)),FP32(R33)]`对上barycentrics，t=abs(FP32(R73))。R32符号对上front/back，R73符号与opaque/候选路径相关。miss为R72=R73=`ffffffff`；未知事件另分类，不能把IAS指针当triangle ID。完整位宽与tag含义仍未知。
- 精简组最后的 TTULD.CLOSE 位是绝对 bit74（第二个 64-bit word 的 bit10）；MF count 是 OPEN/MF 之后的 11 条 TTU 指令。
- 地址、配置常量和输出解释仅在该 fixture/ray 族上验证，不能笼统命名为全部 535 或 SM86 唯一协议。
- 裸 kernel 的结果等待已放在读回之前。任何 TTULD 重映射、循环或计时改动都必须复验依赖与 scoreboard。

## 后续顺序与完成判据

功能线先解**父级IAS恢复**：沿JIT保存/恢复分支对应返回槽和下一次请求输入；用两个instance验证第一child miss而后续hit、第一child hit而后续更近、交换输入顺序。保存原始事件顺序，不能把builder输入顺序当成遍历顺序。不要以逐个instance独立查询取min冒充硬件父级resume；`parent_slot_nonzero_raw`只是R51非零诊断，尚不是“存在待处理兄弟节点”的解码。详见Phase16末尾。

下面为独立的测量线顺序；Phase16的连续两次请求只证明功能，不满足这些计时判据。

1. **先形成正确的重复 TTU 测量。** TTULD 当前覆盖 R72:R73、R32:R33，它们同时是下一次 TTUST 输入。先分离输入/输出，或明确恢复每次被覆盖的输入；不能直接在现序列上加循环。用独立几何 oracle 检查循环尾部结果。
2. **加入设备内计时与对照。** 明确测 TTUGO 到结果可用、完整 TTU group，还是连续请求吞吐。排除准备、global 写回与 launch；计时指令本身也可能争抢资源，校准空序列与依赖等待。当前 Python 时间和原始 OptiX API 时间不能互换。
3. **降低寄存器占用，验证 active lanes / full warp。** 保存未改组对照，紧凑重映射后检查输出，再扫 active lanes、RT warp 数及驻留条件。Ray-by-Ray 已做相近实验，这一步主要服务测量校准。
4. **同 CTA 的选择性干扰矩阵。** 不同 warp 分别承担 TTU 和 FFMA/INT/S2R/shared/LDG；核验同 SM、工作确实重叠、寄存器/shared/occupancy 可比。记录两方向吞吐及运行时钟。
5. **分离局部与下游资源。** 从热小 AS 开始，再扩展 BVH / working set / ray coherence，做不同 SM 对照。只有存在可重复、可否证的差异，才讨论具体端口、队列或 MSHR 归因。

Phase13扩展了几何功能，没有新增性能测量。不要预先把测到的减速命名为 MIO 或 L1 争抢。

## 当前主线与历史资料

完整 megakernel 的 callback/yield、私有VMM/heap扫描和hook目前不列为最小ISA测量的前置任务。旧e7常量不能当作miss oracle；但对应观察、工具知识和驱动行为可作为独立研究记录保存。已有原始JIT/AS和恢复的历史文件可供回看，勿再把“非当前主线”或“缺几何oracle”作为一并删除编码helpers/复现记录的理由。

旧诊断中的两个根本问题是 ray 参数实际没更新，以及 local 模式仍 NOP 了 88 个 spill 槽；寄存器中继还存在等待/覆盖问题。应保留这些方法教训，而不是旧代码中的“永久不可达”“上下文缺失”“路径判死”等结论。

## 文献约束

Ray-by-Ray §III 已在 RTX 5090 上做单 SM 的 lane/warp 资源实验，报告在 RTX 3070 复现结论；Fig.2 的 12-warp 分界是其所报告配置的观测，不能直接当成 RTX 3090 常量。§III-B 是 shared 活动相关性和分配观察，§III-C 的 lane 延迟包含 traceRay/setup 成本。

RTSpMSpM §4.5 已用 RTX 4090 应用研究讨论共享 L1、访存放大和 RT/SIMD 竞争；其 Fig.9 的 12.37× 是替换 shader 后的复合收益，不是纯 L1 竞争惩罚。

研究空间见 [rt-cuda-research-gap.md](../../research/notes/rt-cuda-research-gap.md)。当前是早期业余微架构探索，也可以写成复现/逆向blog；无需以论文贡献作为继续的门槛。ISA隔离、正确复现、错误诊断和工具知识本身就是阶段成果。

## 工作纪律

选择空闲 GPU；fault 变体独立进程加 timeout；不为实验重置共享 GPU。保存 ray 参数、fixture/group hash、实际输出、设备 UUID 和实验配置。更换协议或输入先过功能 oracle，再看性能。

有效固定输入在 fixtures，其他原始镜像在 reference，文件清单见 [INDEX.md](INDEX.md)。不直接修改 fixture；新 AS 另存并建立新的几何 oracle。验证输出使用新文件名，避免覆盖已保存的实验依据。
