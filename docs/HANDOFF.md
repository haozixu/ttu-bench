# 当前交接：Phase19单warp行为测量，复杂IAS暂挂

2026-11-04下一步选择：先做范围受限的小renderer，补不同lane的ray与尾warp，比较软件CUDA BVH、裸TTU、OptiX；相机/着色/软件遍历均用CUDA，只有TTU小后端patch。renderer通过后回到热小AS的紧凑多warp准入/竞争测量，不扩复杂IAS。范围/完成判据见 [MEASUREMENT_ROADMAP](MEASUREMENT_ROADMAP.md)。

README位于 `ttu-bench/` 根目录，详细文档集中在 `docs/`；命令从 `ttu-bench/` 执行。概念与协议边界先读 [TTU 协议入门](TTU_PROTOCOL_GUIDE.md)。旧文档快照在 `docs/reference/history/`，恢复源码仍在 `reference/history/source/`。结果已按阶段归档，Phase17起逐项raw合并为单ZIP，早期保留gzip；中间版本在本地被ignore的archive，原字节均保留。路径/hash映射和读取方法见 [RESULTS.md](RESULTS.md)，不要直接按旧matrix路径读取新目录。


**最新先读 [Phase19](PHASE19.md)。** 152项正常矩阵、194784次逐lane检查全通过，另48项匹配NOP/MF0替换全部预期715。新增 `mixed_timing.py` / `run_mixed_timing.py`，不改Phase18源码hash。正常MF不能直接省掉，但可拆成MF9+ST、MF2+GO/LD，或MF10+ST/GO、MF1+LD；组间FADD正确。后者GO后gap64总窗口仍617，显示GO和结果读回可以分离；未验证CILP/抢占。

FADD为普通FP32 ALU。同warp实际HMMA、global指针追踪、shared指针环也能和TTU重叠。HMMA仅等SB5时即时消费者读旧值，不能只看末尾snapshot；正式版每条>=32 scheduled cycles，计时结束前核对四个消费者。TTU C只等SB1，之后立即copy结果核对；内存SB2另等。128 registers/thread、0 stack/local/static shared，launch有256B dynamic shared，别漏报。测的是带padding/消费的工作包，不是HMMA精确延迟或RT纯延迟。

Phase17：47份→7个物理文件；Phase18：179份→5个；Phase19：211份→5个。旧路径通过manifest aliases/`result_store.py show`解析，原字节/hash不变。每个ZIP只校验一次完整hash（stat变化使缓存失效），成员另校验；共有642份原始记录、543份发布报告、123个发布物理文件，约14.92 MiB，不含manifest及早期未入catalog的小样例。不要直接读取旧matrix里已不存在的单文件路径。

功能线看 [Phase17](PHASE17.md)：`parent_template.cu` 在一个kernel内保存硬件返回的pending引用、查询child、恢复父IAS。平坦1–3个正行列式静态instance、共享triangle GAS、ray flags1/geometry flags0、相对引用类型0x21；两GPU的30项、4,768次launch、11,492次请求（1,484次恢复父级）全通过。两个first-child-only负对照分别错误报告miss或保留更远交点。四实例及其他引用/返回类型仍不支持，不是完整IAS/JIT协议。

**重要纠正：E-0x0c不是通用instance ID，E-0x20不是通用child handle。** 从 `[E+4]` 取得1-based索引，IAS base+0x30取得表偏移，步长0x80，元数据+0x74低24位为ID、+0x60为handle。矩阵仍在E+0x10，child root仍为[E+8]低49位。旧Phase15/16报告与hash保留，现行decoder已修正。正式多实例报告直接包含相邻ID错误反例。

入口为 `run_parent_resume.py` / `validate_parent_resume.py`，场景在 `fixtures/phase17-parent-cases.json`。`parent_resume.py`提供raw transport及ABI，`device_instance.py`保留默认单instance接口并支持按kernel选择marker。parent_probe有88 registers、848字节stack、7856字节结果trace，只有一个active thread，不能直接计时。单instance回归8项/1,344次launch、2,615次请求及固定fixture8项/40次launch均通过。新正式结果在results/phase17，47份原始结果现为7个物理文件，其中41份在records.zip。

[Phase15](PHASE15.md)保留host参考及4096三角形GAS；[Phase16](PHASE16.md)保留独立child阶段。普通JIT新增 `reference/jit/jit16-parent-*` 保存/恢复分支，TTU槽已换NOP，仅供分析。不要丢弃已成功的受限路径，也不要把它推广为任意AS。

更新：2026-11-03。先读 [README.md](../README.md) 和 [PHASE12.md](PHASE12.md)，用 `run_minimal.py hit/miss --check` 验证环境。当前功能基线不依赖 OptiX 启动、驱动调度现场、shared/local 栈或 EoSS Python 包。

前阶段 [Phase14](PHASE14.md)：一般方向/tmin/opaque/culling flags共20208次TTU launch通过，13/19条组主要返回逐位一致。`run_scene`默认ray flags1；flags0的nonopaque多命中可返回较远候选，不能视为完整closest-hit。R73/R32符号承载额外信息，t/u取幅值；geometry flag1旧负结果已纠正。IAS事件与ID/变换字段以后续Phase17纠正为准，父级续遍历已实现受限子集。

前阶段 [Phase13](PHASE13.md)：GPU5/7上单/双三角形与最大1024网格，共16case/980launch通过。`both`比较19条expanded与13条reduced；19条保留所有LD但静态消除P4分支。原20条组在实时AS上有CUDA700，固定fixture回归仍通过，不能当作通用可靠控制组。原flag1负记录已有上述解码纠正。

公开线索：[NVIDIA complet/node专利](../../research/notes/nvidia-bvh-public-records.md)，两份PDF已下载。复杂IAS协议逆向暂挂，四实例新返回路径、等距边界及nonopaque accept/ignore保留为后续问题。当前主线是Phase19单warp行为测量。无需先独立构造全部BVH才能推进。

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
| Phase18已测1/32 active lanes | 每lane独立结果、相同ray；尚无发散ray、饱和吞吐或RT buffer容量结论 |
| 单warp时间窗口与同warp异步重叠 | Phase18有设备内展开重复、时钟和ALU位置对照；没有纯RT延迟、饱和吞吐或跨warp竞争结论 |

硬件可能仍有内部状态/缓冲；自研 kernel 没有显式 shared 分配，不能推出 RT 不使用内部存储，也不能直接否定 OptiX 的 shared 开销观测。

## 必须保留的协议事实

- TTULD 第一个目的寄存器字段为绝对 bits23:16；第二个在 bits71:64。此组允许 R2、R50、R54、R58，旧“必须 4 对齐”错误。
- TTUST 有两个显式源寄存器对，起始寄存器分别在 bits39:32 和 bits71:64；不要忽略第二对。
- triangle hit观察到8/a两种tag，低位对上primitive；`[abs(FP32(R32)),FP32(R33)]`对上barycentrics，t=abs(FP32(R73))。R32符号对上front/back，R73符号与opaque/候选路径相关。miss为R72=R73=`ffffffff`；未知事件另分类，不能把IAS指针当triangle ID。完整位宽与tag含义仍未知。
- 精简组最后的 TTULD.CLOSE 位是绝对 bit74（第二个 64-bit word 的 bit10）；MF count 是 OPEN/MF 之后的 11 条 TTU 指令。
- 地址、配置常量和输出解释仅在该 fixture/ray 族上验证，不能笼统命名为全部 535 或 SM86 唯一协议。
- 裸 kernel 的结果等待已放在读回之前。任何 TTULD 重映射、循环或计时改动都必须复验依赖与 scoreboard。

## 后续顺序与完成判据

**复杂IAS功能线暂时挂起**，恢复时从四实例最小反例继续：从 `results/phase17/pop-four.json` 的0x24引用类型与直接triangle返回出发，明确坐标、instance归属和父状态，不要仅提高count限制。另用隔离实验研究等距命中的身份选择及tmax端点；`device-chain2-overlap` 的51项检查失败（29条TTU记录、22项CPU/OptiX比较）已保留，不能放宽oracle后称为全部通过。当前正式矩阵排除并列命中身份问题。嵌套IAS、不同child GAS、负缩放、motion和用户any-hit仍是后续任务。

Phase17受限路径已通过原计划的“先miss后hit”“先远后近”和输入顺序交换；恢复引用来自TTU，不能退回逐个枚举builder输入来冒充协议续遍历。`parent_slot_nonzero_raw`仍只是旧R51诊断，不能当作pending解码；新路径检查R55数量及显式条目类型。

当前 [Phase18](PHASE18.md) 已完成一个warp、1/32 active lanes的独立输出、每次恢复输入、4次设备内展开请求、四时间戳及ALU/OPEN-gap对照。两GPU172项、2064次launch、6144次warp请求、136224次逐lane检查通过。单lane/13条窗口381周期，32条FADD放等待前仍381、等待后509；同warp可在异步请求未完成时执行独立计算。不能把完整窗口当纯RT延迟，当前逐次snapshot还不能测饱和吞吐。

MF0与匹配NOP相同；OPEN-gap可隐藏部分GO前等待。NVIDIA US11138009B2描述MF等待OPEN资源并屏蔽有限组的抢占、结果另用scoreboard等待，与观察相容，不是本机CILP实测。Phase19已补拆组、MF替换负对照和Tensor/访存位置矩阵。下一步优先查HMMA依赖、减少快照扰动，再引入第二warp。下面保留测量检查顺序，部分基础项已由Phase18完成。

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
