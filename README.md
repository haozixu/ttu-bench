# TTU 微基准：SM86 指令复现与功能基线

代码已按用途整理：`src/` 是ISA、patch、fixture和probe等可复用模块，`tools/` 是实验/验证/结果维护入口，`kernels/` 是CUDA/C++源码，自动生成文件统一在ignored的 `build/`。根目录统一入口为 `python3 bench.py <命令> [参数]`，例如 `python3 bench.py run_minimal hit --check`；`python3 bench.py --help`列出全部命令。无需安装Python包。直接导入模块时，在项目根设置 `PYTHONPATH=src:tools`。

本README位于 `ttu-bench/` 根目录，详细文档集中在 `docs/`。第一次阅读建议先看 [TTU 协议入门](docs/TTU_PROTOCOL_GUIDE.md)，再查 [文件索引](docs/INDEX.md) 和 [当前交接](docs/HANDOFF.md)。本文及其他现行文档中的命令均从 **`ttu-bench/` 项目根目录** 执行；行内代码形式的源码、fixture、结果路径也以该目录为基准，点击链接则按文档位置解析。研究文献仍在 `research/`。

实验结果按阶段收在 `results/phase12/` 至 `results/phase19/`，EoSS结果在 `results/eoss/`。正式结果与关键反例供Git提交，Phase17起逐项报告合并为每阶段一个 `records.zip`，旧阶段保留 `.json.gz`；中间/探索与本地试跑记录保留在被ignore的 `results/archive/`。原始路径、SHA256和现位置见 [manifest](results/manifest.json)，阅读/恢复及提交策略见 [结果归档说明](docs/RESULTS.md)。

最新 [Phase19](docs/PHASE19.md) 已测试具体MACROFUSE行为，以及同warp的Tensor/global/shared重叠：152项正常对照、194784次逐lane检查全通过；48项匹配NOP/MF0替换均如预期报715。一个请求可以拆成多个合法融合组，GO之后、LD.CLOSE之前也能运行普通计算。所有工作在计时结束前都核对即时消费者，HMMA另加保守调度间隔；尚未验证抢占保护或饱和争抢。Phase17/18的226个结果文件现无损合并为12个物理文件，旧路径通过loader兼容。

```bash
CUDA_VISIBLE_DEVICES=5 timeout 25s python3 bench.py run_mixed_timing --lanes 32 --kind hmma --count 8 --check --output results/local/tensor-before.json
CUDA_VISIBLE_DEVICES=5 timeout 25s python3 bench.py run_mixed_timing --lanes 32 --variant split10 --gap 64 --check --output results/local/go-gap-read.json
```

功能基线 [Phase17](docs/PHASE17.md)：一个自研CUDA kernel已能保存硬件返回的待处理引用，查询child后恢复父IAS，完成已测1–3个静态instance的最近命中。两张3090上30项、4,768次launch、11,492次TTU请求通过CPU/OptiX对照，包含“先miss后hit”和“先远后近”的反例及负对照。**仅覆盖平坦IAS、共享triangle GAS、已识别的0x21引用类型；四实例新路径、嵌套、any-hit等尚未实现。** 同时纠正单instance中碰巧成立的ID/handle相邻偏移。4096三角形GAS仍只是最大已测试规模，不是容量上限。

```bash
CUDA_VISIBLE_DEVICES=6 timeout 40s python3 bench.py run_parent_resume --scenario first-farther-later-nearer --check --output results/local/parent-nearer.json
python3 bench.py validate_parent_resume --gpus 2 6 --output results/local/parent-matrix.json
```

在 RTX 3090（GA102 / SM86、driver 535.309.01、CUDA 12.1）上，以自行 patch 的 CUDA kernel 发射 TTU 指令，绕开运行时 OptiX pipeline、shader 回调和驱动 megakernel。当前目的是为 RT/CUDA 同 SM 资源交互研究提供可控的发射与读回基础。

**功能已验证：固定单三角形，以及实时GAS的一般ray/tmin/有限opaque与culling flags；原矩阵到1024三角形，后补4096三角形，核对hit、ID、t、barycentrics与face。Phase18已有单warp设备内计时和独立算术重叠对照；尚无吞吐或跨warp RT/CUDA争抢结果。** 当前精简序列有13条TTU指令，不声称理论最短或首次实现。

目录保留两条实验线：`tools/run_exp.py` 用于 EoSS 独立指令/会话行为复现，`tools/run_minimal.py` 用于真实几何功能验证。前者不需要几何 oracle 来判断 OPEN fault 等行为；架构/驱动差异和工程知识也值得独立保存，详见 [EOSS_REPRO.md](docs/EOSS_REPRO.md)。旧文档记录已恢复到 `docs/reference/history/`，恢复源码仍在 `reference/history/source/`；未找回原文件的入口明确标为重建版。

新增生产 AS 入口 `tools/run_scene.py`：OptiX 只负责 builder 与事后 oracle，测量目标仍是自研 CUDA kernel 的裸 TTU。两张3090上16场景、980次launch通过，见 [PHASE13.md](docs/PHASE13.md)。公开 AS 资料也有新增：NVIDIA 专利提供 complet/node 布局示例，见 [公开资料核查](../research/notes/nvidia-bvh-public-records.md)；目前还没有对上本机格式的完整公开 builder。

```bash
CUDA_VISIBLE_DEVICES=7 timeout 30s python3 bench.py run_scene separated --group both --repeat 5 --check
CUDA_VISIBLE_DEVICES=7 timeout 30s python3 bench.py run_scene grid --count 1024 --group both --check
python3 bench.py validate_scene --gpus 5 7 --output results/local-scenes.json
```

此入口需要本地OptiX8 SDK、nvcc和g++；`both`比较19条expanded与13条reduced。原20条predicated组在实时GAS上不稳定，仍保留`--group full`诊断入口。

2026-11-03更新见 [PHASE14.md](docs/PHASE14.md)：一般ray与flags矩阵共20,208次TTU launch通过，13/19条组主要返回逐位一致。**默认ray flags现在为1（DISABLE_ANYHIT）**；flags0的nonopaque路径存在较远候选、需要续遍历的反例，不能当作完整closest-hit。geometry flag1旧负结果来自t符号解码过窄，已纠正。Phase14的IAS入口仅作原始事件诊断；单instance设备端功能使用上面的新入口。

```bash
CUDA_VISIBLE_DEVICES=5 timeout 30s python3 bench.py run_scene overlap --ray-suite general --group both --check
python3 bench.py validate_scene --gpus 2 6 --ray-suite general --output results/local-general.json
python3 bench.py validate_ray_protocol --gpus 0 5 --output results/local-ray-protocol.json
```

## 快速复现

需要 NVIDIA 驱动、支持 SM86 的 `nvcc` 和 Python 3。固定fixture功能线使用 Python 标准库，不依赖OptiX SDK；实时AS/instance入口另需本地OptiX8 SDK和g++；EoSS 复现线使用本地 `research/community/eoss` 包及原始 encoding 数据。

先查 `nvidia-smi` 选择空闲卡；在 `ttu-bench/` 运行：

```bash
CUDA_VISIBLE_DEVICES=7 timeout 20s python3 bench.py run_minimal hit --check --repeat 5
CUDA_VISIBLE_DEVICES=7 timeout 20s python3 bench.py run_minimal miss --check --repeat 5
CUDA_VISIBLE_DEVICES=7 timeout 20s python3 bench.py run_minimal hit --oz -10 --check
CUDA_VISIBLE_DEVICES=7 timeout 20s python3 bench.py run_minimal hit --dz 2 --check
CUDA_VISIBLE_DEVICES=7 timeout 20s python3 bench.py run_minimal hit --tmax 6 --check
```

输出为 JSON。`--check` 通过解析求交检查硬件结果，不匹配时非零退出；`hit/miss` 仅选择默认 x/y，实际命中由完整 ray 参数决定。基础 hit 距离为 6.875，最后一条 tmax=6 的命令应得到 miss。`--full-group` 使用保存的原始 20 条 TTU 组作对照。

完整回归，每个场景独立进程 / 新 context：

```bash
python3 bench.py validate_minimal --gpus 5 7 --output results/local-functional.json
```

Phase 12 在 GPU 5、7 上共 16 个场景、80 次 launch 通过，记录在 [results/phase12/phase12-functional.json](results/phase12/phase12-functional.json)。错误的 TTU/地址实验可能损坏当前 CUDA context，独立进程和 timeout 用于限制单次实验；timeout 不能保证终止已经挂起的 GPU 工作。

## 当前实现

`src/ttu_isa.py` 提供完整 TTU 族的字节构造、字段解码/修改和融合计数；`src/cubin_builder.py` 提供普通 SASS 和单入口 CUDA ELF patch；`src/eoss_helpers.py` 保留 EoSS canonical bytes 和完整替换策略。`src/sm86_fixture.py` 集中维护已验证的组、root 打包和 ray 输入知识。`src/minimal.py` 与重建的 `tools/build_and_patch.py` 都使用这些组件，避免各自在脚本内复制位运算。

EoSS 行为复现与 CPU 编码检查：

```bash
CUDA_VISIBLE_DEVICES=7 python3 bench.py run_exp all --repeat 3 --output results/local-eoss.json
python3 bench.py validate_encoding
```

`tools/run_exp.py` 为每项实验创建新进程，保留 fault 后的 sync 返回码；超时即停止。scoreboard-write 探索不在默认 `all` 内。已复现单 OPEN 成功、双 OPEN 稳定715、MF0/CCTL 成功。新测 CCTL+scoreboard 超时及与 SM120 的比较限制见 EOSS_REPRO。

精简 TTU 组为：

```text
OPEN → MF(11) → 9 × TTUST → GO → TTULD.CLOSE[0x300]
```

普通指令还负责输入初始化、结果等待和写回，所以“13 条”仅指 TTU 指令数，不是 kernel 总长度。编译容器中的 FADD 链在加载前被替换，不执行。

- 数据来自 [fixtures/](fixtures/)：保存的单三角形 AS 和驱动 JIT 镜像。镜像只作为指令/静态数据来源，不加载执行。
- GPU kernel 使用 80 registers/thread，0 local / shared / stack，当前只有一个 active lane。
- 默认ray为`(.25,.25,-5)+(0,0,1)t`。triangle hit观察到R72的8/a两种tag和低位primitive ID。t=abs(FP32(R73))，u=abs(FP32(R32))；R32符号对上front/back face，R73符号与opaque/候选路径相关，见Phase14。miss为R72=R73=`ffffffff`；未知事件另分类。
- AS 和 dummy root 在普通 CUDA 高 VA 分配；此 fixture 不需要低 4GB 地址。AS 格式与完整输出语义尚未通解。
- `--repeat` 重复多个独立 launch，不是同一 kernel 内的 TTU 循环；Python 耗时不能当作 RT latency。

完整编码、地址打包、几何判据和纠错证据见 [PHASE12.md](docs/PHASE12.md)。

## 研究与下一步

长期希望形成论文贡献；可靠基础设施、复现和blog是有价值的阶段成果，不以新颖性作为前期推进门槛。两代TTU接口对照和复杂AS路线见 [AS_ROADMAP.md](docs/AS_ROADMAP.md)。Phase17已提交为 `867ee16`。**复杂IAS协议逆向暂时挂起**：四实例0x24/直接triangle返回、嵌套、异构child及等距边界保留为后续问题，现有helpers、反例与Phase17代码继续保留。[Phase18](docs/PHASE18.md) 两GPU共172项、2064次launch及136224次逐lane检查通过；完整组后、等待前的独立FADD可与未完成请求重叠。单lane/13条窗口约381周期，32条FADD放在等待前仍381、等待后509；这是含接口/时钟的窗口，不是纯RT延迟。最新扩展及MACROFUSE的资源准入/抢占模型见 [Phase19](docs/PHASE19.md)。Phase17功能容器有848字节stack和完整trace，不能直接当作最小计时核。

[研究空白审视](../research/notes/rt-cuda-research-gap.md)已纳入 Ray-by-Ray 和 RTSpMSpM 全文：前者已有同 SM 的 lane/warp 微基准并报告 Ampere 复现，后者已讨论 RT/SIMD 的 L1 竞争。仅换到 GA102、重做 lane/warp 扫描或发现内存争抢，不足以构成新贡献。

Phase18/19完成了相同ray的full-warp计时与工作重叠验证。2026-11-04的下一步选择是先做单GAS小renderer，补不同lane的ray和尾warp正确性，用软件CUDA BVH、裸TTU、OptiX三个后端比较具体实现；只将小段TTU发射封装成SASS/cubin patch，其余留在CUDA。随后回到紧凑的同CTA多warp准入/竞争测量。范围、计时边界和完成标准见 [后续路线](docs/MEASUREMENT_ROADMAP.md)。

## 阅读与目录

- [INDEX.md](docs/INDEX.md)：文件用途、依赖和结果来源。
- [HANDOFF.md](docs/HANDOFF.md)：已知事实、边界和可执行的后续任务。
- [PHASE12.md](docs/PHASE12.md)：通过验证的协议及实验记录。
- [PHASE13.md](docs/PHASE13.md)：生产GAS、多三角形、返回解码及原组/flags负结果。
- [PHASE14.md](docs/PHASE14.md)：三组指令证据、一般ray/flags、解码纠正、多命中反例与IAS事件。
- [PHASE15.md](docs/PHASE15.md)：静态instance记录解码、host参考实现及4096三角形GAS。
- [PHASE16.md](docs/PHASE16.md)：设备端事件分派/变换/独立child查询及其历史验证范围。
- [PHASE17.md](docs/PHASE17.md)：受限父IAS恢复、多实例最近命中、ID解码纠正及负对照。
- [PHASE18.md](docs/PHASE18.md)：单warp计时、异步算术重叠、OPEN-gap与MACROFUSE专利/实测边界。
- [PHASE19.md](docs/PHASE19.md)：融合组拆分/必要性反例、Tensor和访存重叠、即时消费者校准。
- [EOSS_REPRO.md](docs/EOSS_REPRO.md)：EoSS 复现结果、SM86/SM120 delta、helpers API 和恢复边界。
- [CLEANUP.md](docs/CLEANUP.md)：删除/保留理由与迁移清单。

2026-11-02 的清理误删了 EoSS 入口和可复用知识；已恢复可找回的历史资料、重建入口并抽取 helpers。错误解释继续撤回，原观察及其条件保留。功能测量以通过几何 oracle 的独立 kernel 为基线。
