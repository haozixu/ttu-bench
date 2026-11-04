# 下次任务入口

归档/Phase18/Phase19已提交为 `338b752` / `ed5fa4d` / `3f66f03`；文件整理单独提交。当前代码分src/tools/kernels，使用 `python3 bench.py <命令>`。原报告及hash不改，整理前源码由Git恢复；迁移表见 [INDEX](INDEX.md)。

2026-11-04决定先做小renderer再回多warp，具体看 [MEASUREMENT_ROADMAP](MEASUREMENT_ROADMAP.md)：单GAS/opaque、逐lane不同ray/尾warp、depth/normal、软件CUDA BVH/裸TTU/OptiX三后端。公开PTX没有TTU发射入口，仅将TTU小后端patch，其余用CUDA。构建/steady追踪/着色/传输分开计时，加速比描述具体实现，不称纯硬件收益。计划尚未实现；复杂IAS继续暂挂。

README在 `ttu-bench/` 根目录，详细文档在 `docs/`，命令从该项目根目录执行。Git仓库也位于 `ttu-bench/.git`，用户已提交初始版本；Phase17按用户要求提交为 `867ee16`。本轮用户已授权直接提交；提交时按实验与整理分开。概念见 [TTU_PROTOCOL_GUIDE.md](TTU_PROTOCOL_GUIDE.md)，完整状态见 [HANDOFF.md](HANDOFF.md)。

已提交的 [PHASE17.md](PHASE17.md)：一个CUDA kernel能保存硬件返回的待处理引用，独立查child后恢复父IAS。已测平坦1–3个正行列式静态instance、共享triangle GAS、flags1、相对引用类型0x21；两张3090上30项、4,768次launch、11,492次请求通过CPU/OptiX及逐事件检查，包括两个必要反例。first-child-only负对照确实得到错误几何结果。这不是完整IAS/JIT协议。

重要纠正：E-0x0c/ E-0x20只在单instance中碰巧对应ID/handle。用[E+4]的1-based索引查询IAS表：base+0x30为表偏移，步长0x80，元数据+0x74低24位ID、+0x60 handle。E+0x10逆矩阵和[E+8]低49位child root仍有效。现行decoder和device kernel已修正，历史报告原样保留。

```bash
CUDA_VISIBLE_DEVICES=5 timeout 25s python3 bench.py run_mixed_timing --lanes 32 --kind hmma --count 8 --check --output results/local/tensor-before.json
CUDA_VISIBLE_DEVICES=5 timeout 25s python3 bench.py run_mixed_timing --lanes 32 --variant split10 --gap 64 --check --output results/local/split-go-read.json
```

**复杂IAS逆向继续暂挂，当前主线为 [Phase19](PHASE19.md)。** Phase18保留初始单warp/FADD基线及源码。Phase19正常矩阵152项、1824次launch、5184次warp请求、194784次逐lane检查全通过；另48个预期715（替换MF为匹配NOP/MF0）在独立process验证，无timeout。合法拆成MF9+ST、MF2+GO/LD，或MF10+ST/GO、MF1+LD均通过；GO后gap64FADD总时间仍617。不能推断已验证抢占或跨warp不可交错。

新增 `tools/run_mixed_timing.py` / `tools/validate_mixed_timing.py`，复用Phase18容器，但用128 registers和256B dynamic shared。FADD、HMMA、global/shared读均有独立/等待前/等待后对照。HMMA单靠SB5等候曾让立即消费者读旧值，正式版每条至少32 scheduled cycles，核对四个即时消费者；别删除padding后仅看最终snapshot。global热小环绕L1、shared无bank conflict，均非饱和带宽测量。C只等TTU SB1，随后立即copy四个返回字段验证；工作独立等自己的内存依赖。

下一步按renderer路线补不同lane/尾warp及三后端，完成后再回多warp；恢复测量线时确认HMMA依赖/静态间隔，缩减观测开销，再测双向吞吐与资源竞争。不要把表中差分直接命名为MIO/L1争抢。恢复IAS时从四实例0x24/直接triangle路径开始；保留等距样本51项旧失败，不放宽oracle。

`src/parent_resume.py` / `kernels/parent_template.cu` / `tools/run_parent_resume.py`是正式设备路径；`tools/run_parent_probe.py`是host raw诊断，其copy/promote/pop不是通用算法。共享device_instance支持选择kernel，避免重复callee的marker混计。parent_probe为88 registers、848字节stack、单active thread、完整trace，不能作为最小计时核。单instance8项及固定fixture8项回归通过，编码检查通过。

triangle GAS最大已测4096，不是容量上限，也不知道一次GO访问节点数。默认DISABLE_ANYHIT；flags0的较远候选仍需要独立续遍历协议。测量线可先在稳定opaque GAS上推进正确设备内重复、计时、active lanes/full warp及同SM干扰，不必等待完整OptiX支持。

Phase18入口仍保留，原报告source hash不改；新实验使用Phase19入口。结果按phase12–19归档，manifest累计642份原文件、543份发布报告、123个物理发布文件，约14.92 MiB。Phase17/18的226个文件变为12个，Phase19共211份证据变为5个。逐报告无损存入各phase的records.zip，外面保留矩阵/摘要；新结果先放ignored local。旧路径用 `result_store.load_result()` 或 `python3 bench.py result_store show`，不要直接Path读取；`verify --include-local`本机已校验642份。ZIP是正式证据，不能ignore。所有helpers/EoSS及复杂IAS代码保留；归档、Phase18、Phase19已分别提交。

长期希望论文；基础设施、复现、blog也是阶段成果。EoSS入口/helpers/代际delta独立保留，勿删除；详见 [EOSS_REPRO.md](EOSS_REPRO.md) 和 [AS_ROADMAP.md](AS_ROADMAP.md)。
