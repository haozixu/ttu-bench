# 下次任务入口

README在 `ttu-bench/` 根目录，详细文档在 `docs/`，命令从该项目根目录执行。Git仓库也位于 `ttu-bench/.git`，用户已提交初始版本；后续修改未经请求不要自动commit。概念见 [TTU_PROTOCOL_GUIDE.md](TTU_PROTOCOL_GUIDE.md)，完整状态见 [HANDOFF.md](HANDOFF.md)。

最新 [PHASE17.md](PHASE17.md)：一个CUDA kernel能保存硬件返回的待处理引用，独立查child后恢复父IAS。已测平坦1–3个正行列式静态instance、共享triangle GAS、flags1、相对引用类型0x21；两张3090上30项、4,768次launch、11,492次请求通过CPU/OptiX及逐事件检查，包括两个必要反例。first-child-only负对照确实得到错误几何结果。这不是完整IAS/JIT协议。

重要纠正：E-0x0c/ E-0x20只在单instance中碰巧对应ID/handle。用[E+4]的1-based索引查询IAS表：base+0x30为表偏移，步长0x80，元数据+0x74低24位ID、+0x60 handle。E+0x10逆矩阵和[E+8]低49位child root仍有效。现行decoder和device kernel已修正，历史报告原样保留。

```bash
CUDA_VISIBLE_DEVICES=6 timeout 40s python3 run_parent_resume.py --scenario first-farther-later-nearer --check --output results/local/parent-nearer.json
python3 validate_parent_resume.py --gpus 2 6 --output results/local/parent-matrix.json
```

先查GPU空闲。下一功能任务从四实例0x24引用/直接triangle返回路径开始，见发布的 `results/phase17/pop-four.json`；不能仅把kernel计数限制从3提高。等距/tmax边界另作隔离实验：`device-chain2-overlap`保留51项失败（29条TTU记录、22项CPU/OptiX比较），正式正矩阵没有这些并列身份样本。嵌套、异构child、负缩放、motion及用户any-hit仍未通。

`parent_resume.py` / `parent_template.cu` / `run_parent_resume.py`是正式设备路径；`run_parent_probe.py`是host raw诊断，其copy/promote/pop不是通用算法。共享device_instance支持选择kernel，避免重复callee的marker混计。parent_probe为88 registers、848字节stack、单active thread、完整trace，不能作为最小计时核。单instance8项及固定fixture8项回归通过，编码检查通过。

triangle GAS最大已测4096，不是容量上限，也不知道一次GO访问节点数。默认DISABLE_ANYHIT；flags0的较远候选仍需要独立续遍历协议。测量线可先在稳定opaque GAS上推进正确设备内重复、计时、active lanes/full warp及同SM干扰，不必等待完整OptiX支持。

结果在 `results/phase12` 至 `phase17`、eoss、maintenance，manifest累计252份原始文件、153份发布约5.43 MiB。新实验先放被ignore的 `results/local/`，选定正式证据后同步manifest。用 `python3 result_store.py verify`，本机可加 `--include-local`；矩阵的原始子报告名用 `result_store.load_result()`解析。SDK/缓存及local/archive被ignore，正式gzip、fixtures、JIT与历史源码保留。

长期希望论文；基础设施、复现、blog也是阶段成果。EoSS入口/helpers/代际delta独立保留，勿删除；详见 [EOSS_REPRO.md](EOSS_REPRO.md) 和 [AS_ROADMAP.md](AS_ROADMAP.md)。
