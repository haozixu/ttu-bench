# 下次任务入口

README位于 `ttu-bench/` 根目录，详细文档在 `docs/`，命令从 `ttu-bench/` 执行；概念先看 [TTU_PROTOCOL_GUIDE.md](TTU_PROTOCOL_GUIDE.md)，状态见 [HANDOFF.md](HANDOFF.md)。结果已按阶段分类并压缩，旧路径由manifest或 `result_store.load_result()`解析，见 [RESULTS.md](RESULTS.md)。

最新 [PHASE16.md](PHASE16.md)：一个自研CUDA kernel已能处理单个静态instance的IAS事件、逆变换ray、独立查询child GAS。8项矩阵、1344次kernel launch、2615次TTU请求全部通过CPU/OptiX对照，包括541次instance事件后child miss。`device_instance.py`提供编译器寄存器映射与19条组patch，`instance_template.cu`保留可读CUDA事件处理，`run_device_instance.py`与`validate_device_instance.py`提供检查及复现。

**尚未保存并恢复父级IAS状态。** 下一功能任务是沿JIT父级状态保存/恢复路径对应返回槽和下次请求输入，然后用两个instance验证：第一child miss而后续hit、第一child hit而另一个更近、交换输入顺序。记录实际事件顺序，不要逐个instance独立查询取min后宣称硬件resume成功。`parent_slot_nonzero_raw`仅记录R51非零，尚不是兄弟节点/可恢复状态的解码。

```bash
CUDA_VISIBLE_DEVICES=1 timeout 40s python3 run_device_instance.py --instance identity --ray-index 0 --check --output results/local-device-one.json
python3 validate_device_instance.py --gpus 1 2 --output results/local-device-matrix.json
```

[Phase15](PHASE15.md)有记录字段差分和host参考实现；普通JIT实例分支在 `reference/jit/jit16-instance-ordinary.sass`，需要时用 `disassemble_ttu.py`补齐父级分支。其TTU槽已换NOP，仅供分析，不可运行。

triangle GAS最大已测4096三角形，不是容量上限，也不知道一次GO访问节点数。[Phase14](PHASE14.md)保留一般ray/tmin/有限flags验证与nonopaque较远候选反例；默认ray flags1（DISABLE_ANYHIT），不能把flags0一次GO当完整closest-hit。

测量线仍需可验证的设备内重复、计时、active lanes/full warp和同SM干扰矩阵。Phase16功能容器有288字节stack与搬运，不是最小计时核。固定fixture功能基线仍用 `run_minimal.py`，编码校验用 `validate_encoding.py`。

长期希望论文贡献；当前基础设施、复现和blog也是成果，见 [AS_ROADMAP.md](AS_ROADMAP.md)及 [研究空白审视](../../research/notes/rt-cuda-research-gap.md)。EoSS复现与代际delta独立保留，见 [EOSS_REPRO.md](EOSS_REPRO.md)；勿删除旧helpers/实验记录。历史快照在 `docs/reference/history/`，恢复源码仍在 `reference/history/source/`。

根目录与ttu-bench的 `.gitignore`忽略SDK、编译缓存、Python缓存及local/scratch/archive结果；正式结果的压缩发布集、manifest、fixtures、JIT证据、历史源码和研究PDF保留。用 `python3 result_store.py verify`校验发布集；本机可加 `--include-local`验证全部205份原始结果。设备instance cubin在临时目录构建，退出自动清理。本地当前没有Git仓库元数据。
