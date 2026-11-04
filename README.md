# TTU：SM86微架构探索与小renderer

在RTX3090 / GA102 / SM86上，用自研CUDA kernel直接发射TTU，研究RT与通用SM工作的交互。当前有独立指令helpers、几何协议验证、单warp异步重叠测量，以及CUDA软件BVH / 裸TTU / OptiX三后端renderer。长期可探索论文贡献，基础设施、复现和blog也是有效成果。

当前范围：

- 单GAS opaque triangle最近命中，一般ray/tmin/tmax、有限culling flags；Bunny含地面最大已测69,453面，不是硬件容量。
- 平坦1–3静态instance、共享GAS、正行列式变换和0x21父级引用恢复；复杂IAS逆向暂挂。
- 同warp的FADD、Tensor及global/shared读可隐藏部分TTU等待；尚无多warp饱和竞争或具体共享资源归因。
- renderer有单次地面镜面。Bunny四级固定视角矩阵仅res3/full通过，另两级存在掠射面的浮点求交残差，不能称为不支持该AS。

## 使用

命令从本目录执行。需要Python标准库、CUDA12.1/nvcc、g++、驱动；生产AS/renderer另需上层OptiX8 SDK。先用nvidia-smi选择空闲SM86卡：

```bash
python3 bench.py --help
CUDA_VISIBLE_DEVICES=0 timeout 20s python3 bench.py run_minimal hit --check
ttu_output_dir=$(mktemp -d /tmp/ttu-demo.XXXXXX)
CUDA_VISIBLE_DEVICES=0 timeout 25s python3 bench.py run_mixed_timing --lanes 32 --kind hmma --count 8 --check --output "$ttu_output_dir/hmma.json"
python3 bench.py prepare_renderer_bunny
CUDA_VISIBLE_DEVICES=0 timeout 180s python3 bench.py render_demo --bunny full --validation geometric --mirror-floor --images
```

日常实验只记复现命令和简要结论。renderer默认仅写一份报告并清理临时编译文件；`--images`保存展示图，`--artifacts`才保留raw buffer和编译产物。矩阵默认仅留一份摘要。不要为常规试跑保存多套完整dump，也不要在文档里依赖临时输出。

临时文件统一放在任务专用目录，禁止直接往`/tmp`根写文件；上例完成后删除`ttu_output_dir`所指目录。

## 阅读与代码

| 内容 | 入口 |
|---|---|
| Ray、GAS/IAS、instance、flags等基础概念 | [TTU协议入门](docs/TTU_PROTOCOL_GUIDE.md) |
| 编码、请求/返回、父级恢复、已纠正旧结论 | [协议](docs/TTU_PROTOCOL.md) |
| MACROFUSE、单warp计时及Tensor/访存重叠 | [测量](docs/MEASUREMENTS.md) |
| Bunny、数值反例、镜面及编译优化 | [renderer](docs/RENDERER.md) / [运行说明](examples/renderer/README.md) |
| EoSS复现、SM86/SM120差异 | [EoSS](docs/EOSS_REPRO.md) |
| 当前下一步 | [交接](docs/HANDOFF.md) |
| AS/完整OptiX目标的取舍 | [AS路线](docs/AS_ROADMAP.md) |
| 模块与保留证据 | [索引](docs/INDEX.md) / [结果说明](docs/RESULTS.md) |

`src/`是可复用ISA/patch/fixture组件，`tools/`是CLI，`kernels/`是CUDA/C++，`examples/renderer/`是独立示例。统一入口是bench.py；直接导入模块可设PYTHONPATH=src:tools。helpers、EoSS入口、原始JIT/AS及重要反例保留。文献在上层research目录。
