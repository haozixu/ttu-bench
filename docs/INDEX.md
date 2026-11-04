# 文件索引

先读 [根README](../README.md)。文档按主题维护，实验数据原有编号只作历史标识，不再为每个小功能新建phase。

| 文档 | 内容 |
|---|---|
| [TTU_PROTOCOL_GUIDE](TTU_PROTOCOL_GUIDE.md) | 从图形基础概念解释硬件/软件边界 |
| [TTU_PROTOCOL](TTU_PROTOCOL.md) | SM86编码、GAS/IAS输入输出、父级恢复和关键纠错 |
| [MEASUREMENTS](MEASUREMENTS.md) | MACROFUSE、异步重叠、Tensor/访存及测量夹具 |
| [RENDERER](RENDERER.md) | 三后端、Bunny、数值边界、镜面与优化 |
| [EOSS_REPRO](EOSS_REPRO.md) | 原始/重建身份、指令行为及代际差异 |
| [AS_ROADMAP](AS_ROADMAP.md) | 复杂AS与完整OptiX语义的成本和选择 |
| [HANDOFF](HANDOFF.md) | 当前状态与下一项工作 |
| [RESULTS](RESULTS.md) | 已保留关键证据的读取方法及保留原则 |
| [HISTORY](HISTORY.md) | 原Phase编号的简要内容、后续纠正和现行主题对应 |

## 代码

| 位置 | 用途 |
|---|---|
| src/ttu_isa.py、eoss_helpers.py、cubin_builder.py | 指令族编码、EoSS完整替换、合法CUDA ELF构建 |
| src/sm86_fixture.py、minimal.py、cuda_driver.py | fixture知识、独立发射、Driver/ELF接口 |
| src/optix_scene.py；kernels/scene_bridge.cpp、scene_oracle.cu | 同context生产AS builder和独立OptiX oracle |
| src/instance_protocol.py、device_instance.py、parent_resume.py | instance解码、marker映射、设备父级恢复 |
| src/warp_timing.py、mixed_timing.py | 单warp测量及普通工作包 |
| tools/run_*.py、validate_*.py | 对应实验/回归CLI，均通过bench.py调用 |
| examples/renderer/ | mesh输入、软件BVH、三后端、CPU oracle、镜面和摘要计时 |
| fixtures/ | 原始AS/JIT、父IAS场景、renderer最小反例 |
| reference/ | 原始JIT、普通SASS分析、OptiX/Vulkan参考、恢复历史源码 |
| tools/result_store.py | 发布结果读取、校验和恢复 |

固定fixture无需OptiX，EoSS入口使用上层已有框架依赖。不要删除helpers或将EoSS会话行为实验误当作失败几何实验。含进程VA的动态cubin不能跨进程直接重放。

目录迁移前源码可从Git版本3f66f03恢复；原报告source hash不改。迁移信息在file-layout-migration.json及document-migration.json；后者是当时快照，旧文档现已按主题合并。旧错误表述保留在reference/history作来源资料，不作为现行指南。
