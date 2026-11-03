# 清理前记录的恢复副本

这些文件从清理前读取文件的完整 stdout 恢复，保留当时的表述，包括后来撤回的结论。**用于追溯实验和知识来源；当前协议请查 [TTU 协议入门](../../TTU_PROTOCOL_GUIDE.md)、[PHASE14](../../PHASE14.md) 与 [EOSS_REPRO](../../EOSS_REPRO.md)，不能把历史文件当现行操作指南。**

2026-11-03仅迁移文档位置：四份 `.pre-cleanup.md` 保持逐字节原文，包括其中的旧链接和旧路径。恢复源码仍在 [reference/history/source](../../../reference/history/source/)，未随文档移动。迁移前SHA256见 [迁移清单](../../document-migration.json)。

- [README.pre-cleanup.md](README.pre-cleanup.md)：Phase 1–9 记录，含 EoSS 初始复现、旧字段猜测和驱动路径观察。
- [INDEX.pre-cleanup.md](INDEX.pre-cleanup.md)：旧脚本、资产和依赖索引。
- [HANDOFF.pre-cleanup.md](HANDOFF.pre-cleanup.md)、[NEXT_PROMPT.pre-cleanup.md](NEXT_PROMPT.pre-cleanup.md)：Phase 12 开始之前的交接；其中“缺外核现场”“三条通道判死”等判断已有反例。
- `reference/history/source/`：可从会话输出恢复并与删除清单 SHA256 相符的源码/语料。`build_ttu3.py`、`probe3.cu`、`run_ttu4.py` 保留原字节和原错误注释；不直接运行。它们依赖旧根目录布局，已不是当前入口。

`build_and_patch.py` 和 `run_exp.py` 的原始文件未找到完整副本；根目录同名文件是重建实现，不能称为原文件恢复。它们保留 exp1–5 的实验身份，使用新的可复用构建器。旧实验记录与本轮结果分别保存，不能冒充相同 harness 的精确复现。

恢复完整旧记录是为了保留观察及其上下文；研究新颖性不是资料是否保留的判据。新实验中发现的问题应记录、纠正，并保留原实验身份和失败条件。
