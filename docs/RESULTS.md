# 结果保留与读取

日常实验保留可复现命令和一句结论即可，不要求留完整artifacts。逐ray dump、cubin、反汇编副本、逐case报告、重复矩阵和临时编译目录使用后清理。完整证据只为重要协议性质、推翻结论的反例或难以重现的观察保留，优先小型自包含样本；不维护临时目录清单，也不在文档中引用ignored路径。

已有发布证据保持原字节和hash，保留543份报告、123个物理文件（约14.9MiB）；旧本机重复副本与99项未精选试跑记录已清理。数据目录编号不再决定文档结构：

| 数据集 | 主要保留内容 |
|---|---|
| phase12–14 | 独立TTU正负对照、一般ray/flags、符号纠错、nonopaque候选 |
| phase15–17 | instance记录、ID查表纠错、父级恢复的必要反例、四实例/等距边界 |
| phase18–19 | 单warp重叠、MF替换fault、HMMA即时消费者与夹具校准 |
| eoss、maintenance | 会话行为、SM86/SM120静态比较及依赖重构回归 |
| fixtures/renderer-counterexamples.json | tag4/6的关键word、两条数值失败的ray/triangle/结果 |

phase17–19的逐项报告位于各数据集records.zip，其余保持已有gzip/JSON；不为常规renderer试跑打包发布。一个示例PNG留在renderer的assets目录，其余可按 [运行说明](../examples/renderer/README.md) 再生。

## 读取

manifest中的original_path/aliases解析旧报告名，published_path指向当前存储；ZIP同时记录archive_member及原字节hash。矩阵内部的历史路径/source hash不改，不应直接按旧路径读取文件。

```bash
python3 bench.py result_store verify
python3 bench.py result_store show phase17/device-chain2-overlap.json.gz
python3 bench.py result_store show phase18/resource-functional.json.gz
```

Python可用result_store.load_result()；需要时result_store restore可恢复原相对结构，目标由--destination指定，拒绝越界或覆盖不同内容。旧源码可由Git恢复。重新整理文档不会改写已有结果来匹配当前源码。

新保留证据应有输入、判据、环境、必要原始返回和复现命令；不要用大量hash或维护清单替代可理解的结论。负例本身推翻了什么比是否失败更重要，普通重复失败不必全部留存。
