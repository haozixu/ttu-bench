# 结果保留与读取

日常实验保留可复现命令和一句结论即可，不要求留完整artifacts。逐ray dump、cubin、反汇编副本、逐case报告、重复矩阵和临时编译目录使用后清理。完整证据只为重要协议性质、推翻结论的反例或难以重现的观察保留，优先小型自包含样本；不维护临时目录清单，也不在文档中引用ignored路径。

当前精选158份报告，38个证据文件约4.1MiB，另有一个manifest；保留报告的原字节和hash不变。汇总矩阵仍完整保留，重复回归、第二张卡的重复逐ray报告和大部分常规计时dump已删除；旧本机重复副本与未精选试跑记录也已清理。数据目录编号不再决定文档结构：

| 数据集 | 主要保留内容 |
|---|---|
| phase12–14 | 独立TTU正负对照、一般ray/flags、符号纠错、nonopaque候选 |
| phase15–17 | instance记录、ID查表纠错、父级恢复的必要反例、四实例/等距边界 |
| phase18–19 | 完整汇总矩阵、代表性重叠/对照配对、MF替换fault、HMMA即时消费者与夹具校准 |
| eoss | 会话行为、CCTL超时、SM86/SM120静态比较 |
| fixtures/renderer-counterexamples.json | tag4/6的关键word、两条数值失败的ray/triangle/结果 |

phase13–19的精选逐项报告统一放入各数据集records.zip，摘要、事件示例和资源信息可直接读；EoSS和独立fixture证据保持JSON/文本。原始JIT/AS、helpers和源码保留。不为常规renderer试跑打包发布；一个示例PNG留在renderer的assets目录，其余可按 [运行说明](../examples/renderer/README.md) 再生。

## 读取

manifest只列当前保留证据，original_path/aliases解析保留报告的旧名字，published_path指向当前存储；ZIP同时记录archive_member及原字节hash。矩阵内部的历史路径/source hash不改，不应直接按旧路径读取文件。**完整历史矩阵不代表每条case的原始dump仍在当前树中**；被删除的报告及完整旧目录可从Git提交`358d42f`恢复，不维护另一份本地归档。原有phase16事件示例也已补入manifest。

```bash
python3 bench.py result_store verify
python3 bench.py result_store show phase17/device-chain2-overlap.json.gz
python3 bench.py result_store show phase18/resource-functional.json.gz
```

Python可用result_store.load_result()；需要时result_store restore可恢复原相对结构，目标由--destination指定，拒绝越界或覆盖不同内容。旧源码可由Git恢复。重新整理文档不会改写已有结果来匹配当前源码。

新保留证据应有输入、判据、环境、必要原始返回和复现命令；不要用大量hash或维护清单替代可理解的结论。负例本身推翻了什么比是否失败更重要，普通重复失败不必全部留存。

`tools/publish_results.py`默认只发布指定矩阵和显式选择的`--reports`，`--include-cases`才收录所有case；发布后由调用者清理输入工作目录。`tools/bundle_results.py`验证压缩结果后删除旧压缩副本，不再另存重复归档。
