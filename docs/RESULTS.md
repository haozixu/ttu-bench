# 实验结果归档与Git提交

更新：2026-11-04。正式结论依赖的原始报告、反例和失败保留原字节及SHA256；临时试跑放在ignored目录。为减少目录中的文件数量，**Phase17起采用每阶段一个 `records.zip`，外面只保留矩阵、摘要和少数易读样例**。Phase12–16仍保留原gzip布局。

| 目录 | 发布内容 | 物理文件布局 |
|---|---|---|
| phase12–16 | 固定fixture、GAS/IAS正式矩阵、回归及关键失败 | 原JSON/gzip；本轮未改变 |
| phase17 | 父IAS恢复、两个必要反例、负对照、四实例与等距边界 | 47份报告→7个文件；41份报告在ZIP |
| phase18 | 单warp计时、172case、快照竞争失败及校准 | 179份报告→5个文件；175份在ZIP |
| eoss / maintenance | 指令行为、代际delta、修复后回归 | 原布局，保留 |
| manifest.json | 原始路径、aliases、原字节hash及发布去向 | 每份原报告一项；bundles记录物理ZIP的hash |
| archive/、local/、local-*、scratch/ | 试跑、中间版本、原未压缩副本、旧gzip副本 | 本地保留，ignore |

ZIP直接压缩原始JSON，未在内部再套gzip；支持标准解压工具按成员随机读取。每个阶段bundle有完整文件SHA256，每个成员另有原始SHA256/字节数。**报告数量与物理文件数量分开统计**：manifest.summary.published_reports计报告，published_files/bytes对ZIP去重；旧batch中的published_files/bytes是当时发布快照，不是当前布局。ZIP不ignore，也未采用Git LFS。

Phase17原47个文件变成7个，原始证据一份不少。旧压缩副本保留在本机 `results/archive/bundled/`，clone只需要发布ZIP，不依赖这些副本。已有阶段合并用 [bundle_results.py](../bundle_results.py)，新阶段从local矩阵发布用 [publish_results.py](../publish_results.py)，两者复用原字节，校验ZIP往返后再更新manifest并归档旧物理文件。

## 读取、校验和恢复

命令在ttu-bench根执行，只依赖Python标准库：

```bash
python3 result_store.py verify
python3 result_store.py verify --include-local
python3 result_store.py show phase17/device-chain2-overlap.json.gz
python3 result_store.py restore --destination results/local/restored
```

`show`接受原始名、旧发布路径alias及旧报告中的绝对路径；输出原始字节，可加 `--output`写单文件。具体case的旧名以manifest中的aliases为准。`verify`检查全部发布字节和成员hash；`--include-local`还检查本机归档/原副本，因此clone上通常不可用。`restore`恢复所有发布报告的原始相对路径，拒绝越出目标目录及覆盖不同内容；不会修改发布集。恢复目标默认被ignore。

```python
from result_store import load_result
matrix = load_result('phase16-device-matrix.json')
for case in matrix['cases']:
    report = load_result(case['file'])
    print(case['file'], len(report['records']))
```

矩阵内部旧路径、命令和source hash原样保留，**不要按旧路径直接 `Path.read_text()`**，应用loader解析。旧单文件链接已改为相应ZIP；具体成员由manifest的archive_member字段定位。loader也支持 `phase17/records.zip#<archive_member>`，ZIP路径本身不唯一标识报告。

## 新实验发布原则

新结果先写 `results/local/`，有明确用途后选取正式矩阵、全部引用子报告、关键负结果及必要校准证据，加入manifest。派生observations或example-events便于阅读，不能替代完整raw。早期不完整/错误测量保留于本地；若其失败本身影响结论，则将代表失败一并发布并解释原因。

不因非当前主线删除EoSS、helpers、历史源码或有意义的架构delta。fixtures、JIT和研究PDF不属于结果合并范围。Git仓库位于 `ttu-bench/.git`；本轮仅整理已有结果，未自动提交。此前用户授权的Phase17提交为 `867ee16`。
