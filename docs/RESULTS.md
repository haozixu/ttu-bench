# 实验结果归档与Git提交

更新：2026-11-03。原 `results/` 共205份文件，153,010,348字节（145.9 MiB）。此次整理保留所有原始字节，没有删除实验数据或改写历史报告。

## 提交哪些

| 目录/文件 | 内容 | 提交策略 |
|---|---|---|
| `results/phase12/` | 固定fixture基线、反例、资源与设备信息 | 提交 |
| `results/phase13/` | 正式GAS矩阵及逐case报告，20条组fault、decoder误判等证据 | 提交 |
| `results/phase14/` | 最终v2矩阵及全部逐case报告、跨组对照、较远候选、IAS事件、解码纠错记录 | 提交 |
| `results/phase15/` | 正式host-instance矩阵及6项原报告、4096三角形GAS、回归 | 提交 |
| `results/phase16/` | 正式device-instance矩阵及8项原报告、资源、回归和代表性事件 | 提交 |
| `results/eoss/` | EoSS指令行为、fault/timeout及SM86/SM120静态编码对照 | 提交，保留有意义的delta |
| `results/maintenance/` | 清理后及helpers提取后的功能回归 | 提交 |
| `results/manifest.json` | 全部205份原文件的路径、字节数、SHA256、发布/本地去向 | 提交 |
| `results/archive/exploratory/` | 早期scene、旧矩阵、sign-check、instance初版等中间记录 | 本地保留，ignore |
| `results/archive/originals/` | 已发布gzip对应的原始未压缩副本 | 本地保留，ignore |
| `results/archive/local/`、`local-*`、`local/`、`scratch/` | 本机试跑、解压、临时产物 | ignore |

发布集为106份原始结果，存储共4,300,493字节（4.1 MiB），另加manifest和小型派生样例。超过256 KiB的正式JSON使用标准gzip、固定mtime=0、无损压缩；因此Git中的大型raw日志不再逐行diff，日常审阅看小型矩阵、文档和样例。此规模暂不需要Git LFS。

旧的general-final/closest、旧ray-protocol以及Phase15初版有独立实验条件和源码hash，不能笼统称为完全相同的重复文件。它们保留在本地归档，正式结论选择最终v2/正式矩阵作为主要证据。关键负结果及EoSS记录单独提交，不能因“不是当前主线”丢弃。

`results/phase16/example-events.json`从正式报告逐项提取hit、instance后child miss、IAS直接miss各一个；保留来源路径、hash、完整raw记录和输入ray。它是方便阅读的派生文件，不能代替完整矩阵。

## 如何读取与校验

命令从 `ttu-bench/` 执行，无第三方Python依赖：

```bash
python3 result_store.py verify
python3 result_store.py verify --include-local
python3 result_store.py restore --destination results/local/restored
```

第一条校验全部发布结果的压缩文件hash及解压后的原始hash；第二条同时核查本机归档和未压缩副本，clone上没有归档时不可用；第三条将发布集解压并恢复原始results相对目录，不修改发布文件。遇到已有但不同的目标文件会拒绝覆盖。

程序化读取示例：

```python
from result_store import load_result
matrix = load_result('phase16-device-matrix.json')
for case in matrix['cases']:
    report = load_result(case['file'])
    print(case['file'], len(report['records']))
```

matrix内部的旧路径、命令和source hash全部原样保留。解析旧的`results/...`路径，或报告中带`/results/`的原机器绝对路径，均可使用此loader；不要直接在新目录对旧matrix的`file/report`字段调用`Path.read_text()`。manifest描述本次归档的历史集，新跑出的local结果不会自动加入它。

未来实验先输出到 `results/local-*` 或 `results/local/`，有明确结论后选择正式结果归档，并同步manifest和阶段文档；不要把每次探索都直接提升为正式phase证据。

## 与本次提交的关系

README已回到项目根目录，阶段文档仍在 `docs/`；历史文档快照没有修改。fixtures、原JIT镜像、历史helpers以及研究PDF不属于此次结果裁剪范围。

当前工作区没有 `.git` 元数据，因此这里只整理文件和验证ignore，不执行init、add或commit。另有 `research/community/` 约747 MiB、`research/papers/` 约242 MiB；它们与这4.1 MiB结果是不同来源。若准备提交整个gpu-engine，第三方checkout应另外决定采用来源清单/固定commit/submodule还是vendor；不要将嵌套checkout默认为本项目源码，也不要未经核对直接ignore掉EoSS运行依赖。
