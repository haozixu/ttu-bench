# 原Phase编号简史

原Phase文档已按主题合并，编号仅用于理解旧报告和Git历史。本表记录当时做了什么，并标明后续纠正，不将旧假设当作当前事实；不恢复逐阶段日志和临时产物清单。

## 早期探索：Phase1–11

来源是恢复的清理前README/HANDOFF；其中Phase2–4原本合写，无法可靠细分各编号边界，故沿用合并范围。

| 原编号 | 当时的主要工作 | 后续认识 |
|---|---|---|
| 1（初始复现） | 用EoSS编码patch CUDA容器，测试OPEN、双OPEN fault、MF0、CCTL和融合组执行 | 指令/会话行为有价值，但无有效AS的组执行不证明真实求交 |
| 2–4 | 获取Vulkan两三角形BLAS，尝试AS重定位、配置槽/寄存器编码与低VA访问 | TTULD字段、第二寄存器、control byte与“必须低4GB”等旧推断后被纠正；没有完整节点格式 |
| 5 | 跑通最小OptiX管线，捕获驱动JIT，分析utrav组及输出槽 | 对某遍历家族的观察不能当作整个SM86协议；“LD必须4对齐”后被反例否定 |
| 6 | 对比EoSS与535驱动组，尝试描述符指针及驱动绑定路径 | 不能仅据组结构推导代际断裂，也不能把某条路径的指针解释推广到mttu |
| 7 | 扫描候选描述符布局、定位真实输入获取路线 | 扫描未得到几何正确性，未证明指针截断或普遍地址限制 |
| 8 | 探索指令页/模块捕获，重建驱动遍历协议线索 | 捕获原始资产有价值；“缺外核现场”并非后来的独立路径依赖 |
| 9 | 解码已链接raygen/megakernel/mttu模块、参数及状态流程，尝试直接发射 | 干净执行不等于真实hit/miss，夹具与后续local路径仍有问题 |
| 10 | 校准普通SASS编码，纠正prologue、寄存器与NOP补丁，分析local/shared访问 | 修正了一批工具错误，未建立通用RT共享存储或遍历完成结论 |
| 11 | 在组后直接读回寄存器、尝试中继，定位输入/结果观察问题 | “-1是引擎拒绝、需要驱动外核现场”后来被独立几何实验推翻 |

原始表述保留在历史快照作来源，当前有效知识见 [EoSS复现](EOSS_REPRO.md) 和 [协议](TTU_PROTOCOL.md)。旧工具不是现行入口。

## 功能、测量与演示：Phase12–21

| 原编号 | 主要进展及边界 | 现行主题 |
|---|---|---|
| 12 | 全新CUDA context内独立TTU实现固定三角形hit/miss；13条精简组及正负对照，纠正寄存器字段和驱动现场依赖猜测 | [协议](TTU_PROTOCOL.md) |
| 13 | OptiX仅构建AS/作oracle，裸TTU查询生产GAS；多三角形到1024面，比较13/19/20条组；原20条在生产AS不稳定 | [协议](TTU_PROTOCOL.md) |
| 14 | 一般方向、tmin/tmax、opaque/culling flags与符号解码；13/19组主要返回对照；发现nonopaque较远候选及IAS事件 | [协议](TTU_PROTOCOL.md) |
| 15 | host读取instance逆变换/child引用并独立查询child；GAS补到4096面；当时邻近ID/handle偏移后被多实例否定 | [协议](TTU_PROTOCOL.md) |
| 16 | 一个CUDA kernel内处理instance事件、变换ray和独立child查询；仍没有父IAS续遍历 | [协议](TTU_PROTOCOL.md) |
| 17 | 修正ID/handle为索引查表；保存pending并恢复受限1–3实例父IAS，验证“先miss后hit/先远后近”；四实例0x24和等距边界未解，复杂IAS暂挂 | [协议](TTU_PROTOCOL.md) |
| 18 | 单warp、1/32 active lanes设备内重复与四窗口计时，FADD异步重叠、OPEN-gap及MF0/NOP校准；非饱和吞吐 | [测量](MEASUREMENTS.md) |
| 19 | 合法MF拆组、正常MF替换fault、同warp Tensor/global/shared重叠；发现HMMA末尾正确但即时消费者错误，保留padding限制 | [测量](MEASUREMENTS.md) |
| 20 | 三后端箱子renderer、逐lane不同ray/尾warp与计时；直接O1–O3破坏marker，分离编译正确但收益小 | [renderer](RENDERER.md) |
| 21 | Bunny到69,453面、tag4/6、预热/顺序修正和单次地面镜面；四级矩阵仅res3/full通过，另外两级有掠射数值反例 | [renderer](RENDERER.md) |

Phase12–19的原文可从Git版本`f18d7b4`读取，例如 `git show f18d7b4:docs/PHASE17.md`；Phase20/21在合并前未提交，当前说明已归入renderer主题。发布数据的旧编号和原报告hash保持不变，读取方式见 [结果说明](RESULTS.md)。今后以更新主题文档为主，不为小功能继续增加Phase。
