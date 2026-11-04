# Phase18：单warp计时与MACROFUSE探索

日期：2026-11-03。RTX3090 / SM86，driver535.309.01，CUDA12.1。Phase17已按用户要求提交为 `867ee16`。**复杂IAS协议逆向暂时挂起**；四实例0x24引用、嵌套/异构child及等距边界的实现与证据保留。本阶段使用稳定的单三角形opaque GAS，研究提交、异步完成及等待。

## 已得到的结果

一个CTA、一个warp，分别用1和32个active lanes，在设备内连续发出4个请求；每次恢复输入，每lane独立写回，逐次检查hit/miss、primitive、t、barycentrics、face、lane地址与算术结果。两GPU正式矩阵全部通过。

完整TTU融合组结束后，结果仍可能未完成；同一个warp可以执行独立FADD，再等结果。**MACROFUSE组结束不等于遍历完成，也没有强制该warp立即停止普通计算。** OPEN和MF之间插入短算术链，还可隐藏部分提交前的等待；与专利描述的“MF等待OPEN资源就绪”相容，但没有直接读出MF内部状态。

## MACROFUSE的资料依据

EoSS §6引用的 [US11138009B2：Robust, efficient multiprocessor-coprocessor interface](https://patents.google.com/patent/US11138009B2/en) 已下载 [PDF](../../research/papers/nvidia-patent-US11138009B2-coprocessor-interface.pdf)，另存 [逐段摘录与来源hash](../reference/jit/macrofuse-patent-excerpts.json)。专利描述：

- §0221–0232、Fig.14A：OPEN先申请ticket/执行资源；MF等待连接可用，再将后续N条接口指令当作在CILP抢占和单步调试中不可拆开的宏指令。count限定范围，避免无限关闭中断。OPEN自身在宏组之外。
- §0200–0206、Fig.12：READ可以在结果尚未产生时发出，承诺日后把数据写入指定寄存器；CLOSE结束请求描述，但目标操作可能仍在进行。后续普通指令可先执行，消费结果之前再等待scoreboard。
- §0231列举宏内允许的接口指令类型，没有把普通FADD/NOP列为任意可插入项。因此本阶段不往融合组内部插普通指令，也不扫错误count或缺CLOSE。

这些是专利示例，**不是GA102产品ABI或本机抢占实测**。当前较合理的模型是：MF负责资源准入及有限提交序列的不可中断性；实际遍历异步进行；结果依赖由scoreboard等待。不能推断其他warp不能被调度，更不能把整个SM视为锁住。单warp实验没有直接验证中断屏蔽或跨warp仲裁。

## 测量核与时间窗口

[warp_timing.py](../warp_timing.py) 使用独立 [timing_template.cu](../timing_template.cu) 容器，自行patch执行代码；复用 `cubin_builder.py`、`ttu_isa.py` 和固定fixture知识。原helpers及功能入口保留。容器原FADD链不会执行。

```text
等待前一sample的读取依赖，恢复全部请求输入
A = clock64
完整有效的13条或19条TTU组（含OPEN、MF、CLOSE）
B = clock64                      # 此处不显式等待TTU结果
可选独立FADD链：before-wait
C = clock64，带wait-all
可选独立FADD链：after-wait
D = clock64
逐lane写回raw、时钟和累加器，保护store的源寄存器
```

记录 `B-A`（完整组窗口）、`C-B`（含等待前算术/结果等待）、`D-C`（等待后算术）、`D-A`（总窗口）。时钟读与固定stall也在窗口内。没有在融合组内部插时钟，因此**没有直接测GO到结果的纯延迟**，也没有分离全部RT访存和接口成本。

FADD为依赖链 `x=x+1`，使用独立R88，固定4-cycle调度间隔；最终累加器必须等于执行条数。两个placement只改变链的位置，不改变TTU body。有效组来自原fixture的13/19条组，内部调度位/count不变，MF分别为11和17。

OPEN-gap变体在OPEN后插一个额外clock和0/8/16/32/64条FADD，然后执行原MF及原body。普通指令都在融合组之外，count不变。额外clock、gap和指令布局会影响窗口，不把差分直接叫作MF硬件延迟。

[资源记录](../results/phase18/warp-resource.txt)：96 registers/thread、0 stack/local/shared。每lane样本步长512字节，4个sample连续展开，最多32 lanes；所有lane使用相同ray，尚未验证发散ray。快照在计时窗口之外，但影响后续缓存/取指条件。这是有观测开销的串行延迟实验，**不是紧凑吞吐核**，没有多个未完成请求同时在途。

## 正式验证与计数

[最终v2矩阵](../results/phase18/final-v2-warp-matrix.json)：GPU5和GPU0共172个case，每case独立进程/context、25秒timeout、12次launch，每次4个sample。共2064次kernel launch、6144次warp级TTU组提交、101376次lane-ray请求，以及136224次逐lane功能/时钟/累加器检查，全通过。前两个计数包含的对象不同：2064次launch包括空窗口/NOP/MF0/ALU校准，6144只计有效TTU组。

不同lane不能当作独立warp试验。摘要只取lane0，排除每个case首个launch与每次launch的首个sample，得到33个steady观测；全部冷样本仍保留并必须通过功能检查。UUID、源码/AS/group hash在原报告中。共享服务器没有锁频/锁SM，也没有证明物理独占；周期数仅代表本次配置，不报告纳秒常数或跨架构通用值。

覆盖空窗口、匹配NOP、MF0、13/19条组，hit/空间miss/tmax截断miss，两处算术链及OPEN-gap。原始报告名由manifest映射至 `results/phase18/records.zip`（172份case与3份校准/旧失败，共175个成员），用 `result_store.load_result()`读取。

### 完整组与等待是不同阶段

以下为GPU5的lane0 steady中位数，单位SM周期；GPU0相应基线相同。

| active lanes | 组 | B-A：完整组 | C-B：显式等待窗口 | D-A：总窗口 |
|---|---|---:|---:|---:|
| 1 | 13条 | 127 | 250 | 381 |
| 1 | 19条 | 148 | 301 | 453 |
| 32 | 13条 | 200 | 398 | 602 |
| 32 | 19条 | 237 | 546 | 787 |

额外LD改变提交和返回路径；不能把19与13条之差解释成“每条LD固定几周期”。32lanes小于32倍成本也不能单独证明内部精确执行宽度。

### 同warp算术与异步请求重叠

[派生摘要](../results/phase18/observations.json)从原矩阵提取，不增加实验次数。以下为GPU5中位数，GPU0对应数值一致：

| lanes / TTU组 | FADD数 | 等待前计算：总周期 | 等待后计算：总周期 |
|---|---:|---:|---:|
| 1 / 13条 | 32 | 381 | 509 |
| 1 / 13条 | 128 | 647 | 893 |
| 32 / 13条 | 64 | 602 | 858 |
| 32 / 13条 | 128 | 720 | 1114 |
| 32 / 19条 | 128 | 789 | 1299 |

空窗口12周期；ALU校准为 `12+4*N`，两种位置相同。32条FADD提供128周期计算，单lane/13条总时间仍381；先等待再算则509。位置对照支持同warp计算与未完成请求的重叠，不证明TTU与所有运算/访存都没有竞争，也不是多个warp间的调度结论。

### OPEN与MF之间的等待

13条组OPEN-gap的总窗口中位数：

| OPEN后FADD数 | 1 lane | 32 lanes |
|---:|---:|---:|
| 0，含额外clock | 381 | 602 |
| 8 | 392 | 602 |
| 16 | 424 | 632 |
| 32 | 488 | 695 |
| 64 | 616 | 822 |

OPEN后立即读clock约10周期。8条FADD按校准提供32周期计算，但1lane总时间只增11，32lanes不变。19条组也有相同模式。可隐藏开销位于GO之前，支持OPEN资源申请与随后MF准入等待的解释；仍需更直接的边界/压力对照，才能把具体周期归因到MF。

### MF(0)说明什么

空窗口12周期；插一个MF0后16周期，插匹配相同stall/yield/scoreboard字段的NOP也是16周期。两GPU、1/32lanes均相同。只能说本配置下MF0没有超过匹配NOP的可分辨额外成本；固定4-cycle调度间隔可能掩盖更短延迟。不能称为“MF延迟精确4周期”，也不能从零count推断资源拥堵时正常MF不等待。

## 校准问题与修复

初版snapshot没有为STG设置源寄存器读取依赖，下一sample覆盖时钟寄存器后，首个launch出现C晚于D等矛盾。[旧失败报告](../results/phase18/records.zip)原样保留。现为每次STG设置read barrier0并等待，再允许复用源寄存器；修复后重跑正式矩阵，全部冷/热样本检查时钟单调及lane寻址，没有筛掉失败样本改善数字。

另一个不含TTU的初版容器在REG:96时用STG.128读R92..R95触发715；留寄存器余量（最高源R88..R91）后通过，REG:104的诊断也通过。顶端寄存器/存储约束未继续研究，不能记成TTU限制。局部诊断脚本和失败记录留在ignored local目录。

[空窗口compute-sanitizer记录](../results/phase18/empty-sanitizer-final.txt)为0错误，只验证普通SASS校准路径，仪器化时间不纳入性能矩阵。另核验TTU字节保真、源码和报告hash。普通指令证据见 [SASS](../reference/jit/phase18-timing-ordinary.sass) 与 [来源manifest](../reference/jit/phase18-timing-disassembly.json)：TTU换NOP仅用于反汇编，不可运行。原cubin含进程VA，留在ignored local。

## 复现与下一步

先查空闲GPU，命令从ttu-bench根目录执行：

```bash
CUDA_VISIBLE_DEVICES=5 timeout 25s python3 run_warp_timing.py --group reduced --lanes 1 --alu 32 --placement before-wait --check --output results/local/warp-before.json
CUDA_VISIBLE_DEVICES=5 timeout 25s python3 run_warp_timing.py --group reduced --lanes 32 --open-gap 8 --check --output results/local/warp-open-gap.json
python3 validate_warp_timing.py --gpus 5 0 --output results/local/warp-matrix.json
python3 result_store.py verify
```

后续 [Phase19](PHASE19.md) 已开始MACROFUSE拆分、Tensor及访存位置对照；Phase18源码与原报告保留，避免改变旧结果的source hash。专利指出普通指令不属于允许宏内类型，暂不把“往组里插NOP看看会不会挂”当常规下一步。中断/抢占原子性需要独立方法；当前没有声称已测出。复杂IAS继续暂挂。

归档更新：原179份发布文件现在为5个物理文件，矩阵、observations、资源、sanitizer日志保留可读文件，其余无损合并入records.zip。旧失败链接指向ZIP，成员通过manifest/旧路径读取；不删除失败或改写历史报告。
