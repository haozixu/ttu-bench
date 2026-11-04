# Phase19：MACROFUSE行为与同warp Tensor/访存重叠

2026-11-03，RTX3090 / SM86，driver535.309.01，CUDA12.1。复杂IAS逆向继续暂挂。Phase18原报告及整理前Git源码版本保留，新入口为 [run_mixed_timing.py](../tools/run_mixed_timing.py)、[mixed_timing.py](../src/mixed_timing.py)，拆组/负对照另有 [run_macrofuse_control.py](../tools/run_macrofuse_control.py)。本阶段是有功能检查的行为实验，没有饱和吞吐或具体争抢端口的结论。

## MACROFUSE：这次实际测了什么

以下省略寄存器恢复、时钟和写回，ST仍使用固定fixture的原始输入；所有普通计算都在融合范围之外。13条原组的payload是9ST、GO、LD.CLOSE，共11条。

| 用例 | 指令布局 | 结果 |
|---|---|---|
| normal | `OPEN; MF(11); 9ST; GO; LD.CLOSE` | 命中正确；两类miss另见Phase18 |
| prefix-mf0 | `MF(0); OPEN; MF(11); …` | 通过，窗口约增加4周期 |
| open-mf0 | `OPEN; MF(0); MF(11); …` | 通过，窗口约增加4周期 |
| split9 | `OPEN; MF(9); 9ST; FADD×N; MF(2); GO; LD.CLOSE` | N=0/16/64均通过 |
| split10 | `OPEN; MF(10); 9ST; GO; clock; FADD×N; MF(1); LD.CLOSE` | N=0/16/64均通过 |
| matched NOP | 正常MF替换为相同调度字段的NOP，保留全部payload/CLOSE | CUDA715 |
| MF0 replacement | 正常MF的count改成0，其余字节不动 | CUDA715 |

正例覆盖1/32 active lanes、13条组，19条组另有split9/split10对照；后者第二组count相应是8/7。负例覆盖两GPU、1/32lanes、13/19条组、NOP/MF0替换，各重复3次独立process/context，**48/48均在首次launch返回CUDA715**，无timeout。完整 [负对照矩阵](../results/phase19/final-mf-controls.json) 及原报告在 [records.zip](../results/phase19/records.zip)。错误从synchronize返回，没有fault PC，不能声称已定位到哪条ST/GO。

能得出的结论：

- **融合组边界可以不同于请求边界。** 一个有OPEN/CLOSE的请求可以拆成多段正确计数的融合组，中间执行普通计算；不能再把“整个请求必须在一个MF组里”当作硬件规则。
- 非零count有实际作用。独立MF0能执行，不意味着MF0能替代正常请求的MF；NOP也不能直接代替。两种负例同时改变了组归属/准入条件，尚不能将715唯一归因于“ticket未就绪”或“缺少宏内标记”。
- GO之后、LD.CLOSE之前已经可以插计算，随后仍取回正确结果；**LD.CLOSE不必与GO处于同一个融合组**。这比只观察完整组结束后的重叠，更直接分离了提交、读回与完成等待。

GPU5、32lanes、13条组的lane0 steady中位数（GPU1相同，单位SM周期）：

| 布局 | A→B：完整提交窗口 | B→C：剩余工作/TTU等待 | A→D：总窗口 |
|---|---:|---:|---:|
| normal | 199 | 398 | 617 |
| split9，gap16 | 232 | 365 | 617 |
| split9，gap64 | 424 | 353 | 797 |
| split10，gap0，含GO后clock | 199 | 398 | 617 |
| split10，gap16 | 256 | 341 | 617 |
| split10，gap64 | 448 | 149 | 617 |

split10的A→GO后clock均约183周期，随后64FADD提供256周期工作，总窗口仍617：观察支持GO后已有异步进展，晚发LD.CLOSE仍可取回结果。split9把gap放GO之前，较长gap会增加总时间；但小gap也能被隐藏，不能据此断言全部遍历何时启动，或把差分直接叫MF延迟。

[US11138009B2](https://patents.google.com/patent/US11138009B2/en)描述OPEN异步申请资源、MF等待连接就绪及有限组的抢占/单步保护，与上述结果相容。**本实验没有触发CILP或单步调试，仍未验证中断屏蔽、跨warp不可交错或整个SM锁定。** MF也不是让多条TTU在一个周期内发射的证据。详见 [Phase18专利摘录](PHASE18.md)。

## FADD、Tensor和访存各测了什么

FADD是普通SM的FP32加法指令。它只代表这一种算术工作，不能代表整数、Tensor或访存流水线。

| 工作 | 实际SASS与功能检查 | 限定 |
|---|---|---|
| FADD | R88依赖链，每条+1，终值核对 | 每条固定4-cycle spacing |
| Tensor | `HMMA.16816.F32`，FP16输入全1、FP32四累加器；每条各加16 | 必须32 active lanes；串行累加，非Tensor峰值吞吐 |
| global load | `LDG.E.64.STRONG.GPU`追踪64节点指针环，间隔128B，检查最终地址 | `.cg`对应策略，绕L1；热小环、同warp各lane同址，不能当DRAM带宽测试 |
| shared load | `LDS`追踪每lane两节点环，地址`lane*4 ↔ lane*4+128`，检查奇偶终值 | 无bank conflict；256B dynamic shared；存储初始化在计时外 |

每类工作有独立运行、TTU等待前运行、TTU等待后运行三种对照；匹配组和寄存器声明。当前没有global store、shared带宽压力、冷大working set或跨warp竞争矩阵。

### 时间窗口与完成检查

```text
恢复请求输入、累加器和指针；初始化shared（计时外）
A=clock
完整有效TTU组，或显式拆组例子
B=clock，无TTU等待
before-wait工作，含自身完成/立即消费
C=clock，等待TTU SB1
立即复制R72/R73/R32/R33，稍后与最终raw逐位核对
可选after-wait工作，含自身完成/立即消费
D=clock
逐lane快照，store read barrier保护源寄存器
```

TTU结果SB1、内存SB2、shared初始化读依赖SB3、HMMA设置SB5、快照SB0分开。C只等SB1，避免wait-all把内存和TTU混为一个未知等待。LDG/LDS末尾立即复制结果，HMMA末尾四条FADD立即消费四个累加器，不能只等很晚的global快照看终值。

**HMMA的SB5设置不是已解决的同步协议。** pilot仅设置/wait SB5、没有足够静态间隔时，最终HMMA结果正确，立即FADD消费者却读到旧值。正式版本保守地给每条HMMA至少32个scheduled cycles（15+15+2，包含两个NOP），再等待并消费；所有消费者都必须正确。因此测到的是带padding/检查的Tensor工作包，与TTU重叠成立；不能将这些周期解释成单条HMMA本征延迟，精确依赖/调度仍待研究。

公开资源为128 registers/thread、0 stack/local/static shared；launch另给256B dynamic shared，不能漏报。一个CTA、一个warp、4个展开sample、每case12次launch；每次TTUST全部输入恢复。四条TTU即时copy增加固定16周期，所以不要直接把Phase19的617与Phase18的602差值解释成RT变慢。所有lane同ray，尚无发散ray或多个未完成TTU请求。

### 正式重叠结果

[最终v3矩阵](../results/phase19/final-v3-mixed-matrix.json)：GPU5/1，共152case、1824次正常launch、5184次warp请求、130176次lane-ray、194784次逐lane检查，全通过。另有48次预期fault首次launch，不混入正常launch数。全部冷/热样本均检查，摘要只用lane0、排除首launch与每launch首sample，33个steady观测。共享服务器未锁频或证明物理独占，不能把周期当通用常量。

以下为GPU5、32lanes、13条组的总窗口中位数，单位SM周期；基线617，GPU1的FADD/Tensor/shared行相同，global因内存时钟/路径等条件有差异：

| 独立工作 | TTU等待前 | TTU等待后 |
|---|---:|---:|
| FADD×64 | 617 | 873 |
| HMMA×8（带padding/消费者） | 617 | 922 |
| HMMA×32（同上） | 1385 | 1779 |
| global load×1 | 617 | 860 |
| global load×2 | 699 | 1093 |
| global load×4 | 1215 | 1609 |
| shared load×16 | 617 | 991 |
| shared load×32 | 966 | 1359 |

这些位置对照支持**同warp在TTU结果尚未完成时，可执行实际Tensor及global/shared读工作，并隐藏部分完成时间**。global×2在GPU1为771/1166，方向相同。独立global基线前后位置也有约20周期差异，TTU/快照/初始化会改变发射与缓存条件，不能将表中差值全归因于固定RT成本。

工作较短时总窗口可保持基线，较长时超出等待窗口。**没有证明RT与Tensor/访存无竞争**：单warp、热小AS、依赖链远未饱和，padding/消费者也计时。下一步宜保持正确性检查，缩减观测扰动后加入第二warp，分别测两方向吞吐，再扩大working set判断L1/L2/下游资源影响。

## 校准负结果、SASS与复现

正式raw和额外四份SASS验证报告都在 [records.zip](../results/phase19/records.zip)，另保留三份代表失败：

- `consumer-hmma1.json`：HMMA终值正确、立即消费者前三项错误，说明只看末尾snapshot会误判完成时间。
- `pilot-none-shared-4-normal-0.json`：LDS类型/寻址字段错误，生成了`LDS.U8 …X4`，功能oracle失败；已修正为32bit、无缩放LDS。
- `pilot-v3-shared1.json`：setup的IMAD addend在过短间隔读到旧R110，shared环未建立；计时外增加保守间隔后通过。

两份早期完整矩阵没有立即消费检查，不采用其性能数字，留在ignored local。上面三份失败是旧prototype，source hash与现源码不同；保留证据不表示当前脚本能逐字复现旧版。`reference/jit/phase19-{hmma,global,shared,none}.{sass,json}` 保存实测cubin来源hash、普通SASS及融合槽解码；只为反汇编将TTU替换NOP，**不可运行该替换版本**。原含进程VA的cubin留local。`tools/inspect_mixed_timing.py`检查融合体内仍为TTU指令。

额外用compute-sanitizer memcheck验证不含TTU的global/shared普通SASS路径，各2launch×2sample×32lanes，均0错误；仪器化周期不纳入性能表。日志与报告留 `results/local/phase19/{global,shared}-memcheck.{txt,json}`，未计入211份发布集。

先查空闲GPU，命令从ttu-bench根目录执行：

```bash
CUDA_VISIBLE_DEVICES=5 timeout 25s python3 bench.py run_mixed_timing --group reduced --lanes 32 --kind hmma --count 8 --placement before-wait --check --output results/local/tensor-before.json
CUDA_VISIBLE_DEVICES=5 timeout 25s python3 bench.py run_mixed_timing --lanes 32 --variant split10 --gap 64 --check --output results/local/split-go-read.json
CUDA_VISIBLE_DEVICES=5 timeout 25s python3 bench.py run_mixed_timing --lanes 32 --kind global --count 2 --check --output results/local/global-before.json
python3 bench.py validate_mixed_timing --gpus 5 1 --output results/local/mixed-matrix.json
python3 bench.py validate_macrofuse_control --gpus 5 1 --output results/local/mf-controls.json
python3 bench.py result_store verify
```

负对照runner预期非零，自动矩阵校验明确的715，timeout不算通过。不重置共享GPU。结果发布按 [RESULTS.md](RESULTS.md)：本阶段211份证据无损发布为5个物理文件（单ZIP含207份原报告，外加两个矩阵、observations、资源）。
