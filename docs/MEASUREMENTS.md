# 单warp测量：异步重叠与MACROFUSE

RTX3090 / SM86、driver535.309.01、CUDA12.1。使用热小opaque GAS、一个CTA/warp，1或32 active lanes、所有lane同ray，每次恢复输入并核对输出。是有观测开销的串行请求实验，尚无多warp准入、多个未完成请求的饱和吞吐或资源竞争结论。

## 协议依据与实测

[US11138009B2](https://patents.google.com/patent/US11138009B2/en) §0221–0232/Fig.14A描述OPEN异步申请资源，MF等待连接就绪并将后续有限N条接口指令纳入抢占/单步保护；§0200–0206/Fig.12描述READ先承诺日后写寄存器、CLOSE后目标操作仍可进行。来源摘录在 `reference/jit/macrofuse-patent-excerpts.json`。这是专利示例，不是本机产品ABI或CILP实测。

已测的请求可以这样分组，组间允许普通计算：

```text
OPEN → MF11 → 9ST / GO / LD.CLOSE
OPEN → MF9 → 9ST → FADD gap → MF2 → GO / LD.CLOSE
OPEN → MF10 → 9ST / GO → FADD gap → MF1 → LD.CLOSE
```

正常MF替换匹配NOP或MF0，两GPU、1/32lanes、13/19组、3重复共48项均首次launch CUDA715，无timeout。独立MF0能执行，不等于可以替代正常MF。错误由sync返回，无fault PC，不能区分ticket未就绪与组归属校验。

结论是：融合组边界不必等于请求边界，GO后、LD.CLOSE前也可计算；MF有实际作用。尚未验证抢占屏蔽、跨warp不可交错、整个SM锁住，或多条TTU一个周期发射。

## 时间窗口与正确性

```text
计时外恢复请求/累加器/指针，初始化shared
A = clock
有效TTU组或合法拆组
B = clock（不等TTU）
before-wait独立工作及自身完成检查
C = clock（等待TTU）
立即复制TTU四个主要返回，随后与最终快照核对
after-wait独立工作及自身完成检查
D = clock
计时外逐lane快照，store read barrier保护源寄存器
```

总窗口含时钟、接口、依赖、padding/消费者，不是纯GO→结果延迟。初版 `src/warp_timing.py` 用96 registers、0 stack/local/shared、4个展开sample，C为wait-all。后续 `src/mixed_timing.py` 用128 registers、256B dynamic shared，TTU SB1、内存SB2、初始化SB3、HMMA SB5、快照SB0分开；C只等SB1。即时copy带来固定观测工作，两个版本的总周期不能直接相减成RT变化。

| 工作 | 实现与检查 | 限定 |
|---|---|---|
| FADD | 独立R88加1依赖链，4-cycle spacing，核对终值 | 普通FP32 ALU工作 |
| Tensor | HMMA.16816.F32，FP16全1、四个FP32累加器，每条+16；FADD立即消费四项 | 全32lanes；串行累加，非峰值吞吐 |
| global | LDG.E.64.STRONG.GPU，64节点指针环、128B间隔，立即复制终址 | 绕L1的热小环，同warp各lane同址，非DRAM带宽 |
| shared | LDS，lane*4与lane*4+128两节点环，核对奇偶终值 | 无bank conflict，256B dynamic shared |

重要校准反例：HMMA仅设置/wait SB5时，最终snapshot正确，立即FADD却读到旧值。正式版每条HMMA至少32个scheduled cycles（含两个NOP）再消费；因此只能证明带padding工作包的重叠，精确HMMA依赖尚未解决。其他已纠正夹具问题包括STG源寄存器复用、LDS误编码成U8/X4、setup IMAD过早读旧值；不将其命名成RT新机制。

## 代表结果

初版lane0 steady中位数，单位SM周期：

| lanes / 组 | 完整组窗口 | 剩余等待窗口 | 总窗口 |
|---|---:|---:|---:|
| 1 / 13条 | 127 | 250 | 381 |
| 1 / 19条 | 148 | 301 | 453 |
| 32 / 13条 | 200 | 398 | 602 |
| 32 / 19条 | 237 | 546 | 787 |

单lane/13条的32 FADD放等待前/后，总窗口381/509；32lanes/13条的64 FADD为602/858。完整融合组结束不等于结果完成，同warp独立计算可隐藏部分等待。额外LD改变返回路径，不能将19与13差值除6解释成固定LD延迟；32lanes也不能推算内部执行宽度。

带即时消费者的后续版本，32lanes、13条组基线617周期：

| 独立工作 | 等待前 | 等待后 |
|---|---:|---:|
| FADD×64 | 617 | 873 |
| HMMA×8（含padding） | 617 | 922 |
| HMMA×32 | 1385 | 1779 |
| global load×1 | 617 | 860 |
| global load×2 | 699 | 1093 |
| global load×4 | 1215 | 1609 |
| shared load×16 | 617 | 991 |
| shared load×32 | 966 | 1359 |

split10在GO后放64 FADD，总窗口仍617；split9把同样gap放GO前，总窗口797。支持提交后存在异步进展，尚不能独立定位ticket申请/释放或全部遍历启动时刻。OPEN后、MF前短gap也可被隐藏，与专利相容；不把差分叫MF固有延迟。

初版172项/136224次逐lane检查、后续152项/194784次检查全通过，负对照48项另外计数。摘要仅lane0、剔除首launch/首sample；冷样本仍做正确性检查。两卡多数行一致，global在另一卡为771/1166周期，方向相同。服务器未锁频，也未证明物理独占，周期不是通用常量。

这些结果证明同warp的实际Tensor/shared/global读可与未完成TTU工作部分重叠，**不证明无竞争**。负载未饱和、AS很小、依赖链及检查开销都在计时中。HMMA最终快照正确仍可能误判即时完成，是值得保留的方法反例。

## 复现与下一步

```bash
ttu_output_dir=$(mktemp -d /tmp/ttu-measurements.XXXXXX)
CUDA_VISIBLE_DEVICES=0 timeout 25s python3 bench.py run_mixed_timing --lanes 32 --kind hmma --count 8 --check --output "$ttu_output_dir/hmma.json"
CUDA_VISIBLE_DEVICES=0 timeout 25s python3 bench.py run_mixed_timing --lanes 32 --variant split10 --gap 64 --check --output "$ttu_output_dir/split.json"
CUDA_VISIBLE_DEVICES=0 timeout 25s python3 bench.py run_mixed_timing --lanes 32 --kind global --count 2 --check --output "$ttu_output_dir/global.json"
python3 bench.py validate_macrofuse_control --gpus 0 --output "$ttu_output_dir/mf-controls.json"
```

先选择空闲GPU，完成后清理任务临时目录；禁止直接向`/tmp`根写文件。预期fault用独立process/context，timeout不算通过，不重置共享GPU。已发布矩阵与重要校准反例保留在 `results/phase18` / `results/phase19`，通过 [结果loader](RESULTS.md) 读取。原报告hash不改，不再保留本机重复副本；普通SASS在 `reference/jit/`，TTU换NOP版本不可运行。

下一步按 [交接](HANDOFF.md) 从紧凑多warp做起：核对SMID/驻留、匹配无TTU对照、测双向吞吐，再扩大working set。渲染速度、单warp重叠和active-lane扫描都不能替代资源归因。
