# TTU测量：异步重叠、MACROFUSE与多warp

RTX3090 / SM86、driver535.309.01、CUDA12.1。单warp实验使用热小opaque GAS、一个CTA/warp，1或32 active lanes、所有lane同ray，每次恢复输入并核对输出。后续同CTA多warp基线见末节；当前没有具体共享资源的竞争归因。

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

## 多warp基线与依赖校准（2026-11-06）

`src/multiwarp.py`用单CTA、1/2/4/8/16个完整warp，每warp串行循环32/128/512次，同lane同ray，每次至多一个未完成TTU请求。计时前后CTA barrier；每个warp记录clock64起止，CTA窗口为最晚结束减最早开始。计时循环含返回覆盖输入的恢复、TTU组、保守等待、逐次四word XOR/OR错误累计及循环控制；最终逐lane检查返回、错误累计、计数归零、TID、前后virtual SMID和输出guard。每CTA固定驻留同一SM，virtual SMID检查不能定位SM内部调度分区。

96 registers、0 local/stack/shared；Driver occupancy API报告1/2/4/8/16 warp配置最多可驻留16/10/5/2/1 CTA，但实际只发一个CTA，不把API上限当作实际并发CTA数量。输出只在整个循环结束后写，每lane一份；`--artifacts`才另存cubin/raw。模板预留barrier资源，128-bit快照最高源R88..R91，上方保留寄存器余量。

这里的lane指warp内编号0–31的线程；一次lane请求就是该线程发起的一次ray查询。所有lane使用相同ray，计数仍按各线程各自的查询累计，不代表不同ray的数量或硬件内部事务数。

两卡GPU0/6各45项、每项12次记录launch，合计31,997,952次计时内ray请求均通过，另有≥0.2秒预热且同样校验；汇总剔除前两launch。未锁频、未证明物理独占。每warp512次时，表格表示**每1000个SM周期完成的ray请求数，整个CTA合计**，即`1000 × warps × 32 × iterations / CTA计时窗口`；无TTU列计数的是合成返回次数：

| warps | 13条组（两卡范围） | 19条组（两卡范围） | 无TTU合成循环 |
|---|---:|---:|---:|
| 1 | 46.8 | 35.2 | 314 |
| 2 | 81.6 | 67.0 | 627 |
| 4 | 160.2 | 108.4 | 1255 |
| 8 | 162.9–164.6 | 109.2–109.3 | 2188 |
| 16 | 164.4 | 105.1–105.2 | 2997 |

无TTU对照用四个MOV合成同样返回，保留恢复/检查/循环等，但不匹配TTU指令数量。不能将它与TTU周期直接相减得到硬件成本。默认13/19循环恢复4/24个输入word，这也是两组差别；8 warp改成恢复全部36个输入word仍通过，每1000个SM周期13条约164次、19条约109次，方向未变。只做末次检查也保留趋平现象，但它不能证明中间每次正确。hit/miss/tmax截断及两种输入恢复模式另有通过对照。

目前支持：增加warp可以隐藏串行请求等待；当前热小GAS请求循环在4–8 warp后收益很小，13/19组的稳定吞吐不同。**尚不能确定限制来自TTU准入、返回接口、普通指令发射还是组合，也不能把平台值称为RT Core峰值或队列容量。** 每warp起止窗口有共同交集，但不能由此精确计算同时在硬件内的请求数。

SM86每SM有四个SMSP，因此调度分区的发射、依赖处理或局部TTU接口先受限，也是候选解释。4–8 warp接近每SMSP 1–2 warp只是数量上的相容性，不能由此定位瓶颈；后续共同窗口实验已获得模4调度竞争类，并发现只用两个类也能达到平台，见下节。尚未测得各分区利用率。

### 新的等待校准反例

13条组中，LD.CLOSE之后立即执行带wait-all的CS2R再读返回，8 warp时错误累计可非零，而最终返回快照已正确。对照中仅在wait之后增加15个scheduled cycles仍失败；在LD与wait之间加1/2个NOP则通过，19条组及无TTU组在对应组合通过。GPU1/6均观察到，保留最小样本在 [fixture](../fixtures/multiwarp-wait-counterexample.json)。

这撤回的是“本夹具紧接wait即可安全消费”的假设，不是13条组的全部功能验证。可能涉及scoreboard登记/依赖时序，但没有fault PC或内部轨迹，不命名成硬件机制。默认保守使用两个2-cycle NOP、wait-all CS2R及额外15-cycle NOP；这些时间全部计入窗口，尚未证明是最小充分延迟。逐次检查不能由最终快照替代。

13条组的范围是`OPEN → MF11 → [9ST / GO / LD.CLOSE]`，随后才是上述NOP与带wait的CS2R，wait在MF11范围之外。wait是普通指令控制字段中的依赖等待掩码，不是另一个TTU操作。反例表明当前手写调度不能保证紧接wait后的消费者看到有效返回；它不能确定GO何时被TTU接受、LD的返回承诺何时登记，也不能据此声称请求尚未提交。即使结果消费者位于融合范围内，融合本身也不能替代返回就绪所需的正确依赖处理。

```bash
ttu_output_dir=$(mktemp -d /tmp/ttu-multiwarp.XXXXXX)
CUDA_VISIBLE_DEVICES=0 timeout 25s python3 bench.py run_multiwarp --warps 8 --iterations 512 --output "$ttu_output_dir/baseline.json"
python3 bench.py validate_multiwarp --gpus 0 --output "$ttu_output_dir/matrix.json"
CUDA_VISIBLE_DEVICES=0 timeout 25s python3 bench.py run_multiwarp --warps 8 --iterations 1 --launches 3 --warmup-seconds 0 --wait-lead 0 --consumer-delay 15 --output "$ttu_output_dir/expected-failure.json"
```

最后一条预期非零退出，硬件不应fault；默认等待控制应通过。先用nvidia-smi确认所选卡空闲，完成后删除任务目录。普通矩阵只留命令与本节结论。

## 返回同步、拆组与吞吐归因

### 同步依靠返回依赖槽

本文的SB是scoreboard dependency barrier的简称。SM86 SASS用编号0–5的依赖槽及6-bit等待掩码跟踪变延迟指令；它们是每warp的指令依赖状态，不是六个RT队列，也不是跨warp共享的同步对象。SB1/SB2/SB4分别指槽1/2/4，等待掩码为`1<<1=2`、`1<<2=4`、`1<<4=16`；编号可以由代码分配，没有“SB1专属TTU、SB2专属访存”的固定功能。本项目只是选择这些编号来分开工作包。

可以把生产者的write dependency barrier理解成“结果尚未就绪时，在这个编号下记账”，消费者的wait mask则是“等相应依赖满足后再执行”。这里的write表示跟踪结果写回依赖，不是往一个软件可读的barrier地址写数据。TTUST还使用read dependency barrier，保护异步操作读取的输入寄存器；它与TTULD的返回写回依赖要区分。上述机制不等于内存fence，也不自动同步其他warp。

`stall1/stall2`指指令控制字段中的静态stall值1/2，用于限制同一warp后续指令的发射间隔。它不等待某个事件、不保证RT工作在1/2周期内完成，也不让整个SM停下来；其他warp仍可被调度。实际间隔还能因依赖或资源变长。固定stall与动态依赖等待共同保证手写SASS的调度正确，不能把两者互相替代。

这些术语的主要来源是 [Dissecting the NVIDIA Turing T4](../../research/papers/turing-t4-microbenchmarking.pdf) §2.1，印刷页11–12：六个dependency barrier、read/write索引、wait mask和stall cycles均有解释。当前编码另可核对 [EoSS control.py](../../research/community/eoss-framework/enc/control.py)：upper64的stall在bits41–44，write/read索引在46–48/49–51，wait mask在52–57。SB1/SB2/SB4是本文给槽位起的简称。PTX是交给ptxas继续编译的虚拟ISA，这些控制信息属于编译后的SASS机器指令；普通CUDA/PTX开发通常不必手动处理。公开逆向资料给出机制和字段，本机TTU对照验证其具体用法，不把Turing描述或社区实现当作SM86完整官方ISA保证。

TTULD的SASS控制字段带write dependency barrier索引；原fixture为SB1，消费者控制字段用`wait_mask=1<<1`等待。这里没有建立或等待共享内存中的mbarrier对象：它是warp指令依赖层的同步。给CS2R加wait并非读取时钟本身有同步语义，NOP或直接读取返回的XOR指令携带相同wait也可实现等待。

两张卡的同步矩阵66项符合预期，含14项读旧值的负对照，其余52项通过；最小错误同步对照用单次请求，要求即时检查失败而最终返回正确：

| LD控制 / 消费者等待 | 结果 |
|---|---|
| LD stall1，紧接wait SB1 | CS2R、NOP、首个XOR三种位置均有即时检查错误 |
| LD stall2或4，紧接wait SB1 | 三种位置均通过，无额外前后NOP |
| LD写SB1，wait0或SB2 | 失败，最终返回仍可正确 |
| LD改写SB2，wait SB2 | 通过；仍等SB1则失败 |
| LD改写SB4，wait SB4 | 通过；仍等SB1则失败 |

较紧的`LD stall2 → 首个XOR wait SB1`另覆盖13/19组、hit/miss/tmax短截、1/4/16 warp、128次循环。更长吞吐与角色实验继续逐次核对结果。以上足以把**当前请求的结果可消费同步**定位到匹配的返回依赖等待；也证明只有固定短延迟不能替代它。stall2是本组实测通过的调度设置，不是TTU计算延迟、硬件通用最小值或正式ABI。原19条组最后一条LD本就为stall2，13条组提前CLOSE时继承了第一条LD的stall1；这是旧反例的直接调度差异，内部登记时序仍未确证。

错误同步若连续循环，会提前恢复被LD覆盖的输入，最终快照也可能变成miss；这不能再当作只影响即时消费者的最小反例。因此负对照限制为一次请求，较长循环只用于正确同步的验证。所有请求仍是固定ray，逐次结果相同的检查不能单独排除读取上一轮结果，依赖同步结论主要来自首请求旧值、匹配槽位及即时消费者对照。

默认入口保留旧的保守NOP/wait设置；要复现上述较紧设置，显式传`--ld-stall 2 --wait-lead 0 --consumer-delay 0 --wait-mask 2 --wait-site consumer`。它仍在MF范围外。结果就绪不等于查明GO接受时刻、CLOSE释放时刻或内部队列状态。最小同步和新请求推进样本见 [协议对照](../fixtures/multiwarp-protocol-controls.json)。

### 保持一个连接不阻止其他warp推进

单CTA内warp0执行2048个依赖FADD，约8200周期，放在OPEN后MF前、split9的ST后GO前，或split10的GO后LD前。其他warp先延迟256个FADD再开始请求，使它们的OPEN发生在warp0的间隔内。时间戳只放在MF范围之外，最终返回和FADD计数都核对。

两卡、2/4/8/16 warp，延迟和不延迟及匹配位置控制共80项通过。延迟启动时，最多15个其他warp的`after_open → after_close → after_wait`均落在warp0间隔内；特别是split9/10的间隔在首个MF组完成之后。这排除了“一个尚未CLOSE的连接独占整个SM、其他warp必须等它关闭才能发起并完成请求”的解释。它不证明15个请求同时驻留，也不测ticket容量；OPEN标记本身不是分配完成证明，后续有效结果才证明请求实际推进。

未延迟启动时，16 warp下只有部分peer在该窗口内完成，说明开始顺序影响观测；不能将某一次的完成数量解释成容量。这里也没有把普通指令插进一个完整融合组，因而仍不回答融合组执行期间的逐指令跨warp交错或抢占问题。

### 平台对返回协议敏感

使用较紧的正确等待后，13条组在8–16 warp仍约164次查询/1000周期，19条约109/105；旧的额外NOP不是平台主因。恢复全部36个输入word后，平台方向相同，无TTU对照仍远快于TTU循环，普通恢复/检查开销不足以独立解释平台。

进一步固定9ST、GO、ray、AS及全部36word恢复，只保留原19条组前N条LD并在末条CLOSE。少掉的LD另用MF之外的2-cycle NOP补齐，比较相同总指令数。两卡的1/4/8/16 warp、N=1..7及8/16 warp补齐对照共84项通过。16 warp代表值：

| 返回LD数量 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 查询/1000周期，含NOP补齐 | 164 | 154 | 142 | 133 | 122 | 113 | 105 |

8 warp同样从约167降至109，补NOP前后接近。固定输入/几何工作时，返回协议足以显著改变吞吐，因而不能把13/19平台差异只归因于三角形求交能力或少了几条输入恢复。NOP控制匹配了数量和指定stall，却没有匹配返回流量、MF长度、提前CLOSE及连接生命周期；具体限制仍可能在SMSP侧TTU发射/返回接口、连接占用或共享TTU资源，尚未定位物理单元。

8 warp、全部输入恢复时，同样256个FADD放GO前约138–142次查询/1000周期，GO后LD前约162，结果等待后约136；短gap64大多能隐藏。支持GO后的异步推进，不将差分当作纯遍历延迟。`after-wait`在consumer等待模式下把gap放在即时结果检查之后，避免gap本身掩盖错误等待。

### TTU与计算warp的双向影响

同一CTA内一半warp逐次发TTU并检查，另一半做FADD；轮换连续的角色区间。两种计算包均核对所有累加器：单依赖链4-cycle spacing，及四条交织依赖链1-cycle spacing。每个角色记录自己的起止窗口，计时不含末尾CTA barrier；对照保留同一CTA/角色分配，只让另一角色执行最短工作后等待。另用无TTU合成循环控制普通发射开销。

单链角色矩阵58项通过。8 warp的4TTU+4FADD，TTU约149–152，相同角色的近独占对照约157–159；FADD平均吞吐变化小于1%。16 warp的8+8则TTU约163–166、FADD约56,200次线程加法/1000周期，没有稳定的大幅互相拖慢。

四链计算包另在两卡各24项通过。16 warp的8+8，TTU约162–163，对照约164–168；FADD约98,000，对照约101,000–102,000次线程加法/1000周期。TTU与FADD窗口交叠覆盖计算窗口几乎全部、TTU窗口约85%；8 warp下约75–81%。这些是完整窗口平均值，尾部单独运行会稀释干扰，也不是每周期内部同时执行的证明。未锁频且部分TTU对照有波动，因此只称轻微至中等影响，不给精确普适百分比。

综合而言，旧角色矩阵显示TTU平台仍有普通FP32工作的余量，但完整窗口含尾部，也未定位调度分区。下节用共同窗口和显式warp掩码收紧这两个限制。

这些矩阵按固定顺序执行，没有随机化配对或硬件性能计数器；吞吐数字用于选择下一项对照，不作精确竞争系数。更新后的普通无TTU/四链角色路径另通过compute-sanitizer memcheck，0错误；未对TTU指令做sanitizer插桩，不能据此声称它验证了TTU协议。

### 共同窗口与调度分组

`--window-cycles 300000`用4B动态shared发布同SM的clock低32位，所有活跃角色在epoch+4096以后开始，使用同一个结束deadline。每轮计时，统计结束前完成的轮数；跨过deadline的最后一轮仍检查结果，但不计入角色吞吐。默认固定次数模式保留。显式`--ttu-mask/--compute-mask`按CTA warp编号选角色，其余warp停在最终barrier，保持同一CTA驻留配置。计算包为256个FADD、四条交织依赖链；所有计时、检查和循环开销均在窗口内，不能与旧窗口相减为硬件延迟。

VIRTID按nvcc对PTX `%warpid`的实际生成序列取`(raw>>8)&127`，这里只叫virtual warp ID，不能将它直接当作SMSP ID。两卡上观察到同一次launch的CTA warp模4类保持一致，virtual ID可能整体轮换。另用纯FADD吞吐确认分组：16 warp CTA内，四个活跃warp取0/4/8/12（或其他三个模4类），合计约31,200次线程加法/1000周期；取连续四warp则约57,000。同样四个计算warp、同一共同窗口，前者明显受同一分区限制。这给出了本配置的SMSP竞争等价类，尚未给物理分区编号。

Turing论文§2.2/表2.1用两活跃warp的FFMA吞吐得出`warp_id%4`映射。本机两个FADD warp无论同/异模4类都约28,500，单warp约14,250，不能重复那个两warp判据；四warp才显出差别。GA102的SMSP有不同于Turing的FP32执行配置，当前结果也受发射/依赖/控制字段影响；不要由两warp无差别断言映射不存在，或直接据此反推执行端口数。

共同窗口角色对照的代表值如下。TTU单位为查询/1000周期，FADD单位为线程加法/1000周期；独占列仅指另一角色停在barrier，CTA配置相同。

| TTU / FADD的CTA warp集合 | TTU独占→混合 | FADD独占→混合 |
|---|---:|---:|
| {0} / {4}，同模4类 | 45.87→45.01 | 14,254→13,844–13,872 |
| {0} / {1}，异模4类 | 45.87→45.87 | 14,254→14,254 |
| {0,8} / {4,12}，同一类各两个 | 90.88→89.39 | 28,454→25,395 |
| {0,2,4,6} / {1,3,5,7}，分占两类 | 158.72→158.61 | 56,907→57,016 |
| 偶数8warp / 奇数8warp，分占两类 | 168.7–169.1→168.8–169.2 | 62,396→62,368 |

这比旧尾部平均更明确：FADD受到的影响随调度分组而变，支持存在局部分区内的交互；异组普通FP32压力没有复现明显退化。同时，仅两个模4类中的TTU warp已能达到约169的平台，因此“必须四个SMSP都满了才出现原平台”的解释不成立。仍不能排除各SMSP的TTU接口、共享返回路径或RT侧资源；这组实验没有定位到单个物理单元，也未证明不同类别之间完全隔离。计时未锁频，轻微正向差异不解释为加速。

新mapping/window矩阵在两卡运行，window含打乱case顺序的重复；return-controls也在两卡复测。较长角色共同窗口的最大启动迟到小于100周期，返回控制小于300周期，均远小于300000周期窗口。末轮不计造成的量化误差每warp至多一轮；未控制每周期内部指令交错。普通FADD共同窗口另过memcheck，0错误；sanitizer只作内存正确性检查，其插桩计时不用于以上比较。

### 固定七条返回指令与CLOSE位置

在全部36word恢复、9ST+GO、相同ray/AS的基础上，新增两种保留MF17、七LD及末条CLOSE的控制：`sink`保留前N个原始LD，其余目的寄存器对改成RZ；`repeat`把后续LD改成重复读取首个返回槽并写回同一结果寄存器。两种路径都逐轮检查主要命中word。RZ表示丢弃目的寄存器写回，不证明内部返回传输被省略；repeat同时改变返回槽选择和目的别名，也不能解释为相同字节的纯流量控制。

16 warp、共同窗口下的代表值：

| 原始返回LD保留数N | 1 | 2 | 4 | 7 |
|---|---:|---:|---:|---:|
| prefix：缩短MF并提前CLOSE | 164 | 156 | 134 | 105 |
| sink：七LD/末条CLOSE，其余写RZ | 118 | 111 | 108 | 105 |
| repeat：七LD/末条CLOSE，其余重复首槽/同目的 | 91 | 91 | 96 | 105 |

同样七LD和CLOSE位置，目的写回/槽位/别名选择仍影响吞吐，因此原prefix曲线不能只归因于MF长度或CLOSE位置。另一方面，sink(N=1)仍远低于prefix(N=1)，减少目的写回也不足以复现短组平台。repeat反而更慢，说明“返回有效字越少就一定越快”的模型也不够；可能的槽位服务或写回依赖限制尚未分离，不直接命名为返回带宽。

另将七LD统一拆成MF16+MF1，只有warp0在末条LD.CLOSE前或后执行相同1/64/256个依赖FADD。对照保持两段MF及指令数量不变。当前peer吞吐的前/后差异很小，warp0自身长gap在CLOSE前更慢；后者也可能来自延迟发出末条返回请求以及等待重叠，尚不能认定为连接释放成本。已有“保持连接时其他warp仍可完成”的时间戳对照仍成立。下一步若追释放点，需比较多个holder及单次请求的peer时间线；不靠这组小差异推断ticket数量。

### 复现

```bash
ttu_output_dir=$(mktemp -d /tmp/ttu-interactions.XXXXXX)
python3 bench.py validate_multiwarp_interactions --suite sync --gpus 6,1 --output "$ttu_output_dir/sync.json"
python3 bench.py validate_multiwarp_interactions --suite split --gpus 6,1 --output "$ttu_output_dir/split.json"
python3 bench.py validate_multiwarp_interactions --suite scaling --gpus 6 --output "$ttu_output_dir/scaling.json"
python3 bench.py validate_multiwarp_interactions --suite returns --gpus 6,1 --output "$ttu_output_dir/returns.json"
python3 bench.py validate_multiwarp_interactions --suite mixed --gpus 1 --output "$ttu_output_dir/mixed.json"
python3 bench.py validate_multiwarp_interactions --suite issue --gpus 6,1 --output "$ttu_output_dir/issue.json"
python3 bench.py validate_multiwarp_interactions --suite mapping --gpus 6,1 --seed 711 --output "$ttu_output_dir/mapping.json"
python3 bench.py validate_multiwarp_interactions --suite window --gpus 6,1 --repeats 2 --seed 717 --launches 10 --output "$ttu_output_dir/window.json"
python3 bench.py validate_multiwarp_interactions --suite return-controls --gpus 6,1 --repeats 2 --seed 716 --launches 10 --output "$ttu_output_dir/return-controls.json"
```

GPU编号按空闲情况替换。sync包含预期读旧值的负对照，只有符合预期且没有CUDA fault才算通过；timeout不算通过。各case报告自动删除，只输出一份摘要，常规摘要读完后清理任务目录。trace改变时序，不用于平台吞吐比较；两个FADD包都是可检查的工作负载，不宣称计算峰值。
