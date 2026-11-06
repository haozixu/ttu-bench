# SM86 TTU协议与功能边界

本机环境：RTX3090 / GA102 / SM86，driver535.309.01、CUDA12.1、OptiX8。以下是这套环境中的经验协议；基础图形概念见 [入门](TTU_PROTOCOL_GUIDE.md)。不把固定fixture、opaque GAS和完整OptiX运行时混为一谈。

## 独立发射与AS

`run_minimal`在全新CUDA context中分配AS/dummy root，以自研kernel发射TTU；不创建OptiX pipeline，不调用驱动megakernel。固定三角形来自 `fixtures/optix_as.bin`，原始指令和128B dummyRootNode来自 `fixtures/jit_16.elf`。原驱动module不被加载，dummy包含单位变换数据。AS用普通高VA的cuMemAlloc分配，本fixture修正AS+0xd0自指针即可使用；不证明任意AS都可如此重定位。

固定fixture的地址打包为：

```text
R50 = uint32(dummyVA >> 6)
R51 = uint32(dummyVA >> 38) | 0x02402800
R54 = uint32(rootVA >> 7)
R55 = uint32(rootVA >> 39) | 0x02484000
```

R50曾额外OR8，后来对照JIT证明它是原现场的地址位，已经移除。生产GAS由OptiX在当前context内构建并持有，裸TTU只消费原分配；CPU和单独OptiX追踪作oracle。经验root偏移从AS header+0x20读取，检查对齐/边界；GAS曾见0x300，IAS曾见0x380，不是通用handle解码。

固定fixture验证16case/80launch，跳过TTU及清零dummy指针均不能通过hit oracle。生产GAS随后验证一般方向、非单位方向、tmin/tmax、opaque和有限culling flags；一般ray与flags矩阵共20,208次launch。规则GAS到4096面，renderer的Bunny到69,453面，均为测试规模，不是硬件容量或每ray访问节点数。我们没有解码树拓扑，也没有节点访问轨迹。

## 指令组与编码

| 组 | ST | GO | LD | MF count | 已知范围 |
|---|---:|---:|---:|---:|---|
| 原始20条 | 10，含互斥P4/!P4 | 1 | 7，最后CLOSE | 18 | 固定fixture通过；生产GAS可重复CUDA700 |
| expanded 19条 | P4=false专门化后9条PT | 1 | 7，最后CLOSE | 17 | opaque GAS及受限IAS诊断 |
| reduced 13条 | 9 | 1 | 仅[0x300]，在此CLOSE | 11 | 已测triangle最近命中子集 |

20→19同时改变predicate、MF count、指令位置和输入准备，尚未单独定位20条组故障根因。19→13少读6槽并提前CLOSE；10,104对主要返回逐位一致，但其他槽确实用于pending状态，不能删作无用。13条不是理论最短，指令数不等于内部工作量；两者的计时差见 [测量](MEASUREMENTS.md)。

13条的提交槽如下，每条ST有两个显式寄存器对：

```text
OPEN → MF(11)
ST[000] R42:R43, R52:R53    ST[040] R12:R13, R66:R67
ST[050] R16:R17, R14:R15    ST[080] R72:R73, R40:R41
ST[0c0] R36:R37, R24:R25    ST[0d0] R32:R33, R2:R3
ST[100] R34:R35, R50:R51    ST[180] R56:R57, R54:R55
ST[190] R60:R61, R58:R59
GO → LD.CLOSE[300] R72:R73, R32:R33
```

TTUST寄存器对起点在绝对bits39:32、71:64；TTULD在bits23:16、71:64。每对内部相邻，原有效组含R2/R50/R54/R58，否定“必须4对齐”。CLOSE是绝对bit74，第二64-bit word的bit10。MF count是bits44:40。helper在 `src/ttu_isa.py`，固定知识在 `src/sm86_fixture.py`；不要从某次寄存器分配推导通用ABI。

返回同步还依赖SASS调度字段：本fixture的TTULD写SB1，普通消费者用wait mask2等待。改写SB2/SB4后，等待必须随之改为4/16；没有等待或等错槽，即时消费者可读旧值而最终dump仍正确。13条组最后LD继承了原非末条的stall1；紧接wait前至少需要经校准的调度间隔，目前stall2在已测范围通过。CLOSE、MF结束都不能替代返回依赖等待；详见 [同步与跨warp对照](MEASUREMENTS.md#返回同步拆组与吞吐归因)。

## Ray、flags与triangle返回

| 输入 | 寄存器 |
|---|---|
| origin xyz | R66/R67/R12，另有R24/R25/R36 |
| direction xyz | R14/R15/R16，另有R2/R3/R32 |
| tmin / tmax | R13/R37；R17/R33 |

方向不要求单位化，t是参数距离。face culling映射到R52 bits4/5，DISABLE_ANYHIT映射到R53 bit5，R40/R41由同一配置派生；不是将OptiX枚举直接写进任意槽。已测flags包括禁any-hit及正反面culling，不代表全部OptiX flags都已实现。

triangle子集中，R72低28位对上primitive，abs(FP32(R73))对上t；abs(FP32(R32))与FP32(R33)是u/v，R32符号0/1对应front/back face。opaque返回t曾为正，nonopaque候选曾为负；完整状态语义未解。miss为R72=R73=ffffffff，未知状态必须另报，不能低位像ID就强行解码。

旧decoder仅接受tag8/a，并固定对R73/R32取负，曾误报opaque flag1失败。符号规则已纠正；Bunny又观察到4/6，在renderer当前opaque子集经oracle核对后接受，含义仍未知。最小原始样本见 [counterexamples](../fixtures/renderer-counterexamples.json)。共享fixture decoder没有一并放宽。

重要反例：16面网格，ray=(.25,5.25,1.875)+(0,-1,0)t、geometry/ray flags0，CPU/OptiX最近命中primitive4、t=2，13/19条组返回primitive0、t=5且属性匹配该面。禁用any-hit后正确。这支持“返回候选后需要软件accept/ignore/续遍历”，尚未实现完整协议；不能把一次GO普遍等同完整closest-hit。无any-hit shader也不自动消除这层差别。

## Instance事件与父级恢复

观察到 `R33>>27 == 0x16` 的instance相关事件，此时R72/R73描述地址，不能当triangle ID/t。CUDA读取逆矩阵，变换ray并独立查child GAS；变换后direction不归一化，以保持t。device-instance矩阵8项、1344次launch/2615次请求通过，包含事件后child miss；事件本身不是几何命中。

当前解码：

```text
E = R72 | ((R73 & 0x1ffff) << 32)
index = uint32[E+4]                         # 1-based
M = IAS_base + uint64[IAS_base+0x30] + (index-1)*0x80
instance_id = uint32[M+0x74] & 0xffffff
child_handle = uint64[M+0x60]
child_root = uint64[E+8] & ((1<<49)-1)
inverse_transform = float[12] at E+0x10
```

索引上限来自IAS+0x70；host/device还核对分配边界、root和矩阵。曾将E-0x0c和E-0x20当ID/handle，**已撤回通用规则**；多实例中事件对应instance17，相邻旧位置却给41，单实例只是碰巧正确。JIT读取[E+4]、减1并按0x80查表的代码支持当前解码。

只查第一个child不足以求IAS最近命中：

| 反例 | 首child | 正确最终结果 | first-child-only负对照 |
|---|---|---|---|
| first-miss-later-hit | instance41 miss | instance17 t=5 | 错报miss |
| first-farther-later-nearer | instance41 t=5 | instance17 t≈4.128019 | 错留远处命中 |

19条组的R54:R55提供观察到的续遍历基址/配置，`(R55>>25)&7`是当前显式pending数量，R56:R57、R58:R59、R60:R61是条目。查询child前，SM软件栈保存它们；已实现类型 `((high>>17)&0x1ff)==0x21`。恢复时base_high保留配置但数量设1，输入R50:R51为基址，R34:R35为条目，R54..R61清零，世界ray的tmax缩至当前最近t，然后发一个新请求。更多pending继续保存；不是枚举builder所有instance求min，也不是保持同一次OPEN。

父级矩阵覆盖平坦1–3静态instance、正行列式变换、共享triangle GAS、flags1和0x21引用，两GPU30项/4768次launch/11492次请求通过，含1484次父级恢复请求。实现有16项软件栈、最多8个父请求；这些不是硬件容量。功能容器单active thread、88 registers/848B stack，不能拿来测最小TTU延迟。

复杂IAS逆向暂挂：四实例出现0x24引用，重提后直接返回triangle，instance归属/坐标语义未解；等距重叠有51项身份检查失败而t相同，不能扩大容差改成通过。不同child GAS、嵌套、负行列式、motion、nonopaque accept/ignore、custom primitive仍未完成。相关代码/原始反例保留，恢复时从这些边界继续。

## 实现与复现

`src/device_instance.py`负责编译器marker映射和TTU patch；`src/instance_protocol.py`负责记录解码；`src/parent_resume.py` / `kernels/parent_template.cu`负责设备端pending循环。`run_parent_probe`的copy/promote/pop是host诊断假设，不是通用算法。原helpers和EoSS线保留，见 [EoSS](EOSS_REPRO.md)。

```bash
CUDA_VISIBLE_DEVICES=0 timeout 20s python3 bench.py run_minimal hit --check
CUDA_VISIBLE_DEVICES=0 timeout 30s python3 bench.py run_scene overlap --ray-suite general --group both --check
ttu_output_dir=$(mktemp -d /tmp/ttu-protocol.XXXXXX)
CUDA_VISIBLE_DEVICES=0 timeout 40s python3 bench.py run_parent_resume --scenario first-farther-later-nearer --check --output "$ttu_output_dir/parent.json"
```

先选择空闲GPU，完成后清理任务临时目录。固定fixture独立于OptiX SDK；生产AS需要本地SDK/nvcc/g++。关键旧证据按数据集编号保留在 `results/phase12` 至 `results/phase17`，读取旧case用 [结果说明](RESULTS.md) 中的loader；原报告字节和源码hash不改。JIT分析在 `reference/jit/`，TTU换NOP的反汇编资产只作静态参考，不能运行。

此外撤回的旧判断包括“必须低4GB”“缺驱动私有现场”“-1是拒绝码”“control byte0必须零”。旧runner曾未实际改ray、NOP掉local spill，no-TTU夹具本身也曾fault；这些不足以推导TTU硬件限制。原始JIT/AS与可恢复历史源码仍保留，日常不再维护失效路径的复现清单。
