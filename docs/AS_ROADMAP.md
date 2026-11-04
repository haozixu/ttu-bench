# 从单三角形 TTU 到复杂 AS：探索路线

最新 [Phase17](PHASE17.md)：保存/重新提交硬件pending引用，已测平坦1–3个静态instance的父IAS续遍历与最近命中通过两GPU的30项矩阵、4,768次launch、11,492次请求。仅覆盖共享triangle GAS、正行列式变换、flags1及引用类型0x21；四实例已出现其他路径，不能宣称完整IAS。instance ID/handle改为按事件索引查IAS表，旧相邻偏移规则撤回。[Phase15](PHASE15.md)保留4096三角形GAS结果。

更新：2026-11-03。长期希望形成research contribution与paper；前期基础设施、可靠复现与blog也有价值，不以新颖性作为推进门槛。GAS最大已测4096三角形；一般方向、tmin、opaque与有限culling flags共20208次launch通过，见 [Phase14](PHASE14.md)。单instance设备功能见Phase16，受限多instance父级恢复见Phase17，通用IAS仍未完成。nonopaque多命中已有较远候选反例，不能把单次GO默认当作完整closest-hit；Phase18已有固定GAS的单warp计时与独立算术重叠对照，通用吞吐/竞争测量仍待开展。

公开结构资料并非空白：新补两份NVIDIA专利，给出complet/multi-box和node layout示例；但还没有对上本机535 AS的完整公开builder。具体图号、下载及软件CWBVH/AMD格式的区别见 [资料核查](../../research/notes/nvidia-bvh-public-records.md)。

## 代际变化：目前能说到哪里

本地只有SM86实机，SM120依据EoSS论文/artifact和Rosetta静态定义。完整比较8个TTU类：opcode、按operand alias归一化的位域及condition表达式均相同，原始比较与源文件hash在 `results/eoss/ttu-sm86-sm120-layout.json`。

此前“SM120没有第二寄存器字段”错误：SM86的内部键Rc在SM120叫Ra_URc，但两代都是bits71:64，FORMAT仍有Rc/Rd2。两代OPEN都有DUAL位bit72，LD.CLOSE都是bit74，MF count都是5位bits44:40。canned SM120序列标为CLOSE的bit72不符合两代Rosetta；先按artifact问题记录，不能认定架构改位。

因此目前没有TTU基础接口发生明显代际断裂的证据，也没有SM120同条件运行或延迟数据，不能断言内部实现相同。RT Core代际功能/吞吐可以大幅变化而沿用相近的提交接口，额外能力也可能通过TTU配置、节点类型及普通SASS软件路径体现。

当前差异的来源应分别排查：

| 层次 | 例子 | 怎样辨认 |
|---|---|---|
| 资料/artifact | CLOSE-labelled bit72、JSON键名变化、零操作数字段模板 | 对照完整FORMAT、alias和真JIT字节，不只比较键名或canned常量 |
| 驱动/编译与遍历家族 | 535镜像mttu MF18、同镜像utrav MF6、EoSS构造MF8；本地进一步缩成MF11 | 先在同一镜像比较不同函数；驱动版本、编译flag和shader功能固定后才谈架构 |
| AS与请求协议 | root tag/地址打包、节点/叶类型、是否opaque、有无instance/回调 | 用同一构建方式、输入几何与ray；不能将一条路径的TTUST槽语义推广到所有路径 |
| 测量环境 | CCTL scoreboard写后timeout而论文fault；计数器可见性不同 | 保存精确字节、后续wait、active lanes、driver/tool版本；没有同条件SM120对照前保留为候选差异 |
| 真硬件差异 | RT吞吐、功能、内部缓存/队列及微调度 | 需要实机同负载测量；接口相同本身不能排除 |

本地13条组在单三角形上成立，不意味着比SM120的10条组更复杂或更慢；输入槽、指令搬运量、编译选择和任务语义尚未匹配。

## 优先做什么：OptiX构建，CUDA自研核发TTU

第一步建议继续让OptiX负责AS构建，在同一CUDA context中保留构建结果，再由自研kernel直接发射TTU。OptiX只作为builder和正确性oracle；测量的kernel不调用optixTrace。这样无需先解析整个BVH格式，也无需先掌握离线重定位。

运行时同时保留AS基址、大小、handle、构建输入/flags及依赖分配；handle不能普遍当作裸指针。当前经验root偏移取AS header+0x20：GAS样本为0x300，IAS样本为0x380；原tag仍需验证，不是通用入口解码。OptiX context与相关资源保持有效，初始化/构建不计入TTU时间。

当前采用19条expanded组保留全部7个LD输出，再与13条精简组对照。它将原20条组的P4=false分支静态专门化；原predicated20条组在实时AS上不稳定，仍保留作诊断，不能作为唯一正确性基准。较深/不同叶类型可能需要额外状态或软件处理；省掉的6个LD不能因当前场景成功就视作永久无用。

最小场景阶梯：

1. **两个分离三角形（已做）。** 两条ray分别命中不同primitive，加一条miss；核对身份，不只看t。无instance、无any-hit shader。Phase14已补geometry flags0/1和ray DISABLE_ANYHIT等配置；原Phase13 flag1负结果已纠正为decoder问题。
2. **前后重叠三角形（基础已做）。** 输入顺序交换、tmax截断、tmin排除近处/全部交点及一般方向已通过。通过的opaque子集与最近命中一致；flags0/nonopaque另有较远候选反例，不能推广到全部完成/候选事件语义。
3. **扩大三角形GAS（规则网格已到4096）。** 19/13条组比较已通过；不同空间分布与重叠度待补。三角形数量不等于BVH深度；若要以深度作横轴，需要独立的结构解码或其他证据。
4. **一般ray（基础套件已做）。** dx/dy、负方向、非单位direction、tmin与front face已核对；geometry flags0/1、DISABLE_ANYHIT与face culling已验证。对t/u取幅值，符号另存；边界/退化/极端数值尚未覆盖。
5. **静态instance（受限父级resume已通）。** 恒等/平移/旋转/非均匀缩放、1–3实例、输入顺序交换、共享grid64/separated GAS及两个必要反例均通过。父级引用由TTU返回而非枚举builder输入。复杂IAS逆向暂挂，恢复时解释四实例的新引用/直接triangle返回，另行隔离等距/tmax端点；之后再考虑不同child GAS和嵌套。当前不支持任意IAS。

每一步保存CPU参考与OptiX参考的hit/miss、primitive ID、t、barycentrics及适用时instance ID，连同TTU raw输出。基础非边界射线先使用明确容差，边/顶点、退化和数值极端另作专门实验。不要从opaque子集成功推断整个OptiX语义覆盖。

## 比现有fixture还缺什么

| 缺口 | 为什么需要 | 获取方法 |
|---|---|---|
| 稳定的root描述符、tag与handle映射 | 大AS、GAS/IAS入口不一定同一节点类型或偏移 | 从builder记录base/handle；对照对应JIT的TTUST输入准备，改变一项场景特征看字段变化 |
| 更完整的ray/flags范围 | 基础一般方向/tmin已通，只映射有限flags | 边界/数值、更多flags与primitive类型逐项扩展，保留oracle |
| 返回状态、primitive/leaf引用、barycentrics | 多三角形只看t容易误判，实例/回调需要事件类型 | 恢复原20条组的全部LD输出，对照不同primitive、构建顺序、几何和ray；检查组后解码SASS |
| “完成”与“需软件处理”的区分 | 已有nonopaque较远候选与IAS事件的实证 | 区分opaque最近命中、nonopaque候选和instance事件，沿JIT实现最小accept/ignore/transform/resume状态机 |
| 遍历状态/栈与溢出 | 深BVH、实例、非opaque可能越过当前纯硬件路径 | 看对应JIT路径、栈状态字段和多次TTU提交；不要先假定必需，也不要凭0 local/shared推断永远不需要 |
| AS生命周期与重定位 | 当前只改+0xd0的办法未覆盖任意复杂AS | 先同context原地址重放；再利用OptiX relocation API做受支持重定位，并重新编码TTU root |
| 正确重复、lane/warp执行 | AS越大，准备/等待/状态错误越容易伪装成性能变化 | 先解决LD覆盖下次ST输入，再做设备内计时和逐lane独立输出；当前run_exp的32线程dump不能当逐lane oracle |

复杂AS**不要求立刻通解整个二进制格式**；同地址、同context的生产AS可以先作为不透明输入。只需先找到可靠的root与请求/返回语义。相反，若目标是可控制精确BVH深度/节点布局，或者离线完全独立builder，就需要逐步解析节点格式、量化界限、子引用、叶类型、对齐和重定位。

本地OptiX8头文件提供 `optixAccelGetRelocationInfo`、`optixCheckRelocationCompatibility` 和 `optixAccelRelocate`。官方接口明确IAS引用的子AS、micromap也可能需要重定位，不能把“本fixture只patch一个自指针”推广到任意AS；API返回的新handle仍需接入TTU root编码。跨进程持久化还涉及兼容性、依赖对象和格式稳定性，属于后续独立任务。

## “OptiX能做的都直接用SASS做”有多大

这个目标可以分成三个范围，成本差别很大：

| 范围 | 需要实现 | 建议 |
|---|---|---|
| 自研kernel向RT提交生产AS，取triangle最近命中/遮挡存在性结果 | root/ray/flags、返回解码、必要时续遍历；AS仍可由OptiX构建 | 当前最合适的主目标；从opaque单GAS推进到静态IAS，可覆盖大量微架构问题 |
| 自己提供BVH builder/序列化并可视化渲染 | 节点/叶格式、地址引用、重定位、遍历控制；一个简易CUDA shader/renderer | 很适合blog或逆向兴趣，但可在生产AS多层遍历跑通后选择做；先锁定单架构和驱动 |
| 覆盖目标GPU/SDK上可用OptiX语义，脱离OptiX runtime | any-hit的accept/ignore/terminate、custom intersection、实例与motion、shader/SBT/属性/调用与续栈、嵌套ray，以及所选版本的其他primitive/特性 | 接近自行实现一套RT运行时和编译接口，不能只靠一组TTU覆盖；不建议作为当前完成条件 |

OptiX的closest-hit/miss/any-hit/custom intersection包含普通程序执行；不是每一项功能都由一条TTU在RT Core内完成。任意CUDA程序最终也会成为SASS，这种广义“全用SASS”在原则上很宽，但工程目标应说清：自己接管硬件traversal接口，还是复刻OptiX全部编程模型。后者长期维护私有接口的成本远大于当前microbenchmark。

当前几何功能基线已覆盖多三角形GAS和一般ray，1–3静态instance的受限设备续遍历也已通。[Phase19](PHASE19.md)已补MACROFUSE拆分/必要性反例及Tensor、global/shared重叠。2026-11-04选择先做单GAS小renderer，验证逐lane不同ray、尾warp及软件CUDA/TTU/OptiX三后端，再回紧凑多warp测量；细节见 [MEASUREMENT_ROADMAP](MEASUREMENT_ROADMAP.md)。复杂IAS新返回路径及等距边界继续暂挂；不能由三角形数量推断已知BVH层数。any-hit与custom primitive仍是后续独立篇章。

可以形成三篇独立blog：从opcode到有oracle的裸TTU；生产AS与TTU提交/返回协议；复杂BVH下的lane/warp时序与CUDA并发。即使最后都是复现或明确的负结果，也足以作为有价值的阶段产物。

EoSS renderer源码核查：截至2026-11-02，官方GitHub当前main和Figshare v1未找到Fig.6自建BVH renderer。Figshare TTU README明确将完整hit/miss traversal列入不提供范围，只发布最小patch/counter demo。原始小demo已另存research/community/eoss-ttu-artifact；来源与核查边界见 [核查记录](../../research/notes/eoss-renderer-source-status.md)。

## 另外四份旧文件的复核

这些文件的完整内容曾匹配删除清单hash；本次只读复核，没有恢复为当前工具或执行旧探针：

| 文件 | 实际内容及判断 |
|---|---|
| `mk_simple_dis.txt` | 1050字节，实际是nvdisasm的unrecognized-operation错误/警告，**不是完整megakernel反汇编**。旧INDEX给它的描述过强；现有原JIT可用于重新生成正确语料 |
| `kprobe3.cu` | 通过host-pinned进度标记保留fault前已读页；没有捕获非法load的设备端机制，并不真正fault-tolerant。属于任意VA探测，不是TTU协议/AS必需组件；不恢复为当前入口 |
| `tools/sec_elf.py` | 思路是取section并包最小ELF，但EM_CUDA写54（实际fixture为190）、e_shstrndx写入offset61（正确62）、shstrtab名字偏移等有错误，ET_REL注释与type=2也不符。通用需求可由当前section parser和正规容器实现，不恢复旧实现 |
| `tools/rebuild_cubin.py` | “把裸代码装进真实cubin供反汇编”的思路有用；原实现重排所有section却只调整部分program header，未完整处理NOBITS、对齐填充及代码尺寸相关元数据/符号等。当前固定尺寸patch已覆盖运行需求；以后需要更大反汇编容器时另做受约束实现 |

这四份不含遗漏的TTU族构造知识，也不阻塞上述AS路线。其价值主要是通用工具意图和失败教训，已在这里记录；不要把有bug的工具重新标为可靠基础设施。
