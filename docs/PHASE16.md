# Phase 16：设备端instance事件处理与独立child查询

日期：2026-11-03。RTX 3090 / SM86，driver 535.309.01，CUDA 12.1，OptiX 8。承接 [Phase15](PHASE15.md)。本阶段没有性能测量。

> 后续纠正与进展：[Phase17](PHASE17.md)已实现平坦1–3静态instance的受限父IAS恢复。**本阶段以E-0x0c/E-0x20读ID/handle的相邻偏移规则已撤回**，应以[E+4]索引查IAS实例表；逆矩阵和child root规则仍成立。下文记录当时实现范围与历史实验，报告及hash不改写。

**一个自研CUDA kernel已能完成IAS请求、识别0x16事件、读取实例记录、变换ray、重新开启child GAS查询并读回结果。8项矩阵、1344次kernel launch、2615次TTU请求全部与CPU/OptiX对照通过。尚未保存并恢复父级IAS遍历状态，因此只支持本阶段验证的单个静态instance。**

## 本阶段推进了什么

Phase15由host解码事件、变换ray，再次launch查询child。本阶段host只负责构建AS、提交最初ray和检查最终结果；两次TTU请求及中间处理均在一个kernel内完成。kernel执行期间不调用OptiX、不由host读取事件决定下一步。

流程如下：

1. 用19条expanded组查询IAS，保留全部7个LD槽的返回。
2. 直接miss则结束；未知事件或越界引用返回错误状态。
3. 对0x16事件，用经验地址打包规则得到实例记录，检查IAS范围和对齐。
4. 读取逆矩阵、instance ID及child root；检查child引用落在持有的GAS范围内。
5. 普通CUDA指令变换origin/direction，保留tmin/tmax，不归一化direction。
6. **重新初始化请求输入，OPEN新的child查询**；用同一19条组发射并读回。

其中`CLOSE`关闭了当前TTU会话；这不是“保持同一OPEN等待SM处理后再GO”。是否能恢复整条ray的逻辑遍历，需要另外理解返回状态并在后续请求中重新提交。当前没有实现该协议。

## 可复用组件与patch方法

| 文件 | 用途 |
|---|---|
| [device_instance.py](../src/device_instance.py) | host/device结构、构建、编译器寄存器映射、TTU patch、module及launch接口 |
| [instance_template.cu](../kernels/instance_template.cu) | 普通CUDA事件分派、边界检查、逆矩阵运算、请求重新初始化；不承担计时 |
| [run_device_instance.py](../tools/run_device_instance.py) | 实时AS、输入ray、raw返回、设备变换/记录与host解码交叉检查、CPU及OptiX oracle |
| [validate_device_instance.py](../tools/validate_device_instance.py) | 8项复现矩阵，每项新进程/context、40秒timeout，保存子报告及hash |

`ttu_issue()`用18个64位输入/输出对应fixture里的18个寄存器对。nvcc生成21条连续DADD marker；工具检查opcode、marker立即数和寄存器关系，取得真实寄存器分配，再把19条expanded TTU组的两个显式寄存器字段重映射。前后各放一个wait-all NOP，普通CUDA代码负责输入/输出搬运。输入、输出缓冲独立，第二次请求前重新初始化全部状态，避免直接重复被LD覆盖的输入。

这让事件处理可以用可读CUDA代码逐步扩展，同时继续复用`src/ttu_isa.py`和`sm86_fixture.load_group()`的编码知识。它仍绑定SM86及当前fixture协议，不能直接当作通用内联TTU ABI；编译器变化导致marker布局不符合预期时会拒绝patch。

[资源记录](../results/phase16/phase16-device-resource-usage.txt)来自一次单独构建：`instance_probe`为59 registers/thread、288字节stack、0显式shared。`cuobjdump`同时显示LOCAL:0，不能据此忽略stack上的数组及搬运。此功能容器以`-Xptxas -O0`编译，只有1个thread/active lane，不能替代Phase12的80寄存器/无stack测量容器，也不能用其host时间估计TTU延迟。

## 结果

[矩阵及各子报告的SHA256](../results/phase16/phase16-device-matrix.json)：

| instance / GAS | GPU | instance ID | kernel launches | TTU请求 |
|---|---|---|---:|---:|
| identity / single | 1 | 17 | 144 | 279 |
| translate / single | 1 | 41 | 96 | 182 |
| rotate / single | 1 | 0 | 96 | 182 |
| nonuniform scale / single | 1 | 0xffffff | 96 | 182 |
| identity / grid64 | 2 | 41 | 272 | 536 |
| nonuniform scale / grid64 | 2 | 17 | 272 | 534 |
| identity / overlap2 | 2 | 17 | 184 | 362 |
| rotate / separated2 | 2 | 41 | 184 | 358 |
| 合计 | | | **1344** | **2615** |

使用`general_rays`：一般方向、反向、非单位direction、tmin/tmax裁剪及空间miss。identity/single重复3次，其余重复2次；重复是独立kernel launch，不是设备内测量循环。两张卡分摊场景，不是完整场景×GPU交叉矩阵。ray flags固定为1（DISABLE_ANYHIT），geometry flags为0，变换的行列式均为正。

- 73次IAS直接miss，只需1次TTU请求。
- 1271次IAS返回实例事件，均发射独立child请求。其中541次child miss、730次triangle hit。
- 所有命中都核对primitive、t、barycentrics、front face和instance ID；设备读出的记录地址、child root与变换后ray也与host解码对照。
- **实例事件不是命中承诺。** 那541次child miss是必要的功能覆盖；不能看到0x16就把ray标成hit。
- 所有正式进程正常退出，无CUDA fault；报告保存raw返回、输入几何/矩阵/ray、源码hash、compiler信息、group/cubin hash和GPU UUID/驱动清单。

复现，从`ttu-bench/`运行并选择空闲卡：

```bash
CUDA_VISIBLE_DEVICES=1 timeout 40s python3 bench.py run_device_instance --instance identity --ray-index 0 --check --output results/local-device-one.json
python3 bench.py validate_device_instance --gpus 1 2 --output results/local-device-matrix.json
```

编码校验通过；[固定fixture回归](../results/phase16/phase16-fixture-regression.json)为GPU2的8case/40launch，含原20条组。

## 距离正确的父级恢复还差什么

当前实现处理一个实例后即返回child结果。多个instance时，后面可能还有更近的命中，或者第一个child miss而另一个child hit；这时必须继续父级IAS，不能提前结束。

下一任务应围绕这些反例解码，而不是简单把已知实例逐个独立查询后取min：后者可以做软件参考，但不能验证TTU父级恢复协议。

1. 从JIT的父级状态保存/恢复路径明确每个返回槽和重提交输入的对应关系。现有片段在`reference/jit/jit16-instance-ordinary.sass`，必要时用`tools/disassemble_ttu.py`补齐`0x2b20–0x41e0`及后续栈恢复分支。
2. 构造两个instance、输入顺序交换、沿ray前后重叠；专门加入“第一事件child miss、另一个child hit”和“第一child hit但另一个更近”。需要记录事件顺序，不能仅从builder输入顺序推定访问顺序。
3. 显式保留父级ray、返回槽和当前best t，在child完成后恢复父级，直到有可验证的终止条件；与CPU/OptiX比较最终最近命中。

`parent_slot_nonzero_raw`只记录R51是否非零，没有证明该位等于“存在兄弟节点”或“可以resume”。嵌套IAS、负缩放、instance flags、motion、any-hit accept/ignore、custom primitive和full-warp执行均未覆盖。节点访问数/树深度/硬件容量也没有新增结论。
