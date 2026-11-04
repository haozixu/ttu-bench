# 下一阶段：先小renderer，再回多warp

更新：2026-11-04。Phase17–19证据保留，复杂IAS继续挂起。本文件是计划，尚无renderer实现、图像或新增性能结果。

## 选择与范围

近期先小renderer：当前full-warp实验只有同ray，renderer能补每lane不同ray、尾warp和更丰富几何的验证，也能产生直观的blog成果。它完成后就回到多warp测量，不扩成渲染引擎。若只追求尽早研究资源竞争，多warp更直接；当前选择兼顾业余探索与后续组件复用。

首版限定单个triangle GAS、opaque/禁用any-hit、最近命中、确定性相机，以及depth/normal/primitive着色。不做IAS、motion、any-hit、路径追踪和复杂材质。先用已验证几何，再加简单模型/网格。

## 三后端与正确性

| 后端 | 实现与作用 |
|---|---|
| software CUDA | 自己的简单二叉BVH、迭代遍历、AABB/triangle求交，提供合理剪枝的软件基线 |
| native TTU | OptiX构建并持有生产GAS，小段SASS在普通CUDA kernel里提交/返回，作为测量对象 |
| OptiX | 同一生产GAS、最小追踪程序，作为正确性及公开运行时性能参考 |

实现顺序是先用软件CUDA与公开OptiX后端跑通相同ray/hit buffer，得到不需要手工patch的初步对比；再加入native TTU后端。这样可以独立验证renderer与计时管线，也能分清OptiX运行时路径和裸TTU路径的差别。

三者共享三角形、相机ray、tmin/tmax、culling语义和着色步骤，输出统一hit/primitive/t/barycentrics。先保存无着色hit buffer并与CPU解析/OptiX核对，不能只看图像。边、顶点、等距身份分开验证并明确容差，不默默放宽oracle。O(N) brute force可作小场景oracle或附加参考，不是唯一CUDA性能基线。

先验证1、31、32、33及更大ray数，覆盖尾warp、不同primitive、hit/miss和正反面。所有活跃lane都须发有效请求并关闭会话，重复launch不得泄漏状态。新编译器后端必须重新验证寄存器/依赖，不能假设Phase19证据已覆盖发散ray。

软件BVH与私有OptiX AS树形、压缩、布局、构建质量不同。记录软件builder、leaf容量、最大深度与栈溢出检查；没有私有AS解码就不声称相同树形。加速比回答两个具体实现各有多快，包含算法和布局差异，不能称纯RT Core收益。

## CUDA/PTX与SASS patch

公开PTX没有TTU opcode，也没有普通CUDA下等价optixTrace的公开发射入口。inline PTX只能嵌合法PTX，不能让ptxas接受TTUST等原生助记符。OptiX能从CUDA/PTX程序JIT出追踪代码，但使用其编程接口/运行时，不等于普通PTX直接访问RT Core。

因此采用混合实现：相机、软件BVH、着色、索引、输出全部用CUDA C++，只有小段TTU后端patch。已有 [device_instance.py](../device_instance.py) 的编译器寄存器映射/marker思路可复用，但须重新验证优化选项、marker唯一性、寄存器存活、编译器可见的数据依赖、访存顺序、scoreboard和实际SASS。marker不符合预期就拒绝运行。当前-O0功能容器不是性能核，先做功能版，再单独优化和计时。

这是私有接口的工程hack：编译器不知道TTU语义，patcher承担驱动/编译器原本维护的约束；AS输入仍依赖本机格式。小模块封装、固定SM/工具链、保留原cubin/patched hash和功能检查可使它可审阅，不能使其变成稳定公开ABI。无须手写整个renderer的SASS。

## 计时与首版完成标准

数据留GPU，预生成ray用于trace-only对照；预热后用CUDA event测重复kernel并报告分布、设备与时钟状态。三个后端都完成同步与功能验证。分别列出软件BVH/OptiX AS构建、steady追踪、统一着色、GPU相机+追踪+着色总时间；传输/编码另列。JIT、Python和冷启动不混入steady ray/s，小场景记录launch开销影响。

先看小到中等场景、hit/miss比例与相干/打乱ray，超过已测几何规模先验正确性。4096只是已测试数量，不是上限。小场景/低并行时RT可能受准备开销限制，不预设必然更快；不用含大量snapshot的Phase19核代表RT性能。

首版通过条件：三后端非边界结果一致；不同lane、尾warp和重复launch通过；输出图像及hit buffer；各后端计时边界、分布、资源声明可复现；正式结果单ZIP发布，关键失败保留。达到后即结束renderer阶段。

## 随后多warp的顺序

1. 回到热小AS、紧凑输出，同CTA设置1/2/4/8…个TTU warp。核对SMID、每warp请求数、active lanes与寄存器/shared/驻留；匹配无TTU对照，测准入、完成和总吞吐，不预设12warp常量。
2. 用有效拆组改变OPEN→GO、GO→LD.CLOSE间隔，测其他warp准入/吞吐。延长间隔也增加指令工作，需要匹配控制以区分ticket占用与SM发射成本；台阶不直接命名为buffer容量。
3. 同CTA中TTU与FADD/INT/shared/LDG分别用不同warp，测双向吞吐并核验重叠；轮换角色位置，控制SMSP/scheduler映射。HMMA依赖/padding解释充分后再加Tensor压力。
4. 出现可重复局部选择性变化后，扩大working set、比较同/异SM，区分接口/驻留、L1/队列、L2/DRAM；记录功率/频率并交错A/B。

research目标是解释作用阶段/范围并预测新配置；renderer、加速比、lane扫描本身首先是工程/blog成果，不以论文门槛否定它们。
