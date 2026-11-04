# 小renderer：功能、数值边界与性能

固定相机、opaque triangle最近命中，CUDA软件二叉BVH、裸TTU、OptiX共享float32三角形、ray和24B hit ABI；着色输出depth/normal/primitive，另可演示水平地面一次镜面。运行说明见 [示例README](../examples/renderer/README.md)，示例图使用Stanford University Computer Graphics Laboratory的Bunny数据。

## 实现与范围

软件builder按最长centroid轴median split、leaf≤4，GPU near-first迭代遍历、64项栈、float32 Möller–Trumbore求交。OptiX使用生产GAS和最小raygen/closest-hit/miss；native用同一个GAS，在普通CUDA kernel中准备/解码状态，21条marker替换为wait+19条TTU组+wait。没有自行构造私有BVH、IAS、any-hit或通用材质系统。

公开PTX没有TTU入口，故相机、软件遍历、着色保留CUDA C++，小段issuer用受控SASS patch。默认native monolithic ptxas O0，55 registers/288B stack；软件O3、禁fmad。直接O1–O3破坏连续marker，patcher拒绝，不执行。可选linked把issuer O0和caller O0–O3分开编译，但最终92 registers/288B stack；原对照trace仅改善约1–2%，默认不切换。O2/O3在该工具链曾生成相同binary。

patch检查连续块、36个独立有效寄存器、常量及操作数字段；linked显式分别映射ST输入与LD输出，所有ST须先于LD，未返回pair位置保持。损坏marker对照须被拒绝。此接口仍依赖工具链/驱动私有协议，不能用提高-O代替正确的依赖及搬运优化。

## 模型与验证

Bunny来自[Stanford官方模型库](https://graphics.stanford.edu/data/3Dscanrep/#bunny)，署名Stanford University Computer Graphics Laboratory；研究/免费再分发及商业许可条件见[条款](https://graphics.stanford.edu/data/3Dscanrep/#acknowledge)。来源、包/成员hash在 [asset manifest](../examples/renderer/assets/stanford-bunny.json)，模型本体不入Git。支持三角ASCII PLY；拒绝非三角、非法索引、非有限坐标及float32退化面。

四级使用full模型相同bounds归一化、yaw25°和相机，再加两个地面面；不是逐级缩放。降采样模型有重合/反绕序面，默认保留；显式 `--mesh-deduplicate`按float32顶点集合去重并保留首面，改变了原始几何，报告记来源映射。full无这种重复。

| 模型 | 原始模型面 | 去重移除 | 含地面面数 | 256×192当前结果 |
|---|---:|---:|---:|---|
| res4 | 948 | 40 | 910 | 1个像素几何残差超预算 |
| res3 | 3,851 | 83 | 3,770 | 通过 |
| res2 | 16,301 | 87 | 16,216 | 1个像素几何残差超预算 |
| full | 69,451 | 0 | 69,453 | 通过 |

每次先做1/31/32/33/127/129/257条逐lane不同ray、每后端3重复和尾guard；随后全图跨后端比较、最多257条CPU参考，计时后重新验证shuffled及管线输出。double brute-force CPU oracle另与旧Python参考交叉检查，不作为GPU性能基线。箱子8项尺寸/grid/block回归、镜面回归、linked full小图通过；software+OptiX memcheck零错误，patched native未插桩。

默认strict要求ID/face精确一致、t/u/v满足rel3e-5或abs2e-5。显式geometric策略仍保留ID/face和原t检查，仅当barycentric不一致时，核对其范围、重建交点与ray残差/双方世界坐标差，每项以double CPU oracle复核。预算 `16 * 2^-23 * max(1, abs(vertices), abs(origin), abs(t*direction))`，不是硬件误差保证。

### 两级为什么失败

两例ID/face都一致，t也通过原容差；失败的是barycentric重建点到ray的残差。ray接近沿三角形表面飞行，求交中的小分母放大float32消去/舍入误差；LOD简化会改变局部面朝向，失败数不随面数单调变化。

| 样本 | ray与面夹角约 | 软件残差 | OptiX残差 | 当前预算 |
|---|---:|---:|---:|---:|
| res4 pixel38316 / primitive193 | 0.92° | 1.42e-5 | 7.13e-7 | 1.34e-5 |
| res2 pixel40092 / primitive14209 | 0.24° | 3.36e-5 | 5.83e-7 | 1.34e-5 |

这两例中软件t反而更接近double参考，说明“t较准”和“t/u/v内部一致”是不同检查；不能只归咎于某个后端，更不是TTU不支持这两级AS。res4只是略超当前保守预算，res2更明显。若改善软件基线，应独立比较稳健/watertight算法，而非统一扩大容差。最小三角形/ray/双方结果在 [counterexamples](../fixtures/renderer-counterexamples.json)，不需要保留整图dump。

另有res2在256×256公共边返回相邻primitive3408/3409、full在128×128超预算的观察；ID差异仍失败。full主图89项软件barycentric严格差异经CPU复核通过，native与OptiX主图满足原严格比较；normal图字节相同不表示float hit buffer逐位相同。成功仅覆盖记录的输入/相机/分辨率。

模型还暴露旧decoder只接受8/a的错误：raw R72=60000011或40000011仍对应primitive17与正确属性，renderer现经验接受4/6/8/a。tag语义尚未知，改变仅限当前opaque renderer子集，原始关键word保留在同一个小fixture中。

## 主追踪与镜面

RTX3090、256×192=49152条ray，9样本×100launch，单位µs/批，中位数的代表观察：

| 输入 | 软件BVH | native | OptiX |
|---|---:|---:|---:|
| res3 coherent | 136.4 | 31.4 | 15.1 |
| res3 shuffled | 128.2 | 31.4 | 16.4 |
| full coherent | 263.6 | 33.0 | 15.8 |
| full shuffled | 194.6 | 33.2 | 17.7 |

CUDA event由C++连续发射，AS/ray常驻GPU、传输在event外；每后端持续≥0.2秒预热、交替ray顺序并轮换后端。之前3次预热有idle升频影响，旧数值不可直接当优化对照；仍未锁频。full GAS约4.5MB，软件40601节点/深度15，树形/压缩/布局/构建质量不同，软件/native约8倍是具体实现比较，不能称纯RT硬件收益。

full打乱ray让软件快约26%、OptiX慢约12%，两卡观察方向相同；native变化很小也可能被接口开销遮住。当前尚未将其归因于分歧、缓存或某共享资源，4.5MB也不自动代表DRAM负载。

单次镜面仅用于水平地面：SM根据primary命中身份分类，用已知平面解析计算交点，起点向上偏移0.001、tmin0.001、方向(dx,-dy,dz)，再调用同一追踪后端并合成normal伪彩色。直接用各后端hit.t曾使二次输入不同，当前解析平面方案保证三后端二次ray字节相同；不代表通用曲面材质。

13,083条真实镜面ray、2,398条二次命中；没有compaction，实际二次dispatch仍49152条，其他是dummy miss。GPU1/5三后端normal/mirror图像一致。GPU5二次trace约277.3/33.2/17.6µs，完整camera→primary→生成→secondary→合成约620/85/57µs。分阶段独立重复的cache状态不同，median之和不必等于完整管线；端到端速度不用于定位RT/SM共享资源。

当前演示范围足够。后续优先紧凑多warp；若先改renderer，针对现有数值反例或缩减native接口搬运，不继续堆模型、材质或复杂IAS。
