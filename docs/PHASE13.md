# Phase 13：同 context 生产 GAS 与多三角形裸 TTU

日期：2026-11-02。RTX 3090 / SM86，driver 535.309.01，CUDA 12.1，OptiX 8 SDK。

2026-11-03纠正与扩展见 [Phase14](PHASE14.md)：一般方向/tmin/opaque与culling flags已通过；旧`-R73`/`-R32`解码假设过窄，flag1此前的t误判已纠正。flags0的nonopaque多命中出现较远候选，原+Z通过不能推导通用closest-hit。本文保留当时实验条件；当前CLI默认ray flags1，复现旧raw配置需显式`--ray-flags 0`。IAS root也不再固定0x300。

**新增能力：同 CUDA context 内构建 triangle GAS，直接向 TTU 提交，再以 CPU 和 OptiX 比较 hit/miss、primitive、参数 t 与 barycentrics。** TTU kernel 不执行 `optixTrace`；AS builder 和后续 oracle 使用 OptiX。未测性能、未实现独立 BVH builder。

## 可复现结果

```bash
CUDA_VISIBLE_DEVICES=7 timeout 30s python3 bench.py run_scene separated --group both --repeat 5 --check
CUDA_VISIBLE_DEVICES=7 timeout 30s python3 bench.py run_scene overlap-reversed --group both --check
CUDA_VISIBLE_DEVICES=7 timeout 30s python3 bench.py run_scene grid --count 1024 --group both --check
python3 bench.py validate_scene --gpus 5 7 --output results/local-scenes.json
```

`both` 比较 **19 条 expanded 与 13 条 reduced**。原始 20 条组以 `--group full` 保留作诊断，不能将其与 expanded 混称。

[正式矩阵](../results/phase13/phase13-scenes.json)：GPU5 / GPU7，每张卡 8 场景、每条 ray 每组 5 次，共 **16 cases / 980 launches 全部通过**；逐 case 原始寄存器、构建输入、源码/group/AS hash、GPU UUID 清单与 CUDA_VISIBLE_DEVICES 保存在 [逐场景记录](../results/phase13/phase13-scenes-cases/)。

| 场景 | 验证内容 |
|---|---|
| single | 固定 fixture 之外的实时构建单三角形 |
| separated | 两条 ray 命中不同 primitive，距离相同，必须核对 ID |
| overlap / overlap-reversed | 交换前后两个三角形的输入顺序，最近命中的 ID 随之从 0 变为 1 |
| grid 16 / 64 / 257 / 1024 | 命中第一、中间、最后 primitive；最大已验证 ID 1023 |
| 每个场景附加 rays | 空间 miss、tmax=6 截断，以及不同命中位置；命中 barycentrics 包括 (.125,.625)、(.375,.5)、(.0625,.4375) |

CPU 使用 Möller–Trumbore；OptiX 返回 primitive、ray t、barycentrics。基础 t=6.875，另两位置 t=6.5 / 6.3125。非边界命中采用绝对/相对容差 1e-5。所有 rays 仍为 +Z、tmin=0；这些场景没有相等 t 的竞争命中。

## 可复用组件

- `kernels/scene_bridge.cpp` / `src/optix_scene.py`：在调用者当前 context 中构建并持有 GAS、vertices、scratch、OptiX module/pipeline/SBT；公开 base / size / traversable handle 和 oracle。首次运行编译 `.so` / `.ptx`，需本地 OptiX SDK、nvcc、g++。
- `kernels/scene_oracle.cu`：固定 payload 的最小 raygen / closest-hit / miss，仅用于参考结果。
- `sm86_fixture.load_group(expanded=True)`：将原组的 P4=false 分支静态专门化，移除 P4=true 的 ST[0x200]，其余 predicate 改 PT；保留全部 7 个 LD，重新计算 MF(17)，总计 19 条 TTU。13 条组保持原 hash。
- `sm86_fixture.triangle_result`：当前 triangle 子集的经验返回解码。hit 下 `-FP32(R73)` 是 t，`[-FP32(R32), FP32(R33)]` 对上 barycentrics；`R72 & 0x0fffffff` 对上 primitive。**只验证实际覆盖的 ID，不宣称完整字段宽度/ABI。**
- `tools/run_scene.py`：CPU/OptiX/TTU 三方比较，保存 R0–R75；raw TTU 在第一次 oracle trace 之前执行。`--ray-index` 隔离某条 ray，`--oracle-only` 单独验证 builder，`--as-dump` 保存当前 AS 原始字节。
- `tools/validate_scene.py`：每个场景/设备独立进程与 context，默认 30 秒 timeout，失败即停止矩阵；不重置 GPU。

root 暂仍采用 fixture 的 **AS base+0x300 与原 tag**。当前各规模成功不代表这是通用 handle 解码。实时 AS 不搬迁、不 patch 自指针，builder 保持分配有效；与 Phase12 的离线单三角形 AS 不同。

## 必须保留的负结果与纠正

1. **原始 predicated 20 条组在实时 GAS 上不稳定。** [当前源码的复现](../results/phase13/phase13-full-predicated-negative.json) 在前三次成功后 CUDA700；固定 fixture 上原组仍通过 5 次 hit 和 miss 回归。cuda-gdb 下实时场景反而通过；保留 module 不 unload、CUDA_LAUNCH_BLOCKING、修改 R2P mask 均未解决。19 条组和13条组通过矩阵，因此问题范围缩到提交组的条件执行/融合/初始化及其环境，但**尚未确认根因**。不能称为“更换 ray 必然 fault”或“多三角形不受支持”。
2. **geometry flag=1旧负记录（已解释）。** [flag=1 对照](../results/phase13/phase13-opaque-flag-negative.json) 的OptiX/CPU匹配，TTU返回正t而旧decoder无条件取负。Phase14修正解码后flag1已通过；不再把它列为未通的硬件功能。raw记录保留。
3. **返回 tag 不只有 a。** 257 场景的 hit 为 `80000000 / 80000080 / 80000100`，t/ID/barycentrics 正确。最初只接受 a-tag 的 decoder 导致 [负记录](../results/phase13/phase13-decoder-a-only-negative.json)；现接受观察到的 8/a 两类，tag 原值写入记录。bit29 的真实语义未解释，不能当作 miss 或跳过正确性检查。
4. **R50 的旧 `|8` 是不应固化的地址位。** 原 JIT 0x7f0–0x840 对低字是右移/搬运，没有额外 OR8；现改成 `uint32(dummyVA>>6)`，R51 保留已验证的高字 tag。这个纠正没有单独解决上面的20条组故障。更新后的 [Phase12回归](../results/phase13/phase13-baseline-regression.json) 仍为16 cases / 80 launches通过。

早期`scene-*.json`现收在本地被ignore的 `results/archive/exploratory/`，schema和decoder较旧，有些只检查hit/t、有些oracle-only；最终能力以正式矩阵及关联case为准。不要因早期没有primitive检查就改写成当时已完成全部验证。

## 下一步

先固定现有生产 AS 功能基线，补完整 dx/dy、负方向、非单位方向、tmin 和有限 flags。原20条组问题可独立排查，以19条组保留额外返回槽作对照；无需为了继续而先解决所有旧组行为。一般 ray 正确后再试一个静态 instance，识别 transform、instance ID 与返回/续遍历状态。

结构研究已有新线索：[公开资料核查](../../research/notes/nvidia-bvh-public-records.md) 中的 NVIDIA complet 专利。它们值得指导 AS 差分，但尚不足以自建本机任意 BVH。多三角形数量不等于树深度；当前结果也未验证任意复杂 AS、通用最近命中语义、full warp、设备内循环或资源争抢。
