# Phase 15：静态instance记录解码与独立child查询

日期：2026-11-03。RTX 3090 / SM86，driver 535.309.01，CUDA 12.1，OptiX 8。承接 [Phase14](PHASE14.md)。本阶段没有性能测量。

后续进展见 [Phase16](PHASE16.md)：设备端事件处理与独立child查询已通，父级IAS恢复仍待实现。下文保留本阶段的host参考实现及当时的下一步计划。

**新增：从真实IAS事件读出的逆变换、instance ID、child handle和child root已作差分验证；host处理事件后发起独立child GAS查询可与CPU/OptiX匹配。尚未实现保留父级状态的裸TTU续遍历。** 这不是完整IAS支持，不能用来验证多instance最近命中。

## 1024是测试规模，不是容量上限

`validate_scene.py` 选取16/64/257/1024作为回归规模；`run_scene.geometry('grid', count)` 主动生成对应数量的三角形，按间隔3排列在二维网格上，再交给OptiX builder。bridge将 `numVertices` 设为三角形数量的3倍。此前1024只是最大已测试输入，没有从AS格式或硬件能力推导该上限。

本阶段直接扩大到4096：

```bash
CUDA_VISIBLE_DEVICES=2 timeout 40s python3 run_scene.py grid --count 4096 --ray-suite general --group both --repeat 1 --check --output results/local-grid4096.json
```

[正式结果](../results/phase15/phase15-grid4096.json.gz)：136条ray × 13/19条组，共272次launch全部通过CPU/OptiX对照，AS大小270592字节，命中ID覆盖到4095。每条ray每组只有一个GO，遍历过程中无软件节点访问或重新提交。该结果仍不能给出内部访问节点数、BVH深度或硬件上限。

## JIT事件分支提供了什么信息

[实例分支SASS](../reference/jit/jit16-instance-ordinary.sass) 与 [来源manifest](../reference/jit/jit16-instance-disassembly.json) 保留原始地址、原指令字与源hash。提取时只为反汇编将TTU换成NOP；输出不能运行。可复现工具为 [disassemble_ttu.py](../disassemble_ttu.py)：

```bash
python3 disassemble_ttu.py fixtures/jit_16.elf --range fb0:1190 --range 29b0:2b30 --range 3100:33a0 --range 41e0:4bc0 --range 5e80:6520 --range 8500:8590 --output /tmp/ttu-instance.sass --manifest /tmp/ttu-instance.json
```

关键静态证据：

- `0xfd0–0x1000`：从R33提取高5位，检查0x16和额外返回状态。
- `0x29b0–0x2b10`：进入0x16事件路径；R41的状态决定是否处理当前层。
- `0x3100–0x33a0`、`0x35xx–0x41xx`：处理带tag的traversable引用及保存父级状态，含shared/local路径。当前独立child工具不执行这些保存步骤。
- `0x41f0–0x42d0`：从事件地址和高地址部分构造instance描述符，读取记录中的控制位。
- `0x47d0–0x4970`：从事件地址 `+0x10/+0x20/+0x30` 读取三行矩阵，用FMUL/FFMA/FADD计算局部ray。
- `0x60d0–0x64b0`：解引用child handle，按类型处理，读取child AS头部 `+0x20`，构造root描述符。
- `0x8500–0x8580`：检查状态并回到下一次请求准备入口。

因此可以具体指出这条捕获路径中SM普通指令承担的实例变换工作；不能推广为所有硬件/驱动路径都必须相同。

## 当前经验记录布局

以事件指向的地址记作E；下面只适用于本阶段验证过的静态实例样本。

| 位置 | 已验证内容 | 依据 |
|---|---|---|
| 返回 `R33>>27` | 0x16事件类别 | 与JIT分支一致，不是完整事件枚举 |
| 返回 `R72 + ((R73 & 0x1ffff)<<32)` | E | 位于IAS分配范围，不能把R73当浮点t |
| `E+0x08` 的64位值 | 带tag的child root，低49位对上GAS base+root offset | 每个事件与builder原分配及GAS头部交叉核对；完整tag含义未知 |
| `E+0x10..E+0x3f` | 3×4世界到局部的逆变换 | identity/translate/rotate/scale，矩阵与build输入相乘为单位矩阵 |
| `E-0x20` 的64位值 | child handle候选 | 与builder提供的GAS handle逐位一致 |
| `E-0x0c` 的32位值低24位 | instance ID候选 | build输入与OptiX交叉检查，覆盖0、17、41、0xffffff |

事件的矩阵区域与应用提交的OptixInstance输入记录不是同一个起点。我们检查的是builder产物，不能仅照抄公开OptixInstance结构定义解释所有内部字段。地址掩码、相对偏移和ID布局尚不保证适用于其他节点类型、嵌套、motion、驱动或架构。

## 新工具及其正确性范围

[instance_protocol.py](../instance_protocol.py) 提供有边界检查的事件解码和ray变换。[run_instance_probe.py](../run_instance_probe.py) 的流程是：

1. 在同一CUDA context构建一个静态instance的IAS及其GAS。
2. 发射19/13条组，保存原始IAS返回。
3. 对0x16事件，host读取IAS镜像中的记录，检查child引用、ID、逆矩阵。
4. host用解出的矩阵变换ray，保持direction不归一化，保留t区间。
5. **独立开启新的child GAS查询**，不携带或恢复父级遍历状态。
6. 与世界坐标CPU结果及随后执行的OptiX结果核对ID、t、u/v和face。

IAS也可能返回潜在instance事件而child实际miss；工具会继续核对子查询。对于直接返回miss的IAS结果，则核对CPU/OptiX是否也miss。未知事件或CUDA fault均使检查失败。当前仅接受一个静态instance，正行列式变换；没有多instance、负缩放或instance flags覆盖。

复现命令均从 `ttu-bench/` 执行：

```bash
CUDA_VISIBLE_DEVICES=1 timeout 40s python3 run_instance_probe.py --instance identity --output results/local-instance-identity.json --check
CUDA_VISIBLE_DEVICES=1 timeout 40s python3 run_instance_probe.py --instance translate --instance-id 41 --output results/local-instance-translate.json --check
CUDA_VISIBLE_DEVICES=1 timeout 40s python3 run_instance_probe.py --instance rotate --instance-id 0 --output results/local-instance-rotate.json --check
CUDA_VISIBLE_DEVICES=1 timeout 40s python3 run_instance_probe.py --instance scale --instance-id 0xffffff --output results/local-instance-scale.json --check
CUDA_VISIBLE_DEVICES=2 timeout 40s python3 run_instance_probe.py --instance identity --scene grid --count 64 --instance-id 41 --output results/local-instance-grid.json --check
CUDA_VISIBLE_DEVICES=2 timeout 40s python3 run_instance_probe.py --instance scale --scene grid --count 64 --output results/local-instance-scaled-grid.json --check
```

[正式矩阵](../results/phase15/phase15-instance-matrix.json)：上述6项在GPU1/2全部通过，共928次IAS查询、874次独立child查询，合计1802次裸TTU launch。CPU/OptiX oracle比较另计，不是TTU launch数。13/19条两组都通过；两张卡按不同场景分配，不是完整的场景×GPU交叉矩阵。早期 `phase15-instance-*-initial` 和无gpu前缀结果为探索版本，以正式矩阵及各文件源码hash为准。

共用CUDA API绑定改动后，编码校验通过；[固定fixture回归](../results/phase15/phase15-fixture-regression.json) 在GPU2的8case/40launch通过，含原20条组。代码语法检查与现行文档链接检查通过。

初次identity探索在全部结果匹配后，host清理时因缺少cuMemFree_v2的ctypes签名将64位地址截断，返回CUDA1；已补充free和context destroy的API绑定。正式矩阵均正常退出，无GPU fault；原初始报告的数值匹配不等于其进程整体成功。

## 下一步：从“独立查child”到“正确恢复遍历”

先锁定一个identity instance，用上述已验证记录字段构造最小设备端事件处理，保留19条组返回；区分重新开启一次硬件会话与恢复整条ray的遍历状态。随后至少需要验证：

- 父级待处理兄弟分支和返回路径如何保存，哪些额外返回槽必须重提交。
- 子GAS结果或miss后如何恢复父级；不能只在child命中时提前返回。
- 两个instance前后重叠、输入顺序交换、近处miss而远处hit时仍与最近命中oracle一致。
- instance ID从实际命中路径获取，不用build输入或host预选答案代替。

当前host工具是事件字段与局部ray的参考实现，不进入将来的TTU计时区，也不作为已完成的ISA最小IAS路径。nonopaque accept/ignore/resume仍是另一项未完成的状态机任务。
