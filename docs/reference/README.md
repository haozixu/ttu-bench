# 参考资产，不是当前运行入口

本页位于 `docs/reference/`，参考资产仍保存在项目根目录的 `reference/`；下文行内资产路径与命令以 `ttu-bench/` 为基准。固定输入功能 runner 读取 [fixtures](../../fixtures/)。参考文件用于追溯原始编码、几何生成方式与 AS 格式，不能替代通过解析求交的功能验证。

## history/

清理前README/INDEX/HANDOFF/NEXT_PROMPT及部分原源码的恢复副本。包含后来撤回的解释，先读 [历史记录说明](history/README.md)。文档快照在 `docs/reference/history/`，恢复源码在 `reference/history/source/`。独立指令行为复现也有保存价值，不以几何oracle或research新颖性作为唯一门槛；当前复现入口与delta见 [EOSS_REPRO](../EOSS_REPRO.md)。

## jit/

实例事件分支的普通SASS新证据见 [jit16-instance-ordinary.sass](../../reference/jit/jit16-instance-ordinary.sass) / [来源manifest](../../reference/jit/jit16-instance-disassembly.json)，对应 [Phase15](../PHASE15.md)。`disassemble_ttu.py` 可复现提取；TTU换NOP只用于反汇编，输出不可执行。

其余20个 `jit_00..20.elf` 原始快照（`jit_16.elf` 在 fixtures）。它们来自前期 OptiX 8 / driver535.309.01 进程中的 JIT 捕获，可能包含不同遍历家族、构建核或已链接/未链接模块。保留原字节用于后续比较，不保证每个镜像是标准工具可直接反汇编或独立发射的完整 CUDA ELF。

原始文件大小/hash 见 [cleanup-2026-11-02.json](../../cleanup-2026-11-02.json) 的 moved 项。当前 AS 生成器没有重建这些 JIT 快照的功能。

## optix/

`optix_min.cu` 是原始 device 程序；`build_fixture.cpp` 是移除旧描述符猜测和扫描分支后的 host。它用 `(1,0,0),(0,2,0),(0,0,3)` 建立 GAS，执行一次 OptiX hit，向当前工作目录写出 `optix_as.bin`。

这是几何/生成过程的参考，**不保证重建出与已验证 fixture 逐字节相同的 AS**。在单独目录运行，避免覆盖 fixtures；新 AS 需要新的布局验证和 hash。示例从 ttu-bench 目录执行，CUDA路径可按机器调整：

```bash
mkdir -p /tmp/ttu-optix-reference
nvcc -ptx -arch=sm_86 -I ../OptiX-8.0.0/include reference/optix/optix_min.cu -o /tmp/ttu-optix-reference/optix_min.ptx
g++ -std=c++11 -I ../OptiX-8.0.0/include -I /home/hzx/cuda-12.1/include reference/optix/build_fixture.cpp -L /home/hzx/cuda-12.1/lib64 -lcuda -ldl -o /tmp/ttu-optix-reference/build_fixture
```

选择空闲GPU后，在 `/tmp/ttu-optix-reference` 中运行 `CUDA_VISIBLE_DEVICES=7 timeout 20s ./build_fixture optix_min.ptx`。本次清理只做编译核验，没有重新运行这个生成器或替换固定输入。

## vulkan/

`vk_bvh_dump.c` 是旧的双三角形 BLAS 生成参考；`as_dump_0/1/2.bin` 为不同 VA 的原始镜像。它们不是当前 OptiX 单三角形 fixture，不能交给当前 runner。文件内的旧 CUDA/Vulkan 直接读址探针只说明当时特定映射的行为，不能推出两个API天然不能共享内存。

本轮未重新验证 Vulkan 生成器，不把源内旧注释当作通用硬件结论。
