# 2026-11-02 清理说明

本文件保留原清理的范围，并记录其纠正。用户允许删除误导路线，但把未通过几何oracle的EoSS复现、编码helpers和工程差异一起删除超出了合理清理边界。

## 过度清理的纠正

已从清理前会话输出恢复旧README/INDEX/HANDOFF/NEXT_PROMPT到 `reference/history/`；其中错误解释保留原文并由目录README警示。部分源码可与删除前hash逐字节核对，恢复为历史副本。原 `run_exp.py` / `build_and_patch.py` 完整内容未找回，现行 `tools/`下同名入口是重建版，不能声称原样恢复。

新增共用 `src/ttu_isa.py`、`src/eoss_helpers.py`、`src/cubin_builder.py`、`src/sm86_fixture.py`，并让minimal实际复用。EoSS复现、SM86/SM120 delta和负结果详见 [EOSS_REPRO.md](EOSS_REPRO.md)。研究新颖性、几何功能正确性、指令行为复现分别评价；不再以前两项缺失否定后一项的保存价值。原删除manifest保持原状，以便审计恢复身份。

## 保留与迁移

- `fixtures/` 保存当前 runner 使用的 `jit_16.elf` 与 `optix_as.bin`，字节不变；代码路径同步更新。
- `reference/jit/` 保留其余20个原始 JIT 镜像，可能用于比较遍历家族和不同配置，不把它们称为全部可独立重放的 kernel。
- `reference/optix/` 保留 device 程序和精简后的 AS 生成参考；从旧 host 删除描述符猜测、失效扫描及其参数入口。
- `reference/vulkan/` 保留三个不同 VA 的原始 AS 和生成参考，仅用于格式比较。重复的 `as_dump.bin` 与 `as_dump_0.bin` 完全相同，删除前者。
- Phase12的功能结果、反例对照、GPU信息、资源报告和单次结果JSON保留，现收在 `results/phase12/`。曾保存的 cubin 含进程 VA，不能跨进程重放，已删除；runner 可重新生成。

## 删除内容与理由

| 删除族 | 原因 |
|---|---|
| `build_ttu*`、`run_ttu*`、`run_exp.py`、`probe*`、`movref.cu`、旧 build_and_patch | 原来将无几何oracle与无保留价值混淆，删除过多；现恢复可找回历史副本、重建EoSS入口并提取helpers。功能线不能替代全部指令/协议实验 |
| `run_mk.py` | 完整 megakernel 后半段未跑通，混合大量旧补丁、重复定义和无效分支；只抽取当前需要的 CUDA/ELF 部分到 cuda_driver.py |
| `hook*`、`detour.c`、`patch_capture.cpp`、`watch_capture.cpp` | 为上传拦截/私有绑定侦察而建，不是当前 ISA 功能路径依赖 |
| `scan*`、`stage_scan.cpp`、`heapdiff.cpp`、`rela_scan.cpp`、`kprobe*`、`drive_kprobe.py`、`test_dtoh.cpp` | 为寻找所谓缺失驱动现场而做的扫描/探针；独立 context 已排除此依赖，保留会诱导重走旧路线 |
| `heap_mid/post.bin`、`xtab_*.bin`、`vmm_dump.bin` | 旧进程地址与私有状态快照，当前实验不引用；不保留为通用驱动机制证据 |
| `_text_*.bin`、旧反汇编 txt、e7/probe3 cubin、旧 ELF 工具 | 重复或不完整派生物，部分曾将 TTU NOP 化；原始 JIT 保留，现有 ELF parser 可直接读原 section |
| 旧模板 cubin、pyc | 可由当前源文件再生的缓存；验证可能重新创建 |

共移除43个旧文本/代码文件及25个二进制/缓存文件，原内容合计34,520,456字节；另重构旧 OptiX host 并重写入口文档。详细路径、大小、删除前 SHA256 与迁移目标记录在 `docs/reference/cleanup-2026-11-02.json`。清单不包含被删除文件的内容。

## 文档与验证

README、INDEX、HANDOFF、NEXT_PROMPT 和 `research/notes/ttu-reversing-implications.md` 已替换旧内容。PHASE12 保留通过验证的协议和有限纠错表，移除指向已删除 runner 的命令。文献判断已纳入 Ray-by-Ray / RTSpMSpM 全文，不再将这些内容标为缺全文或无人研究。

功能入口不导入run_mk或EoSS；恢复的EoSS复现入口使用本地框架。清理后的GPU5/7回归在 `results/maintenance/cleanup-functional.json`；提取helpers后的GPU5回归在 `results/maintenance/helpers-functional.json`，8场景40次launch通过。CPU编码检查核对原20条组、精简hash、EoSS保真patch及历史源码hash；新EoSS结果单独保存，超时不改写为成功或忽略。
