# 当前状态与下一步

环境为RTX3090 / SM86、driver535.309.01、CUDA12.1、OptiX8。协议、测量、renderer各有一个主题文档；不要继续按小功能拆phase，也不要在多份文档重复实验日志。

## 当前边界

- [协议](TTU_PROTOCOL.md)：opaque单GAS、一般ray/有限flags已验证；受限1–3静态instance能保存pending、查child、恢复父级。复杂IAS暂挂，保留四实例0x24/直接triangle返回、并列身份及nonopaque候选问题。
- [测量](MEASUREMENTS.md)：一个请求可拆多个合法MF组；正常MF不能替换NOP/MF0。同warp的FADD/HMMA/global/shared读可与未完成TTU部分重叠。HMMA仍用保守padding，不能当本征延迟。
- [renderer](RENDERER.md)：Bunny含地面69,453面与单次地面镜面，三后端已有一致图像；res4/res2固定视角各一项几何残差失败。tag4/6有最小raw样本，语义未解；不要放宽ID/face。
- native默认monolithic O0。直接O1–O3破坏marker；linked caller可优化但寄存器上升且收益小，guard必须保留。

## 下一步：紧凑多warp

1. 先用稳定热小opaque GAS，同CTA设1/2/4/8个TTU warp，减少逐sample快照，核对SMID、warp请求/输出及寄存器/shared/驻留。匹配无TTU对照，测完成与总吞吐。
2. 使用已有合法拆组，分别延长OPEN→GO、GO→LD.CLOSE，观察其他warp的准入/吞吐；匹配新增普通指令成本，不把台阶直接叫buffer容量。
3. 不同warp分别做TTU与FADD/INT/shared/LDG，测双向影响、轮换角色位置并证明重叠。HMMA依赖解释充分后再加Tensor压力。
4. 可重复选择性干扰出现后，再扩大AS/访存工作集、比较同/异SM并控制频率，区分接口/驻留和L1/L2/DRAM。

renderer无需继续堆模型/材质；若先改善它，独立对照稳健/watertight软件求交，或缩减native接口/数组搬运。小图与端到端加速比不能替代控制测量。

## 保存与工作方式

常规结果只留命令及简要结论。raw、cubin、逐case报告默认不留；特别重要的性质、推翻旧结论的反例才保留必要完整证据。重要旧发布数据、源码helpers、原始JIT/AS未删。复现和blog也是有效成果，不因缺少research novelty否定工具积累。

renderer默认报告、可选images/artifacts，普通编译目录自动清理；矩阵默认单摘要。临时结果不出现在公开文档里。不要新增COMMIT_PLAN或重复的NEXT_PROMPT；本文件就是交接入口。当前renderer仍未提交，提交操作按用户后续安排。
