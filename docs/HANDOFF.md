# 当前状态与下一步

环境为RTX3090 / SM86、driver535.309.01、CUDA12.1、OptiX8。协议、测量、renderer各有一个主题文档；不要继续按小功能拆phase，也不要在多份文档重复实验日志。

## 当前边界

- [协议](TTU_PROTOCOL.md)：opaque单GAS、一般ray/有限flags已验证；受限1–3静态instance能保存pending、查child、恢复父级。复杂IAS暂挂，保留四实例0x24/直接triangle返回、并列身份及nonopaque候选问题。
- [测量](MEASUREMENTS.md)：一个请求可拆多个合法MF组；正常MF不能替换NOP/MF0。同warp的FADD/HMMA/global/shared读可与未完成TTU部分重叠。HMMA仍用保守padding，不能当本征延迟。
- 多warp：匹配返回依赖槽后可安全消费；保持连接时其他warp仍能推进。共同窗口与FADD分组对照支持CTA warp模4的SMSP竞争类：同类有干扰，异类很小。仅两个类的TTU已达约169查询/1000周期；返回目的/槽位改变在七LD、末条CLOSE不变时仍影响吞吐。具体物理瓶颈及连接释放点未定位。
- [renderer](RENDERER.md)：Bunny含地面69,453面与单次地面镜面，三后端已有一致图像；res4/res2固定视角各一项几何残差失败。tag4/6有最小raw样本，语义未解；不要放宽ID/face。
- native默认monolithic O0。直接O1–O3破坏marker；linked caller可优化但寄存器上升且收益小，guard必须保留。

## 下一步：紧凑多warp

1. `validate_multiwarp_interactions`新增mapping/window/return-controls，支持固定seed打乱case及重复；具体命令、共同窗口定义和SB/stall资料来源只在MEASUREMENTS维护。默认同步仍保守，较紧配置显式指定。
2. 优先补变化输入的逐轮正确性控制，再分离“重复返回槽”和“重复目的寄存器”的影响。RZ不等于内部零流量，repeat也不是纯流量控制；不要把平台称RT Core峰值。
3. 在已确认的同/异调度类中加入INT/shared/LDG压力，必要时控制频率和扩大AS/访存工作集。HMMA依赖解释充分后再加Tensor压力。
4. 若继续追连接释放点，用多个holder和单次peer时间线收紧CLOSE前/后对照；当前一个holder只使peer小幅变化，不能据此推ticket容量。复杂IAS仍暂挂。

renderer无需继续堆模型/材质；若先改善它，独立对照稳健/watertight软件求交，或缩减native接口/数组搬运。小图与端到端加速比不能替代控制测量。

## 保存与工作方式

常规结果只留命令及简要结论。raw、cubin、逐case报告默认不留；特别重要的性质、推翻旧结论的反例才保留必要完整证据。重要旧发布数据、源码helpers、原始JIT/AS未删。复现和blog也是有效成果，不因缺少research novelty否定工具积累。

renderer默认报告、可选images/artifacts，普通编译目录自动清理；矩阵默认单摘要。临时结果不出现在公开文档里。不要新增COMMIT_PLAN或重复的NEXT_PROMPT；本文件就是交接入口。renderer、整理及本轮多warp代码/测量记录已提交；下一次从上述待办继续。
