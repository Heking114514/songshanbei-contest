# H3 第九阶段本地候选

更新时间：2026-11-09T08:36:23.874829+08:00（北京时间）。

已验方案固定在 `5ec6c504dc78c89cff2f776d6d2505e373e17dd9` 和标签 `h3-phase8-verified-20261109`；本分支 `experiment/h3-phase9-local` 新增精确shape/stride注意力分派、完整上游冻结回放和原始DiT热点采集。本轮没有连接平台或操作AFS，没有Linux/gfx936编译或GPU测试。

入口 [phase9_pipeline.py](../../scripts/experiments/phase9_pipeline.py) 默认只生成计划，显式--execute才允许未来Linux节点运行。[phase9_attention.py](../../scripts/experiments/phase9_attention.py)从SHA绑定的36组旧实测生成策略，默认离线，不重编Triton；未来额外编译宿主分派器。未知布局用v4，已选子库失败直接报错，子HIP模块保持进程生命周期。每种key记录一次完整stride。

冻结在Diffusers去噪循环前保存整个block state和scheduler状态，replay只恢复一次；首个完整DiT输入逐字段SHA一致才继续，后续正常迭代。固定语义参数、输入媒体、解析后模型配置和上游源码，尚未核全模型权重。CPU测试使用固定Diffusers wheel真实循环、denoiser与Euler方法体，改变模型后差异会传播到后续输入。

profiler输出原始Chrome trace，逐GPU区间并集和通信/计算交集统计；缺任一所选GPU kernel会拒绝采集完成。名称分类是启发式，不能把相加kernel时间或重叠率当端到端收益。冻结、哈希和profiler属于诊断；性能需另跑无探针重复轮次。

新增17项CPU检查、前轮23项回归、AST/help和Bash语法/source副作用检查通过。Windows编译模拟ABI子库验证路由、并发计划、模块生命周期、错build/缺库及错误透传，这不代表真实Linux/GPU已验。证据见 [phase9/](phase9/)。完整交接正文位于工作区 `handover/operations/phase9_local_optimization.md`。

第八阶段稳定步5.095→4.395秒、主流程297.54→238.92秒仍是此前native v4对比；本轮无新增模型性能结论，前轮同DiT输入音频1.975%差异与音质未验收边界继续保留。既有缓存链见 [phase8检查点](phase8_checkpoint.md)。实验Git保存在源码仓库自己的.git，与正式submit_repo独立。
