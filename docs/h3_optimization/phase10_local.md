# H3 第十阶段本地结构改造

更新时间：2026-11-09T16:12:31+08:00（北京时间）。已验fa701ffd7cbcdc65303085b538b245a9a4c9a9e4已推送远端exp；本阶段新改动只留本地，不上机、不读写AFS，不改正式main。

新增phase10_flash_kernel/build_flash：完整KV块和尾块分离，可选V预取；旧BF16/FP32在线softmax顺序、真实mixed-V stride与ABI保留。构建器默认仅计划，--execute-build才编gfx936 AOT，独立插件不覆盖旧dispatcher。phase10_attention_worker固定第九阶段最佳bundle SHA，比较字段versus_previous_best，含重复/Graph/小形状math检查。

新增phase10_profile：外部hipprof采集native工作线程及Graph replay，默认collect只计划，--execute-trace才执行。日志实时输出和15秒心跳；已知OPS_*/STR_TABLE schema按物理GPU验证覆盖并求区间并集/重叠。仅stats、缺GPU事件、未知schema或缺卡失败；真实工具/schema待GPU节点确认。phase9_profile不再把pid当device。

GEMM probe新增paired测量，phase10_gemm_timing提供基线/候选交错调度、每次候选前后测BLAS、检查位一致/10%漂移；selector的--require-paired使用相邻较快基线，继续四组标量、两独立报告和2%收益门槛。16候选和模型默认Lt关闭保留，没有新Lt性能结论。

本地60项检查通过（18新增、19第九阶段、23第八阶段）。CPU kernel shim实际执行新旧函数体并核dense参考，不能替代Triton编译/GPU数值、Graph或性能验收。首轮SQLite关闭缺陷已修复，失败与成功日志均保留在工作区phase10。

未来性能对照为4.394788秒/步的第九阶段最佳mixed-V包，manifest SHA：1f7aeecd37c092fae4fbd8a6adbfde667e6fcea56e3650c8ed2777e7f8d759e8。第九阶段相对第八阶段4.394925秒没有新增提速证据；19.70%包含VAE变化，固定VAE4的历史pipeline改善约12.4%–12.5%。音频23.2%/9.95%/4.45%不是同输入连续改进，音质未验收。

规范操作入口为工作区handover/operations/phase10_local_optimization.md；旧phase9_local.md保留历史验证边界。后续按已有父链恢复native缓存，-j120精确增量守卫，不clean，本候选只另编AOT。
