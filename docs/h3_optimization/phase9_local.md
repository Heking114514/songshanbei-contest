# H3 第九阶段精确分派与验证

更新时间：2026-11-09T12:56:27.506753+08:00（北京时间）。本轮完整TP8验证已完成，八卡主动释放并确认清理。后续改动仅本地，AFS只读，正式stable不覆盖。

真实H3的Q/K/O连续、V是三倍packed-QKV stride，旧compact/padded策略未匹配，完整模型回退v4。phase8_attention_worker.py新增--layout mixed-v；phase9_attention.py的--layout-evidence验证当前worker源码/子库SHA与build ID、精确布局、数值/Graph/重复及原始计时后才增规则，失败不修改输入策略。27570长序列启用m128-n32-w4-s1，708短序列仍v4；未知布局仍v4。

同节点同上游完整三组对照：候选稳定同步步5.094603→4.394788秒（降13.736%）、主流程274.336→240.291秒（降12.410%）、外层368.883→345.459秒（降6.350%）。v4自重放全部49次输入及媒体位一致；候选首输入相同，从第二次DiT输入出现音视频状态差异。候选视频SSIM98.890947%、音频PCM相对L2差异4.452999%、相关系数0.9991575，听感未验收，仍实验属性。一个固定用例和同步/冻结诊断有扰动，不当无探针生产收益。

冻结仅循环前恢复一次block state与scheduler，后续正常迭代。真实profiler缺部分GPU native kernel，严格失败，无通信重叠结论。1206对象SHA/mtime不变、-j120零native编译，只约3.7秒宿主dispatcher构建；既有子AOT未重编。19项CPU回归通过。

完整记录在工作区experiments/h3-opt-20261107-220727/phase9_live3/run.md，保留清单同目录preserve-list.md。本地results.tar.gz 85成员全部验SHA，冻结包21.4MB已验内外SHA，phase9-dispatch-mixed全bundle已回收。AFS父final-v2、round2-v4-delta、本地audio8-native-delta及attention36.tar.gz仍必须保留。每次新租约按父链恢复并核零编译，不复用过期sid。

Git分支experiment/h3-phase9-local；初稿62efba4已在用户撤回前推送exp，本次mixed布局修正只提交本地。最终提交见仓库外phase9_live3/git-local-final.json。正式交接当前正文为handover/operations/phase9_local_optimization.md。

2026-11-09T16:12:31+08:00状态更新：用户重新授权推送，fa701ff已推远端exp并核实一致；后续第十阶段只存本地且不上机，详见[本地改造](phase10_local.md)。第九阶段是恢复第八阶段已有约4.395秒/步，不能报告新增提速。
