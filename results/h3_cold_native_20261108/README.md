# 2026-11-08 空 AFS、新八卡正式 native 验收

更新时间：2026-11-08T22:45:35+08:00（北京时间）。

本目录保存该次真实运行的原始证据副本及视频；evidence-files.json 登记复制来源、大小和 SHA。原始日志与 JSON 保留当时绝对路径和状态，不改写成发布后的 commit 身份。

## 结论与人工介入

从空 AFS、无旧环境和无旧对象的新八卡节点完成了环境、源码冷编译、完整 native 推理与 AFS 保存。**不是首次默认命令全程无干预成功**：首次 stage 选择 /tmp，因平台权限拒绝而退出1；手动设置 H3_NODE_ROOT 到本 sid 私有根下、workspace 同级目录后重跑成功。该修正只改变执行环境变量，没有修改当时载荷源码。

运行中终端 frames 读取曾为空；为观察进度，人工暂停并立即恢复后台任务，添加日志尾部快照读取。该干预发生在观测链，没有修改推理参数或跳过计算。prepare/build/完整生成/结果保存由重启后的同一正式脚本完成。

本提交将 H3_NODE_ROOT 改为启动必填，移除失败的 /tmp 默认路径，并补本地回归。**这次代码修正没有另申请新节点复跑，不能把本次运行归为修正后版本的无干预实测。**

## 真实结果

| 阶段 | 结果 |
|---|---|
| 上传 | 13文件，357530853字节，服务端大小/SHA通过 |
| 环境准备 | rc=0，约7分36秒 |
| 源码构建 | rc=0，-j120，约22分34秒，1194对象全部本轮新编译 |
| native | pipeline_rc=0、save_rc=0；2026-11-08T22:10:24+08:00完成 |
| 推理规格 | 完整层数、A dog falls asleep、seed42、50步请求/49次DiT、124帧、768×928、24fps、1音频流 |
| native耗时 | DiT433.297秒，pipeline484.989秒，端到端563.771秒；不含环境/编译 |
| 本地原版compare | rc=0、Size/Length PASS、SSIM96.819%；使用本轮native和历史reference |

本轮八卡默认入口没有 check，没有开发冒烟或提前计算预热，首用 JIT 在真实推理中发生。之后本地使用未修改官方 minimax_h3_compare.py 的 provided 参数完成比较与独立媒体验证。服务器 check 工作流尚未实机验收；没有同轮 PyTorch 性能基线，不拼接加速比，也不声称音质已验收。双方 math_bf16，历史 reference 为 head_batch4。

本轮实际源身份由 invocation.json 和 acceptance-summary.json 中的载荷/源码指纹绑定，基准 Git HEAD 为发布前版本加当时工作树。后续文档和节点路径校验修正使发布树不同于已运行载荷；下次必须从固定新 commit 重新制包。

## 踩坑与处置

- AFS API 上传不自动建父目录：控制器补逐层 mkdir；不属于模型代码。
- 上传曾出现413、raw后端403和下载忙：续用上传ID/offset，退回平台声明的64KiB JSON块，四文件并行；下载校验有限退避，已提交内容不重传。这些是工作区控制器修正，未把其个人平台控制代码发布为正式入口。
- /tmp建工程根被拒：运行时显式设置节点私有目录；本提交入口提前要求 H3_NODE_ROOT。workspace有限额，不用于大型工程树。
- 实时增长日志跨块下载会 FILE_CHANGED_DURING_DOWNLOAD：使用原子尾部快照观察，结束后核验不可变原始日志。
- 额外JIT保存的两次terminal/input被拒 INVALID_TERMINAL_INPUT，确切原因未查明：没有保存JIT，不宣称它已持久化。1194对象源码/安装缓存已经保存，但新缓存跨节点恢复尚未验证。

八卡后来按租约自然到期，旧 sid 不复用。历史AFS容量仅为时点数据，后续写入必须重新用专门API查询。

## 复现与证据

下一次运行遵循仓库 [REPRODUCE.md](../../REPRODUCE.md)，在新八卡初始workspace中先设置节点工程目录，再进入 AFS 交付包。原始失败见 first-entry-failure.log，生成参数和源码身份见 invocation.json，视频规格见 native-check.json，保存清单见 saved-manifest.json，比较原文见 official-comparison.log。

原始 saved-manifest.json 列出的是 AFS 上的完整9份产物，本目录仅摘录其中证据，不冒充该清单的完整镜像。本轮未发布 Release，也未验证最终版本匿名获取或官方换例。
