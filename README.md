# MiniMax-H3 FL2VA 提交工程

更新时间：2026-11-08T22:48:17+08:00（北京时间）。

正式入口为根目录 run_submission.sh。先上传交付材料到个人AFS，再申请独占八卡：

- bash run_submission.sh：构建环境、-j120构建所需H3目标、完整生成，保存 result.mp4 到指定AFS新目录。
- bash run_submission.sh check：同样完整生成，再使用独立 reference 目录的既有参考视频跑原版官方比较与独立检查。

正式流程无冒烟或缩层，首用JIT计入真实推理。编译/推理日志实时打印并保存，每30秒心跳；保留计算和保存退出码。源码、环境和对象放节点目录；必须在cd到AFS前按 [完整命令](REPRODUCE.md) 设置 H3_NODE_ROOT，避免平台禁止的/tmp及受限workspace。

2026-11-08已完成空AFS、新八卡环境、1194对象源码冷构建、完整124帧native及AFS保存，pipeline_rc/save_rc均0。首次/tmp错误后人工修正路径，运行中也处理过日志观测，不能称首次全程无干预成功。随后本地原版官方compare：Size/Length PASS、SSIM96.819%。[原始证据和视频](results/h3_cold_native_20261108/README.md)随仓库保存。

本提交要求显式节点目录并补本地回归；修正后版本尚未新节点复验，服务器check流程和官方换例尚未实机验收。使用历史reference，无双方同轮加速比，音质未独立验收。另一agent的候选算子未合入本工程。

完整上传结构、依赖、制包和执行见 [REPRODUCE.md](REPRODUCE.md)；参考视频见 [reference](reference/README.md)；报告事实见 [复现与提交说明](复现与提交说明.md)；来源与边界见 [VERSIONS.md](VERSIONS.md)。

自用入口 scripts/run_h3_development.sh 保留单卡冒烟和八卡短推理；旧 scripts/run_h3_submission.sh 草案已停用。
离线依赖Release尚未发布，最终公开版本匿名获取未验证。提交后仍需从该commit重新制作交付包，旧final包保持原始快照。
