# MiniMax-H3 FL2VA 交付包运行说明

更新时间：2026-11-08T23:32:42+08:00（北京时间）。

本目录是从源码制作好的交付包，已经包含源码、离线依赖和参考视频。

1. 将整个目录上传到个人AFS根目录，命名为 `h3-submission`，保留所有分片、清单、scripts和reference目录。
2. 申请独占8卡。资源就绪后，将个人AFS最新完整容量统计 `storage-info.json` 上传到AFS根目录。已登录的平台客户端可用 `ifab --json afs-status > storage-info.json` 导出，统计距启动应不超过15分钟。
3. 在新八卡Bash初始目录执行以下命令，替换个人AFS路径。

~~~bash
set -euo pipefail
export IFAB_AFS_HOME=/path/to/your/afs
export H3_NODE_ROOT="$(dirname "$PWD")/h3-submission-native"
export MODEL_PATH=/public/home/aclsylqidf/shared/MiniMax-H3
export BUILD_JOBS=120
export H3_AFS_INFO_JSON="$IFAB_AFS_HOME/storage-info.json"
export H3_AFS_OUTPUT_DIR="$IFAB_AFS_HOME/h3-results/run-$(date -u +%Y%m%dT%H%M%SZ)"
cd "$IFAB_AFS_HOME/h3-submission"
unset H3_SHORT_SMOKE H3_INFINILM_VIDEO H3_REFERENCE_VIDEO
bash run_submission.sh
~~~

脚本自动安装环境、-j120编译、完整生成视频并保存AFS。需要生成后与随包参考视频比较时，将最后一行改为：

~~~bash
bash run_submission.sh check
~~~

实际安装环境位于节点 `$H3_NODE_ROOT/work/`：`venv/` 是Python入口，`python/` 是安装后的依赖，`install/` 是原生库，`toolchain/` 和 `xmake/` 是工具。节点环境随资源回收，同节点重试保留该目录可增量复用。

视频保存在 `$H3_AFS_OUTPUT_DIR/result.mp4`。到平台AFS文件管理的 `h3-results/run-时间戳/` 下载视频及日志：

| 内容 | 文件 |
|---|---|
| 视频 | `result.mp4` |
| 推理与总流程日志 | `generation.log`、`workflow.log` |
| 输入与参数 | `invocation.json` |
| 规格及运行状态 | `native-check.json`、`submission-status.json` |
| 保存清单 | `saved-manifest.json` |
| 官方比较原文，仅check模式 | `compare/compare.log` |
| 比较检查结果，仅check模式 | `evaluation.json` |

脚本退出0、末行pipeline_rc=0/save_rc=0，并存在AFS视频和保存清单，表示生成及保存成功。每次输出使用AFS内尚不存在的新目录。

参考视频仅适用于随包登记的测试用例。换图片需要登记FIRST_FRAME/H3_INPUT_SHA256；运行check还需更新对应reference视频和case.json并重新制包。平台Python/Torch/DTK与共享模型由赛事环境提供。
