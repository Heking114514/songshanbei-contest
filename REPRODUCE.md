# MiniMax-H3 FL2VA 运行说明

更新时间：2026-11-09T00:45:35+08:00（北京时间）。

## 放文件

将本仓库源码上传到 `个人AFS/songshanbei-contest/`，保证 `run_submission.sh` 位于该目录第一层。

将 `requirements/h3_linux_cp311.lock.json` 中的 55 个 wheel 原名放入 `songshanbei-contest/offline-wheels/`。获取方法见 [PACKAGING.md](PACKAGING.md)。`third_party/h3/` 中的 FlagTree、clang/lld 和 xmake 已随源码提供；输入图和参考视频也在仓库中。

准备后的结构和输出列表见 [README](README.md)。模型权重使用平台共享目录，不上传到个人 AFS。

## 运行

申请独占 8 卡。资源就绪后，用已登录的平台客户端执行：

~~~bash
ifab --json afs-status > storage-info.json
~~~

将文件放入个人 AFS 根目录。统计需完整，且距启动不超过 15 分钟；源码、wheel 和结果应合计小于个人 AFS 配额。

在新八卡 Bash 的初始工作目录执行，替换个人 AFS 路径：

~~~bash
set -euo pipefail
export IFAB_AFS_HOME=/path/to/your/afs
export H3_NODE_ROOT="$(dirname "$PWD")/h3-native"
export MODEL_PATH=/public/home/aclsylqidf/shared/MiniMax-H3
export BUILD_JOBS=120
export H3_AFS_INFO_JSON="$IFAB_AFS_HOME/storage-info.json"
export H3_AFS_OUTPUT_DIR="$IFAB_AFS_HOME/h3-results/run-$(date -u +%Y%m%dT%H%M%SZ)"
cd "$IFAB_AFS_HOME/songshanbei-contest"
unset H3_SHORT_SMOKE H3_INFINILM_VIDEO H3_REFERENCE_VIDEO
bash run_submission.sh
~~~

默认流程为：核验离线输入 → 复制源码到节点 → 安装环境 → 编译 H3 所需目标 → 完整生成视频 → 保存到 AFS。参数为 8 卡、seed 42、50 步请求、124 帧，Prompt 为 `A dog falls asleep`。

需要生成后运行官方比较时，将最后一行改为：

~~~bash
bash run_submission.sh check
~~~

`check` 使用本次新生成视频与 `reference/reference_fl2va.mp4` 调用原版官方比较器，再检查比较输出和视频规格；它不会重新生成参考视频。

## 拿结果

到平台 AFS 文件管理中的 `$H3_AFS_OUTPUT_DIR/` 下载：

| 内容 | 文件 |
|---|---|
| 视频 | `result.mp4` |
| 推理与总流程日志 | `generation.log`、`workflow.log` |
| 输入与参数 | `invocation.json` |
| 规格及运行状态 | `native-check.json`、`submission-status.json` |
| 保存清单 | `saved-manifest.json` |
| 官方比较原文，仅check模式 | `compare/compare.log` |
| 比较检查结果，仅check模式 | `evaluation.json` |

退出 0、末行 `pipeline_rc=0` 和 `save_rc=0`，并存在 AFS 视频及保存清单，表示生成及保存成功。失败时查看 `workflow.log` 和 `submission-status.json`；保存失败时，日志末行会给出节点产物位置。

## 重试与增量编译

同一节点、同一 AFS 源码目录重试时，保留 `H3_NODE_ROOT`；为本次结果设置新的 `H3_AFS_OUTPUT_DIR`，并更新容量统计。未变化的源码保持时间戳，已有环境和编译对象继续复用。源码更新后只复制变化文件，交由构建系统增量编译。

节点源码存在独立修改时，入口会拒绝覆盖；请先保存该修改，或为不同实验指定新的节点目录。新节点需要重新安装环境和构建，不能复用已回收节点的目录。

## 更换测试图片

默认图片为 `inputs/first_frame.png`。换图片后，运行前指定图片路径和它的 SHA256：

~~~bash
export FIRST_FRAME="$IFAB_AFS_HOME/songshanbei-contest/inputs/new_frame.png"
export H3_INPUT_SHA256=替换为新图片的64位SHA256
~~~

可用 `sha256sum inputs/new_frame.png` 获取校验值。Prompt 当前固定为 `A dog falls asleep`，其他生成参数如上。

`check` 还要求将 `reference/reference_fl2va.mp4` 换成新用例对应的参考视频，并更新 `reference/case.json` 中的图片、视频 SHA256、视频字节数和生成参数。旧参考视频不能用于新图片。参考视频来源与登记说明见 [reference/README.md](reference/README.md)；新用例仍需实际验收。
