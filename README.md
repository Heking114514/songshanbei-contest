# songshanbei-contest

更新时间：2026-11-08T23:32:42+08:00（北京时间）。

## 嵩山杯 MiniMax-H3 FL2VA 初赛

统一输入图片：`inputs/first_frame.png`

统一 FL2VA Prompt：`A dog falls asleep`

### 复现指令

1. **准备上传目录。** 本仓库是源码。`h3-submission/` 是 [制包步骤](PACKAGING.md) 生成的交付目录，内含源码和全部离线依赖；克隆仓库后需先制包。已经收到该交付目录时，可直接使用。

2. **上传文件。** 将整个 `h3-submission/` 上传到个人AFS根目录，保留目录结构。`payload.part*`、`delivery-manifest.json`、`scripts/` 和 `reference/` 全部上传。离线依赖已包含在分片中。

3. **申请独占8卡并运行。** 资源就绪后，使用已登录的平台客户端导出容量统计，将生成的 `storage-info.json` 上传到个人AFS根目录：

~~~bash
ifab --json afs-status > storage-info.json
~~~

统计需完整且距启动不超过15分钟。随后打开新八卡Bash，在初始工作目录执行以下命令，将AFS路径改为自己的实际路径：

~~~bash
set -euo pipefail
export IFAB_AFS_HOME=/path/to/your/afs

# 在进入AFS前，确定本节点的环境和编译目录。
export H3_NODE_ROOT="$(dirname "$PWD")/h3-submission-native"
export MODEL_PATH=/public/home/aclsylqidf/shared/MiniMax-H3
export BUILD_JOBS=120
export H3_AFS_INFO_JSON="$IFAB_AFS_HOME/storage-info.json"
export H3_AFS_OUTPUT_DIR="$IFAB_AFS_HOME/h3-results/run-$(date -u +%Y%m%dT%H%M%SZ)"

cd "$IFAB_AFS_HOME/h3-submission"
unset H3_SHORT_SMOKE H3_INFINILM_VIDEO H3_REFERENCE_VIDEO
bash run_submission.sh
~~~

脚本自动完成环境安装、`-j120` 编译、完整推理和AFS结果保存；编译与推理进度实时显示。

如果本次需要生成后运行官方比较，将上述最后一行改为：

~~~bash
bash run_submission.sh check
~~~

两种方式选一种。`check` 会新生成native视频，再与随包 `reference/reference_fl2va.mp4` 比较，参考视频直接复用。

### 输出位置

运行结果保存在启动时设置的 `H3_AFS_OUTPUT_DIR`，例如 `个人AFS/h3-results/run-时间戳/`。在平台AFS文件管理中打开该目录即可下载。

| 内容 | 本次结果目录内的路径 |
|---|---|
| 生成的视频 | `result.mp4` |
| 推理日志与各阶段耗时 | `generation.log` |
| 环境、编译及推理总日志 | `workflow.log` |
| 输入与生成参数 | `invocation.json` |
| 视频规格与指纹 | `native-check.json` |
| 运行状态与保存清单 | `submission-status.json`、`saved-manifest.json` |
| 官方比较原文，仅check模式 | `compare/compare.log` |
| 比较检查结果，仅check模式 | `evaluation.json` |

脚本退出0、末行 `pipeline_rc=0` 和 `save_rc=0`，且AFS中存在视频和保存清单，表示生成及保存成功。

### 文件和环境放在哪里

| 位置 | 内容 |
|---|---|
| 克隆得到的源码仓库 | 源码、脚本、依赖清单、三份构建工具、输入图及参考视频 |
| 制包生成的 `h3-submission/` | 上传用的源码和离线依赖分片、清单及启动入口 |
| 个人AFS `h3-submission/` | 上传后的交付包，长期保留 |
| 八卡节点 `$H3_NODE_ROOT/work/` | 脚本安装的Python环境、依赖、工具与原生库 |
| 个人AFS `h3-results/` | 保存的视频、日志和评测结果 |

节点环境由脚本创建，随节点资源回收；相同节点和相同交付包重试时保留 `H3_NODE_ROOT`，复用环境与增量编译产物。平台提供Python3.11、Torch2.9.0/HIP6.3和DTK/gfx936，模型使用 `MODEL_PATH` 指向的共享权重。

### 离线依赖与参考视频

- 从源码制作交付目录：见 [PACKAGING.md](PACKAGING.md)，包括55个wheel的下载、放置和打包命令。
- 完整节点环境结构：见 [ENVIRONMENT.md](ENVIRONMENT.md)。
- 当前交付包由制包命令生成，尚未发布Release下载入口。
- 换测试图片后，`check` 需要对应的新参考视频及 `reference/case.json`；当前入口Prompt固定为 `A dog falls asleep`。
