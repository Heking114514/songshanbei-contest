# songshanbei-contest

更新时间：2026-11-09T00:45:35+08:00（北京时间）。

## 嵩山杯 MiniMax-H3 FL2VA 初赛

统一输入图片：`inputs/first_frame.png`

统一 FL2VA Prompt：`A dog falls asleep`

### 复现指令

1. **上传源码。** 下载本仓库，将整个源码目录上传到个人 AFS 根目录。下面统一将这个目录命名为 `songshanbei-contest/`，其中应直接包含 `run_submission.sh`、`InfiniCore/`、`InfiniLM/` 等文件。
2. **放入离线依赖。** 将配套的 **55 个 wheel 文件**原名放入源码目录的 `offline-wheels/`，不要再套一层目录。三份工具已在仓库的 `third_party/h3/` 中，无需另外下载。wheel 的获取与校验方法见 [离线依赖说明](PACKAGING.md)；**当前尚未发布 Release**。
3. **申请独占 8 卡并运行。** 资源就绪后，导出个人 AFS 的最新容量统计，将 `storage-info.json` 放到 AFS 根目录：

~~~bash
ifab --json afs-status > storage-info.json
~~~

统计需完整且距启动不超过 15 分钟。上传完成后的目录应为：

~~~text
个人AFS/
├── storage-info.json
└── songshanbei-contest/             # 本仓库源码
    ├── run_submission.sh
    ├── InfiniCore/
    ├── InfiniLM/
    ├── ninetoothed/
    ├── scripts/
    ├── requirements/
    ├── third_party/h3/             # 仓库已带的三份工具
    ├── offline-wheels/             # 放入55个wheel，约183MiB
    ├── inputs/first_frame.png
    └── reference/                  # 仓库已带的参考视频及用例登记
~~~

打开新八卡 Bash，**先在初始工作目录执行以下命令**，将 AFS 路径替换成自己的实际路径：

~~~bash
set -euo pipefail
export IFAB_AFS_HOME=/path/to/your/afs

# 在进入AFS前，确定本节点的环境和编译目录。
export H3_NODE_ROOT="$(dirname "$PWD")/h3-native"
export MODEL_PATH=/public/home/aclsylqidf/shared/MiniMax-H3
export BUILD_JOBS=120
export H3_AFS_INFO_JSON="$IFAB_AFS_HOME/storage-info.json"
export H3_AFS_OUTPUT_DIR="$IFAB_AFS_HOME/h3-results/run-$(date -u +%Y%m%dT%H%M%SZ)"

cd "$IFAB_AFS_HOME/songshanbei-contest"
unset H3_SHORT_SMOKE H3_INFINILM_VIDEO H3_REFERENCE_VIDEO
bash run_submission.sh
~~~

脚本自动核验源码和依赖、复制源码到节点、安装环境、以 `-j120` 编译 H3 所需目标、执行完整推理并保存结果。编译与推理进度实时显示。

需要本次生成后运行官方比较时，将最后一行改为：

~~~bash
bash run_submission.sh check
~~~

两种方式选一种。`check` 会新生成本次视频，再与 `reference/reference_fl2va.mp4` 比较；参考视频直接复用。

### 输出位置

结果保存在 `H3_AFS_OUTPUT_DIR`，例如 `个人AFS/h3-results/run-时间戳/`。到平台 AFS 文件管理中打开该目录即可下载。

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

脚本退出 0，末行 `pipeline_rc=0`、`save_rc=0`，且 AFS 中存在视频和保存清单，表示生成及保存成功。

### 环境在哪里

- **AFS 源码目录**：上传的仓库、55 个 wheel、输入图与参考视频。
- **节点 `$H3_NODE_ROOT/source/`**：脚本复制的源码及编译产物。
- **节点 `$H3_NODE_ROOT/work/`**：脚本安装的 Python 依赖、工具链及原生库；环境由脚本自动加载。
- **AFS `h3-results/`**：最终视频和日志。

相同节点重试时保留同一个 `H3_NODE_ROOT`，脚本只更新变化的源码，复用环境和增量编译产物；每次结果使用新的 `H3_AFS_OUTPUT_DIR`。节点回收后，节点环境也会回收。

平台需提供 Python 3.11、Torch 2.9.0/HIP 6.3、DTK/gfx936 和共享模型权重。详细目录见 [ENVIRONMENT.md](ENVIRONMENT.md)，运行说明见 [REPRODUCE.md](REPRODUCE.md)。换测试图片的登记方法及参考视频要求见 [REPRODUCE.md](REPRODUCE.md#更换测试图片)。
