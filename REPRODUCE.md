# MiniMax-H3 FL2VA 交付包操作说明

更新时间：2026-11-08T23:27:59+08:00（北京时间）。

本目录已经包含源码、离线Python依赖、构建工具和参考视频。将整个目录上传到个人AFS后，在独占八卡Bash中运行。

## 这个目录和运行环境是什么关系

你现在拿到的是制作好的上传交付包。它由源码仓库中的 `python scripts/h3_delivery.py package ... --output ../h3-submission` 生成；目录名 `h3-submission` 来自打包命令的输出参数。

本目录保存的是源码与离线依赖压缩分片。将其上传到AFS后启动 `run_submission.sh`，脚本才会在八卡节点 `H3_NODE_ROOT` 下解压源码、安装环境和编译。

~~~text
AFS/h3-submission/                   # 你上传的交付包，保留在AFS
    ↓ 运行run_submission.sh
$H3_NODE_ROOT/                      # 节点私有工程根
├── source/                         # 自动解压的源码
├── wheelhouse/                     # 自动解压的原始wheel
├── work/
│   ├── venv/bin/python3            # 本工程Python入口
│   ├── python/                     # 安装后的Python依赖
│   ├── install/                    # 编译安装的原生库
│   ├── toolchain/                  # clang/lld
│   ├── xmake/                      # 构建工具
│   └── jit/                        # 推理JIT缓存
└── runs/本次运行/                   # 节点上的本次结果
    ↓ 自动保存
$H3_AFS_OUTPUT_DIR/result.mp4        # AFS上的最终视频
~~~

Python/Torch和DTK由平台提供，共享模型目录由 `MODEL_PATH` 指定。本工程的Python依赖、工具和原生库安装在节点 `$H3_NODE_ROOT/work/`，由脚本自动加载。节点环境随资源回收，AFS交付包和已保存结果长期保留。

## 1. 上传到 AFS

将整个 `h3-submission/` 上传到个人AFS根目录，保留所有文件名和子目录。AFS上的结构应为：

~~~text
个人AFS/
├── h3-submission/
│   ├── run_submission.sh
│   ├── REPRODUCE.md
│   ├── UPLOAD.md
│   ├── delivery-manifest.json
│   ├── payload.part000
│   ├── payload.part001
│   ├── payload.part002
│   ├── payload.part003             # 分片数量以实际清单为准
│   ├── scripts/
│   │   ├── h3_delivery.py
│   │   └── h3_source_identity.py
│   └── reference/
│       ├── reference_fl2va.mp4
│       ├── case.json
│       └── README.md
├── storage-info.json
└── h3-results/                     # 完成运行后自动保存结果
~~~

`payload.part*` 全部上传，保持分片形式；脚本会在计算节点合并、解压和编译。`reference/` 与分片一并上传。

上传结束、八卡资源就绪后，准备个人AFS的最新完整容量统计 `storage-info.json`，放到AFS根目录。使用已安装并登录的平台 `ifab` 客户端时，可导出后上传此文件：

~~~bash
ifab --json afs-status > storage-info.json
~~~

统计应完整且距启动不超过15分钟；排队较久需重新导出。个人AFS上限4GiB，脚本默认预留128MiB结果空间和32MiB余量。该统计文件是启动必需输入。

## 2. 在八卡 Bash 中启动

申请独占8卡资源并打开 Bash。在新打开的初始工作目录中执行下面整段命令；先确定节点构建目录，再进入AFS。

平台通常已设置 `IFAB_AFS_HOME`；若没有，先执行 `export IFAB_AFS_HOME=/实际个人AFS根目录`。`MODEL_PATH` 指向平台共享 MiniMax-H3 权重目录，路径不同则替换。

~~~bash
set -euo pipefail
: "${IFAB_AFS_HOME:?请先设置个人AFS根目录}"

# 在初始工作目录中设置；同一交付包重试时保留这个路径。
export H3_NODE_ROOT="$(dirname "$PWD")/h3-submission-native"
export MODEL_PATH=/public/home/aclsylqidf/shared/MiniMax-H3
export H3_BASE_PYTHON=/usr/local/bin/python3
export BUILD_JOBS=120
export H3_AFS_INFO_JSON="$IFAB_AFS_HOME/storage-info.json"
export H3_AFS_OUTPUT_DIR="$IFAB_AFS_HOME/h3-results/native-$(date -u +%Y%m%dT%H%M%SZ)"

cd "$IFAB_AFS_HOME/h3-submission"
unset H3_SHORT_SMOKE H3_INFINILM_VIDEO H3_REFERENCE_VIDEO
bash run_submission.sh
~~~

脚本依次完成：校验和展开交付包 → 安装离线依赖与工具 → `-j120` 编译所需H3/Core/绑定 → 完整生成视频 → 校验视频 → 保存到 `H3_AFS_OUTPUT_DIR`。

默认用例为随包 `inputs/first_frame.png`、Prompt `A dog falls asleep`、seed42、50步请求、124帧、完整模型层数。正式入口直接运行完整推理。编译和推理日志实时显示，每30秒输出心跳。

目标运行环境为赛事 Linux x86_64、Python3.11、平台 Torch2.9.0/HIP6.3、DTK/gfx936。Python依赖和构建工具从交付包安装，Torch与模型权重使用平台已有版本。

`H3_NODE_ROOT` 必须是节点私有目录，位于初始workspace同级；源码、环境和编译产物放在那里。相同包重试时复用同一路径进行增量编译。每次 `H3_AFS_OUTPUT_DIR` 则应设置为个人AFS内尚不存在的新目录。

## 3. 需要官方比较时，加 check

默认模式只生成并保存视频。需要生成后比较时，完成上述环境变量配置，将最后的运行命令改为以下两行；两种模式选择一种即可：

~~~bash
export H3_AFS_OUTPUT_DIR="$IFAB_AFS_HOME/h3-results/check-$(date -u +%Y%m%dT%H%M%SZ)"
bash run_submission.sh check
~~~

`check` 仍会生成一份新的完整native视频，然后使用AFS交付目录中 `reference/reference_fl2va.mp4` 运行原版官方比较，并检查结果。参考视频只需随包上传一次，后续运行直接复用。

参考视频必须与当前图片、Prompt、seed和生成参数匹配。换输入图时，可用 `FIRST_FRAME` 指定图片，并用 `H3_INPUT_SHA256` 登记其SHA256；如需 `check`，还必须准备该用例对应的参考视频、更新 `reference/case.json` 并重新制包。当前入口的Prompt固定为 `A dog falls asleep`。

## 4. 从 AFS 取回产物

本次结果目录就是启动前设置的 `H3_AFS_OUTPUT_DIR`。运行结束后可查看：

~~~bash
printf '本次结果目录：%s\n' "$H3_AFS_OUTPUT_DIR"
ls -lh "$H3_AFS_OUTPUT_DIR"
~~~

在平台AFS文件管理中进入 `h3-results/`，打开对应的 `native-时间戳/` 或 `check-时间戳/`，下载 `result.mp4`；需要完整运行记录时，下载该次结果目录中的文件。

| 内容 | 相对于本次结果目录的路径 |
|---|---|
| 生成的视频 | `result.mp4` |
| 推理日志及各阶段耗时 | `generation.log` |
| 环境、编译和推理总日志 | `workflow.log` |
| 输入、生成参数及源码指纹 | `invocation.json` |
| 环境与加载库信息 | `origins.json`、`preflight.json` |
| 视频帧数、尺寸和指纹 | `native-check.json` |
| 运行状态 | `submission-status.json` |
| 保存文件清单及SHA256 | `saved-manifest.json` |
| 官方比较原文，仅check模式 | `compare/compare.log` |
| 比较检查结果，仅check模式 | `evaluation.json` |

例如，若结果目录为 `h3-results/native-20261108T140000Z/`，视频就在：

~~~text
个人AFS/h3-results/native-20261108T140000Z/result.mp4
~~~

成功标志：脚本退出码为0，末行 `pipeline_rc=0` 且 `save_rc=0`；AFS内有 `result.mp4`、状态为 `complete` 的 `submission-status.json` 和 `saved-manifest.json`。check模式还会生成官方比较日志和 `evaluation.json`。
