# MiniMax-H3 FL2VA：放置依赖、运行与获取结果

更新时间：2026-11-08T23:15:02+08:00（北京时间）。

操作顺序：准备离线依赖并制作交付目录 → 上传个人AFS → 申请独占八卡 → 运行脚本 → 从AFS下载视频和日志。

## 1. 放置离线依赖并制作交付目录

如果已经拿到制作好的 `h3-submission/`，直接从第2步开始。以下准备工作在源码仓库根目录执行，需要 Python 3.11 或更新版本。

55个 Python wheel 放在仓库同级的 `h3-wheelhouse/` 中，直接放文件，不再套一层目录；文件名、大小和 SHA256 必须符合 `requirements/h3_linux_cp311.lock.json`。三份构建工具已经随源码放在 `third_party/h3/`。

~~~text
工作目录/
├── songshanbei-contest/             # 源码仓库
│   ├── run_submission.sh
│   ├── scripts/
│   ├── requirements/h3_linux_cp311.lock.json
│   ├── third_party/h3/
│   │   ├── flagtree-0.7.0+hcu3.6-cp311-cp311-linux_x86_64.whl
│   │   ├── hcu-jit-clang18-minimal.tar.gz
│   │   └── xmake-v3.0.3.gz.run
│   ├── inputs/first_frame.png
│   └── reference/
│       ├── reference_fl2va.mp4
│       ├── case.json
│       └── README.md
├── h3-wheelhouse/                   # 55个锁定wheel，约183MiB
└── h3-submission/                   # 下一条打包命令生成
~~~

已有55个wheel时直接放入 `../h3-wheelhouse/`。需要下载时，下面的命令按依赖锁中的地址下载并核验；再次运行会复用已通过校验的文件：

~~~bash
python - <<'WHEELS'
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from h3_bootstrap import fetch

wheelhouse = Path("../h3-wheelhouse")
wheelhouse.mkdir(parents=True, exist_ok=True)
lock = json.loads(Path("requirements/h3_linux_cp311.lock.json").read_text())
for wheel in lock["wheels"]:
    fetch(wheel, wheelhouse)
print("离线wheel准备完成：", wheelhouse.resolve())
WHEELS
~~~

在同一仓库根目录制作交付包：

~~~bash
python scripts/h3_delivery.py package \
  --repo . \
  --wheelhouse ../h3-wheelhouse \
  --output ../h3-submission
~~~

`../h3-submission` 必须是尚不存在的新目录。命令会校验依赖，合并源码和离线wheel，生成最多96MiB一片的 `payload.part*`，以及上传清单 `UPLOAD.md` 和校验清单 `delivery-manifest.json`。

生成的交付目录已包含运行所需的源码、55个wheel和三份构建工具；上传时不必另传 `h3-wheelhouse/`。模型权重使用平台共享目录，不放进交付包。

## 2. 上传到 AFS

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

## 3. 在八卡 Bash 中启动

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

## 4. 需要官方比较时，加 check

默认模式只生成并保存视频。需要生成后比较时，完成上述环境变量配置，将最后的运行命令改为以下两行；两种模式选择一种即可：

~~~bash
export H3_AFS_OUTPUT_DIR="$IFAB_AFS_HOME/h3-results/check-$(date -u +%Y%m%dT%H%M%SZ)"
bash run_submission.sh check
~~~

`check` 仍会生成一份新的完整native视频，然后使用AFS交付目录中 `reference/reference_fl2va.mp4` 运行原版官方比较，并检查结果。参考视频只需随包上传一次，后续运行直接复用。

参考视频必须与当前图片、Prompt、seed和生成参数匹配。换输入图时，可用 `FIRST_FRAME` 指定图片，并用 `H3_INPUT_SHA256` 登记其SHA256；如需 `check`，还必须准备该用例对应的参考视频、更新 `reference/case.json` 并重新制包。当前入口的Prompt固定为 `A dog falls asleep`。

## 5. 从 AFS 取回产物

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
