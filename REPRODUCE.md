# MiniMax-H3 技术补充

更新时间：2026-11-09T18:36:08+08:00（北京时间）。

下载、上传、启动与取视频统一见 [README](README.md)。本文只保留环境、可选设置及维护细节，合并原 ENVIRONMENT、PACKAGING、VERSIONS、BASELINE 和“复现与提交说明”。

## 环境在哪里

AFS 保留源码、原始依赖 ZIP 和最终结果。脚本在新终端初始 `workspace` 的同级创建 `h3-native-编号/`：

| 节点目录 | 内容 |
|---|---|
| `source/` | 从 AFS 同步的源码及编译对象 |
| `work/venv/`、`work/python/` | Python 入口与已安装依赖 |
| `work/downloads/` | 从 ZIP 解压的 55 个 wheel |
| `work/install/` | 本轮构建的原生库 |
| `work/toolchain/`、`work/xmake/`、`work/jit/` | 编译工具及 JIT 缓存 |
| `runs/native-时间戳-进程号/` | 本次节点产物，完成后保存到 AFS |

平台提供 Linux x86_64、Python 3.11、Torch 2.9.0/HIP 6.3、DTK/gfx936 和共享权重；脚本不安装其他 Torch。无需手工激活环境。新节点需要重新安装和冷编译，同节点、同 AFS 源码目录重试可复用环境和对象；每次使用新的结果目录。

文件完整性采用存在性、大小、mtime、ZIP CRC 和媒体解码检查，不执行 SHA256 校验。来源审计核对模块和原生库路径。源码没有变化时保持 mtime，节点源码存在独立修改时拒绝覆盖；大小及 mtime 均未变的内容替换不在此检查范围内。

## 可选设置与换图

标准用例无需设置变量。模型路径不同或更换首帧时，可在新终端启动前指定：

```bash
export MODEL_PATH=/实际模型目录
export FIRST_FRAME="${IFAB_AFS_HOME:-$HOME}/songshanbei-contest/inputs/new_frame.png"
bash "${IFAB_AFS_HOME:-$HOME}/songshanbei-contest/run_submission.sh"
```

新图片必须存在且能解码。统一首帧 `inputs/first_frame.png` 原样提取自官方报告模板的 `word/media/image1.png`，800×965、1,393,139字节，未缩放或重编码；输出视频尺寸由 pipeline 预处理决定。

Prompt 当前固定为 `A dog falls asleep`，seed42、50步请求、124帧。更换 Prompt 或其他官方参数需要适配并重新验证，不能仅凭换图入口承诺通过。

高级变量：`H3_NODE_ROOT` 可指定 AFS/workspace 之外的节点盘目录；`H3_AFS_OUTPUT_DIR` 可指定 AFS 内尚不存在的新结果目录。默认每次保存上限 128MiB，AFS 需有足够空间。

## 可选服务器 check

默认流程生成视频后保存，下载到本地再做官方比较。若确需在服务器生成后比较，使用：

```bash
bash "${IFAB_AFS_HOME:-$HOME}/songshanbei-contest/run_submission.sh" check
```

此命令会**重新生成 native 视频**，然后与已有 `reference/reference_fl2va.mp4` 调用未修改的官方比较器，不生成 reference。比较原文为结果目录的 `compare/compare.log`，检查结果为 `evaluation.json`。近期冷验收使用默认入口及本地比较，未重新实测服务器 check。

参考视频仅适用于 `reference/case.json` 对应的图片和参数。换图后做比较，必须同步更换参考视频及登记信息，包括 `first_frame_bytes`、`reference_bytes`、Prompt、seed、requested_steps、num_frames，详见 [参考资产说明](reference/README.md)。不能拿旧参考视频比较新用例。

## 依赖与发布维护

- 55 个普通 wheel 由 [offline-wheel Release](https://github.com/Heking114514/songshanbei-contest/releases/tag/offline-wheel) 分发，ZIP 为 189,769,761字节，解压后共 192,118,057字节。原名放仓库根目录，不在 AFS 解压。
- 已有散文件也可放 `offline-wheels/`；目录内有 wheel 时优先使用，必须齐全。两种方式选一种，ZIP 不放进该目录。
- 文件名、版本、下载地址和大小见 [Python依赖锁](requirements/h3_linux_cp311.lock.json)。锁中的历史 SHA 字段不参与运行校验。
- 4 份工具随源码位于 `third_party/h3/`，包括 HCU FlagTree、clang/lld、xmake；来源、补丁与许可见 [工具说明](third_party/h3/README.md) 和 [工具清单](third_party/h3/manifest.json)。
- 发布 Release 只需上传 `h3-offline-wheels-linux-cp311.zip`，不必上传 SHA256SUMS。ZIP 内保持 `offline-wheels/文件.whl`，发布前确认成员齐全、大小正确、可完整解压。

维护者需要重新下载散文件时，在**本地源码根目录**用 Python 3.11 或更新版本运行；该步骤只下载，不安装 Linux wheel：

```bash
python - <<'WHEELS'
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from h3_bootstrap import fetch

wheelhouse = Path("offline-wheels")
wheelhouse.mkdir(exist_ok=True)
lock = json.loads(Path("requirements/h3_linux_cp311.lock.json").read_text())
for wheel in lock["wheels"]:
    fetch(wheel, wheelhouse)
print("离线依赖已下载到", wheelhouse.resolve())
WHEELS
```

## 实现与源码来源

H3 DiT 使用 InfiniLM/InfiniCore 原生 TP8，注意力采用 math_bf16；音频 SnakeBeta 使用 NineToothed。条件编码、VAE、调度和媒体处理保留 PyTorch/Diffusers。参考端为历史完整 legacy 输出，按4个独立 head 分批；复用参考视频的比较不产生同轮双方加速比。

| 组件 | 整合时来源 commit |
|---|---|
| InfiniLM | `79350958fe06f8e7cd7684ff0773ea06d3449a85` |
| InfiniCore | `5a674e269717adc578c0d599846c9818e6d741fa` |
| NineToothed 快照 | `6ce8568f8d208ba55afe4a5508bdc9896062b756` |

这些是导入来源，不代表当前运行版本；当前源码版本以 `git rev-parse HEAD` 为准。NineToothed 快照对应的官方上游 commit 未独立核验。

日常算法自验入口为 [run_h3_development.sh](scripts/run_h3_development.sh)，与正式入口分开。历史运行证据保留于 `results/`，各次记录只代表当时版本，不作为当前操作指令。最新实测摘要见 README。
