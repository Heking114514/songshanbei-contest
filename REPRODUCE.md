# MiniMax-H3 环境与运行说明

下载、上传、启动与产物获取见 [README](README.md)。

## 运行环境与目录

平台环境为 Linux x86_64、Python 3.11、PyTorch 2.9.0/HIP 6.3、DTK/gfx936，并提供共享 MiniMax-H3 权重。脚本使用平台 PyTorch，自动安装仓库内的 Python 依赖和构建工具，无需手工激活环境。

AFS 保存上传的仓库与最终结果。环境、原生库和编译对象位于节点盘；节点目录在初始 `workspace` 同级自动创建：

| 节点目录 | 内容 |
|---|---|
| `source/` | 源码与编译对象 |
| `work/venv/`、`work/python/` | Python 入口与依赖 |
| `work/downloads/` | 从仓库复制的 55 个 wheel |
| `work/install/` | 本轮构建的原生库 |
| `work/toolchain/`、`work/xmake/`、`work/jit/` | 编译工具与 JIT 缓存 |
| `runs/native-时间戳-进程号/` | 本次节点产物，完成后保存到 AFS |

同节点、同源码目录重试可复用环境与未变化的编译对象；新节点执行首次安装与构建。每次运行使用新的结果目录。

## 模型与首帧设置

默认模型目录为 `/public/home/aclsylqidf/shared/MiniMax-H3`。标准用例使用 `inputs/first_frame.png`，Prompt 为 `A dog falls asleep`，seed42、50 步请求、124 帧。

模型路径不同或需要更换首帧时，在新八卡 Bash 终端执行：

```bash
export MODEL_PATH=/实际模型目录
export FIRST_FRAME="${IFAB_AFS_HOME:-$HOME}/songshanbei-contest/inputs/new_frame.png"
bash "${IFAB_AFS_HOME:-$HOME}/songshanbei-contest/run_submission.sh"
```

新图片必须存在且能解码。正式入口的 Prompt、seed、步数和帧数使用上述固定值；底层 `InfiniLM/examples/minimax_h3.py fl2va` 提供相应 CLI 参数。

`H3_NODE_ROOT` 可指定 AFS 和 workspace 之外的节点目录；`H3_AFS_OUTPUT_DIR` 可指定 AFS 内尚不存在的新结果目录。默认每次保存上限 128 MiB。

## 官方视频比较

默认入口完成生成并保存视频。若需生成后自动调用官方比较器，使用：

```bash
bash "${IFAB_AFS_HOME:-$HOME}/songshanbei-contest/run_submission.sh" check
```

该命令生成新的 native 视频，再与 `reference/reference_fl2va.mp4` 比较，不重新生成参考视频。输出位于本次 AFS 结果目录的 `compare/compare.log` 和 `evaluation.json`。

已有两份视频时，可在具备比较依赖的本地环境中，进入**仓库根目录**后只运行比较：

```bash
python InfiniLM/examples/minimax_h3_compare.py \
  --tasks fl2va \
  --model-path . \
  --first-frame inputs/first_frame.png \
  --infinilm-fl2va-video results/fl2va/result.mp4 \
  --reference-fl2va-video reference/reference_fl2va.mp4 \
  --output-dir local-compare
```

参考视频与 `reference/case.json` 登记的输入、Prompt 及生成参数对应；更换测试用例时须使用对应参考视频，见 [参考资产说明](reference/README.md)。

## 本次运行结果

已在空 AFS、新八卡环境完成一次环境准备、冷编译、完整推理和 AFS 保存，运行中无需人工修补或重启。入口总耗时约 37 分 03 秒，其中源码暂存约 71 秒、环境准备 28 秒、编译及来源检查 22 分 41 秒。模型 CLI 加载至输出 713.249 秒，生成 pipeline 462.360 秒，DiT 的 49 次调用共 411.476 秒；这些时间采用不同统计边界。

视频为 124 帧、768×928、24fps，含 AAC 双声道 32000Hz 音频。与随附参考视频的官方比较结果为 Size PASS、Length PASS、Average SSIM 95.826%。[比较日志](results/fl2va/official-comparison.log)、[评测数据](results/fl2va/evaluation.json)与[生成视频](results/fl2va/result.mp4)位于同一目录。

## 依赖与实现

- Python 依赖位于 `third_party/offline-wheels/`，文件名、版本和大小见 [依赖清单](requirements/h3_linux_cp311.lock.json)。运行时只从本地文件安装，缺失文件直接报错。
- HCU FlagTree、clang/lld 和 xmake 位于 `third_party/h3/`，用途、来源与许可见 [构建依赖说明](third_party/h3/README.md)。
- DiT 使用 InfiniLM/InfiniCore 原生八卡张量并行，Attention 为显式 QKᵀ、scale、Softmax、PV；RMSNorm、MRoPE 使用官方实现。
- 音频 SnakeBeta 通过 NineToothed Python JIT 生成内核，由 HCU FlagTree 编译。条件编码、VAE 和调度保留 PyTorch/Diffusers 路径；native 进程的 PyTorch SDPA 使用数学 BF16 模式。
- 开发验证入口为 [run_h3_development.sh](scripts/run_h3_development.sh)。
