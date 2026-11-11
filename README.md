# MiniMax-H3 八卡阶段提交工作区

统一首帧：`inputs/first_frame.png`；统一 FL2VA Prompt：**A dog falls asleep**。

## 一、操作流程

### 1. 源码与依赖下载

下载 **main 分支**的完整仓库。Python 离线依赖位于 `third_party/offline-wheels/`，共 55 个独立 `.whl` 文件；构建工具位于 `third_party/h3/`。

涉及修改的第三方项目及对应源码、补丁位置如下：

| 项目 | 上游仓库 | 本仓库中的修改位置（main 分支） |
|---|---|---|
| InfiniCore | [InfiniTensor/InfiniCore](https://github.com/InfiniTensor/InfiniCore) | [InfiniCore/](https://github.com/Heking114514/songshanbei-contest/tree/main/InfiniCore) |
| InfiniLM | [InfiniTensor/InfiniLM](https://github.com/InfiniTensor/InfiniLM) | [InfiniLM/](https://github.com/Heking114514/songshanbei-contest/tree/main/InfiniLM) |
| 九齿 NineToothed | [InfiniTensor/ninetoothed](https://github.com/InfiniTensor/ninetoothed) | [ninetoothed/](https://github.com/Heking114514/songshanbei-contest/tree/main/ninetoothed) |
| FlagTree（构建配置补丁） | [flagos-ai/FlagTree](https://github.com/flagos-ai/FlagTree) | [patch_flagtree_build.py](https://github.com/Heking114514/songshanbei-contest/blob/main/third_party/h3/patch_flagtree_build.py) |

InfiniCore、InfiniLM 和 NineToothed 的源码整合在本仓库中，分支为 `main`。

### 2. 文件上传

将完整的 `songshanbei-contest/` 文件夹上传到**个人 AFS 根目录**，保持以下目录结构：

```text
个人AFS/
└── songshanbei-contest/
    ├── run_submission.sh
    ├── InfiniCore/
    ├── InfiniLM/
    ├── ninetoothed/
    ├── scripts/
    ├── requirements/
    ├── third_party/
    │   ├── offline-wheels/      # 55 个独立 .whl 文件
    │   └── h3/                  # HCU 编译后端与构建工具
    ├── inputs/first_frame.png
    └── reference/
```

### 3. 脚本启动

申请**独占 8 卡**，打开新的 Bash 终端，在**初始目录直接执行，无需先 cd**：

```bash
bash "${IFAB_AFS_HOME:-$HOME}/songshanbei-contest/run_submission.sh"
```

运行环境需提供 Python 3.11、Torch 2.9.0/HIP 6.3、DTK/gfx936，以及共享模型 `/public/home/aclsylqidf/shared/MiniMax-H3`。Python 依赖和构建工具由脚本自动准备，无需手工安装或激活环境。

脚本自动将源码与依赖暂存到节点盘，完成离线安装、工具链准备、`-j120` 编译、完整八卡推理、视频检查和 AFS 保存，进度实时打印。启动前检查文件与依赖完整性，运行时不联网补包。

默认参数为完整模型、8 卡、seed42、50 步请求、124 帧。上述命令不加 `check`，生成视频后即可从 AFS 获取结果；官方比较在视频下载后于本地进行。

### 4. 产物获取

终端末行出现 **`pipeline_rc=0 save_rc=0`** 后，到 AFS 文件管理打开终端显示的结果目录：

```text
个人AFS/h3-results/run-时间戳-进程号/
```

| 文件 | 内容 |
|---|---|
| `result.mp4` | 生成视频 |
| `generation.log` | 推理日志 |
| `workflow.log` | 环境、编译和推理日志 |
| `invocation.json` | 输入及运行参数 |
| `native-check.json` | 视频规格检查 |
| `submission-status.json`、`saved-manifest.json` | 运行状态、保存清单 |

确认视频和保存清单已在 AFS 中，再**释放八卡资源**。节点临时目录会随资源回收，最终产物以 AFS 结果目录为准。

### 5. 总结

完整复现命令及结果获取方式如下：

```bash
# ==========================================================
# 1. 本地 Git Bash：获取 main 源码和离线依赖
# ==========================================================

git clone --branch main --single-branch \
  https://github.com/Heking114514/songshanbei-contest.git

# 将整个 songshanbei-contest 文件夹上传到个人 AFS 根目录。
# 保留全部源码和 third_party/ 下的独立依赖文件，无需另下压缩包。
#
# 上传后应当是：
# 个人AFS/songshanbei-contest/run_submission.sh
# 个人AFS/songshanbei-contest/InfiniCore/
# 个人AFS/songshanbei-contest/InfiniLM/
# 个人AFS/songshanbei-contest/ninetoothed/
# 个人AFS/songshanbei-contest/scripts/
# 个人AFS/songshanbei-contest/third_party/offline-wheels/  # 55 个 .whl
# 个人AFS/songshanbei-contest/third_party/h3/
# 个人AFS/songshanbei-contest/inputs/first_frame.png
# 个人AFS/songshanbei-contest/reference/
# 其余仓库文件也完整保留。

# ==========================================================
# 2. 平台申请独占 8 卡，打开新的 Bash 终端
# 以下命令在新终端初始目录执行，无需先 cd。
# ==========================================================

bash "${IFAB_AFS_HOME:-$HOME}/songshanbei-contest/run_submission.sh"

# 脚本自动完成：
# 源码暂存到节点盘 → 复制独立wheel → Python环境安装
# → 工具链准备 → -j120编译 → 完整八卡推理
# → 检查视频 → 保存视频和日志到AFS
#
# 默认首帧：inputs/first_frame.png
# Prompt：A dog falls asleep
# 参数：完整模型、8卡、seed42、50步请求、124帧
# 不加 check；官方比较在视频下载后于本地进行。

# ==========================================================
# 3. 运行结束后获取结果
# ==========================================================

# 终端末行出现 pipeline_rc=0 save_rc=0，表示运行和保存成功。
# 到 AFS 文件管理中打开：
# 个人AFS/h3-results/run-时间戳-进程号/
#
# 下载：
# result.mp4                生成视频
# generation.log            推理日志
# workflow.log              环境、编译和推理日志
# invocation.json           输入及运行参数
# native-check.json         视频规格检查
# submission-status.json    运行状态
# saved-manifest.json       保存清单
#
# 确认结果已保存后，释放八卡资源。
```

## 二、完成情况与后续改进

### 1. 完成了什么

1. **完整 FL2VA 八卡推理。** DiT 主干通过 InfiniLM/InfiniCore 原生算子与八卡张量并行执行，条件编码、VAE 和调度保留 PyTorch/Diffusers 路径，完成从首帧图片和 Prompt 到音视频输出的完整流程。

2. **算子与编译路径。** Attention 使用显式 QKᵀ、缩放、Softmax 和 PV 流程，RMSNorm、MRoPE 使用官方实现；音频 SnakeBeta 通过 NineToothed Python JIT 生成计算内核，包含指数、正弦与归一化计算。

3. **从零复现验证。** 已在空 AFS、新八卡环境完成一次无人工干预的环境准备、冷编译、完整推理和结果保存，入口总耗时约 37 分 03 秒。视频为 124 帧、768×928、24fps，含音频；本地使用未修改的官方比较器，对随附历史参考视频得到 Size/Length PASS、Average SSIM 95.826%。比较输出见 [official-comparison.log](results/fl2va/official-comparison.log)，对应视频见 [result.mp4](results/fl2va/result.mp4)。

### 2. 后续改进

- **数值一致性**：对照条件编码、注意力、音视频潜变量和解码输出，定位输出差异及 SSIM 波动，补充音频波形与听感检查。
- **算子性能**：优化显式 Attention、矩阵乘和多卡通信，在相同输入与环境下验证数值和端到端性能。
- **复现效率**：统一换图与 Prompt 参数、运行记录和媒体检查，补充多用例验证；减少头文件时间戳变化及重复准备引起的重编译，验证跨节点缓存恢复。

环境目录、换图、可选 check 及依赖维护统一见 [技术补充](REPRODUCE.md)。
