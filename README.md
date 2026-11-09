# MiniMax-H3 八卡阶段提交工作区

更新时间：2026-11-10T03:18:45+08:00（北京时间）。

**版本状态：整改计算代码已保存为本地提交 `4f1ebbc`，尚未推送远端。使用本轮版本时应取本地完整源码；下方远端 `main` 下载命令仅获取已发布版本，不包含本地尚未推送的整改。**

统一首帧：`inputs/first_frame.png`；统一 FL2VA Prompt：**A dog falls asleep**。

## 一、操作流程

### 1. 源码与依赖下载

远端发布后的常规获取方式是下载本仓库 **main 分支源码**，再从 [离线依赖 ZIP](https://github.com/Heking114514/songshanbei-contest/releases/download/offline-wheel/h3-offline-wheels-linux-cp311.zip) 下载 **`h3-offline-wheels-linux-cp311.zip`**。仓库为私有仓库，GitHub 账号需要读取权限；也可以登录网页后手动下载。

本轮仍使用现有离线依赖，无需更换 ZIP 或增加 wheel。仓库包含源码、构建工具、首帧图片和参考视频；ZIP 包含 55 个 Python 依赖。**源码要展开，依赖 ZIP 保留原名，不要解压。** 完整命令见下方“5. 总结”。

### 2. 文件上传

将整个 `songshanbei-contest/` 文件夹上传到**个人 AFS 根目录**，保留所有仓库文件。ZIP 与 `run_submission.sh` 必须同级：

```text
个人AFS/
└── songshanbei-contest/
    ├── run_submission.sh
    ├── h3-offline-wheels-linux-cp311.zip
    ├── InfiniCore/
    ├── InfiniLM/
    ├── ninetoothed/
    ├── scripts/
    ├── requirements/
    ├── third_party/
    ├── inputs/first_frame.png
    └── reference/
```

### 3. 脚本启动

申请**独占 8 卡**，打开新的 Bash 终端，在**初始目录直接执行，无需先 cd**：

```bash
bash "${IFAB_AFS_HOME:-$HOME}/songshanbei-contest/run_submission.sh"
```

脚本依次执行源码暂存、依赖解压与安装、工具链准备、`-j120` 编译、完整八卡推理和结果保存，进度实时打印。本次整改版已通过增量续跑，按本页完整流程从干净环境一键启动的复验仍待完成。

平台需提供 Python 3.11、Torch 2.9.0/HIP 6.3、DTK/gfx936 和共享模型 `/public/home/aclsylqidf/shared/MiniMax-H3`。默认使用完整模型、seed42、50步请求、124帧；**不加 check，官方比较在视频下载后于本地进行。**

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

以下保留常规远端获取与运行流程。当前整改尚未推送；使用本地版本时，以本地完整源码目录替代第一步 clone 的源码，并保留相同目录结构和依赖 ZIP。

```bash
# ==========================================================
# 1. 本地 Git Bash：获取 main 源码和离线依赖
# 仓库为私有仓库，需要当前 GitHub 账号有读取权限。
# ==========================================================

git clone --branch main --single-branch \
  https://github.com/Heking114514/songshanbei-contest.git

cd songshanbei-contest

# 也可以从 Release 页面手动下载此文件。
curl -fL --retry 3 -C - \
  -o h3-offline-wheels-linux-cp311.zip \
  "https://github.com/Heking114514/songshanbei-contest/releases/download/offline-wheel/h3-offline-wheels-linux-cp311.zip"

# 将整个 songshanbei-contest 文件夹上传到个人 AFS 根目录。
# 源码保持展开状态，依赖 ZIP 保持原名，不要解压。
#
# 上传后应当是：
# 个人AFS/songshanbei-contest/run_submission.sh
# 个人AFS/songshanbei-contest/h3-offline-wheels-linux-cp311.zip
# 个人AFS/songshanbei-contest/InfiniCore/
# 个人AFS/songshanbei-contest/InfiniLM/
# 个人AFS/songshanbei-contest/scripts/
# 个人AFS/songshanbei-contest/third_party/
# 个人AFS/songshanbei-contest/inputs/first_frame.png
# 个人AFS/songshanbei-contest/reference/
# 其余仓库文件也完整保留。

# ==========================================================
# 2. 平台申请独占 8 卡，打开新的 Bash 终端
# 以下命令在新终端初始目录执行，无需先 cd。
# ==========================================================

bash "${IFAB_AFS_HOME:-$HOME}/songshanbei-contest/run_submission.sh"

# 脚本自动完成：
# 源码暂存到节点盘 → ZIP解压 → Python环境安装
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

- H3 DiT 使用 InfiniLM/InfiniCore 原生 TP8，Attention 计算沿用官方已有的显式 QKᵀ/scale/softmax/PV；RMSNorm、MRoPE 恢复官方实现，实验 ATen conv/groupnorm/Snake 后端排除在正式构建之外。
- 音频 SnakeBeta 使用 NineToothed Python JIT，exp(alpha)、exp(beta)、sin 和归一化均在生成的九齿 kernel 内完成。native 条件编码保留数学 BF16 SDPA；条件编码、VAE 和调度继续使用原有 PyTorch/Diffusers 路径。
- 本地69项流程回归、SnakeBeta三组FP32和MHA7头五种序列长度断言通过；MHA最大绝对误差为0，SnakeBeta最大绝对误差约5.25e-6。
- 2026-11-10完成124帧、50步请求（49次DiT）、seed42的完整八卡FL2VA，推理及AFS保存均退出0。该次运行使用提交前的整改工作树，计算代码现已保存为本地提交 `4f1ebbc`。

当前结论为“主要算子边界已整改、单个完整用例跑通”；多用例连续验证、干净环境一键复验和交付材料仍待收口，不宣称已完全通过赛事合规验收。底层官方CLI支持输入参数，正式封装仍固定Prompt、seed、步数与帧数，换例限制见[技术补充](REPRODUCE.md)。

### 2. 后续改进

- **数值一致性**：对照条件编码、注意力、音视频潜变量和解码输出，定位两端差异及不同运行间的 SSIM 波动，补充音频波形与听感检查。
- **算子性能**：针对显式 Attention、基础矩阵乘和通信开展同输入、同环境对照，数值验证通过后再更新性能结论。
- **交付与复现**：贯通正式封装的换例参数及记录/自检，补充不同图片和Prompt的连续完整验证、干净环境一键复验；修复头文件mtime与重复prepare/build引起的LM重编，验证跨节点缓存恢复。

环境目录、换图、可选 check 及依赖维护统一见 [技术补充](REPRODUCE.md)。
