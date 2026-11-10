# MiniMax-H3 八卡阶段提交工作区

更新时间：2026-11-10T13:00:42+08:00（北京时间）。

**版本状态：本次 main 文档提交为 final_modify，基于已完成干净环境八卡复验的 final1110 版本。按下方命令获取 main 源码即可复现。**

统一首帧：`inputs/first_frame.png`；统一 FL2VA Prompt：**A dog falls asleep**。

## 一、操作流程

### 1. 源码与依赖下载

获取方式是下载本仓库 **main 分支源码**，再从 [离线依赖 ZIP](https://github.com/Heking114514/songshanbei-contest/releases/download/offline-wheel/h3-offline-wheels-linux-cp311.zip) 下载 **`h3-offline-wheels-linux-cp311.zip`**。

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

脚本依次执行源码暂存、依赖解压与安装、工具链准备、`-j120` 编译、完整八卡推理和结果保存，进度实时打印。本次整改版已在清空个人 AFS、重新上传源码和原始 ZIP 后，用新八卡按本页原命令完成从零安装、冷编译、完整推理及 AFS 保存；正式入口启动后无需人工修补或重启。

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

以下为本次 `main` 版本的获取与运行流程。源码展开后保留完整目录结构，原始依赖 ZIP 放在仓库根目录。

```bash
# ==========================================================
# 1. 本地 Git Bash：获取 main 源码和离线依赖
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

1. **完成 MiniMax-H3 的原生八卡推理迁移。** 围绕比赛对完整模型推理的要求，将 DiT 主干接入 InfiniLM/InfiniCore 原生执行和八卡张量并行，串联条件编码、去噪、音视频解码与媒体输出，形成从统一图片和 Prompt 到完整音视频生成的端到端流程。条件编码、VAE 和调度保留原有 PyTorch/Diffusers 路径，模型主干通过原生算子与多卡通信执行，并已完成完整 FL2VA 用例验证。

2. **形成清晰、可检查的算子实现边界。** Attention 使用官方已有的显式 QKᵀ、缩放、Softmax 和 PV 流程，RMSNorm、MRoPE 使用官方实现；音频 SnakeBeta 通过 NineToothed Python JIT 生成计算内核，将指数、正弦和归一化计算纳入九齿编译链。对应实现配套 SnakeBeta、Attention 和流程回归检查，验证关键计算的数值结果与调用接口，为赛事代码审查、结果核对及后续性能优化提供可追溯的依据。

3. **交付可从源码启动的复现流程。** 将源码、固定依赖、构建工具、统一输入和参考资产整理为一套交付材料，使用现有离线依赖 ZIP，通过单条启动命令完成环境准备、-j120 编译、完整八卡推理和 AFS 结果保存。构建与推理信息实时输出，结果目录包含视频、参数、状态和日志；已完成干净环境下从源码准备到生成并保存音视频的完整复验，便于按同一流程复现、排查失败并核对产物。

### 2. 后续改进

- **数值一致性**：对照条件编码、注意力、音视频潜变量和解码输出，定位两端差异及不同运行间的 SSIM 波动，补充音频波形与听感检查。
- **算子性能**：针对显式 Attention、基础矩阵乘和通信开展同输入、同环境对照，数值验证通过后再更新性能结论。
- **交付与复现**：贯通正式封装的换例参数及记录/自检，补充不同图片和Prompt的连续完整验证；修复头文件mtime与重复prepare/build引起的LM重编，验证跨节点缓存恢复。

环境目录、换图、可选 check 及依赖维护统一见 [技术补充](REPRODUCE.md)。
