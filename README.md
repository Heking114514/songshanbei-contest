# MiniMax-H3 八卡阶段提交工作区

更新时间：2026-11-09T18:36:08+08:00（北京时间）。

统一首帧：`inputs/first_frame.png`；统一 FL2VA Prompt：**A dog falls asleep**。

## 一、操作流程

### 1. 源码与依赖下载

下载本仓库 **main 分支源码**，再从 [offline-wheel Release](https://github.com/Heking114514/songshanbei-contest/releases/tag/offline-wheel) 下载 **`h3-offline-wheels-linux-cp311.zip`**。仓库为私有仓库，GitHub 账号需要读取权限；也可以登录网页后手动下载。

仓库包含源码、构建工具、首帧图片和参考视频；ZIP 包含 55 个 Python 依赖。**源码要展开，依赖 ZIP 保留原名，不要解压。** 完整命令见下方“5. 总结”。

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

脚本自动解压依赖、安装环境、准备工具链、`-j120` 编译、完整八卡推理并保存结果，进度实时打印。无需手工切换环境或分步编译。

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

- 完成 H3 原生 DiT 八卡并行和音频 SnakeBeta 算子适配；条件编码、VAE 等组件继续使用 PyTorch/Diffusers。
- 完成源码与离线依赖交付，一条命令自动构建、生成视频并保存日志；同节点重试可复用环境和未变化的编译对象。
- 2026-11-09 实测从零到保存视频 **34分50秒**，全过程无人工修补。视频为 124帧、768×928、24fps，含音轨；下载后运行原版官方比较器，Size、Length 均 PASS，SSIM **96.579%**。

比较使用历史参考视频（math_bf16、参考端按4个独立 head 分批），没有同轮 PyTorch 加速比；SSIM 不评价音频质量。

### 2. 后续改进

- **数值一致性**：对照条件编码、注意力、音视频潜变量和解码输出，定位两端差异及不同运行间的 SSIM 波动，补充音频波形与听感检查。
- **算子性能**：针对注意力、矩阵乘和通信开展同输入、同环境对照，数值验证通过后再合入优化。
- **复现效率**：减少冷编译开销，完善跨节点缓存恢复和失败后增量重试，并补充不同首帧用例的完整验证。

环境目录、换图、可选 check 及依赖维护统一见 [技术补充](REPRODUCE.md)。
