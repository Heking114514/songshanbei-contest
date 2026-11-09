# songshanbei-contest

更新时间：2026-11-09T15:24:04+08:00（北京时间）。

## 嵩山杯 MiniMax-H3 FL2VA 初赛

统一首帧：`inputs/first_frame.png`

统一 FL2VA Prompt：`A dog falls asleep`

### 复现：上传源码和 ZIP，一行启动

1. 将本仓库源码上传到个人 AFS 根目录，目录命名为 **`songshanbei-contest/`**。目录内应直接有 `run_submission.sh`、`InfiniCore/`、`InfiniLM/`，源码需是展开的文件夹。
2. 从[离线依赖 Release](https://github.com/Heking114514/songshanbei-contest/releases/tag/offline-wheel)下载 **`h3-offline-wheels-linux-cp311.zip`**，放到上述目录，**与 `run_submission.sh` 同级，保持原名，无需手工解压**。依赖获取说明见 [PACKAGING.md](PACKAGING.md)；仓库本身不包含这个 ZIP。
3. 申请独占 8 卡，打开新 Bash，**在终端初始目录直接粘贴**：

~~~bash
bash "${IFAB_AFS_HOME:-$HOME}/songshanbei-contest/run_submission.sh"
~~~

**启动后无需人工解压、安装、切换环境或分步编译。** 脚本自动完成：检查文件完整性 → 把源码复制到节点盘 → 把依赖 ZIP 解压到节点盘 → 安装 Python 依赖与解包工具链 → `-j120` 编译 → 完整 8 卡推理 → 把视频和日志保存到个人 AFS。进度实时打印，错误会停止后续步骤并输出原因。无需计算或填写 SHA256。

### 文件必须这样放

~~~text
个人AFS/
└── songshanbei-contest/
    ├── run_submission.sh
    ├── h3-offline-wheels-linux-cp311.zip   # 另行下载，原名放这里，不解压
    ├── InfiniCore/
    ├── InfiniLM/
    ├── ninetoothed/
    ├── scripts/
    ├── requirements/
    ├── third_party/h3/                    # 仓库自带4份工具
    ├── inputs/first_frame.png             # 仓库自带首帧
    └── reference/                        # 仓库自带参考视频和case.json
~~~

ZIP 内为 `offline-wheels/` 目录及 55 个 wheel，脚本直接解压到**节点**的 `work/downloads/`，不会在 AFS 展开一份依赖。也兼容已解压的 `songshanbei-contest/offline-wheels/*.whl`；两种方式选一种即可，同时存在且目录中有wheel时优先使用该目录；只有README的目录不影响ZIP识别。请勿把 ZIP 放在 `offline-wheels/` 内。

### 从下载到运行的完整命令

下面分为两个执行位置：先在本地 Git Bash 下载，再上传文件；申请八卡后，在新的 Bash 终端运行。私有仓库和 Release 需要账号具备读取权限；也可在浏览器登录后下载源码和 ZIP。

~~~bash
# 1. 本地 Git Bash：获取 main 源码。
git clone --branch main --single-branch \
  https://github.com/Heking114514/songshanbei-contest.git
cd songshanbei-contest

# 下载离线依赖；也可以从 Release 页面手动下载。
curl -fL --retry 3 -C - \
  -o h3-offline-wheels-linux-cp311.zip \
  "https://github.com/Heking114514/songshanbei-contest/releases/download/offline-wheel/h3-offline-wheels-linux-cp311.zip"

# 2. 将整个 songshanbei-contest 文件夹上传到个人 AFS 根目录。
# 源码保持展开，依赖 ZIP 保持原名、不解压。
# 上传后 run_submission.sh 和 ZIP 必须同级：
# 个人AFS/songshanbei-contest/run_submission.sh
# 个人AFS/songshanbei-contest/h3-offline-wheels-linux-cp311.zip
# 其余源码目录也全部保留。

# 3. 申请独占 8 卡，打开新的 Bash。
# 下面这一行在新终端初始目录执行，不要先 cd。
bash "${IFAB_AFS_HOME:-$HOME}/songshanbei-contest/run_submission.sh"

# 脚本自动完成源码暂存、ZIP解压、环境准备、-j120编译、
# 完整八卡推理、视频检查与AFS保存；无需手工切换环境。
# 首帧：inputs/first_frame.png
# Prompt：A dog falls asleep
# 完整模型、8卡、seed42、50步请求、124帧。

# 4. 末行 pipeline_rc=0 save_rc=0 表示运行和保存成功。
# 在 AFS 文件管理下载以下目录中的 result.mp4：
# 个人AFS/h3-results/run-时间戳-进程号/
# generation.log 是推理日志，workflow.log 是全流程日志。
# invocation.json、native-check.json 记录参数与视频检查；
# submission-status.json、saved-manifest.json 记录状态与保存清单。
# 确认结果保存后，释放八卡资源。
~~~

默认不加 check。本轮验收将视频下载后在本地运行官方比较器；若需要在八卡流程内比较，使用下节的 check 命令。两种方式无需同时执行。

### 需要官方比较时

把启动命令加上 `check`：

~~~bash
bash "${IFAB_AFS_HOME:-$HOME}/songshanbei-contest/run_submission.sh" check
~~~

`check` 会完成本次完整推理，再与 `reference/reference_fl2va.mp4` 调用原版官方比较器。参考视频直接复用；默认不加 `check` 时只生成和保存视频。两条命令选一条运行。

### 结果从哪里拿

每次自动保存到 **`个人AFS/h3-results/run-时间戳-进程号/`**，终端显示完整路径。到 AFS 文件管理中打开该目录下载：

| 内容 | 本次结果目录内的文件 |
|---|---|
| 生成的视频 | `result.mp4` |
| 推理日志 | `generation.log` |
| 环境、编译和推理总日志 | `workflow.log` |
| 输入与完整生成参数 | `invocation.json` |
| 视频规格与文件记录 | `native-check.json` |
| 运行状态、保存清单 | `submission-status.json`、`saved-manifest.json` |
| 官方比较原文，仅 check | `compare/compare.log` |
| 比较检查结果，仅 check | `evaluation.json` |

退出 0、末行 `pipeline_rc=0 save_rc=0`，且 AFS 中存在 `result.mp4` 和 `saved-manifest.json`，表示生成与保存成功。终端中的节点临时目录不是最终取视频的位置。

### 环境和增量复用

平台需提供 Python 3.11、Torch 2.9.0/HIP 6.3、DTK/gfx936 和共享模型。默认模型目录为 `/public/home/aclsylqidf/shared/MiniMax-H3`，源码、工具、首帧与参考视频由仓库提供，55 个普通 wheel 由 ZIP 提供。

脚本在节点初始 `workspace` 的同级自动创建 `h3-native-编号/`：`source/` 保存源码和编译对象，`work/` 保存环境和解压依赖。同节点重试复用环境及未变化的编译对象；节点回收后，下次新节点重新构建。AFS 内的源码、ZIP和已保存视频保留。依赖只解压到节点，不额外占用AFS展开空间；AFS仍需有保存本次结果的空间。

标准参数为完整层数、8卡、seed42、50步请求、124帧。更换首帧及模型路径等可选设置见 [REPRODUCE.md](REPRODUCE.md)，完整目录见 [ENVIRONMENT.md](ENVIRONMENT.md)。
