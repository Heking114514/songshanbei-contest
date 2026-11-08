# songshanbei-contest

更新时间：2026-11-09T01:19:29+08:00（北京时间）。

## 嵩山杯 MiniMax-H3 FL2VA 初赛

统一输入图片：`inputs/first_frame.png`

统一 FL2VA Prompt：`A dog falls asleep`

### 复现指令

1. **上传源码。** 将本仓库源码上传到个人 AFS 根目录，目录统一命名为 `songshanbei-contest/`；该目录内直接包含 `run_submission.sh`、`InfiniCore/` 和 `InfiniLM/`。
2. **放入依赖。** 将 **55 个 wheel 文件**原名放入仓库内的 `offline-wheels/`。三份工具、输入图和参考视频已随仓库提供。获取方法见 [离线依赖说明](PACKAGING.md)；当前尚未发布配套 Release。
3. **申请独占 8 卡。** 打开新 Bash，**就在终端初始目录直接粘贴下面一行，无需先进入 AFS**：

~~~bash
bash "${IFAB_AFS_HOME:-$HOME}/songshanbei-contest/run_submission.sh"
~~~

脚本自动选择节点工作目录，安装环境、以 `-j120` 编译 H3 所需目标，完整生成视频并保存到 AFS。进度实时打印，启动时显示环境和结果目录。

需要本次生成后运行官方比较，改用：

~~~bash
bash "${IFAB_AFS_HOME:-$HOME}/songshanbei-contest/run_submission.sh" check
~~~

两种方式选一种。`check` 使用本次新生成视频，与随仓库提供的 `reference/reference_fl2va.mp4` 比较，参考视频直接复用。

**如果习惯先进入源码目录，也可以从新终端初始目录执行：**

~~~bash
cd "${IFAB_AFS_HOME:-$HOME}/songshanbei-contest"
bash run_submission.sh
~~~

需要比较时，把最后一行改为 `bash run_submission.sh check`。

### 文件怎么放

~~~text
个人AFS/
└── songshanbei-contest/
    ├── run_submission.sh
    ├── InfiniCore/
    ├── InfiniLM/
    ├── ninetoothed/
    ├── scripts/
    ├── requirements/
    ├── third_party/h3/             # 仓库已带三份工具
    ├── offline-wheels/             # 放入55个wheel，约183MiB
    ├── inputs/first_frame.png
    └── reference/                  # 仓库已带参考视频及用例登记
~~~

### 结果从哪里拿

每次运行自动保存到 **`个人AFS/h3-results/run-时间戳-进程号/`**。启动和结束时都会打印完整路径，到平台 AFS 文件管理中打开该目录下载。

| 内容 | 本次结果目录内的路径 |
|---|---|
| 生成的视频 | `result.mp4` |
| 推理日志 | `generation.log` |
| 环境、编译及推理总日志 | `workflow.log` |
| 输入与生成参数 | `invocation.json` |
| 视频规格与指纹 | `native-check.json` |
| 运行状态与保存清单 | `submission-status.json`、`saved-manifest.json` |
| 官方比较原文，仅check模式 | `compare/compare.log` |
| 比较检查结果，仅check模式 | `evaluation.json` |

退出 0，末行 `pipeline_rc=0`、`save_rc=0`，且 AFS 中存在视频和保存清单，表示生成及保存成功。

### 环境在哪里

源码和 wheel 保留在 AFS；脚本在当前节点初始 `workspace` 的同级目录自动创建 `h3-native-编号/`。其中 `source/` 保存源码和编译对象，`work/` 保存依赖、工具链和原生库。相同节点、相同 AFS 源码目录重复运行会复用环境和增量编译产物。

以上命令适用于赛事海光 8 卡环境：平台提供 Python 3.11、Torch 2.9.0/HIP 6.3、DTK/gfx936，默认共享模型为 `/public/home/aclsylqidf/shared/MiniMax-H3`。平台环境不同时的可选配置及换图方法见 [REPRODUCE.md](REPRODUCE.md)，详细目录见 [ENVIRONMENT.md](ENVIRONMENT.md)。
