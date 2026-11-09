# MiniMax-H3 FL2VA 运行说明

更新时间：2026-11-09T11:12:27+08:00（北京时间）。

## 放文件

上传本仓库到 `个人AFS/songshanbei-contest/`，将 `h3-offline-wheels-linux-cp311.zip` 原名放到仓库根目录，与 `run_submission.sh` 同级，无需手工解压。四份工具、输入图和参考视频已随仓库提供。获取和核验依赖见 [PACKAGING.md](PACKAGING.md)。

## 一行启动

申请独占 8 卡，打开新 Bash，**在终端初始目录直接执行，无需先cd到AFS**：

~~~bash
bash "${IFAB_AFS_HOME:-$HOME}/songshanbei-contest/run_submission.sh"
~~~

需要本次完整生成后运行官方比较，改为：

~~~bash
bash "${IFAB_AFS_HOME:-$HOME}/songshanbei-contest/run_submission.sh" check
~~~

也可以从新终端初始目录先进入仓库，再运行：

~~~bash
cd "${IFAB_AFS_HOME:-$HOME}/songshanbei-contest"
bash run_submission.sh
~~~

在仓库内需要比较时执行 `bash run_submission.sh check`。脚本从当前或前一个目录识别本节点workspace，自动选择其同级工作目录；若先切换过多个目录，推荐重新打开Bash后使用一行启动命令。

默认流程为：核验离线输入 → 同步节点源码 → 自动解压wheel ZIP到节点盘 → 安装环境 → -j120编译 → 完整生成 → AFS保存。参数保持8卡、完整层数、seed42、50步请求、124帧、Prompt为 `A dog falls asleep`。check在生成完成后调用原版官方比较器，复用已有参考视频。

## 拿结果

每次自动保存到 `个人AFS/h3-results/run-时间戳-进程号/`，完整路径会显示在终端。视频为 `result.mp4`，日志和其他产物见 [README输出表](README.md#结果从哪里拿)。

退出0、末行pipeline_rc=0/save_rc=0，且存在AFS视频及saved-manifest.json才表示生成和保存成功。保存失败时，终端末行提供节点产物位置。

## 默认设置与可选覆盖

标准赛事环境直接使用上述命令即可。默认设置为：

| 设置 | 默认值 |
|---|---|
| 个人AFS根 | 平台IFAB_AFS_HOME，未设置时使用HOME |
| 源码及依赖 | 启动脚本所在目录及同级h3-offline-wheels-linux-cp311.zip；兼容offline-wheels/ |
| 节点环境 | 当前或前一个workspace的同级h3-native-编号/ |
| 共享模型 | /public/home/aclsylqidf/shared/MiniMax-H3 |
| 编译并行数 | 120 |
| 结果目录 | AFS/h3-results/run-时间戳-进程号/ |
| 单次保存上限 | 128MiB |

共享模型路径不同时，可在启动前 `export MODEL_PATH=/实际模型路径`。需要指定节点目录时，设置 `H3_NODE_ROOT` 为AFS和workspace之外的节点盘目录；需要固定输出位置时，设置 `H3_AFS_OUTPUT_DIR` 为AFS内尚不存在的新目录。日常运行无需设置这些变量。

平台需提供Python3.11、Torch2.9.0/HIP6.3、DTK/gfx936和共享权重。容量检查不作为启动前置步骤；结果按单次上限保存，个人AFS仍需有足够可用空间。

## 增量重试

相同节点、相同AFS源码目录自动选择同一节点目录，复用环境和对象；每次生成使用新输出目录。未变化的源码不重写，变化文件交给构建系统增量处理。节点源码存在独立修改时拒绝覆盖。

节点回收后环境和对象随之回收，新节点重新安装和构建；AFS源码、wheel及已保存结果继续保留。结构见 [ENVIRONMENT.md](ENVIRONMENT.md)。

## 更换测试图片

运行前指定新图片，文件需存在且能解码：

~~~bash
export FIRST_FRAME="${IFAB_AFS_HOME:-$HOME}/songshanbei-contest/inputs/new_frame.png"
~~~

无需计算或填写 SHA256。Prompt当前固定，其他参数如上。check还需更新 `reference/reference_fl2va.mp4` 和 `reference/case.json`，绑定新图片及新参考视频。登记字段见 [reference说明](reference/README.md)，不能用旧参考视频比较新图片。

运行过程仅执行存在、大小、归档可解压和媒体可解码检查，不执行SHA256校验。增量同步采用大小与mtime，未变化的源码和编译对象不重写。

推荐上传原名ZIP；不需要额外解压命令。已有offline-wheels/目录也可继续使用，两种格式选一种；同时存在且目录中有wheel时优先目录。ZIP和源码留在AFS，解压后的依赖及编译对象仅在节点盘。

准备阶段使用目录项元数据减少AFS路径查询；正式预检在真实推理进程启动时执行，检查失败会在加载模型前退出。新节点仍从源码冷构建，完整生成参数不变。
