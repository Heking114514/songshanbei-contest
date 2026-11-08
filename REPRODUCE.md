# MiniMax-H3 FL2VA 正式复现

更新时间：2026-11-08T22:48:17+08:00（北京时间）。

正式入口只有根目录 run_submission.sh。先上传本交付目录到个人 AFS，再申请独占八卡；进入八卡 Bash 后构建环境、-j120 构建 H3 及所需 Core/绑定，然后完整推理。默认只生成视频并保存到指定 AFS 新目录；加 check 会在同一次完整生成之后，使用随包已有参考视频调用原版官方 compare，再独立检查结果。两种模式都不冒烟、不缩层、不减少帧/步、不先跑 Snake 或矩阵预热。

本轮已从清空后的AFS和新八卡完成环境、-j120源码构建、完整native生成与AFS保存，pipeline_rc/save_rc均0；首次/tmp权限错误通过下文显式H3_NODE_ROOT修正，不能称默认路径首次无误。随后在本地用本轮native与既有reference调用原版官方比较器，Size/Length PASS、SSIM96.819%；服务器check模式尚未实测，没有双方同轮生成加速比。最终源码发布SHA、匿名获取及正式Word报告待收尾；现有制包快照与后续命令说明改动须在最终发布时重新绑定。

## 上传内容与目录

使用打包工具生成的新目录整体上传，保留名称，不在 AFS 解压 payload：

~~~text
个人AFS/
├── h3-submission/
│   ├── run_submission.sh
│   ├── REPRODUCE.md
│   ├── UPLOAD.md
│   ├── delivery-manifest.json
│   ├── scripts/
│   │   ├── h3_delivery.py
│   │   └── h3_source_identity.py
│   ├── reference/
│   │   ├── reference_fl2va.mp4
│   │   ├── case.json
│   │   └── README.md
│   └── payload.part000、payload.part001……（每片最多96 MiB）
├── storage-info.json（上传后、八卡就绪时取得的容量API完整统计）
└── h3-results/本次唯一目录/（运行完成自动创建）
~~~

payload 内含 source/ 和 wheelhouse/。source 包含 InfiniCore、InfiniLM、ninetoothed、scripts、requirements、third_party/h3、inputs、许可证和源码身份清单；不含历史 AFS、个人凭证、旧虚拟环境、原生对象或 JIT 缓存。reference 独立存放，不在 payload 重复收录。源码导出保留真实基准 Git SHA、当时脏工作树状态和逐文件 SHA，不 git init 伪造提交。

| 资产 | 大小/来源 | 用途 |
|---|---|---|
| 锁定55个 Linux cp311 wheel | 192,118,057字节；requirements/h3_linux_cp311.lock.json | 离线 Python 包，逐项大小/SHA校验 |
| FlagTree HCU wheel | 72,231,003字节 | triton HCU 后端 |
| clang18/lld 最小工具链 | 74,463,838字节 | 真实推理首次 JIT |
| xmake 3.0.3 | 3,030,123字节 | 原生构建 |
| reference_fl2va.mp4 | 1,554,000字节 | 现有模板用例的完整参考视频 |
| 统一图片 | inputs/first_frame.png | 默认输入，SHA登记 |
| 模型权重 | 平台共享 MiniMax-H3 | 只读，不上传个人AFS |
| Python/Torch/DTK/驱动 | 平台镜像 | CPython3.11、Torch2.9.0/HIP6.3、DTK/gfx936 |

离线依赖合计341,843,021字节，另加源码、参考视频和清单。上传总字节和每个分片 SHA 以生成的 delivery-manifest.json / UPLOAD.md 为准。包内无需下载 PyPI 依赖，不替换平台 Torch。新设备指符合上述平台镜像的全新节点，不是缺少驱动和共享权重的任意机器。

本地制作命令，须在指定 Git Bash 中执行；路径替换为已核验的55-wheel目录与新的输出目录：

~~~bash
python scripts/h3_delivery.py package   --repo .   --wheelhouse /path/to/verified/wheelhouse   --output /path/to/new/h3-submission
~~~

Windows Git Bash 下原生 Python 的参数用 C:/ 或 D:/ 绝对路径。不要用 MSYS python3。Release 尚未发布；后续可将 payload 分片与 manifest 挂在固定 Release，源码保留 Git，并把确切下载地址填入提交材料。完整 AFS 灾备快照不作为正式依赖。

## 上传与容量统计

先用平台个人 AFS 上传本交付目录，包括 reference。单片96 MiB适应当前平台128 MiB单文件上传限制；不要为了500 MB分片触发上传拒绝。上传前先查容量并核对上传总量、结果预留及32 MiB余量；容量只看 ifab afs info / GET /api/persistent-storage，不扫盘。

上传结束并且八卡资源就绪后，取得新鲜、complete=true、over_quota=false 的 API JSON，放到 AFS 的 storage-info.json。支持原始 storage 对象，也支持 {"storage":...} 外层。用现有本地平台客户端可导出：

~~~bash
python ifab/ifab.py afs-status --json > storage-info.json
~~~

这是工作区已有客户端的本地命令，不是提交工程内的脚本；官方也可从平台等价接口取得 JSON。只上传统计 JSON，不上传登录凭证、token 或客户端配置。排队超过15分钟，应在资源就绪后刷新统计。pending_uploads 与回收站按 API 语义纳入预算，不能把后台扫描中的0用量当空盘。

脚本在构建前按最新统计预留默认128 MiB输出和32 MiB余量，硬上限4 GiB。AFS只读分片，合并/解压/构建都在节点盘；输出超过预留上限会停止保存并保留节点现场。其他任务同时写 AFS 时必须协调预留，时点统计不能保证其他写入受控。不要自动清空旧 AFS 或旧结果。

## 八卡执行命令

申请独占8卡后，在其 Bash 内运行以下载荷。本地派发只能使用指定 Git Bash，按交接手册整体base64；禁止 PowerShell、cmd、WSL包装。

~~~bash
set -euo pipefail
export IFAB_AFS_HOME=/path/to/your/personal/afs
# 在新八卡Bash初始workspace中、cd到AFS之前执行；保留同一路径供增量重试。
export H3_NODE_ROOT="$(dirname "$PWD")/h3-submission-native"
cd "$IFAB_AFS_HOME/h3-submission"
export MODEL_PATH=/path/to/shared/MiniMax-H3
export H3_BASE_PYTHON=/usr/local/bin/python3
export BUILD_JOBS=120
export H3_AFS_INFO_JSON="$IFAB_AFS_HOME/storage-info.json"
export H3_AFS_OUTPUT_DIR="$IFAB_AFS_HOME/h3-results/native-$(date -u +%Y%m%dT%H%M%SZ)"
unset H3_SHORT_SMOKE H3_INFINILM_VIDEO H3_REFERENCE_VIDEO
bash run_submission.sh
~~~

默认完整生成并保存视频，不比较。需要完整生成后比较时，在同样配置下使用：

~~~bash
export H3_AFS_OUTPUT_DIR="$IFAB_AFS_HOME/h3-results/check-$(date -u +%Y%m%dT%H%M%SZ)"
bash run_submission.sh check
~~~

这两条是二选一的完整运行方式。check 并非对上一次 native 只评测：它仍构建/复用环境并新生成 native，然后直接与现有 reference 比较。官方 compare 同时接收本轮 native 与既有 reference 的 provided 参数，零重复生成。不存在 run_submisson_compare.sh 第二正式入口；旧 scripts/run_h3_submission.sh 草案入口已停用。

本平台必须按上文显式设置 H3_NODE_ROOT，使用刚打开Bash的sid私有根下、workspace同级的目录；含 source、wheelhouse、work 和 runs。首次运行选择 /tmp 被平台权限拒绝；本提交已移除该默认值，未设置 H3_NODE_ROOT 会在启动时明确报错，stage也要求 --node。此修正通过本地回归，尚未另做新节点无干预复验。不要将大构建树放workspace或AFS。同一交付包、同一路径重试复用已验 SHA 的源码、55-wheel环境、xmake配置、对象和 JIT；不clean。换包或环境锁不兼容时用新目录。新设备没有旧对象，首次原生构建不可省略；默认 -j120 构建 H3 必需目标及其 Core/SDK/绑定依赖，不构建无关模型。

所有环境/编译/推理信息实时 tee 到终端与 workflow.log，每30秒心跳。首用 JIT 在真实推理中发生，计入本次实际时间。HIP只查八卡可见性，预检不运行计算预热。构建导入审计核对包、so和实际加载库来源，不能从旧私有环境借库。prepare、build或生成失败后立即停止后续计算，保存预算内日志；pipeline_rc 和 save_rc 分开，保存失败也返回非零。

## 参数、换例和参考视频

当前官方FL2VA参数：Prompt "A dog falls asleep"、seed42、50步请求、124帧、完整层数、默认预处理。日志49次DiT是调度行为。模板图输出768×928、24fps。双方运行设置为 math_bf16；随包 reference 历史采用每批4个独立head，原版比较公式不改。

reference/case.json 绑定模板图SHA、Prompt、seed、步数、帧数、原版compare脚本SHA及参考视频SHA。check 在构建前核对，换图但没换对应reference会拒绝。图片名称不是身份，不能把别人的 sample_first_frame.jpg 与本PNG视为同一字节。

默认生成可通过 FIRST_FRAME 和 H3_INPUT_SHA256 登记官方新图片。新用例要 check 时，必须准备对应的参考视频、更新 case.json 并重新制包，使上传清单同步。参考视频可以只生成一次后复用，但其来源必须与本用例一致。当前官方比较器固定 Prompt/seed/步数/帧数；若官方改Prompt或规则，需要按官方版本适配入口、校验和登记，不擅自关闭验证，也不将旧reference用于新例。当前新例尚未实测。

## 保存位置与验收

H3_AFS_OUTPUT_DIR 必须是个人 AFS 内不存在的新目录，且位于交付目录之外。保存逐文件核 SHA，临时文件通过后才改名，最后写 saved-manifest.json；不覆盖已有结果。视频始终在 H3_AFS_OUTPUT_DIR/result.mp4。

| 结果 | 默认模式 | check模式 |
|---|---|---|
| result.mp4、generation.log | 有 | 有 |
| invocation.json、origins.json、preflight.json | 有 | 有 |
| native-check.json、submission-status.json | 有 | 有 |
| workflow.log、afs-reservation.json、saved-manifest.json | 有 | 有 |
| compare/compare.log（官方原文）及预检/状态 | 无 | 有 |
| evaluation.json（独立check） | 无 | 有 |
| 新reference视频 | 不生成 | 不生成，复用AFS reference目录 |

验收要求：脚本退出0；最后一行 pipeline_rc=0、save_rc=0；AFS内 submission-status.json 为 complete、saved-manifest.json 存在且 pipeline_rc=0。native需可完整解码为124帧、24fps、有效32倍数尺寸和一条音频流；check另要求官方Size/Length PASS、SSIM数字完整、双方实际规格一致，输入及视频SHA未变化。音频流存在不代表音质一致；官方未给SSIM阈值，不自行发明。

已有reference通过provided路径复用，官方性能表可能显示n/a，不报告双方同轮加速比。历史94.277%只属于历史运行。本轮冷启动完整native与AFS保存成功，本地原版compare为96.819%；原始失败、参数、视频和比较证据见 [本轮记录](results/h3_cold_native_20261108/README.md)。首次节点路径由人工修正，日志观测也有人工处理，因此不能称首次全程无干预成功；服务器check工作流未实测。

## 自用开发与历史边界

自用仍是 scripts/run_h3_development.sh，与正式入口分开。在节点源码目录配置同一 H3_REPRO_WORK 增量更新：

~~~bash
export H3_REPRO_WORK=/node/local/h3-dev-work
export H3_WHEELHOUSE=/node/local/wheelhouse
export MODEL_PATH=/path/to/shared/MiniMax-H3
export BUILD_JOBS=120 H3_DEV_GPUS=1
bash scripts/run_h3_development.sh all
~~~

默认prepare→build→Snake JIT→单卡缩层双端smoke→禁新JIT复跑；仅开发自验。H3_DEV_GPUS=8则prepare→build→eight-smoke。正式run_submission不调用这些阶段。单卡夹具使用缩层、22帧256²和短步数，不能当正式质量成绩。旧 reproduce_h3.sh preflight/compare 保留真实single-ready门禁，只用于历史开发分支。

历史短推理与94.277%比较保留各自证据；本轮独立完成空AFS、新八卡冷构建和完整native、本地96.819%比较。下一步是从固定新commit重新制包，复验显式节点路径下的无干预运行、服务器check与官方换例，确认匿名获取并填写正式报告。原生对象缓存已另存AFS，完整Python/JIT恢复和本轮缓存跨节点复用未验收。
