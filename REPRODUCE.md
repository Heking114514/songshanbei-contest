# H3 官方复现命令与验收范围

更新时间：2026-11-08T12:40:52+08:00（北京时间）。

这是仓库内的完整命令入口，覆盖两条工作流：单卡用于后续算法更新后的增量编译与双端推理自验；八卡用于从环境准备到双方全新生成的官方FL2VA提交。另保留八卡短推理用于快速确认链路。允许使用按SHA校验的预下载依赖；失败在同一checkout/work增量续跑，不clean。本提交包含所需入口、固定依赖资产与运行边界，发布整理没有新增服务器任务。当前命令对应已知模板用例；官方更换图片/Prompt或脚本时，先按下方“官方更换测试用例”处理已知适配缺口，不能将当前封装视作任意新用例已验收。

本轮新环境与原生构建已在八卡节点完成；1157个旧对象保持SHA不变，只补37个对象，合计1194。总入口已累计两次prepare/build/eight-smoke全部rc=0，八卡完整层数短推理两次均保存视频，复跑1194对象SHA全部不变。第三次构建及八卡预检通过，权重加载后缺最终退出码；租约deadline后平台鉴权401，不计为完整成功。首次构建跨两轮租约，随后总入口复跑复用1194对象；并非一次不间断冷启动耗时。历史94.277%为旧环境resume compare，不替代本轮验收；[结构化执行摘要](results/h3_reproduction_20261108.json)保留实测配置及脚本指纹。

## 工作流选择

| 目的 | 完整顺序 | 结果用途 |
|---|---|---|
| 单卡开发自验 | prepare→build→jit→smoke（双端首次运行及禁JIT复跑） | 算法/接口更新后自验，使用缩层夹具 |
| 八卡官方提交 | prepare→build→来源审计→八卡Snake验证→run_h3_compare.sh | 完整模型、官方参数、双方本轮新生成和原版FL2VA结果 |
| 八卡快速链路验证 | reproduce_h3_all.sh：prepare→build→eight-smoke | 完整层数、低分辨率少步数；非正式质量/性能成绩 |

所有本地命令只通过指定Git Bash，禁止PowerShell/pwsh、cmd、WSL和裸bash.exe别名。下面是节点Bash载荷；H3_SOURCE_CHECKOUT须设为本轮实际节点源码路径。不要在本地Windows直接执行Linux节点载荷。

## 八卡快速链路验证（非提交）

将完整仓库、third_party/h3固定资产及55个锁定wheel准备在节点本地。平台Torch、共享官方权重为前提。所有本地命令只由指定Git Bash派发；下面是在节点执行的Bash载荷：

```bash
set -euo pipefail
cd "$H3_SOURCE_CHECKOUT"
export H3_REPRO_WORK="$PWD/.h3-reproduction"
export H3_BASE_PYTHON=/usr/local/bin/python3 BUILD_JOBS=120
export MODEL_PATH=/public/home/aclsylqidf/shared/MiniMax-H3
# 可选：预先直接传到节点的55个wheel目录；每份大小/SHA必须匹配锁。
if [ -n "${H3_NODE_WHEELHOUSE:-}" ]; then
    export H3_WHEELHOUSE="$H3_NODE_WHEELHOUSE"
fi
bash --noprofile --norc scripts/reproduce_h3_all.sh
```

不设置H3_WHEELHOUSE时，prepare从锁定URL下载；在赛事节点实测很慢，推荐先直传依赖。总入口依次prepare、build、eight-smoke；每阶段实时tee，30秒心跳，保留真实退出码。新目录首次构建；同目录再次执行复用正确对象与依赖。不得更换work路径后期待绝对路径缓存自然命中，跨节点缓存须另行校验/迁移。

八卡冒烟保持完整模型层数、统一图片和Prompt，22帧、2步请求（本轮日志实际1次DiT去噪）、256×256、seed42，允许首次JIT。低分辨率及极少去噪步数会影响细节和成片质量；完整层数不能等同完整正式推理参数。它证明推理链路，不能当正式124帧/50步/768×928成绩。完整视频产生后生成reports/eight-smoke-result.json；本轮完整短视频已生成并验收；用户要求的正常进入推理目标已满足。

## 固定源码与输入

交付根目录包含 `InfiniCore/`、`InfiniLM/`、`ninetoothed/`、`scripts/`、`requirements/`、`third_party/h3/`、`inputs/`，不需要仓库外 handover 脚本。此前公开基线是`834488108a0af0636590e2349240ac336163ecc5`，该旧commit不含本次新增入口与依赖。复现时选择包含本文、scripts/requirements/third_party资产的完整提交；运行时记录git rev-parse HEAD，不能以旧基线代替。

源码与依赖提交cb6812d15cb2ececfe7b91130ff7491b4b6805c1已通过SSH推送并核对远程ref；匿名网页/API/raw当前返回404，公开可访问性未确认。以下匿名clone流程需仓库实际公开可读后再执行。用包含所需内容的实际完整40位SHA设置H3_COMMIT，clone到新节点目录，并以detached HEAD固定版本。匿名clone实机整链仍待验收；以下命令不代表已完成该项：

```bash
set -euo pipefail
: "${H3_COMMIT:?设置含全部修复与依赖的最终公开40位commit}"
: "${H3_NODE_ROOT:?设置平台提供的节点工作目录，不能位于个人AFS}"
cd "$H3_NODE_ROOT"
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0
git clone --no-checkout https://github.com/Heking114514/songshanbei-contest.git h3-official-repro
cd h3-official-repro
git checkout --detach "$H3_COMMIT"
test "$(git rev-parse HEAD)" = "$H3_COMMIT"
test -z "$(git status --porcelain)"
```

当前模板用例的统一图片：`inputs/first_frame.png`，原始字节 SHA-256：
`b97c3487ca35da0c803fba3787cf2b7f936d3d4595548ae5b7b576c19934be54`。

当前模板用例的FL2VA Prompt：`A dog falls asleep`。该例使用seed42、124帧、50步请求，已验收输出768×928、24fps。官方换例时输入和参数以当轮规则为准；768×928不作为所有图片的固定断言。不传缩小分辨率/步数的参数，不启用短时长 override，不提供已有视频来代替本次正式生成。

## 平台和依赖

目标是赛事海光 Linux x86_64：CPython 3.11、平台 Torch 2.9.0 / HIP 6.3、gfx936、DTK `/opt/dtk`。Torch 必须来自平台 `/usr/local`，不能被普通 PyPI Torch 覆盖。基础 Python 默认 `/usr/local/bin/python3`；同环境下可显式设置 `H3_BASE_PYTHON`。平台需有 pip、venv、Git、make、C/C++ 构建工具，且允许从锁定 URL 下载 Python wheel。

`requirements/h3_linux_cp311.lock.json` 锁定 55 个 wheel 的版本、URL、大小、SHA 和依赖元数据；对应 txt 便于审阅。共下载 192118057 字节，闭包检查通过；**本轮已在新Python目录安装并通过八卡短推理**；依赖随本提交提供，匿名clone实机整链仍待验收。venv 使用平台 site-packages 保留 Torch，锁定依赖安装在独立 `work/python`；入口重建 PATH/PYTHONPATH/动态库路径，并审计实际导入来源。

随附第三方资产共 149724964 字节：HCU FlagTree wheel、clang/lld JIT 工具链、xmake 3.0.3 源码安装包。其来源、SHA、许可与构建补丁见 [第三方说明](third_party/h3/README.md) 和 manifest。它们属于编译依赖；H3 原生库与 JIT 内核将在新目录首次生成。这里不恢复个人 venv、旧 H3 二进制或旧 JIT。

权重保持共享只读，默认 `/public/home/aclsylqidf/shared/MiniMax-H3`，须同时包含 `FL2VA` 和 `diffusers`；评审可通过 `MODEL_PATH` 指定等价的官方完整权重目录。不能复制权重到个人 AFS。节点工作区是允许写入的路径；系统/tmp可能不可写，Git需禁读受限/etc/gitconfig。节点磁盘需容纳源码、wheel、依赖、原生对象、完整输出及单卡缩层夹具；55 个 wheel 的下载大小不能当成解压后峰值预算。

## 单卡开发自验：编译到双端推理结束

用于后续算法/接口更新后的自验，使用独立开发副本及其现有对象缓存。首次建立环境；后续相同源码/work路径增量。完整命令：

```bash
set -euo pipefail
: "${H3_SOURCE_CHECKOUT:?设置本轮节点开发源码根}"
cd "$H3_SOURCE_CHECKOUT"
export H3_REPRO_WORK="$PWD/.h3-reproduction"
export H3_BASE_PYTHON=/usr/local/bin/python3 BUILD_JOBS=120
export MODEL_PATH=/public/home/aclsylqidf/shared/MiniMax-H3
export FIRST_FRAME="$PWD/inputs/first_frame.png"
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0
export H3_EXPECTED_GPUS=1
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0
unset GPU_DEVICE_ORDINAL H3_SHORT_SMOKE H3_INFINILM_VIDEO H3_REFERENCE_VIDEO
unset H3_SINGLE_WORK H3_SMOKE_RUN OUTPUT_DIR

for phase in prepare build jit smoke; do
    bash --noprofile --norc scripts/reproduce_h3.sh "$phase"
done
```

prepare安装锁定依赖与工具；build以-j120构建所需Core/LM目标链并审计来源；jit先允许首次SnakeBeta JIT、再起新进程禁新JIT验证；smoke创建本地缩层夹具，依次运行reference/native和原版比较函数，然后禁止新JIT完整复跑双端。阶段实时tee、30秒心跳、保留退出码。

单卡使用1层DiT和选定的1层Qwen隐藏态，reference夹具含2层Qwen来对齐；vision/VAE保留完整。输出22帧256×256、2步请求、seed42。它验证流程和局部数值回归，不能证明完整层数质量、长序列显存、TP8通信或正式加速比。原共享权重只读，派生夹具仅在节点single-smoke目录。

完成标准是本轮prepare/build/jit/smoke全部rc=0，smoke-result.json、cached-smoke-result.json和reports/single-ready.json对应本次日志，双端视频实际可解码、Size/Length及SSIM真实执行。没有统一SSIM阈值可代替算法误差审核；改优化后还须覆盖真实shape/stride/dtype专项用例。当前新bootstrap环境单卡jit/smoke组合仍未单独实机验收，不能把旧单卡成功或本轮八卡成功挪作此分支的运行回执。

C++/头文件更新后的常用增量命令，在相同环境变量下运行：

```bash
unset H3_SINGLE_WORK H3_SMOKE_RUN OUTPUT_DIR
for phase in build jit smoke; do
    bash --noprofile --norc scripts/reproduce_h3.sh "$phase"
done
```

只改Python/helper且不改原生接口时，运行受影响的jit/smoke；不为无关文件重编。smoke默认新建时间戳输出目录，build/work不变，失败旧日志不覆盖，旧single-ready不当本次通过。涉及多卡切分/通信的优化最后仍需八卡完整验证。

## 八卡官方提交：环境到完整FL2VA比较

八卡直接流程允许全部阶段在八卡完成，不要求先制造单卡回执。保持完整层数、统一图/Prompt、seed42、124帧、50步请求、默认768×928/24fps，双方本轮全新生成。reference历史子进程wall为3015.783秒（约50.3分钟），native另一轮加载至保存为542.131秒（约9分钟）；两者计时口径不同、来自不同轮次，不作官方加速比。规划窗口时完整双方计算另留预检/保存，宜约90分钟，首次环境/编译时间另加，以当前池规则为准。1小时租约不能保证从零到完整双方比较结束，不降低参数凑时长。

```bash
set -euo pipefail
: "${H3_SOURCE_CHECKOUT:?设置本轮节点完整源码根}"
cd "$H3_SOURCE_CHECKOUT"
export H3_SOURCE_ROOT="$PWD" H3_REPRO_WORK="$PWD/.h3-reproduction"
export H3_BASE_PYTHON=/usr/local/bin/python3 BUILD_JOBS=120
export MODEL_PATH=/public/home/aclsylqidf/shared/MiniMax-H3
export FIRST_FRAME="$PWD/inputs/first_frame.png"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export H3_EXPECTED_GPUS=8
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0
unset GPU_DEVICE_ORDINAL H3_SHORT_SMOKE H3_INFINILM_VIDEO H3_REFERENCE_VIDEO
unset H3_SINGLE_WORK H3_SMOKE_RUN OUTPUT_DIR

for phase in prepare build; do
    bash --noprofile --norc scripts/reproduce_h3.sh "$phase"
done

source scripts/h3_repro_env.sh
python -u scripts/h3_bootstrap.py audit --work "$H3_REPRO_WORK" \
    --output "$H3_REPRO_WORK/reports/official-origins.json"
# 全八卡直接从新环境运行，首次允许JIT，记录实际编译开销。
export H3_FORBID_JIT_COMPILATION=0
python -u scripts/validate_snake_8card.py --gpus 0,1,2,3,4,5,6,7 \
    --output "$H3_REPRO_WORK/reports/official-snake-eight.json"

run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
export OUTPUT_DIR="$H3_REPRO_WORK/official-fl2va-$run_id"
run_formal() (
    while sleep 30; do
        printf '[%s] H3_OFFICIAL_HEARTBEAT output=%s\n' "$(date -u +%FT%TZ)" "$OUTPUT_DIR"
    done &
    heartbeat_pid=$!
    trap 'kill "$heartbeat_pid" 2>/dev/null || true' EXIT
    bash --noprofile --norc scripts/run_h3_compare.sh
)
run_formal 2>&1 | tee "$H3_REPRO_WORK/logs/official-entry-$run_id.log"
printf 'H3_OFFICIAL_COMPLETE output=%s\n' "$OUTPUT_DIR"
```

wrapper先做GPU/包/媒体/输入和官方compare SHA预检，再调用原版InfiniLM/examples/minimax_h3_compare.py，仅FL2VA；native GPU0..7，reference text GPU0..3、transformer GPU4..7，memory-fraction0.4。原版compare固定正式Prompt/seed/步数/帧数；没有provided视频，不使用短时长或缩层override。

首次允许JIT，实际编译开销保留日志。双方math_bf16、reference每批4个独立attention head的调度需在报告中披露。若测预热后性能，应另记预热和计时口径，不删日志冒充纯计算时间。

这条流程使用现有run_h3_compare.sh。reproduce_h3.sh的preflight/compare分支仍要求真实single-ready.json，属于先单卡后八卡路径，不是本页八卡直接路径；不能伪造回执。reproduce_h3_all.sh仍以eight-smoke结束，不能当正式提交入口。

结束核对OUTPUT_DIR/status.json的rc=0，以及compare.log的真实FL2VA Size/Length/SSIM与timing表；保留reference_fl2va.mp4、infinilm_fl2va.mp4、两个生成器日志、preflight.json、来源报告、构建日志、源码/依赖/参数及每份文件大小/SHA。双方都必须本轮success，不能以provided代替再声称同轮加速比。正式报告复制本轮compare原文，并核对解码124帧/768×928及音频流；音频流存在不等于音频一致性验收。

原版compare的LF归一SHA为24f3439e11e98d979ec8cee5f76284079dad87cb4668cf0f109b7904e8641bc3。模板没有SSIM合格阈值，SSIM不是相同像素比例。

## 增量、搬迁与回收

编译失败后保留源码、`build/`、`.xmake/`、安装目录和 work；修复后在**相同路径、相同 work**重新运行 build。脚本无 clean；配置 key 不变且实际 xmake.conf 仍在时跳过 configure，普通 C++ 改动交给 xmake 依赖判断。ABI、工具/构建配置、Lua 或安装路径变化会刷新配置。环境锁变化应换新验收目录，不覆盖旧稳定缓存。

prepare 的已验证下载可复用；中断的 Python staging 会留作诊断并重新安装，成功环境按 marker 复用。smoke 重试默认创建新的时间戳测试目录，不动原生对象。仅改报告或 helper 不需要原生重编；改动后应重新做对应流程检查。

跨实例迁移前在节点打包源码 checkout、新扩展、安装、venv/python、toolchain/xmake、JIT/ninetoothed 和报告；保留 .git 以供身份审计。把包下载到本地或另有足够配额的持久存储，下一节点解包。本轮已跨节点恢复源码/1157原生对象并核验SHA不变，只补37对象，随后新环境推理通过。Python环境/固定工具链按锁重建；不能将此称为整个venv或新增JIT的跨节点零编译恢复已通过。换目录须迁移路径元数据、核验对象字节与mtime_ns并重新审计，禁止直接重编全部对象掩盖恢复问题。正式compare门禁仍另行验收。

不能默认把约数 GiB 的新环境再写进接近满额的个人 AFS。AFS 上限 4 GiB，容量只查专用 API，不扫盘；任何写入前需新鲜完整统计与峰值预算。本轮已按预算把源码/原生缓存及首轮视频/日志保存为AFS base+delta，并下载校验；没有把整个Python环境写入AFS。额外第二次成功证据的小包AFS上传未commit，仅本地副本已核验。

## 官方更换测试用例

正式交付需要支持官方当轮的图片/Prompt和生成规则。每例先登记ID、任务、输入路径/SHA、Prompt原文、seed、帧数、步数、尺寸/预处理要求和官方脚本版本；两端使用同一份登记数据。模型实现与算法不得对模板图片、特定Prompt或旧输出做特例处理。

当前有以下适配缺口，本文只记录流程，尚未修改代码：

| 位置 | 已核实的限制 | 后续换例要求 |
|---|---|---|
| scripts/h3_preflight.py的INPUT_SHA | 固定旧模板图，新FIRST_FRAME会被拒绝 | 从当轮用例登记绑定期望SHA，保留解码/硬件/来源检查 |
| 官方compare的FL2VA_PROMPT | 固定A dog falls asleep，没有Prompt命令行参数 | 优先使用官方新脚本/配置；按官方允许方式传Prompt，双方一致，不擅改比较算法 |
| 官方compare常量 | seed42、50步、124帧固定 | 官方只换输入时保持；如新规范改参数，统一跟随新规范 |
| 预检COMPARE_SHA | 绑定现版官方脚本 | 官方更新脚本时核实来源与新SHA，不能简单关闭校验 |
| 输出尺寸检查 | 当前模板图为768×928 | 按新用例的有效尺寸与官方预处理核对，双方相同且符合规范 |

底层InfiniLM/examples/minimax_h3.py支持--first-frame、--text、--seed、--steps、--num-frames及成对的--height/--width。底层支持参数不等于外层compare/预检已经完成通用换例适配，当前不能仅换图片路径就宣称整链可用。

换例后的执行顺序：环境/原生库准备一次 → 登记本例输入和官方参数 → 核实形状、token长度、TP切分与显存 → 必要的首次JIT → 官方compare分别新生成双方完整视频 → 按本例规范比较并保存结果 → 新目录执行下一例。每例独立保存实际参数、双方视频、日志、SSIM、耗时、退出码和大小/SHA。

只换图片/Prompt/seed通常无需C++重编；新尺寸、文本长度或帧数可能触发新JIT，不能假定旧例缓存覆盖所有shape。编译缓存和正确匹配的内核缓存可复用，旧例的文本嵌入、latent或视频不能冒充新例输出。每例是否首次JIT/是否预热要记录，性能比较采用一致口径。超出支持范围须明确报错并按规则修复，不能静默改小分辨率、缩层或少跑步数。

仍只按本轮要求做FL2VA；若官方改变任务，按新规范调整任务入口并单独验收。当前新环境完整双方compare和换例/多例适配均未验收，旧例94.277%不能代替新例成绩。

## 已有结果与发布门槛

[已有官方原文](results/h3_official_compare_20261107.txt) 为完整模型：Size PASS、Length PASS、平均 SSIM 94.277%。该轮新生成 reference、复用已核验 native 视频，native 计时 n/a；无同轮加速比。SSIM 不等于相同像素比例，也不代表音频一致性。

当前用户要求的八卡环境准备、原生构建、正常进入推理已完成，且两次完整短推理保存视频；首次原生构建跨两轮租约，后续总入口复用本轮1194对象，不能宣称一个不中断的冷启动调用已经实测。本提交提供当前修复/依赖及其真实验收边界，正式交付仍需：固定实际发布SHA并匿名clone实机验收；如走single-ready门禁则补单卡分支；足时八卡双方全新生成及原版compare；保存同轮参数/日志/视频SHA并填写正式报告。不得把本轮短推理当正式compare成绩。
