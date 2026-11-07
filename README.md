# MiniMax-H3 八卡阶段提交工作区

本仓库包含 InfiniLM、InfiniCore、NineToothed。完整H3原生TP8的124帧/50步FL2VA已成功生成；官方compare已执行，但参考进程因HIP检测不到GPU失败，报告SKIPPED，尚无SSIM结论。

**复现与官方提交材料见 [复现与提交说明.md](复现与提交说明.md)**，包括统一图片/提示词、-j120源码构建到运行、已编译资产恢复、真实compare输出及失败分析。组件来源见 `BASELINE.md` 与 `VERSIONS.md`。

目录包含三个组件：

```text
submit_repo/
├── InfiniLM/       # stage1：我方 songshan_minimax 基线 + stage1_work
├── InfiniCore/     # stage1：我方 pull_20261106 基线
├── ninetoothed/     # stage1：我方 NineToothed 快照基线
├── BASELINE.md
├── VERSIONS.md
└── README.md
```

## 平台与模型

- 使用平台单节点八卡实例；开始推理前需有有效租约，并通过实际 HIP/Torch 逐卡计算预检。
- 平台操作统一走 CLI：`/d/Desktop/songshan workspace/ifab/ifab.py`。
- `ifab_frontend` 仅作浏览器端参考，不作为自动化入口。
- 已有 token 时先用 `python ifab.py --json status`；只有 token 失效才重新登录。
- 已核实正式权重位于共享目录 `/public/home/aclsylqidf/shared/MiniMax-H3`，包括 `FL2VA` 和 `diffusers`，无需复制到个人 AFS。
- `$IFAB_AFS_HOME/handover` 保存构建文件、环境、输入和日志。个人配额为 4 GiB，应保留足够的编译空间。

```bash
cd "/d/Desktop/songshan workspace/ifab"
python ifab.py --json status

# 以下两行在平台 Bash 内执行。
export MODEL_PATH=/public/home/aclsylqidf/shared/MiniMax-H3
ls "$MODEL_PATH/FL2VA" "$MODEL_PATH/diffusers"
```

## 先读

1. `VERSIONS.md`：整合仓库、三个组件的来源和当前验证状态。
2. `../八卡阶段代码边界与改造审计.md`：赛事边界、P0、工单和验收。
3. `InfiniLM/stage1_work/README.md`：单卡阶段证据和限制。
4. `../reference/`：官方 InfiniLM、InfiniCore 参考仓库；只做 diff，不作为提交基线。

## Git Bash 约定

所有命令在 **Git Bash** 中执行。禁止 PowerShell/pwsh 执行或包装：

```bash
cd "/d/Desktop/songshan workspace/submit_repo"
pwd
git status --short --branch
```

## 单卡组件来源

| 组件 | 历史基线分支 | 内容 |
|---|---|---|
| InfiniLM | `stage1` | 我方 `songshan_minimax` 基线 + `stage1_work/` 证据 |
| InfiniCore | `stage1` | 我方 `pull_20261106` 基线 + SnakeBeta/接口/后端增量 |
| NineToothed | `stage1` | 我方快照基线；官方 commit 待确认 |

## 推送前

三个组件属于同一个整合 Git 仓库，远程为 `https://github.com/Heking114514/songshanbei-contest`，当前工作分支为 `codex/eight-card-handover`。本轮提交在该分支保存，不在历史 `submit/` 的三个仓库分别提交。

提交源码、测试、运行入口、统一首帧与复现说明；排除权重、编译产物、Python/pytest 缓存和凭证。`BASELINE.md` 保留原单卡来源，`VERSIONS.md` 记录本轮验证边界；当前提交号通过 `git rev-parse HEAD` 查询。

## 不可修改

- 模型权重、测试数据、隐藏用例；
- `InfiniLM/examples/minimax_h3_compare.py`；
- 评测程序和基线；
- 不得用预计算结果或外部在线服务替代推理。

## 当前验证状态与下一步

- 已完成：H3原生构建；1194对象跨租约零重编、单文件增量验证；单卡接口修复；运行包换目录零JIT恢复；八卡22帧冒烟；完整模型TP8正式124帧/50步请求生成与媒体回收校验。
- 正式配置：统一图片、A dog falls asleep、seed42、默认分辨率768×928、24fps、math_bf16。原生正式视频证据见 `../handover/results/h3-formal-20261107T0956/`。
- 未完成：legacy完整参考生成、官方SSIM/差异比较、全模型参考数值对齐。已跑compare返回2，参考进程在导入设备层失败，没有开始加载模型。
- 下一次使用健康八卡会话，先直接HIP/Torch/逐卡计算预检，再用未修改官方compare补跑参考。平台Active不能替代GPU实测。
- 当前资产直接恢复即可，不需要重编或pip。首次源码构建入口 `scripts/build_and_compare_h3.sh` 使用-j120；该新wrapper只经过本地检查，尚未新租约端到端验收。
- `scripts/run_h3_compare.sh`统一BF16 math配置和预检。resume可通过官方provided参数复用native视频，但该侧本轮性能为n/a；正式性能用full重新生成双方。
- `scripts/run_h3_smoke.sh`已改为22帧/256×256/2步，仅短冒烟启用时长override，正式compare不启用。
- 所有命令仅Git Bash执行、完整脚本整体base64。AFS用量只查专门API，保留4GiB上限与容量余量。

代码边界见 `../八卡阶段代码边界与改造审计.md`。实时进展见 `../handover/接手进展.md`。
