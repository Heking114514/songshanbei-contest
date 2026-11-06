# Version Lock

本目录是八卡阶段的提交工作区。三个仓库都以**我方改好的代码**作为 baseline，不再以官方 clone 作为提交起点；官方仓库只作为差异审计参考。

| Component | Local path | Branch | Baseline | Submission commit | Status |
|---|---|---|---|---|---|
| InfiniLM | `InfiniLM/` | `stage1` | `songshan_minimax@a3bdf10` + `stage1_work` | `79350958fe06f8e7cd7684ff0773ea06d3449a85` | 我方修改代码基线 + cache 修复 |
| InfiniCore | `InfiniCore/` | `stage1` | `pull_20261106/InfiniCore` | `5a674e269717adc578c0d599846c9818e6d741fa` | 我方修改代码基线 + build 修复 |
| NineToothed | `ninetoothed/` | `stage1` | `pull_20261106/ninetoothed` 快照 | `6ce8568` | 快照基线，官方 commit 待确认 |

## 官方参考（不是提交基线）

| Component | Official reference | 用途 |
|---|---|---|
| InfiniLM | `minimax_h3@65deb91` | 仅做 diff 对照 |
| InfiniCore | `minimax-h3@f32e2fb` | 仅做 diff 对照 |

## 运行资源

- 八卡资源：已获得
- 平台操作统一走 CLI：`/d/Desktop/songshan workspace/ifab/ifab.py`
- `ifab_frontend` 仅作浏览器端参考，不作为自动化入口
- 模型权重：已在平台 AFS 空间，从 `$IFAB_AFS_HOME` 下加载
- 不登录 SCNet Slurm/SSH；使用新八卡平台

## 重要说明

- `InfiniLM` 和 `InfiniCore` 现在以我方修改代码为 baseline；官方 commit 只保留在对照记录中。
- `InfiniLM@7935095` 包含 `reset_cache(nullptr)` 的 engine/worker 空配置修复。
- `InfiniCore@5a674e2` 包含 `__has_include` 兼容头和 Python/pybind11 include 自动探测，去掉了 `/public/software/.../python3.10` 硬编码。
- 这两项修复尚未在目标八卡环境重新构建验证。
- `ninetoothed` 的 GitHub 拉取曾被网络重置，当前仓库来自 `pull_20261106` 快照，**不是已核验的官方 commit**。正式提交前必须与官方 `InfiniTensor/ninetoothed` 对齐，并把真实 commit 更新到本表。
- 三个仓库当前都没有配置 public `origin`；推送前由负责人添加目标 public 仓库地址。
- 不把模型权重、token、密码、隐藏测试数据放进任何仓库。
- `InfiniLM/stage1_work/` 是单卡阶段证据和工具，不是八卡正式运行时；其中历史脚本仍包含旧路径，需按八卡工单修复。

## 构建参数占位

```text
OS:
DTK:
HIP:
Python:
PyTorch:
xmake:
InfiniCore flags:
InfiniLM flags:
NineToothed install:
Build date:
```

## 八卡运行目标

```text
Task: FL2VA
Prompt: A dog falls asleep
Steps: 50
Frames: 124
FPS: 24
Audio: 32 kHz, 2 channels
Entry: InfiniLM/examples/minimax_h3.py fl2va
Compare: InfiniLM/examples/minimax_h3_compare.py --tasks fl2va
```
