# 单卡基线固化

本仓库是 MiniMax-H3 八卡阶段的单卡基线快照。

## 组件来源

| 目录 | 来源 | 来源 commit | 集成方式 |
|---|---|---|---|
| `InfiniLM/` | `songshan_minimax` + `stage1_work` | `79350958fe06f8e7cd7684ff0773ea06d3449a85` | git subtree |
| `InfiniCore/` | `pull_20261106/InfiniCore` | `5a674e269717adc578c0d599846c9818e6d741fa` | git subtree |
| `ninetoothed/` | `pull_20261106/ninetoothed` 快照 | `6ce8568f8d208ba55afe4a5508bdc9896062b756` | 快照目录 |

## 说明

- `InfiniLM` 包含 `reset_cache(nullptr)` 的 engine/worker 修复。
- `InfiniCore` 包含 `c10` 兼容头、Python include 和 pybind11 include 的可移植构建修复。
- `ninetoothed` 官方来源 commit 尚未核验；当前是 `pull_20261106` 快照。
- 官方参考代码不在此仓库；在本地 `reference/` 目录，仅用于差异审计。

## 固化信息

- 固化日期：2026-11-06
- 仓库：`git@github.com:Heking114514/songshanbei-contest.git`
- 目标 commit message：`单卡基线固化11062027`
