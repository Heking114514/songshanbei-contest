# MiniMax-H3 八卡阶段提交工作区

这个目录按官方要求准备三个独立仓库：

```text
submit/
├── InfiniLM/       # stage1：我方 songshan_minimax 基线 + stage1_work
├── InfiniCore/     # stage1：我方 pull_20261106 基线
├── ninetoothed/     # stage1：我方 NineToothed 快照基线
├── VERSIONS.md
└── README.md
```

## 平台与模型

- 八卡资源已经获得，直接使用新八卡平台；不再登录 SCNet Slurm/SSH。
- 平台操作统一走 CLI：`/d/Desktop/songshan workspace/ifab/ifab.py`。
- `ifab_frontend` 仅作浏览器端参考，不作为自动化入口。
- 已有 token 时先用 `python ifab.py --json status`；只有 token 失效才重新登录。
- MiniMax-H3 权重已经在平台 AFS 空间，不需要重新下载。
- 权重路径从 `$IFAB_AFS_HOME` 下确认，运行时设置 `MODEL_PATH`，不要把权重复制进项目。

```bash
cd "/d/Desktop/songshan workspace/ifab"
python ifab.py --json status

echo "$IFAB_AFS_HOME"
find "$IFAB_AFS_HOME" -maxdepth 4 \
  \( -iname 'MiniMax-H3' -o -iname 'FL2VA' -o -iname 'diffusers' \) \
  -print 2>/dev/null
```

## 先读

1. `VERSIONS.md`：三仓的基线、分支和提交号。
2. `../八卡阶段代码边界与改造审计.md`：赛事边界、P0、工单和验收。
3. `InfiniLM/stage1_work/README.md`：单卡阶段证据和限制。
4. `../reference/`：官方 InfiniLM、InfiniCore 参考仓库；只做 diff，不作为提交基线。

## Git Bash 约定

所有命令在 **Git Bash** 中执行。禁止 PowerShell/pwsh 执行或包装：

```bash
cd "/d/Desktop/songshan workspace/submit"
pwd
git -C InfiniLM status --short --branch
git -C InfiniCore status --short --branch
git -C ninetoothed status --short --branch
```

## 仓库现状

| 仓库 | 分支 | 内容 |
|---|---|---|
| InfiniLM | `stage1` | 我方 `songshan_minimax` 基线 + `stage1_work/` 证据 |
| InfiniCore | `stage1` | 我方 `pull_20261106` 基线 + SnakeBeta/接口/后端增量 |
| NineToothed | `stage1` | 我方快照基线；官方 commit 待确认 |

## 推送前

1. 给每个仓库创建 public remote，不要把当前本地 `upstream` 当提交目标。
2. 把 `VERSIONS.md` 的 commit 更新为推送后的真实 SHA。
3. 检查没有 `*.so`、`*.pkl`、权重、token、密码。
4. 运行 `git status --short`，确认工作区干净。

## 不可修改

- 模型权重、测试数据、隐藏用例；
- `InfiniLM/examples/minimax_h3_compare.py`；
- 评测程序和基线；
- 不得用预计算结果或外部在线服务替代推理。

## 下一步

优先完成：

1. `reset_cache(None)` 修复；
2. 文本/RoPE 缓存失效键；
3. 构建口径统一；
4. 八卡短冒烟；
5. 124 帧/50 步 FL2VA；
6. 官方 compare；
7. 连续用例和模板报告。

细节见 `../八卡阶段代码边界与改造审计.md`。
