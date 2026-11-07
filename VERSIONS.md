# Version Lock
> 2026-11-07 19:49:12 +08:00 发布状态：本轮原生 TP8 源码已提交并推送到 codex/eight-card-handover，源码 commit 为 ee93718a59c4c2251aee5a5c5da6c6b33daa085b，匿名 GitHub API 已确认可访问。后续文档修订以分支最新 HEAD 为准。原生正式生成已完成；完整官方 compare 和公开源码构建实测仍待完成。


更新时间：2026-11-07 19:44:41 +08:00（北京时间）。

本目录是包含三个组件的整合 Git 仓库。当前分支为 `codex/eight-card-handover`，远程为 `https://github.com/Heking114514/songshanbei-contest`；组件子目录没有各自独立的 `.git`。当前发布源码以本仓库的 `git rev-parse HEAD` 为准。

## 历史单卡来源

单卡整合基线为 `acac3452bb0b7a059cb388bd928d181b02925843`（2026-11-06）。下表保留组件导入来源，不代表本轮八卡增量的独立提交号。

| Component | Local path | Source commit |
|---|---|---|
| InfiniLM | `InfiniLM/`，我方 `songshan_minimax` + `stage1_work` | `79350958fe06f8e7cd7684ff0773ea06d3449a85` |
| InfiniCore | `InfiniCore/`，我方 `pull_20261106/InfiniCore` | `5a674e269717adc578c0d599846c9818e6d741fa` |
| NineToothed | `ninetoothed/`，我方快照 | `6ce8568f8d208ba55afe4a5508bdc9896062b756` |

NineToothed 官方来源 commit 尚未核验；本轮没有修改其源码。不得把我方快照 SHA 宣称为已核验的官方版本。

## 本轮增量

- InfiniLM：请求级 conditioning revision 经 Python、pybind、engine、TP worker 传到 H3；文本/位置快照检测缓存失效；`begin_generation()` 清理请求边界；跳过会被缓存索引替代的原 float64 位置转换；支持 Diffusers 的 PIL 帧批次输出校验。
- InfiniCore：允许指定 Torch 路径；统一海光 DTK/CUDA 兼容头文件的 Thrust/CUB 包含顺序和编译参数。
- 增加 CPU 回归测试、构建/冒烟/正式比较入口、BF16 math SDPA 配置和 GPU/依赖预检。
- 统一首帧保存在 `inputs/first_frame.png`，来源、大小和 SHA 见 `inputs/README.md`；没有修改或重编码。

## 官方参考（只用于差异审计）

| Component | Reference branch | Commit |
|---|---|---|
| InfiniLM | `minimax_h3` | `65deb91f840ebaef752d028fad2f3217ce0950f4` |
| InfiniCore | `minimax-h3` | `f32e2fbb2a9b2ca912b5711f4f3836c403196a74` |

官方 compare 源码 `InfiniLM/examples/minimax_h3_compare.py` 保持原样，LF 归一后的 SHA-256 为 `24f3439e11e98d979ec8cee5f76284079dad87cb4668cf0f109b7904e8641bc3`。

## 已验证平台与配置

- Python 3.11、平台海光 Torch 2.9 / HIP 6.3、gfx936、DTK `/opt/dtk`。
- Transformers 5.5.0、Hugging Face Hub 1.23.0、Diffusers 0.40.0、PyAV 16.0.1、FlagTree 0.7.0+hcu3.6 / Triton 3.6。
- Core：release、ATen/CCL 开启；NineToothed 使用 Python JIT 路径。LM：同一 Python/Torch C++ ABI。
- 首次源码构建入口 `scripts/build_hygon.sh` 默认 `BUILD_JOBS=120`；已有稳定对象按交接流程增量恢复，不能无故全量重编。
- 共享权重为 `/public/home/aclsylqidf/shared/MiniMax-H3`，不在个人 AFS，也不入 Git。个人 AFS 仅保存环境、编译缓存和证据，保持低于 4 GiB；用量只查专门 API。
- 本地命令与远端派发统一使用 Git Bash，完整脚本整体 base64。运行包恢复和平台会话调度还依赖外部 `handover/scripts` 及平台资产，详见复现说明。

## 验证状态

- 原生 H3 构建、1194 对象跨租约零重编和单文件增量已验证。
- 单卡接口及真实 float64 位置输入回归通过；运行包换目录且禁止新 JIT 编译恢复通过。
- 完整模型 TP8：22 帧短冒烟通过；124 帧、50 步请求、seed 42、提示词 `A dog falls asleep` 的正式生成通过。
- 正式产物：768×928、24 fps、H.264，AAC 双声道 32000 Hz；请求 50 步，调度器实际 49 次 DiT。使用 math BF16 SDPA；该精度配置需随结果披露。
- 正式视频 SHA-256：`ef0fa80ae779545881c6bc233494c2950c1edc974f0bce985c782582cc19e0ed`。
- 官方 compare 已启动但因 HIP 检测不到 GPU 而 `SKIPPED`，退出码 2；尚无 legacy 参考视频、SSIM 或全模型数值对齐结论。
- 新的构建至 compare wrapper、预检和 hook 只经过本地检查，尚未在健康新租约完成端到端验收。CPU pytest 用例不能仅凭语法检查宣称执行通过。
- 原八卡租约已于 2026-11-07 18:31:51 +08:00 到期；后续运行先确认有效实例，不能以旧 API 的 Active 状态代替 GPU 计算检查。

完整参数、真实结果与恢复条件见 [复现与提交说明.md](复现与提交说明.md)。
