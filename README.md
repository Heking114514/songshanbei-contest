# MiniMax-H3 八卡提交仓库

更新时间：2026-11-08T12:40:52+08:00（北京时间）。

本仓库整合InfiniLM、InfiniCore和NineToothed，包含H3环境准备、-j120增量构建、单卡开发自验、八卡短推理和官方FL2VA比较入口。完整命令见[REPRODUCE.md](REPRODUCE.md)，版本/依赖见[VERSIONS.md](VERSIONS.md)，报告内容见[复现与提交说明](复现与提交说明.md)。

## 已核实的运行范围

| 验收 | 实际结果 | 边界 |
|---|---|---|
| 历史完整native TP8生成 | 完整层数、124帧、50步请求/49次DiT、768×928 | 历史独立运行，加载至保存542.131秒 |
| 历史原版FL2VA compare | Size/Length PASS，SSIM94.277% | 新生成reference、复用已核验native；native时间n/a，没有同轮加速比 |
| 新环境源码构建 | 新Python依赖目录和1194个原生对象完成 | 首次跨两轮八卡租约；恢复1157对象SHA不变，只补37 |
| 新环境总入口复跑 | prepare/build/eight-smoke累计两次rc=0 | 完整层数，但仅22帧、256×256、2步请求；复跑1194对象SHA不变 |

新环境短推理摘要及实测脚本指纹见[结构化记录](results/h3_reproduction_20261108.json)，历史官方结果原文见[compare输出](results/h3_official_compare_20261107.txt)。SSIM不是相同像素比例，也不代表音频一致性或所有测试用例已通过。

**尚未完成：干净环境到双方全新生成完整官方compare的整链验收、最终发布版本的匿名clone实机验收，以及官方换图/换Prompt的通用入口适配。** 本次提供的源码和材料不改变这些运行边界。历史约50分钟reference与约9分钟native来自不同运行且计时口径不同，不能相除作为官方加速比。

## 选择工作流

| 用途 | 入口与顺序 |
|---|---|
| 算法/接口更新后的单卡自验 | reproduce_h3.sh：prepare→build→jit→smoke；缩层双端推理及禁新JIT复跑 |
| 八卡快速链路验证 | reproduce_h3_all.sh：prepare→build→eight-smoke；首次允许JIT，生成短视频 |
| 八卡正式提交 | prepare→build→source新环境→来源审计/八卡检查→run_h3_compare.sh；完整双方新生成 |

正式八卡直接流程的完整载荷在REPRODUCE.md。reproduce_h3.sh的preflight/compare分支保留真实single-ready门禁，属于先单卡后八卡的另一流程；不要伪造回执。短推理成功不等于正式compare通过。

## 源码、依赖和输入

- InfiniCore/、InfiniLM/、ninetoothed/属于同一个Git仓库。
- requirements/锁定55个CPython3.11/Linux wheel的版本、URL、大小和SHA；平台Torch由赛事基础环境提供，不安装PyPI Torch。
- third_party/h3/随附HCU FlagTree、clang/lld和xmake源码包，均有来源、SHA与许可。H3原生库、模型权重和旧JIT缓存不随包发布。
- 当前模板输入为inputs/first_frame.png、Prompt A dog falls asleep、seed42。图片/官方脚本的固定SHA和compare的Prompt常量仍需按官方新用例适配；仅换FIRST_FRAME路径会被现有预检拒绝。

目标平台：赛事海光Linux x86_64、CPython3.11、平台Torch2.9.0/HIP6.3、gfx936、DTK /opt/dtk。MODEL_PATH默认指向共享MiniMax-H3官方权重，评审可指定等价目录。BF16 math SDPA与reference每批4个独立attention head的调度需在报告披露，原版compare保持不变。

所有本地和派发命令统一指定Git Bash完整可执行路径，整段Bash载荷先base64；禁止PowerShell/pwsh、cmd、WSL及裸bash.exe别名。已有构建保持相同源码/work路径增量，不clean；日志实时tee、每30秒心跳。AFS上限4GiB，只查专门接口，按新鲜完整统计和峰值预算保存；节点环境和共享权重不能未经预算整份复制到AFS。

仓库：[songshanbei-contest](https://github.com/Heking114514/songshanbei-contest)，分支codex/eight-card-handover。复现时固定包含本文的完整40位commit（git rev-parse HEAD）；旧8344881只作历史基线，不包含本次新增入口与固定资产。

## 远程提交与访问状态

2026-11-08T12:40:52+08:00：源码与固定资产已通过SSH推送到codex/eight-card-handover，源码提交为cb6812d15cb2ececfe7b91130ff7491b4b6805c1，远程ref已核对。匿名访问仓库网页、GitHub API和raw文件均返回HTTP404；当前公开可访问性未确认，不能宣称匿名clone已通过。此记录仅说明提交/访问状态，不改变运行验收范围。
