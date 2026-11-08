# 版本、依赖与验证边界

更新时间：2026-11-08T12:33:23+08:00（北京时间）。

本仓库是三个组件的整合 Git 仓库，分支 codex/eight-card-handover。组件子目录没有独立 .git。以下区分源码来源、历史完整成绩和本提交新增的独立复现入口。运行时以包含本文的完整commit为准；历史执行发生在该版本公开前，不伪造当时的源码身份。

## 版本身份

| 版本 | 含义 |
|---|---|
| acac3452bb0b7a059cb388bd928d181b02925843 | 历史单卡整合基线 |
| ee93718a59c4c2251aee5a5c5da6c6b33daa085b | 已公开原生 TP8 及构建修复 |
| 834488108a0af0636590e2349240ac336163ecc5 | 本次新增入口之前的历史公开基线；新环境执行时以它加候选脚本构建 |
| 包含本文的提交（git rev-parse HEAD） | 包含预检/参考分头/hook/wrapper、独立环境与增量入口、依赖锁和第三方资产；实际验收边界见下文 |

不能把 8344881 写成包含本轮所有修复的最终复现版本。交付以包含本轮内容的完整40位SHA、匿名clone内容和实际验收证据为准；发布源码不能代替GPU整链验收。

## 组件导入来源

| 组件 | 我方源 commit |
|---|---|
| InfiniLM（songshan_minimax + stage1_work） | 79350958fe06f8e7cd7684ff0773ea06d3449a85 |
| InfiniCore（pull_20261106） | 5a674e269717adc578c0d599846c9818e6d741fa |
| NineToothed 快照 | 6ce8568f8d208ba55afe4a5508bdc9896062b756 |

NineToothed 的官方上游 commit 尚未独立核验；本轮不改其源码。基线说明见 BASELINE.md。

官方只读差异参考：InfiniLM minimax_h3 分支 65deb91f840ebaef752d028fad2f3217ce0950f4；InfiniCore minimax-h3 分支 f32e2fbb2a9b2ca912b5711f4f3836c403196a74。

## 源码范围

InfiniLM 已含请求级 conditioning revision、缓存失效/请求边界处理、位置缓存路径、Diffusers 输出容器兼容。InfiniCore 已含 Torch 路径选择、DTK/CUDA 头文件与 ABI 构建兼容、NineToothed SnakeBeta。参考端独立 head 分批通过仓库 helper 注入；原版 compare 不改。

官方 compare 的 LF SHA-256：
24f3439e11e98d979ec8cee5f76284079dad87cb4668cf0f109b7904e8641bc3。

统一图片原始 SHA-256：
b97c3487ca35da0c803fba3787cf2b7f936d3d4595548ae5b7b576c19934be54。

## 独立复现固定依赖

目标平台：Linux x86_64、CPython3.11、/usr/local 平台 Torch2.9.0 / HIP6.3、gfx936、DTK /opt/dtk。不安装 PyPI Torch。

[依赖锁](requirements/h3_linux_cp311.lock.json) 固定55个 wheel（共192118057字节），包括 Transformers5.5.0、Hub1.23.0、Diffusers0.40.0、accelerate1.11.0、numpy1.26.4、PyAV16.0.1、safetensors0.8.0、scikit-image0.25.2、CPU torchvision0.24.0+cpu。闭包已本地校验，已在本轮新Python目录安装并通过八卡短推理；历史完整compare环境并非逐包等同这份新锁。锁中的scope保留制锁时的候选说明，不改锁字节冒充新的运行回执。

[第三方 manifest](third_party/h3/manifest.json) 固定 HCU FlagTree0.7.0+hcu3.6（Triton3.6）、clang-18/lld、xmake3.0.3源码包，共149724964字节。FlagTree 来源 commit47970467c4d75a9cfffb84426383c7e7a9f2557a，补丁/许可与来源见 third_party/h3/README.md。H3 原生扩展与 JIT 不在此包内，新环境首次生成。

## 已有运行与本轮检查

已有实测：H3 构建/导入，1194对象跨租约零变化、单文件增量，单卡真实双端缩层冒烟，完整 native TP8 124帧/50步生成，完整官方 FL2VA 比较94.277%。正式配置统一图、Prompt A dog falls asleep、seed42、768×928/24fps、math_bf16、reference head_batch4；本轮 reference 新生成、native provided，性能 n/a、无同轮加速比。

native视频 SHA-256：ef0fa80ae779545881c6bc233494c2950c1edc974f0bce985c782582cc19e0ed。
reference视频 SHA-256：1ca2e40a53b85479d3e9e06a2644801fc91324ba71c197ba484f1d33ee05056b。

本轮新入口已在新Python目录完成锁定依赖安装和1194对象源码构建，首次构建跨两轮八卡租约；恢复1157对象SHA不变，只补37。随后仓库总入口prepare/build/eight-smoke累计两次rc=0，复跑1194对象SHA不变。推理为完整模型层数、22帧/256×256/2步请求，不能当正式compare。运行摘要及实测脚本指纹见[执行记录](results/h3_reproduction_20261108.json)。

尚未完成新环境双方全新生成的完整官方compare、这套bootstrap单卡jit/smoke组合、最终公开版本匿名clone实机验收和换例入口适配。当前预检图片SHA、官方脚本SHA和compare Prompt绑定模板用例；底层参数可变不代表外层适配已完成。Python环境和新增JIT的跨节点零编译恢复也未验收。

完整命令及下一步验收条件见 [REPRODUCE.md](REPRODUCE.md)。已有结果原文见 [结果](results/h3_official_compare_20261107.txt)。
