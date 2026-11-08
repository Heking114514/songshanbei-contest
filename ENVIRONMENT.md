# 源码、交付包与运行环境的位置

更新时间：2026-11-08T23:32:04+08:00（北京时间）。

运行步骤见 [README](README.md)，从源码制包见 [PACKAGING.md](PACKAGING.md)。

## 先分清：源码、交付包、运行环境

`h3-submission/` 是打包命令生成的目录，名字来自 `--output ../h3-submission`。仓库克隆下来时没有这个目录；[制包说明](PACKAGING.md)中的命令会创建它。

| 东西 | 放在哪里 | 包含什么 | 何时创建 |
|---|---|---|---|
| 源码仓库 | 你克隆仓库的目录 | InfiniCore、InfiniLM、脚本、依赖清单、三份工具、图片和参考视频 | 克隆仓库时 |
| 离线wheel文件夹 | 仓库同级 `h3-wheelhouse/` | 55个尚未安装的Python依赖文件 | 下载或放入依赖时 |
| 上传交付包 | 仓库同级 `h3-submission/`，然后上传至AFS同名目录 | 压缩后的源码、依赖、工具，以及启动脚本、清单和参考视频 | 运行打包命令时 |
| 已安装的运行环境 | 八卡节点 `$H3_NODE_ROOT/work/` | Python环境、安装后的依赖、工具和原生库 | 运行正式脚本时 |
| 最终结果 | AFS `$H3_AFS_OUTPUT_DIR/` | 视频、日志及校验结果 | 推理结束保存时 |

正式操作分为两处：**在源码仓库中制作交付包；在AFS交付包目录中启动推理。** 仓库内的 `run_submission.sh` 会随打包一起复制出去，启动时需要同目录下的 `delivery-manifest.json` 和 `payload.part*`。

## 整个环境在节点上是什么结构

安装好的环境由脚本在八卡节点创建。`H3_NODE_ROOT` 是你在启动命令中设置的节点工程根目录，其下结构为：

~~~text
$H3_NODE_ROOT/                       # 八卡节点上的工程目录
├── source/                         # 自动解压的源码
│   ├── InfiniCore/                 # Core源码及构建文件
│   ├── InfiniLM/                   # H3源码及构建文件
│   ├── ninetoothed/
│   └── scripts/
├── wheelhouse/                     # 自动解压的55个原始wheel
├── work/                           # 实际运行环境，脚本自动安装和加载
│   ├── venv/bin/python3            # 本工程使用的Python入口
│   ├── python/                    # 安装后的Python依赖和FlagTree
│   ├── install/                   # 编译安装的InfiniCore/InfiniLM原生库
│   ├── toolchain/                 # clang/lld工具链
│   ├── xmake/                     # xmake构建工具
│   ├── downloads/                 # 核验后的离线wheel副本
│   ├── config/                    # 增量构建配置
│   ├── jit/                       # 推理时生成的JIT缓存
│   └── prepared.json              # 环境准备完成记录
└── runs/
    └── native-时间戳-进程号/        # check模式则为check-时间戳-进程号
        ├── result.mp4
        ├── generation.log
        └── workflow.log
~~~

Python3.11和Torch2.9.0由平台 `/usr/local/` 提供，DTK由 `/opt/dtk/` 提供；脚本在此基础上安装本工程依赖。模型权重从 `MODEL_PATH` 指定的共享目录只读加载。

例如将 `H3_NODE_ROOT` 设置为 `/节点私有目录/h3-submission-native`，实际Python入口就是 `/节点私有目录/h3-submission-native/work/venv/bin/python3`，安装后的依赖就在同级 `work/python/`。脚本自动配置并使用它们，无需另行激活环境。

节点上的环境和编译产物随节点资源回收；AFS中的上传包和已保存结果长期保留。同一节点、相同交付包重试时，保留同一个 `H3_NODE_ROOT` 可复用环境和增量编译产物。新节点可使用AFS上传包重新安装和构建。
