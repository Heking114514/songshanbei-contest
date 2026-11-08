# 源码、离线依赖与运行环境的位置

更新时间：2026-11-09T00:45:35+08:00（北京时间）。

上传、启动和取结果见 [README](README.md)，获取依赖见 [PACKAGING.md](PACKAGING.md)。

## AFS 上放什么

| 位置 | 内容 | 如何准备 |
|---|---|---|
| `个人AFS/songshanbei-contest/` | 本仓库源码、脚本、三份工具、输入图及参考视频 | 上传源码仓库 |
| `个人AFS/songshanbei-contest/offline-wheels/` | 55 个尚未安装的普通 wheel | 下载后原名放入 |
| `个人AFS/storage-info.json` | 最新完整容量统计 | 资源就绪后导出并上传 |
| `个人AFS/h3-results/` | 视频、日志和比较结果 | 运行结束由脚本保存 |

AFS 中的源码和 wheel 是安装输入。实际 Python 环境、编译对象和 JIT 缓存由脚本在节点盘创建，避免占用个人 AFS。

## 节点上的环境结构

`H3_NODE_ROOT` 是启动时指定的节点目录，必须位于个人 AFS 和受限的 `workspace` 目录之外。在新八卡 Bash 的初始工作目录中，`$(dirname "$PWD")/h3-native` 表示初始目录的同级目录。

~~~text
$H3_NODE_ROOT/
├── source-stage.json               # 上传源码的同步与校验记录
├── source/                         # 从AFS复制的源码
│   ├── InfiniCore/                 # Core源码及此节点的编译产物
│   ├── InfiniLM/                   # H3源码及此节点的编译产物
│   ├── ninetoothed/
│   └── scripts/
├── work/                           # 实际环境，脚本自动安装和加载
│   ├── venv/bin/python3            # 本工程的Python入口
│   ├── python/                    # 安装后的55个依赖及FlagTree
│   ├── install/                   # InfiniCore/InfiniLM原生库
│   ├── toolchain/                 # clang/lld工具链
│   ├── xmake/                     # xmake构建工具
│   ├── downloads/                 # 从AFS复制并核验的离线wheel
│   ├── config/                    # 增量构建配置
│   ├── jit/                       # 推理时生成的JIT缓存
│   └── prepared.json              # 环境准备完成记录
└── runs/
    └── native-时间戳-进程号/        # check模式使用check前缀
        ├── result.mp4
        ├── generation.log
        └── workflow.log
~~~

Python 3.11 和 Torch 2.9.0/HIP 6.3 由平台 `/usr/local/` 提供，DTK/gfx936 工具由 `/opt/dtk/` 提供。脚本在此基础上创建 Python 入口，将依赖安装到 `work/python/`，将编译后的原生库安装到 `work/install/`。模型从 `MODEL_PATH` 指向的共享目录只读加载。

例如 `H3_NODE_ROOT=/节点私有目录/h3-native`，Python 入口就是 `/节点私有目录/h3-native/work/venv/bin/python3`。脚本自动设置其搜索路径和原生库路径，无需手工激活环境。

## 什么时候能复用

同一节点和同一 AFS 源码目录重试时，保留 `H3_NODE_ROOT`。入口按内容校验，未变化的源码不重写；环境和编译产物留在原位置，源码更新交由构建系统增量处理。旧编译目录不会从上传源码中带入。

节点资源回收后，上述节点目录随之回收。AFS 中上传的源码、wheel 和已保存结果继续保留；新节点重新安装和构建。
