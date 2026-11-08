# 源码、离线依赖与运行环境的位置

更新时间：2026-11-09T01:19:29+08:00（北京时间）。

启动命令与取结果见 [README](README.md)，依赖获取见 [PACKAGING.md](PACKAGING.md)。

## AFS目录

| 位置 | 内容 |
|---|---|
| `个人AFS/songshanbei-contest/` | 上传的源码、三份工具、图片和参考视频 |
| `个人AFS/songshanbei-contest/offline-wheels/` | 放入的55个普通wheel |
| `个人AFS/h3-results/run-时间戳-进程号/` | 自动保存的本次视频、日志及比较结果 |

## 节点目录

脚本根据新终端的初始workspace自动选择其同级 `h3-native-编号/`；编号由AFS源码路径确定，相同节点和源码目录重试时保持稳定。启动时打印完整 `H3_NODE_ROOT` 路径。

~~~text
$H3_NODE_ROOT/
├── source-stage.json               # 源码同步与校验记录
├── source/                         # 从AFS复制的源码及编译产物
│   ├── InfiniCore/
│   ├── InfiniLM/
│   ├── ninetoothed/
│   └── scripts/
├── work/                           # 自动安装和加载的环境
│   ├── venv/bin/python3            # Python入口
│   ├── python/                    # 安装后的依赖及FlagTree
│   ├── install/                   # InfiniCore/InfiniLM原生库
│   ├── toolchain/                 # clang/lld
│   ├── xmake/
│   ├── downloads/                 # 校验后的离线wheel副本
│   ├── config/                    # 增量构建配置
│   ├── jit/
│   └── prepared.json
└── runs/
    └── native-时间戳-进程号/        # check模式使用check前缀
        ├── result.mp4
        ├── generation.log
        └── workflow.log
~~~

Python3.11与Torch2.9.0/HIP6.3由平台/usr/local提供，DTK/gfx936由/opt/dtk提供。脚本在此基础上安装依赖和构建原生库，并自动配置使用，无需激活环境。模型从MODEL_PATH指向的共享目录只读加载。

## 执行目录与复用

推荐在新八卡终端初始目录运行README中的一行命令。如果从初始目录直接cd进仓库，也可以执行 `bash run_submission.sh`，入口通过前一个workspace定位节点盘。

若终端已切换过多个目录，重新打开Bash使用推荐命令，或通过H3_NODE_ROOT指定已有节点工程目录。不会自动回退到/tmp、AFS或受限workspace内构建。

同节点同源码目录重复运行，未变化的源码保持mtime，环境和对象保留；源码更新后增量编译。节点回收后，节点环境和对象回收；AFS源码、wheel和结果保留。
