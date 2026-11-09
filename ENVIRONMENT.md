# 源码、离线依赖与运行环境的位置

更新时间：2026-11-09T11:12:27+08:00（北京时间）。

启动命令与取结果见 [README](README.md)，依赖获取见 [PACKAGING.md](PACKAGING.md)。

## AFS目录

| 位置 | 内容 |
|---|---|
| `个人AFS/songshanbei-contest/` | 上传的源码、四份工具、图片和参考视频 |
| `个人AFS/songshanbei-contest/h3-offline-wheels-linux-cp311.zip` | 推荐：原名放置的离线包，自动在节点解压 |
| `个人AFS/songshanbei-contest/offline-wheels/` | 可选：已解压的55个普通wheel；与ZIP二选一 |
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
│   ├── downloads/                 # 大小检查后的离线wheel副本
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

当前部署优化：源码扫描使用scandir复用目录项元数据、逐目录拒绝符号链接，避免每个文件反复解析全部祖先路径；复制及复制后核验默认16线程（H3_STAGE_WORKERS，可设1..32）；仅用存在性、字节数和mtime检查同步状态，不读取文件内容计算SHA；节点文件大小或mtime发生独立变化时拒绝覆盖，未变文件保持mtime。xmake优先使用随源码提供的官方3.0.3自包含工具，无法运行时回退源码构建；H3原生库始终源码编译。正式工作流只执行一次完整来源审计，直接保存本次origins.json；环境脚本从prepared.json读取平台Torch位置，不为查询路径重复导入Torch。

完整性策略：工具与wheel核对存在性及清单字节数，安装/解压工具负责读取归档格式；图片和视频进行真实解码。来源审计仍确认Python模块和实际映射原生库来自本次节点目录。保存到AFS使用.partial文件、大小检查和原子改名，不做内容哈希。构建配置直接记录配置值及工具大小/mtime，避免为配置缓存再次读取整个二进制。相同大小且mtime未变化的内容替换不在此完整性检查的检测范围内。

ZIP处理：源端只读取ZIP中央目录、成员名和大小，解压直接写入节点work/downloads/，解压中检查格式/CRC并逐文件原子改名。该目录后续作为离线wheel来源，避免额外复制。已完整提取的文件同节点重试直接复用；失败的.partial不会用于安装。AFS中不生成offline-wheels副本。

正式入口的GPU可见性、图片/模型目录、依赖和编码器检查，在真实生成进程启动时执行，失败仍在模型加载前退出并保存preflight.json。构建末尾的原生导入检查并入来源审计；独立调用build_hygon.sh仍默认保留导入检查。wheel仅在源码stage时解压/暂存一次，不在正式worker重复调用。
