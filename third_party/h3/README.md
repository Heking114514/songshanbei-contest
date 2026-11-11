# H3 固定第三方构建依赖

55 个普通 Python wheel 位于 `third_party/offline-wheels/`，随 Git 仓库提供；本目录保存 HCU 编译后端和构建工具。使用者整体上传仓库即可，无需另外下载或解压外层依赖包。文件名、大小与版本见 [工具清单](manifest.json) 和 [Python依赖锁](../../requirements/h3_linux_cp311.lock.json)。

安装流程只检查存在性、大小、归档格式/CRC和可运行版本，不执行文件哈希校验；H3 原生库仍由源码编译，不随依赖提供旧构建对象或 JIT 缓存。

| 文件 | 字节数 | 用途 |
|---|---:|---|
| `flagtree-0.7.0+hcu3.6-cp311-cp311-linux_x86_64.whl` | 72,231,003 | CPython 3.11 / Linux x86_64 的 HCU 编译后端 |
| `hcu-jit-clang18-minimal.tar.gz` | 74,463,838 | clang-18、lld 及相对链接 |
| `xmake-v3.0.3.gz.run` | 3,030,123 | xmake 3.0.3 源码回退工具 |
| `xmake-bundle-v3.0.3.linux.x86_64` | 2,761,352 | 优先使用的官方自包含 xmake 工具 |

## FlagTree 来源与构建补丁

上游：[flagos-ai/FlagTree](https://github.com/flagos-ai/FlagTree)，固定 commit 为 [47970467c4d75a9cfffb84426383c7e7a9f2557a](https://github.com/flagos-ai/FlagTree/commit/47970467c4d75a9cfffb84426383c7e7a9f2557a)。

[patch_flagtree_build.py](patch_flagtree_build.py) 保留实际构建修复：尊重 profiler 关闭选项、添加生成 target 依赖、开发工具设为 EXCLUDE_FROM_ALL。正式启动直接安装本目录的固定 wheel，不重新构建整套 FlagTree。许可见 [LICENSE.flagtree](LICENSE.flagtree)。

## JIT 工具链与 xmake

clang/lld 来自厂商镜像 `harbor.baai.ac.cn/flagtree/flagtree-hcu-py310-torch2.10.0-dtk26.04-ubuntu22.04`。脚本展开到节点工具目录，按清单核对两个二进制的大小，再使用平台 DTK。许可见 [LICENSE.llvm](LICENSE.llvm)。

xmake 使用官方 [v3.0.3 Linux x86_64 自包含工具](https://github.com/xmake-io/xmake/releases/download/v3.0.3/xmake-bundle-v3.0.3.linux.x86_64)，无法运行时回退到随仓库提供的 [v3.0.3 源码工具](https://github.com/xmake-io/xmake/releases/download/v3.0.3/xmake-v3.0.3.gz.run) 构建安装。许可见 [LICENSE.xmake](LICENSE.xmake)。

完整的上传、启动和取产物步骤见 [README](../../README.md)。
