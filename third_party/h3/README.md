# H3 固定第三方构建依赖

更新时间：2026-11-08T12:33:23+08:00（北京时间）。

本目录随源码交付目标平台的固定依赖；不包含 InfiniCore/InfiniLM 散装原生扩展、模型权重、旧 JIT 或个人 venv。文件大小以 [manifest.json](manifest.json) 为准，bootstrap 安装前只核对存在性和大小；历史SHA作为来源记录，不参与通用资产校验。H3 注意力归档例外：正式入口在逐成员 SHA、内部 `bundle.json`、ABI 和 dispatcher 身份全部匹配后才部署。本提交随附这五份固定资产；新环境实际执行摘要见[验收记录](../../results/h3_reproduction_20261108.json)。

| 文件 | 字节数 | 用途 |
|---|---:|---|
| flagtree-0.7.0+hcu3.6-cp311-cp311-linux_x86_64.whl | 72231003 | CPython3.11/Linux x86_64 的 HCU Triton3.6 编译器/运行时 |
| hcu-jit-clang18-minimal.tar.gz | 74463838 | clang-18、lld 与相对链接，用平台 DTK 提供 ROCm 支持 |
| xmake-v3.0.3.gz.run | 3030123 | xmake3.0.3 源码回退包 |
| xmake-bundle-v3.0.3.linux.x86_64 | 2761352 | 官方自包含工具，优先直接运行，启动时检查大小和可运行版本 |
| phase10-best-bundle.tar.gz | 132170 | gfx936 第九/十阶段最佳注意力 dispatcher、5 个子库、ABI 和策略；启动时逐成员校验后部署 |

五份资产合计 152618486 字节。现有 wheel 来自此前完成的 HCU 构建；本轮已在新Python目录导入，并完成首次JIT及八卡短推理；这不等于完整官方compare或任意新shape已验收。不能用 PyPI Triton 或另一 Python/DTK 版本的 wheel 随意替代。

## FlagTree 来源与补丁

源 commit：`47970467c4d75a9cfffb84426383c7e7a9f2557a`，上游 [FlagTree](https://github.com/flagos-ai/FlagTree)。固定 [源码归档](https://codeload.github.com/flagos-ai/FlagTree/tar.gz/47970467c4d75a9cfffb84426383c7e7a9f2557a) SHA-256：
`31cd442562f0278f4f0f9cd02df3efd2d9c045516e263fc1fcf01bd738aeba48`。

[patch_flagtree_build.py](patch_flagtree_build.py) 保存实际构建修复：尊重 profiler 关闭选项、添加生成 target 的依赖、开发工具 EXCLUDE_FROM_ALL。它接受解包源码目录，不含个人 AFS 路径。manifest 中 archive_sha256 指历史裁剪构建包，不是上游源码 tar，也不是本目录 wheel；不能混用这些 SHA。

wheel SHA-256：
`c7e67f30152702395455ccc440949fecfce5ed4c63a04ede23dfe1802061b714`。
许可原文由 wheel 提取，见 [LICENSE.flagtree](LICENSE.flagtree)。本次默认使用该固定 wheel，bootstrap 不重新编译整套 FlagTree。这里提供来源和补丁，不声称已独立复验 wheel 的源码重建流程。

## JIT 工具链

来源镜像仓库：
`harbor.baai.ac.cn/flagtree/flagtree-hcu-py310-torch2.10.0-dtk26.04-ubuntu22.04`。

来源 layer：
`sha256:630032d42d85494e5c3ea8fc6173fb4ec207d8550d0c456d999b9a584c8f3600`。
manifest 逐个固定 clang-18/lld 二进制 SHA。编译器版本来源以该厂商镜像层为准，尚未单独确定其精确 LLVM 源码 revision。LLVM 18 系列上游许可原文见 [LICENSE.llvm](LICENSE.llvm)，取自 [llvmorg-18.1.8](https://github.com/llvm/llvm-project/blob/llvmorg-18.1.8/llvm/LICENSE.TXT)，不把这一许可文本版本当作厂商编译器源码版本。

该本地归档与旧 AFS ready 归档的封装字节不同，但两份实际 clang/lld 二进制 SHA 相同；新复现仍首次生成 JIT，不能恢复旧内核来替代首次编译。wrapper 仅规范 --version 的 InstalledDir，实际编译仍使用这些二进制与 /opt/dtk。原先单卡预热→八卡preflight/compare分支禁止新JIT；八卡直接部署/短推理和正式直接流程允许首次JIT并保留开销，详见仓库复现说明。不能把某个旧shape的缓存成功当成所有新用例零JIT已通过。

## xmake 来源

来源：[官方 v3.0.3 Release](https://github.com/xmake-io/xmake/releases/download/v3.0.3/xmake-v3.0.3.gz.run)。本地 SHA 与发布 digest 已核对：
`50842cf3a77436bc97fccc7c74935a0ed002bc44afa69cf05fb2b5b8463279d9`。
许可见 [LICENSE.xmake](LICENSE.xmake)，取自相同 tag。此前已在节点从该源码包执行解包、configure、make/install并验证版本。当前优先把官方自包含bundle复制到节点work/xmake/bin/xmake，检查版本和show --help；不能运行时自动回退上述源码构建。bundle已按官方Release digest校验，本轮新节点兼容性仍待实测。它是工具依赖，H3原生对象仍从本轮源码首次构建。

Python 第三方 wheel 的固定来源和各自 metadata 见 [依赖锁](../../requirements/h3_linux_cp311.lock.json)；安装包保留其许可元数据。

官方bundle来源：[v3.0.3 Linux x86_64](https://github.com/xmake-io/xmake/releases/download/v3.0.3/xmake-bundle-v3.0.3.linux.x86_64)，SHA256：`cf4def6e60db1b35d1af4c0064abb069b4b89d90aea3d57b551ef4c697141190`。这只是构建工具，InfiniCore/InfiniLM原生对象仍从源码编译。
