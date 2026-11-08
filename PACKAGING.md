# 离线 wheel 的准备与发布

更新时间：2026-11-09T00:45:35+08:00（北京时间）。

正式运行直接使用本仓库源码和 `offline-wheels/` 中的依赖，放置和启动方法见 [README](README.md)。

## 哪些文件需要另外提供

| 文件 | 数量与大小 | 分发位置 |
|---|---|---|
| 普通 Python wheel | 55 个，192,118,057 字节（约183MiB） | 配套下载资产；当前尚未发布 Release |
| HCU FlagTree、最小 clang/lld、xmake | 3 份工具，约143MiB | 已在本仓库 `third_party/h3/` |
| 源码、输入图和参考视频 | 见仓库目录 | 已在本仓库 |
| Torch、DTK、共享模型权重 | 由赛事平台提供 | 使用平台现有环境与共享路径 |

只有 55 个普通 wheel 需要另行准备。保持文件原名，直接放在仓库根目录的 `offline-wheels/` 中。`requirements/h3_linux_cp311.lock.json` 记录每个文件的名称、版本、下载地址、字节数和 SHA256。

## 当前获取方法

当前还没有可用的配套 Release 下载入口。可以在联网电脑进入仓库根目录，使用 Python 3.11 或更新版本，按锁定清单下载全部 wheel：

~~~bash
python - <<'WHEELS'
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from h3_bootstrap import fetch

wheelhouse = Path("offline-wheels")
wheelhouse.mkdir(parents=True, exist_ok=True)
lock = json.loads(Path("requirements/h3_linux_cp311.lock.json").read_text())
for wheel in lock["wheels"]:
    fetch(wheel, wheelhouse)
print("离线wheel准备完成：", wheelhouse.resolve())
WHEELS
~~~

这只下载文件，既不在本机安装 Linux wheel，也不编译。再次执行会复用通过校验的文件。下载完后，把源码和这个目录一起上传到 AFS；也可以先上传源码，再把 wheel 放到对应目录。

已有 wheel 时，用下面的命令只做核验：

~~~bash
python - <<'VERIFY'
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from h3_bootstrap import verified, verify_assets

verify_assets()
rows = json.loads(Path("requirements/h3_linux_cp311.lock.json").read_text())["wheels"]
bad = [r["filename"] for r in rows if not verified(Path("offline-wheels") / r["filename"], r)]
if bad:
    raise SystemExit("缺失或校验失败：\n" + "\n".join(bad))
print("55个wheel及3份工具校验通过")
VERIFY
~~~

## 维护者发布 Release

发布时，将锁定清单对应的 55 个 wheel 作为 Release 资产上传，保持原名和原始字节；源码、工具和参考视频继续由仓库提供。发布前核对文件数量、总字节数及逐文件 SHA256，并确认下载后的文件能够通过上述核验。

Release 发布完成后，在 README 中补充真实发布页和下载命令。当前文档不提供未发布的下载地址。`offline-wheels/` 中的二进制不进入 Git 历史。
