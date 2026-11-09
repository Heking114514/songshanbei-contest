# 离线 wheel 的准备与发布

更新时间：2026-11-09T11:12:27+08:00（北京时间）。

正式运行使用本仓库源码和仓库根目录的 `h3-offline-wheels-linux-cp311.zip`，脚本自动解压到节点盘，放置和启动方法见 [README](README.md)。

## 哪些文件需要另外提供

| 文件 | 数量与大小 | 分发位置 |
|---|---|---|
| 普通 Python wheel | 55 个，192,118,057 字节（约183MiB） | [offline-wheel Release](https://github.com/Heking114514/songshanbei-contest/releases/tag/offline-wheel) |
| HCU FlagTree、最小 clang/lld、xmake | 4 份工具，约145MiB | 已在本仓库 `third_party/h3/` |
| 源码、输入图和参考视频 | 见仓库目录 | 已在本仓库 |
| Torch、DTK、共享模型权重 | 由赛事平台提供 | 使用平台现有环境与共享路径 |

只有 55 个普通 wheel 需要另行准备。推荐以ZIP原名放在仓库根目录，由脚本自动解压；也兼容原名放在 `offline-wheels/` 的散文件。`requirements/h3_linux_cp311.lock.json` 记录每个文件的名称、版本、下载地址、字节数；历史 SHA256 字段保留作资产记录，下载和运行时不读取或比对它。

## 当前获取方法

推荐从[offline-wheel Release](https://github.com/Heking114514/songshanbei-contest/releases/tag/offline-wheel)直接下载原名ZIP。在联网电脑的**源码仓库根目录（run_submission.sh所在目录）**，也可执行：

~~~bash
curl -fL --retry 3 -C - -o h3-offline-wheels-linux-cp311.zip \
  "https://github.com/Heking114514/songshanbei-contest/releases/download/offline-wheel/h3-offline-wheels-linux-cp311.zip"
~~~

把源码目录连同这个ZIP上传AFS后，按README一行启动，无需解压。以下散文件下载方式仅供维护或已有wheel时使用；使用Python3.11或更新版本，在源码仓库根目录运行：

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
print("55个wheel及4份工具存在且大小完整")
VERIFY
~~~

## 维护者发布 Release

发布时，上传压缩包 `h3-offline-wheels-linux-cp311.zip` 即可，无需上传 `SHA256SUMS`。压缩包内为 `offline-wheels/` 目录及锁定的 55 个 wheel，保持原名和原始字节；源码、工具和参考视频继续由仓库提供。发布前核对文件数量、大小和 ZIP 能否完整解压，并确认解压后的文件能够通过上述完整性检查。

使用者将 ZIP 原名放到个人AFS的源码仓库根目录，与 `run_submission.sh` 同级。**不需要本地解压，也不需要服务器手工解压。** 申请八卡后按README一行启动，脚本直接读取ZIP并解压到节点 `work/downloads/`，按清单核对成员、数量和字节数，解压时检查ZIP自身CRC；不执行SHA256检查。AFS仅保留ZIP，不再额外展开一份wheel。

已准备好散文件的用户仍可上传 `offline-wheels/`，与ZIP方式二选一。同时存在且目录中有wheel时优先使用该目录；只有README的目录不影响ZIP识别。ZIP中的目录结构必须是 `offline-wheels/各个文件.whl`，不能再多嵌套一级；缺文件、成员大小不符或无法解压时脚本在安装前报错并给出文件名。

上述Release及原名资产已于2026-11-09核实存在。`offline-wheels/` 中的二进制不进入 Git 历史。
