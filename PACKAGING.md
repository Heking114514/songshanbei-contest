# 从源码制作上传交付包

更新时间：2026-11-08T23:32:04+08:00（北京时间）。

本文供制作交付包的人使用。运行者拿到交付目录后，按照 [README](README.md) 上传和运行即可。

## 准备依赖并打包

进入克隆得到的源码仓库，即能看到 `run_submission.sh`、`InfiniCore/` 和 `InfiniLM/` 的目录，再执行本节命令。这里将创建原先不存在的 `h3-wheelhouse/` 和 `h3-submission/`。若已收到制作好的交付包，可直接按README上传并运行。制包电脑需要 Python 3.11 或更新版本。

55个 Python wheel 放在仓库同级的 `h3-wheelhouse/` 中，直接放文件，不再套一层目录；文件名、大小和 SHA256 必须符合 `requirements/h3_linux_cp311.lock.json`。三份构建工具已经随源码放在 `third_party/h3/`。

~~~text
工作目录/
├── 源码仓库/                       # 你克隆的目录，名称以实际为准
│   ├── run_submission.sh
│   ├── scripts/
│   ├── requirements/h3_linux_cp311.lock.json
│   ├── third_party/h3/
│   │   ├── flagtree-0.7.0+hcu3.6-cp311-cp311-linux_x86_64.whl
│   │   ├── hcu-jit-clang18-minimal.tar.gz
│   │   └── xmake-v3.0.3.gz.run
│   ├── inputs/first_frame.png
│   └── reference/
│       ├── reference_fl2va.mp4
│       ├── case.json
│       └── README.md
├── h3-wheelhouse/                   # 55个锁定wheel，约183MiB
└── h3-submission/                   # 下一条打包命令生成
~~~

已有55个wheel时直接放入 `../h3-wheelhouse/`。需要下载时，下面的命令按依赖锁中的地址下载并核验；再次运行会复用已通过校验的文件：

~~~bash
python - <<'WHEELS'
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from h3_bootstrap import fetch

wheelhouse = Path("../h3-wheelhouse")
wheelhouse.mkdir(parents=True, exist_ok=True)
lock = json.loads(Path("requirements/h3_linux_cp311.lock.json").read_text())
for wheel in lock["wheels"]:
    fetch(wheel, wheelhouse)
print("离线wheel准备完成：", wheelhouse.resolve())
WHEELS
~~~

在同一仓库根目录制作交付包：

~~~bash
python scripts/h3_delivery.py package \
  --repo . \
  --wheelhouse ../h3-wheelhouse \
  --output ../h3-submission
~~~

`../h3-submission` 必须是尚不存在的新目录。命令会校验依赖，合并源码和离线wheel，生成最多96MiB一片的 `payload.part*`，以及上传清单 `UPLOAD.md` 和校验清单 `delivery-manifest.json`。

生成的交付目录已包含运行所需的源码、55个wheel和三份构建工具；上传时不必另传 `h3-wheelhouse/`。模型权重使用平台共享目录，不放进交付包。


打包完成后，将生成的整个目录提供给运行者。目录名由 --output 参数决定；本文统一使用 h3-submission。
