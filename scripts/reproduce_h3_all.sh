#!/usr/bin/env bash
# Complete eight-GPU deployment; rerun in the same checkout/work for incremental builds.
set -euo pipefail
umask 007
TOOLS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export H3_SOURCE_ROOT="$(cd "$TOOLS/.." && pwd)"
export H3_REPRO_WORK="${H3_REPRO_WORK:-$H3_SOURCE_ROOT/.h3-reproduction}"
export H3_BASE_PYTHON="${H3_BASE_PYTHON:-/usr/local/bin/python3}"
export BUILD_JOBS="${BUILD_JOBS:-120}"
export MODEL_PATH="${MODEL_PATH:-/public/home/aclsylqidf/shared/MiniMax-H3}"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0
unset GPU_DEVICE_ORDINAL
env -u PYTHONPATH -u PYTHONHOME "$H3_BASE_PYTHON" "$TOOLS/h3_bootstrap.py" validate-work --work "$H3_REPRO_WORK"
if [ -n "${H3_WHEELHOUSE:-}" ]; then
    "$H3_BASE_PYTHON" -u - "$TOOLS/../requirements/h3_linux_cp311.lock.json" <<'STAGE'
from pathlib import Path
import hashlib,json,os,shutil,sys
source=Path(os.environ["H3_WHEELHOUSE"]).resolve()
dest=Path(os.environ["H3_REPRO_WORK"])/"downloads"
dest.mkdir(parents=True,exist_ok=True)
def valid(p,row):
    return p.is_file() and p.stat().st_size==row["size"] and hashlib.sha256(p.read_bytes()).hexdigest()==row["sha256"]
for row in json.loads(Path(sys.argv[1]).read_text())["wheels"]:
    p=source/row["filename"];q=dest/row["filename"]
    if valid(q,row):
        print("WHEEL_PRESTAGED",q.name,flush=True)
        continue
    if not valid(p,row):
        raise RuntimeError("Missing/corrupt offline wheel: "+str(p))
    if p.resolve()!=q.resolve():
        stage=q.with_suffix(q.suffix+".offline-partial")
        shutil.copy2(p,stage)
        if not valid(stage,row):raise RuntimeError("Wheel copy SHA mismatch: "+q.name)
        stage.replace(q)
    print("WHEEL_OFFLINE_SHA_OK",q.name,flush=True)
STAGE
fi
for phase in prepare build eight-smoke; do
    bash --noprofile --norc "$TOOLS/reproduce_h3.sh" "$phase"
done
printf '[%s] H3_EIGHT_ALL_COMPLETE work=%s\n' "$(date -u +%FT%TZ)" "$H3_REPRO_WORK"
