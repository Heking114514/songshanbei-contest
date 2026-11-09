#!/usr/bin/env bash
# NODE LOCAL ONLY: no resource request, build, download or AFS write.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
h3_phase6_results="$(python3 - <<'CHECK'
import os
from pathlib import Path
work = Path(os.environ["H3_PHASE6_WORKSPACE"]).resolve()
root = Path(os.environ["H3_PHASE6_LIBRARY_ROOT"]).resolve()
out = Path(os.environ["H3_PHASE6_RESULTS"]).resolve()
afs = os.environ.get("IFAB_AFS_HOME")
if afs and work.is_relative_to(Path(afs).resolve()):
    raise SystemExit("Workspace is inside AFS")
if not root.is_relative_to(work) or not out.is_relative_to(work):
    raise SystemExit("Library/results are outside private workspace")
print(out)
CHECK
)"
mkdir -p "$h3_phase6_results"
h3_phase6_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
(while sleep 30; do printf '[%s] PHASE6_HEARTBEAT\n' "$(TZ=Asia/Shanghai date +%FT%T%:z)"; done) &
h3_phase6_heart=$!
trap 'kill "$h3_phase6_heart" 2>/dev/null || true' EXIT
printf '[%s] PHASE6_START private_workspace_only\n' "$(TZ=Asia/Shanghai date +%FT%T%:z)"
set +e
python3 -u "$h3_phase6_dir/phase6_gemm_probe.py" \
    --workspace-root "$H3_PHASE6_WORKSPACE" \
    --library-root "$H3_PHASE6_LIBRARY_ROOT" \
    --output "$h3_phase6_results/gemm-probe.json" "$@" \
    2>&1 | tee "$h3_phase6_results/gemm-probe.log"
h3_phase6_status=("${PIPESTATUS[@]}")
if (( h3_phase6_status[0] != 0 )); then exit "${h3_phase6_status[0]}"; fi
if (( h3_phase6_status[1] != 0 )); then exit "${h3_phase6_status[1]}"; fi
printf '[%s] PHASE6_SCREENING_COMPLETE model_acceptance=false\n' "$(TZ=Asia/Shanghai date +%FT%T%:z)"
