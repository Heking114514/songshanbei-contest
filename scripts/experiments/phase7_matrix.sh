#!/usr/bin/env bash
# Node-local validation wrapper only; no resource submission, upload or AFS writes.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
if (( $# < 1 )); then printf '%s\n' 'usage: phase7_matrix.sh flash|overlap ARGS' >&2; exit 2; fi
h3_phase7_group="$1"
shift
h3_phase7_results="$(python3 - <<'CHECK'
import os
from pathlib import Path
work=Path(os.environ["H3_PHASE7_WORKSPACE"]).resolve()
library=Path(os.environ["H3_PHASE7_LIBRARY_ROOT"]).resolve()
results=Path(os.environ["H3_PHASE7_RESULTS"]).resolve()
afs=os.environ.get("IFAB_AFS_HOME")
if afs and work.is_relative_to(Path(afs).resolve()):
    raise SystemExit("Workspace is inside AFS")
if not library.is_relative_to(work) or not results.is_relative_to(work):
    raise SystemExit("Library/results outside private workspace")
probe=os.environ.get("H3_PHASE7_OVERLAP_PROBE")
if probe and not Path(probe).resolve().is_relative_to(work):
    raise SystemExit("Overlap probe outside private workspace")
print(results)
CHECK
)"
mkdir -p "$h3_phase7_results"
h3_phase7_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
case "$h3_phase7_group" in
    flash)
        h3_phase7_command=(python3 -u "$h3_phase7_dir/phase7_flash_probe.py"
            --workspace-root "$H3_PHASE7_WORKSPACE"
            --library-root "$H3_PHASE7_LIBRARY_ROOT"
            --output "$h3_phase7_results/flash.json" "$@")
        ;;
    overlap)
        if [[ -z "${H3_PHASE7_OVERLAP_PROBE:-}" || ! -x "$H3_PHASE7_OVERLAP_PROBE" ]]; then
            printf '%s\n' 'Set H3_PHASE7_OVERLAP_PROBE to the already compiled private executable' >&2
            exit 2
        fi
        h3_phase7_command=("$H3_PHASE7_OVERLAP_PROBE" "$@")
        ;;
    *) printf '%s\n' 'Unknown validation group' >&2; exit 2;;
esac
(while sleep 30; do printf '[%s] PHASE7_HEARTBEAT %s\n' "$(TZ=Asia/Shanghai date +%FT%T%:z)" "$h3_phase7_group"; done) &
h3_phase7_heart=$!
trap 'kill "$h3_phase7_heart" 2>/dev/null || true' EXIT
printf '[%s] PHASE7_BEGIN %s\n' "$(TZ=Asia/Shanghai date +%FT%T%:z)" "$h3_phase7_group"
set +e
"${h3_phase7_command[@]}" 2>&1 | tee "$h3_phase7_results/$h3_phase7_group.log"
h3_phase7_status=("${PIPESTATUS[@]}")
if (( h3_phase7_status[0] != 0 )); then exit "${h3_phase7_status[0]}"; fi
if (( h3_phase7_status[1] != 0 )); then exit "${h3_phase7_status[1]}"; fi
printf '[%s] PHASE7_COMPLETE %s model_acceptance=false\n' "$(TZ=Asia/Shanghai date +%FT%T%:z)" "$h3_phase7_group"
