#!/usr/bin/env bash
# Runs on an already prepared isolated node. No allocation/build/AFS operations.
# H3_PHASE5_LIBRARY_ROOT and H3_PHASE5_RESULTS must be set explicitly.
set -Eeuo pipefail
: "${H3_PHASE5_LIBRARY_ROOT:?Set the isolated native library root}"
: "${H3_PHASE5_RESULTS:?Set a node-local results directory}"
phase5_scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p -- "$H3_PHASE5_RESULTS"
phase5_extra=()
case "${1:-}" in
    "") ;;
    --real-size) phase5_extra=(--real-size) ;;
    *) printf 'Usage: phase5_matrix.sh [--real-size]\n' >&2; exit 2 ;;
esac
phase5_heartbeat_pid=""
phase5_cleanup() {
    if [[ -n "$phase5_heartbeat_pid" ]]; then
        kill "$phase5_heartbeat_pid" 2>/dev/null || true
        wait "$phase5_heartbeat_pid" 2>/dev/null || true
    fi
}
trap phase5_cleanup EXIT
phase5_run() {
    local phase5_name="$1"
    shift
    printf '[%s] PHASE5_STAGE_START %s\n' "$(date -u +%FT%TZ)" "$phase5_name"
    (
        while sleep 30; do
            printf '[%s] PHASE5_HEARTBEAT %s\n' "$(date -u +%FT%TZ)" "$phase5_name"
        done
    ) &
    phase5_heartbeat_pid="$!"
    # PIPESTATUS is handled by pipefail; do not hide a probe error behind tee.
    python -u "$@" --library-root "$H3_PHASE5_LIBRARY_ROOT" \
        --output "$H3_PHASE5_RESULTS/$phase5_name.json" \
        2>&1 | tee "$H3_PHASE5_RESULTS/$phase5_name.log"
    phase5_cleanup
    phase5_heartbeat_pid=""
    printf '[%s] PHASE5_STAGE_COMPLETE %s\n' "$(date -u +%FT%TZ)" "$phase5_name"
}
# Strict gates first. Every stage runs in a fresh Python process.
phase5_run phase4-native "$phase5_scripts/phase4_probe.py" --group native "${phase5_extra[@]}"
phase5_run phase5-rms-strict "$phase5_scripts/phase5_probe.py" --group rms --blocks 0 "${phase5_extra[@]}"
phase5_run phase5-qk-strict "$phase5_scripts/phase5_probe.py" --group qk "${phase5_extra[@]}"
phase5_run phase4-video "$phase5_scripts/phase4_probe.py" --group video
# Explicitly different reduction policies are only screened after strict gates.
phase5_run phase5-rms-tuning "$phase5_scripts/phase5_probe.py" --group rms --blocks 128,256,512,1024 "${phase5_extra[@]}"
python - "$H3_PHASE5_RESULTS" <<'PY_SUMMARY'
import json,sys
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
root=Path(sys.argv[1])
names=["phase4-native","phase5-rms-strict","phase5-qk-strict","phase4-video","phase5-rms-tuning"]
reports={name:json.loads((root/(name+".json")).read_text()) for name in names}
assert all(row["complete"] for row in reports.values())
summary={"finished_at":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
         "complete":True,"stages":{k:len(v["cases"]) for k,v in reports.items()},
         "scope":"local probes only; real DiT/TP8/media acceptance still required"}
(root/"matrix-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print("PHASE5_MATRIX_COMPLETE",json.dumps(summary),flush=True)
PY_SUMMARY
