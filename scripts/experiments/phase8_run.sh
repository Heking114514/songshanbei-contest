#!/usr/bin/env bash
# Future node wrapper. No API calls, resource requests, restore, or AFS writes.
h3_phase8_run_main() {
    if [[ $# -eq 0 || "$1" == "--help" ]]; then
        cat <<'HELP'
Usage: H3_PHASE8_LOG_DIR=NEW_PRIVATE_DIR phase8_run.sh MODE [python arguments]
Modes: attention | pipeline | replay-audio | replay-vae
Pass --workspace-root PRIVATE_ROOT and all arguments required by the selected tool.
Attention defaults to plan-only; --execute is required for a future GPU sweep.
H3_PHASE8_PYTHON selects the prepared node Python (default python3).
Every run uses a new log directory, live tee, 15-second heartbeat and both exit codes.
This wrapper never requests a node, restores a cache, builds native targets or writes AFS.
HELP
        return 0
    fi
    local mode="$1" script
    shift
    case "$mode" in
        attention) script=phase8_attention_tune.py ;;
        pipeline) script=phase8_pipeline.py ;;
        replay-audio) script=phase8_replay_audio.py ;;
        replay-vae) script=phase8_replay_vae.py ;;
        *) printf 'Unknown phase8 mode: %s\n' "$mode" >&2; return 2 ;;
    esac
    local scripts_dir python_bin log_dir
    scripts_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || return
    python_bin="${H3_PHASE8_PYTHON:-python3}"
    log_dir="${H3_PHASE8_LOG_DIR:-}"
    if [[ -z "$log_dir" ]]; then
        printf 'H3_PHASE8_LOG_DIR must be a new private directory\n' >&2
        return 2
    fi
    "$python_bin" - "$scripts_dir" "$log_dir" "$@" <<'PY_CHECK' || return
import argparse
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from phase8_trace import private_output
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--workspace-root", type=Path, required=True)
known, _ = parser.parse_known_args(sys.argv[3:])
directory = private_output(known.workspace_root, Path(sys.argv[2]))
directory.mkdir(parents=True, exist_ok=False)
print("PHASE8_LOG_DIR", directory, flush=True)
PY_CHECK
    (
        sleeper=""
        trap 'if [[ -n "$sleeper" ]]; then kill "$sleeper" 2>/dev/null || true; wait "$sleeper" 2>/dev/null || true; fi' EXIT
        trap 'exit 0' TERM INT
        while true; do
            sleep 15 & sleeper=$!
            wait "$sleeper" || exit 0
            sleeper=""
            printf 'PHASE8_HEARTBEAT %s %s\n' "$mode" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        done
    ) &
    local heartbeat=$!
    trap 'kill "$heartbeat" 2>/dev/null || true; wait "$heartbeat" 2>/dev/null || true' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    # Run in this function with errexit disabled, to preserve both pipeline exit codes.
    local -a codes
    "$python_bin" -u "$scripts_dir/$script" "$@" 2>&1 | tee "$log_dir/run.log"
    codes=("${PIPESTATUS[@]}")
    kill "$heartbeat" 2>/dev/null || true
    wait "$heartbeat" 2>/dev/null || true
    trap - EXIT INT TERM
    printf 'PHASE8_EXIT python=%s tee=%s\n' "${codes[0]}" "${codes[1]}" | tee "$log_dir/exit.log"
    if [[ "${codes[0]}" -ne 0 ]]; then return "${codes[0]}"; fi
    return "${codes[1]}"
}
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set +e
    h3_phase8_run_main "$@"
    exit $?
fi
