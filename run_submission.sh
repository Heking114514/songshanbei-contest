#!/usr/bin/env bash
set -euo pipefail
umask 007
mode=native
if [ "$#" -gt 1 ]; then printf 'Usage: bash run_submission.sh [check]\n' >&2; exit 64; fi
case "${1:-}" in
    "") ;;
    check) mode=check;;
    -h|--help) printf 'Usage: bash run_submission.sh [check]\nDefault: full native generation and AFS save. check: also compare against the bundled reference.\n'; exit 0;;
    *) printf 'Usage: bash run_submission.sh [check]\n' >&2; exit 64;;
esac
: "${IFAB_AFS_HOME:=${HOME:?Personal AFS home is unavailable}}"
export IFAB_AFS_HOME
export MODEL_PATH="${MODEL_PATH:-/public/home/aclsylqidf/shared/MiniMax-H3}"
export H3_AFS_OUTPUT_DIR="${H3_AFS_OUTPUT_DIR:-$IFAB_AFS_HOME/h3-results/run-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
export BUILD_JOBS="${BUILD_JOBS:-120}"
export H3_BASE_PYTHON="${H3_BASE_PYTHON:-/usr/local/bin/python3}"
export H3_DELIVERY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export H3_WHEELHOUSE="${H3_WHEELHOUSE:-$H3_DELIVERY_DIR/offline-wheels}"
stage_args=(--source "$H3_DELIVERY_DIR" --wheelhouse "$H3_WHEELHOUSE")
if [ -n "${H3_NODE_ROOT:-}" ]; then stage_args+=(--node "$H3_NODE_ROOT"); fi
H3_NODE_ROOT="$(env -u PYTHONPATH -u PYTHONHOME "$H3_BASE_PYTHON" -u \
    "$H3_DELIVERY_DIR/scripts/h3_source_stage.py" "${stage_args[@]}")"
export H3_NODE_ROOT
# Both ZIP and directory inputs are staged exactly once on node storage.
export H3_WHEELHOUSE="$H3_NODE_ROOT/work/downloads"
printf 'H3_SOURCE_DIR=%s\nH3_NODE_ROOT=%s\nH3_AFS_OUTPUT_DIR=%s\n' \
    "$H3_DELIVERY_DIR" "$H3_NODE_ROOT" "$H3_AFS_OUTPUT_DIR"
exec bash --noprofile --norc "$H3_NODE_ROOT/source/scripts/h3_submission_common.sh" "$mode"
