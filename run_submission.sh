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
: "${IFAB_AFS_HOME:?Set personal AFS root}"
: "${MODEL_PATH:?Set platform shared MiniMax-H3 model directory}"
: "${H3_AFS_OUTPUT_DIR:?Set a new AFS result directory}"
: "${H3_AFS_INFO_JSON:?Set fresh GET /api/persistent-storage JSON after upload}"
: "${H3_NODE_ROOT:?Set node storage outside AFS and workspace; see REPRODUCE.md}"
export H3_BASE_PYTHON="${H3_BASE_PYTHON:-/usr/local/bin/python3}"
export H3_DELIVERY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export H3_WHEELHOUSE="${H3_WHEELHOUSE:-$H3_DELIVERY_DIR/offline-wheels}"
export H3_NODE_ROOT
H3_NODE_ROOT="$(env -u PYTHONPATH -u PYTHONHOME "$H3_BASE_PYTHON" -u \
    "$H3_DELIVERY_DIR/scripts/h3_source_stage.py" --source "$H3_DELIVERY_DIR" \
    --wheelhouse "$H3_WHEELHOUSE" --node "$H3_NODE_ROOT")"
exec bash --noprofile --norc "$H3_NODE_ROOT/source/scripts/h3_submission_common.sh" "$mode"
