#!/usr/bin/env bash
set -euo pipefail
umask 007
TOOLS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export H3_SOURCE_ROOT="${H3_SOURCE_ROOT:-$(cd "$TOOLS/.." && pwd)}"
if [ "${H3_ENV_READY:-0}" != 1 ]; then
    source "$H3_SOURCE_ROOT/scripts/env_hygon.sh"
fi
: "${MODEL_PATH:?Set MODEL_PATH to the shared MiniMax-H3 directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a NEW node-local directory}"
export FIRST_FRAME="${FIRST_FRAME:-$H3_SOURCE_ROOT/inputs/first_frame.png}"
test ! -e "$OUTPUT_DIR" || { printf 'Refusing existing OUTPUT_DIR: %s\n' "$OUTPUT_DIR" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
export OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
export H3_REFERENCE_MAX_HEADS="${H3_REFERENCE_MAX_HEADS:-0}"
export H3_COMPARE_BF16_MATH="${H3_COMPARE_BF16_MATH:-0}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TOOLS/h3_compare_hook:$TOOLS:${PYTHONPATH:-}"
finish() {
    rc=$?
    trap - EXIT
    printf '{"rc":%s}\n' "$rc" > "$OUTPUT_DIR/status.json"
    printf 'H3_COMPARE_WRAPPER_END rc=%s output=%s\n' "$rc" "$OUTPUT_DIR"
    exit "$rc"
}
trap finish EXIT
python -u "$TOOLS/h3_preflight.py" --source-root "$H3_SOURCE_ROOT" \
    --model-path "$MODEL_PATH" --first-frame "$FIRST_FRAME" \
    --output "$OUTPUT_DIR/preflight.json" 2>&1 | tee "$OUTPUT_DIR/preflight.log"
provided=()
if [ -n "${H3_INFINILM_VIDEO:-}" ]; then
    test -s "$H3_INFINILM_VIDEO"
    provided+=(--infinilm-fl2va-video "$H3_INFINILM_VIDEO")
fi
if [ -n "${H3_REFERENCE_VIDEO:-}" ]; then
    test -s "$H3_REFERENCE_VIDEO"
    provided+=(--reference-fl2va-video "$H3_REFERENCE_VIDEO")
fi
python -u "$H3_SOURCE_ROOT/InfiniLM/examples/minimax_h3_compare.py" \
    --tasks fl2va --model-path "$MODEL_PATH" --first-frame "$FIRST_FRAME" \
    --infinilm-gpus 0,1,2,3,4,5,6,7 --legacy-gpus 0,1,2,3,4,5,6,7 \
    --legacy-text-gpus 0,1,2,3 --legacy-transformer-gpus 4,5,6,7 \
    --legacy-memory-fraction 0.4 \
    "${provided[@]}" --output-dir "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/compare.log"
