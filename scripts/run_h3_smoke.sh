#!/usr/bin/env bash
set -euo pipefail
umask 007
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "${H3_ENV_READY:-0}" != 1 ]; then
    source "$REPO_ROOT/scripts/env_hygon.sh"
fi
export H3_SOURCE_ROOT="$REPO_ROOT" H3_SHORT_SMOKE=1
: "${MODEL_PATH:?Set MODEL_PATH to the shared MiniMax-H3 directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a persistent output directory}"
FIRST_FRAME="${FIRST_FRAME:-$REPO_ROOT/inputs/first_frame.png}"
mkdir -p "$OUTPUT_DIR"
python -u "$REPO_ROOT/scripts/h3_inference_bf16.py" fl2va \
    --model-path "$MODEL_PATH" --first-frame "$FIRST_FRAME" \
    --text 'A dog falls asleep' --gpus 0,1,2,3,4,5,6,7 \
    --num-frames 22 --steps 2 --seed 42 --height 256 --width 256 \
    --output "$OUTPUT_DIR/smoke.mp4" 2>&1 | tee "$OUTPUT_DIR/smoke.log"
