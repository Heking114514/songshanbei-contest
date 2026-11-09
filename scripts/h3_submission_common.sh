#!/usr/bin/env bash
# Internal worker for run_submission.sh [check].
set -euo pipefail
umask 007
mode="${1:?native or check}"
case "$mode" in native|check) ;; *) exit 64;; esac
TOOLS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export H3_SOURCE_ROOT="$(cd "$TOOLS/.." && pwd)"
export H3_REPRO_WORK="$H3_NODE_ROOT/work"
export H3_WHEELHOUSE="${H3_WHEELHOUSE:-$H3_DELIVERY_DIR/offline-wheels}"
export BUILD_JOBS="${BUILD_JOBS:-120}"
export FIRST_FRAME="${FIRST_FRAME:-$H3_SOURCE_ROOT/inputs/first_frame.png}"
export H3_EXPECTED_GPUS=8 H3_FORBID_JIT_COMPILATION=0 H3_GPU_CHECK_MODE=inventory
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0
if [ -n "${H3_SHORT_SMOKE:-}" ] && [ "$H3_SHORT_SMOKE" != 0 ]; then
    printf 'Formal submission refuses H3_SHORT_SMOKE\n' >&2; exit 64
fi
if [ -n "${H3_INFINILM_VIDEO:-}" ] || [ -n "${H3_REFERENCE_VIDEO:-}" ]; then
    printf 'Formal submission generates new native and uses the bundled reference for check; unset provided video variables\n' >&2; exit 64
fi
unset GPU_DEVICE_ORDINAL H3_SHORT_SMOKE H3_SINGLE_WORK H3_SMOKE_RUN OUTPUT_DIR
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
export H3_RUN_DIR="$H3_NODE_ROOT/runs/$mode-$run_id"
mkdir -p "$H3_RUN_DIR"
export H3_SUBMISSION_MODE="$mode"
export H3_SUBMISSION_INVOCATION="$H3_RUN_DIR/invocation.json"
export OUTPUT_DIR="$H3_RUN_DIR/compare"
export H3_NATIVE_VIDEO="$H3_RUN_DIR/result.mp4"
export H3_ATTENTION_BUNDLE_DIR="$H3_RUN_DIR/attention-bundle"
env -u PYTHONPATH -u PYTHONHOME "$H3_BASE_PYTHON" -u "$TOOLS/h3_attention_bundle.py" verify
if [ "$mode" = check ]; then
    export H3_REFERENCE_VIDEO="$H3_DELIVERY_DIR/reference/reference_fl2va.mp4"
    env -u PYTHONPATH -u PYTHONHOME "$H3_BASE_PYTHON" -u "$TOOLS/h3_submission_record.py" verify-reference
fi
env -u PYTHONPATH -u PYTHONHOME "$H3_BASE_PYTHON" -u "$TOOLS/h3_delivery.py" reserve --run "$H3_RUN_DIR"
worker() (
    set -euo pipefail
    heartbeat() {
        tick=""
        trap 'if [ -n "$tick" ]; then kill "$tick" 2>/dev/null || true; fi; exit 0' TERM INT
        while :; do
            sleep 30 & tick=$!
            wait "$tick" || exit 0
            tick=""
            printf '[%s] H3_FORMAL_HEARTBEAT mode=%s run=%s\n' "$(date -u +%FT%TZ)" "$mode" "$H3_RUN_DIR"
        done
    }
    heartbeat & heart=$!
    trap 'kill "$heart" 2>/dev/null || true; wait "$heart" 2>/dev/null || true' EXIT
    printf '[%s] H3_FORMAL_BEGIN mode=%s jobs=%s\n' "$(date -u +%FT%TZ)" "$mode" "$BUILD_JOBS"
    bash --noprofile --norc "$TOOLS/reproduce_h3.sh" prepare
    export H3_BUILD_AUDIT_OUTPUT="$H3_RUN_DIR/origins.json"
    bash --noprofile --norc "$TOOLS/reproduce_h3.sh" build
    source "$TOOLS/h3_repro_env.sh"
    while IFS= read -r variable; do unset "$variable"; done < <(env | sed -n 's/^\(H3_EXPERIMENT_[A-Za-z0-9_]*\)=.*/\1/p')
    python -u "$TOOLS/h3_attention_bundle.py" deploy --output "$H3_ATTENTION_BUNDLE_DIR"
    export H3_EXPERIMENT_FLASH_GFX936=1
    export H3_FLASH_LIBRARY="$H3_ATTENTION_BUNDLE_DIR/libh3_flash_dispatch.so"
    export H3_EXPERIMENT_RMS_ADALN=1 H3_EXPERIMENT_GATE_RMS_ADALN=1
    export H3_EXPERIMENT_QK_NORM_ROPE=1 H3_EXPERIMENT_ADDCMUL_ROWS=1
    export H3_EXPERIMENT_ADALN_INPLACE=1 H3_EXPERIMENT_ADALN_BATCH=1
    export H3_EXPERIMENT_ROPE_INPLACE=1 H3_EXPERIMENT_VAE_DEVICES=4
    export H3_EXPERIMENT_MHA_QUERY_CHUNK=0 H3_EXPERIMENT_MHA_HEAD_BATCH=0
    export H3_EXPERIMENT_SWIGLU_ROWS=0 H3_EXPERIMENT_SNAKE_PREPARED=0
    export H3_EXPERIMENT_SNAKE_STRIDED=0 H3_EXPERIMENT_ADALN_INDEXED=0
    export H3_EXPERIMENT_FUSED_RMS_BLOCK=0 H3_EXPERIMENT_VIDEO_POSTPROCESS=0
    export H3_EXPERIMENT_DETERMINISTIC_BLAS=0 H3_EXPERIMENT_GEMM_LT=0
    export H3_EXPERIMENT_TP_OVERLAP_ROWS=0
    python -u "$TOOLS/h3_submission_record.py" begin
    export H3_COMPARE_BF16_MATH=1
    export PYTHONPATH="$TOOLS/h3_compare_hook:$TOOLS:${PYTHONPATH:-}"
    # Full native generation exactly once. First-use JIT is part of inference.
    H3_PREFLIGHT_OUTPUT="$H3_RUN_DIR/preflight.json" \
    python -u "$H3_SOURCE_ROOT/InfiniLM/examples/minimax_h3.py" fl2va \
        --model-path "$MODEL_PATH" --first-frame "$FIRST_FRAME" --text "A dog falls asleep" \
        --gpus 0,1,2,3,4,5,6,7 --steps 50 --seed 42 --num-frames 124 \
        --output "$H3_NATIVE_VIDEO" 2>&1 | tee "$H3_RUN_DIR/generation.log"
    python -u "$TOOLS/h3_submission_record.py" native-check
    if [ "$mode" = check ]; then
        # Both provided videos: official comparison performs no generation.
        export H3_INFINILM_VIDEO="$H3_NATIVE_VIDEO"
        bash --noprofile --norc "$TOOLS/run_h3_compare.sh"
        python -u "$TOOLS/h3_submission_record.py" check
    fi
)
set +e
worker 2>&1 | tee "$H3_RUN_DIR/workflow.log"
codes=("${PIPESTATUS[@]}")
set -e
rc="${codes[0]}"
if [ "$rc" -eq 0 ] && [ "${codes[1]}" -ne 0 ]; then rc="${codes[1]}"; fi
# Stdlib receipt also covers prepare/build failures. Save bounded failure evidence.
set +e
env -u PYTHONPATH -u PYTHONHOME "$H3_BASE_PYTHON" -u "$TOOLS/h3_submission_record.py" status --rc "$rc"
status_rc=$?
if [ "$rc" -eq 0 ] && [ "$status_rc" -ne 0 ]; then rc="$status_rc"; fi
env -u PYTHONPATH -u PYTHONHOME "$H3_BASE_PYTHON" -u "$TOOLS/h3_delivery.py" persist --run "$H3_RUN_DIR" --rc "$rc"
save_rc=$?
set -e
printf '[%s] H3_FORMAL_END mode=%s pipeline_rc=%s save_rc=%s node=%s afs=%s\n' \
    "$(date -u +%FT%TZ)" "$mode" "$rc" "$save_rc" "$H3_RUN_DIR" "$H3_AFS_OUTPUT_DIR"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
exit "$save_rc"
