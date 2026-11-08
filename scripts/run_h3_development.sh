#!/usr/bin/env bash
# Developer validation: incremental build, then single-GPU or TP8 smoke.
set -euo pipefail
umask 007
TOOLS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
usage() {
    cat <<'HELP'
Usage: bash scripts/run_h3_development.sh [all|prepare|build|jit|smoke|eight-smoke]
Default all. H3_DEV_GPUS=1: prepare -> build -> jit -> smoke.
H3_DEV_GPUS=8: prepare -> build -> eight-smoke.
Set H3_REPRO_WORK to node storage, MODEL_PATH to shared weights, and optionally
H3_WHEELHOUSE to the 55 locked wheels. BUILD_JOBS defaults to 120.
This entry produces development smoke results, not submission scores.
HELP
}
phase="${1:-all}"
if [ "$phase" = --help ] || [ "$phase" = -h ]; then usage; exit 0; fi
if [ "$#" -gt 1 ]; then usage >&2; exit 64; fi
case "$phase" in all|prepare|build|jit|smoke|eight-smoke) ;; *) usage >&2; exit 64;; esac
gpus="${H3_DEV_GPUS:-1}"
case "$gpus" in
    1) devices=0 ;;
    8) devices=0,1,2,3,4,5,6,7 ;;
    *) printf 'H3_DEV_GPUS must be 1 or 8.\n' >&2; exit 64 ;;
esac
if { [ "$phase" = jit ] || [ "$phase" = smoke ]; } && [ "$gpus" != 1 ]; then
    printf 'jit/smoke use one GPU; use eight-smoke for TP8.\n' >&2; exit 64
fi
if [ "$phase" = eight-smoke ] && [ "$gpus" != 8 ]; then
    printf 'Set H3_DEV_GPUS=8 for eight-smoke.\n' >&2; exit 64
fi
export H3_SOURCE_ROOT="$(cd "$TOOLS/.." && pwd)"
export H3_REPRO_WORK="${H3_REPRO_WORK:-$H3_SOURCE_ROOT/.h3-development}"
export H3_BASE_PYTHON="${H3_BASE_PYTHON:-/usr/local/bin/python3}"
export BUILD_JOBS="${BUILD_JOBS:-120}"
export MODEL_PATH="${MODEL_PATH:-/public/home/aclsylqidf/shared/MiniMax-H3}"
export FIRST_FRAME="${FIRST_FRAME:-$H3_SOURCE_ROOT/inputs/first_frame.png}"
export HIP_VISIBLE_DEVICES="$devices" CUDA_VISIBLE_DEVICES="$devices" ROCR_VISIBLE_DEVICES="$devices"
export H3_EXPECTED_GPUS="$gpus" H3_GPU_CHECK_MODE=compute
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0
unset GPU_DEVICE_ORDINAL H3_SHORT_SMOKE H3_INFINILM_VIDEO H3_REFERENCE_VIDEO
unset H3_SINGLE_WORK H3_SMOKE_RUN OUTPUT_DIR
env -u PYTHONPATH -u PYTHONHOME "$H3_BASE_PYTHON" "$TOOLS/h3_bootstrap.py" \
    validate-work --work "$H3_REPRO_WORK"
prepare() {
    if [ -n "${H3_WHEELHOUSE:-}" ]; then
        env -u PYTHONPATH -u PYTHONHOME "$H3_BASE_PYTHON" -u "$TOOLS/h3_bootstrap.py" \
            stage-wheels --work "$H3_REPRO_WORK" --wheelhouse "$H3_WHEELHOUSE"
    fi
    bash --noprofile --norc "$TOOLS/reproduce_h3.sh" prepare
}
if [ "$phase" = all ]; then
    prepare
    bash --noprofile --norc "$TOOLS/reproduce_h3.sh" build
    if [ "$gpus" = 1 ]; then
        bash --noprofile --norc "$TOOLS/reproduce_h3.sh" jit
        bash --noprofile --norc "$TOOLS/reproduce_h3.sh" smoke
    else
        bash --noprofile --norc "$TOOLS/reproduce_h3.sh" eight-smoke
    fi
elif [ "$phase" = prepare ]; then
    prepare
else
    bash --noprofile --norc "$TOOLS/reproduce_h3.sh" "$phase"
fi
printf '[%s] H3_DEVELOPMENT_COMPLETE phase=%s gpus=%s work=%s\n' \
    "$(date -u +%FT%TZ)" "$phase" "$gpus" "$H3_REPRO_WORK"
