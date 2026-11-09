#!/usr/bin/env bash
# Node-side phases. Invoke the complete remote payload through Git Bash/base64.
set -euo pipefail
umask 007
TOOLS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export H3_SOURCE_ROOT="$(cd "$TOOLS/.." && pwd)"
export H3_REPRO_WORK="${H3_REPRO_WORK:-$H3_SOURCE_ROOT/.h3-reproduction}"
export H3_BASE_PYTHON="${H3_BASE_PYTHON:-/usr/local/bin/python3}"
export BUILD_JOBS="${BUILD_JOBS:-120}"
usage() {
    printf 'Usage: bash scripts/reproduce_h3.sh prepare|build|jit|smoke|eight-smoke|preflight|compare\n'
    printf 'Single GPU: prepare -> build -> jit -> smoke. Eight GPUs: preflight -> compare.\n'
}
if [ "${1:-}" = --help ] || [ "$#" -eq 0 ]; then usage; exit 0; fi
worker() {
    case "$1" in
        prepare)
            unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
            export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
            export XMAKE_GLOBALDIR="$H3_REPRO_WORK/xmake-global"
            unset XMAKE_PROGRAM_DIR
            "$H3_BASE_PYTHON" -u "$TOOLS/h3_bootstrap.py" prepare --work "$H3_REPRO_WORK"
            return
            ;;
    esac
    source "$TOOLS/h3_repro_env.sh"
    export MODEL_PATH="${MODEL_PATH:-/public/home/aclsylqidf/shared/MiniMax-H3}"
    export FIRST_FRAME="${FIRST_FRAME:-$H3_SOURCE_ROOT/inputs/first_frame.png}"
    mkdir -p "$H3_REPRO_WORK/reports"
    case "$1" in
        build)
            export BUILD_SKIP_CONFIGURE=0 BUILD_VERBOSE=1 BUILD_IMPORT_CHECK=0
            bash --noprofile --norc "$TOOLS/build_hygon.sh"
            python -u "$TOOLS/h3_bootstrap.py" audit --work "$H3_REPRO_WORK" \
                --output "${H3_BUILD_AUDIT_OUTPUT:-$H3_REPRO_WORK/reports/build-origins.json}"
            ;;
        jit)
            export H3_FORBID_JIT_COMPILATION=0
            python -u "$TOOLS/validate_snake_8card.py" --gpus 0 \
                --output "$H3_REPRO_WORK/reports/snake-first-jit.json"
            export H3_FORBID_JIT_COMPILATION=1
            python -u "$TOOLS/validate_snake_8card.py" --gpus 0 \
                --output "$H3_REPRO_WORK/reports/snake-cached.json"
            ;;
        smoke)
            export H3_EXPECTED_GPUS=1 H3_FORBID_JIT_COMPILATION=0
            python -u "$TOOLS/h3_preflight.py" --source-root "$H3_SOURCE_ROOT" \
                --model-path "$MODEL_PATH" --first-frame "$FIRST_FRAME" --expected-gpus 1 \
                --output "$H3_REPRO_WORK/reports/preflight-single.json"
            export H3_SINGLE_WORK="${H3_SINGLE_WORK:-$H3_REPRO_WORK/single-smoke-$(date -u +%Y%m%dT%H%M%S)-$$}"
            test ! -e "$H3_SINGLE_WORK/smoke-result.json" || {
                printf 'Existing smoke result; inspect it before rerunning.\n' >&2; return 64;
            }
            mkdir -p "$H3_SINGLE_WORK"
            python -u "$TOOLS/h3_compare_single_smoke.py" prepare
            export H3_SMOKE_RUN=smoke
            python -u "$TOOLS/h3_compare_single_smoke.py" compare
            export H3_FORBID_JIT_COMPILATION=1 H3_SMOKE_RUN=cached-smoke
            python -u "$TOOLS/h3_compare_single_smoke.py" compare
            cp "$H3_SINGLE_WORK/cached-smoke-result.json" "$H3_REPRO_WORK/reports/single-ready.json"
            ;;
        eight-smoke)
            bash --noprofile --norc "$TOOLS/h3_eight_smoke.sh"
            ;;
        preflight|compare)
            export H3_EXPECTED_GPUS=8 H3_FORBID_JIT_COMPILATION=1
            unset H3_SHORT_SMOKE H3_INFINILM_VIDEO H3_REFERENCE_VIDEO
            test -s "$H3_REPRO_WORK/reports/single-ready.json" || {
                printf 'Complete single-GPU smoke including the no-JIT replay first.\n' >&2; return 65;
            }
            python -u "$TOOLS/h3_bootstrap.py" rebase-cache --work "$H3_REPRO_WORK"
            python -u "$TOOLS/h3_bootstrap.py" audit --work "$H3_REPRO_WORK" \
                --output "$H3_REPRO_WORK/reports/eight-origins.json"
            python -u "$TOOLS/validate_snake_8card.py" --gpus 0,1,2,3,4,5,6,7 \
                --output "$H3_REPRO_WORK/reports/snake-eight-cached.json"
            if [ "$1" = preflight ]; then
                python -u "$TOOLS/h3_preflight.py" --source-root "$H3_SOURCE_ROOT" \
                    --model-path "$MODEL_PATH" --first-frame "$FIRST_FRAME" --expected-gpus 8 \
                    --output "$H3_REPRO_WORK/reports/preflight-eight.json"
            else
                export OUTPUT_DIR="${OUTPUT_DIR:-$H3_REPRO_WORK/compare-$(date -u +%Y%m%dT%H%M%SZ)}"
                bash --noprofile --norc "$TOOLS/run_h3_compare.sh"
            fi
            ;;
        *) usage >&2; return 64 ;;
    esac
}
if [ "${1:-}" = --worker ]; then worker "$2"; exit 0; fi
phase="$1"
case "$phase" in prepare|build|jit|smoke|eight-smoke|preflight|compare) ;; *) usage >&2; exit 64;; esac
# This check uses stdlib only and runs before mkdir or any installation.
env -u PYTHONPATH -u PYTHONHOME "$H3_BASE_PYTHON" "$TOOLS/h3_bootstrap.py" \
    validate-work --work "$H3_REPRO_WORK"
mkdir -p "$H3_REPRO_WORK/logs"
log="$H3_REPRO_WORK/logs/$phase-$(date -u +%Y%m%dT%H%M%SZ).log"
logged_worker() (
    heartbeat() {
        tick=""
        trap 'if [ -n "$tick" ]; then kill "$tick" 2>/dev/null || true; fi; exit 0' TERM INT
        while :; do
            sleep 30 & tick=$!
            wait "$tick" || exit 0
            tick=""
            printf '[%s] H3_PHASE_HEARTBEAT %s\n' "$(date -u +%FT%TZ)" "$phase"
        done
    }
    heartbeat &
    heart=$!
    trap 'kill "$heart" 2>/dev/null || true; wait "$heart" 2>/dev/null || true' EXIT
    printf '[%s] H3_PHASE_BEGIN %s jobs=%s\n' "$(date -u +%FT%TZ)" "$phase" "$BUILD_JOBS"
    bash --noprofile --norc "$0" --worker "$phase"
)
set +e
logged_worker 2>&1 | tee "$log"
status=("${PIPESTATUS[@]}")
set -e
rc="${status[0]}"
if [ "$rc" -eq 0 ] && [ "${status[1]}" -ne 0 ]; then rc="${status[1]}"; fi
printf '[%s] H3_PHASE_END %s rc=%s log=%s\n' "$(date -u +%FT%TZ)" "$phase" "$rc" "$log"
exit "$rc"
