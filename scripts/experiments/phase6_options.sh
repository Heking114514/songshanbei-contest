#!/usr/bin/env bash
# Source defines functions only; no build, resource requests or testing.
h3_phase6_baseline() {
    export H3_EXPERIMENT_GEMM_LT=0
    unset H3_GEMM_LT_TABLE
}
h3_phase6_heuristic() {
    if (( $# != 2 )); then
        printf '%s\n' 'usage: h3_phase6_heuristic RAW_INDEX WORKSPACE_MB' >&2
        return 2
    fi
    case "$1" in 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15) ;; *) return 2;; esac
    case "$2" in 0|8|32|64|128|256) ;; *) return 2;; esac
    export H3_EXPERIMENT_GEMM_LT=1 H3_GEMM_LT_TRACE=1
    export H3_GEMM_LT_INDEX="$1" H3_GEMM_LT_WORKSPACE_MB="$2"
    export H3_GEMM_LT_ALIGNMENT=16 H3_GEMM_LT_MIN_TOKENS=4096
    unset H3_GEMM_LT_TABLE
}
h3_phase6_table() {
    if (( $# != 3 )); then
        printf '%s\n' 'usage: h3_phase6_table TABLE_PATH VERIFIED_RUNTIME_SHA256 WORKSPACE_MB' >&2
        return 2
    fi
    if [[ ! -f "$1" || ! "$2" =~ ^[a-f0-9]{64}$ ]]; then return 2; fi
    case "$3" in 0|8|32|64|128|256) ;; *) return 2;; esac
    export H3_EXPERIMENT_GEMM_LT=1 H3_GEMM_LT_TRACE=1
    export H3_GEMM_LT_TABLE="$1" H3_GEMM_LT_RUNTIME_TAG="$2"
    export H3_GEMM_LT_TABLE_ONLY=1 H3_GEMM_LT_INDEX=0
    export H3_GEMM_LT_ALIGNMENT=16 H3_GEMM_LT_MIN_TOKENS=4096
    export H3_GEMM_LT_WORKSPACE_MB="$3"
}
