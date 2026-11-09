#!/usr/bin/env bash
# Definitions only. Select phase2/3/4 first, then this stage, in a new process.
# These flags imply device-indexed modulation in the H3 block when eligible.
h3_phase5_baseline() {
    export H3_EXPERIMENT_RMS_ADALN=0
    export H3_EXPERIMENT_GATE_RMS_ADALN=0
    export H3_EXPERIMENT_QK_NORM_ROPE=0
    export H3_EXPERIMENT_FUSED_RMS_BLOCK=0
}
h3_phase5_rms_adaln() {
    h3_phase5_baseline
    export H3_EXPERIMENT_RMS_ADALN=1
}
h3_phase5_gate_rms_adaln() {
    h3_phase5_baseline
    export H3_EXPERIMENT_GATE_RMS_ADALN=1
}
h3_phase5_qk_norm_rope() {
    h3_phase5_baseline
    export H3_EXPERIMENT_QK_NORM_ROPE=1
}
h3_phase5_candidate() {
    h3_phase5_baseline
    export H3_EXPERIMENT_RMS_ADALN=1
    export H3_EXPERIMENT_GATE_RMS_ADALN=1
    export H3_EXPERIMENT_QK_NORM_ROPE=1
}
h3_phase5_tuned_candidate() {
    case "$1" in
        128|256|512|1024) ;;
        *) printf 'Expected RMS block: 128/256/512/1024\n' >&2; return 2 ;;
    esac
    h3_phase5_candidate
    export H3_EXPERIMENT_FUSED_RMS_BLOCK="$1"
}
