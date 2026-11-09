#!/usr/bin/env bash
# Defines opt-in presets only. Sourcing this file does not launch work or set flags.
# Call a function explicitly BEFORE starting a fresh Python/model process.

h3_phase2_baseline() {
    export H3_EXPERIMENT_MHA_QUERY_CHUNK=0
    export H3_EXPERIMENT_MHA_HEAD_BATCH=0
    export H3_EXPERIMENT_ADDCMUL_ROWS=0
    export H3_EXPERIMENT_SWIGLU_ROWS=0
    export H3_EXPERIMENT_RMS_SMALL_ROWS=0
    export H3_EXPERIMENT_ADALN_INPLACE=0
    export H3_EXPERIMENT_SNAKE_PREPARED=0
    export H3_EXPERIMENT_SNAKE_STRIDED=0
    export H3_EXPERIMENT_SNAKE_TILE_WIDTH=512
    export H3_EXPERIMENT_SNAKE_NUM_WARPS=4
}

h3_phase2_pointwise() {
    h3_phase2_baseline
    export H3_EXPERIMENT_ADDCMUL_ROWS=1
    export H3_EXPERIMENT_SWIGLU_ROWS=1
    export H3_EXPERIMENT_RMS_SMALL_ROWS=1
    export H3_EXPERIMENT_ADALN_INPLACE=1
}

h3_phase2_snake_prepared() {
    h3_phase2_baseline
    export H3_EXPERIMENT_SNAKE_PREPARED=1
}

h3_phase2_snake_strided() {
    h3_phase2_snake_prepared
    export H3_EXPERIMENT_SNAKE_STRIDED=1
}

h3_phase2_candidate() {
    h3_phase2_pointwise
    export H3_EXPERIMENT_MHA_QUERY_CHUNK=2048
    export H3_EXPERIMENT_SNAKE_PREPARED=1
    export H3_EXPERIMENT_SNAKE_STRIDED=1
}

# Independent alternative; compose with baseline/candidate explicitly.
h3_phase2_head_batch() {
    export H3_EXPERIMENT_MHA_HEAD_BATCH=4
}
