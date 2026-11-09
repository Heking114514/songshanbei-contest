#!/usr/bin/env bash
# Defines functions only. Source does not set flags, build, or launch a task.
# These presets cover phase3 ONLY; select phase2 separately before a new process.

h3_phase3_baseline() {
    export H3_EXPERIMENT_ADALN_BATCH=0
    export H3_EXPERIMENT_ROPE_INPLACE=0
    export H3_EXPERIMENT_VIDEO_POSTPROCESS=0
    export H3_EXPERIMENT_VIDEO_CHUNK=8
    export H3_EXPERIMENT_VIDEO_WORKERS=4
}

h3_phase3_adaln_batch() {
    h3_phase3_baseline
    export H3_EXPERIMENT_ADALN_BATCH=1
}

h3_phase3_rope_inplace() {
    h3_phase3_baseline
    export H3_EXPERIMENT_ROPE_INPLACE=1
}

h3_phase3_video_postprocess() {
    h3_phase3_baseline
    export H3_EXPERIMENT_VIDEO_POSTPROCESS=1
}

h3_phase3_candidate() {
    h3_phase3_baseline
    export H3_EXPERIMENT_ADALN_BATCH=1
    export H3_EXPERIMENT_ROPE_INPLACE=1
    export H3_EXPERIMENT_VIDEO_POSTPROCESS=1
}
