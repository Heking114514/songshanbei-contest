#!/usr/bin/env bash
# Source defines functions only; it does not configure, build, or run anything.
# These presets reset phase4 flags, including VIDEO_POSTPROCESS shared with
# phase3. Select phase2/phase3 first, then phase4, before starting new processes.

h3_phase4_baseline() {
    export H3_EXPERIMENT_ADALN_INDEXED=0
    export H3_EXPERIMENT_DETERMINISTIC_BLAS=0
    export H3_EXPERIMENT_VIDEO_POSTPROCESS=0
    export H3_EXPERIMENT_VIDEO_CHUNK=8
    export H3_EXPERIMENT_VIDEO_WORKERS=4
}

h3_phase4_adaln_indexed() {
    h3_phase4_baseline
    export H3_EXPERIMENT_ADALN_INDEXED=1
}

h3_phase4_video_postprocess() {
    h3_phase4_baseline
    export H3_EXPERIMENT_VIDEO_POSTPROCESS=1
}

h3_phase4_deterministic_blas() {
    h3_phase4_baseline
    export H3_EXPERIMENT_DETERMINISTIC_BLAS=1
}

h3_phase4_candidate() {
    h3_phase4_baseline
    export H3_EXPERIMENT_ADALN_INDEXED=1
    export H3_EXPERIMENT_VIDEO_POSTPROCESS=1
}
