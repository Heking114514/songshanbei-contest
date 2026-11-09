#!/usr/bin/env bash
# Source only: defines presets; never requests resources, builds, or runs inference.
h3_phase8_baseline() {
    export H3_EXPERIMENT_VAE_DEVICES=0
    unset H3_PHASE8_NATIVE_TRACE_DIR
}
h3_phase8_vae() {
    case "$1" in 2|4|8) ;; *) return 2 ;; esac
    export H3_EXPERIMENT_VAE_DEVICES="$1"
    export H3_EXPERIMENT_VAE_RESERVE_MIB=2048
}
