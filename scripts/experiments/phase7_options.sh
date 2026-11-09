#!/usr/bin/env bash
# Source defines functions only. No server/resource/AFS operations.
h3_phase7_baseline() {
    export H3_EXPERIMENT_FLASH_GFX936=0 H3_EXPERIMENT_TP_OVERLAP_ROWS=0
    unset H3_FLASH_LIBRARY
}
h3_phase7_flash() {
    if (( $# != 1 )) || [[ "$1" != /* || ! -f "$1" ]]; then
        printf '%s\n' 'usage: h3_phase7_flash ABSOLUTE_PRIVATE_PLUGIN_PATH' >&2
        return 2
    fi
    export H3_EXPERIMENT_FLASH_GFX936=1 H3_FLASH_LIBRARY="$1"
    export H3_EXPERIMENT_MHA_QUERY_CHUNK=0 H3_EXPERIMENT_MHA_HEAD_BATCH=0
}
h3_phase7_overlap() {
    if (( $# != 1 )); then return 2; fi
    case "$1" in 1024|2048|4096|8192) ;; *) return 2;; esac
    export H3_EXPERIMENT_TP_OVERLAP_ROWS="$1"
}
