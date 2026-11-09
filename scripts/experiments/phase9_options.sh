#!/usr/bin/env bash
# Sourcing defines functions only. No allocation, AFS access, build or inference.
h3_phase9_v4() {
    export H3_EXPERIMENT_FLASH_GFX936=1
    export H3_FLASH_LIBRARY="$1"
    export H3_EXPERIMENT_VAE_DEVICES=4
}
h3_phase9_dispatch() {
    # Caller supplies the restored private node bundle; SHA verification is mandatory.
    "${H3_PYTHON:-python3}" "${BASH_SOURCE[0]%/*}/phase9_attention.py" --verify-bundle "$1" || return
    export H3_EXPERIMENT_FLASH_GFX936=1
    export H3_FLASH_LIBRARY="$1/libh3_flash_dispatch.so"
    export H3_EXPERIMENT_VAE_DEVICES=4
}
