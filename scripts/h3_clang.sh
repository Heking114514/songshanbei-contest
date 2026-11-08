#!/usr/bin/env bash
set -euo pipefail
: "${H3_JIT_TOOLCHAIN:?}"
if [ "$#" -eq 1 ] && [ "$1" = --version ]; then
    "$H3_JIT_TOOLCHAIN/bin/clang-18" --rocm-path="${DTK_ROOT:-/opt/dtk}" --version |
        sed 's|^InstalledDir:.*|InstalledDir: /opt/h3-jit-toolchain/bin|'
    exit 0
fi
if [ "${H3_FORBID_JIT_COMPILATION:-0}" = 1 ]; then
    printf 'UNEXPECTED_JIT_COMPILATION %s\n' "$*" >&2
    exit 91
fi
exec "$H3_JIT_TOOLCHAIN/bin/clang-18" --rocm-path="${DTK_ROOT:-/opt/dtk}" "$@"
