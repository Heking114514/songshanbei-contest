#!/usr/bin/env bash
# First source build in a node-local checkout, using the existing platform env.
# Current AFS deployment should use reproduce_h3_compare.sh instead.
set -euo pipefail
TOOLS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export H3_SOURCE_ROOT="$(cd "$TOOLS/.." && pwd)"
: "${IFAB_AFS_HOME:?}"
: "${H3_RUNTIME_DIR:?First restore the verified runtime archive on node storage}"
: "${H3_RUNTIME_ENV:?Path to h3_runtime_env.sh delivered with the runtime helpers}"
case "$H3_SOURCE_ROOT/" in "$IFAB_AFS_HOME/"*) printf 'Build the fresh checkout on node storage\n' >&2; exit 64;; esac
source "$H3_RUNTIME_ENV"
# A new source checkout and install stay on the node. Do not overwrite AFS stable assets.
export INFINI_ROOT="$H3_SOURCE_ROOT/.h3-install"
export BUILD_SETTINGS_FILE="$IFAB_AFS_HOME/handover/current-logs/torch_build_settings.txt"
export H3_PYTHON="$IFAB_AFS_HOME/handover/venv/bin/python3"
export H3_FORBID_JIT_COMPILATION=0
mkdir -p "$H3_SOURCE_ROOT/.h3-xmake"
tar --no-same-owner -xzf "$IFAB_AFS_HOME/handover/tools/xmake-built-v1.tar.gz" -C "$H3_SOURCE_ROOT/.h3-xmake"
export PATH="$H3_SOURCE_ROOT/.h3-xmake/bin:$PATH"
export XMAKE_PROGRAM_DIR="$H3_SOURCE_ROOT/.h3-xmake/share/xmake"
export XMAKE_GLOBALDIR="$H3_SOURCE_ROOT/.h3-xmake-global"
export BUILD_JOBS=120 BUILD_VERBOSE=1 BUILD_SKIP_CONFIGURE=0
bash --noprofile --norc "$TOOLS/build_hygon.sh" 2>&1 | tee "$H3_SOURCE_ROOT/build-h3.log"
# Preserve runtime compatibility directories ahead of the package sources.
export PYTHONPATH="$H3_RUNTIME_DIR/compat55:$H3_RUNTIME_DIR/compat:$H3_RUNTIME_DIR/python:$H3_RUNTIME_DIR/runtime:$H3_SOURCE_ROOT/InfiniCore/python:$H3_SOURCE_ROOT/InfiniLM/python:$H3_SOURCE_ROOT/ninetoothed/src:$ROOT/current-python"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$LD_LIBRARY_PATH"
export H3_ENV_READY=1 H3_FORBID_JIT_COMPILATION=1
export MODEL_PATH=/public/home/aclsylqidf/shared/MiniMax-H3
export FIRST_FRAME="$H3_SOURCE_ROOT/inputs/first_frame.png"
export OUTPUT_DIR="$H3_SOURCE_ROOT/h3-compare-$(date -u +%Y%m%dT%H%M%SZ)"
unset H3_INFINILM_VIDEO H3_REFERENCE_VIDEO
bash --noprofile --norc "$TOOLS/run_h3_compare.sh"
