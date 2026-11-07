#!/usr/bin/env bash
# Source after activating the Python environment used to build the extensions.
H3_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export INFINI_ROOT="${INFINI_ROOT:-${IFAB_AFS_HOME:?Set INFINI_ROOT or IFAB_AFS_HOME}/handover/current-install}"
export DTK_ROOT="${DTK_ROOT:-/opt/dtk}"
export HYGON_CUDA_HOME="${HYGON_CUDA_HOME:-$DTK_ROOT/cuda/cuda}"
export PYTHONPATH="$H3_REPO_ROOT/InfiniCore/python:$H3_REPO_ROOT/InfiniLM/python:$H3_REPO_ROOT/ninetoothed/src:${PYTHONPATH:-}"
if [ -n "${H3_PYTHON_DEPS:-}" ]; then
    export PYTHONPATH="$H3_PYTHON_DEPS:$PYTHONPATH"
elif [ -n "${IFAB_AFS_HOME:-}" ] && [ -d "$IFAB_AFS_HOME/handover/current-python" ]; then
    export PYTHONPATH="$IFAB_AFS_HOME/handover/current-python:$PYTHONPATH"
fi
H3_TORCH_LIB="$(python -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent/"lib")')"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$H3_TORCH_LIB:$DTK_ROOT/lib:$DTK_ROOT/lib64:$HYGON_CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$HYGON_CUDA_HOME/bin:$DTK_ROOT/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
unset H3_TORCH_LIB

# Smoke JIT output belongs to the current node workspace, never the AFS home.
if [ -n "${EIGHT_WORK:-}" ]; then
    export NINETOOTHED_CACHE_DIR="$EIGHT_WORK/ninetoothed"
fi
