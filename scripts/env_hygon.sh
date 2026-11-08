#!/usr/bin/env bash
# Source after activating the environment used to build the native extensions.
H3_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export INFINI_ROOT="${INFINI_ROOT:-$H3_REPO_ROOT/.h3-install}"
export DTK_ROOT="${DTK_ROOT:-/opt/dtk}"
export HYGON_CUDA_HOME="${HYGON_CUDA_HOME:-$DTK_ROOT/cuda/cuda}"
export PYTHONPATH="$H3_REPO_ROOT/InfiniCore/python:$H3_REPO_ROOT/InfiniLM/python:$H3_REPO_ROOT/ninetoothed/src:${PYTHONPATH:-}"
if [ -n "${H3_PYTHON_DEPS:-}" ]; then
    export PYTHONPATH="$H3_PYTHON_DEPS:$PYTHONPATH"
fi
H3_TORCH_LIB="$(python -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent/"lib")')"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$H3_TORCH_LIB:$DTK_ROOT/lib:$DTK_ROOT/lib64:$HYGON_CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$HYGON_CUDA_HOME/bin:$DTK_ROOT/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset H3_TORCH_LIB
if [ -n "${EIGHT_WORK:-}" ]; then
    export NINETOOTHED_CACHE_DIR="$EIGHT_WORK/ninetoothed"
fi
