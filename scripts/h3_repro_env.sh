#!/usr/bin/env bash
# Source only after prepare; independent of personal AFS/old runtime packages.
H3_SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export H3_SOURCE_ROOT
: "${H3_REPRO_WORK:?Set a node-local reproduction work directory}"
test -s "$H3_REPRO_WORK/prepared.json"
export VIRTUAL_ENV="$H3_REPRO_WORK/venv"
export PATH="$VIRTUAL_ENV/bin:$H3_REPRO_WORK/xmake/bin:/usr/local/bin:/usr/bin:/bin"
export DTK_ROOT="${DTK_ROOT:-/opt/dtk}" HYGON_ARCH="${HYGON_ARCH:-gfx936}"
export HYGON_CUDA_HOME="$DTK_ROOT/cuda/cuda"
export PATH="$HYGON_CUDA_HOME/bin:$DTK_ROOT/bin:$PATH"
export INFINI_ROOT="$H3_REPRO_WORK/install"
export H3_PYTHON="$VIRTUAL_ENV/bin/python3"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset PYTHONHOME
export PYTHONPATH="$H3_REPRO_WORK/python:$H3_SOURCE_ROOT/InfiniCore/python:$H3_SOURCE_ROOT/InfiniLM/python:$H3_SOURCE_ROOT/ninetoothed/src:$H3_SOURCE_ROOT/scripts"
H3_TORCH_LIB="$(python -c 'import json,os; from pathlib import Path; p=Path(os.environ["H3_REPRO_WORK"])/"prepared.json"; print(Path(json.loads(p.read_text())["platform"]["torch_path"]).parent/"lib")')"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$H3_TORCH_LIB:$DTK_ROOT/lib:$DTK_ROOT/lib64:$HYGON_CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
unset H3_TORCH_LIB
if [ -d "$H3_REPRO_WORK/xmake/share/xmake" ]; then
    export XMAKE_PROGRAM_DIR="$H3_REPRO_WORK/xmake/share/xmake"
else
    unset XMAKE_PROGRAM_DIR
fi
export XMAKE_GLOBALDIR="$H3_REPRO_WORK/xmake-global"
export H3_CONFIG_CACHE="$H3_REPRO_WORK/config"
export TMPDIR="$H3_REPRO_WORK/tmp" TRITON_HOME="$H3_REPRO_WORK/triton"
export TRITON_CACHE_DIR="$H3_REPRO_WORK/jit" NINETOOTHED_CACHE_DIR="$H3_REPRO_WORK/ninetoothed"
export XDG_CACHE_HOME="$H3_REPRO_WORK/cache" HF_HOME="$H3_REPRO_WORK/cache/huggingface"
export TORCH_EXTENSIONS_DIR="$H3_REPRO_WORK/torch" FLAGTREE_CACHE_DIR="$H3_REPRO_WORK/cache/flagtree"
export H3_JIT_TOOLCHAIN="$H3_REPRO_WORK/toolchain"
export TRITON_HIP_CLANG_PATH="$H3_REPRO_WORK/clang-hcu"
export TRITON_HIP_LLD_PATH="$H3_JIT_TOOLCHAIN/bin/ld.lld"
export TRITON_CACHE_MANAGER=h3_triton_cache:SiblingCacheManager
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0
export H3_ENV_READY=1
mkdir -p "$INFINI_ROOT" "$H3_CONFIG_CACHE" "$TMPDIR" "$TRITON_HOME" "$TRITON_CACHE_DIR" \
    "$NINETOOTHED_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR"
