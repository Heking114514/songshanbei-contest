#!/usr/bin/env bash
# Run inside the selected Python environment. Platform Torch must already work.
set -euo pipefail
umask 007
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON="${H3_PYTHON:-$(command -v python3)}"
export DTK_ROOT="${DTK_ROOT:-/opt/dtk}"
export HYGON_CUDA_HOME="${HYGON_CUDA_HOME:-$DTK_ROOT/cuda/cuda}"
export HYGON_ARCH="${HYGON_ARCH:-gfx936}"
export INFINI_ROOT="${INFINI_ROOT:-$REPO_ROOT/.h3-install}"
export PATH="$HYGON_CUDA_HOME/bin:$DTK_ROOT/bin:$PATH"
export OMP_NUM_THREADS="${BUILD_PYTHON_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${BUILD_PYTHON_THREADS:-1}"
export MKL_NUM_THREADS="${BUILD_PYTHON_THREADS:-1}"
printf '[environment] Reading Python and platform Torch build settings\n'
if [ -n "${BUILD_SETTINGS_FILE:-}" ]; then
    PYTHON_BUILD_SETTINGS="$(cat -- "$BUILD_SETTINGS_FILE")"
else
PYTHON_BUILD_SETTINGS="$("${BUILD_SETTINGS_PYTHON:-$PYTHON}" -u - <<'PY'
import os
import pathlib
import sysconfig
import torch
assert torch.version.hip, 'The selected Python must contain platform Hygon Torch'
print(sysconfig.get_path('include'))
pybind_include = os.environ.get('PYBIND11_INCLUDE')
if not pybind_include:
    import pybind11
    pybind_include = pybind11.get_include()
print(pybind_include)
print(int(torch._C._GLIBCXX_USE_CXX11_ABI))
print(pathlib.Path(torch.__file__).parent / 'lib')
print(torch.__version__)
print(torch.version.hip)
PY
)"
fi
mapfile -t PYTHON_BUILD_VALUES <<< "$PYTHON_BUILD_SETTINGS"
test "${#PYTHON_BUILD_VALUES[@]}" -eq 6
export PYTHON_INCLUDE="${PYTHON_BUILD_VALUES[0]}"
export PYBIND11_INCLUDE="${PYTHON_BUILD_VALUES[1]}"
export INFINILM_PYTHON_INCLUDE="$PYTHON_INCLUDE"
export INFINILM_PYBIND11_INCLUDE="$PYBIND11_INCLUDE"
export INFINILM_CXX11_ABI="${PYTHON_BUILD_VALUES[2]}"
TORCH_LIB="${PYTHON_BUILD_VALUES[3]}"
export INFINICORE_TORCH_DIR="$(dirname -- "$TORCH_LIB")"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$TORCH_LIB:$DTK_ROOT/lib:$DTK_ROOT/lib64:$HYGON_CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
JOBS="${BUILD_JOBS:-120}"
XMAKE_DIAGNOSTICS=()
if [ "${BUILD_VERBOSE:-0}" = 1 ]; then
    XMAKE_DIAGNOSTICS=(-vD)
fi
test -x "$HYGON_CUDA_HOME/bin/nvcc"
command -v xmake
printf 'Torch %s HIP %s ABI %s\n' "${PYTHON_BUILD_VALUES[4]}" "${PYTHON_BUILD_VALUES[5]}" "$INFINILM_CXX11_ABI"
configure_component() {
    component="$1"
    shift
    if [ "${BUILD_SKIP_CONFIGURE:-0}" = 1 ]; then
        printf '[configure] explicit skip %s\n' "$component"
        return
    fi
    key=""
    marker=""
    if [ -n "${H3_CONFIG_CACHE:-}" ]; then
        mkdir -p "$H3_CONFIG_CACHE"
        key="$("$PYTHON" "$REPO_ROOT/scripts/h3_bootstrap.py" config-key "$component")"
        marker="$H3_CONFIG_CACHE/$component.json"
        if [ -f "$marker" ] && [ "$(cat "$marker")" = "$key" ] \
            && [ -s .xmake/linux/x86_64/xmake.conf ]; then
            printf '[configure] unchanged %s; reuse existing xmake configuration\n' "$component"
            return
        fi
    fi
    xmake f "$@"
    if [ -n "$marker" ]; then
        printf '%s\n' "$key" > "$marker.partial"
        mv -f "$marker.partial" "$marker"
    fi
}
printf '[1/5] Configure InfiniCore (ATen on, CCL on, NineToothed Python JIT)\n'
cd "$REPO_ROOT/InfiniCore"
configure_component InfiniCore "${XMAKE_DIAGNOSTICS[@]}" -y -m release --ccache=n --hygon-dcu=y --hygon-arch="$HYGON_ARCH" --cuda="$HYGON_CUDA_HOME" --aten=y --ccl=y --ninetoothed=n --omp=n --cxxflags="-D_GLIBCXX_USE_CXX11_ABI=$INFINILM_CXX11_ABI"
printf '[2/5] Build and install InfiniCore\n'
for dependency in infiniop infinirt infiniccl; do
    xmake build "${XMAKE_DIAGNOSTICS[@]}" -y -j "$JOBS" "$dependency"
    xmake install -y "$dependency"
done
test -s "$INFINI_ROOT/include/infiniop.h"
xmake build "${XMAKE_DIAGNOSTICS[@]}" -y -j "$JOBS" infinicore_cpp_api
xmake install -y infinicore_cpp_api
xmake build "${XMAKE_DIAGNOSTICS[@]}" -y -j "$JOBS" _infinicore
xmake install -y _infinicore
printf '[3/5] Build and install InfiniLM\n'
cd "$REPO_ROOT/InfiniLM"
configure_component InfiniLM "${XMAKE_DIAGNOSTICS[@]}" -y -m release --ccache=n --cxx11-abi="$INFINILM_CXX11_ABI"
xmake build "${XMAKE_DIAGNOSTICS[@]}" -y -j "$JOBS" _infinilm
xmake install -y _infinilm
printf '[4/5] Register source packages\n'
export PYTHONPATH="$REPO_ROOT/InfiniCore/python:$REPO_ROOT/InfiniLM/python:$REPO_ROOT/ninetoothed/src:${PYTHONPATH:-}"
if [ "${BUILD_IMPORT_CHECK:-1}" = 1 ]; then
"$PYTHON" - <<'PY'
import infinicore, infinilm
from infinicore.nn.functional import snake_beta_ninetoothed
print("INFINICORE",infinicore.__file__)
print("INFINILM",infinilm.__file__)
print("SNAKE_BETA",snake_beta_ninetoothed)
PY
else
    printf '[imports] validated by the following build origin audit\n'
fi
printf '[5/5] BUILD_COMPLETE\n'
