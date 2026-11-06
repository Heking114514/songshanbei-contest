#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}"
CORE="${ROOT}/Projects/InfiniCore"
LM="${ROOT}/Projects/InfiniLM"
WORK="${ROOT}/work/infinilm"
CORE_BUILD="${CORE}/build/linux/x86_64/release"
LM_BUILD="${LM}/build/linux/x86_64/release"

source "${ROOT}/setup_env.sh"
module load compiler/gcc/11.4.0

echo "[1/7] Configure InfiniCore: $(date -Is)"
cd "${CORE}"
xmake f -y \
  --cc=gcc \
  --cxx=g++ \
  --ld=g++ \
  --hygon-dcu=y \
  --hygon-arch=gfx936 \
  --aten=y \
  --ccl=y

echo "[2/7] Build InfiniCore C++ API: $(date -Is)"
xmake -j4 -y infinicore_cpp_api

echo "[3/7] Build and install Python operator bindings: $(date -Is)"
xmake -j4 -y _infinicore
xmake install -y infinicore_cpp_api _infinicore

# The editable Python package loads native libraries from this directory.
install -m 755 "${CORE_BUILD}"/_infinicore*.so "${CORE}/python/infinicore/"
install -m 755 "${CORE_BUILD}"/libinfiniop.so "${CORE}/python/infinicore/lib/"
install -m 755 "${CORE_BUILD}"/libinfinirt.so "${CORE}/python/infinicore/lib/"
install -m 755 "${CORE_BUILD}"/libinfiniccl.so "${CORE}/python/infinicore/lib/"
install -m 755 "${CORE_BUILD}"/libinfinicore_cpp_api.so "${CORE}/python/infinicore/lib/"

echo "[4/7] Build InfiniLM: $(date -Is)"
cd "${LM}"
xmake -j4 -y _infinilm
xmake install -y _infinilm
install -m 755 "${LM_BUILD}"/_infinilm*.so "${LM}/python/infinilm/lib/"

echo "[5/7] Verify native libraries: $(date -Is)"
ldd "${CORE}/python/infinicore/lib/_infinicore"*.so
ldd "${LM}/python/infinilm/lib/_infinilm"*.so

echo "[6/7] Run batched operator validation: $(date -Is)"
cd "${WORK}"
export NINETOOTHED_TRITON_COMPILE="${WORK}/tools/triton_compile.py"
export NINETOOTHED_TRITON_LINK="${WORK}/tools/triton_link.py"
export NINETOOTHED_TRITON_TARGET="hip:gfx936:64"
export TMPDIR="${HOME}/.cache/ninetoothed-tmp"
mkdir -p "${TMPDIR}"
python kernels/aot_snake_beta.py --out "${WORK}/kernels/aot_snake_beta"
python kernels/aot_snake_beta_runtime.py --out results/aot_snake_beta_runtime.json
python kernels/final_validation.py --out results/final_validation.json
python kernels/mha_baseline_compare.py --out results/mha_baseline_compare.json
python kernels/group_norm_validation.py --out results/group_norm_validation.json
python kernels/audio_ops_validation.py --out results/audio_ops_validation.json
python kernels/snake_beta_nt_dynamic.py --tile-width 512 --warps 4 --lengths 207,2048,16384,331200 --out results/snake_beta_nt_dynamic.json
python scripts/snake_wrapper_check.py
python scripts/audio_hook_check.py
python scripts/weight_norm_check.py
python scripts/media_contract_check.py

echo "[7/7] Completed: $(date -Is)"
