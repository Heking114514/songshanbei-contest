#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}"
WORK="${ROOT}/work/infinilm"
CORE="${ROOT}/Projects/InfiniCore"

source "${ROOT}/setup_env.sh"
module load compiler/gcc/11.4.0
cd "${WORK}"

export NINETOOTHED_TRITON_COMPILE="${WORK}/tools/triton_compile.py"
export NINETOOTHED_TRITON_LINK="${WORK}/tools/triton_link.py"
export NINETOOTHED_TRITON_TARGET="hip:gfx936:64"
export TMPDIR="${HOME}/.cache/ninetoothed-tmp"
mkdir -p "${TMPDIR}" results/screenshots

python -u kernels/final_validation.py --out results/final_validation.json
python -u kernels/mha_baseline_compare.py --out results/mha_baseline_compare.json
python -u kernels/group_norm_validation.py --out results/group_norm_validation.json
python -u kernels/audio_ops_validation.py --out results/audio_ops_validation.json
python -u kernels/snake_beta_nt_dynamic.py --tile-width 512 --warps 4 \
  --lengths 207,2048,16384,331200 --out results/snake_beta_nt_dynamic.json
python -u kernels/aot_snake_beta.py --out "${WORK}/kernels/aot_snake_beta"
python -u kernels/aot_snake_beta_runtime.py --out results/aot_snake_beta_runtime.json
python -u scripts/snake_wrapper_check.py
python -u scripts/audio_hook_check.py
python -u scripts/weight_norm_check.py
python -u scripts/media_contract_check.py

(
  cd "${CORE}"
  python test/infinicore/ops/group_norm.py --hygon --num_iterations 20
  python test/infinicore/ops/conv1d.py --hygon --num_iterations 20
  python test/infinicore/ops/conv_transpose1d.py --hygon --num_iterations 20
  python test/infinicore/ops/snake_beta.py --hygon --num_iterations 20
) | tee results/official_ops_tests.log

python -u scripts/operator_summary.py | tee results/operator_summary.txt
