#!/usr/bin/env bash
# Compatibility entry. Prefer explicit one-GPU phases and eight-GPU compare.
set -euo pipefail
TOOLS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -gt 0 ]; then
    exec bash --noprofile --norc "$TOOLS/reproduce_h3.sh" "$@"
fi
printf 'Run prepare, build, jit, smoke on one GPU; then preflight, compare on eight GPUs.\n'
printf 'Example: bash scripts/build_and_compare_h3.sh prepare\n'
printf 'See REPRODUCE.md for clean-clone prerequisites and validation status.\n'
