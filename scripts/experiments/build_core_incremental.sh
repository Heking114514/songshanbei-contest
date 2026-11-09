#!/usr/bin/env bash
# Use only AFTER a node-local copy has a verified build configuration and objects.
set -euo pipefail
TOOLS="$(cd -- "$(dirname -- "$0")" && pwd)"
SOURCE="$(cd "$TOOLS/../.." && pwd)"
: "${INFINI_ROOT:?Set a new install directory inside this experimental source copy}"
: "${BUILD_LOG:?Set a NEW node-local build log path}"
SOURCE_REAL="$(realpath "$SOURCE")"
INSTALL_REAL="$(realpath -m "$INFINI_ROOT")"
case "$INSTALL_REAL/" in
    "$SOURCE_REAL/"*) ;;
    *) printf 'Install must stay inside the experimental copy: %s\n' "$SOURCE_REAL" >&2; exit 64 ;;
esac
if [ -n "${IFAB_AFS_HOME:-}" ]; then
    case "$SOURCE_REAL/" in
        "$(realpath "$IFAB_AFS_HOME")/"*) printf 'Use node-local experiment storage\n' >&2; exit 64 ;;
    esac
fi
test -f "$SOURCE/../baseline-manifest.json" || {
    printf 'Missing experimental baseline manifest\n' >&2; exit 64;
}
test -d "$SOURCE/InfiniCore/.xmake" || {
    printf 'No verified build configuration here; prepare the copied build first\n' >&2; exit 64;
}
test -d "$SOURCE/InfiniCore/build/.objs" || {
    printf 'No restored object cache; refusing an accidental full build\n' >&2; exit 64;
}
test ! -e "$BUILD_LOG" || { printf 'Build log already exists\n' >&2; exit 64; }
mkdir -p "$(dirname "$BUILD_LOG")"
cd "$SOURCE/InfiniCore"
xmake build -vD -y -j "${BUILD_JOBS:-120}" infinicore_cpp_api 2>&1 | tee "$BUILD_LOG"
xmake install -y infinicore_cpp_api 2>&1 | tee -a "$BUILD_LOG"
printf 'H3_EXPERIMENT_CORE_BUILD_COMPLETE install=%s\n' "$INFINI_ROOT"
