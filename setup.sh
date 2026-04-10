#!/usr/bin/env bash
# setup.sh — Initialize PIMeval submodule, build it, and symlink lnorm/rmsnorm into PIMbench.
#
# Usage (from a fresh clone):
#   git clone <your-repo-url>
#   cd assignment_pimeval
#   bash setup.sh
#
# Or with parallel build jobs:
#   bash setup.sh -j8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIMEVAL_DIR="${SCRIPT_DIR}/PIMeval-PIMbench"
PIMBENCH_DIR="${PIMEVAL_DIR}/PIMbench"

# Parse optional -j flag for parallel make
JOBS=""
if [[ "${1:-}" =~ ^-j[0-9]*$ ]]; then
    JOBS="$1"
fi

# ---------- Step 1: Initialize and update the PIMeval submodule ----------
echo "==> Initializing PIMeval-PIMbench submodule..."
cd "$SCRIPT_DIR"
git submodule update --init --recursive
echo "    Done."

# ---------- Step 2: Build PIMeval (libpimeval + PIMbench) ----------
echo "==> Building PIMeval (make perf ${JOBS})..."
cd "$PIMEVAL_DIR"
make perf $JOBS
echo "    Done."

# ---------- Step 3: Symlink lnorm and rmsnorm into PIMbench ----------
WORKLOADS=(lnorm rmsnorm)

for workload in "${WORKLOADS[@]}"; do
    src="${SCRIPT_DIR}/${workload}"
    dest="${PIMBENCH_DIR}/${workload}"

    if [ ! -d "$src" ]; then
        echo "WARNING: Source directory ${src} does not exist, skipping."
        continue
    fi

    if [ -L "$dest" ]; then
        echo "    Symlink ${dest} already exists, re-creating..."
        rm "$dest"
    elif [ -d "$dest" ]; then
        echo "ERROR: ${dest} exists as a real directory. Remove it manually before symlinking."
        exit 1
    fi

    # Use a relative symlink so the repo stays portable
    ln -s "../../${workload}" "$dest"
    echo "==> Symlinked: ${dest} -> ../../${workload}"
done

# ---------- Step 4: Verify ----------
echo ""
echo "==> Verification:"
for workload in "${WORKLOADS[@]}"; do
    link="${PIMBENCH_DIR}/${workload}"
    if [ -L "$link" ] && [ -d "$link" ]; then
        echo "    OK: ${link} -> $(readlink "$link")"
    else
        echo "    FAIL: ${link} is not a valid symlink"
        exit 1
    fi
done

echo ""
echo "Setup complete. You can now build lnorm/rmsnorm from PIMbench:"
echo "  cd ${PIMBENCH_DIR}/lnorm/PIM && make perf USE_OPENMP=1"
echo "  cd ${PIMBENCH_DIR}/rmsnorm/PIM && make perf USE_OPENMP=1"
echo ""
echo "Edit your source files in-place at:"
echo "  ${SCRIPT_DIR}/lnorm/"
echo "  ${SCRIPT_DIR}/rmsnorm/"
