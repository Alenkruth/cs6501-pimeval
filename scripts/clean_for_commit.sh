#!/usr/bin/env bash
# clean_for_commit.sh — Remove symlinks and build artifacts before committing.
# Re-run setup.sh after committing to restore the symlinks.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIMBENCH_DIR="${REPO_ROOT}/PIMeval-PIMbench/PIMbench"

WORKLOADS=(lnorm rmsnorm)

# ---------- Remove symlinks from PIMbench ----------
for workload in "${WORKLOADS[@]}"; do
    link="${PIMBENCH_DIR}/${workload}"
    if [ -L "$link" ]; then
        rm "$link"
        echo "Removed symlink: ${link}"
    fi
done

echo "Clean. Run setup.sh to restore symlinks after committing."
