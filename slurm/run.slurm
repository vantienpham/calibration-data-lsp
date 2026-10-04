#!/bin/bash
# Generic job runner used by scripts/submit.py:
#
#   sbatch --job-name=NAME --time=... [--partition=...] slurm/run.slurm \
#     scripts/<script>.py <args...>
#
# Everything after the script name is passed to the interpreter unchanged, and the
# command is echoed into the log, so every job records what produced it. Resource
# flags go on the sbatch command line; the #SBATCH lines below are defaults.

#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-08:00:00

set -euo pipefail

# Run from the submission directory (the repository root), so relative paths in
# the plans resolve against it.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs out/runs

# Compute nodes often have no internet access. Offline mode makes a missing cache
# entry fail at once with its name, instead of after a network timeout. Warm the
# cache first with scripts/prefetch.py on a node that has access.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

echo "job       : ${SLURM_JOB_NAME:-?} (${SLURM_JOB_ID:-?})"
echo "node      : $(hostname)"
echo "started   : $(date -Is)"
echo "git       : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "command   : uv run --no-sync $*"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
echo

# --no-sync: use the environment built by slurm/setup_env.sh as it is, instead of
# re-syncing it at every job start (concurrent jobs would race on it).
uv run --no-sync "$@"

echo
echo "finished  : $(date -Is)"
