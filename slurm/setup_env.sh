#!/usr/bin/env bash
# Build the Python environment and check it. Idempotent; safe to re-run.
#
#   bash slurm/setup_env.sh
#
# Needs uv (https://docs.astral.sh/uv/) on PATH. Run it from a node with internet
# access; on a cluster, that is usually the login node.

set -euo pipefail
cd "$(dirname "$0")/.."

# uv.lock pins every package; its torch is 2.6.0+cu124 (CUDA 12.4), the build the
# experiments ran with, which needs an NVIDIA driver >= 525.
echo "=== uv sync ==="
command -v uv >/dev/null || { echo "error: uv is not on PATH" >&2; exit 1; }
uv sync --extra dev

echo "=== directories ==="
mkdir -p logs out/runs out/calib out/stats out/subspaces

echo "=== environment ==="
uv run --no-sync python - <<'PY'
import sys
import torch
import transformers
print(f"python       : {sys.version.split()[0]}")
print(f"torch        : {torch.__version__} (CUDA {torch.version.cuda})")
print(f"transformers : {transformers.__version__}")
print(f"gpu visible  : {torch.cuda.is_available()}")
PY

echo "=== CPU tests ==="
uv run --no-sync python -m pytest -q

cat <<'EOF'

Next:
  1. On a GPU node, check that the build runs kernels:
       uv run --no-sync python scripts/verify_gpu.py
  2. With internet access, cache the models and data (Llama needs a Hugging Face
     token with access to the gated meta-llama repositories):
       uv run --no-sync python scripts/prefetch.py
  3. Run a two-minute end-to-end check:
       uv run --no-sync python scripts/submit.py smoke --local
EOF
