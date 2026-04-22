#!/usr/bin/env bash
set -euo pipefail
REPO=${SC4001_REPO:-/workspace/SC4001-FinalProject}
DATA=${SC4001_DATA:-/workspace/data/flowers-102}
CKPT=${SC4001_CKPT:-/workspace/checkpoints}
cd "$REPO"
python -m src.analyze \
  --runs-jsonl "$REPO/results/runs.jsonl" \
  --results-dir "$REPO/results" \
  --figures-dir "$REPO/figures" \
  --checkpoint-dir "$CKPT" \
  --data-root "$DATA"
