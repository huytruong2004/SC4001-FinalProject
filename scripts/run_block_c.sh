#!/usr/bin/env bash
set -euo pipefail

DATA=${SC4001_DATA:-/workspace/data/flowers-102}
CKPT=${SC4001_CKPT:-/workspace/checkpoints}
REPO=${SC4001_REPO:-/workspace/SC4001-FinalProject}
RESULTS="$REPO/results/runs.jsonl"

cd "$REPO"
mkdir -p "$(dirname "$RESULTS")" "$CKPT"

for cfg in A5_cutmix_attsup; do
  for seed in 0 1 2; do
    echo "=== $cfg seed=$seed ==="
    mkdir -p "$CKPT/${cfg}_seed${seed}"
    python -m src.train --config configs/${cfg}.yaml --seed $seed \
      --data-root "$DATA" --checkpoint-dir "$CKPT" \
      --results-path "$RESULTS" \
      2>&1 | tee -a "$CKPT/${cfg}_seed${seed}/train.stdout.log"
  done
done
