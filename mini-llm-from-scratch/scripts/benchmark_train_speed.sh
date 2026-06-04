#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON:-python3}"
export PYTHONPATH=src

for preset in large_250m large_320m large_370m large_450m large_560m; do
  "$PYTHON_BIN" -m mini_llm.benchmark_train_speed \
    --preset "$preset" \
    --batch-sizes 4 8 12 16 20 24 32 \
    --block-size 1024 \
    --vocab-size 32000 \
    --warmup-steps 5 \
    --timed-steps 20 \
    --device cuda \
    --dtype bfloat16 \
    --output "results/benchmarks/train_speed_${preset}.csv"
done
