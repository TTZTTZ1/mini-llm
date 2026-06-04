#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON:-python3}"
export PYTHONPATH=src

for preset in mid_134m mid_163m mid_191m mid_219m mid_212m; do
  "$PYTHON_BIN" -m mini_llm.benchmark_train_speed \
    --preset "$preset" \
    --batch-sizes 16 24 32 40 48 56 64 \
    --block-size 1024 \
    --vocab-size 32000 \
    --warmup-steps 5 \
    --timed-steps 20 \
    --device cuda \
    --dtype bfloat16 \
    --output "results/benchmarks/train_speed_${preset}.csv"
done
