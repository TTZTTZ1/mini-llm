#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON:-python3}"
export PYTHONPATH=src

for strategy in greedy temperature top_k top_p; do
  "$PYTHON_BIN" -m mini_llm.generate \
    --config configs/small.yaml \
    --checkpoint results/runs/small_main/checkpoint.pt \
    --prompt "人工智能的发展" \
    --strategy "$strategy" \
    --device cuda \
    --output "results/runs/small_main/sample_${strategy}.txt"
done
