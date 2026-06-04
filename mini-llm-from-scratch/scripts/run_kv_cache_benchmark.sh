#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON:-python3}"
export PYTHONPATH=src

"$PYTHON_BIN" -m mini_llm.benchmark_kv_cache \
  --config configs/small.yaml \
  --checkpoint results/runs/small_main/checkpoint.pt \
  --prompt "人工智能的发展" \
  --lengths 64 128 256 512 \
  --device cuda \
  --output results/runs/small_main/kv_cache_benchmark.csv
