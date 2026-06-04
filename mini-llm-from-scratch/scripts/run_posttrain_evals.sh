#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python3}"
export PYTHONPATH=src

CONFIG="${CONFIG:-configs/final_212m_rope_ctx1024.yaml}"
RUN_DIR="${RUN_DIR:-results/runs/final_212m_rope_ctx1024}"
CHECKPOINT="${CHECKPOINT:-$RUN_DIR/checkpoint.pt}"
DEVICE="${DEVICE:-cuda}"
PROMPT="${PROMPT:-人工智能的发展}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
KV_LENGTHS="${KV_LENGTHS:-128 256 512 1024}"

for strategy in greedy temperature top_k top_p; do
  "$PYTHON_BIN" -m mini_llm.generate \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --prompt "$PROMPT" \
    --strategy "$strategy" \
    --device "$DEVICE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --output "$RUN_DIR/sample_${strategy}.txt"
done

"$PYTHON_BIN" -m mini_llm.benchmark_kv_cache \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --prompt "$PROMPT" \
  --lengths $KV_LENGTHS \
  --device "$DEVICE" \
  --output "$RUN_DIR/kv_cache_benchmark.csv"
