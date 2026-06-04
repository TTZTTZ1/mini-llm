#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python3}"
export PYTHONPATH=src

CONFIG="${CONFIG:-configs/final_212m_rope_ctx1024.yaml}"
DEVICE="${DEVICE:-cuda}"

args=(--config "$CONFIG" --device "$DEVICE")
if [ -n "${MAX_STEPS_OVERRIDE:-}" ]; then
  args+=(--max_steps "$MAX_STEPS_OVERRIDE")
fi
if [ -n "${OUT_DIR_OVERRIDE:-}" ]; then
  args+=(--out_dir "$OUT_DIR_OVERRIDE")
fi

"$PYTHON_BIN" -m mini_llm.train "${args[@]}"
