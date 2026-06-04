#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python3}"
export PYTHONPATH=src

CONFIG="${CONFIG:-configs/final_212m_rope_ctx1024.yaml}"
DEVICE="${DEVICE:-cuda}"
DEFAULT_RUN_DIR="results/runs/final_212m_rope_ctx1024"
RUN_DIR="${OUT_DIR_OVERRIDE:-$DEFAULT_RUN_DIR}"
LOG_DIR="${LOG_DIR:-$RUN_DIR}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_${TIMESTAMP}.log}"

args=(--config "$CONFIG" --device "$DEVICE")
if [ -n "${MAX_STEPS_OVERRIDE:-}" ]; then
  args+=(--max_steps "$MAX_STEPS_OVERRIDE")
fi
if [ -n "${OUT_DIR_OVERRIDE:-}" ]; then
  args+=(--out_dir "$OUT_DIR_OVERRIDE")
fi

mkdir -p "$LOG_DIR"
{
  echo "started_at=$(date +"%Y-%m-%dT%H:%M:%S%z")"
  echo "config=$CONFIG"
  echo "device=$DEVICE"
  echo "log_file=$LOG_FILE"
  echo "command=$PYTHON_BIN -m mini_llm.train ${args[*]}"
} | tee "$LOG_FILE"

"$PYTHON_BIN" -m mini_llm.train "${args[@]}" 2>&1 | tee -a "$LOG_FILE"
