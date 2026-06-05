#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python3}"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

DEVICE="${DEVICE:-cuda}"
DEFAULT_CONFIGS="configs/final_212m_rope_ctx1024.yaml configs/ablation_212m_pos_abs.yaml configs/ablation_212m_pos_none.yaml configs/ablation_212m_context_512.yaml configs/ablation_212m_context_1536.yaml configs/scale_134m_rope_ctx1024.yaml"
CONFIGS="${CONFIGS:-$DEFAULT_CONFIGS}"
LOG_DIR="${LOG_DIR:-results/logs}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$LOG_DIR"

for cfg in $CONFIGS; do
  echo "Running experiment config: $cfg"
  config_name="$(basename "$cfg" .yaml)"
  log_file="$LOG_DIR/${config_name}_${TIMESTAMP}.log"
  args=(--config "$cfg" --device "$DEVICE")
  if [ -n "${MAX_STEPS_OVERRIDE:-}" ]; then
    args+=(--max_steps "$MAX_STEPS_OVERRIDE")
  fi
  {
    echo "started_at=$(date +"%Y-%m-%dT%H:%M:%S%z")"
    echo "config=$cfg"
    echo "device=$DEVICE"
    echo "log_file=$log_file"
    echo "command=$PYTHON_BIN -m mini_llm.train ${args[*]}"
  } | tee "$log_file"
  "$PYTHON_BIN" -m mini_llm.train "${args[@]}" 2>&1 | tee -a "$log_file"
done
