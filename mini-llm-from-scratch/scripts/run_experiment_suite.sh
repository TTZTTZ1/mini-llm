#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python3}"
export PYTHONPATH=src

DEVICE="${DEVICE:-cuda}"
DEFAULT_CONFIGS="configs/final_212m_rope_ctx1024.yaml configs/ablation_212m_pos_abs.yaml configs/ablation_212m_pos_none.yaml configs/ablation_212m_context_512.yaml configs/ablation_212m_context_1536.yaml configs/scale_134m_rope_ctx1024.yaml"
CONFIGS="${CONFIGS:-$DEFAULT_CONFIGS}"

for cfg in $CONFIGS; do
  echo "Running experiment config: $cfg"
  args=(--config "$cfg" --device "$DEVICE")
  if [ -n "${MAX_STEPS_OVERRIDE:-}" ]; then
    args+=(--max_steps "$MAX_STEPS_OVERRIDE")
  fi
  "$PYTHON_BIN" -m mini_llm.train "${args[@]}"
done
