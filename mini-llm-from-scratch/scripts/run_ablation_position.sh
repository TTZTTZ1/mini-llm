#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON:-python3}"
export PYTHONPATH=src

for cfg in configs/ablation_pos_none.yaml configs/ablation_pos_abs.yaml configs/ablation_pos_rope.yaml; do
  "$PYTHON_BIN" -m mini_llm.train --config "$cfg" --device cuda
done
