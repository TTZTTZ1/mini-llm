#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON:-python3}"
export PYTHONPATH=src

for cfg in configs/ablation_context_128.yaml configs/ablation_context_256.yaml configs/ablation_context_512.yaml; do
  "$PYTHON_BIN" -m mini_llm.train --config "$cfg" --device cuda
done
