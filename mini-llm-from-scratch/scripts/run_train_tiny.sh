#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON:-python3}"
export PYTHONPATH=src

"$PYTHON_BIN" -m mini_llm.train --config configs/tiny.yaml --device cuda
