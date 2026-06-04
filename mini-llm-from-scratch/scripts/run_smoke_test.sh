#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON:-python3}"
export PYTHONPATH=src

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
from mini_llm.data import write_token_bins

ids = [i % 128 for i in range(4096)]
Path("data/processed").mkdir(parents=True, exist_ok=True)
write_token_bins(ids, Path("data/processed/smoke_train.bin"), Path("data/processed/smoke_val.bin"), val_fraction=0.1)
PY

"$PYTHON_BIN" -m pytest -q
"$PYTHON_BIN" -m mini_llm.train --config configs/smoke.yaml --max_steps 20 --device cpu
