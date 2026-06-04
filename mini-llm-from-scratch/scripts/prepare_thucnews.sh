#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON:-python3}"
export PYTHONPATH=src

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
from mini_llm.data import clean_text

raw_dir = Path("data/raw")
out = Path("data/processed/thucnews_clean.txt")
texts = []
for path in sorted(raw_dir.rglob("*.txt")):
    texts.append(path.read_text(encoding="utf-8", errors="ignore"))
if not texts:
    raise SystemExit("no .txt files found under data/raw")
cleaned = clean_text("\n".join(texts), min_chars=10)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(cleaned, encoding="utf-8")
print("chars", len(cleaned))
PY

"$PYTHON_BIN" -m mini_llm.tokenizer_train \
  --input data/processed/thucnews_clean.txt \
  --tokenizer data/tokenizer/bpe_16000.json \
  --vocab-size 16000

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
from mini_llm.data import write_token_bins
from mini_llm.tokenizer_train import encode_file

ids = encode_file(Path("data/tokenizer/bpe_16000.json"), Path("data/processed/thucnews_clean.txt"))
write_token_bins(ids, Path("data/processed/train.bin"), Path("data/processed/val.bin"), val_fraction=0.05)
print("tokens", len(ids))
PY
