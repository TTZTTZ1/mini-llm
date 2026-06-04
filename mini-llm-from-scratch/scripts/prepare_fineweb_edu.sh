#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python3}"
export PYTHONPATH=src

DATASET_NAME="${DATASET_NAME:-HuggingFaceFW/fineweb-edu}"
DATASET_CONFIG="${DATASET_CONFIG:-sample-10BT}"
SPLIT="${SPLIT:-train}"
TEXT_COLUMN="${TEXT_COLUMN:-text}"
VOCAB_SIZE="${VOCAB_SIZE:-32000}"
TOKENIZER_TRAIN_DOCS="${TOKENIZER_TRAIN_DOCS:-200000}"
TOKENIZER_TRAIN_CHARS="${TOKENIZER_TRAIN_CHARS:-50000000}"
TARGET_TRAIN_TOKENS="${TARGET_TRAIN_TOKENS:-3200000000}"
TARGET_VAL_TOKENS="${TARGET_VAL_TOKENS:-20000000}"
VAL_FRACTION="${VAL_FRACTION:-}"
MIN_CHARS="${MIN_CHARS:-128}"

TOKENIZER_CORPUS="${TOKENIZER_CORPUS:-data/processed/fineweb_edu_tokenizer_corpus.txt}"
TOKENIZER_PATH="${TOKENIZER_PATH:-data/tokenizer/fineweb_edu_bpe_32000.json}"
TRAIN_BIN="${TRAIN_BIN:-data/processed/fineweb_edu_train.bin}"
VAL_BIN="${VAL_BIN:-data/processed/fineweb_edu_val.bin}"
MANIFEST="${MANIFEST:-data/processed/fineweb_edu_manifest.json}"

args=(
  --dataset-name "$DATASET_NAME"
  --dataset-config "$DATASET_CONFIG"
  --split "$SPLIT"
  --text-column "$TEXT_COLUMN"
  --tokenizer-corpus "$TOKENIZER_CORPUS"
  --tokenizer "$TOKENIZER_PATH"
  --train-bin "$TRAIN_BIN"
  --val-bin "$VAL_BIN"
  --manifest "$MANIFEST"
  --vocab-size "$VOCAB_SIZE"
  --tokenizer-train-docs "$TOKENIZER_TRAIN_DOCS"
  --tokenizer-train-chars "$TOKENIZER_TRAIN_CHARS"
  --target-train-tokens "$TARGET_TRAIN_TOKENS"
  --target-val-tokens "$TARGET_VAL_TOKENS"
  --min-chars "$MIN_CHARS"
)

if [ -n "$VAL_FRACTION" ]; then
  args+=(--val-fraction "$VAL_FRACTION")
fi

if [ "${FORCE_TOKENIZER:-0}" = "1" ]; then
  args+=(--force-tokenizer)
fi

if [ -n "${LOCAL_TEXT_FILE:-}" ]; then
  args+=(--local-text "$LOCAL_TEXT_FILE")
elif [ -n "${LOCAL_TEXT:-}" ]; then
  read -r -a local_text_paths <<< "$LOCAL_TEXT"
  args+=(--local-text "${local_text_paths[@]}")
fi

"$PYTHON_BIN" -m mini_llm.pretrain_data "${args[@]}"
