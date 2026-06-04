from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from mini_llm.data import TOKEN_DTYPE, clean_text
from mini_llm.tokenizer_train import SPECIAL_TOKENS, train_bpe_tokenizer


DEFAULT_DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DEFAULT_DATASET_CONFIG = "sample-10BT"
DEFAULT_TEXT_COLUMN = "text"


@dataclass(frozen=True)
class CorpusStats:
    documents: int
    characters: int


@dataclass(frozen=True)
class TokenBinStats:
    documents: int
    train_tokens: int
    val_tokens: int
    tokenizer_path: str
    train_bin: str
    val_bin: str


def iter_local_texts(paths: list[Path]) -> Iterator[str]:
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    yield text


def iter_hf_texts(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    text_column: str,
    streaming: bool,
) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the `datasets` package to stream HuggingFace datasets") from exc

    kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
    if dataset_config:
        kwargs["name"] = dataset_config
    dataset = load_dataset(dataset_name, **kwargs)
    for row in dataset:
        text = row.get(text_column)
        if isinstance(text, str) and text.strip():
            yield text


def collect_tokenizer_corpus(
    texts: Iterable[str],
    output_path: Path,
    max_docs: int,
    max_chars: int,
    min_chars: int,
) -> CorpusStats:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    documents = 0
    characters = 0
    with output_path.open("w", encoding="utf-8") as f:
        for raw_text in texts:
            text = clean_text(raw_text, min_chars=min_chars)
            if not text:
                continue
            if documents >= max_docs or characters >= max_chars:
                break
            remaining = max_chars - characters
            if len(text) > remaining:
                text = text[:remaining]
            f.write(text)
            f.write("\n")
            documents += 1
            characters += len(text)
    if documents == 0:
        raise ValueError("tokenizer corpus is empty after cleaning")
    return CorpusStats(documents=documents, characters=characters)


def train_tokenizer(corpus_path: Path, tokenizer_path: Path, vocab_size: int) -> None:
    train_bpe_tokenizer([corpus_path], tokenizer_path, vocab_size)


def _write_ids(handle, ids: list[int], limit: int) -> int:
    if limit <= 0 or not ids:
        return 0
    trimmed = ids[:limit]
    np.asarray(trimmed, dtype=TOKEN_DTYPE).tofile(handle)
    return len(trimmed)


def target_val_fraction(target_train_tokens: int, target_val_tokens: int) -> float:
    if target_train_tokens <= 0:
        raise ValueError("target_train_tokens must be positive")
    if target_val_tokens <= 0:
        raise ValueError("target_val_tokens must be positive")
    return target_val_tokens / (target_train_tokens + target_val_tokens)


def _val_stride(val_fraction: float | None, target_train_tokens: int, target_val_tokens: int) -> int:
    if val_fraction is None:
        val_fraction = target_val_fraction(target_train_tokens, target_val_tokens)
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    return max(2, round(1 / val_fraction))


def write_streaming_token_bins(
    texts: Iterable[str],
    tokenizer_path: Path,
    train_bin: Path,
    val_bin: Path,
    target_train_tokens: int,
    target_val_tokens: int,
    val_fraction: float | None,
    min_chars: int,
) -> TokenBinStats:
    if target_train_tokens <= 0:
        raise ValueError("target_train_tokens must be positive")
    if target_val_tokens <= 0:
        raise ValueError("target_val_tokens must be positive")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError(f"tokenizer must contain <eos>; expected special tokens {SPECIAL_TOKENS}")

    train_bin.parent.mkdir(parents=True, exist_ok=True)
    val_bin.parent.mkdir(parents=True, exist_ok=True)

    documents = 0
    train_tokens = 0
    val_tokens = 0
    stride = _val_stride(val_fraction, target_train_tokens, target_val_tokens)
    train_tmp = train_bin.with_suffix(train_bin.suffix + ".tmp")
    val_tmp = val_bin.with_suffix(val_bin.suffix + ".tmp")
    for tmp_path in (train_tmp, val_tmp):
        tmp_path.unlink(missing_ok=True)

    try:
        with train_tmp.open("wb") as train_f, val_tmp.open("wb") as val_f:
            for raw_text in texts:
                if train_tokens >= target_train_tokens and val_tokens >= target_val_tokens:
                    break
                text = clean_text(raw_text, min_chars=min_chars)
                if not text:
                    continue
                ids = tokenizer.encode(text).ids
                if not ids:
                    continue
                ids.append(eos_id)
                route_to_val = documents % stride == stride - 1
                if val_tokens >= target_val_tokens:
                    route_to_val = False
                if train_tokens >= target_train_tokens:
                    route_to_val = True

                if route_to_val:
                    written = _write_ids(val_f, ids, target_val_tokens - val_tokens)
                    val_tokens += written
                else:
                    written = _write_ids(train_f, ids, target_train_tokens - train_tokens)
                    train_tokens += written
                if written:
                    documents += 1

            if train_tokens < target_train_tokens or val_tokens < target_val_tokens:
                raise ValueError(
                    "not enough text to build token bins: "
                    f"train={train_tokens}/{target_train_tokens}, val={val_tokens}/{target_val_tokens}"
                )
            train_f.flush()
            val_f.flush()
            os.fsync(train_f.fileno())
            os.fsync(val_f.fileno())
        train_tmp.replace(train_bin)
        val_tmp.replace(val_bin)
    except Exception:
        train_tmp.unlink(missing_ok=True)
        val_tmp.unlink(missing_ok=True)
        raise
    return TokenBinStats(
        documents=documents,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        tokenizer_path=str(tokenizer_path),
        train_bin=str(train_bin),
        val_bin=str(val_bin),
    )


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_text_iterator(args: argparse.Namespace) -> Iterator[str]:
    if args.local_text:
        return iter_local_texts(args.local_text)
    return iter_hf_texts(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        text_column=args.text_column,
        streaming=not args.no_streaming,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare streaming pretraining data for mini-LLM experiments.")
    parser.add_argument("--local-text", nargs="*", type=Path)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--tokenizer-corpus", type=Path, default=Path("data/processed/fineweb_edu_tokenizer_corpus.txt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/fineweb_edu_bpe_32000.json"))
    parser.add_argument("--train-bin", type=Path, default=Path("data/processed/fineweb_edu_train.bin"))
    parser.add_argument("--val-bin", type=Path, default=Path("data/processed/fineweb_edu_val.bin"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/fineweb_edu_manifest.json"))
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--tokenizer-train-docs", type=int, default=200_000)
    parser.add_argument("--tokenizer-train-chars", type=int, default=50_000_000)
    parser.add_argument("--target-train-tokens", type=int, default=3_200_000_000)
    parser.add_argument("--target-val-tokens", type=int, default=20_000_000)
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--min-chars", type=int, default=128)
    parser.add_argument("--force-tokenizer", action="store_true")
    args = parser.parse_args()

    if args.force_tokenizer or not args.tokenizer.exists():
        corpus_stats = collect_tokenizer_corpus(
            _build_text_iterator(args),
            args.tokenizer_corpus,
            max_docs=args.tokenizer_train_docs,
            max_chars=args.tokenizer_train_chars,
            min_chars=args.min_chars,
        )
        train_tokenizer(args.tokenizer_corpus, args.tokenizer, args.vocab_size)
    else:
        corpus_stats = CorpusStats(documents=0, characters=0)

    token_stats = write_streaming_token_bins(
        _build_text_iterator(args),
        tokenizer_path=args.tokenizer,
        train_bin=args.train_bin,
        val_bin=args.val_bin,
        target_train_tokens=args.target_train_tokens,
        target_val_tokens=args.target_val_tokens,
        val_fraction=args.val_fraction,
        min_chars=args.min_chars,
    )
    write_manifest(
        args.manifest,
        {
            "dataset_name": args.dataset_name if not args.local_text else "local_text",
            "dataset_config": args.dataset_config if not args.local_text else None,
            "split": args.split,
            "text_column": args.text_column,
            "vocab_size": args.vocab_size,
            "tokenizer_corpus": asdict(corpus_stats),
            "token_bins": asdict(token_stats),
        },
    )
    print(
        "prepared "
        f"train_tokens={token_stats.train_tokens} "
        f"val_tokens={token_stats.val_tokens} "
        f"documents={token_stats.documents}"
    )


if __name__ == "__main__":
    main()
