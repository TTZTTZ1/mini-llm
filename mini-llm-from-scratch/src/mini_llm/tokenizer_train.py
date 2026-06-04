from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def train_bpe_tokenizer(files: list[Path], output_path: Path, vocab_size: int) -> None:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train([str(path) for path in files], trainer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))


def encode_file(tokenizer_path: Path, text_path: Path) -> list[int]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    text = text_path.read_text(encoding="utf-8")
    return tokenizer.encode(text).ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--vocab-size", required=True, type=int)
    parser.add_argument("--output-ids", type=Path)
    args = parser.parse_args()

    train_bpe_tokenizer([args.input], args.tokenizer, args.vocab_size)
    ids = encode_file(args.tokenizer, args.input)
    if args.output_ids:
        args.output_ids.parent.mkdir(parents=True, exist_ok=True)
        args.output_ids.write_text(" ".join(map(str, ids)), encoding="utf-8")
    print(f"tokens {len(ids)}")


if __name__ == "__main__":
    main()
