from pathlib import Path

import numpy as np

from mini_llm.data import TOKEN_DTYPE
from mini_llm.pretrain_data import (
    collect_tokenizer_corpus,
    iter_local_texts,
    target_val_fraction,
    train_tokenizer,
    write_streaming_token_bins,
)


def test_iter_local_texts_yields_nonempty_lines(tmp_path):
    text_path = tmp_path / "corpus.txt"
    text_path.write_text("\nalpha beta gamma\n\nsecond document\n", encoding="utf-8")

    assert list(iter_local_texts([text_path])) == ["alpha beta gamma", "second document"]


def test_collect_tokenizer_corpus_cleans_and_limits_documents(tmp_path):
    corpus_path = tmp_path / "tokenizer_corpus.txt"
    docs = ["tiny", "alpha beta gamma delta", "second useful document"]

    stats = collect_tokenizer_corpus(
        docs,
        corpus_path,
        max_docs=2,
        max_chars=10_000,
        min_chars=5,
    )

    text = corpus_path.read_text(encoding="utf-8")
    assert "tiny" not in text
    assert "alpha beta gamma delta" in text
    assert "second useful document" in text
    assert stats.documents == 2
    assert stats.characters == len("alpha beta gamma delta") + len("second useful document")


def test_write_streaming_token_bins_hits_exact_targets(tmp_path):
    tokenizer_corpus = tmp_path / "tokenizer_corpus.txt"
    tokenizer_path = tmp_path / "toy_tokenizer.json"
    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"

    docs = [
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
        "news model training attention cache rope transformer language data",
        "another useful pretraining paragraph with enough repeated tokens",
    ] * 20
    tokenizer_corpus.write_text("\n".join(docs), encoding="utf-8")
    train_tokenizer(tokenizer_corpus, tokenizer_path, vocab_size=128)

    stats = write_streaming_token_bins(
        docs,
        tokenizer_path=tokenizer_path,
        train_bin=train_bin,
        val_bin=val_bin,
        target_train_tokens=80,
        target_val_tokens=20,
        val_fraction=None,
        min_chars=5,
    )

    train_tokens = np.memmap(train_bin, dtype=TOKEN_DTYPE, mode="r")
    val_tokens = np.memmap(val_bin, dtype=TOKEN_DTYPE, mode="r")
    assert train_tokens.shape[0] == 80
    assert val_tokens.shape[0] == 20
    assert stats.train_tokens == 80
    assert stats.val_tokens == 20
    assert stats.documents > 0


def test_target_val_fraction_matches_requested_token_budget():
    fraction = target_val_fraction(target_train_tokens=3_200_000_000, target_val_tokens=20_000_000)

    assert round(1 / fraction) == 161


def test_write_streaming_token_bins_keeps_existing_bins_on_failure(tmp_path):
    tokenizer_corpus = tmp_path / "tokenizer_corpus.txt"
    tokenizer_path = tmp_path / "toy_tokenizer.json"
    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"
    train_bin.write_bytes(b"existing-train")
    val_bin.write_bytes(b"existing-val")

    docs = ["alpha beta gamma delta epsilon"] * 3
    tokenizer_corpus.write_text("\n".join(docs), encoding="utf-8")
    train_tokenizer(tokenizer_corpus, tokenizer_path, vocab_size=128)

    import pytest

    with pytest.raises(ValueError, match="not enough text"):
        write_streaming_token_bins(
            docs,
            tokenizer_path=tokenizer_path,
            train_bin=train_bin,
            val_bin=val_bin,
            target_train_tokens=10_000,
            target_val_tokens=1_000,
            val_fraction=None,
            min_chars=5,
        )

    assert train_bin.read_bytes() == b"existing-train"
    assert val_bin.read_bytes() == b"existing-val"
    assert not train_bin.with_suffix(".bin.tmp").exists()
    assert not val_bin.with_suffix(".bin.tmp").exists()
