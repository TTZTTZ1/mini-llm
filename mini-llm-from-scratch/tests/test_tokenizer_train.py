from pathlib import Path

from mini_llm.tokenizer_train import encode_file, train_bpe_tokenizer


def test_train_bpe_tokenizer_and_encode(tmp_path):
    corpus = Path("tests/fixtures/tiny_corpus.txt")
    tokenizer_path = tmp_path / "tok.json"
    train_bpe_tokenizer([corpus], tokenizer_path, vocab_size=128)
    assert tokenizer_path.exists()

    ids = encode_file(tokenizer_path, corpus)
    assert len(ids) > 10
    assert all(isinstance(x, int) for x in ids)
