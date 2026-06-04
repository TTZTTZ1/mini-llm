from pathlib import Path

import numpy as np
import torch

from mini_llm.data import TokenBlockDataset, clean_text, write_token_bins


def test_clean_text_removes_blank_lines_and_controls():
    raw = "第一行\n\n\x00第二行\r\n  \n第三行"
    cleaned = clean_text(raw, min_chars=2)
    assert cleaned == "第一行\n第二行\n第三行"


def test_write_token_bins_and_block_dataset(tmp_path):
    ids = list(range(100))
    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"
    write_token_bins(ids, train_bin, val_bin, val_fraction=0.2)
    assert np.memmap(train_bin, dtype=np.uint32, mode="r").shape[0] == 80
    assert np.memmap(val_bin, dtype=np.uint32, mode="r").shape[0] == 20

    ds = TokenBlockDataset(train_bin, block_size=8)
    x, y = ds[0]
    assert x.shape == torch.Size([8])
    assert y.shape == torch.Size([8])
    assert torch.equal(y, x + 1)
