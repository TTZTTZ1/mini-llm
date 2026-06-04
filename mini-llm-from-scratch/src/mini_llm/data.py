from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def clean_text(text: str, min_chars: int = 5) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    kept_chars: list[str] = []
    for ch in text:
        if ch in {"\n", "\t"}:
            kept_chars.append(ch)
            continue
        if unicodedata.category(ch).startswith("C"):
            continue
        kept_chars.append(ch)
    lines = []
    for line in "".join(kept_chars).split("\n"):
        stripped = line.strip()
        if stripped and len(stripped) >= min_chars:
            lines.append(stripped)
    return "\n".join(lines)


TOKEN_DTYPE = np.uint32


def write_token_bins(ids: list[int], train_bin: Path, val_bin: Path, val_fraction: float) -> None:
    if not ids:
        raise ValueError("ids must not be empty")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    boundary = int(len(ids) * (1 - val_fraction))
    if boundary <= 0 or boundary >= len(ids):
        raise ValueError("token split would create an empty train or val set")
    train_bin.parent.mkdir(parents=True, exist_ok=True)
    val_bin.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(ids[:boundary], dtype=TOKEN_DTYPE).tofile(train_bin)
    np.asarray(ids[boundary:], dtype=TOKEN_DTYPE).tofile(val_bin)


class TokenBlockDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, bin_path: Path, block_size: int, dtype: np.dtype = TOKEN_DTYPE):
        self.bin_path = Path(bin_path)
        self.block_size = block_size
        self.data = np.memmap(self.bin_path, dtype=dtype, mode="r")
        if self.data.shape[0] <= block_size:
            raise ValueError(f"{bin_path} must contain more tokens than block_size")

    def __len__(self) -> int:
        return max(0, int(self.data.shape[0]) - self.block_size)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        chunk = np.asarray(self.data[idx : idx + self.block_size + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1].copy()).long()
        y = torch.from_numpy(chunk[1:].copy()).long()
        return x, y


def random_batch(dataset: TokenBlockDataset, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randint(0, len(dataset), (batch_size,))
    xs, ys = zip(*(dataset[int(i)] for i in indices), strict=True)
    return torch.stack(xs).to(device), torch.stack(ys).to(device)
