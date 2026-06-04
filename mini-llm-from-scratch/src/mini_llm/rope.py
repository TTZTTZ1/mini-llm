from __future__ import annotations

import torch


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    half_dim = head_dim // 2
    freq_seq = torch.arange(half_dim, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (freq_seq / half_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    start_pos: int = 0,
) -> torch.Tensor:
    seq_len = x.size(-2)
    x_float = x.float()
    x_even = x_float[..., 0::2]
    x_odd = x_float[..., 1::2]
    cos_slice = cos[start_pos : start_pos + seq_len].view(1, 1, seq_len, -1)
    sin_slice = sin[start_pos : start_pos + seq_len].view(1, 1, seq_len, -1)
    rotated_even = x_even * cos_slice - x_odd * sin_slice
    rotated_odd = x_even * sin_slice + x_odd * cos_slice
    out = torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)
    return out.to(dtype=x.dtype)
