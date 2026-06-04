from __future__ import annotations

import math


def perplexity(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def tokens_per_second(num_tokens: int, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0:
        return 0.0
    return num_tokens / elapsed_seconds


def metrics_header() -> list[str]:
    return ["step", "train_loss", "val_loss", "val_ppl", "tokens_per_second", "peak_memory_mb"]
