from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import torch

from mini_llm.model import GPTConfig, GPTLanguageModel
from mini_llm.utils import autocast_context, select_device, set_seed


BENCHMARK_PRESETS = {
    "small": {
        "n_layer": 8,
        "n_head": 8,
        "n_embd": 512,
        "dropout": 0.1,
        "position_encoding": "rope",
    },
    "mid_134m": {
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "dropout": 0.1,
        "position_encoding": "rope",
    },
    "mid_163m": {
        "n_layer": 16,
        "n_head": 12,
        "n_embd": 768,
        "dropout": 0.1,
        "position_encoding": "rope",
    },
    "mid_191m": {
        "n_layer": 20,
        "n_head": 12,
        "n_embd": 768,
        "dropout": 0.1,
        "position_encoding": "rope",
    },
    "mid_219m": {
        "n_layer": 24,
        "n_head": 12,
        "n_embd": 768,
        "dropout": 0.1,
        "position_encoding": "rope",
    },
    "mid_212m": {
        "n_layer": 16,
        "n_head": 14,
        "n_embd": 896,
        "dropout": 0.1,
        "position_encoding": "rope",
    },
    "large_250m": {
        "n_layer": 16,
        "n_head": 16,
        "n_embd": 1024,
        "dropout": 0.1,
        "position_encoding": "rope",
    },
    "large_320m": {
        "n_layer": 20,
        "n_head": 16,
        "n_embd": 1024,
        "dropout": 0.1,
        "position_encoding": "rope",
    },
    "large_370m": {
        "n_layer": 24,
        "n_head": 16,
        "n_embd": 1024,
        "dropout": 0.1,
        "position_encoding": "rope",
    },
    "large_450m": {
        "n_layer": 20,
        "n_head": 20,
        "n_embd": 1280,
        "dropout": 0.1,
        "position_encoding": "rope",
    },
    "large_560m": {
        "n_layer": 24,
        "n_head": 20,
        "n_embd": 1280,
        "dropout": 0.1,
        "position_encoding": "rope",
    },
}


CSV_FIELDS = [
    "preset",
    "status",
    "batch_size",
    "block_size",
    "tokens_per_step",
    "timed_steps",
    "seconds",
    "tokens_per_second",
    "peak_memory_allocated_mb",
    "peak_memory_reserved_mb",
    "error",
]


def build_preset_config(preset: str, vocab_size: int, block_size: int) -> GPTConfig:
    if preset not in BENCHMARK_PRESETS:
        choices = ", ".join(sorted(BENCHMARK_PRESETS))
        raise ValueError(f"unknown preset {preset!r}; choose one of: {choices}")
    values = BENCHMARK_PRESETS[preset]
    return GPTConfig(
        vocab_size=vocab_size,
        n_layer=values["n_layer"],
        n_head=values["n_head"],
        n_embd=values["n_embd"],
        block_size=block_size,
        dropout=values["dropout"],
        position_encoding=values["position_encoding"],
    )


def tokens_per_step(batch_size: int, block_size: int) -> int:
    return batch_size * block_size


def is_oom_error(exc: RuntimeError) -> bool:
    return "out of memory" in str(exc).lower()


def memory_stats_mb(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    allocated = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    reserved = torch.cuda.max_memory_reserved(device) / 1024 / 1024
    return allocated, reserved


def random_batch(cfg: GPTConfig, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size), dtype=torch.long, device=device)
    y = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size), dtype=torch.long, device=device)
    return x, y


def train_step(
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    cfg: GPTConfig,
    batch_size: int,
    device: torch.device,
    dtype: str,
) -> None:
    x, y = random_batch(cfg, batch_size, device)
    with autocast_context(device, dtype):
        _, loss = model(x, targets=y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def benchmark_batch_size(
    cfg: GPTConfig,
    batch_size: int,
    warmup_steps: int,
    timed_steps: int,
    device: torch.device,
    dtype: str,
    learning_rate: float,
    weight_decay: float,
    preset: str = "custom",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "preset": preset,
        "status": "ok",
        "batch_size": batch_size,
        "block_size": cfg.block_size,
        "tokens_per_step": tokens_per_step(batch_size, cfg.block_size),
        "timed_steps": timed_steps,
        "seconds": 0.0,
        "tokens_per_second": 0.0,
        "peak_memory_allocated_mb": 0.0,
        "peak_memory_reserved_mb": 0.0,
        "error": "",
    }

    model = GPTLanguageModel(cfg).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    try:
        for _ in range(warmup_steps):
            train_step(model, optimizer, cfg, batch_size, device, dtype)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(timed_steps):
            train_step(model, optimizer, cfg, batch_size, device, dtype)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        allocated, reserved = memory_stats_mb(device)
        row["seconds"] = seconds
        row["tokens_per_second"] = row["tokens_per_step"] * timed_steps / max(seconds, 1e-9)
        row["peak_memory_allocated_mb"] = allocated
        row["peak_memory_reserved_mb"] = reserved
    except RuntimeError as exc:
        if not is_oom_error(exc):
            raise
        row["status"] = "oom"
        row["error"] = str(exc).splitlines()[0]
        if device.type == "cuda":
            torch.cuda.empty_cache()
            allocated, reserved = memory_stats_mb(device)
            row["peak_memory_allocated_mb"] = allocated
            row["peak_memory_reserved_mb"] = reserved
    finally:
        del optimizer
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return row


def write_csv(output: Path, rows: list[dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(
    preset: str,
    batch_sizes: list[int],
    block_size: int,
    vocab_size: int,
    warmup_steps: int,
    timed_steps: int,
    device: torch.device,
    dtype: str,
    learning_rate: float,
    weight_decay: float,
    stop_on_oom: bool,
) -> list[dict[str, Any]]:
    cfg = build_preset_config(preset, vocab_size=vocab_size, block_size=block_size)
    rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        print(f"benchmark preset={preset} batch_size={batch_size} block_size={block_size}")
        row = benchmark_batch_size(
            cfg,
            batch_size=batch_size,
            warmup_steps=warmup_steps,
            timed_steps=timed_steps,
            device=device,
            dtype=dtype,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            preset=preset,
        )
        rows.append(row)
        print(
            "status={status} tokens_per_second={tokens_per_second:.2f} "
            "peak_reserved_mb={peak_memory_reserved_mb:.1f}".format(**row)
        )
        if row["status"] == "oom" and stop_on_oom:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="large_250m", choices=sorted(BENCHMARK_PRESETS))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[4, 8, 12, 16, 20, 24, 32])
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--continue-after-oom", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/benchmarks/train_speed_large_250m.csv"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    rows = run_sweep(
        preset=args.preset,
        batch_sizes=args.batch_sizes,
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        warmup_steps=args.warmup_steps,
        timed_steps=args.timed_steps,
        device=device,
        dtype=args.dtype,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        stop_on_oom=not args.continue_after_oom,
    )
    write_csv(args.output, rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
