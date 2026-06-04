from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from mini_llm.checkpoint import save_checkpoint
from mini_llm.config import load_config, project_root_from_config, resolve_project_path
from mini_llm.data import TokenBlockDataset, random_batch
from mini_llm.metrics import metrics_header, perplexity, tokens_per_second
from mini_llm.model import GPTLanguageModel, config_from_model_config
from mini_llm.utils import autocast_context, peak_memory_mb, select_device, set_seed


@torch.no_grad()
def estimate_loss(model, dataset: TokenBlockDataset, batch_size: int, eval_iters: int, device: torch.device, dtype: str) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        xb, yb = random_batch(dataset, batch_size, device)
        with autocast_context(device, dtype):
            _, loss = model(xb, targets=yb)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def train(config_path: Path, max_steps: int | None = None, device_override: str | None = None, out_dir_override: Path | None = None) -> Path:
    cfg = load_config(config_path)
    project_root = project_root_from_config(config_path)
    train_cfg = cfg.train
    set_seed(train_cfg.seed)
    device = select_device(device_override or train_cfg.device)
    out_dir = resolve_project_path(project_root, out_dir_override or train_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data = TokenBlockDataset(resolve_project_path(project_root, cfg.data.train_bin), cfg.model.block_size)
    val_data = TokenBlockDataset(resolve_project_path(project_root, cfg.data.val_bin), cfg.model.block_size)
    model = GPTLanguageModel(config_from_model_config(cfg.model)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    steps = max_steps or train_cfg.max_steps
    metrics_path = out_dir / "metrics.csv"
    best_val = float("inf")
    start_time = time.perf_counter()

    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(metrics_header())

        for step in range(1, steps + 1):
            xb, yb = random_batch(train_data, train_cfg.batch_size, device)
            with autocast_context(device, train_cfg.dtype):
                _, loss = model(xb, targets=yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()

            if step % train_cfg.eval_interval == 0 or step == steps:
                val_loss = estimate_loss(model, val_data, train_cfg.batch_size, train_cfg.eval_iters, device, train_cfg.dtype)
                elapsed = time.perf_counter() - start_time
                seen_tokens = step * train_cfg.batch_size * cfg.model.block_size
                row = [
                    step,
                    float(loss.item()),
                    val_loss,
                    perplexity(val_loss),
                    tokens_per_second(seen_tokens, elapsed),
                    peak_memory_mb(device),
                ]
                writer.writerow(row)
                f.flush()
                print(f"step={step} train_loss={row[1]:.4f} val_loss={val_loss:.4f} val_ppl={row[3]:.2f}")
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(out_dir / "checkpoint.pt", model, optimizer, cfg, step, val_loss)

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--out_dir", type=Path)
    args = parser.parse_args()
    train(args.config, max_steps=args.max_steps, device_override=args.device, out_dir_override=args.out_dir)


if __name__ == "__main__":
    main()
