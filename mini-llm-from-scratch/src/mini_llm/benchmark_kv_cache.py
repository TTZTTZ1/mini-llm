from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from mini_llm.checkpoint import load_checkpoint
from mini_llm.config import load_config, project_root_from_config, resolve_project_path
from mini_llm.generate import _encode_prompt
from mini_llm.metrics import tokens_per_second
from mini_llm.model import GPTLanguageModel, config_from_model_config
from mini_llm.utils import peak_memory_mb, select_device


@torch.no_grad()
def greedy_no_cache(model: GPTLanguageModel, prompt: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    was_training = model.training
    model.eval()
    try:
        idx = prompt.clone()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.config.block_size :]
            logits, _ = model(idx_cond)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def greedy_with_cache(model: GPTLanguageModel, prompt: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    was_training = model.training
    model.eval()
    try:
        idx = prompt.clone()
        logits, _, past = model(idx, use_cache=True, start_pos=0)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, next_token), dim=1)
        start_pos = idx.size(1) - 1
        for _ in range(max_new_tokens - 1):
            logits, _, past = model(next_token, past_kv=past, use_cache=True, start_pos=start_pos)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat((idx, next_token), dim=1)
            start_pos += 1
        return idx
    finally:
        if was_training:
            model.train()


def benchmark(model: GPTLanguageModel, prompt: torch.Tensor, lengths: list[int], device: torch.device) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    model.eval()
    for length in lengths:
        effective_length = min(length, max(1, model.config.block_size - prompt.size(1)))
        for mode, fn in [("no_cache", greedy_no_cache), ("with_cache", greedy_with_cache)]:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize()
            start = time.perf_counter()
            fn(model, prompt, effective_length)
            if device.type == "cuda":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            rows.append(
                {
                    "length": length,
                    "effective_length": effective_length,
                    "mode": mode,
                    "seconds": seconds,
                    "tokens_per_second": tokens_per_second(effective_length, seconds),
                    "peak_memory_mb": peak_memory_mb(device),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = project_root_from_config(args.config)
    device = select_device(args.device)
    model = GPTLanguageModel(config_from_model_config(cfg.model)).to(device)
    checkpoint_path = resolve_project_path(project_root, args.checkpoint)
    tokenizer_path = resolve_project_path(project_root, cfg.data.tokenizer_path)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    prompt_ids = _encode_prompt(args.prompt, tokenizer_path, cfg.model.vocab_size)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    check_len = min(8, max(1, model.config.block_size - prompt.size(1)))
    no_cache = greedy_no_cache(model, prompt, check_len)
    with_cache = greedy_with_cache(model, prompt, check_len)
    if not torch.equal(no_cache, with_cache):
        raise RuntimeError("KV cache correctness check failed")
    print("correctness passed")

    rows = benchmark(model, prompt, args.lengths, device)
    output_path = resolve_project_path(project_root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["length", "effective_length", "mode", "seconds", "tokens_per_second", "peak_memory_mb"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
