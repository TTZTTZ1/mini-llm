from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mini_llm.checkpoint import load_checkpoint
from mini_llm.config import load_config, project_root_from_config, resolve_project_path
from mini_llm.model import GPTLanguageModel, config_from_model_config
from mini_llm.utils import select_device


def filter_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, k)
    threshold = values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def filter_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    mask = cumulative > p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    out = torch.full_like(logits, float("-inf"))
    return out.scatter(-1, sorted_indices, sorted_logits)


def sample_next_token(
    logits: torch.Tensor,
    strategy: str,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    if strategy == "greedy":
        return torch.argmax(logits, dim=-1, keepdim=True)
    scaled = logits / max(temperature, 1e-6)
    if strategy == "top_k":
        scaled = filter_top_k(scaled, top_k)
    elif strategy == "top_p":
        scaled = filter_top_p(scaled, top_p)
    elif strategy != "temperature":
        raise ValueError("strategy must be greedy, temperature, top_k, or top_p")
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_tokens(
    model: GPTLanguageModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    strategy: str,
    temperature: float,
    top_k: int,
    top_p: float,
    use_cache: bool = False,
) -> torch.Tensor:
    model.eval()
    idx = input_ids
    past = None
    if use_cache:
        logits, _, past = model(idx, use_cache=True, start_pos=0)
        next_token = sample_next_token(logits[:, -1, :], strategy, temperature, top_k, top_p)
        idx = torch.cat((idx, next_token), dim=1)
        start_pos = idx.size(1) - 1
        for _ in range(max_new_tokens - 1):
            logits, _, past = model(next_token, past_kv=past, use_cache=True, start_pos=start_pos)
            next_token = sample_next_token(logits[:, -1, :], strategy, temperature, top_k, top_p)
            idx = torch.cat((idx, next_token), dim=1)
            start_pos += 1
        return idx

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        next_token = sample_next_token(logits[:, -1, :], strategy, temperature, top_k, top_p)
        idx = torch.cat((idx, next_token), dim=1)
    return idx


def _encode_prompt(prompt: str, tokenizer_path: Path | None, vocab_size: int) -> list[int]:
    if tokenizer_path is not None and tokenizer_path.exists():
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer.encode(prompt).ids
    return [ord(ch) % vocab_size for ch in prompt] or [0]


def _decode_tokens(ids: list[int], tokenizer_path: Path | None) -> str:
    if tokenizer_path is not None and tokenizer_path.exists():
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer.decode(ids)
    return " ".join(map(str, ids))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--strategy", default="greedy")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--output", type=Path)
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
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    max_new_tokens = args.max_new_tokens or cfg.generate.max_new_tokens
    out = generate_tokens(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        strategy=args.strategy,
        temperature=cfg.generate.temperature,
        top_k=cfg.generate.top_k,
        top_p=cfg.generate.top_p,
    )
    text = _decode_tokens(out[0].tolist(), tokenizer_path)
    if args.output:
        output_path = resolve_project_path(project_root, args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
