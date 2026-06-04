from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F

from mini_llm.rope import apply_rope, build_rope_cache


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    dropout: float
    position_encoding: str


KVCache = tuple[torch.Tensor, torch.Tensor]


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    return mask.view(1, 1, seq_len, seq_len)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.config = config
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, KVCache | None]:
        batch, seq_len, embd = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(embd, dim=-1)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        if self.config.position_encoding == "rope":
            total_for_cache = start_pos + seq_len
            cos, sin = build_rope_cache(total_for_cache, self.head_dim, x.device)
            q = apply_rope(q, cos, sin, start_pos=start_pos)
            k = apply_rope(k, cos, sin, start_pos=start_pos)

        if past_kv is not None:
            past_k, past_v = past_kv
            k_all = torch.cat((past_k, k), dim=-2)
            v_all = torch.cat((past_v, v), dim=-2)
        else:
            k_all = k
            v_all = v

        scores = (q @ k_all.transpose(-2, -1)) / sqrt(self.head_dim)
        total_len = k_all.size(-2)
        q_positions = torch.arange(start_pos, start_pos + seq_len, device=x.device).view(seq_len, 1)
        k_positions = torch.arange(total_len, device=x.device).view(1, total_len)
        mask = (k_positions > q_positions).view(1, 1, seq_len, total_len)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        y = attn @ v_all
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, embd)
        y = self.resid_dropout(self.proj(y))
        new_cache = (k_all, v_all) if use_cache else None
        return y, new_cache


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, KVCache | None]:
        attn_out, new_kv = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache, start_pos=start_pos)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv


class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.position_encoding not in {"none", "learned_abs", "rope"}:
            raise ValueError("position_encoding must be none, learned_abs, or rope")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = (
            nn.Embedding(config.block_size, config.n_embd)
            if config.position_encoding == "learned_abs"
            else None
        )
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_kv: list[KVCache] | None = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ):
        batch, seq_len = idx.shape
        if start_pos + seq_len > self.config.block_size:
            raise ValueError("sequence length exceeds block_size")
        x = self.token_embedding(idx)
        if self.position_embedding is not None:
            positions = torch.arange(start_pos, start_pos + seq_len, device=idx.device)
            x = x + self.position_embedding(positions).view(1, seq_len, -1)
        x = self.drop(x)

        if past_kv is None:
            past_kv = [None] * len(self.blocks)  # type: ignore[list-item]
        new_kvs: list[KVCache] = []
        for block, layer_past in zip(self.blocks, past_kv, strict=True):
            x, new_kv = block(x, past_kv=layer_past, use_cache=use_cache, start_pos=start_pos)
            if use_cache and new_kv is not None:
                new_kvs.append(new_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        if use_cache:
            return logits, loss, new_kvs
        return logits, loss


def config_from_model_config(model_cfg) -> GPTConfig:
    return GPTConfig(
        vocab_size=model_cfg.vocab_size,
        n_layer=model_cfg.n_layer,
        n_head=model_cfg.n_head,
        n_embd=model_cfg.n_embd,
        block_size=model_cfg.block_size,
        dropout=model_cfg.dropout,
        position_encoding=model_cfg.position_encoding,
    )
