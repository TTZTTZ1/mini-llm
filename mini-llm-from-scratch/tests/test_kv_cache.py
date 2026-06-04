import torch

from mini_llm.benchmark_kv_cache import greedy_no_cache, greedy_with_cache
from mini_llm.model import GPTConfig, GPTLanguageModel


def test_kv_cache_greedy_matches_no_cache():
    torch.manual_seed(0)
    cfg = GPTConfig(
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=32,
        dropout=0.3,
        position_encoding="rope",
    )
    model = GPTLanguageModel(cfg)
    model.train()
    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    no_cache = greedy_no_cache(model, prompt, max_new_tokens=8)
    with_cache = greedy_with_cache(model, prompt, max_new_tokens=8)
    assert torch.equal(no_cache, with_cache)
    assert model.training
