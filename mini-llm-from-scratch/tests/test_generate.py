import torch

from mini_llm.generate import filter_top_k, filter_top_p, sample_next_token


def test_filter_top_k_masks_all_but_k():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = filter_top_k(logits, k=2)
    assert torch.isneginf(out[0, 0])
    assert torch.isneginf(out[0, 1])
    assert out[0, 2].item() == 3.0
    assert out[0, 3].item() == 4.0


def test_filter_top_p_keeps_highest_until_threshold():
    logits = torch.tensor([[10.0, 9.0, 1.0, 0.0]])
    out = filter_top_p(logits, p=0.8)
    assert not torch.isneginf(out[0, 0])
    assert torch.isneginf(out[0, 3])


def test_greedy_sample_returns_argmax():
    logits = torch.tensor([[0.1, 2.0, 1.0]])
    token = sample_next_token(logits, strategy="greedy", temperature=1.0, top_k=0, top_p=1.0)
    assert token.item() == 1
