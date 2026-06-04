import torch

from mini_llm.rope import apply_rope, build_rope_cache


def test_rope_cache_shapes():
    cos, sin = build_rope_cache(seq_len=8, head_dim=16, device=torch.device("cpu"))
    assert cos.shape == (8, 8)
    assert sin.shape == (8, 8)


def test_apply_rope_preserves_shape():
    x = torch.randn(2, 4, 8, 16)
    cos, sin = build_rope_cache(seq_len=8, head_dim=16, device=torch.device("cpu"))
    out = apply_rope(x, cos, sin, start_pos=0)
    assert out.shape == x.shape
