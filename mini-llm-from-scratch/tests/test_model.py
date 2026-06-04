import torch

from mini_llm.model import GPTConfig, GPTLanguageModel, build_causal_mask


def test_causal_mask_blocks_future_positions():
    mask = build_causal_mask(seq_len=4, device=torch.device("cpu"))
    assert mask.shape == (1, 1, 4, 4)
    assert mask[0, 0, 0, 1].item() is True
    assert mask[0, 0, 3, 0].item() is False


def test_model_forward_with_loss_for_all_position_modes():
    for pos in ["none", "learned_abs", "rope"]:
        cfg = GPTConfig(
            vocab_size=128,
            n_layer=2,
            n_head=2,
            n_embd=32,
            block_size=16,
            dropout=0.0,
            position_encoding=pos,
        )
        model = GPTLanguageModel(cfg)
        idx = torch.randint(0, 128, (2, 16))
        logits, loss = model(idx, targets=idx)
        assert logits.shape == (2, 16, 128)
        assert loss is not None
        assert torch.isfinite(loss)


def test_model_rejects_too_long_sequence():
    cfg = GPTConfig(
        vocab_size=128,
        n_layer=1,
        n_head=2,
        n_embd=32,
        block_size=8,
        dropout=0.0,
        position_encoding="learned_abs",
    )
    model = GPTLanguageModel(cfg)
    idx = torch.randint(0, 128, (1, 9))
    try:
        model(idx)
    except ValueError as exc:
        assert "block_size" in str(exc)
    else:
        raise AssertionError("expected ValueError")
