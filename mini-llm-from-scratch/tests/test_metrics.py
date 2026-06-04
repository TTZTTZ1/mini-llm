from mini_llm.metrics import perplexity, tokens_per_second


def test_perplexity_from_loss():
    assert round(perplexity(0.0), 4) == 1.0
    assert round(perplexity(1.0), 4) == 2.7183


def test_tokens_per_second():
    assert tokens_per_second(num_tokens=1000, elapsed_seconds=2.0) == 500.0
