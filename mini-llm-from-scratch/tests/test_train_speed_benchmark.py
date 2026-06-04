from pathlib import Path

import torch

from mini_llm.benchmark_train_speed import (
    BENCHMARK_PRESETS,
    build_preset_config,
    benchmark_batch_size,
    tokens_per_step,
    write_csv,
)
from mini_llm.model import GPTConfig


def test_build_large_preset_config_overrides_vocab_and_block_size():
    cfg = build_preset_config("large_250m", vocab_size=32000, block_size=1024)

    assert cfg.vocab_size == 32000
    assert cfg.block_size == 1024
    assert cfg.n_layer == 16
    assert cfg.n_head == 16
    assert cfg.n_embd == 1024
    assert cfg.position_encoding == "rope"


def test_benchmark_presets_include_larger_model_candidates():
    expected = {
        "large_250m": (16, 16, 1024),
        "large_320m": (20, 16, 1024),
        "large_370m": (24, 16, 1024),
        "large_450m": (20, 20, 1280),
        "large_560m": (24, 20, 1280),
    }

    for preset, (n_layer, n_head, n_embd) in expected.items():
        assert preset in BENCHMARK_PRESETS
        cfg = build_preset_config(preset, vocab_size=32000, block_size=1024)
        assert cfg.n_layer == n_layer
        assert cfg.n_head == n_head
        assert cfg.n_embd == n_embd
        assert cfg.position_encoding == "rope"


def test_tokens_per_step_uses_batch_times_context():
    assert tokens_per_step(batch_size=12, block_size=1024) == 12288


def test_benchmark_batch_size_runs_one_cpu_training_step():
    cfg = GPTConfig(
        vocab_size=128,
        n_layer=1,
        n_head=2,
        n_embd=32,
        block_size=16,
        dropout=0.0,
        position_encoding="learned_abs",
    )

    row = benchmark_batch_size(
        cfg,
        batch_size=2,
        warmup_steps=0,
        timed_steps=1,
        device=torch.device("cpu"),
        dtype="float32",
        learning_rate=1e-3,
        weight_decay=0.0,
    )

    assert row["status"] == "ok"
    assert row["batch_size"] == 2
    assert row["block_size"] == 16
    assert row["tokens_per_step"] == 32
    assert row["tokens_per_second"] > 0
    assert row["seconds"] > 0


def test_write_csv_writes_benchmark_rows(tmp_path):
    output = tmp_path / "benchmark.csv"
    rows = [
        {
            "preset": "large_250m",
            "status": "ok",
            "batch_size": 4,
            "block_size": 1024,
            "tokens_per_step": 4096,
            "timed_steps": 20,
            "seconds": 1.0,
            "tokens_per_second": 81920.0,
            "peak_memory_allocated_mb": 1000.0,
            "peak_memory_reserved_mb": 1200.0,
            "error": "",
        }
    ]

    write_csv(output, rows)

    text = output.read_text(encoding="utf-8")
    assert "tokens_per_second" in text
    assert "large_250m" in text
