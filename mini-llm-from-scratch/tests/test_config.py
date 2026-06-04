from pathlib import Path

from dataclasses import replace

import pytest

from mini_llm.config import load_config, project_root_from_config, resolve_project_path, resolve_run_dir
from mini_llm.model import GPTLanguageModel, config_from_model_config


def test_load_smoke_config_has_linux_relative_paths():
    cfg = load_config(Path("configs/smoke.yaml"))
    assert cfg.project.name == "mini-llm-from-scratch"
    assert cfg.data.train_bin == "data/processed/smoke_train.bin"
    assert not cfg.data.train_bin.startswith("/Users/")
    assert cfg.model.position_encoding == "learned_abs"
    assert cfg.train.device == "cpu"


def test_resolve_run_dir_uses_project_relative_path(tmp_path):
    cfg = load_config(Path("configs/smoke.yaml"))
    run_dir = resolve_run_dir(cfg, root=tmp_path)
    assert run_dir == tmp_path / "results" / "runs" / "smoke"


def test_project_root_from_config_and_path_resolution():
    root = project_root_from_config(Path("configs/smoke.yaml"))
    assert root.name == "mini-llm-from-scratch"
    assert resolve_project_path(root, "data/processed/train.bin") == root / "data" / "processed" / "train.bin"
    absolute = root / "data" / "processed" / "train.bin"
    assert resolve_project_path(root, absolute) == absolute


def test_load_config_rejects_bad_device_and_dtype(tmp_path):
    cfg = load_config(Path("configs/smoke.yaml"))
    bad_device = replace(cfg.train, device="gpu")
    raw = {
        "project": cfg.project.__dict__,
        "data": cfg.data.__dict__,
        "model": cfg.model.__dict__,
        "train": bad_device.__dict__,
        "generate": cfg.generate.__dict__,
    }
    path = tmp_path / "bad_device.yaml"
    import yaml

    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="device"):
        load_config(path)

    raw["train"] = replace(cfg.train, dtype="bf16").__dict__
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="dtype"):
        load_config(path)


def test_final_212m_config_matches_training_budget():
    cfg = load_config(Path("configs/final_212m_rope_ctx1024.yaml"))

    assert cfg.model.n_layer == 16
    assert cfg.model.n_head == 14
    assert cfg.model.n_embd == 896
    assert cfg.model.block_size == 1024
    assert cfg.model.position_encoding == "rope"
    assert cfg.train.batch_size == 24
    assert cfg.train.max_steps == 130209
    assert cfg.train.batch_size * cfg.model.block_size * cfg.train.max_steps >= 3_200_000_000
    assert cfg.data.train_bin == "data/processed/fineweb_edu_train.bin"
    assert cfg.data.tokenizer_path == "data/tokenizer/fineweb_edu_bpe_32000.json"
    params = sum(p.numel() for p in GPTLanguageModel(config_from_model_config(cfg.model)).parameters())
    assert 210_000_000 <= params <= 214_000_000
