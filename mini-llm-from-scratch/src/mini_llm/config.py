from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import yaml


POSITION_ENCODINGS = {"none", "learned_abs", "rope"}
SAMPLING_STRATEGIES = {"greedy", "temperature", "top_k", "top_p"}
DEVICES = {"cpu", "cuda", "mps"}
DTYPES = {"float32", "float16", "bfloat16"}


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    run_name: str


@dataclass(frozen=True)
class DataConfig:
    raw_dir: str
    cleaned_text: str
    train_bin: str
    val_bin: str
    tokenizer_path: str
    vocab_size: int
    val_fraction: float


@dataclass(frozen=True)
class ModelConfig:
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    vocab_size: int
    dropout: float
    position_encoding: str


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    device: str
    dtype: str
    batch_size: int
    max_steps: int
    eval_interval: int
    eval_iters: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    out_dir: str


@dataclass(frozen=True)
class GenerateConfig:
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float


@dataclass(frozen=True)
class ExperimentConfig:
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    generate: GenerateConfig


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required config key: {key}")
    return mapping[key]


def load_config(path: Path) -> ExperimentConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    project = ProjectConfig(**_require(raw, "project"))
    data = DataConfig(**_require(raw, "data"))
    model = ModelConfig(**_require(raw, "model"))
    train = TrainConfig(**_require(raw, "train"))
    generate = GenerateConfig(**_require(raw, "generate"))

    if model.position_encoding not in POSITION_ENCODINGS:
        allowed = ", ".join(sorted(POSITION_ENCODINGS))
        raise ValueError(f"position_encoding must be one of {allowed}")
    if train.device not in DEVICES:
        allowed = ", ".join(sorted(DEVICES))
        raise ValueError(f"device must be one of {allowed}")
    if train.dtype not in DTYPES:
        allowed = ", ".join(sorted(DTYPES))
        raise ValueError(f"dtype must be one of {allowed}")
    if data.vocab_size != model.vocab_size:
        raise ValueError("data.vocab_size must match model.vocab_size")
    if not 0 < data.val_fraction < 1:
        raise ValueError("data.val_fraction must be between 0 and 1")
    return ExperimentConfig(project=project, data=data, model=model, train=train, generate=generate)


def resolve_run_dir(cfg: ExperimentConfig, root: Path = Path(".")) -> Path:
    return root / cfg.train.out_dir


def project_root_from_config(config_path: Path) -> Path:
    config_path = config_path.resolve()
    if config_path.parent.name != "configs":
        return config_path.parent
    return config_path.parent.parent


def resolve_project_path(root: Path, value: str | PathLike[str] | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path
