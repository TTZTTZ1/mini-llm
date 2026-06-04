# Mini LLM From Scratch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Linux-first, GitHub-synchronized mini Decoder-only Transformer project that can train tiny/small Chinese GPT-like language models, run context/position/sampling/KV-cache experiments, and produce reproducible metrics, figures, and experiment notes.

**Architecture:** The current repository at `/Users/ttz/Documents/大模型推理` remains the Git root. The new implementation lives under `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/` as a focused Python project with reusable modules in `src/mini_llm/`, Linux bash entrypoints in `scripts/`, configs in `configs/`, tests in `tests/`, and generated experiment outputs under ignored `results/` paths. Mac is used for coding and CPU smoke tests; CloudStudio Linux/A800 is the source of truth for preprocessing, training, generation, and benchmarks, with code synchronized through Git + GitHub.

**Tech Stack:** Python 3.10+, PyTorch, HuggingFace `tokenizers`, PyYAML, NumPy, pandas, matplotlib, pytest, tqdm, Linux bash, Git, GitHub private repository, CloudStudio Linux with CUDA/A800.

---

## Clarification Check

No blocking clarification is required to write this plan. The plan makes these concrete decisions:

- Use the current Git repository `/Users/ttz/Documents/大模型推理` as the synchronization repository.
- Create the mini LLM project as `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/`.
- Use GitHub as the shared remote so CloudStudio can clone/pull code from the same repository.
- Prefer a private GitHub repository named `mini-llm-from-scratch`.
- Use THUCNews as the first Chinese dataset source, with CLUECorpusSmall sampling reserved as an extension.
- Keep resume/package wording outside this project plan.

If the GitHub repository should instead be a new standalone repo containing only `mini-llm-from-scratch/`, change only Task 0 before implementation.

## File Structure And Responsibilities

Create this tree:

```text
/Users/ttz/Documents/大模型推理/
├── .gitignore
├── docs/
│   └── superpowers/
│       └── plans/
│           └── 2026-06-04-mini-llm-from-scratch-implementation.md
└── mini-llm-from-scratch/
    ├── README.md
    ├── requirements.txt
    ├── configs/
    │   ├── smoke.yaml
    │   ├── tiny.yaml
    │   ├── small.yaml
    │   ├── ablation_context_128.yaml
    │   ├── ablation_context_256.yaml
    │   ├── ablation_context_512.yaml
    │   ├── ablation_pos_none.yaml
    │   ├── ablation_pos_abs.yaml
    │   └── ablation_pos_rope.yaml
    ├── data/
    │   ├── raw/.gitkeep
    │   ├── processed/.gitkeep
    │   └── tokenizer/.gitkeep
    ├── experiments/
    │   ├── ablation_context.md
    │   ├── ablation_kv_cache.md
    │   ├── ablation_position.md
    │   ├── ablation_sampling.md
    │   └── ablation_scale.md
    ├── report/
    │   └── experiment_report.md
    ├── scripts/
    │   ├── prepare_thucnews.sh
    │   ├── run_ablation_context.sh
    │   ├── run_ablation_position.sh
    │   ├── run_kv_cache_benchmark.sh
    │   ├── run_sampling.sh
    │   ├── run_smoke_test.sh
    │   ├── run_train_small.sh
    │   └── run_train_tiny.sh
    ├── src/
    │   └── mini_llm/
    │       ├── __init__.py
    │       ├── benchmark_kv_cache.py
    │       ├── checkpoint.py
    │       ├── config.py
    │       ├── data.py
    │       ├── generate.py
    │       ├── metrics.py
    │       ├── model.py
    │       ├── plot.py
    │       ├── rope.py
    │       ├── tokenizer_train.py
    │       ├── train.py
    │       └── utils.py
    └── tests/
        ├── fixtures/
        │   ├── tiny_corpus.txt
        │   └── tiny_tokens.bin
        ├── test_config.py
        ├── test_data.py
        ├── test_generate.py
        ├── test_kv_cache.py
        ├── test_metrics.py
        ├── test_model.py
        ├── test_rope.py
        └── test_tokenizer_train.py
```

Responsibilities:

- `.gitignore`: protect raw data, checkpoints, generated logs, and large binary artifacts from accidental commits.
- `mini-llm-from-scratch/README.md`: Mac smoke-test commands, Linux server setup, GitHub sync workflow, and formal experiment commands.
- `requirements.txt`: dependencies installable on Linux; Mac users can install the same list except CUDA-specific torch wheels are selected by PyTorch install command.
- `configs/*.yaml`: complete experiment settings; no hardcoded `/Users/...` or Linux absolute paths.
- `scripts/*.sh`: Linux-first bash entrypoints for CloudStudio; also usable from Mac for CPU smoke checks where noted.
- `src/mini_llm/config.py`: load YAML into typed dataclasses and resolve project-relative paths.
- `src/mini_llm/data.py`: clean Chinese text, split train/val, create binary token files, and serve contiguous next-token batches.
- `src/mini_llm/tokenizer_train.py`: train BPE tokenizer from cleaned text and encode corpus.
- `src/mini_llm/rope.py`: RoPE cache construction and rotation helpers.
- `src/mini_llm/model.py`: Decoder-only Transformer, learned/none/RoPE position modes, causal attention, optional KV cache.
- `src/mini_llm/train.py`: training loop, evaluation, metrics CSV, checkpoints, seed control, CPU/MPS/CUDA device selection.
- `src/mini_llm/generate.py`: greedy, temperature, top-k, and top-p generation from checkpoints.
- `src/mini_llm/benchmark_kv_cache.py`: no-cache vs with-cache correctness and speed benchmark.
- `src/mini_llm/metrics.py`: perplexity, running averages, CSV row formatting, speed/memory summaries.
- `src/mini_llm/plot.py`: read metrics/benchmark CSV files and save PNG figures without GUI calls.
- `tests/*`: CPU-friendly tests that validate shape, masking, tokenizer/data behavior, generation filters, and KV-cache equivalence.

---

## Task 0: GitHub Synchronization Foundation

**Files:**
- Create: `/Users/ttz/Documents/大模型推理/.gitignore`
- Modify: repository git remote configuration

**Tests:**
- `git status --short`
- `git remote -v`
- `gh repo view mini-llm-from-scratch --json name,visibility`

- [ ] **Step 1: Confirm current Git state**

Run:

```bash
cd /Users/ttz/Documents/大模型推理
git status --short
git branch --show-current
git remote -v
```

Expected:

```text
git status --short prints no tracked/untracked changes before this task starts
git branch --show-current prints codex/llm-inference-benchmark
git remote -v prints no remote before GitHub setup
```

- [ ] **Step 2: Add root `.gitignore`**

File: `/Users/ttz/Documents/大模型推理/.gitignore`

```gitignore
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/

.DS_Store
.env
.venv/
venv/

mini-llm-from-scratch/data/raw/*
mini-llm-from-scratch/data/processed/*
mini-llm-from-scratch/data/tokenizer/*
!mini-llm-from-scratch/data/raw/.gitkeep
!mini-llm-from-scratch/data/processed/.gitkeep
!mini-llm-from-scratch/data/tokenizer/.gitkeep

mini-llm-from-scratch/results/
mini-llm-from-scratch/checkpoints/
mini-llm-from-scratch/logs/
*.pt
*.pth
*.bin
*.npy
*.npz
*.log
```

- [ ] **Step 3: Verify GitHub CLI auth on Mac**

Run:

```bash
gh auth status
```

Expected: output includes `Logged in to github.com`. If it prints `You are not logged into any GitHub hosts`, run:

```bash
gh auth login
```

Expected: interactive login succeeds and a later `gh auth status` includes `Logged in to github.com`.

- [ ] **Step 4: Create private GitHub remote for the current repository**

Run:

```bash
cd /Users/ttz/Documents/大模型推理
gh repo create mini-llm-from-scratch --private --source=. --remote=origin --push
```

Expected:

```text
Created repository
Added remote origin
Pushed commits to origin
```

- [ ] **Step 5: Verify remote**

Run:

```bash
git remote -v
gh repo view mini-llm-from-scratch --json name,visibility --jq '.name + " " + .visibility'
```

Expected:

```text
git remote -v prints origin fetch and push URLs ending with /mini-llm-from-scratch.git
mini-llm-from-scratch PRIVATE
```

- [ ] **Step 6: Server clone workflow check**

Run on CloudStudio Linux:

```bash
gh auth status
gh repo clone mini-llm-from-scratch
cd mini-llm-from-scratch
git branch --show-current
```

Expected:

```text
gh auth status includes Logged in to github.com
git clone completes
branch command prints codex/llm-inference-benchmark or the default branch selected by GitHub
```

- [ ] **Step 7: Commit GitHub sync metadata**

Run on Mac:

```bash
cd /Users/ttz/Documents/大模型推理
git add .gitignore
git commit -m "chore: configure mini llm artifact ignores"
git push origin HEAD
```

Expected:

```text
[codex/llm-inference-benchmark <sha>] chore: configure mini llm artifact ignores
git push completes without rejected updates
```

---

## Task 1: Create Mini LLM Project Skeleton

**Files:**
- Create all files and directories listed in File Structure under `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/`

**Tests:**
- `find mini-llm-from-scratch -maxdepth 3 -type d | sort`
- `python -m py_compile` for package marker

- [ ] **Step 1: Create directories**

Run:

```bash
cd /Users/ttz/Documents/大模型推理
mkdir -p mini-llm-from-scratch/{configs,data/{raw,processed,tokenizer},experiments,report,scripts,src/mini_llm,tests/fixtures}
```

Expected: command exits with status 0.

- [ ] **Step 2: Create `.gitkeep` files**

Files:

```text
/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/data/raw/.gitkeep
/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/data/processed/.gitkeep
/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/data/tokenizer/.gitkeep
```

Each file content is empty.

- [ ] **Step 3: Create package marker**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/src/mini_llm/__init__.py`

```python
"""Mini Decoder-only Transformer language model project."""
```

- [ ] **Step 4: Create requirements**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/requirements.txt`

```text
matplotlib
numpy
pandas
pytest
pyyaml
tokenizers
torch
tqdm
```

- [ ] **Step 5: Create initial README**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/README.md`

```markdown
# Mini LLM From Scratch

Linux-first PyTorch implementation of a small Chinese Decoder-only Transformer language model.

## Development Split

- Mac: edit code, run CPU smoke tests, inspect logs, write reports.
- CloudStudio Linux/A800: preprocess data, train models, generate samples, run KV-cache benchmarks.

## GitHub Sync

Mac:

```bash
git add mini-llm-from-scratch docs/superpowers/plans .gitignore
git commit -m "feat: implement mini llm task"
git push origin HEAD
```

CloudStudio:

```bash
git pull origin HEAD
cd mini-llm-from-scratch
```

## Local Smoke Test

```bash
cd mini-llm-from-scratch
PYTHONPATH=src pytest -q
```
```

- [ ] **Step 6: Create empty experiment/report files**

Files and content:

```text
experiments/ablation_context.md -> "# Context Length Ablation\n"
experiments/ablation_kv_cache.md -> "# KV Cache Benchmark\n"
experiments/ablation_position.md -> "# Position Encoding Ablation\n"
experiments/ablation_sampling.md -> "# Sampling Strategy Ablation\n"
experiments/ablation_scale.md -> "# Model Scale Ablation\n"
report/experiment_report.md -> "# Mini LLM Experiment Report\n"
```

- [ ] **Step 7: Verify skeleton**

Run:

```bash
cd /Users/ttz/Documents/大模型推理
find mini-llm-from-scratch -maxdepth 3 -type d | sort
PYTHONPATH=mini-llm-from-scratch/src python -m py_compile mini-llm-from-scratch/src/mini_llm/__init__.py
```

Expected:

```text
directory listing includes configs, data/raw, data/processed, data/tokenizer, experiments, report, scripts, src/mini_llm, tests/fixtures
py_compile exits with status 0
```

- [ ] **Step 8: Commit**

Run:

```bash
git add mini-llm-from-scratch .gitignore
git commit -m "chore: create mini llm project skeleton"
git push origin HEAD
```

Expected: commit and push succeed.

---

## Task 2: Add Config System And Experiment YAMLs

**Files:**
- Create: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/src/mini_llm/config.py`
- Create: all `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/configs/*.yaml`
- Create: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/tests/test_config.py`

**Tests:**
- `PYTHONPATH=src pytest tests/test_config.py -q`

- [ ] **Step 1: Write failing config tests**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/tests/test_config.py`

```python
from pathlib import Path

from mini_llm.config import load_config, resolve_run_dir


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
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd /Users/ttz/Documents/大模型推理/mini-llm-from-scratch
PYTHONPATH=src pytest tests/test_config.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'mini_llm.config'
```

- [ ] **Step 3: Implement config dataclasses**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/src/mini_llm/config.py`

Implementation requirements:

```text
Define dataclasses ProjectConfig, DataConfig, ModelConfig, TrainConfig, GenerateConfig, ExperimentConfig.
Implement load_config(path: Path) -> ExperimentConfig using yaml.safe_load.
Implement resolve_run_dir(cfg: ExperimentConfig, root: Path = Path(".")) -> Path.
Keep all path values as strings from YAML until resolved by caller.
Reject position_encoding values outside none, learned_abs, rope with ValueError.
Reject sampling values outside greedy, temperature, top_k, top_p with ValueError.
```

- [ ] **Step 4: Add YAML configs**

Files:

```text
configs/smoke.yaml
configs/tiny.yaml
configs/small.yaml
configs/ablation_context_128.yaml
configs/ablation_context_256.yaml
configs/ablation_context_512.yaml
configs/ablation_pos_none.yaml
configs/ablation_pos_abs.yaml
configs/ablation_pos_rope.yaml
```

`configs/smoke.yaml` content:

```yaml
project:
  name: mini-llm-from-scratch
  run_name: smoke
data:
  raw_dir: data/raw
  cleaned_text: data/processed/smoke_clean.txt
  train_bin: data/processed/smoke_train.bin
  val_bin: data/processed/smoke_val.bin
  tokenizer_path: data/tokenizer/smoke_tokenizer.json
  vocab_size: 256
  val_fraction: 0.1
model:
  n_layer: 2
  n_head: 2
  n_embd: 64
  block_size: 32
  vocab_size: 256
  dropout: 0.0
  position_encoding: learned_abs
train:
  seed: 42
  device: cpu
  dtype: float32
  batch_size: 4
  max_steps: 20
  eval_interval: 10
  eval_iters: 2
  learning_rate: 0.001
  weight_decay: 0.0
  grad_clip: 1.0
  out_dir: results/runs/smoke
generate:
  max_new_tokens: 32
  temperature: 1.0
  top_k: 20
  top_p: 0.9
```

`configs/tiny.yaml` differs from smoke:

```yaml
project:
  name: mini-llm-from-scratch
  run_name: tiny_main
data:
  raw_dir: data/raw
  cleaned_text: data/processed/thucnews_clean.txt
  train_bin: data/processed/train.bin
  val_bin: data/processed/val.bin
  tokenizer_path: data/tokenizer/bpe_16000.json
  vocab_size: 16000
  val_fraction: 0.05
model:
  n_layer: 4
  n_head: 4
  n_embd: 256
  block_size: 256
  vocab_size: 16000
  dropout: 0.1
  position_encoding: learned_abs
train:
  seed: 42
  device: cuda
  dtype: bfloat16
  batch_size: 64
  max_steps: 5000
  eval_interval: 250
  eval_iters: 50
  learning_rate: 0.0003
  weight_decay: 0.1
  grad_clip: 1.0
  out_dir: results/runs/tiny_main
generate:
  max_new_tokens: 128
  temperature: 0.9
  top_k: 50
  top_p: 0.9
```

`configs/small.yaml` differs from tiny:

```yaml
project:
  name: mini-llm-from-scratch
  run_name: small_main
model:
  n_layer: 8
  n_head: 8
  n_embd: 512
  block_size: 512
  vocab_size: 16000
  dropout: 0.1
  position_encoding: learned_abs
train:
  seed: 42
  device: cuda
  dtype: bfloat16
  batch_size: 32
  max_steps: 10000
  eval_interval: 500
  eval_iters: 50
  learning_rate: 0.00025
  weight_decay: 0.1
  grad_clip: 1.0
  out_dir: results/runs/small_main
```

Context ablation configs copy `tiny.yaml` and set:

```text
ablation_context_128.yaml: project.run_name=context_128, model.block_size=128, train.out_dir=results/runs/context_128
ablation_context_256.yaml: project.run_name=context_256, model.block_size=256, train.out_dir=results/runs/context_256
ablation_context_512.yaml: project.run_name=context_512, model.block_size=512, train.out_dir=results/runs/context_512
```

Position ablation configs copy `tiny.yaml` and set:

```text
ablation_pos_none.yaml: project.run_name=pos_none, model.position_encoding=none, train.out_dir=results/runs/pos_none
ablation_pos_abs.yaml: project.run_name=pos_abs, model.position_encoding=learned_abs, train.out_dir=results/runs/pos_abs
ablation_pos_rope.yaml: project.run_name=pos_rope, model.position_encoding=rope, train.out_dir=results/runs/pos_rope
```

- [ ] **Step 5: Run tests**

Run:

```bash
cd /Users/ttz/Documents/大模型推理/mini-llm-from-scratch
PYTHONPATH=src pytest tests/test_config.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 6: Commit**

Run:

```bash
cd /Users/ttz/Documents/大模型推理
git add mini-llm-from-scratch/configs mini-llm-from-scratch/src/mini_llm/config.py mini-llm-from-scratch/tests/test_config.py
git commit -m "feat: add mini llm experiment configs"
git push origin HEAD
```

Expected: commit and push succeed.

---

## Task 3: Implement Data Cleaning, Tokenizer Training, And Binary Encoding

**Files:**
- Create: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/src/mini_llm/tokenizer_train.py`
- Create: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/src/mini_llm/data.py`
- Create: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/tests/fixtures/tiny_corpus.txt`
- Create: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/tests/test_tokenizer_train.py`
- Create: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/tests/test_data.py`

**Tests:**
- `PYTHONPATH=src pytest tests/test_tokenizer_train.py tests/test_data.py -q`

- [ ] **Step 1: Add text fixture**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/tests/fixtures/tiny_corpus.txt`

```text
今天的人工智能新闻发布了新的研究结果。
小型语言模型可以帮助我们理解大模型的基本原理。

训练语言模型需要文本、分词器、损失函数和生成方法。
```

- [ ] **Step 2: Write failing tokenizer tests**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/tests/test_tokenizer_train.py`

```python
from pathlib import Path

from mini_llm.tokenizer_train import train_bpe_tokenizer, encode_file


def test_train_bpe_tokenizer_and_encode(tmp_path):
    corpus = Path("tests/fixtures/tiny_corpus.txt")
    tokenizer_path = tmp_path / "tok.json"
    train_bpe_tokenizer([corpus], tokenizer_path, vocab_size=128)
    assert tokenizer_path.exists()

    ids = encode_file(tokenizer_path, corpus)
    assert len(ids) > 10
    assert all(isinstance(x, int) for x in ids)
```

- [ ] **Step 3: Write failing data tests**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/tests/test_data.py`

```python
from pathlib import Path

import numpy as np
import torch

from mini_llm.data import clean_text, write_token_bins, TokenBlockDataset


def test_clean_text_removes_blank_lines_and_controls():
    raw = "第一行\\n\\n\\x00第二行\\r\\n  \\n第三行"
    cleaned = clean_text(raw, min_chars=2)
    assert cleaned == "第一行\\n第二行\\n第三行"


def test_write_token_bins_and_block_dataset(tmp_path):
    ids = list(range(100))
    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"
    write_token_bins(ids, train_bin, val_bin, val_fraction=0.2)
    assert np.memmap(train_bin, dtype=np.uint16, mode="r").shape[0] == 80
    assert np.memmap(val_bin, dtype=np.uint16, mode="r").shape[0] == 20

    ds = TokenBlockDataset(train_bin, block_size=8)
    x, y = ds[0]
    assert x.shape == torch.Size([8])
    assert y.shape == torch.Size([8])
    assert torch.equal(y, x + 1)
```

- [ ] **Step 4: Run tests to verify failure**

Run:

```bash
cd /Users/ttz/Documents/大模型推理/mini-llm-from-scratch
PYTHONPATH=src pytest tests/test_tokenizer_train.py tests/test_data.py -q
```

Expected:

```text
ModuleNotFoundError for mini_llm.tokenizer_train or mini_llm.data
```

- [ ] **Step 5: Implement tokenizer helpers**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/src/mini_llm/tokenizer_train.py`

Implementation requirements:

```text
train_bpe_tokenizer(files: list[Path], output_path: Path, vocab_size: int) -> None
  - Use tokenizers.Tokenizer with BPE model.
  - Use Whitespace pre-tokenizer.
  - Include special tokens ["<pad>", "<unk>", "<bos>", "<eos>"].
  - Save JSON tokenizer to output_path.

encode_file(tokenizer_path: Path, text_path: Path) -> list[int]
  - Load tokenizer JSON.
  - Read UTF-8 text.
  - Return encoded token ids.

main()
  - CLI args: --input, --tokenizer, --vocab-size, --output-ids optional.
  - Print token count when encoding succeeds.
```

- [ ] **Step 6: Implement data helpers**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/src/mini_llm/data.py`

Implementation requirements:

```text
clean_text(text: str, min_chars: int = 5) -> str
  - Normalize CRLF to LF.
  - Remove control characters except newline and tab.
  - Strip each line.
  - Drop blank lines.
  - Drop lines with fewer than min_chars characters.
  - Join remaining lines with "\n".

write_token_bins(ids: list[int], train_bin: Path, val_bin: Path, val_fraction: float) -> None
  - Use np.uint16 when max(ids) < 65536.
  - Use np.uint32 otherwise.
  - Split by index boundary int(len(ids) * (1 - val_fraction)).
  - Write raw binary arrays with tofile().

TokenBlockDataset(bin_path: Path, block_size: int)
  - Load memmap as np.uint16 by default.
  - __len__ returns max(0, num_tokens - block_size).
  - __getitem__(idx) returns torch.long x=ids[idx:idx+block_size], y=ids[idx+1:idx+block_size+1].
```

- [ ] **Step 7: Run tests**

Run:

```bash
cd /Users/ttz/Documents/大模型推理/mini-llm-from-scratch
PYTHONPATH=src pytest tests/test_tokenizer_train.py tests/test_data.py -q
```

Expected:

```text
4 passed
```

- [ ] **Step 8: Commit**

Run:

```bash
cd /Users/ttz/Documents/大模型推理
git add mini-llm-from-scratch/src/mini_llm/tokenizer_train.py mini-llm-from-scratch/src/mini_llm/data.py mini-llm-from-scratch/tests/test_tokenizer_train.py mini-llm-from-scratch/tests/test_data.py mini-llm-from-scratch/tests/fixtures/tiny_corpus.txt
git commit -m "feat: add tokenizer and token data pipeline"
git push origin HEAD
```

Expected: commit and push succeed.

---

## Task 4: Implement RoPE And Decoder-only Transformer Core

**Files:**
- Create: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/src/mini_llm/rope.py`
- Create: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/src/mini_llm/model.py`
- Create: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/tests/test_rope.py`
- Create: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/tests/test_model.py`

**Tests:**
- `PYTHONPATH=src pytest tests/test_rope.py tests/test_model.py -q`

- [ ] **Step 1: Write failing RoPE tests**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/tests/test_rope.py`

```python
import torch

from mini_llm.rope import build_rope_cache, apply_rope


def test_rope_cache_shapes():
    cos, sin = build_rope_cache(seq_len=8, head_dim=16, device=torch.device("cpu"))
    assert cos.shape == (8, 8)
    assert sin.shape == (8, 8)


def test_apply_rope_preserves_shape():
    x = torch.randn(2, 4, 8, 16)
    cos, sin = build_rope_cache(seq_len=8, head_dim=16, device=torch.device("cpu"))
    out = apply_rope(x, cos, sin, start_pos=0)
    assert out.shape == x.shape
```

- [ ] **Step 2: Write failing model tests**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/tests/test_model.py`

```python
import torch

from mini_llm.model import GPTConfig, GPTLanguageModel, build_causal_mask


def test_causal_mask_blocks_future_positions():
    mask = build_causal_mask(seq_len=4, device=torch.device("cpu"))
    assert mask.shape == (1, 1, 4, 4)
    assert mask[0, 0, 0, 1].item() is True
    assert mask[0, 0, 3, 0].item() is False


def test_model_forward_with_loss_for_all_position_modes():
    for pos in ["none", "learned_abs", "rope"]:
        cfg = GPTConfig(vocab_size=128, n_layer=2, n_head=2, n_embd=32, block_size=16, dropout=0.0, position_encoding=pos)
        model = GPTLanguageModel(cfg)
        idx = torch.randint(0, 128, (2, 16))
        logits, loss = model(idx, targets=idx)
        assert logits.shape == (2, 16, 128)
        assert loss is not None
        assert torch.isfinite(loss)


def test_model_rejects_too_long_sequence():
    cfg = GPTConfig(vocab_size=128, n_layer=1, n_head=2, n_embd=32, block_size=8, dropout=0.0, position_encoding="learned_abs")
    model = GPTLanguageModel(cfg)
    idx = torch.randint(0, 128, (1, 9))
    try:
        model(idx)
    except ValueError as exc:
        assert "block_size" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
cd /Users/ttz/Documents/大模型推理/mini-llm-from-scratch
PYTHONPATH=src pytest tests/test_rope.py tests/test_model.py -q
```

Expected:

```text
ModuleNotFoundError for mini_llm.rope or mini_llm.model
```

- [ ] **Step 4: Implement RoPE**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/src/mini_llm/rope.py`

Implementation requirements:

```text
build_rope_cache(seq_len: int, head_dim: int, device: torch.device, base: float = 10000.0) -> tuple[Tensor, Tensor]
  - head_dim must be even.
  - Return cos and sin with shape (seq_len, head_dim // 2).

apply_rope(x: Tensor, cos: Tensor, sin: Tensor, start_pos: int = 0) -> Tensor
  - x shape is (batch, heads, seq, head_dim).
  - Split last dim into even/odd pairs.
  - Apply rotary transform using cos/sin slice [start_pos:start_pos+seq].
  - Return same shape and dtype as x.
```

- [ ] **Step 5: Implement model**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/src/mini_llm/model.py`

Implementation requirements:

```text
GPTConfig dataclass fields: vocab_size, n_layer, n_head, n_embd, block_size, dropout, position_encoding.
build_causal_mask(seq_len, device) returns bool mask with True above diagonal, shape (1, 1, seq_len, seq_len).
CausalSelfAttention supports position_encoding="rope" and optional past_kv.
TransformerBlock has LayerNorm -> attention -> residual -> LayerNorm -> MLP -> residual.
GPTLanguageModel:
  - token embedding always enabled.
  - learned position embedding only when position_encoding="learned_abs".
  - no position tensor when position_encoding="none".
  - forward(idx, targets=None, past_kv=None, use_cache=False, start_pos=0) returns (logits, loss) or (logits, loss, new_kv).
  - loss is cross_entropy(logits.view(-1, vocab), targets.view(-1)).
  - raise ValueError when sequence length exceeds block_size.
```

- [ ] **Step 6: Run tests**

Run:

```bash
cd /Users/ttz/Documents/大模型推理/mini-llm-from-scratch
PYTHONPATH=src pytest tests/test_rope.py tests/test_model.py -q
```

Expected:

```text
5 passed
```

- [ ] **Step 7: Commit**

Run:

```bash
cd /Users/ttz/Documents/大模型推理
git add mini-llm-from-scratch/src/mini_llm/rope.py mini-llm-from-scratch/src/mini_llm/model.py mini-llm-from-scratch/tests/test_rope.py mini-llm-from-scratch/tests/test_model.py
git commit -m "feat: implement decoder only transformer core"
git push origin HEAD
```

Expected: commit and push succeed.

---

## Task 5: Implement Metrics, Checkpointing, And Training Loop

**Files:**
- Create: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/src/mini_llm/metrics.py`
- Create: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/src/mini_llm/checkpoint.py`
- Create: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/src/mini_llm/train.py`
- Create: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/tests/test_metrics.py`

**Tests:**
- `PYTHONPATH=src pytest tests/test_metrics.py -q`
- CPU smoke training command

- [ ] **Step 1: Write failing metrics tests**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/tests/test_metrics.py`

```python
from mini_llm.metrics import perplexity, tokens_per_second


def test_perplexity_from_loss():
    assert round(perplexity(0.0), 4) == 1.0
    assert round(perplexity(1.0), 4) == 2.7183


def test_tokens_per_second():
    assert tokens_per_second(num_tokens=1000, elapsed_seconds=2.0) == 500.0
```

- [ ] **Step 2: Run metrics tests to verify failure**

Run:

```bash
cd /Users/ttz/Documents/大模型推理/mini-llm-from-scratch
PYTHONPATH=src pytest tests/test_metrics.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'mini_llm.metrics'
```

- [ ] **Step 3: Implement metrics**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/src/mini_llm/metrics.py`

Implementation requirements:

```text
perplexity(loss: float) -> float returns math.exp(loss), capped at inf if overflow.
tokens_per_second(num_tokens: int, elapsed_seconds: float) -> float returns 0.0 for non-positive elapsed.
metrics_header() -> list[str] returns ["step", "train_loss", "val_loss", "val_ppl", "tokens_per_second", "peak_memory_mb"].
```

- [ ] **Step 4: Implement checkpoint helpers**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/src/mini_llm/checkpoint.py`

Implementation requirements:

```text
save_checkpoint(path, model, optimizer, cfg, step, val_loss) writes torch.save dict.
load_checkpoint(path, map_location) returns dict.
Checkpoint dict keys: model_state_dict, optimizer_state_dict, config, step, val_loss.
Ensure parent directory exists before saving.
```

- [ ] **Step 5: Implement training CLI**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/src/mini_llm/train.py`

Implementation requirements:

```text
CLI: python -m mini_llm.train --config configs/smoke.yaml --max_steps 20 --device cpu
Load config and allow --max_steps, --device, --out_dir overrides.
Set seed for random, numpy, torch.
Select device cpu/mps/cuda; cuda path supports bf16 autocast.
Build TokenBlockDataset for train_bin and val_bin.
Build DataLoader or random batch sampler.
Evaluate val loss every eval_interval.
Append metrics.csv with header from metrics_header().
Save checkpoint.pt after each evaluation if val_loss improves.
Print progress lines containing step, train_loss, val_loss.
```

- [ ] **Step 6: Create smoke token files for CPU training**

Run:

```bash
cd /Users/ttz/Documents/大模型推理/mini-llm-from-scratch
python - <<'PY'
from pathlib import Path
from mini_llm.data import write_token_bins
ids = [i % 128 for i in range(4096)]
Path("data/processed").mkdir(parents=True, exist_ok=True)
write_token_bins(ids, Path("data/processed/smoke_train.bin"), Path("data/processed/smoke_val.bin"), val_fraction=0.1)
PY
```

Expected: `data/processed/smoke_train.bin` and `data/processed/smoke_val.bin` exist.

- [ ] **Step 7: Run tests and smoke training**

Run:

```bash
cd /Users/ttz/Documents/大模型推理/mini-llm-from-scratch
PYTHONPATH=src pytest tests/test_metrics.py -q
PYTHONPATH=src python -m mini_llm.train --config configs/smoke.yaml --max_steps 20 --device cpu
```

Expected:

```text
2 passed
training output includes step=10 and step=20
results/runs/smoke/metrics.csv exists
results/runs/smoke/checkpoint.pt exists
```

- [ ] **Step 8: Commit**

Run:

```bash
cd /Users/ttz/Documents/大模型推理
git add mini-llm-from-scratch/src/mini_llm/metrics.py mini-llm-from-scratch/src/mini_llm/checkpoint.py mini-llm-from-scratch/src/mini_llm/train.py mini-llm-from-scratch/tests/test_metrics.py
git commit -m "feat: add training loop and checkpointing"
git push origin HEAD
```

Expected: commit and push succeed.

---

## Task 6: Implement Generation And Sampling Strategies

**Files:**
- Create: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/src/mini_llm/generate.py`
- Create: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/tests/test_generate.py`

**Tests:**
- `PYTHONPATH=src pytest tests/test_generate.py -q`

- [ ] **Step 1: Write failing generation tests**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/tests/test_generate.py`

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd /Users/ttz/Documents/大模型推理/mini-llm-from-scratch
PYTHONPATH=src pytest tests/test_generate.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'mini_llm.generate'
```

- [ ] **Step 3: Implement generation helpers and CLI**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/src/mini_llm/generate.py`

Implementation requirements:

```text
filter_top_k(logits, k) masks logits outside top k with -inf; k <= 0 leaves logits unchanged.
filter_top_p(logits, p) sorts probabilities and masks tokens after cumulative probability exceeds p; p >= 1 leaves logits unchanged.
sample_next_token(logits, strategy, temperature, top_k, top_p) supports greedy, temperature, top_k, top_p.
generate_tokens(model, input_ids, max_new_tokens, strategy, temperature, top_k, top_p, use_cache=False) returns token tensor.
CLI args: --config, --checkpoint, --prompt, --strategy, --device, --max_new_tokens, --output.
CLI writes generated text or token ids to output file when --output is supplied.
```

- [ ] **Step 4: Run tests**

Run:

```bash
cd /Users/ttz/Documents/大模型推理/mini-llm-from-scratch
PYTHONPATH=src pytest tests/test_generate.py -q
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Commit**

Run:

```bash
cd /Users/ttz/Documents/大模型推理
git add mini-llm-from-scratch/src/mini_llm/generate.py mini-llm-from-scratch/tests/test_generate.py
git commit -m "feat: add generation sampling strategies"
git push origin HEAD
```

Expected: commit and push succeed.

---

## Task 7: Implement KV Cache Correctness And Benchmark

**Files:**
- Create: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/src/mini_llm/benchmark_kv_cache.py`
- Create: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/tests/test_kv_cache.py`

**Tests:**
- `PYTHONPATH=src pytest tests/test_kv_cache.py -q`

- [ ] **Step 1: Write failing KV-cache tests**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/tests/test_kv_cache.py`

```python
import torch

from mini_llm.model import GPTConfig, GPTLanguageModel
from mini_llm.benchmark_kv_cache import greedy_no_cache, greedy_with_cache


def test_kv_cache_greedy_matches_no_cache():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=64, n_layer=2, n_head=2, n_embd=32, block_size=32, dropout=0.0, position_encoding="rope")
    model = GPTLanguageModel(cfg)
    model.eval()
    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    no_cache = greedy_no_cache(model, prompt, max_new_tokens=8)
    with_cache = greedy_with_cache(model, prompt, max_new_tokens=8)
    assert torch.equal(no_cache, with_cache)
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cd /Users/ttz/Documents/大模型推理/mini-llm-from-scratch
PYTHONPATH=src pytest tests/test_kv_cache.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'mini_llm.benchmark_kv_cache'
```

- [ ] **Step 3: Implement KV-cache benchmark helpers**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/src/mini_llm/benchmark_kv_cache.py`

Implementation requirements:

```text
greedy_no_cache(model, prompt, max_new_tokens) runs full-context forward each step.
greedy_with_cache(model, prompt, max_new_tokens) runs prompt prefill once, then one-token decode with past_kv.
benchmark(model, prompt, lengths, device) returns rows with length, mode, seconds, tokens_per_second, peak_memory_mb.
CLI args: --config, --checkpoint, --prompt, --lengths 64 128 256 512, --device, --output.
CLI first runs correctness comparison for greedy no-cache vs cache.
CSV output columns: length, mode, seconds, tokens_per_second, peak_memory_mb.
```

- [ ] **Step 4: Run tests**

Run:

```bash
cd /Users/ttz/Documents/大模型推理/mini-llm-from-scratch
PYTHONPATH=src pytest tests/test_kv_cache.py -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit**

Run:

```bash
cd /Users/ttz/Documents/大模型推理
git add mini-llm-from-scratch/src/mini_llm/benchmark_kv_cache.py mini-llm-from-scratch/tests/test_kv_cache.py
git commit -m "feat: add kv cache correctness benchmark"
git push origin HEAD
```

Expected: commit and push succeed.

---

## Task 8: Add Plotting, Experiment Summaries, And Linux Scripts

**Files:**
- Create: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/src/mini_llm/plot.py`
- Create or modify: all `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/scripts/*.sh`
- Modify: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/README.md`

**Tests:**
- `bash -n scripts/*.sh`
- `PYTHONPATH=src python -m mini_llm.plot --help`

- [ ] **Step 1: Implement plotting CLI**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/src/mini_llm/plot.py`

Implementation requirements:

```text
CLI args: --run_dir, --output_dir.
Read metrics.csv from run_dir.
Save loss_curve.png containing train_loss and val_loss.
Save ppl_curve.png containing val_ppl.
Use matplotlib Agg backend so Linux headless server works.
Print saved file paths.
```

- [ ] **Step 2: Add Linux scripts**

Files and commands:

`scripts/run_smoke_test.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src pytest -q
PYTHONPATH=src python -m mini_llm.train --config configs/smoke.yaml --max_steps 20 --device cpu
```

`scripts/run_train_tiny.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m mini_llm.train --config configs/tiny.yaml --device cuda
```

`scripts/run_train_small.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m mini_llm.train --config configs/small.yaml --device cuda
```

`scripts/run_ablation_context.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for cfg in configs/ablation_context_128.yaml configs/ablation_context_256.yaml configs/ablation_context_512.yaml; do
  PYTHONPATH=src python -m mini_llm.train --config "$cfg" --device cuda
done
```

`scripts/run_ablation_position.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for cfg in configs/ablation_pos_none.yaml configs/ablation_pos_abs.yaml configs/ablation_pos_rope.yaml; do
  PYTHONPATH=src python -m mini_llm.train --config "$cfg" --device cuda
done
```

`scripts/run_sampling.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for strategy in greedy temperature top_k top_p; do
  PYTHONPATH=src python -m mini_llm.generate \
    --config configs/small.yaml \
    --checkpoint results/runs/small_main/checkpoint.pt \
    --prompt "人工智能的发展" \
    --strategy "$strategy" \
    --device cuda \
    --output "results/runs/small_main/sample_${strategy}.txt"
done
```

`scripts/run_kv_cache_benchmark.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m mini_llm.benchmark_kv_cache \
  --config configs/small.yaml \
  --checkpoint results/runs/small_main/checkpoint.pt \
  --prompt "人工智能的发展" \
  --lengths 64 128 256 512 \
  --device cuda \
  --output results/runs/small_main/kv_cache_benchmark.csv
```

`scripts/prepare_thucnews.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m mini_llm.tokenizer_train \
  --input data/processed/thucnews_clean.txt \
  --tokenizer data/tokenizer/bpe_16000.json \
  --vocab-size 16000
```

- [ ] **Step 3: Make scripts executable**

Run:

```bash
cd /Users/ttz/Documents/大模型推理/mini-llm-from-scratch
chmod +x scripts/*.sh
```

Expected: command exits with status 0.

- [ ] **Step 4: Update README with exact workflows**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/README.md`

Required sections:

```text
Mac setup:
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  PYTHONPATH=src pytest -q

GitHub sync:
  git add mini-llm-from-scratch docs/superpowers/plans .gitignore
  git commit -m "..."
  git push origin HEAD

CloudStudio setup:
  git clone or git pull
  cd mini-llm-from-scratch
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  nvidia-smi

Formal scripts:
  ./scripts/run_train_tiny.sh
  ./scripts/run_train_small.sh
  ./scripts/run_ablation_context.sh
  ./scripts/run_ablation_position.sh
  ./scripts/run_sampling.sh
  ./scripts/run_kv_cache_benchmark.sh
```

- [ ] **Step 5: Validate scripts**

Run:

```bash
cd /Users/ttz/Documents/大模型推理/mini-llm-from-scratch
bash -n scripts/*.sh
PYTHONPATH=src python -m mini_llm.plot --help
```

Expected:

```text
bash -n exits with status 0
plot help output includes --run_dir and --output_dir
```

- [ ] **Step 6: Commit**

Run:

```bash
cd /Users/ttz/Documents/大模型推理
git add mini-llm-from-scratch/src/mini_llm/plot.py mini-llm-from-scratch/scripts mini-llm-from-scratch/README.md
git commit -m "feat: add experiment scripts and plotting"
git push origin HEAD
```

Expected: commit and push succeed.

---

## Task 9: Full Local Verification Before Server Run

**Files:**
- Modify only files that fail tests in previous tasks.

**Tests:**
- Full CPU test suite
- py_compile for all Python modules
- smoke training

- [ ] **Step 1: Run complete Mac verification**

Run:

```bash
cd /Users/ttz/Documents/大模型推理/mini-llm-from-scratch
PYTHONPATH=src pytest -q
PYTHONPATH=src python -m py_compile src/mini_llm/*.py
./scripts/run_smoke_test.sh
```

Expected:

```text
pytest output shows all tests passed
py_compile exits with status 0
smoke script prints training progress and creates results/runs/smoke/metrics.csv
```

- [ ] **Step 2: Confirm no large artifacts are tracked**

Run:

```bash
cd /Users/ttz/Documents/大模型推理
git status --short
git check-ignore -v mini-llm-from-scratch/results/runs/smoke/checkpoint.pt
git check-ignore -v mini-llm-from-scratch/data/processed/smoke_train.bin
```

Expected:

```text
git check-ignore prints .gitignore rules for checkpoint.pt and smoke_train.bin
git status does not show results/runs or data/processed binary files as untracked
```

- [ ] **Step 3: Commit final local fixes**

Run:

```bash
cd /Users/ttz/Documents/大模型推理
git add mini-llm-from-scratch docs/superpowers/plans .gitignore
git commit -m "test: verify mini llm local smoke workflow"
git push origin HEAD
```

Expected: commit and push succeed, or `git commit` reports nothing to commit if no fixes were needed.

---

## Task 10: CloudStudio Linux Environment And Data Preparation

**Files:**
- Data files are generated on server and ignored by Git.
- Modify: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/experiments/ablation_scale.md` after successful preparation.

**Tests:**
- `nvidia-smi`
- `python -c "import torch; print(torch.cuda.is_available())"`
- tokenizer and binary files exist

- [ ] **Step 1: Pull code on CloudStudio**

Run on CloudStudio Linux:

```bash
git clone git@github.com:$(gh api user --jq .login)/mini-llm-from-scratch.git
cd mini-llm-from-scratch
git pull origin HEAD
cd mini-llm-from-scratch
```

Expected:

```text
clone succeeds
pwd ends with /mini-llm-from-scratch/mini-llm-from-scratch
```

- [ ] **Step 2: Create Linux environment**

Run on CloudStudio Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
nvidia-smi
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

Expected:

```text
torch.cuda.is_available() prints True
GPU name output includes A800
```

- [ ] **Step 3: Run server smoke test**

Run:

```bash
./scripts/run_smoke_test.sh
```

Expected:

```text
pytest output shows all tests passed
training output includes step=10 and step=20
results/runs/smoke/metrics.csv exists
```

- [ ] **Step 4: Prepare THUCNews text**

Run:

```bash
mkdir -p data/raw data/processed data/tokenizer
python - <<'PY'
from pathlib import Path
from mini_llm.data import clean_text
raw_dir = Path("data/raw")
out = Path("data/processed/thucnews_clean.txt")
texts = []
for path in sorted(raw_dir.rglob("*.txt")):
    texts.append(path.read_text(encoding="utf-8", errors="ignore"))
cleaned = clean_text("\n".join(texts), min_chars=10)
out.write_text(cleaned, encoding="utf-8")
print("chars", len(cleaned))
PY
```

Expected:

```text
chars <number greater than 5000000>
data/processed/thucnews_clean.txt exists
```

- [ ] **Step 5: Train tokenizer and write train/val bins**

Run:

```bash
PYTHONPATH=src python -m mini_llm.tokenizer_train \
  --input data/processed/thucnews_clean.txt \
  --tokenizer data/tokenizer/bpe_16000.json \
  --vocab-size 16000

python - <<'PY'
from pathlib import Path
from mini_llm.tokenizer_train import encode_file
from mini_llm.data import write_token_bins
ids = encode_file(Path("data/tokenizer/bpe_16000.json"), Path("data/processed/thucnews_clean.txt"))
write_token_bins(ids, Path("data/processed/train.bin"), Path("data/processed/val.bin"), val_fraction=0.05)
print("tokens", len(ids))
PY
```

Expected:

```text
tokenizer training completes
tokens <number greater than 10000000>
data/tokenizer/bpe_16000.json exists
data/processed/train.bin exists
data/processed/val.bin exists
```

- [ ] **Step 6: Record preparation summary**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/experiments/ablation_scale.md`

Append this structure with real numbers from server output:

```markdown
## Data Preparation

- Dataset: THUCNews subset
- Cleaned characters: recorded from `chars` output
- Encoded tokens: recorded from `tokens` output
- Tokenizer vocab size: 16000
- Train/val split: 95% / 5%
```

- [ ] **Step 7: Commit preparation notes from Mac after pulling server-created markdown**

Run on Mac:

```bash
cd /Users/ttz/Documents/大模型推理
git pull origin HEAD
git add mini-llm-from-scratch/experiments/ablation_scale.md
git commit -m "docs: record mini llm data preparation"
git push origin HEAD
```

Expected: commit and push succeed. Large raw/tokenized files remain ignored and are not pushed.

---

## Task 11: Run Formal Training Experiments On A800

**Files:**
- Generated ignored outputs under `mini-llm-from-scratch/results/runs/*`
- Modify: experiment markdown files with result tables and observations

**Tests:**
- `metrics.csv` exists for each run
- `checkpoint.pt` exists for tiny and small
- `val_loss` and `val_ppl` columns exist

- [ ] **Step 1: Train tiny model**

Run on CloudStudio Linux:

```bash
cd mini-llm-from-scratch
source .venv/bin/activate
./scripts/run_train_tiny.sh
```

Expected:

```text
training progress prints eval steps
results/runs/tiny_main/metrics.csv exists
results/runs/tiny_main/checkpoint.pt exists
```

- [ ] **Step 2: Train small model**

Run:

```bash
./scripts/run_train_small.sh
```

Expected:

```text
training progress prints eval steps
results/runs/small_main/metrics.csv exists
results/runs/small_main/checkpoint.pt exists
```

- [ ] **Step 3: Run context length ablation**

Run:

```bash
./scripts/run_ablation_context.sh
```

Expected:

```text
results/runs/context_128/metrics.csv exists
results/runs/context_256/metrics.csv exists
results/runs/context_512/metrics.csv exists
```

- [ ] **Step 4: Run position encoding ablation**

Run:

```bash
./scripts/run_ablation_position.sh
```

Expected:

```text
results/runs/pos_none/metrics.csv exists
results/runs/pos_abs/metrics.csv exists
results/runs/pos_rope/metrics.csv exists
```

- [ ] **Step 5: Plot curves**

Run:

```bash
for run in tiny_main small_main context_128 context_256 context_512 pos_none pos_abs pos_rope; do
  PYTHONPATH=src python -m mini_llm.plot --run_dir "results/runs/$run" --output_dir "results/runs/$run/figures"
done
```

Expected:

```text
each run directory contains figures/loss_curve.png
each run directory contains figures/ppl_curve.png
```

- [ ] **Step 6: Write experiment markdown summaries**

Files:

```text
experiments/ablation_scale.md
experiments/ablation_context.md
experiments/ablation_position.md
```

Required content:

```text
Run name
Config path
Final val loss
Final perplexity
Average tokens/s
Peak GPU memory
One short observation per run
```

- [ ] **Step 7: Commit summaries only**

Run on Mac after syncing markdown and selected figures:

```bash
cd /Users/ttz/Documents/大模型推理
git pull origin HEAD
git add mini-llm-from-scratch/experiments mini-llm-from-scratch/report
git commit -m "docs: summarize mini llm training experiments"
git push origin HEAD
```

Expected: commit and push succeed. Checkpoints and raw metrics remain ignored unless small CSV/PNG artifacts are intentionally copied into a tracked report artifact directory.

---

## Task 12: Run Sampling And KV Cache Experiments

**Files:**
- Modify: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/experiments/ablation_sampling.md`
- Modify: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/experiments/ablation_kv_cache.md`

**Tests:**
- generated sample files exist
- KV-cache CSV exists and correctness check passes

- [ ] **Step 1: Run sampling strategies**

Run on CloudStudio Linux:

```bash
cd mini-llm-from-scratch
source .venv/bin/activate
./scripts/run_sampling.sh
```

Expected:

```text
results/runs/small_main/sample_greedy.txt exists
results/runs/small_main/sample_temperature.txt exists
results/runs/small_main/sample_top_k.txt exists
results/runs/small_main/sample_top_p.txt exists
```

- [ ] **Step 2: Run KV-cache benchmark**

Run:

```bash
./scripts/run_kv_cache_benchmark.sh
```

Expected:

```text
output prints correctness passed
results/runs/small_main/kv_cache_benchmark.csv exists
CSV contains rows for no_cache and with_cache at lengths 64, 128, 256, 512
```

- [ ] **Step 3: Write sampling summary**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/experiments/ablation_sampling.md`

Required content:

```text
Prompt used
Strategy table: greedy, temperature, top_k, top_p
For each strategy: repeated tokens observation, coherence observation, representative sample file path
```

- [ ] **Step 4: Write KV-cache summary**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/experiments/ablation_kv_cache.md`

Required content:

```text
Correctness result
Benchmark table: length, no_cache tokens/s, with_cache tokens/s, speedup, peak memory
Short explanation: why cache speeds up autoregressive decoding and why it costs memory
```

- [ ] **Step 5: Commit summaries**

Run on Mac:

```bash
cd /Users/ttz/Documents/大模型推理
git pull origin HEAD
git add mini-llm-from-scratch/experiments/ablation_sampling.md mini-llm-from-scratch/experiments/ablation_kv_cache.md
git commit -m "docs: summarize sampling and kv cache experiments"
git push origin HEAD
```

Expected: commit and push succeed.

---

## Task 13: Final Project Verification

**Files:**
- Modify: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/report/experiment_report.md`
- Modify: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/README.md`

**Tests:**
- Full local test suite
- Linux smoke script
- Git clean check

- [ ] **Step 1: Write final report**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/report/experiment_report.md`

Required sections:

```text
Project goal
Environment: Mac development, Linux CloudStudio A800 training
Dataset and tokenizer
Model architecture
Training setup
Scale comparison
Context length ablation
Position encoding ablation
Sampling strategy comparison
KV Cache correctness and benchmark
Limitations
Reproducibility commands
```

- [ ] **Step 2: Final README update**

File: `/Users/ttz/Documents/大模型推理/mini-llm-from-scratch/README.md`

Required sections:

```text
Repository sync through GitHub
Mac local verification
CloudStudio setup
Data preparation
Training commands
Experiment commands
Artifact policy: checkpoints/data ignored, summaries tracked
```

- [ ] **Step 3: Run local verification**

Run:

```bash
cd /Users/ttz/Documents/大模型推理/mini-llm-from-scratch
PYTHONPATH=src pytest -q
PYTHONPATH=src python -m py_compile src/mini_llm/*.py
bash -n scripts/*.sh
```

Expected:

```text
all tests passed
py_compile exits with status 0
bash -n exits with status 0
```

- [ ] **Step 4: Run server verification**

Run on CloudStudio Linux:

```bash
cd mini-llm-from-scratch
source .venv/bin/activate
git pull origin HEAD
./scripts/run_smoke_test.sh
```

Expected:

```text
git pull completes
pytest output shows all tests passed
smoke training creates results/runs/smoke/metrics.csv
```

- [ ] **Step 5: Commit final docs**

Run on Mac:

```bash
cd /Users/ttz/Documents/大模型推理
git add mini-llm-from-scratch/README.md mini-llm-from-scratch/report/experiment_report.md docs/superpowers/plans/2026-06-04-mini-llm-from-scratch-implementation.md
git commit -m "docs: finalize mini llm implementation plan and report"
git push origin HEAD
```

Expected: commit and push succeed.

---

## Self-Review

Spec coverage:

- Git + GitHub synchronization is covered in Task 0 and repeated in server tasks.
- Linux-first execution is covered in scripts, README requirements, and CloudStudio tasks.
- File creation and responsibility mapping is covered in File Structure And Responsibilities.
- Tests are specified for config, data/tokenizer, RoPE/model, metrics/training, generation, KV cache, scripts, and final verification.
- Formal experiments include two model sizes, three context lengths, three position encoding modes, sampling, and KV cache.
- Resume writing is intentionally excluded from implementation scope.

Placeholder scan:

- The plan avoids deferred implementation labels and uses fixed file paths, fixed commands, fixed config names, and fixed commit messages.

Type consistency:

- `GPTConfig`, `GPTLanguageModel`, `load_config`, `TokenBlockDataset`, `build_rope_cache`, `apply_rope`, `perplexity`, `sample_next_token`, `greedy_no_cache`, and `greedy_with_cache` are introduced before later tasks reference them.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-04-mini-llm-from-scratch-implementation.md`. Two execution options:

**1. Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** - execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
