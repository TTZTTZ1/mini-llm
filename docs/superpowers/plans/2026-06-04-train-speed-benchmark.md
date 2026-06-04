# Train Speed Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a server-side benchmark that measures training throughput and GPU memory for large mini-LLM presets across batch sizes.

**Architecture:** The benchmark uses synthetic random token batches so dataset size and disk IO do not affect measurements. A Python module performs warmup and timed training steps, catches CUDA OOM, and writes CSV rows; a shell script provides the default large_250m sweep for A800 tuning.

**Tech Stack:** Python, PyTorch, existing `mini_llm.model`, bash, pytest.

---

### Task 1: Benchmark Module

**Files:**
- Create: `mini-llm-from-scratch/src/mini_llm/benchmark_train_speed.py`
- Test: `mini-llm-from-scratch/tests/test_train_speed_benchmark.py`

- [ ] Write failing tests for preset construction, token math, and a one-step CPU benchmark.
- [ ] Implement preset configs, a single-batch-size benchmark function, CSV writing, and CLI parsing.
- [ ] Run `PYTHONPATH=src .venv/bin/pytest -q tests/test_train_speed_benchmark.py`.

### Task 2: Shell Entry Point

**Files:**
- Create: `mini-llm-from-scratch/scripts/benchmark_train_speed.sh`
- Modify: `mini-llm-from-scratch/tests/test_scripts.py`
- Modify: `mini-llm-from-scratch/README.md`

- [ ] Add shell script that defaults to `large_250m`, block size 1024, batch sizes `4 8 12 16 24 32`, warmup 5, timed 20.
- [ ] Add script test for Linux-safe syntax and expected module invocation.
- [ ] Document the server command and how to interpret CSV output.

### Task 3: Verification and Publish

**Files:**
- Verify all modified files.

- [ ] Run `PYTHONPATH=src .venv/bin/pytest -q`.
- [ ] Run `bash -n scripts/*.sh`.
- [ ] Commit locally and sync only mini-llm files to `TTZTTZ1/mini-llm`.
