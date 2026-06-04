import os
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from mini_llm.data import TOKEN_DTYPE


def test_download_thucnews_script_exists_and_is_linux_safe():
    script = Path("scripts/download_thucnews.sh")
    assert script.exists()
    assert os.access(script, os.X_OK)
    text = script.read_text(encoding="utf-8")
    assert "THUCNEWS_SOURCE" in text
    assert "dirtycomputer/THUCNews" in text
    assert "thunlp.oss-cn-qingdao.aliyuncs.com/THUCNews.zip" in text
    assert "/Users/" not in text
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_download_thucnews_cnews_uses_existing_huggingface_file(tmp_path):
    script = Path("scripts/download_thucnews.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl_log = tmp_path / "curl.log"
    fake_curl = bin_dir / "curl"
    fake_curl.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
out=""
url=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -o)
      shift
      out="$1"
      ;;
    http*)
      url="$1"
      ;;
  esac
  shift || true
done
echo "$url" >> "$CURL_LOG"
mkdir -p "$(dirname "$out")"
printf "sample text\\n" > "$out"
""",
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
            "CURL_LOG": str(curl_log),
            "FORCE": "1",
            "THUCNEWS_CNEWS_BASE_URL": "https://example.test",
            "THUCNEWS_DEST_DIR": str(tmp_path / "raw"),
            "THUCNEWS_DOWNLOAD_DIR": str(tmp_path / "downloads"),
        }
    )

    subprocess.run(["bash", str(script)], check=True, env=env)

    urls = curl_log.read_text(encoding="utf-8").splitlines()
    assert urls == ["https://example.test/cnews.train.txt"]


def test_benchmark_train_speed_script_runs_large_preset():
    script = Path("scripts/benchmark_train_speed.sh")
    assert script.exists()
    assert os.access(script, os.X_OK)
    text = script.read_text(encoding="utf-8")
    assert "mini_llm.benchmark_train_speed" in text
    assert "for preset in large_250m" in text
    assert '--preset "$preset"' in text
    assert "large_320m" in text
    assert "large_370m" in text
    assert "large_450m" in text
    assert "large_560m" in text
    assert "--batch-sizes 4 8 12 16 20 24 32" in text
    assert "train_speed_${preset}.csv" in text
    assert "/Users/" not in text
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_benchmark_midrange_train_speed_script_runs_midrange_presets():
    script = Path("scripts/benchmark_midrange_train_speed.sh")
    assert script.exists()
    assert os.access(script, os.X_OK)
    text = script.read_text(encoding="utf-8")
    assert "mid_134m" in text
    assert "mid_163m" in text
    assert "mid_191m" in text
    assert "mid_219m" in text
    assert "mid_212m" in text
    assert "--batch-sizes 16 24 32 40 48 56 64" in text
    assert "train_speed_${preset}.csv" in text
    assert "/Users/" not in text
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_prepare_fineweb_edu_script_uses_streaming_dataset_defaults():
    script = Path("scripts/prepare_fineweb_edu.sh")
    assert script.exists()
    assert os.access(script, os.X_OK)
    text = script.read_text(encoding="utf-8")
    assert "mini_llm.pretrain_data" in text
    assert "HuggingFaceFW/fineweb-edu" in text
    assert "sample-10BT" in text
    assert "TARGET_TRAIN_TOKENS" in text
    assert "3200000000" in text
    assert "VAL_FRACTION" in text
    assert "0.01" not in text
    assert "/Users/" not in text
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_run_final_and_suite_scripts_are_config_driven():
    for script_name in ["run_final_212m.sh", "run_experiment_suite.sh", "run_posttrain_evals.sh"]:
        script = Path("scripts") / script_name
        assert script.exists()
        assert os.access(script, os.X_OK)
        text = script.read_text(encoding="utf-8")
        assert "configs/final_212m_rope_ctx1024.yaml" in text
        assert "/Users/" not in text
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_run_final_212m_script_saves_stdout_and_stderr_log(tmp_path):
    script = Path("scripts/run_final_212m.sh")
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(
        """#!/usr/bin/env bash
printf 'stdout marker: %s\\n' "$*"
printf 'stderr marker\\n' >&2
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    out_dir = tmp_path / "run"

    env = os.environ.copy()
    env.update(
        {
            "PYTHON": str(fake_python),
            "OUT_DIR_OVERRIDE": str(out_dir),
            "MAX_STEPS_OVERRIDE": "1",
            "DEVICE": "cpu",
        }
    )

    subprocess.run(["bash", str(script)], check=True, env=env)

    logs = list(out_dir.glob("train_*.log"))
    assert len(logs) == 1
    log_text = logs[0].read_text(encoding="utf-8")
    assert "stdout marker:" in log_text
    assert "stderr marker" in log_text
    assert "mini_llm.train" in log_text


def test_run_experiment_suite_saves_per_config_logs(tmp_path):
    script = Path("scripts/run_experiment_suite.sh")
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(
        """#!/usr/bin/env bash
printf 'suite stdout: %s\\n' "$*"
printf 'suite stderr\\n' >&2
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    log_dir = tmp_path / "logs"

    env = os.environ.copy()
    env.update(
        {
            "PYTHON": str(fake_python),
            "LOG_DIR": str(log_dir),
            "CONFIGS": "configs/final_212m_rope_ctx1024.yaml configs/scale_134m_rope_ctx1024.yaml",
            "MAX_STEPS_OVERRIDE": "1",
            "DEVICE": "cpu",
        }
    )

    subprocess.run(["bash", str(script)], check=True, env=env)

    logs = sorted(log_dir.glob("*.log"))
    assert len(logs) == 2
    all_text = "\n".join(log.read_text(encoding="utf-8") for log in logs)
    assert "suite stdout:" in all_text
    assert "suite stderr" in all_text
    assert "configs/final_212m_rope_ctx1024.yaml" in all_text
    assert "configs/scale_134m_rope_ctx1024.yaml" in all_text


def test_prepare_fineweb_edu_script_runs_local_text_end_to_end(tmp_path):
    script = Path("scripts/prepare_fineweb_edu.sh")
    local_text = tmp_path / "local corpus.txt"
    local_text.write_text(
        "\n".join(
            [
                "alpha beta gamma delta epsilon zeta eta theta iota kappa",
                "language model training data attention cache transformer rope",
                "news text pretraining validation tokens dataset stream",
            ]
            * 20
        ),
        encoding="utf-8",
    )
    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"
    manifest = tmp_path / "manifest.json"

    env = os.environ.copy()
    env.update(
        {
            "PYTHON": os.environ.get("PYTHON", sys.executable),
            "PYTHONPATH": "src",
            "LOCAL_TEXT_FILE": str(local_text),
            "TOKENIZER_CORPUS": str(tmp_path / "tokenizer_corpus.txt"),
            "TOKENIZER_PATH": str(tmp_path / "tokenizer.json"),
            "TRAIN_BIN": str(train_bin),
            "VAL_BIN": str(val_bin),
            "MANIFEST": str(manifest),
            "VOCAB_SIZE": "128",
            "TOKENIZER_TRAIN_DOCS": "100",
            "TOKENIZER_TRAIN_CHARS": "10000",
            "TARGET_TRAIN_TOKENS": "80",
            "TARGET_VAL_TOKENS": "20",
            "MIN_CHARS": "5",
            "FORCE_TOKENIZER": "1",
        }
    )

    subprocess.run(["bash", str(script)], check=True, env=env)

    assert np.memmap(train_bin, dtype=TOKEN_DTYPE, mode="r").shape[0] == 80
    assert np.memmap(val_bin, dtype=TOKEN_DTYPE, mode="r").shape[0] == 20
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["dataset_name"] == "local_text"
    assert payload["token_bins"]["train_tokens"] == 80
    assert payload["token_bins"]["val_tokens"] == 20
