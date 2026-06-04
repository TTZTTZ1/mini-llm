import os
import subprocess
from pathlib import Path


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
    assert "--preset large_250m" in text
    assert "--batch-sizes 4 8 12 16 24 32" in text
    assert "/Users/" not in text
    subprocess.run(["bash", "-n", str(script)], check=True)
