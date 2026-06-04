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
