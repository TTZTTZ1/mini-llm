#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

SOURCE="${THUCNEWS_SOURCE:-cnews}"
DEST_DIR="${THUCNEWS_DEST_DIR:-data/raw/thucnews}"
FORCE="${FORCE:-0}"
DOWNLOAD_DIR="${THUCNEWS_DOWNLOAD_DIR:-data/raw/_downloads}"
EXTRACT_DIR="${THUCNEWS_EXTRACT_DIR:-data/raw/_extract}"
CNEWS_BASE_URL="${THUCNEWS_CNEWS_BASE_URL:-https://huggingface.co/datasets/dirtycomputer/THUCNews/resolve/main}"
FULL_URL="${THUCNEWS_FULL_URL:-https://thunlp.oss-cn-qingdao.aliyuncs.com/THUCNews.zip}"

download_file() {
  local url="$1"
  local output="$2"
  mkdir -p "$(dirname "$output")"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --retry-delay 2 -o "$output" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$output" "$url"
  else
    echo "Neither curl nor wget is available. Install one of them and retry." >&2
    exit 1
  fi
}

has_existing_text() {
  [ -d "$DEST_DIR" ] && find "$DEST_DIR" -type f -name '*.txt' -print -quit | grep -q .
}

if [ "$FORCE" != "1" ] && has_existing_text; then
  echo "Existing THUCNews text files found under $DEST_DIR; set FORCE=1 to re-download."
  exit 0
fi

rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR" "$DOWNLOAD_DIR"

case "$SOURCE" in
  cnews)
    echo "Downloading THUCNews cnews subset into $DEST_DIR"
    for split in train val test; do
      download_file "$CNEWS_BASE_URL/cnews.${split}.txt" "$DEST_DIR/cnews.${split}.txt"
    done
    ;;
  full)
    archive="$DOWNLOAD_DIR/THUCNews.zip"
    echo "Downloading full THUCNews zip into $archive"
    download_file "$FULL_URL" "$archive"
    rm -rf "$EXTRACT_DIR"
    mkdir -p "$EXTRACT_DIR"
    THUCNEWS_ARCHIVE="$archive" THUCNEWS_DEST_DIR="$DEST_DIR" THUCNEWS_MAX_FILES="${THUCNEWS_MAX_FILES:-0}" python - <<'PY'
from pathlib import Path
import os
import zipfile

archive = Path(os.environ["THUCNEWS_ARCHIVE"])
dest = Path(os.environ["THUCNEWS_DEST_DIR"])
max_files = int(os.environ.get("THUCNEWS_MAX_FILES", "0"))
count = 0

with zipfile.ZipFile(archive) as zf:
    for info in zf.infolist():
        if info.is_dir() or not info.filename.endswith(".txt"):
            continue
        parts = Path(info.filename).parts
        rel_parts = parts[1:] if len(parts) > 1 else parts
        if not rel_parts:
            continue
        target = dest / Path(*rel_parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, target.open("wb") as out:
            out.write(src.read())
        count += 1
        if max_files > 0 and count >= max_files:
            break

print(f"extracted_text_files {count}")
if count == 0:
    raise SystemExit("no .txt files found in THUCNews zip")
PY
    ;;
  *)
    echo "Unsupported THUCNEWS_SOURCE=$SOURCE. Use cnews or full." >&2
    exit 1
    ;;
esac

echo "THUCNews text files ready under $DEST_DIR"
find "$DEST_DIR" -type f -name '*.txt' | head -n 5
