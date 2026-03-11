#!/usr/bin/env bash
set -euo pipefail

# Download color-constancy datasets used for WPA evaluation.
# Modes:
#   quick: download metadata/small files (fast)
#   full : additionally download larger image archives
#
# Usage:
#   bash scripts/download_datasets.sh quick
#   bash scripts/download_datasets.sh full

MODE="${1:-quick}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT_DIR/datasets/raw"
META_DIR="$ROOT_DIR/datasets/metadata"

mkdir -p "$DATA_DIR"/{rendered_wb,shi_gehler,cubepp} "$META_DIR"

fetch() {
  local url="$1"
  local out="$2"
  echo "[download] $url"
  curl -fL --retry 4 --retry-delay 3 -C - -o "$out" "$url"
}

echo "[info] mode=$MODE"

echo "[step] Fetch Rendered WB page and extract links..."
RWB_PAGE="$META_DIR/rendered_wb_dataset.html"
fetch "https://yorkucvil.github.io/projects/public_html/sRGB_WB_correction/dataset.html" "$RWB_PAGE"

RWB_URLS="$META_DIR/rendered_wb_sync_links.txt"
python - "$RWB_PAGE" "$RWB_URLS" <<'PY'
import re
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
urls = sorted(set(re.findall(r"https://ln2\.sync\.com/dl/[^\s\"'<>]+", src)))
Path(sys.argv[2]).write_text("\n".join(urls) + ("\n" if urls else ""), encoding="utf-8")
print(f"[info] extracted_sync_links={len(urls)}")
PY

echo "[step] Fetch WB_sRGB helper repo metadata snapshot..."
fetch "https://codeload.github.com/mahmoudnafifi/WB_sRGB/zip/refs/heads/master" \
  "$DATA_DIR/rendered_wb/WB_sRGB_master.zip"

echo "[step] Fetch Shi-Gehler metadata..."
fetch "https://www.cs.sfu.ca/~colour/data/shi_gehler/" "$META_DIR/shi_gehler_index.html"

echo "[step] Download Shi-Gehler ground truth..."
fetch "https://www.cs.sfu.ca/~colour/data/shi_gehler/groundtruth_568.zip" \
  "$DATA_DIR/shi_gehler/groundtruth_568.zip"

if [[ "$MODE" == "full" ]]; then
  echo "[step] Download Shi-Gehler PNG archives (large)..."
  fetch "https://www.cs.sfu.ca/~colour/data2/shi_gehler/png_canon1d.zip" \
    "$DATA_DIR/shi_gehler/png_canon1d.zip"
  fetch "https://www.cs.sfu.ca/~colour/data2/shi_gehler/png_canon5d_1.zip" \
    "$DATA_DIR/shi_gehler/png_canon5d_1.zip"
  fetch "https://www.cs.sfu.ca/~colour/data2/shi_gehler/png_canon5d_2.zip" \
    "$DATA_DIR/shi_gehler/png_canon5d_2.zip"
  fetch "https://www.cs.sfu.ca/~colour/data2/shi_gehler/png_canon5d_3.zip" \
    "$DATA_DIR/shi_gehler/png_canon5d_3.zip"
fi

echo "[step] Fetch Cube++ README metadata..."
fetch "https://raw.githubusercontent.com/Visillect/CubePlusPlus/master/README.md" \
  "$META_DIR/cubepp_readme.md"

if [[ "$MODE" == "full" ]]; then
  echo "[step] Download SimpleCube++ (about 2GB)..."
  fetch "https://zenodo.org/record/4153431/files/SimpleCube%2B%2B.zip?download=1" \
    "$DATA_DIR/cubepp/SimpleCubepp.zip"
fi

echo "[done] Dataset collection finished (mode=$MODE)."
echo "[hint] Rendered WB sync links saved at: $RWB_URLS"

