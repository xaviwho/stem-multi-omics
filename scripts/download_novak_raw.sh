#!/usr/bin/env bash
# Download the 8 Novak 2022 raw files from PRIDE for Path 3 (local MaxQuant rerun).
#
# Files are sequentially numbered Skupin11..Skupin18 with PRIDE IDs
# Qplus_00029245..Qplus_00029252. Each is ~1.1 GB → ~9 GB total.
# Resumable via curl -C - so re-runs pick up where they left off.
#
# Usage:
#     bash scripts/download_novak_raw.sh [--start N] [--end M]
# Defaults: --start 11 --end 18 (all 8 files).
# Pass --start 12 to skip Skupin11 if already downloaded.

set -euo pipefail

START=11
END=18
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START=$2; shift 2 ;;
    --end)   END=$2; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

OUT_DIR="data/raw/novak2022_raw"
mkdir -p "$OUT_DIR"

# Maps Skupin## -> PRIDE Qplus ID.
# Qplus_00029245 = Skupin11, ..., Qplus_00029252 = Skupin18.
declare -A QPLUS_ID
for i in $(seq 11 18); do
  qid=$(printf "%08d" $((29245 + i - 11)))
  QPLUS_ID[$i]=$qid
done

BASE="https://ftp.pride.ebi.ac.uk/pride/data/archive/2022/02/PXD028283"
for i in $(seq "$START" "$END"); do
  qid=${QPLUS_ID[$i]}
  src_name="Qplus_${qid}_MQQC_HS_SR_Skupin${i}.raw"
  dst_path="${OUT_DIR}/Skupin${i}.raw"
  url="${BASE}/${src_name}"

  echo "[$(date +%H:%M:%S)] Skupin${i} <- ${url}"
  curl -fL -C - --retry 3 --retry-delay 5 -o "$dst_path" "$url"
  size=$(stat -c%s "$dst_path" 2>/dev/null || echo 0)
  echo "[$(date +%H:%M:%S)] Skupin${i} done ($((size / 1024 / 1024)) MB)"
done

echo
echo "All requested files downloaded to ${OUT_DIR}/"
ls -la "$OUT_DIR"
