"""Aggregate per-raw-file metadata JSONs into a CSV vial/timestamp table.

Produces `artifacts/novak2022_vial_table.csv` with one row per raw file.
The `sample_identity` column is left blank for Dittmar to fill in (or for
the post-hoc abundance matcher to populate after MaxQuant runs).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "novak2022_raw"
OUT_PATH = ROOT / "artifacts" / "novak2022_vial_table.csv"

STEMS = [f"Skupin{i}" for i in range(11, 19)]


def main() -> int:
    rows = []
    for stem in STEMS:
        meta_path = RAW_DIR / f"{stem}-metadata.json"
        if not meta_path.exists():
            print(f"  [skip] no metadata for {stem}", file=sys.stderr)
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sample = {e["name"]: e["value"] for e in meta.get("SampleData", [])}
        file_props = {e["name"]: e["value"] for e in meta.get("FileProperties", [])}
        method_full = sample.get("device acquisition method", "")
        method_short = method_full.split("\\")[-1] if method_full else ""
        rows.append({
            "file_stem": stem,
            "vial": sample.get("Vial", ""),
            "row": sample.get("Row", ""),
            "sample_number": sample.get("sample number", ""),
            "injection_volume_uL": sample.get("injection volume setting", ""),
            "acquired_utc": file_props.get("Content Creation Date", ""),
            "operator": sample.get("User name", ""),
            "method": method_short,
            "sample_identity": "",   # filled by Dittmar reply or post-hoc match
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"wrote {len(rows)} rows -> {OUT_PATH}")
    for r in rows:
        print(
            f"  {r['file_stem']}: vial={r['vial']:<5} row={r['row']:<3} "
            f"inj_uL={r['injection_volume_uL']:<5} acquired={r['acquired_utc']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
