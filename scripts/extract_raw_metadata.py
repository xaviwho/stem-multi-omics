"""Extract sample-identity metadata from Thermo .raw files via ThermoRawFileParser.

Goal: discover whether the .raw file headers Novak/Dittmar deposited contain
sample names that map Skupin11..Skupin18 to (CTRL/PINK1, D25/D40, replicate).
If they do, Path 3 is unblocked without a follow-up email to Dittmar.

Usage
-----
    python scripts/extract_raw_metadata.py <raw_file> [<raw_file> ...]

Outputs a small per-file summary with the most relevant sample-identifying
fields (Sample.Name, Sample.Comment, Sample.Vial, Sample.ID, Instrument.Name)
plus a one-line verdict: "informative" if a non-generic sample name was
recovered, "generic" if the field just echoes the file name, "missing" if
the field is absent.

Requires the ThermoRawFileParser binary at tools/ThermoRawFileParser/
ThermoRawFileParser.exe (download script is scripts/setup_trfp.sh, but
the binary already lives at that path after the previous setup turn).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRFP = ROOT / "tools" / "ThermoRawFileParser" / "ThermoRawFileParser.exe"


def _looks_generic(name: str, file_stem: str) -> bool:
    """A sample name is 'generic' if it just echoes the file name."""
    if not name:
        return True
    n = name.strip().lower()
    fs = file_stem.lower()
    # Exact match, file-stem prefix, or just the Skupin## tail
    if n == fs or n.startswith(fs) or fs.startswith(n):
        return True
    if re.fullmatch(r"skupin\s*\d+", n):
        return True
    return False


def _extract_one(raw_path: Path) -> dict:
    """Run TRFP -m 0 (JSON metadata) on one .raw file."""
    if not TRFP.exists():
        raise RuntimeError(
            f"ThermoRawFileParser not found at {TRFP}. "
            "Download with: scripts/setup_trfp.sh"
        )
    # --metadata=0 → JSON; --format=4 → no spectra output (metadata only).
    # Note: -s (stdout) cannot be combined with -o (output directory). We let
    # TRFP write the metadata file to disk, then read it back.
    proc = subprocess.run(
        [str(TRFP), "-i", str(raw_path), "-m", "0", "-f", "4",
         "-o", str(raw_path.parent)],
        capture_output=True, text=True, check=False,
    )
    # TRFP writes metadata to <stem>-metadata.json next to the raw file.
    meta_file = raw_path.with_name(raw_path.stem + "-metadata.json")
    if not meta_file.exists():
        raise RuntimeError(
            f"TRFP did not produce metadata for {raw_path}.\n"
            f"  stdout: {proc.stdout[:300]}\n  stderr: {proc.stderr[:300]}"
        )
    return json.loads(meta_file.read_text(encoding="utf-8"))


def _interesting_fields(meta: dict) -> dict[str, str]:
    """Pull the fields most likely to carry the human sample identity.

    TRFP's metadata JSON has the shape
        { "SampleData": [{"name": "sample name", "value": "Skupin11"}, ...], ... }
    so we collect entries from SampleData (and a few from FileProperties)
    whose `name` matches our shortlist.
    """
    sample_field_names = {
        "sample name", "sample number", "sample id",
        "Vial", "Row", "dilution factor", "injection volume setting",
        "User name", "Client", "Comment", "device acquisition method",
    }
    file_field_names = {"Content Creation Date"}

    out: dict[str, str] = {}
    for entry in meta.get("SampleData", []):
        name = str(entry.get("name", "")).strip()
        value = entry.get("value", "")
        if name in sample_field_names and value not in (None, ""):
            out[name] = str(value).strip()
    for entry in meta.get("FileProperties", []):
        name = str(entry.get("name", "")).strip()
        value = entry.get("value", "")
        if name in file_field_names and value not in (None, ""):
            out[name] = str(value).strip()
    return out


def main(argv: list[str]) -> int:
    if not argv:
        raw_dir = ROOT / "data" / "raw" / "novak2022_raw"
        files = sorted(raw_dir.glob("*.raw"))
        if not files:
            print(f"no .raw files found in {raw_dir}")
            return 2
    else:
        files = [Path(p) for p in argv]

    print(f"{'file':<32}{'verdict':<14}fields")
    print("-" * 80)
    for f in files:
        try:
            meta = _extract_one(f)
        except Exception as e:
            print(f"{f.stem:<32}{'error':<14}{e}")
            continue

        fields = _interesting_fields(meta)
        sample_name = fields.get("sample name", "")
        verdict = (
            "informative" if not _looks_generic(sample_name, f.stem) else
            "generic" if sample_name else "missing"
        )
        kv = "  ".join(f"{k}={v!r}" for k, v in fields.items())
        print(f"{f.stem:<24}{verdict:<14}{kv}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
