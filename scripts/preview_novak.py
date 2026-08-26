"""Preview the Novak 2022 proteomic loader + 3D state assembly.

Loads PXD028283 selectedProteins.txt (or the full proteinGroups.txt if
Skupin shares it), shows the CTRL sample structure detected, the per-stage
Pr_t scores, and the assembled (200, 3, 3) tensor that extends the ICUFN
2D baseline.

Use this to sanity-check the loader before running the full pipeline. With
the public 127-protein curation, expect Pr to be 0/NaN for the canonical
G_Pr markers — that's the M-A finding, not a bug. With the full proteome
(when received), at least TH and ALDH1A1 should produce real Pr values.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Minimal .env loader (no python-dotenv dep).
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip())

from stem_dt import data, markers, scores  # noqa: E402

ICUFN = os.environ.get("ICUFN_REPO_PATH") or str(ROOT.parent / "stem-cell-digital-twin")
NOVAK = os.environ["NOVAK2022_PROTEINGROUPS_PATH"]


def main() -> int:
    print(f"[novak] file: {NOVAK}")
    tc = data.load_proteomic_timecourse(NOVAK, source="Novak2022")
    print(f"[novak] proteins (after gene-name expansion): {tc.abundance.shape[0]}")
    print(f"[novak] CTRL samples found: {len(tc.samples)}")
    print(tc.samples[["sample_label", "condition", "stage"]].to_string())

    print("\n[novak] G_Pr coverage in this file:")
    for g in markers.G_Pr:
        present = g in tc.abundance.index
        print(f"  {g:<10} {'FOUND' if present else 'missing'}")

    print("\n[Pr_t] per-stage Pr score from Novak data:")
    pr = scores.per_stage_pr_score(tc, list(markers.G_Pr))
    for stage in ("D11", "D30", "D52"):
        r = pr[stage]
        print(
            f"  {stage:<6} mean={r.mean:>8.3f}  std={r.std:>6.3f}  "
            f"n_reps={r.n_replicates}  found={list(r.markers_found)}  "
            f"missing={list(r.markers_missing)}"
        )

    print("\n[3D] assembling (P, D, Pr) state tensor on top of ICUFN baseline ...")
    baseline = data.load_icufn_baseline(ICUFN)
    state_3d = scores.assemble_3d_state(baseline.state, pr, seed=42)
    print(f"[3D] state shape: {state_3d.shape}")
    for t_idx, stage in enumerate(baseline.stage_order):
        col = state_3d[:, t_idx, 2]
        print(
            f"  Pr at {stage}: mean={col.mean():.3f}  std={col.std():.3f}  "
            f"min={col.min():.3f}  max={col.max():.3f}"
        )

    found_any = any(np.isfinite(pr[s].mean) for s in pr)
    if not found_any:
        print("\nNOTE: G_Pr markers are absent from the public 127-protein curation.")
        print("      Loader is working correctly; Pr is degenerate until full")
        print("      proteinGroups.txt is available (Skupin email pending).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
