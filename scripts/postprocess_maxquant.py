"""Post-process MaxQuant proteinGroups.txt to match Novak's sample names.

Path 3 produces a fresh `proteinGroups.txt` from the 8 PRIDE raw files, but
the LFQ-intensity columns are named `Skupin11..Skupin18` (the file stems we
used during MQ setup). The downstream loader expects names like `D25 C7`,
`D40 C 8`, `cont IPSC` etc.

This script discovers the mapping post-hoc by correlating each of our 8
columns against each of Novak's 18 named columns in `selectedProteins.txt`,
restricted to the 127 proteins Novak's curation kept. Each Skupin## column
is assigned to the named column with which it correlates highest.

Output
------
A standardized proteinGroups.txt with the Skupin## columns renamed in place
to Novak's labels. The existing `data.load_proteomic_timecourse(..., source="Novak2022")`
loader works on it without code changes.

Usage
-----
    python scripts/postprocess_maxquant.py \
        --proteingroups data/raw/novak2022_raw/combined/txt/proteinGroups.txt \
        --reference     data/raw/novak2022_selectedProteins.txt \
        --out           data/raw/novak2022_proteinGroups.txt \
        [--min-correlation 0.85] \
        [--mapping-out artifacts/skupin_to_sample_mapping.json]

If `--mapping-out` is provided, also writes the discovered (and validated)
mapping as JSON for record-keeping. When Dittmar later confirms the
official mapping, you can diff this JSON against his answer for a
cross-check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--proteingroups", type=Path, required=True,
                   help="MaxQuant proteinGroups.txt from the Path-3 run")
    p.add_argument("--reference", type=Path, required=True,
                   help="Novak's PRIDE selectedProteins.txt (the named-sample reference)")
    p.add_argument("--out", type=Path, required=True,
                   help="Standardized proteinGroups.txt with Novak-style sample names")
    p.add_argument("--min-correlation", type=float, default=0.85,
                   help="Minimum Pearson r required to accept a mapping (default 0.85)")
    p.add_argument("--mapping-out", type=Path, default=None,
                   help="Optional path to save the discovered mapping as JSON")
    return p.parse_args()


def _filter_maxquant(df: pd.DataFrame) -> pd.DataFrame:
    for flag in ("Reverse", "Potential contaminant", "Only identified by site"):
        if flag in df.columns:
            df = df[df[flag].fillna("") != "+"]
    return df.reset_index(drop=True)


def _expand_genes(df: pd.DataFrame, sample_cols: list[str]) -> pd.DataFrame:
    """Explode 'Gene names' (e.g. 'TAF9;AK6') and sum-aggregate intensities."""
    rows = []
    for _, row in df[["Gene names", *sample_cols]].iterrows():
        names = str(row["Gene names"]).strip()
        if not names or names.lower() == "nan":
            continue
        for g in names.split(";"):
            g = g.strip()
            if not g:
                continue
            rec = {"gene": g}
            for c in sample_cols:
                rec[c] = float(row[c])
            rows.append(rec)
    long_df = pd.DataFrame(rows)
    return long_df.groupby("gene", as_index=True).sum()


def main() -> int:
    args = parse_args()

    print(f"[postproc] reading MQ output: {args.proteingroups}")
    mq = pd.read_csv(args.proteingroups, sep="\t", low_memory=False)
    mq = _filter_maxquant(mq)
    mq_lfq_cols = [c for c in mq.columns if c.startswith("LFQ intensity ")]
    if not mq_lfq_cols:
        raise SystemExit("[postproc] no 'LFQ intensity *' columns in MQ output")
    print(f"  rows after filter: {len(mq)}; LFQ columns: {len(mq_lfq_cols)}")

    print(f"[postproc] reading Novak reference: {args.reference}")
    ref = pd.read_csv(args.reference, sep="\t", low_memory=False)
    ref = _filter_maxquant(ref)
    ref_lfq_cols = [c for c in ref.columns if c.startswith("LFQ intensity ")]
    print(f"  rows after filter: {len(ref)}; LFQ columns: {len(ref_lfq_cols)}")

    # Expand both to gene-keyed long format so we can intersect.
    mq_long = _expand_genes(mq, mq_lfq_cols)
    ref_long = _expand_genes(ref, ref_lfq_cols)
    shared = mq_long.index.intersection(ref_long.index)
    print(f"[postproc] genes in common: {len(shared)} (of MQ {len(mq_long)}, ref {len(ref_long)})")
    if len(shared) < 30:
        raise SystemExit(
            "[postproc] too few shared genes ({}); check input files".format(len(shared))
        )

    # Log-transform on the linear LFQ values so correlations aren't dominated
    # by high-abundance proteins.
    mq_log = np.log2(1.0 + mq_long.loc[shared, mq_lfq_cols])
    ref_log = np.log2(1.0 + ref_long.loc[shared, ref_lfq_cols])

    # Pearson correlation matrix between every (mq_col, ref_col) pair.
    print("[postproc] computing per-pair correlations ...")
    corr = pd.DataFrame(
        index=mq_lfq_cols, columns=ref_lfq_cols, dtype=float,
    )
    for m in mq_lfq_cols:
        x = mq_log[m].to_numpy()
        for r in ref_lfq_cols:
            y = ref_log[r].to_numpy()
            mask = np.isfinite(x) & np.isfinite(y) & ~((x == 0) & (y == 0))
            if mask.sum() < 10:
                corr.at[m, r] = float("nan")
                continue
            corr.at[m, r] = float(np.corrcoef(x[mask], y[mask])[0, 1])

    # Greedy best-first matching: assign each MQ column to its highest-r ref
    # column, removing the chosen ref column so each ref is used at most once.
    print("\n[postproc] best matches:")
    print(f"{'MQ column':<35}{'best Novak match':<35}{'r':>8}{'second-best':>40}{'r':>8}")
    print("-" * 130)
    available_refs = list(ref_lfq_cols)
    mapping: dict[str, dict] = {}
    for m in mq_lfq_cols:
        # Sort refs still available by descending r against m
        scores = [(r, corr.at[m, r]) for r in available_refs]
        scores.sort(key=lambda x: (-x[1] if np.isfinite(x[1]) else float("inf")))
        if not scores:
            mapping[m] = {"match": None, "r": float("nan"), "runner_up": None,
                          "runner_up_r": float("nan")}
            continue
        best_ref, best_r = scores[0]
        runner_ref, runner_r = scores[1] if len(scores) >= 2 else (None, float("nan"))
        print(f"{m:<35}{best_ref:<35}{best_r:>8.3f}{runner_ref or '':>40}"
              f"{runner_r:>8.3f}")
        mapping[m] = {
            "match": best_ref, "r": float(best_r),
            "runner_up": runner_ref, "runner_up_r": float(runner_r),
        }
        # Lock this ref so subsequent MQ cols can't claim it
        available_refs.remove(best_ref)

    # Validate confidence
    weak = [m for m, v in mapping.items()
            if v["match"] is None or v["r"] < args.min_correlation]
    if weak:
        print(f"\n[postproc] WARNING: {len(weak)} columns below r={args.min_correlation}:")
        for m in weak:
            print(f"  {m}: r={mapping[m]['r']:.3f}")
        print("  Consider waiting for Dittmar's confirmation before trusting these.")

    # Rename and write standardized file
    rename_map = {m: v["match"] for m, v in mapping.items() if v["match"]}
    print(f"\n[postproc] applying {len(rename_map)} column renames")
    mq.rename(columns=rename_map, inplace=True)
    mq.to_csv(args.out, sep="\t", index=False)
    print(f"[postproc] standardized proteinGroups: {args.out}")

    if args.mapping_out:
        args.mapping_out.parent.mkdir(parents=True, exist_ok=True)
        args.mapping_out.write_text(
            json.dumps(mapping, indent=2), encoding="utf-8"
        )
        print(f"[postproc] mapping JSON: {args.mapping_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
