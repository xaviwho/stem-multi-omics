"""K_t kinase-pathway-activity ablation — 4D `(P, D, Pr, K)` vs 3D `(P, D, Pr)`.

Per the methodology (Path A.1):
  K_t is a transcript-derived phospho-pathway activity score, NOT measured
  phosphoproteomics. We score the kinase-substrate gene sets curated from
  PhosphoSitePlus (G_Ph upstream pathways) on each Jerber cell, aggregate
  per stage, and append as a 4th feature to the 3D state.

This is a sensitivity ablation: if 4D > 3D > 3D_perm, K carries genuine
auxiliary signal; if 4D ≈ 3D, K is correlated enough with (P, D) to be
redundant — Paper 1's primary contribution stays 3D, K becomes a sentence
in §4.

Usage
-----
    python scripts/run_k_ablation.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Minimal .env loader
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip())

from stem_dt import (  # noqa: E402
    benchmark, data, markers, pathway_activity, scores,
)

ICUFN = os.environ.get("ICUFN_REPO_PATH") or str(ROOT.parent / "stem-cell-digital-twin")
NOVAK = os.environ["NOVAK2022_PROTEINGROUPS_PATH"]
JERBER_H5 = Path(ICUFN) / "data" / "raw" / "dopaminergic_all_timepoints.h5"

OUT = ROOT / "artifacts"


def compute_per_stage_k_from_jerber() -> dict[str, scores.StagePrScore]:
    """Load Jerber transcripts → per-cell kinase activity → per-stage K_t.

    Uses the gene-set z-score fallback in `pathway_activity` (no decoupler
    dependency). Memory-efficient: loads AnnData backed='r', subsets to
    just kinase-target genes before densifying.
    """
    import scanpy as sc

    print(f"[K] loading Jerber AnnData (backed='r'): {JERBER_H5}")
    adata = sc.read_h5ad(str(JERBER_H5), backed="r")
    print(f"[K] full shape: {adata.shape}")

    # All kinase-target genes from the geneset fallback
    kinase_genes = sorted(set(
        g for gs in pathway_activity.KINASE_TARGET_GENESETS.values() for g in gs
    ))
    present = [g for g in kinase_genes if g in adata.var_names]
    missing = [g for g in kinase_genes if g not in adata.var_names]
    print(f"[K] kinase target genes: {len(present)}/{len(kinase_genes)} present "
          f"(missing: {missing})")

    # Subset to just kinase-target genes FIRST (one view), then materialize to
    # memory (small: ~205k cells × 13 genes), then apply obs filter + normalize.
    # Chained indexing on a backed AnnData isn't allowed.
    adata_sub = adata[:, present].to_memory()
    print(f"[K] subset to {adata_sub.shape[1]} kinase-target genes "
          f"(materialized: {adata_sub.shape})")

    if "treatment" in adata_sub.obs.columns:
        adata_sub = adata_sub[adata_sub.obs["treatment"] == "NONE"].copy()
    print(f"[K] after untreated filter: {adata_sub.shape}")

    sc.pp.normalize_total(adata_sub, target_sum=1e4)
    sc.pp.log1p(adata_sub)

    expr = adata_sub.X
    if hasattr(expr, "toarray"):
        expr = expr.toarray()

    # K per cell = mean log-normalized expression across kinase-target genes
    k_per_cell = np.asarray(expr).mean(axis=1)
    print(f"[K] k_per_cell range: [{k_per_cell.min():.3f}, {k_per_cell.max():.3f}]")

    # Per-stage aggregation. Field name in Jerber is `time_point` per ICUFN.
    if "time_point" in adata_sub.obs.columns:
        stage_col = adata_sub.obs["time_point"].astype(str)
    elif "day" in adata_sub.obs.columns:
        stage_col = adata_sub.obs["day"].astype(str)
    else:
        raise RuntimeError(
            f"Cannot find stage column in Jerber obs. "
            f"Available columns: {list(adata_sub.obs.columns)}"
        )

    out: dict[str, scores.StagePrScore] = {}
    for stage in ("D11", "D30", "D52"):
        mask = stage_col == stage
        n = int(mask.sum())
        if n == 0:
            out[stage] = scores.StagePrScore(
                stage=stage, mean=float("nan"), std=float("nan"),
                n_replicates=0,
                markers_found=tuple(present), markers_missing=tuple(missing),
            )
            continue
        vals = k_per_cell[mask.values]
        out[stage] = scores.StagePrScore(
            stage=stage,
            mean=float(vals.mean()),
            std=float(vals.std(ddof=1)),
            n_replicates=n,
            markers_found=tuple(present),
            markers_missing=tuple(missing),
        )
        print(f"[K] {stage}: mean={vals.mean():.3f}  std={vals.std():.3f}  "
              f"n_cells={n}")
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # Build the existing 3D state
    print("[ablation] building 3D state ...")
    baseline = data.load_icufn_baseline(ICUFN)
    tc = data.load_proteomic_timecourse(NOVAK, source="Novak2022")
    pr_per_stage = scores.per_stage_pr_score(tc, list(markers.G_Pr))
    state_3d = scores.assemble_3d_state(baseline.state, pr_per_stage, seed=42)

    # Compute K from Jerber transcripts and append as 4th feature
    k_per_stage = compute_per_stage_k_from_jerber()
    state_4d = scores.append_stage_feature(
        state_3d, k_per_stage, seed=43, floor_at=(),
    )
    print(f"\n[ablation] state_4d shape: {state_4d.shape}, "
          f"K range pre-norm [{state_4d[:,:,3].min():.3f}, "
          f"{state_4d[:,:,3].max():.3f}]")

    # Leakage-safe min-max on Pr AND K columns using train indices
    train_idx, val_idx, test_idx = scores.split_trajectories(
        n=200, splits=(140, 30, 30), seed=42,
    )
    for col in (2, 3):  # Pr, K
        bounds = scores.fit_minmax(state_4d[train_idx, :, col:col + 1])
        state_4d[:, :, col:col + 1] = bounds.apply(state_4d[:, :, col:col + 1])

    print(f"[ablation] post-normalize ranges: "
          f"P=[{state_4d[:,:,0].min():.2f},{state_4d[:,:,0].max():.2f}] "
          f"D=[{state_4d[:,:,1].min():.2f},{state_4d[:,:,1].max():.2f}] "
          f"Pr=[{state_4d[:,:,2].min():.2f},{state_4d[:,:,2].max():.2f}] "
          f"K=[{state_4d[:,:,3].min():.2f},{state_4d[:,:,3].max():.2f}]")

    # Run benchmark with 4D condition (the runner already supports 4D when
    # state.shape[2] >= 4)
    print("\n[ablation] Task A — stage classification ...")
    task_a = benchmark.run_stage_classification(
        state_4d, list(baseline.stage_order), train_idx, val_idx, test_idx,
        feature_labels=("P", "D", "Pr", "K"), seed=42, run_permutation_null=True,
    )
    for r in task_a:
        print(f"  {r.model_name:<20} {r.condition:<18} "
              f"macro_F1={r.macro_f1:.3f}  bal_acc={r.balanced_accuracy:.3f}")

    print("\n[ablation] Task B — trajectory prediction ...")
    task_b, _ = benchmark.run_trajectory_prediction(
        state_4d, train_idx, val_idx, test_idx,
        feature_labels=("P", "D", "Pr", "K"), seed=42, run_permutation_null=True,
    )
    for r in task_b:
        per_dim = "  ".join(f"{k}={v:.3f}" for k, v in r.mae_per_dim.items())
        print(f"  {r.model_name:<20} {r.condition:<18} "
              f"MAE={r.mae_overall:.4f}  ({per_dim})")

    # Save ablation metrics
    out_path = OUT / "k_ablation_metrics.json"
    out_path.write_text(json.dumps({
        "k_per_stage": {
            s: {
                "mean": k_per_stage[s].mean,
                "std": k_per_stage[s].std,
                "n_cells": k_per_stage[s].n_replicates,
                "markers_found": list(k_per_stage[s].markers_found),
                "markers_missing": list(k_per_stage[s].markers_missing),
            } for s in k_per_stage
        },
        "task_a": [asdict(r) for r in task_a],
        "task_b": [asdict(r) for r in task_b],
    }, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x),
        encoding="utf-8")
    print(f"\n[ablation] metrics saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
