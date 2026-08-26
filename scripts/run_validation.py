"""Run Stage 4 validation against the assembled 3D state tensor.

Pipeline:
  1. Load ICUFN baseline + Novak Pr → assemble (200, 3, 3) tensor (degenerate
     Pr until the full Novak proteinGroups arrives).
  2. Compute per-stage cross-modal Spearman correlations + bootstrap CIs.
  3. Compute trajectory coherence (3D vs 2D-baseline-only, for the comparison
     histogram in figures.coherence_histogram).
  4. Fit MOFA+ for modality-resolved variance decomposition (skipped with a
     warning if mofapy2 is not installed — the rest still completes).
  5. Save everything to artifacts/validation_metrics.json + a state .npz.
  6. Render every figure into artifacts/figures/.

Usage
-----
    python scripts/run_validation.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# .env loader (no dotenv dep)
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip())

from stem_dt import data, figures, markers, scores, validation  # noqa: E402

ICUFN = os.environ.get("ICUFN_REPO_PATH") or str(ROOT.parent / "stem-cell-digital-twin")
NOVAK = os.environ["NOVAK2022_PROTEINGROUPS_PATH"]

OUT_DIR = ROOT / "artifacts"
FIG_DIR = OUT_DIR / "figures"


def _to_jsonable(obj):
    """Recursively convert numpy types to JSON-friendly Python primitives."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics: dict = {}

    print("[validation] loading ICUFN baseline ...")
    baseline = data.load_icufn_baseline(ICUFN)
    state_2d = baseline.state                                      # (200, 3, 2)
    print(f"  state_2d shape: {state_2d.shape}")

    print("[validation] loading Novak proteomics + computing per-stage Pr ...")
    tc = data.load_proteomic_timecourse(NOVAK, source="Novak2022")
    pr_per_stage = scores.per_stage_pr_score(tc, list(markers.G_Pr))
    metrics["pr_per_stage"] = {
        s: {
            "mean": pr_per_stage[s].mean,
            "std": pr_per_stage[s].std,
            "n_replicates": pr_per_stage[s].n_replicates,
            "markers_found": list(pr_per_stage[s].markers_found),
            "markers_missing": list(pr_per_stage[s].markers_missing),
        }
        for s in pr_per_stage
    }

    print("[validation] assembling 3D state tensor ...")
    state_3d = scores.assemble_3d_state(state_2d, pr_per_stage, seed=42)
    print(f"  state_3d shape: {state_3d.shape}")

    # --- 4.1 Cross-modal correlations
    print("[validation] cross-modal Spearman correlations + bootstrap CIs ...")
    corr = validation.cross_modal_correlations(
        state_3d, list(baseline.stage_order), method="spearman",
        n_bootstrap=1000, seed=42,
    )
    metrics["correlations"] = _to_jsonable(corr)

    # --- 4.3 Trajectory coherence: 2D (baseline) vs 3D (stochastic Pr)
    # vs 3D (deterministic Pr). Deterministic gives exactly +1 by construction
    # (Pr is monotone across stages and identical for all trajectories), so it
    # is reported only as the structural-sanity-check anchor — the *stochastic*
    # row is the meaningful one with biological per-cell variance.
    print("[validation] trajectory coherence ...")
    coh_2d = validation.trajectory_coherence(state_2d)
    coh_3d = validation.trajectory_coherence(state_3d)
    state_3d_det = scores.assemble_3d_state(
        state_2d, pr_per_stage, seed=42, deterministic=True,
    )
    coh_3d_det = validation.trajectory_coherence(state_3d_det)
    metrics["coherence_2d"] = _to_jsonable(coh_2d)
    metrics["coherence_3d"] = _to_jsonable(coh_3d)
    metrics["coherence_3d_deterministic"] = _to_jsonable(coh_3d_det)
    print(f"  coherence_2d                 mean={coh_2d.mean():+.4f} ± {coh_2d.std():.4f}")
    print(f"  coherence_3d (stochastic Pr) mean={coh_3d.mean():+.4f} ± {coh_3d.std():.4f}")
    print(f"  coherence_3d (determ.    Pr) mean={coh_3d_det.mean():+.4f} ± {coh_3d_det.std():.4f}")

    # --- 4.2 MOFA+ variance decomposition
    print("[validation] MOFA+ variance decomposition ...")
    try:
        mofa_out = validation.fit_mofa(
            state_3d, list(baseline.stage_order),
            feature_labels=("P", "D", "Pr"), n_factors=5, seed=42,
        )
        metrics["mofa_variance"] = _to_jsonable(mofa_out["variance_explained"])
        # Persist factor scores separately (large; not paper-table material)
        np.savez(OUT_DIR / "mofa_factors.npz", factors=mofa_out["factor_scores"])
        print(f"  MOFA+ done; variance keys: {list(metrics['mofa_variance'].keys())}")
    except RuntimeError as e:
        print(f"  [skip] MOFA+ unavailable: {e}")
        metrics["mofa_variance"] = None

    # --- save metrics + state for figure rendering
    metrics_path = OUT_DIR / "validation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[validation] metrics saved: {metrics_path}")

    # Persist the assembled state tensor + (placeholder) predicted_d52 so the
    # hero figure can render even before the benchmark runs. The placeholder
    # is the observed D52 (perfect prediction) — figures show no residuals
    # until run_benchmark.py overwrites this with real model outputs.
    n_test = 30
    rng = np.random.default_rng(42)
    test_idx = rng.permutation(state_3d.shape[0])[:n_test]
    placeholder_pred = state_3d[test_idx, -1, :].copy()
    state_npz_path = OUT_DIR / "state_for_figures.npz"
    np.savez(
        state_npz_path,
        state=state_3d, test_idx=test_idx, predicted_d52=placeholder_pred,
    )
    print(f"[validation] state .npz saved: {state_npz_path}")

    # --- render figures
    print("[validation] rendering figures ...")
    written = figures.render_all_from_artifacts(
        state_path=state_npz_path,
        metrics_path=metrics_path,
        out_dir=FIG_DIR,
    )
    for p in written:
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
