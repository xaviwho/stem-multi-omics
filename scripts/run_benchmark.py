"""Run Stage 5 benchmarks: Tasks A & B at 2D / 3D / 3D-permuted.

Reuses the ICUFN seed=42 split; adds Pr from Novak 2022 as the third dim
(currently degenerate until full proteinGroups arrives, see project memory).

Pipeline:
  1. Load ICUFN (P, D), assemble 3D state with Novak Pr.
  2. Reproduce 140/30/30 split (seed=42).
  3. Task A (stage classification): LR / RF / Transformer at 2D, 3D, 3D-perm.
  4. Task B (trajectory prediction [s_11, s_30] -> s_52): Ridge / RF /
     Transformer DT at 2D, 3D, 3D-perm.
  5. Persist Transformer predictions for the hero figure.
  6. Save aggregate metrics to artifacts/benchmark_metrics.json + merge into
     validation_metrics.json so the figure runner picks up everything.

Usage
-----
    python scripts/run_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip())

from stem_dt import benchmark, data, figures, markers, provenance, scores  # noqa: E402

ICUFN = os.environ.get("ICUFN_REPO_PATH") or str(ROOT.parent / "stem-cell-digital-twin")
NOVAK = os.environ["NOVAK2022_PROTEINGROUPS_PATH"]
PSP = os.environ.get("PSP_KINASE_SUBSTRATE_PATH") or None

OUT = ROOT / "artifacts"
FIG = OUT / "figures"
PROV = OUT / "provenance"


def run_benchmark_once(
    seed: int = 42,
    *,
    run_pr_only: bool = False,
    excluded_proteomic_samples: tuple[str, ...] = (),
    verbose: bool = True,
) -> dict:
    """Run Tasks A+B at one seed and return the raw metrics dict.

    Pulled out as a function so `run_multi_seed.py` and `run_loro.py` can
    drive it without forking the orchestration logic. Returns the same
    dict that the CLI writes to `benchmark_metrics.json`.

    Parameters
    ----------
    seed
        Controls trajectory assembly, train/val/test split, Pr sampling,
        permutation-null shuffles, and every model's random_state. Multi-
        seed CIs (§5.3) use this knob.
    excluded_proteomic_samples
        Sample IDs to exclude when computing per-stage `(μ_s, σ_s)` — the
        LORO probe (§5.1) drives this with the 7 Dittmar CTRL sample IDs.
    """
    baseline = data.load_icufn_baseline(ICUFN)
    tc = data.load_proteomic_timecourse(NOVAK, source="Novak2022")
    pr = scores.per_stage_pr_score(
        tc, list(markers.G_Pr),
        exclude_samples=tuple(excluded_proteomic_samples),
    )
    state = scores.assemble_3d_state(baseline.state, pr, seed=seed)
    stage_order = list(baseline.stage_order)
    feat_labels = ("P", "D", "Pr")
    if verbose:
        print(f"[bench] seed={seed} state shape: {state.shape}, "
              f"Pr raw range [{state[:,:,2].min():.2f}, {state[:,:,2].max():.2f}]")

    train_idx, val_idx, test_idx = scores.split_trajectories(
        n=200, splits=(140, 30, 30), seed=seed,
    )

    # Leakage-safe min-max on Pr (training split only).
    pr_train = state[train_idx, :, 2:3]
    pr_bounds = scores.fit_minmax(pr_train)
    state[:, :, 2:3] = pr_bounds.apply(state[:, :, 2:3])

    task_a = benchmark.run_stage_classification(
        state, stage_order, train_idx, val_idx, test_idx,
        feature_labels=feat_labels, seed=seed,
        run_permutation_null=True, run_pr_only=run_pr_only,
    )
    if verbose:
        for r in task_a:
            print(f"  {r.model_name:<20} {r.condition:<16} "
                  f"macro_F1={r.macro_f1:.3f}  bal_acc={r.balanced_accuracy:.3f}")

    task_b, predictions_by_cond = benchmark.run_trajectory_prediction(
        state, train_idx, val_idx, test_idx,
        feature_labels=feat_labels, seed=seed, run_permutation_null=True,
    )
    if verbose:
        for r in task_b:
            per_dim = "  ".join(f"{k}={v:.3f}" for k, v in r.mae_per_dim.items())
            print(f"  {r.model_name:<20} {r.condition:<16} "
                  f"MAE={r.mae_overall:.4f}  ({per_dim})")

    confusion_per_condition: dict[str, list[list[int]]] = {}
    for r in task_a:
        if r.model_name == "Transformer":
            confusion_per_condition[r.condition] = r.confusion

    mae_per_condition: dict[str, dict[str, float]] = {}
    for r in task_b:
        if r.model_name == "Transformer":
            mae_per_condition[r.condition] = r.mae_per_dim

    return {
        "seed": seed,
        "excluded_proteomic_samples": list(excluded_proteomic_samples),
        "task_a": [asdict(r) for r in task_a],
        "task_b": [asdict(r) for r in task_b],
        "task_a_confusion": confusion_per_condition,
        "task_b_mae": mae_per_condition,
        "predictions_3d": predictions_by_cond.get("3D"),  # np.ndarray
        "state": state,
        "test_idx": test_idx,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for trajectory assembly, split, Pr sampling, "
                             "and all model random_states.")
    parser.add_argument("--run-pr-only", action="store_true",
                        help="Also evaluate Task A on the Pr column alone.")
    parser.add_argument("--no-provenance", action="store_true",
                        help="Skip the provenance manifest step (useful inside "
                             "multi-seed sweeps where hashes are identical).")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if not args.no_provenance:
        print("[bench] computing provenance records for the Paper 1 inputs ...")
        records = provenance.build_paper1_ingestion_records(
            icufn_repo_path=ICUFN,
            novak_proteingroups_path=NOVAK,
            psp_kinase_substrate_path=PSP,
        )
        for r in records:
            print(f"  {r.source_name:<22} hash={r.raw_data_hash[:16]}…  "
                  f"version={r.source_version}")
        provenance.register_paper1_provenance(records, PROV)

    print(f"\n[bench] Task A + B at seed={args.seed} ...")
    result = run_benchmark_once(
        seed=args.seed,
        run_pr_only=args.run_pr_only,
        verbose=True,
    )

    bench_path = OUT / "benchmark_metrics.json"
    bench_path.write_text(json.dumps({
        "seed": result["seed"],
        "task_a": result["task_a"],
        "task_b": result["task_b"],
        "task_a_confusion": result["task_a_confusion"],
        "task_b_mae": result["task_b_mae"],
    }, indent=2), encoding="utf-8")
    print(f"\n[bench] benchmark metrics: {bench_path}")

    val_path = OUT / "validation_metrics.json"
    if val_path.exists():
        merged = json.loads(val_path.read_text(encoding="utf-8"))
    else:
        merged = {}
    merged["task_a_confusion"] = result["task_a_confusion"]
    merged["task_b_mae"] = result["task_b_mae"]
    val_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"[bench] merged into: {val_path}")

    pred_d52 = result["predictions_3d"]
    if pred_d52 is not None:
        np.savez(
            OUT / "state_for_figures.npz",
            state=result["state"], test_idx=result["test_idx"],
            predicted_d52=pred_d52,
        )
        print(f"[bench] state .npz updated with Transformer 3D predictions")

    print("\n[bench] rendering figures with benchmark output ...")
    written = figures.render_all_from_artifacts(
        state_path=OUT / "state_for_figures.npz",
        metrics_path=val_path,
        out_dir=FIG,
    )
    for p in written:
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
