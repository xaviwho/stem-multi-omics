"""Deterministic-Pr ablation — does per-cell Gaussian sampling add anything?

Reviewer probe: if Pr operates purely at stage level (within-stage permutation
preserves performance, even improving RF macro F1 to 1.000), why sample
per-cell at all? Why not assign the stage mean directly to every cell?

This script runs the same Tasks A and B benchmarks with two Pr-assembly modes:
  1. **Stochastic** (default): per-cell Pr ~ N(stage_mean, stage_std)
  2. **Deterministic**:        per-cell Pr = stage_mean (no per-cell variance)

If results are essentially identical, the per-cell sampling is decorative
and the methodology should disclose that the Pr signal lives entirely at
stage level. If deterministic is materially worse, the per-cell variance
adds something the stochastic-Pr captures (worth investigating).
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

for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip())

from stem_dt import benchmark, data, markers, scores  # noqa: E402

ICUFN = os.environ.get("ICUFN_REPO_PATH") or str(ROOT.parent / "stem-cell-digital-twin")
NOVAK = os.environ["NOVAK2022_PROTEINGROUPS_PATH"]
OUT = ROOT / "artifacts"


def run_one_mode(*, deterministic: bool) -> tuple[list, list]:
    baseline = data.load_icufn_baseline(ICUFN)
    tc = data.load_proteomic_timecourse(NOVAK, source="Novak2022")
    pr = scores.per_stage_pr_score(tc, list(markers.G_Pr))
    state = scores.assemble_3d_state(
        baseline.state, pr, seed=42, deterministic=deterministic,
    )

    train_idx, val_idx, test_idx = scores.split_trajectories(
        n=200, splits=(140, 30, 30), seed=42,
    )

    # Same leakage-safe min-max on the Pr column as run_benchmark.py
    pr_train = state[train_idx, :, 2:3]
    pr_bounds = scores.fit_minmax(pr_train)
    state[:, :, 2:3] = pr_bounds.apply(state[:, :, 2:3])

    print(f"  Pr per-stage variance after assembly:")
    for t_idx, stage in enumerate(baseline.stage_order):
        col = state[:, t_idx, 2]
        print(f"    {stage}: mean={col.mean():.4f}  std={col.std():.5f}")

    task_a = benchmark.run_stage_classification(
        state, list(baseline.stage_order), train_idx, val_idx, test_idx,
        feature_labels=("P", "D", "Pr"), seed=42, run_permutation_null=True,
    )
    task_b, _ = benchmark.run_trajectory_prediction(
        state, train_idx, val_idx, test_idx,
        feature_labels=("P", "D", "Pr"), seed=42, run_permutation_null=True,
    )
    return task_a, task_b


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("MODE 1: STOCHASTIC Pr (per-cell ~ N(stage_mean, stage_std))")
    print("=" * 64)
    task_a_stoch, task_b_stoch = run_one_mode(deterministic=False)

    print("\n" + "=" * 64)
    print("MODE 2: DETERMINISTIC Pr (per-cell = stage_mean exactly)")
    print("=" * 64)
    task_a_det, task_b_det = run_one_mode(deterministic=True)

    # Side-by-side comparison
    print("\n" + "=" * 64)
    print("COMPARISON (3D condition only)")
    print("=" * 64)
    print(f"\n{'Task A — macro F1':<30}{'Stochastic':>14}{'Deterministic':>16}{'delta':>10}")
    print("-" * 70)
    for ts, td in zip(task_a_stoch, task_a_det):
        if ts.condition == "3D" and td.condition == "3D":
            d = td.macro_f1 - ts.macro_f1
            sign = "+" if d >= 0 else ""
            print(f"  {ts.model_name:<28}{ts.macro_f1:>14.4f}"
                  f"{td.macro_f1:>16.4f}{sign}{d:>9.4f}")

    print(f"\n{'Task B — overall MAE':<30}{'Stochastic':>14}{'Deterministic':>16}{'delta':>10}")
    print("-" * 70)
    for ts, td in zip(task_b_stoch, task_b_det):
        if ts.condition == "3D" and td.condition == "3D":
            d = td.mae_overall - ts.mae_overall
            sign = "+" if d >= 0 else ""
            print(f"  {ts.model_name:<28}{ts.mae_overall:>14.4f}"
                  f"{td.mae_overall:>16.4f}{sign}{d:>9.4f}")

    # Save full result tables
    out_path = OUT / "pr_ablation_metrics.json"
    out_path.write_text(json.dumps({
        "stochastic": {
            "task_a": [asdict(r) for r in task_a_stoch],
            "task_b": [asdict(r) for r in task_b_stoch],
        },
        "deterministic": {
            "task_a": [asdict(r) for r in task_a_det],
            "task_b": [asdict(r) for r in task_b_det],
        },
    }, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x),
        encoding="utf-8")
    print(f"\nFull metrics saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
