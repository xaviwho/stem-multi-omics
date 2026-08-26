"""Verify the ICUFN-bridge by reproducing the published 2D state statistics.

Loads the saved trajectory tensor + normalization bounds from the ICUFN paper
repo, applies the bounds + clip, regenerates the seed=42 split, and prints
per-stage population medians + train-split MAE bounds for comparison against
the published Kanu et al. 2026 values:

  D11: P_med = 0.155   D_med = 0.026
  D30: P_med = 0.000   D_med = 0.112
  D52: P_med = 0.042   D_med = 0.075

If these match, the bridge is sound and we can extend the state with Pr / K
without re-running the upstream Jerber preprocessing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stem_dt import data, scores  # noqa: E402

# Resolve ICUFN repo path: env var first, then default sibling directory.
ICUFN = os.environ.get("ICUFN_REPO_PATH") or str(ROOT.parent / "stem-cell-digital-twin")


def main() -> int:
    print(f"[bridge] loading ICUFN baseline from: {ICUFN}")
    baseline = data.load_icufn_baseline(ICUFN)
    state = baseline.state
    print(f"[bridge] state.shape={state.shape}, "
          f"min={state.min():.4f}, max={state.max():.4f}")
    print(f"[bridge] stage order: {baseline.stage_order}")
    print(f"[bridge] bounds (metadata, already applied): {baseline.bounds}")

    print("\n=== per-stage population medians on 200 sampled trajectories ===")
    print(f"{'stage':<6}{'P_med':>10}{'D_med':>10}{'P_mean':>10}{'D_mean':>10}")
    for t_idx, name in enumerate(baseline.stage_order):
        s = state[:, t_idx, :]
        p_med, d_med = float(np.median(s[:, 0])), float(np.median(s[:, 1]))
        p_mean, d_mean = float(s[:, 0].mean()), float(s[:, 1].mean())
        print(f"{name:<6}{p_med:>10.4f}{d_med:>10.4f}{p_mean:>10.4f}{d_mean:>10.4f}")

    # Published full-population medians (from data/processed/dopaminergic_population_dynamics.csv).
    # Matching across only 200 sampled trajectories has higher sampling variance
    # than the 50k+ population statistic — accept up to 0.05 absolute drift.
    pop_published = {
        "D11": (0.155, 0.026),
        "D30": (0.000, 0.112),
        "D52": (0.042, 0.075),
    }
    print("\n=== sample-vs-population check (tolerance 0.05) ===")
    all_ok = True
    for t_idx, name in enumerate(baseline.stage_order):
        s = state[:, t_idx, :]
        p_med, d_med = float(np.median(s[:, 0])), float(np.median(s[:, 1]))
        ep, ed = pop_published[name]
        ok = abs(p_med - ep) < 0.05 and abs(d_med - ed) < 0.05
        all_ok &= ok
        flag = "OK" if ok else "DRIFT"
        print(f"{name:<6} pop P={ep:.3f} D={ed:.3f}  "
              f"sample P={p_med:.3f} D={d_med:.3f}  [{flag}]")

    print("\n=== monotonicity sanity (ICUFN paper claims ~78% / ~85%) ===")
    P_per_traj = state[:, :, 0]
    D_per_traj = state[:, :, 1]
    pct_p_decline = float(np.mean(P_per_traj[:, 0] > P_per_traj[:, 2]) * 100)
    pct_d_rise = float(np.mean(D_per_traj[:, 0] < D_per_traj[:, 2]) * 100)
    print(f"P(D11) > P(D52): {pct_p_decline:.1f}% of trajectories")
    print(f"D(D11) < D(D52): {pct_d_rise:.1f}% of trajectories")

    print("\n=== seed=42 split reproduction ===")
    tr, va, te = scores.split_trajectories(n=200, splits=(140, 30, 30), seed=42)
    print(f"train n={len(tr)}, val n={len(va)}, test n={len(te)}")
    print(f"first 5 train idx: {tr[:5].tolist()}")
    print(f"first 5 test  idx: {te[:5].tolist()}")
    assert sorted(np.concatenate([tr, va, te]).tolist()) == list(range(200))
    print("split partitions [0..199] cleanly")

    print("\n=== bridge verification " +
          ("PASSED" if all_ok else "FAILED -- investigate before extending") +
          " ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
