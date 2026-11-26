"""Confidence intervals for the Task B 2D -> 3D MAE change (CMPB revision, R1-3.2).

Reads the per-seed values already written by ``scripts/run_multi_seed.py``
(``artifacts/multi_seed_metrics.json``); no model is retrained.

For each model family it reports, across the 20 seeds:

* overall MAE change  (3D - 2D)            -> the headline -28..-31 % figure
* P/D-only MAE change (3D MAE_PD - 2D MAE) -> shows the original axes are unchanged

each with
  - a percentile bootstrap 95 % CI (10,000 resamples of the seeds, fixed RNG), and
  - a paired t 95 % CI (for reference; seeds are repeated runs, not
    biological replicates, so the CI describes run-to-run variability of
    the pipeline, not population-level uncertainty).

Relative change is computed as ratio-of-means, 100 * mean(3D - 2D) / mean(2D),
re-evaluated inside every bootstrap resample, which matches the Delta % row of
the manuscript's Task B table.

Outputs
-------
artifacts/revision/mae_ci.json
artifacts/revision/mae_ci.tex   (table rows, ready to paste)

Usage
-----
    python scripts/revision_mae_ci.py
    python scripts/revision_mae_ci.py --metrics path/to/multi_seed_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS = ROOT / "artifacts" / "multi_seed_metrics.json"
OUT = ROOT / "artifacts" / "revision"

MODEL_ORDER = ("Ridge", "RandomForest", "Transformer")


def _bootstrap(v2: np.ndarray, v3: np.ndarray, n_boot: int, rng: np.random.Generator):
    """Percentile bootstrap over seeds (paired: the same seed indices for 2D and 3D)."""
    n = v2.size
    idx = rng.integers(0, n, size=(n_boot, n))
    b2, b3 = v2[idx], v3[idx]
    d_mean = (b3 - b2).mean(axis=1)
    rel = 100.0 * d_mean / b2.mean(axis=1)
    return (np.percentile(d_mean, [2.5, 97.5]).tolist(),
            np.percentile(rel, [2.5, 97.5]).tolist())


def _t_ci(diffs: np.ndarray) -> list[float]:
    n = diffs.size
    se = diffs.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    m = diffs.mean()
    return [float(m - h), float(m + h)]


def _summarise(v2: np.ndarray, v3: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    diffs = v3 - v2
    boot_abs, boot_rel = _bootstrap(v2, v3, n_boot, rng)
    return {
        "n_seeds": int(diffs.size),
        "mae_2d_mean": float(v2.mean()),
        "mae_3d_mean": float(v3.mean()),
        "delta_mean": float(diffs.mean()),
        "delta_ci95_bootstrap": boot_abs,
        "delta_ci95_t": _t_ci(diffs),
        "rel_pct": float(100.0 * diffs.mean() / v2.mean()),
        "rel_pct_ci95_bootstrap": boot_rel,
        "n_seeds_3d_better": int((diffs < 0).sum()),
    }


def _per_seed(entry: dict, key: str) -> np.ndarray:
    vals = entry.get(key)
    if vals is None:
        raise KeyError(key)
    return np.asarray(vals, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed.")
    args = ap.parse_args()

    if not args.metrics.exists():
        raise SystemExit(
            f"{args.metrics} not found. Copy artifacts/ from the machine that ran "
            "scripts/run_multi_seed.py, or rerun it (needs the Jerber/ICUFN and "
            "Novak proteinGroups inputs)."
        )
    ms = json.loads(args.metrics.read_text(encoding="utf-8"))
    tb = ms["task_b_summary"]

    results: dict[str, dict] = {}
    for model in [m for m in MODEL_ORDER if m in tb] + [m for m in tb if m not in MODEL_ORDER]:
        by_cond = tb[model]
        if "2D" not in by_cond or "3D" not in by_cond:
            continue
        v2 = _per_seed(by_cond["2D"], "mae_overall_per_seed")
        v3 = _per_seed(by_cond["3D"], "mae_overall_per_seed")
        if v2.size != v3.size:
            raise ValueError(f"{model}: 2D and 3D seed counts differ ({v2.size} vs {v3.size})")
        res = {"overall": _summarise(v2, v3, args.n_boot, args.seed)}

        # P/D-only: 2D overall MAE already averages P and D only; prefer the
        # explicit PD column when present.
        try:
            v2_pd = _per_seed(by_cond["2D"], "mae_PD_per_seed")
        except KeyError:
            v2_pd = v2
        try:
            v3_pd = _per_seed(by_cond["3D"], "mae_PD_per_seed")
            res["PD_only"] = _summarise(v2_pd, v3_pd, args.n_boot, args.seed)
        except KeyError:
            pass
        results[model] = res

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mae_ci.json").write_text(json.dumps({
        "source": str(args.metrics),
        "seeds": ms.get("seeds"),
        "n_boot": args.n_boot,
        "bootstrap_rng_seed": args.seed,
        "results": results,
    }, indent=2), encoding="utf-8")

    # LaTeX rows for the Task B table (one column per model).
    models = list(results)
    def fmt_ci(lo_hi, scale=1.0, nd=4):
        lo, hi = (x * scale for x in lo_hi)
        return f"[{lo:.{nd}f}, {hi:.{nd}f}]"
    rows = [
        r"$\Delta$ MAE 95\% CI (bootstrap) & " + " & ".join(
            fmt_ci(results[m]["overall"]["delta_ci95_bootstrap"]) for m in models) + r" \\",
        r"$\Delta$\% 95\% CI (bootstrap) & " + " & ".join(
            fmt_ci(results[m]["overall"]["rel_pct_ci95_bootstrap"], nd=1) for m in models) + r" \\",
    ]
    if all("PD_only" in results[m] for m in models):
        rows.append(r"$\Delta$ MAE$_{PD}$ 95\% CI (bootstrap) & " + " & ".join(
            fmt_ci(results[m]["PD_only"]["delta_ci95_bootstrap"]) for m in models) + r" \\")
    (OUT / "mae_ci.tex").write_text(
        "% columns: " + ", ".join(models) + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

    # ASCII-only console summary (Windows cp1252-safe).
    print(f"[mae-ci] {args.metrics.name}: {len(ms.get('seeds', []))} seeds, {args.n_boot} bootstrap resamples")
    for m, r in results.items():
        o = r["overall"]
        lo, hi = o["delta_ci95_bootstrap"]
        plo, phi = o["rel_pct_ci95_bootstrap"]
        print(f"  {m:<13} overall dMAE {o['delta_mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"rel {o['rel_pct']:+.1f}% [{plo:+.1f}, {phi:+.1f}]  "
              f"3D better in {o['n_seeds_3d_better']}/{o['n_seeds']} seeds")
        if "PD_only" in r:
            p = r["PD_only"]
            lo, hi = p["delta_ci95_bootstrap"]
            print(f"  {'':<13} P/D-only dMAE {p['delta_mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print(f"[mae-ci] wrote {OUT / 'mae_ci.json'} and {OUT / 'mae_ci.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
