"""Multi-seed CI sweep for the headline Task A + Task B numbers — Plan #2.

Re-runs the full Tasks A+B benchmark across 20 seeds (parameterizing
trajectory assembly, Pr sampling, train/val/test split, permutation-null
shuffles, and every model's random_state). Aggregates to mean ± std per
(model, condition) and runs a paired Wilcoxon signed-rank test across
seeds for the 2D→3D MAE delta, with Cohen's d_z effect size and
Holm-Bonferroni correction across the three Task B model families.

Output: `artifacts/multi_seed_metrics.json` consumed by `build_paper.py`
to replace single-seed point estimates in Tables 3–4 with mean ± std and
to report the Task B improvement as "−Δ% (Wilcoxon p=…, d_z=…)".

Usage
-----
    python scripts/run_multi_seed.py                  # 20 seeds (default)
    python scripts/run_multi_seed.py --n-seeds 5      # quick smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip())

from run_benchmark import run_benchmark_once  # noqa: E402

OUT = ROOT / "artifacts"


def _cohens_dz(diffs: np.ndarray) -> float:
    """Paired-test effect size: mean / std of the per-seed differences."""
    if diffs.size < 2:
        return float("nan")
    sd = float(diffs.std(ddof=1))
    return float(diffs.mean() / sd) if sd > 0 else float("inf")


def _holm_bonferroni(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down correction across a family of paired tests."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    n = len(items)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, p * (n - i))
        running_max = max(running_max, adj)
        adjusted[k] = running_max
    return adjusted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=42,
                        help="Seeds used: base, base+1, ..., base+n_seeds-1.")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.base_seed, args.base_seed + args.n_seeds))

    print(f"[multi-seed] running benchmark across {len(seeds)} seeds: "
          f"{seeds[0]}…{seeds[-1]}")
    per_seed: list[dict] = []
    for i, s in enumerate(seeds):
        print(f"\n[multi-seed] seed {i+1}/{len(seeds)}  ({s}) ...")
        res = run_benchmark_once(seed=s, run_pr_only=True, verbose=False)
        per_seed.append({
            "seed": s,
            "task_a": res["task_a"],
            "task_b": res["task_b"],
        })

    # ----- Aggregate Task A: macro_f1 by (model, condition) across seeds.
    def _gather_a(field: str) -> dict[tuple[str, str], list[float]]:
        agg: dict[tuple[str, str], list[float]] = {}
        for rec in per_seed:
            for r in rec["task_a"]:
                agg.setdefault((r["model_name"], r["condition"]), []).append(r[field])
        return agg

    def _gather_b(field: str) -> dict[tuple[str, str], list[float]]:
        agg: dict[tuple[str, str], list[float]] = {}
        for rec in per_seed:
            for r in rec["task_b"]:
                agg.setdefault((r["model_name"], r["condition"]), []).append(r[field])
        return agg

    a_f1 = _gather_a("macro_f1")
    b_mae = _gather_b("mae_overall")

    # ----- per-dimension MAE for the Task B P/D-only deflation guard.
    def _gather_b_per_dim() -> dict[tuple[str, str], dict[str, list[float]]]:
        agg: dict[tuple[str, str], dict[str, list[float]]] = {}
        for rec in per_seed:
            for r in rec["task_b"]:
                bucket = agg.setdefault((r["model_name"], r["condition"]), {})
                for dim, v in r["mae_per_dim"].items():
                    bucket.setdefault(dim, []).append(float(v))
        return agg
    b_per_dim = _gather_b_per_dim()

    def _summarize(vals: list[float]) -> dict[str, float]:
        a = np.array(vals, dtype=float)
        if a.size == 0:
            return {"n": 0, "mean": float("nan"), "std": float("nan"),
                    "min": float("nan"), "max": float("nan")}
        return {
            "n": int(a.size),
            "mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()),
            "max": float(a.max()),
        }

    task_a_summary: dict[str, dict[str, dict[str, float]]] = {}
    for (model, cond), vals in a_f1.items():
        task_a_summary.setdefault(model, {})[cond] = {
            "macro_f1_per_seed": vals,
            **{f"macro_f1_{k}": v for k, v in _summarize(vals).items()},
        }

    task_b_summary: dict[str, dict[str, dict[str, float]]] = {}
    for (model, cond), vals in b_mae.items():
        per_dim = b_per_dim.get((model, cond), {})
        # PD-mean = average over the two original dimensions (excludes Pr).
        if "P" in per_dim and "D" in per_dim:
            pd_pairs = list(zip(per_dim["P"], per_dim["D"]))
            pd_mean_per_seed = [(p + d) / 2 for p, d in pd_pairs]
        else:
            pd_mean_per_seed = []
        entry: dict[str, object] = {
            "mae_overall_per_seed": vals,
            **{f"mae_overall_{k}": v for k, v in _summarize(vals).items()},
        }
        for dim, dvals in per_dim.items():
            entry[f"mae_{dim}_per_seed"] = dvals
            for k, v in _summarize(dvals).items():
                entry[f"mae_{dim}_{k}"] = v
        if pd_mean_per_seed:
            entry["mae_PD_per_seed"] = pd_mean_per_seed
            for k, v in _summarize(pd_mean_per_seed).items():
                entry[f"mae_PD_{k}"] = v
        task_b_summary.setdefault(model, {})[cond] = entry

    # ----- Paired tests on Task B 2D vs 3D (the headline -30% claim).
    try:
        from scipy.stats import wilcoxon
        wilcoxon_available = True
    except ImportError:
        wilcoxon_available = False

    paired_b: dict[str, dict] = {}
    raw_pvalues: dict[str, float] = {}
    for model in task_b_summary:
        v2 = np.array(task_b_summary[model].get("2D", {}).get("mae_overall_per_seed", []), dtype=float)
        v3 = np.array(task_b_summary[model].get("3D", {}).get("mae_overall_per_seed", []), dtype=float)
        if v2.size == 0 or v3.size == 0 or v2.size != v3.size:
            continue
        diffs = v3 - v2                          # negative → 3D improves on 2D
        rel_pct = float(100.0 * diffs.mean() / v2.mean()) if v2.mean() != 0 else float("nan")
        dz = _cohens_dz(diffs)
        p = None
        if wilcoxon_available and diffs.size >= 5 and np.any(diffs != 0):
            try:
                stat, p_two = wilcoxon(v3, v2, zero_method="wilcox", alternative="less")
                p = float(p_two)
            except ValueError:
                p = None
        paired_b[model] = {
            "n_seeds": int(diffs.size),
            "delta_mae_per_seed": diffs.tolist(),
            "delta_mae_mean": float(diffs.mean()),
            "delta_mae_std":  float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0,
            "delta_mae_relative_pct": rel_pct,
            "cohens_dz": dz,
            "wilcoxon_p_one_sided_3d_lt_2d": p,
        }
        if p is not None:
            raw_pvalues[model] = p

    if raw_pvalues:
        for model, p_adj in _holm_bonferroni(raw_pvalues).items():
            paired_b[model]["wilcoxon_p_holm"] = p_adj

    # Write JSON FIRST so the artifact survives any stdout-encoding hiccup
    # on Windows cp1252 — earlier run crashed printing the Delta glyph and
    # took the JSON write down with it.
    out_path = OUT / "multi_seed_metrics.json"
    out_path.write_text(json.dumps({
        "seeds": seeds,
        "task_a_summary": task_a_summary,
        "task_b_summary": task_b_summary,
        "paired_b_2d_vs_3d": paired_b,
        "note": (
            f"Across {len(seeds)} seeds. Paired Wilcoxon signed-rank tests "
            "compare per-seed MAE under 2D vs 3D for each model family; "
            "Cohen's d_z is the paired effect size; Holm-Bonferroni "
            "correction is applied across the three Task B model families."
        ),
    }, indent=2), encoding="utf-8")
    print(f"[multi-seed] wrote {out_path}")

    # ASCII-safe summary (no Delta / non-Latin chars) so Windows cp1252 stdout
    # cannot break the run.
    summary_table: list[str] = []
    summary_table.append("\n[multi-seed] Task A 3D macro F1 (mean +/- std across seeds):")
    for model, by_cond in task_a_summary.items():
        s = by_cond.get("3D", {})
        summary_table.append(
            f"  {model:<20} {s.get('macro_f1_mean', float('nan')):.3f} "
            f"+/- {s.get('macro_f1_std', float('nan')):.3f}"
        )
    summary_table.append("\n[multi-seed] Task B 2D-to-3D MAE delta (Wilcoxon + Holm):")
    for model, t in paired_b.items():
        p_raw = t.get("wilcoxon_p_one_sided_3d_lt_2d")
        p_holm = t.get("wilcoxon_p_holm")
        summary_table.append(
            f"  {model:<20} dMAE={t['delta_mae_mean']:+.4f}  "
            f"({t['delta_mae_relative_pct']:+.0f}%)  "
            f"d_z={t['cohens_dz']:+.2f}  "
            f"p_raw={p_raw if p_raw is None else f'{p_raw:.4g}'}  "
            f"p_holm={p_holm if p_holm is None else f'{p_holm:.4g}'}"
        )
    print("\n".join(summary_table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
