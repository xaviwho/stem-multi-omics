"""Per-replicate proteomics view + sample-mapping table (CMPB revision).

Addresses
---------
R2-4.2  distribution plot of individual proteomic biological replicates per stage
R2-2.2  table linking sample labels to biological condition and model stage
R2-4.1  stage-correspondence summary (sources, stages, replicate counts)
R2-2.1  explicit missing-value accounting (LFQ = 0 -> "not detected")

Reads the canonical (Dittmar) proteinGroups file through the same loader the
benchmark uses (``data.load_proteomic_timecourse(..., "Novak2022")``), so the
numbers are identical to the ones that fed ``per_stage_pr_score``. No model is
retrained.

Outputs (artifacts/revision/)
-----------------------------
pr_per_replicate.csv        one row per CTRL replicate: per-marker log2(1+LFQ),
                            detected flags, and the replicate's Pr score
pr_stage_summary.csv        per-stage mean / SD / n, plus the (mu, sigma) the
                            trajectory generator actually samples from
sample_mapping.tex          LaTeX table: sample label -> condition -> source day
                            -> model stage, with the replicate's Pr
fig_pr_replicates.{png,pdf} strip plot, one panel per marker + Pr

With ``--to-paper`` the PNG is also copied to paper_dt1/images/.

Usage
-----
    python scripts/revision_proteomics_replicates.py \
        [--proteingroups $NOVAK2022_PROTEINGROUPS_PATH] [--to-paper]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stem_dt import data, markers  # noqa: E402

OUT = ROOT / "artifacts" / "revision"
PAPER_IMAGES = ROOT / "paper_dt1" / "images"

STAGES = ("D11", "D30", "D52")
SOURCE_DAY = {"D11": "iPSC", "D30": "D25", "D52": "D40"}
# The trajectory generator (scores.assemble_3d_state) floors these stages at a
# biological-zero mean with sigma/2; keep in sync with its `pr_floor_at` default.
PR_FLOOR_AT = ("D11",)

# Same stage colours as the manuscript's other figures (figures.STAGE_COLORS);
# stage identity is also carried by x-position and marker shape.
STAGE_COLORS = {"D11": "#4C72B0", "D30": "#DD8452", "D52": "#55A467"}
STAGE_MARKERS = {"D11": "o", "D30": "s", "D52": "^"}
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e6e6e6"


def _load(path: Path):
    tc = data.load_proteomic_timecourse(path, source="Novak2022")
    # Canonical CTRL labels only (exclude the interim Skupin## columns if the
    # file happens to carry them).
    keep = [s for s in tc.samples.index if not s.startswith("Skupin")]
    return tc.abundance[keep], tc.samples.loc[keep]


def per_replicate_table(abundance: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    gpr = list(markers.G_Pr)
    found = [m for m in gpr if m in abundance.index]
    missing = [m for m in gpr if m not in abundance.index]
    if missing:
        print(f"[replicates] G_Pr markers absent from file: {missing}")
    rows = []
    for sid, meta in samples.iterrows():
        rec = {
            "sample_label": sid,
            "condition": meta["condition"],
            "source_day": SOURCE_DAY.get(meta["stage"], ""),
            "model_stage": meta["stage"],
        }
        logs = []
        for m in found:
            lfq = float(abundance.loc[m, sid])
            lfq = 0.0 if not np.isfinite(lfq) else lfq
            v = float(np.log2(1.0 + lfq))
            rec[f"{m}_log2"] = v
            rec[f"{m}_detected"] = lfq > 0
            logs.append(v)
        # Identical to scores.per_stage_pr_score: mean log2(1+LFQ) over markers,
        # with LFQ = 0 (not quantified) entering as 0.
        rec["Pr"] = float(np.mean(logs)) if logs else float("nan")
        rec["n_markers_detected"] = int(sum(rec[f"{m}_detected"] for m in found))
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["model_stage"] = pd.Categorical(df["model_stage"], STAGES, ordered=True)
    return df.sort_values(["model_stage", "sample_label"]).reset_index(drop=True)


def stage_summary(rep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for st in STAGES:
        v = rep.loc[rep["model_stage"] == st, "Pr"].to_numpy(dtype=float)
        mean = float(v.mean()) if v.size else float("nan")
        sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
        if st in PR_FLOOR_AT:
            gen_mu, gen_sd = 0.0, max(sd / 2.0, 1e-6)
        else:
            gen_mu, gen_sd = mean, max(sd, 1e-6)
        rows.append({
            "model_stage": st, "source_day": SOURCE_DAY[st], "n_replicates": int(v.size),
            "Pr_mean": mean, "Pr_sd": sd, "Pr_min": float(v.min()) if v.size else np.nan,
            "Pr_max": float(v.max()) if v.size else np.nan,
            "generator_mu": gen_mu, "generator_sigma": gen_sd,
            "generator_floored": st in PR_FLOOR_AT,
        })
    return pd.DataFrame(rows)


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def mapping_tex(rep: pd.DataFrame, summ: pd.DataFrame) -> str:
    found = [m for m in markers.G_Pr if f"{m}_log2" in rep.columns]
    lines = [
        r"% Auto-generated by scripts/revision_proteomics_replicates.py",
        r"\begin{tabular}{llllc" + "c" * len(found) + "c}",
        r"\toprule",
        "Sample label & Condition & Source day & Model stage & $n$ & "
        + " & ".join(found) + r" & $Pr$ \\",
        r"\midrule",
    ]
    n_by_stage = dict(zip(summ["model_stage"], summ["n_replicates"]))
    for _, r in rep.iterrows():
        cells = []
        for m in found:
            cells.append(f"{r[f'{m}_log2']:.2f}" if r[f"{m}_detected"] else "n/d")
        lines.append(
            f"\\texttt{{{_tex_escape(r['sample_label'])}}} & {r['condition']} & "
            f"{r['source_day']} & {r['model_stage']} & {n_by_stage[r['model_stage']]} & "
            + " & ".join(cells) + f" & {r['Pr']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def plot(rep: pd.DataFrame, summ: pd.DataFrame, out_base: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    found = [m for m in markers.G_Pr if f"{m}_log2" in rep.columns]
    panels = [(m, f"{m}_log2", f"{m}_detected") for m in found] + [("$Pr$ (mean of markers)", "Pr", None)]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10.5,
        "axes.titlesize": 11, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, len(panels), figsize=(2.15 * len(panels), 3.1), sharey=True)
    axes = np.atleast_1d(axes)
    x_of = {st: i for i, st in enumerate(STAGES)}
    rng = np.random.default_rng(0)  # deterministic jitter

    for ax, (title, col, det_col) in zip(axes, panels):
        for st in STAGES:
            sub = rep[rep["model_stage"] == st]
            if sub.empty:
                continue
            n = len(sub)
            jitter = (np.linspace(-0.12, 0.12, n) if n > 1 else np.zeros(1)) + rng.uniform(-0.01, 0.01, n)
            xs = x_of[st] + jitter
            ys = sub[col].to_numpy(dtype=float)
            det = sub[det_col].to_numpy(dtype=bool) if det_col else np.ones(n, bool)
            # detected: filled; not detected (LFQ = 0): hollow, drawn at 0
            ax.scatter(xs[det], ys[det], s=46, marker=STAGE_MARKERS[st], color=STAGE_COLORS[st],
                       edgecolor="white", linewidth=1.2, zorder=3)
            if (~det).any():
                ax.scatter(xs[~det], ys[~det], s=46, marker=STAGE_MARKERS[st], facecolor="white",
                           edgecolor=STAGE_COLORS[st], linewidth=1.4, zorder=3)
            m = float(np.mean(ys))
            ax.plot([x_of[st] - 0.26, x_of[st] + 0.26], [m, m], color=INK, lw=1.6,
                    solid_capstyle="round", zorder=2)
        if col == "Pr":
            # what the trajectory generator samples from (mu +/- sigma)
            for _, s in summ.iterrows():
                x = x_of[s["model_stage"]] + 0.36
                ax.errorbar(x, s["generator_mu"], yerr=s["generator_sigma"], fmt="D", ms=4,
                            color=MUTED, ecolor=MUTED, elinewidth=1.2, capsize=2.5, zorder=2)
        ax.set_title(title, color=INK, pad=6)
        ax.set_xticks(range(len(STAGES)))
        ax.set_xticklabels([f"{SOURCE_DAY[s]}\n({s})" for s in STAGES])
        ax.set_xlim(-0.55, len(STAGES) - 0.35)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(r"$\log_2(1 + \mathrm{LFQ})$")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], ls="", marker="o", ms=7, color=MUTED, label="replicate (detected)"),
        Line2D([], [], ls="", marker="o", ms=7, markerfacecolor="white", markeredgecolor=MUTED,
               label="replicate (not detected, LFQ = 0)"),
        Line2D([], [], color=INK, lw=1.6, label="stage mean"),
        Line2D([], [], ls="", marker="D", ms=4, color=MUTED,
               label=r"trajectory generator $\mu \pm \sigma$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proteingroups", type=Path,
                    default=os.environ.get("NOVAK2022_PROTEINGROUPS_PATH") or None,
                    help="Canonical proteinGroups file (default: $NOVAK2022_PROTEINGROUPS_PATH)")
    ap.add_argument("--to-paper", action="store_true",
                    help="Also copy the PNG to paper_dt1/images/")
    args = ap.parse_args()
    if not args.proteingroups or not Path(args.proteingroups).exists():
        raise SystemExit("Pass --proteingroups or set NOVAK2022_PROTEINGROUPS_PATH "
                         "(the canonical Dittmar proteinGroups file).")

    abundance, samples = _load(Path(args.proteingroups))
    rep = per_replicate_table(abundance, samples)
    summ = stage_summary(rep)

    OUT.mkdir(parents=True, exist_ok=True)
    rep.to_csv(OUT / "pr_per_replicate.csv", index=False)
    summ.to_csv(OUT / "pr_stage_summary.csv", index=False)
    (OUT / "sample_mapping.tex").write_text(mapping_tex(rep, summ), encoding="utf-8")
    plot(rep, summ, OUT / "fig_pr_replicates")
    if args.to_paper:
        PAPER_IMAGES.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT / "fig_pr_replicates.png", PAPER_IMAGES / "fig_pr_replicates.png")

    with pd.option_context("display.width", 140, "display.max_columns", 30):
        print(rep[["sample_label", "source_day", "model_stage", "Pr", "n_markers_detected"]]
              .to_string(index=False))
        print()
        print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n[replicates] wrote {OUT}/ (pr_per_replicate.csv, pr_stage_summary.csv, "
          f"sample_mapping.tex, fig_pr_replicates.png/.pdf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
