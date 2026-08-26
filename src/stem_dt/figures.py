"""Manuscript-ready figures for Paper 1.

All figures consume already-computed artifacts (state tensors + saved metric
JSONs from validation/benchmark runners). They do *not* re-run analyses.

Figures
-------
- :func:`hero_state_space_forecast` -- 3D `(P, D, Pr)` headline figure with
  population background + held-out test trajectories + observed-vs-predicted
  D52 markers + residual lines. The visualization-priority figure.
- :func:`projections_2d` -- three side-by-side 2D scatters `(P,D)`, `(P,Pr)`,
  `(D,Pr)` for reviewers who prefer flat axes.
- :func:`correlation_heatmap` -- per-stage Spearman ρ heatmap with
  bootstrap-CI annotations.
- :func:`mofa_variance_bar` -- stacked bar of MOFA+ variance explained per
  modality per factor.
- :func:`coherence_histogram` -- 2D vs 3D trajectory-coherence overlay.
- :func:`task_a_confusion` -- Task A confusion matrices, one per condition
  (2D / 3D / 3D-permuted).
- :func:`task_b_mae_bars` -- Task B MAE comparison with per-feature breakdown.

Conventions
-----------
- All functions take a `save_path` argument; if `None`, the function returns
  the matplotlib `Figure` object so the caller can `plt.show()` interactively.
- Saved figures use 300 DPI PNG (manuscript-ready); set `dpi=` to override.
- Color palette is colorblind-safe via matplotlib's `tab10`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Shared style helpers.
# ---------------------------------------------------------------------------

STAGE_COLORS = {"D11": "#4C72B0", "D30": "#DD8452", "D52": "#55A467"}
CONDITION_COLORS = {
    "2D":               "#9DA1A6",
    "3D":               "#2E7D32",
    "3D_perm_within":   "#F6B26B",
    "3D_perm_across":   "#C5544D",
    "3D_perm":          "#F6B26B",
    "3D perm within":   "#F6B26B",   # human-readable label variants used in legends
    "3D perm across":   "#C5544D",
    "4D":               "#6A3D9A",
}
STATE_LABELS = ("P", "D", "Pr", "K")


def _apply_paper_style() -> None:
    """Configure seaborn + matplotlib for journal-ready figures."""
    import matplotlib as mpl
    import seaborn as sns

    sns.set_theme(
        context="paper",
        style="ticks",
        palette="deep",
        font="DejaVu Sans",
        rc={
            "axes.spines.top":    False,
            "axes.spines.right":  False,
            "axes.titleweight":   "bold",
            "axes.titlesize":     12.5,
            "axes.labelsize":     11,
            "axes.labelweight":   "bold",
            "xtick.labelsize":    10,
            "ytick.labelsize":    10,
            "xtick.major.width":  1.0,
            "ytick.major.width":  1.0,
            "legend.fontsize":    9.5,
            "legend.title_fontsize": 10.5,
            "legend.frameon":     False,
            "figure.titlesize":   13,
            "figure.titleweight": "bold",
            "figure.dpi":         140,
            "savefig.dpi":        600,
            "savefig.bbox":       "tight",
            "axes.grid":          False,
            "axes.linewidth":     1.0,
            "lines.linewidth":    1.6,
            "patch.linewidth":    0.7,
        },
    )
    mpl.rcParams["font.weight"]    = "medium"  # body text slightly heavier
    mpl.rcParams["pdf.fonttype"]   = 42        # embed TrueType (editable in Illustrator)
    mpl.rcParams["ps.fonttype"]    = 42
    mpl.rcParams["text.color"]     = "#1a1a1a"  # near-black, less harsh than pure black
    mpl.rcParams["axes.labelcolor"] = "#1a1a1a"
    mpl.rcParams["xtick.color"]    = "#1a1a1a"
    mpl.rcParams["ytick.color"]    = "#1a1a1a"


def _save_or_return(fig, save_path: str | Path | None, dpi: int = 600):
    if save_path is None:
        return fig
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return Path(save_path)


def save_paper_figure(fig, base_path: str | Path) -> list[Path]:
    """Save a figure as both vector PDF and 600 DPI PNG.

    On Windows, PDF/PNG viewers hold an exclusive lock on opened files. If
    the target path is locked we fall back to `<base>.new.pdf` / `.new.png`
    and emit a notice — the user can close the viewer and rename, or just
    look at the `.new.*` version.
    """
    import matplotlib.pyplot as plt

    base = Path(base_path).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for ext in (".pdf", ".png"):
        target = base.with_suffix(ext)
        try:
            fig.savefig(target, dpi=600, bbox_inches="tight")
        except PermissionError:
            fallback = base.with_suffix(f".new{ext}")
            fig.savefig(fallback, dpi=600, bbox_inches="tight")
            print(f"  [locked] {target.name} held by another process; "
                  f"wrote to {fallback.name} instead")
            target = fallback
        out.append(target)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Hero figure -- the single image that tells the multi-omics-DT story.
# ---------------------------------------------------------------------------

@dataclass
class ForecastBundle:
    """Inputs to the hero figure.

    Attributes
    ----------
    state : (N, T, D) np.ndarray
        Full 3D state tensor across all 200 trajectories.
    test_idx : (n_test,) int array
        Indices into axis 0 of `state` selecting the held-out trajectories.
    predicted_d52 : (n_test, D) np.ndarray
        Model prediction at D52 for each held-out trajectory.
    stage_order : tuple[str, ...]
        Labels for axis 1 of `state`, e.g. ("D11", "D30", "D52").
    feature_labels : tuple[str, ...]
        Labels for axis 2 of `state`, e.g. ("P", "D", "Pr").
    """

    state: np.ndarray
    test_idx: np.ndarray
    predicted_d52: np.ndarray
    stage_order: tuple[str, ...] = ("D11", "D30", "D52")
    feature_labels: tuple[str, ...] = ("P", "D", "Pr")


def hero_state_space_forecast(
    bundle: ForecastBundle,
    save_path: str | Path | None = None,
    elev: float = 25.0,
    azim: float = -50.0,
    population_alpha: float = 0.06,
    test_alpha: float = 0.85,
    figsize: tuple[float, float] = (9, 7),
    title: str = "Multi-omics digital twin: state space + D52 forecast",
):
    """3D `(P, D, Pr)` hero figure: population + test + forecast + residuals.

    Layered rendering:
      1. All N trajectories drawn as faint gray polylines (background).
      2. `bundle.test_idx` trajectories highlighted, colored by per-test
         residual magnitude (red ≈ large error; viridis ≈ small).
      3. Observed D52 = green ●; Predicted D52 = red ▲; thin segment
         connecting them (the residual vector, geometrically interpretable).
      4. Stage centroids drawn as large diamond markers so the population
         path is readable through the cloud.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  -- registers 3d projection

    s = bundle.state
    if s.shape[2] < 3:
        raise ValueError(
            f"hero figure needs at least 3 features (P, D, Pr); got shape {s.shape}"
        )
    n_traj, n_stages, _ = s.shape
    if n_stages != len(bundle.stage_order):
        raise ValueError("state stage axis does not match stage_order length")

    test_idx = np.asarray(bundle.test_idx)
    pred = np.asarray(bundle.predicted_d52)
    obs = s[test_idx, -1, :]                          # observed D52
    residuals = np.linalg.norm(pred[:, :3] - obs[:, :3], axis=1)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    # --- Layer 1: faint background population
    for i in range(n_traj):
        ax.plot(
            s[i, :, 0], s[i, :, 1], s[i, :, 2],
            color="0.3", lw=0.4, alpha=population_alpha, zorder=1,
        )

    # --- Layer 2: test trajectories colored by residual magnitude
    cmap = plt.cm.viridis_r
    if residuals.max() > residuals.min():
        norm_res = (residuals - residuals.min()) / (residuals.max() - residuals.min())
    else:
        norm_res = np.zeros_like(residuals)
    for k, i in enumerate(test_idx):
        ax.plot(
            s[i, :, 0], s[i, :, 1], s[i, :, 2],
            color=cmap(norm_res[k]), lw=1.2, alpha=test_alpha, zorder=2,
        )

    # --- Layer 3: D52 observed (●) vs predicted (▲), with residual segments
    ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2],
               c="#2ca02c", marker="o", s=40,
               edgecolors="black", linewidths=0.5, label="Observed D52", zorder=4)
    ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2],
               c="#d62728", marker="^", s=40,
               edgecolors="black", linewidths=0.5, label="Predicted D52", zorder=4)
    for k in range(len(test_idx)):
        ax.plot(
            [obs[k, 0], pred[k, 0]],
            [obs[k, 1], pred[k, 1]],
            [obs[k, 2], pred[k, 2]],
            color="0.4", lw=0.6, alpha=0.6, zorder=3,
        )

    # --- Layer 4: stage centroids as a population trajectory
    centroids = s.mean(axis=0)                        # (T, D)
    ax.plot(centroids[:, 0], centroids[:, 1], centroids[:, 2],
            color="black", lw=2.0, ls="--", alpha=0.9, zorder=5)
    for t_idx, stage in enumerate(bundle.stage_order):
        ax.scatter(*centroids[t_idx, :3],
                   marker="D", s=140, zorder=6,
                   c=STAGE_COLORS.get(stage, "gray"),
                   edgecolors="black", linewidths=1.0,
                   label=f"{stage} centroid")

    ax.set_xlabel(bundle.feature_labels[0])
    ax.set_ylabel(bundle.feature_labels[1])
    ax.set_zlabel(bundle.feature_labels[2])
    ax.set_title(title)

    # Deduplicate legend (centroid scatter adds 3 entries)
    handles, labels = ax.get_legend_handles_labels()
    seen, dedup_h, dedup_l = set(), [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        dedup_h.append(h)
        dedup_l.append(l)
    ax.legend(dedup_h, dedup_l, loc="upper left", fontsize=8, framealpha=0.9)

    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# Supporting figure: 2D projections.
# ---------------------------------------------------------------------------

def projections_2d(
    state: np.ndarray,
    stage_order: tuple[str, ...] = ("D11", "D30", "D52"),
    feature_labels: tuple[str, ...] = ("P", "D", "Pr"),
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 4),
):
    """3 × 2D scatters: (P,D), (P,Pr), (D,Pr) colored by stage."""
    import matplotlib.pyplot as plt

    if state.shape[2] < 3:
        raise ValueError("projections_2d expects ≥3 features")

    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (i, j) in zip(axes, pairs):
        for t_idx, stage in enumerate(stage_order):
            ax.scatter(
                state[:, t_idx, i], state[:, t_idx, j],
                s=14, alpha=0.6,
                color=STAGE_COLORS.get(stage, "gray"), label=stage,
                edgecolors="none",
            )
        ax.set_xlabel(feature_labels[i])
        ax.set_ylabel(feature_labels[j])
        ax.set_title(f"{feature_labels[i]} vs {feature_labels[j]}")
        ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# Cross-modal correlation heatmap (per stage).
# ---------------------------------------------------------------------------

def correlation_heatmap(
    correlations: dict,
    feature_labels: tuple[str, ...] = ("P", "D", "Pr"),
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (11, 3.6),
    annotate_ci: bool = True,
):
    """Per-stage Spearman ρ heatmap, paper-quality with annotated CIs.

    `correlations` is the output of :func:`validation.cross_modal_correlations`:
        {stage: {"rho": (D,D), "ci_low": (D,D), "ci_high": (D,D)}}

    Each cell shows ρ in bold + bracketed 95% CI in smaller text. Cells whose
    CI excludes 0 (statistically distinguishable from independence) are marked
    with a small filled circle in the corner.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    _apply_paper_style()

    stages = list(correlations.keys())
    fig, axes = plt.subplots(1, len(stages), figsize=figsize)
    if len(stages) == 1:
        axes = [axes]

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    last_im = None

    for ax, stage in zip(axes, stages):
        rho = np.asarray(correlations[stage]["rho"], dtype=float)
        # Sentinel: render NaN cells as a neutral grey BG distinct from ρ≈0 (white).
        with np.errstate(invalid="ignore"):
            display = np.where(np.isnan(rho), 0.0, rho)
        nan_mask = ~np.isfinite(rho)

        im = ax.imshow(display, cmap=cmap, vmin=-1, vmax=1,
                       aspect="equal", interpolation="nearest")
        last_im = im

        # Overlay grey background + diagonal hatch on n/a cells so they
        # clearly read as "undefined" rather than "near zero".
        from matplotlib.patches import Rectangle
        for ii, jj in zip(*np.where(nan_mask)):
            ax.add_patch(Rectangle(
                (jj - 0.5, ii - 0.5), 1, 1,
                facecolor="#E5E5E5", edgecolor="white", linewidth=0.6,
                hatch="///", zorder=2,
            ))

        ax.set_xticks(range(len(feature_labels)))
        ax.set_yticks(range(len(feature_labels)))
        ax.set_xticklabels(feature_labels, fontsize=11.5, weight="bold")
        ax.set_yticklabels(feature_labels, fontsize=11.5, weight="bold")
        ax.set_title(stage, fontsize=12.5, weight="bold", pad=6)
        ax.tick_params(length=0)
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(False)

        ci_low = correlations[stage].get("ci_low")
        ci_high = correlations[stage].get("ci_high")
        ci_low = np.asarray(ci_low, dtype=float) if ci_low is not None else None
        ci_high = np.asarray(ci_high, dtype=float) if ci_high is not None else None

        for i in range(rho.shape[0]):
            for j in range(rho.shape[1]):
                if i == j:
                    ax.text(j, i, "1.00", ha="center", va="center",
                            fontsize=10, color="0.35", style="italic",
                            weight="bold", zorder=3)
                    continue

                r = rho[i, j]
                if not np.isfinite(r):
                    ax.text(j, i, "n/a", ha="center", va="center",
                            fontsize=12, color="0.25", weight="bold",
                            zorder=3)
                    continue

                color = "white" if abs(r) > 0.5 else "#1a1a1a"
                ax.text(j, i, f"{r:+.2f}",
                        ha="center", va="center",
                        fontsize=12.5, color=color, weight="bold",
                        zorder=3)

                if annotate_ci and ci_low is not None:
                    lo, hi = ci_low[i, j], ci_high[i, j]
                    if np.isfinite(lo) and np.isfinite(hi):
                        ax.text(j, i + 0.34, f"[{lo:+.2f}, {hi:+.2f}]",
                                ha="center", va="center",
                                fontsize=8, color=color, weight="semibold",
                                alpha=0.9, zorder=3)
                        # Filled-circle marker if CI excludes 0
                        if lo > 0 or hi < 0:
                            ax.plot(j + 0.36, i - 0.36, marker="o",
                                    markersize=5, color=color,
                                    markeredgecolor="white",
                                    markeredgewidth=0.5, zorder=4)

    cbar = fig.colorbar(last_im, ax=axes, fraction=0.014, pad=0.02,
                        ticks=[-1, -0.5, 0, 0.5, 1], shrink=0.85)
    cbar.set_label("Spearman ρ", fontsize=10.5, weight="bold")
    cbar.ax.tick_params(labelsize=9)
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# MOFA+ variance bar.
# ---------------------------------------------------------------------------

def mofa_variance_bar(
    variance_explained: dict,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (8, 4),
):
    """Stacked bar of % variance explained, per modality per factor.

    `variance_explained` is a dict produced by the MOFA+ runner:
        {modality_label: np.ndarray of shape (n_factors,) % values}
    """
    import matplotlib.pyplot as plt

    if not variance_explained:
        raise ValueError("empty variance_explained dict")
    modalities = list(variance_explained.keys())
    n_factors = len(next(iter(variance_explained.values())))

    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(n_factors)
    factor_idx = np.arange(n_factors)
    for mod in modalities:
        vals = np.asarray(variance_explained[mod])
        ax.bar(factor_idx, vals, bottom=bottom, label=mod, edgecolor="white")
        bottom += vals

    ax.set_xticks(factor_idx)
    ax.set_xticklabels([f"F{i+1}" for i in range(n_factors)])
    ax.set_xlabel("MOFA+ factor")
    ax.set_ylabel("% variance explained")
    ax.set_title("Modality-resolved variance decomposition")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# Trajectory coherence histograms.
# ---------------------------------------------------------------------------

def coherence_histogram(
    coherence_2d: np.ndarray,
    coherence_3d: np.ndarray,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (7, 4),
):
    """Overlay 2D vs 3D trajectory-coherence distributions."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    bins = np.linspace(-1, 1, 31)
    ax.hist(coherence_2d, bins=bins, alpha=0.55, label="2D (P, D)",
            color="#1f77b4", edgecolor="white")
    ax.hist(coherence_3d, bins=bins, alpha=0.55, label="3D (P, D, Pr)",
            color="#ff7f0e", edgecolor="white")
    ax.axvline(0, color="0.4", lw=0.8, ls="--")
    ax.set_xlabel("Trajectory coherence (cosine of consecutive displacements)")
    ax.set_ylabel("count")
    ax.set_title("Trajectory coherence: 2D vs 3D")
    ax.legend()
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# Task A confusion matrices.
# ---------------------------------------------------------------------------

def task_a_confusion(
    confusions: dict,
    classes: tuple[str, ...] = ("D11", "D30", "D52"),
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 4),
):
    """Side-by-side confusion matrices keyed by condition label.

    `confusions` is e.g. {"2D": np.ndarray(3,3), "3D": ..., "3D-perm": ...}
    """
    import matplotlib.pyplot as plt

    keys = list(confusions.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=figsize)
    if len(keys) == 1:
        axes = [axes]

    for ax, k in zip(axes, keys):
        cm = np.asarray(confusions[k]).astype(float)
        # Row-normalize so each row sums to 1 (= per-true-class recall view)
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(k)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                txt = f"{int(cm[i, j])}\n({cm_norm[i, j]:.2f})"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if cm_norm[i, j] > 0.5 else "black")

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="row-normalized")
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# Task B MAE bar comparison.
# ---------------------------------------------------------------------------

def task_b_mae_bars(
    metrics: dict,
    feature_labels: tuple[str, ...] = ("P", "D", "Pr"),
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (9, 5),
):
    """Grouped bar chart: per-condition (2D/3D/4D) test MAE per feature.

    `metrics` is e.g.
        {
          "2D": {"P": 0.085, "D": 0.076},
          "3D": {"P": 0.07, "D": 0.06, "Pr": 0.09},
          ...
        }
    Conditions appear as bar groups; features as bars within a group.
    """
    import matplotlib.pyplot as plt

    conditions = list(metrics.keys())
    feature_set = sorted({f for d in metrics.values() for f in d})
    width = 0.8 / max(len(feature_set), 1)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(conditions))
    for i, feat in enumerate(feature_set):
        vals = [metrics[c].get(feat, np.nan) for c in conditions]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width,
               label=feat, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Test MAE")
    ax.set_title("Task B: trajectory prediction MAE per feature")
    ax.legend(title="feature", fontsize=9)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# Convenience runner: render all figures from saved JSON / npz artifacts.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Paper figures (Fig 1–4) — match the Results section spec.
# ---------------------------------------------------------------------------

def fig1_pipeline(
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 6),
):
    """Fig 1 — Pipeline flowchart: 5 stages from data sources to outputs.

    Pure matplotlib so it commits with the codebase. For a more polished
    publication graphic, the user should re-draw in BioRender / draw.io.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, text, color="#1f77b4"):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08",
            facecolor=color, edgecolor="black", linewidth=0.8, alpha=0.55,
        ))
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center", fontsize=9, weight="bold")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
            color="0.3", linewidth=1.2,
        ))

    # Stage 1 (data sources) — left column
    box(0.2, 4.5, 2.4, 0.8, "Jerber 2021\n(scRNA-seq, n=161,584)", "#9bcfe9")
    box(0.2, 3.3, 2.4, 0.8, "Novak 2022 (Path 3)\n(MaxQuant, 2,965 prots)", "#a5d6a7")
    box(0.2, 2.1, 2.4, 0.8, "PSP kinase substrates\n(K_t pathway prior)", "#ffe082")

    # Stage 2 (state construction)
    box(3.4, 3.4, 2.4, 1.2, "Score construction\nP, D (transcript)\nPr (proteomic)", "#ce93d8")

    # Stage 3 (validation + provenance side-branch)
    box(6.6, 5.2, 2.4, 0.7, "Validation\n(corr / coh / MOFA+)", "#f48fb1")
    box(6.6, 0.8, 2.4, 0.7, "PureChain provenance\n(SHA-256 + on-chain)", "#80cbc4")

    # Stage 4 (models)
    box(6.6, 4.0, 2.4, 0.7, "Ridge / RF / Tx-DT\n(2D vs 3D vs perm)", "#90a4ae")

    # Stage 5 (outputs)
    box(9.8, 4.6, 2.0, 0.7, "Task A:\nMacro F1 (3-class)", "#ef9a9a")
    box(9.8, 3.4, 2.0, 0.7, "Task B:\nMAE D52", "#ef9a9a")

    # Arrows
    arrow(2.6, 4.9, 3.4, 4.4)
    arrow(2.6, 3.7, 3.4, 4.0)
    arrow(2.6, 2.5, 3.4, 3.6)
    arrow(5.8, 4.0, 6.6, 4.4)
    arrow(5.8, 4.0, 6.6, 5.5)
    arrow(5.8, 4.0, 6.6, 1.0)
    arrow(9.0, 4.4, 9.8, 5.0)
    arrow(9.0, 4.4, 9.8, 3.7)

    ax.text(1.4, 5.6, "Stage 1 — data sources",
            ha="center", fontsize=9, style="italic")
    ax.text(4.6, 4.8, "Stage 2 — state",
            ha="center", fontsize=9, style="italic")
    ax.text(7.8, 5.95, "Stage 4 — validation",
            ha="center", fontsize=9, style="italic")
    ax.text(7.8, 4.85, "Stage 5 — benchmark",
            ha="center", fontsize=9, style="italic")
    ax.text(7.8, 1.55, "Stage 3 — provenance",
            ha="center", fontsize=9, style="italic")
    ax.text(10.8, 5.4, "Outputs", ha="center", fontsize=9, style="italic")

    fig.suptitle("Multi-omics digital twin construction — pipeline overview",
                 fontsize=11)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


def fig2_2d_vs_3d_scatter(
    state: np.ndarray,
    stage_order: tuple[str, ...] = ("D11", "D30", "D52"),
    feature_labels: tuple[str, ...] = ("P", "D", "Pr"),
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (12.5, 5.6),
):
    """Fig 2 — 2D vs 3D state space; paper-quality with marginal KDEs.

    Two panels:
      Left:  scatter on (P, D) only — shows 2D-baseline overlap between stages.
             Marginal KDE strips on top + right communicate per-axis distributions.
      Right: same scatter, with **per-trajectory polylines** linking the same
             cell across stages and **bubble size encoding Pr** — the new
             dimension that resolves stage overlap. Stage centroids drawn as
             diamond markers.

    Both panels use the same x/y axis scale and stage palette so visual
    comparison is direct.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import gridspec

    _apply_paper_style()

    if state.shape[2] < 3:
        raise ValueError("fig2 requires (N, T, 3+) state tensor")

    fig = plt.figure(figsize=figsize)
    outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.25)

    n_traj, n_stages, _ = state.shape
    colors = [STAGE_COLORS.get(s, "gray") for s in stage_order]
    centroids = state.mean(axis=0)  # (T, D)

    def _build_panel(g_outer, title: str, with_pr_size: bool, with_traj: bool):
        # 2x2 sub-grid: top-marg, top-empty, scatter, right-marg
        gs = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=g_outer,
            width_ratios=[5, 1], height_ratios=[1, 5],
            hspace=0.05, wspace=0.05,
        )
        ax_top   = fig.add_subplot(gs[0, 0])
        ax_main  = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

        # --- top marginal: KDEs of P per stage
        for t_idx, stage in enumerate(stage_order):
            sns.kdeplot(
                x=state[:, t_idx, 0], ax=ax_top, fill=True, alpha=0.25,
                color=colors[t_idx], linewidth=0.9, bw_adjust=0.85,
            )
        ax_top.set_xticks([]); ax_top.set_yticks([])
        ax_top.set_xlabel(""); ax_top.set_ylabel("")
        for s in ("top", "right", "left"):
            ax_top.spines[s].set_visible(False)
        ax_top.spines["bottom"].set_linewidth(0.4)

        # --- right marginal: KDEs of D per stage (rotated)
        for t_idx, stage in enumerate(stage_order):
            sns.kdeplot(
                y=state[:, t_idx, 1], ax=ax_right, fill=True, alpha=0.25,
                color=colors[t_idx], linewidth=0.9, bw_adjust=0.85,
            )
        ax_right.set_xticks([]); ax_right.set_yticks([])
        ax_right.set_xlabel(""); ax_right.set_ylabel("")
        for s in ("top", "right", "bottom"):
            ax_right.spines[s].set_visible(False)
        ax_right.spines["left"].set_linewidth(0.4)

        # --- main scatter
        if with_traj:
            # Faint per-trajectory polylines, one per cell, drawn before scatter
            for i in range(n_traj):
                ax_main.plot(state[i, :, 0], state[i, :, 1],
                             color="0.5", alpha=0.06, lw=0.4, zorder=1)

        for t_idx, stage in enumerate(stage_order):
            xs = state[:, t_idx, 0]; ys = state[:, t_idx, 1]
            if with_pr_size:
                pr = state[:, t_idx, 2]
                # Larger floor so D11 (Pr≈0) remains visible; previous floor
                # of 10 made D11 dots disappear behind the larger D30/D52
                # bubbles (~170 pt²). 28 + 140·Pr keeps D11 readable.
                sizes = 28 + 140 * np.clip(pr, 0, 1)
            else:
                sizes = 30
            # Draw D11 LAST so its small dots aren't occluded by larger
            # D30/D52 bubbles. Reorder by mean(Pr) ascending → D11 paints last.
            pass
            ax_main.scatter(
                xs, ys, s=sizes, color=colors[t_idx], label=stage,
                alpha=0.65, edgecolors="white", linewidths=0.7,
                zorder=3 if stage == "D11" else 2,
            )

        # Stage centroid markers
        for t_idx, stage in enumerate(stage_order):
            ax_main.scatter(
                centroids[t_idx, 0], centroids[t_idx, 1], marker="D",
                s=200, color=colors[t_idx], edgecolors="black",
                linewidths=1.3, zorder=4,
            )
            ax_main.annotate(
                stage, xy=(centroids[t_idx, 0], centroids[t_idx, 1]),
                xytext=(8, 8), textcoords="offset points",
                fontsize=10.5, weight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="0.5", lw=0.6, alpha=0.85),
            )

        ax_main.set_xlabel(feature_labels[0], weight="bold")
        ax_main.set_ylabel(feature_labels[1], weight="bold")
        leg = ax_main.legend(title="Stage", loc="upper right", fontsize=10)
        leg.get_title().set_fontweight("bold")
        ax_main.tick_params(axis="both", labelsize=10)

        ax_top.set_title(title, fontsize=11.5, pad=6, weight="bold")
        return ax_main

    ax_left = _build_panel(
        outer[0], title="2D state (P, D)",
        with_pr_size=False, with_traj=False,
    )
    ax_right = _build_panel(
        outer[1],
        title="3D state (P, D, Pr) — bubble size encodes Pr;\n"
              "grey lines connect sampled trajectories",
        with_pr_size=True, with_traj=True,
    )

    # Sync axes ranges across panels for direct visual comparison
    xlim = (
        min(ax_left.get_xlim()[0], ax_right.get_xlim()[0]),
        max(ax_left.get_xlim()[1], ax_right.get_xlim()[1]),
    )
    ylim = (
        min(ax_left.get_ylim()[0], ax_right.get_ylim()[0]),
        max(ax_left.get_ylim()[1], ax_right.get_ylim()[1]),
    )
    ax_left.set_xlim(xlim); ax_left.set_ylim(ylim)
    ax_right.set_xlim(xlim); ax_right.set_ylim(ylim)

    return _save_or_return(fig, save_path)


def fig5_dlt_verification(
    *,
    contract_address: str = "0xFA954E93c389D54B05FBA832276B7C7a42bAeD51",
    chain_id: str = "900520900520",
    novak_block: int = 1_203_043,
    novak_gas: int = 472_168,
    block_time_s: float = 1.0,
    verify_latency_ms: float = 42.0,
    hash_throughput_mb_s: float = 150.0,
    confirmation_depth: int = 5_118,
    novak_file_mb: float = 4.4,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (12.5, 7.5),
):
    """Fig 5 — PureChain verification flow: re-hash → on-chain query → tamper branch.

    Two-lane sequence diagram (Local pipeline ↔ PureChain contract) with all
    empirical numbers from the live deployment annotated inline. Used in the
    Results section to give reviewers the audit-trail mechanism in one image.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
    import seaborn as sns

    _apply_paper_style()

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 16); ax.set_ylim(0, 11)
    ax.axis("off")

    # ----- Lane backgrounds
    LOCAL_BG  = "#F2EFEC"
    CHAIN_BG  = "#E8F2EC"
    LOCAL_ACC = "#4A4A4A"
    CHAIN_ACC = "#2E7D32"

    ax.add_patch(Rectangle((0.3, 1.85), 6.6, 7.65, facecolor=LOCAL_BG,
                           edgecolor=LOCAL_ACC, linewidth=1.2, zorder=1))
    ax.add_patch(Rectangle((9.1, 1.85), 6.6, 7.65, facecolor=CHAIN_BG,
                           edgecolor=CHAIN_ACC, linewidth=1.2, zorder=1))

    # ----- Lane headers (sit above the lane backgrounds; title sits above them)
    ax.text(3.6, 10.20, "LOCAL PIPELINE", ha="center", va="center",
            fontsize=13, weight="bold", color=LOCAL_ACC)
    ax.text(12.4, 10.20, "PURECHAIN MAINNET", ha="center", va="center",
            fontsize=13, weight="bold", color=CHAIN_ACC)
    ax.text(3.6, 9.75, "scripts/run_benchmark.py + provenance.py",
            ha="center", va="center", fontsize=8.5,
            color=LOCAL_ACC, style="italic")
    ax.text(12.4, 9.75, f"MultiOmicsProvenance @ {contract_address[:10]}…",
            ha="center", va="center", fontsize=8.5,
            color=CHAIN_ACC, style="italic")

    # ----- Helper: stage box on a lane
    def lane_box(x, y, w, h, title, subtitle, *, lane="local"):
        bg = "white"
        edge = LOCAL_ACC if lane == "local" else CHAIN_ACC
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.18,rounding_size=0.15",
            facecolor=bg, edgecolor=edge, linewidth=1.4, zorder=3,
        ))
        ax.text(x + w/2, y + h*0.62, title, ha="center", va="center",
                fontsize=10.5, weight="bold", color="#1a1a1a", zorder=4)
        ax.text(x + w/2, y + h*0.27, subtitle, ha="center", va="center",
                fontsize=8.5, color="#3a3a3a", zorder=4)

    def lane_step(idx, y, title, sub, lane="local"):
        x = 0.7 if lane == "local" else 9.5
        # Step bubble
        ax.add_patch(FancyBboxPatch(
            (x, y), 0.7, 0.7, boxstyle="circle,pad=0.08",
            facecolor=LOCAL_ACC if lane == "local" else CHAIN_ACC,
            edgecolor="none", zorder=4,
        ))
        ax.text(x + 0.35, y + 0.35, str(idx), ha="center", va="center",
                fontsize=11, weight="bold", color="white", zorder=5)
        # Content
        bx = x + 0.9
        lane_box(bx, y - 0.05, 5.1, 0.8, title, sub, lane=lane)

    # ----- Steps
    lane_step(1, 8.5,
              "Re-hash the file on disk",
              f"SHA-256 streaming • {hash_throughput_mb_s:.0f} MB/s • {novak_file_mb:.1f} MB → {int(1000*novak_file_mb/hash_throughput_mb_s)} ms",
              lane="local")

    lane_step(2, 6.7,
              "Call verifyIngestion(key, h)",
              "Read-only EVM call • no transaction submitted",
              lane="local")

    # Right side: contract receives + computes
    lane_step(3, 6.7,
              "Contract reads stored record",
              f"keccak(name, hash, signer) → ingestion[key]",
              lane="chain")

    lane_step(4, 4.9,
              "Returns (matched, recorded, version)",
              f"matched = (recorded_hash == h) • {verify_latency_ms:.0f} ms RPC roundtrip",
              lane="chain")

    # ----- Cross-lane arrows
    def cross_arrow(x0, y0, x1, y1, label, color="0.3", side="below"):
        arr = FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>",
            mutation_scale=18, color=color, linewidth=1.6, zorder=2,
        )
        ax.add_patch(arr)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my + (0.32 if side == "above" else -0.32), label,
                ha="center", va="center", fontsize=9, weight="semibold",
                color=color, zorder=3,
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec="0.7", lw=0.5))

    # Arrow: step 2 → step 3 (right-going)
    cross_arrow(6.8, 7.05, 9.55, 7.05, "RPC call", color=LOCAL_ACC, side="above")
    # Arrow: step 4 → step 5 (left-going, return)
    cross_arrow(9.55, 5.25, 6.8, 5.25, "RPC return", color=CHAIN_ACC, side="below")

    # ----- Step 5: branch outcomes (back on the local lane, stacked vertically)
    bubble_y = 2.80
    ax.add_patch(FancyBboxPatch(
        (0.7, bubble_y), 0.7, 0.7, boxstyle="circle,pad=0.08",
        facecolor=LOCAL_ACC, edgecolor="none", zorder=4,
    ))
    ax.text(1.05, bubble_y + 0.35, "5", ha="center", va="center",
            fontsize=11, weight="bold", color="white", zorder=5)

    # Two-branch outcome boxes (stacked: True above, False below)
    OK_BG  = "#DDEFE0"
    BAD_BG = "#F8DEDE"
    OK_BORDER  = "#2E7D32"
    BAD_BORDER = "#C62828"

    branch_x, branch_w = 1.60, 5.10

    # Matched = True branch (upper)
    true_y = 3.30
    ax.add_patch(FancyBboxPatch(
        (branch_x, true_y), branch_w, 0.65,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=OK_BG, edgecolor=OK_BORDER, linewidth=1.4, zorder=3,
    ))
    ax.text(branch_x + 0.18, true_y + 0.325, "✓",
            ha="left", va="center", fontsize=14, weight="bold",
            color=OK_BORDER, zorder=4)
    ax.text(branch_x + 0.55, true_y + 0.46, "matched = True",
            ha="left", va="center", fontsize=10, weight="bold",
            color=OK_BORDER, zorder=4)
    ax.text(branch_x + 0.55, true_y + 0.20, "integrity confirmed → proceed",
            ha="left", va="center", fontsize=8.5, color="#1a1a1a", zorder=4)

    # Matched = False branch (lower)
    false_y = 2.45
    ax.add_patch(FancyBboxPatch(
        (branch_x, false_y), branch_w, 0.65,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=BAD_BG, edgecolor=BAD_BORDER, linewidth=1.4, zorder=3,
    ))
    ax.text(branch_x + 0.18, false_y + 0.325, "✗",
            ha="left", va="center", fontsize=14, weight="bold",
            color=BAD_BORDER, zorder=4)
    ax.text(branch_x + 0.55, false_y + 0.46, "matched = False",
            ha="left", va="center", fontsize=10, weight="bold",
            color=BAD_BORDER, zorder=4)
    ax.text(branch_x + 0.55, false_y + 0.20, "tamper detected → halt",
            ha="left", va="center", fontsize=8.5, color="#1a1a1a", zorder=4)

    # Connector from step 5 bubble splitting into both branches
    ax.add_patch(FancyArrowPatch(
        (1.4, bubble_y + 0.35), (branch_x - 0.02, true_y + 0.325),
        arrowstyle="-", color=LOCAL_ACC, linewidth=1.2, zorder=2,
        connectionstyle="arc3,rad=-0.15",
    ))
    ax.add_patch(FancyArrowPatch(
        (1.4, bubble_y + 0.35), (branch_x - 0.02, false_y + 0.325),
        arrowstyle="-", color=LOCAL_ACC, linewidth=1.2, zorder=2,
        connectionstyle="arc3,rad=0.15",
    ))

    # ----- Sidecar: empirical bottom strip (sits below the lane backgrounds)
    strip_y = 0.35
    ax.add_patch(Rectangle((0.3, strip_y), 15.4, 1.15,
                            facecolor="white", edgecolor="0.7", linewidth=0.8,
                            zorder=2))
    ax.text(0.55, strip_y + 0.92, "Live measurements (2026-05-13)",
            ha="left", va="center", fontsize=9.5, weight="bold",
            color="#1a1a1a")

    metrics = [
        ("Tx-confirm latency",    f"~{block_time_s:.1f} s/block"),
        ("verifyIngestion RPC",   f"{verify_latency_ms:.0f} ms (n=5)"),
        ("Gas per write",         f"{novak_gas:,}"),
        ("Cost",                  "0 PCG (gas-free)"),
        ("Confirmation depth",    f"{confirmation_depth:,} blocks"),
        ("Tamper-demo",           "PASS"),
    ]
    # Distribute six columns evenly across the strip's usable width.
    x_left, x_right = 0.55, 15.45
    col_w = (x_right - x_left) / len(metrics)
    for i, (k, v) in enumerate(metrics):
        cx = x_left + i * col_w
        ax.text(cx, strip_y + 0.55, k, ha="left", va="center",
                fontsize=8, color="#5a5a5a", weight="semibold")
        ax.text(cx, strip_y + 0.25, v, ha="left", va="center",
                fontsize=9, color="#1a1a1a", weight="bold")

    # ----- Title (above the lane headers, centered across both lanes)
    ax.text(8, 10.75,
            "PureChain verification flow — local re-hash → on-chain query → match / tamper branch",
            ha="center", va="center", fontsize=12.5, weight="bold",
            color="#1a1a1a")

    return _save_or_return(fig, save_path)


def fig4_benchmark_bars(
    task_a: dict[str, dict[str, float]],
    task_b: dict[str, dict[str, float]],
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (12.5, 4.8),
    *,
    task_a_std: dict[str, dict[str, float]] | None = None,
    task_b_std: dict[str, dict[str, float]] | None = None,
    n_seeds: int | None = None,
):
    """Fig 4 — Task A F1 + Task B MAE grouped bars with permutation nulls.

    Paper-quality with seaborn styling:
      - Per-bar value labels above each bar
      - Chance reference line on Task A (1/3 for 3-class)
      - Optional error bars (±1 std) when `task_a_std` / `task_b_std` are
        supplied — paired with multi-seed sweep output from
        `artifacts/multi_seed_metrics.json`
      - Coordinated palette across panels via CONDITION_COLORS

    When `n_seeds` is provided, the figure subtitle annotates the sample
    size so the error bars are unambiguously interpretable.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    _apply_paper_style()

    models_a = list(task_a.keys())
    models_b = list(task_b.keys())
    width = 0.27

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    def _bar_panel(ax, models, table, conditions, ylabel, title,
                   value_fmt: str, ylim, std_table: dict | None = None,
                   chance_line: float | None = None):
        x = np.arange(len(models))
        for i, cond in enumerate(conditions):
            vals = [table[m].get(cond, np.nan) for m in models]
            errs = ([std_table[m].get(cond, 0.0) if std_table and m in std_table
                     else 0.0 for m in models]
                    if std_table is not None else None)
            offsets = (i - (len(conditions) - 1) / 2) * width
            color = CONDITION_COLORS.get(cond, "0.6")
            bars = ax.bar(x + offsets, vals, width=width,
                          label=cond.replace("_", " "),
                          color=color, edgecolor="white", linewidth=0.9,
                          alpha=0.97,
                          yerr=errs if errs is not None else None,
                          error_kw=dict(ecolor="#3a3a3a", elinewidth=1.1,
                                        capsize=2.8, capthick=1.0,
                                        zorder=5) if errs is not None else None)
            for j, (rect, v) in enumerate(zip(bars, vals)):
                if not np.isfinite(v):
                    continue
                # Float label above the error bar tip so they don't collide.
                err = errs[j] if errs is not None else 0.0
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + err + (ylim[1] - ylim[0]) * 0.018,
                        value_fmt.format(v),
                        ha="center", va="bottom", fontsize=9,
                        color="#1a1a1a", weight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10.5, weight="bold")
        ax.set_ylabel(ylabel, fontsize=11, weight="bold")
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=12.5, weight="bold", pad=8)
        leg = ax.legend(title="Condition", loc="upper right",
                        fontsize=10, title_fontsize=10.5,
                        ncol=1, frameon=False)
        leg.get_title().set_fontweight("bold")
        for txt in leg.get_texts():
            txt.set_fontweight("semibold")
        sns.despine(ax=ax)

        if chance_line is not None:
            ax.axhline(chance_line, ls=(0, (3, 3)), lw=1.2,
                       color="0.45", zorder=1)
            ax.text(len(models) - 0.5,
                    chance_line + (ylim[1] - ylim[0]) * 0.025,
                    f"chance ({chance_line:.2f})", fontsize=9,
                    color="0.35", ha="right", style="italic",
                    weight="semibold")

    _bar_panel(
        axes[0], models_a, task_a,
        conditions=["2D", "3D", "3D_perm_across"],
        ylabel="Macro F1", title="Task A — stage classification",
        value_fmt="{:.2f}", ylim=(0, 1.16),
        std_table=task_a_std,
        chance_line=1 / 3,
    )

    # Task B: lower is better; condition list uses '3D_perm' (within) per benchmark.py.
    # The benchmark module stores it as '3D_perm' (not '3D_perm_within') for legacy
    # reasons; we rename only the legend label so the figure matches the paper's
    # nomenclature ("3D perm within" — the appropriate null for trajectory tasks).
    perm_b_key = "3D_perm" if any("3D_perm" in d for d in task_b.values()) else "3D_perm_within"

    # Tight y-axis: bars run ~0.06–0.10, give breathing room without wasting half
    # the panel on empty whitespace below the bars. Error bars widen the
    # visible top of each bar by up to ±0.01, so we add a touch more headroom
    # when std data is present.
    b_min = min(v for d in task_b.values() for v in d.values() if np.isfinite(v))
    b_max = max(v for d in task_b.values() for v in d.values() if np.isfinite(v))
    if task_b_std is not None:
        b_max += max((s.get(c, 0.0) for s in task_b_std.values()
                      for c in s), default=0.0)
    pad   = (b_max - b_min) * 0.55
    y_lo  = max(0.0, b_min - pad * 0.4)
    y_hi  = b_max + pad

    # Temporarily rewrite the dict-key for the legend label, restore after
    relabeled = {m: {("3D perm within" if k == perm_b_key else k): v
                     for k, v in d.items()} for m, d in task_b.items()}
    relabeled_std = None
    if task_b_std is not None:
        relabeled_std = {m: {("3D perm within" if k == perm_b_key else k): v
                             for k, v in d.items()}
                         for m, d in task_b_std.items()}

    _bar_panel(
        axes[1], models_b, relabeled,
        conditions=["2D", "3D", "3D perm within"],
        ylabel="Test MAE  (lower is better)",
        title="Task B — trajectory prediction",
        value_fmt="{:.3f}", ylim=(y_lo, y_hi),
        std_table=relabeled_std,
    )

    if n_seeds:
        suptitle = (f"Benchmark: 2D baseline vs 3D multi-omics state vs permutation null  "
                    f"(error bars = ±1 std across {n_seeds} seeds)")
    else:
        suptitle = "Benchmark: 2D baseline vs 3D multi-omics state vs permutation null"
    fig.suptitle(suptitle, fontsize=12.5, weight="bold", y=1.05)
    return _save_or_return(fig, save_path)


def render_all_from_artifacts(
    state_path: str | Path,
    metrics_path: str | Path,
    out_dir: str | Path,
) -> list[Path]:
    """Render every figure in this module from saved validation+benchmark output.

    Expected on-disk inputs (filled by run_validation.py and run_benchmark.py):
      - state_path: .npz with keys `state` (N, T, D) and `predicted_d52` (n_test, D),
        plus `test_idx` (n_test,)
      - metrics_path: .json with keys
        `correlations`, `mofa_variance`, `coherence_2d`, `coherence_3d`,
        `task_a_confusion`, `task_b_mae`
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    npz = np.load(state_path)
    state = npz["state"]
    test_idx = npz["test_idx"]
    pred = npz["predicted_d52"]

    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))

    # 1. Hero
    written.append(hero_state_space_forecast(
        ForecastBundle(state=state, test_idx=test_idx, predicted_d52=pred),
        save_path=out_dir / "fig_hero_3d.png",
    ))
    # 2. 2D projections
    written.append(projections_2d(state, save_path=out_dir / "fig_projections_2d.png"))
    # 3. Correlations (decode lists -> np arrays first)
    if "correlations" in metrics:
        corr = {
            stage: {
                k: np.asarray(v) for k, v in stage_data.items()
            }
            for stage, stage_data in metrics["correlations"].items()
        }
        written.append(correlation_heatmap(corr, save_path=out_dir / "fig_correlations.png"))
    # 4. MOFA+ (skip if mofapy2 wasn't available at validation time)
    if metrics.get("mofa_variance"):
        written.append(mofa_variance_bar(
            metrics["mofa_variance"], save_path=out_dir / "fig_mofa_variance.png",
        ))
    # 5. Coherence
    if "coherence_2d" in metrics and "coherence_3d" in metrics:
        written.append(coherence_histogram(
            np.asarray(metrics["coherence_2d"]), np.asarray(metrics["coherence_3d"]),
            save_path=out_dir / "fig_coherence.png",
        ))
    # 6. Task A (skip if benchmark hasn't run)
    if metrics.get("task_a_confusion"):
        written.append(task_a_confusion(
            {k: np.asarray(v) for k, v in metrics["task_a_confusion"].items()},
            save_path=out_dir / "fig_task_a_confusion.png",
        ))
    # 7. Task B (skip if benchmark hasn't run)
    if metrics.get("task_b_mae"):
        written.append(task_b_mae_bars(
            metrics["task_b_mae"], save_path=out_dir / "fig_task_b_mae.png",
        ))
    return written
