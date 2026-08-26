"""Score construction: P_t, D_t, Pr_t, Ph_t and leakage-safe min-max scaling.

Methodology cross-reference: §2.1–§2.5, §1.4 Step 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-cell raw score functions (mean of log-transformed values over markers).
# Inputs are expected already log-transformed at the modality level; these
# functions just compute the marker-set mean (§2.1–§2.4).
# ---------------------------------------------------------------------------

def _marker_mean(matrix: pd.DataFrame, markers: list[str]) -> pd.Series:
    """Mean over the subset of `markers` actually present as columns of `matrix`.

    Missing markers are dropped silently and counted toward the unmapped-marker
    log emitted by the pipeline driver (see §1.4 Step 1).
    """
    present = [m for m in markers if m in matrix.columns]
    if not present:
        return pd.Series(np.nan, index=matrix.index)
    return matrix[present].mean(axis=1)


def pluripotency_score(log_expr: pd.DataFrame, markers: list[str]) -> pd.Series:
    """P_t^raw — mean log1p(expression) over G_P. §2.1."""
    return _marker_mean(log_expr, markers)


def dopaminergic_score(log_expr: pd.DataFrame, markers: list[str]) -> pd.Series:
    """D_t^raw — mean log1p(expression) over G_D. §2.2."""
    return _marker_mean(log_expr, markers)


def proteomics_score(log_abundance: pd.DataFrame, markers: list[str]) -> pd.Series:
    """Pr_t^raw — mean log2(1+abundance) over G_Pr. §2.3."""
    return _marker_mean(log_abundance, markers)


# ---------------------------------------------------------------------------
# Pr_t score per stage from a ProteomicTimeCourse, with graceful handling
# of missing markers + per-stage replicate aggregation. Extends §2.3.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StagePrScore:
    """Per-stage `Pr_t` summary statistic, computed across CTRL replicates."""

    stage: str
    mean: float
    std: float
    n_replicates: int
    markers_found: tuple[str, ...]
    markers_missing: tuple[str, ...]


def per_stage_pr_score(
    timecourse,
    markers: list[str],
    stages: tuple[str, ...] = ("D11", "D30", "D52"),
    exclude_samples: tuple[str, ...] = (),
) -> dict[str, StagePrScore]:
    """Compute Pr_t per stage from a `ProteomicTimeCourse`.

    For each replicate sample at each stage, computes
    ``mean(log2(1 + LFQ_intensity))`` over markers present in `abundance`.
    Markers absent from `abundance.index` are dropped with audit.

    `exclude_samples` removes the named CTRL sample IDs from the per-stage
    mean / std computation — used by the LORO (leave-one-replicate-out)
    robustness probe (§5.1) to verify Task A generalizes to a replicate that
    never contributed to the stage mean against which it is classified.

    Returns
    -------
    dict {stage: StagePrScore}
        Empty stages (no replicates / no markers found) get
        ``StagePrScore(stage, mean=nan, std=nan, n_replicates=0, ...)``.
    """
    abundance = timecourse.abundance
    samples = timecourse.samples

    found = [m for m in markers if m in abundance.index]
    missing = tuple(m for m in markers if m not in abundance.index)
    exclude = set(exclude_samples)

    out: dict[str, StagePrScore] = {}
    for stage in stages:
        rep_ids = [s for s in samples.index[samples["stage"] == stage].tolist()
                   if s not in exclude]
        if not rep_ids or not found:
            out[stage] = StagePrScore(
                stage=stage, mean=float("nan"), std=float("nan"),
                n_replicates=len(rep_ids), markers_found=tuple(found),
                markers_missing=missing,
            )
            continue
        # Per-replicate score: mean log2(1 + abundance) across markers found.
        per_rep = np.log2(1.0 + abundance.loc[found, rep_ids]).mean(axis=0)
        out[stage] = StagePrScore(
            stage=stage,
            mean=float(per_rep.mean()),
            std=float(per_rep.std(ddof=1)) if len(per_rep) > 1 else 0.0,
            n_replicates=len(rep_ids),
            markers_found=tuple(found),
            markers_missing=missing,
        )
    return out


# ---------------------------------------------------------------------------
# 3D state-tensor assembly: extend ICUFN's (200, 3, 2) (P, D) baseline with
# a Pr column. Per-trajectory Pr is sampled from the empirical CTRL
# replicate distribution at the matched stage (mean ± SD as Gaussian).
# ---------------------------------------------------------------------------

def append_stage_feature(
    state: np.ndarray,
    feature_per_stage: dict[str, "StagePrScore"],
    stage_order: tuple[str, ...] = ("D11", "D30", "D52"),
    seed: int = 42,
    floor_at: tuple[str, ...] = (),
) -> np.ndarray:
    """Append one per-stage feature column to an existing (N, T, D) state.

    Used for the K_t ablation: extends an existing 3D `(P, D, Pr)` state to
    4D `(P, D, Pr, K)` without reshuffling the lower dimensions. Per-cell
    feature values are sampled from N(stage_mean, stage_std) per the same
    convention as :func:`assemble_3d_state`.
    """
    if state.shape[1] != len(stage_order):
        raise ValueError(
            f"state.shape[1] ({state.shape[1]}) != len(stage_order) ({len(stage_order)})"
        )
    n_traj = state.shape[0]
    rng = np.random.default_rng(seed)
    new_col = np.empty((n_traj, len(stage_order)), dtype=np.float32)

    for t_idx, stage in enumerate(stage_order):
        rec = feature_per_stage.get(stage)
        if rec is None or not np.isfinite(rec.mean):
            new_col[:, t_idx] = 0.0
            continue
        if stage in floor_at:
            mean, std = 0.0, max(rec.std / 2.0, 1e-6)
        else:
            mean, std = rec.mean, max(rec.std, 1e-6)
        new_col[:, t_idx] = rng.normal(mean, std, size=n_traj).astype(np.float32)

    return np.concatenate(
        [state.astype(np.float32), new_col[:, :, None]], axis=-1
    )


def assemble_3d_state(
    icufn_state: np.ndarray,
    pr_per_stage: dict[str, "StagePrScore"],
    stage_order: tuple[str, ...] = ("D11", "D30", "D52"),
    seed: int = 42,
    pr_floor_at: tuple[str, ...] = ("D11",),
    deterministic: bool = False,
) -> np.ndarray:
    """Build the (N, T, 3) state tensor `(P, D, Pr)`.

    Parameters
    ----------
    icufn_state : (N, T, 2)
        Published `(P, D)` from `data.load_icufn_baseline().state`.
    pr_per_stage : output of :func:`per_stage_pr_score`
    stage_order
        Stages along axis 1 of `icufn_state`.
    seed : int
        Per-trajectory Pr is sampled from N(stage_mean, stage_std) so that
        cell-level Pr has biologically-plausible variance even though
        Novak's per-stage n=2.
    pr_floor_at
        Stages at which Pr is set to a biological-zero floor (D11 by default,
        since DA markers are not yet expressed at floor-plate stage). The
        floor is encoded as `mean = 0, std = stage_std / 2` so per-cell
        sampling produces a tight bunch near zero rather than a single
        constant — preserves Pr-axis variance for downstream analysis.

    Returns
    -------
    (N, T, 3) state tensor with `Pr_t` appended as the third feature.
    Pr values are NOT yet min-max normalized to [0, 1]; apply
    :class:`MinMaxBounds` afterwards (or rely on the leakage-safe pipeline).

    `deterministic=True` assigns the per-stage mean to every cell at that
    stage (zero per-cell variance). Used by the deterministic-Pr ablation
    to test whether per-cell Gaussian sampling adds anything beyond the
    stage-level signal.
    """
    if icufn_state.shape[1] != len(stage_order):
        raise ValueError(
            f"icufn_state.shape[1] ({icufn_state.shape[1]}) != "
            f"len(stage_order) ({len(stage_order)})"
        )
    n_traj = icufn_state.shape[0]
    rng = np.random.default_rng(seed)
    pr_col = np.empty((n_traj, len(stage_order)), dtype=np.float32)

    for t_idx, stage in enumerate(stage_order):
        rec = pr_per_stage.get(stage)
        if rec is None or not np.isfinite(rec.mean):
            # No data → flat zero column for this stage. Audit-loggable.
            pr_col[:, t_idx] = 0.0
            continue
        if stage in pr_floor_at:
            # Biological-zero floor: tight distribution near zero.
            mean, std = 0.0, max(rec.std / 2.0, 1e-6)
        else:
            mean, std = rec.mean, max(rec.std, 1e-6)
        if deterministic:
            pr_col[:, t_idx] = mean
        else:
            pr_col[:, t_idx] = rng.normal(mean, std, size=n_traj).astype(np.float32)

    # Stack: (N, T, 3) = concat([(N,T,2), (N,T,1)])
    return np.concatenate(
        [icufn_state.astype(np.float32), pr_col[:, :, None]], axis=-1
    )


def phospho_score(
    log_phospho: pd.DataFrame, sites: list[tuple[str, str]]
) -> pd.Series:
    """Ph_t^raw — mean log2(1+intensity) over G_Ph. §2.4.

    `sites` is a list of (gene_symbol, residue); `log_phospho` columns must
    be encoded as ``f"{gene}_{residue}"`` to match.
    """
    site_keys = [f"{g}_{r}" for g, r in sites]
    return _marker_mean(log_phospho, site_keys)


# ---------------------------------------------------------------------------
# Leakage-safe min-max scaling fitted on training trajectories only (§1.4 Step 4).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MinMaxBounds:
    """Per-dimension min/max fitted on the training split."""

    lo: np.ndarray  # shape (D,)
    hi: np.ndarray  # shape (D,)

    def apply(self, x: np.ndarray) -> np.ndarray:
        denom = np.where(self.hi - self.lo == 0, 1.0, self.hi - self.lo)
        return np.clip((x - self.lo) / denom, 0.0, 1.0)


def fit_minmax(train_scores: np.ndarray) -> MinMaxBounds:
    """Fit min-max bounds on the training-split raw scores.

    `train_scores` shape: (n_train_trajectories, n_timepoints, n_dims).
    Bounds are computed across the (trajectory, timepoint) axes so that one
    set of (lo, hi) is shared across stages — consistent with the master's
    thesis convention.
    """
    flat = train_scores.reshape(-1, train_scores.shape[-1])
    return MinMaxBounds(lo=flat.min(axis=0), hi=flat.max(axis=0))


# ---------------------------------------------------------------------------
# Trajectory assembly (§2.5).
# ---------------------------------------------------------------------------

def assemble_trajectories(
    scores_by_stage: dict[str, pd.DataFrame],
    n_trajectories: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Sample one cell per stage per trajectory.

    Parameters
    ----------
    scores_by_stage : dict[stage_label, DataFrame]
        Each DataFrame is (cells × dims), rows = per-cell scaled scores at that stage.
    n_trajectories : int
        N (default 200, master's thesis convention).
    seed : int
        Sampling seed; together with `n_trajectories` defines the trajectory tensor.

    Returns
    -------
    tensor : np.ndarray of shape (N, T, D)
    stage_order : list[str]
        Stage labels in the order they index axis 1 of `tensor`.
    """
    rng = np.random.default_rng(seed)
    stage_order = list(scores_by_stage.keys())  # caller controls order
    n_dims = next(iter(scores_by_stage.values())).shape[1]
    tensor = np.empty((n_trajectories, len(stage_order), n_dims), dtype=np.float32)
    for t_idx, stage in enumerate(stage_order):
        df = scores_by_stage[stage]
        idx = rng.integers(0, len(df), size=n_trajectories)
        tensor[:, t_idx, :] = df.iloc[idx].to_numpy()
    return tensor, stage_order


def split_trajectories(
    n: int = 200, splits: tuple[int, int, int] = (140, 30, 30), seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the master's thesis 140/30/30 split with seed=42."""
    if sum(splits) != n:
        raise ValueError(f"splits {splits} must sum to {n}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr, n_va, _ = splits
    return perm[:n_tr], perm[n_tr : n_tr + n_va], perm[n_tr + n_va :]
