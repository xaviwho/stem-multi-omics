"""State-space validation: cross-modal correlation, MOFA+, trajectory coherence.

Methodology cross-reference: §4.1–§4.3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# §4.1 — Cross-modal correlation analysis.
# ---------------------------------------------------------------------------

def _corr_matrix(x: np.ndarray, method: str) -> np.ndarray:
    """Pairwise correlation matrix for an (N, D) array."""
    from scipy.stats import pearsonr, spearmanr

    n_dims = x.shape[1]
    out = np.eye(n_dims)
    if method == "spearman":
        # spearmanr returns the full matrix when given an (N, D) array with axis=0.
        rho, _ = spearmanr(x, axis=0)
        if np.isscalar(rho):  # D=2 case returns a scalar
            out[0, 1] = out[1, 0] = float(rho)
            return out
        return np.asarray(rho)
    if method == "pearson":
        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                r, _ = pearsonr(x[:, i], x[:, j])
                out[i, j] = out[j, i] = float(r)
        return out
    raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")


def cross_modal_correlations(
    state_tensor: np.ndarray,
    stage_labels: list[str],
    method: str = "spearman",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, np.ndarray]]:
    """Pairwise correlations among state dims per timepoint, with bootstrap CIs.

    Parameters
    ----------
    state_tensor : np.ndarray, shape (N, T, D)
    stage_labels : list[str], length T
    method : "spearman" | "pearson"
    n_bootstrap : int
        0 disables bootstrapping; otherwise computes percentile 95% CIs from
        `n_bootstrap` resamples of cells with replacement.

    Returns
    -------
    dict {stage: {"rho": (D,D), "ci_low": (D,D), "ci_high": (D,D)}}
        ci_low / ci_high are present only when n_bootstrap > 0.
    """
    if state_tensor.shape[1] != len(stage_labels):
        raise ValueError("len(stage_labels) must equal state_tensor.shape[1]")

    rng = np.random.default_rng(seed)
    n_cells = state_tensor.shape[0]
    out: dict[str, dict[str, np.ndarray]] = {}

    for t_idx, stage in enumerate(stage_labels):
        x = state_tensor[:, t_idx, :]
        rho = _corr_matrix(x, method)
        entry: dict[str, np.ndarray] = {"rho": rho}

        if n_bootstrap > 0:
            samples = np.empty((n_bootstrap, *rho.shape), dtype=np.float64)
            for b in range(n_bootstrap):
                idx = rng.integers(0, n_cells, size=n_cells)
                samples[b] = _corr_matrix(x[idx], method)
            entry["ci_low"] = np.percentile(samples, 2.5, axis=0)
            entry["ci_high"] = np.percentile(samples, 97.5, axis=0)

        out[stage] = entry

    return out


# ---------------------------------------------------------------------------
# §4.2 — MOFA+ variance decomposition.
# ---------------------------------------------------------------------------

def fit_mofa(
    score_tensor: np.ndarray,
    stage_labels: list[str],
    feature_labels: tuple[str, ...] = ("P", "D", "Pr"),
    n_factors: int = 5,
    seed: int = 42,
) -> dict:
    """Fit MOFA+ across modalities, grouped by stage (§4.2).

    Each feature dimension (P, D, Pr, ...) is treated as a separate "view"
    of size 1 — MOFA+ then decomposes per-modality vs. shared variance.
    With only 1 feature per view, MOFA+ degenerates to a structured-PCA;
    that's still useful here because it gives the modality-resolved variance
    breakdown the methodology asks for, even though we don't have many
    proteins per cell yet.

    Parameters
    ----------
    score_tensor : (N, T, D) np.ndarray
        State tensor; cells × stages × features.
    stage_labels : list[str], length T
        Group labels (one per stage along axis 1).
    feature_labels : tuple[str, ...], length D
        View labels along axis 2.
    n_factors : int
        K = 5 by default per methodology §4.2 (elbow criterion).

    Returns
    -------
    dict
        {
          "variance_explained": {modality_label: ndarray of shape (n_factors,)
                                 with % values, per-modality across all groups},
          "factor_scores":      ndarray (N * T, n_factors),
          "n_cells_total":      int,
          "stages":             list[str],
          "modalities":         list[str],
        }

    Notes
    -----
    `mofapy2` is imported lazily so the rest of this module imports cleanly
    without it. Install with `pip install mofapy2`.
    """
    if score_tensor.ndim != 3:
        raise ValueError(f"expected 3D tensor, got shape {score_tensor.shape}")
    n_traj, n_stages, n_feats = score_tensor.shape
    if n_stages != len(stage_labels):
        raise ValueError("stage_labels length must match axis 1")
    if n_feats != len(feature_labels):
        raise ValueError("feature_labels length must match axis 2")

    try:
        from mofapy2.run.entry_point import entry_point
    except ImportError as e:
        raise RuntimeError(
            "mofapy2 not installed. Install with `pip install mofapy2`."
        ) from e

    # Build the long-format DataFrame MOFA+ expects: rows = (sample, group, view, feature).
    import pandas as pd
    rows: list[dict] = []
    for traj in range(n_traj):
        for t_idx, stage in enumerate(stage_labels):
            for f_idx, feat in enumerate(feature_labels):
                rows.append({
                    "sample":  f"traj_{traj}_{stage}",
                    "group":   stage,
                    "view":    feat,                       # one feature per view
                    "feature": feat,                       # placeholder name
                    "value":   float(score_tensor[traj, t_idx, f_idx]),
                })
    long_df = pd.DataFrame(rows)

    ent = entry_point()
    ent.set_data_options(scale_views=False, scale_groups=False)
    ent.set_data_df(long_df)
    ent.set_model_options(factors=n_factors, spikeslab_weights=True, ard_weights=True)
    ent.set_train_options(
        iter=500, convergence_mode="medium", seed=seed, verbose=False,
    )
    ent.build()
    ent.run()

    # Extract variance explained: shape (n_factors, n_views, n_groups).
    # Aggregate across groups by summing then re-normalize against per-modality totals.
    r2_arr = ent.model.calculate_variance_explained(total=False)
    # MOFA+ returns a list-of-arrays-per-group; stack to (n_groups, n_factors, n_views).
    r2_by_group = np.stack([np.asarray(g) for g in r2_arr], axis=0)
    if r2_by_group.shape[1] != n_factors:
        # Some MOFA versions return (n_views, n_factors); normalize layout.
        r2_by_group = r2_by_group.transpose(0, 2, 1)
    # Per-modality average across stages; values are fractions, multiply 100 → %.
    per_view_mean = r2_by_group.mean(axis=0) * 100.0     # (n_factors, n_views)
    variance_explained = {
        view: per_view_mean[:, i] for i, view in enumerate(feature_labels)
    }

    # Factor scores: MOFA+ returns per-group; concatenate them in stage order.
    z_per_group = ent.model.getExpectations()["Z"]["E"]  # list of (n_samples_in_group, n_factors)
    factor_scores = np.concatenate(z_per_group, axis=0)

    return {
        "variance_explained": variance_explained,
        "factor_scores": factor_scores,
        "n_cells_total": int(factor_scores.shape[0]),
        "stages": list(stage_labels),
        "modalities": list(feature_labels),
    }


# ---------------------------------------------------------------------------
# §4.3 — Trajectory coherence.
# ---------------------------------------------------------------------------

def trajectory_coherence(state_tensor: np.ndarray) -> np.ndarray:
    """Cosine similarity of consecutive displacement vectors.

    For T timepoints, returns mean cosine over the T-2 consecutive pairs.
    With T=3 (the master's-thesis convention) this collapses to a single
    cosine: cos(s_1 - s_0, s_2 - s_1).

    NOTE(M-B): the methodology had `(1/2) × cos(...)` with one cosine,
    which gave range [-0.5, 0.5] inconsistent with the stated [-1, 1]
    interpretation. Implemented here as the unscaled cosine.

    Parameters
    ----------
    state_tensor : np.ndarray, shape (N, T, D), T ≥ 3

    Returns
    -------
    coherence : np.ndarray, shape (N,) in [-1, 1]
    """
    if state_tensor.shape[1] < 3:
        raise ValueError("trajectory_coherence requires at least 3 timepoints")

    disp = np.diff(state_tensor, axis=1)            # (N, T-1, D)
    a = disp[:, :-1, :]                             # (N, T-2, D)
    b = disp[:, 1:, :]                              # (N, T-2, D)

    num = (a * b).sum(axis=-1)                      # (N, T-2)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    den = np.where(den == 0, 1.0, den)              # avoid /0; cos undefined → 0
    cos = num / den
    return cos.mean(axis=1)
