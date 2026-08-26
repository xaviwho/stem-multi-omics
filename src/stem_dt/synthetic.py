"""Synthetic state-tensor generator for pipeline development.

Produces (N, T, D=4) trajectories with realistic-ish DA differentiation
dynamics so validation (§4) and benchmark (§5) code can be built and
unit-tested before real data is wired up.

Per-stage behavior:
    P  decreases monotonically D11 → D52 (pluripotency loss)
    D  increases monotonically D11 → D52 (DA-marker gain)
    Pr lags D (transcript→protein lag)
    Ph approximates a sigmoidal activation peaking around D30

Two knobs for hypothesis testing:
    `pr_signal`, `ph_signal` ∈ [0, 1] — set both to 0 to produce a null
    where Pr and Ph are pure noise (used for the M-F permutation null
    sanity check on Task A).
"""

from __future__ import annotations

import numpy as np


def _stage_means(pr_signal: float, ph_signal: float) -> np.ndarray:
    """Per-stage means for (P, D, Pr, Ph) at D11, D30, D52."""
    return np.array(
        [
            # P,    D,    Pr,                Ph
            [0.85, 0.10, 0.05 * pr_signal,   0.10 * ph_signal],   # D11
            [0.40, 0.55, 0.45 * pr_signal,   0.85 * ph_signal],   # D30 (Ph peak)
            [0.10, 0.90, 0.85 * pr_signal,   0.55 * ph_signal],   # D52
        ],
        dtype=np.float32,
    )


def generate_synthetic_state_tensor(
    n: int = 200,
    stages: tuple[str, ...] = ("D11", "D30", "D52"),
    pr_signal: float = 1.0,
    ph_signal: float = 1.0,
    noise_std: float = 0.08,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Generate (N, T, 4) ∈ [0, 1] state tensor with optional Pr/Ph nulling.

    Parameters
    ----------
    pr_signal, ph_signal
        0.0 = the dimension is pure noise (no stage signal); 1.0 = full
        signal. Used for permutation-null checks (M-F) — set to 0 for the
        Pr/Ph dims to confirm the 4D model loses its lift over 2D.
    noise_std
        Per-cell Gaussian noise around the stage mean; clipped to [0, 1].

    Returns
    -------
    tensor : np.ndarray of shape (N, T, 4)
    stage_order : list[str]
    """
    if len(stages) != 3:
        raise ValueError("synthetic generator hard-codes T=3 stage means")
    rng = np.random.default_rng(seed)
    means = _stage_means(pr_signal, ph_signal)             # (T, 4)
    tensor = means[None, :, :] + rng.normal(
        0.0, noise_std, size=(n, len(stages), 4)
    ).astype(np.float32)
    np.clip(tensor, 0.0, 1.0, out=tensor)
    return tensor, list(stages)


def permute_dims(
    tensor: np.ndarray, dims_to_permute: tuple[int, ...] = (2, 3), seed: int = 42
) -> np.ndarray:
    """Shuffle the listed dims across cells *within each timepoint*.

    Implements the M-F permutation null: if Pr/Ph (dims 2, 3) carry real
    stage signal, shuffling them across cells at each stage destroys the
    cell-level association while preserving the marginal distribution.
    Task A trained on the permuted tensor should perform like the 2D
    baseline; if it doesn't, the 4D advantage is from added capacity,
    not added information.
    """
    out = tensor.copy()
    rng = np.random.default_rng(seed)
    for t in range(out.shape[1]):
        for d in dims_to_permute:
            perm = rng.permutation(out.shape[0])
            out[:, t, d] = out[perm, t, d]
    return out
