"""Tasks A & B benchmark suite for Paper 1.

Both tasks are evaluated under three conditions:
    "2D"       : (P, D)
    "3D"       : (P, D, Pr)         -- the multi-omics extension
    "3D_perm"  : (P, D, Pr_shuffled) -- M-F permutation null on the Pr dim

Plus optional "4D" if a K dim is appended later (sensitivity ablation).

Models trained per task per condition:
    Task A (stage classification, 3-class):
        LogisticRegression, RandomForest, Transformer encoder
    Task B (trajectory prediction, [s_11, s_30] -> s_52):
        Ridge, RandomForest, Transformer DT

The same train/val/test split (140/30/30, seed=42) is used everywhere.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from . import scores
from .transformer import TransformerConfig, fit_predict_transformer


# ---------------------------------------------------------------------------
# Dataclasses for structured outputs.
# ---------------------------------------------------------------------------

@dataclass
class StageClassificationResult:
    model_name: str
    condition: str                 # "2D" / "3D" / "3D_perm" / "4D"
    macro_f1: float
    balanced_accuracy: float
    confusion: list[list[int]]     # row=true, col=predicted
    n_features: int


@dataclass
class TrajectoryPredictionResult:
    model_name: str
    condition: str
    mae_overall: float
    mae_per_dim: dict[str, float]
    n_features: int


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _slice_features(state: np.ndarray, dims: tuple[int, ...]) -> np.ndarray:
    return state[:, :, list(dims)].astype(np.float32)


def _flatten_for_classifier(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell-per-stage classification: each stage of each trajectory is a sample.

    Returns (X, y) where X has shape (N*T, D) and y is integer-encoded stage.
    """
    n, t, d = state.shape
    X = state.reshape(n * t, d)
    y = np.repeat(np.arange(t), n)        # 0/1/2 for stages along axis 1 (T-major)
    # The above repeats across cells per stage — but state is (cell, stage, feat),
    # so reshaping cell-major. Recompute y to match cell-major order:
    y = np.tile(np.arange(t), n)
    return X.astype(np.float32), y


def _permute_pr_within_stage(state: np.ndarray, pr_dim: int = 2, seed: int = 42) -> np.ndarray:
    """Within-stage permutation null: shuffle Pr among cells of the same stage.

    Preserves per-stage marginal distribution; destroys per-cell association
    between Pr and the rest of the state. Tests: "does Pr carry CELL-LEVEL
    information beyond what its stage already implies?" If the model retains
    its lift over 2D under this null, the cell-level Pr signal is the lift's
    source. This is the appropriate null for **Task B** (trajectory prediction)
    where per-cell Pr values inform per-cell future-state forecasts.

    NOT the right null for Task A (stage classification): within-stage shuffle
    leaves the stage-Pr relationship intact, so a model learning "Pr value
    range → stage" continues to work. Use ``_permute_pr_across_stages`` for
    that test.
    """
    out = state.copy()
    rng = np.random.default_rng(seed)
    for t in range(out.shape[1]):
        perm = rng.permutation(out.shape[0])
        out[:, t, pr_dim] = out[perm, t, pr_dim]
    return out


def _permute_pr_across_stages(state: np.ndarray, pr_dim: int = 2, seed: int = 42) -> np.ndarray:
    """Across-stage permutation null: shuffle Pr globally across all cell-stages.

    Destroys both the cell-level association AND the stage-Pr relationship,
    leaving Pr as a stage-uninformative random feature. Tests: "does Pr
    carry STAGE-DISCRIMINATING signal at all?" If 3D_perm_global drops back
    to 2D performance, the 3D lift comes from real Pr-stage coupling. This
    is the appropriate null for **Task A** (stage classification).
    """
    out = state.copy()
    rng = np.random.default_rng(seed)
    n_traj, n_stages, _ = out.shape
    flat = out[:, :, pr_dim].reshape(-1)
    flat = flat[rng.permutation(flat.size)]
    out[:, :, pr_dim] = flat.reshape(n_traj, n_stages)
    return out


# ---------------------------------------------------------------------------
# Task A: stage classification (LR / RF / Transformer).
# ---------------------------------------------------------------------------

def _stage_classification_one(
    name: str,
    state: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    condition: str,
    seed: int = 42,
) -> list[StageClassificationResult]:
    """Train and evaluate the three Task-A models on a single state tensor."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

    n_traj, n_stages, n_feat = state.shape

    # Use cells from training/test trajectories at all stages.
    def split(idx):
        s = state[idx]                          # (k, T, D)
        X = s.reshape(-1, n_feat)
        y = np.tile(np.arange(n_stages), len(idx))
        return X.astype(np.float32), y

    Xtr, ytr = split(train_idx)
    Xva, yva = split(val_idx)
    Xte, yte = split(test_idx)

    out: list[StageClassificationResult] = []
    classes = list(range(n_stages))

    # Logistic Regression
    lr = LogisticRegression(max_iter=2000, random_state=seed)
    lr.fit(Xtr, ytr)
    pred_lr = lr.predict(Xte)
    out.append(StageClassificationResult(
        model_name="LogisticRegression", condition=condition,
        macro_f1=float(f1_score(yte, pred_lr, average="macro", labels=classes, zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(yte, pred_lr)),
        confusion=confusion_matrix(yte, pred_lr, labels=classes).tolist(),
        n_features=n_feat,
    ))

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, random_state=seed, n_jobs=-1,
    )
    rf.fit(Xtr, ytr)
    pred_rf = rf.predict(Xte)
    out.append(StageClassificationResult(
        model_name="RandomForest", condition=condition,
        macro_f1=float(f1_score(yte, pred_rf, average="macro", labels=classes, zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(yte, pred_rf)),
        confusion=confusion_matrix(yte, pred_rf, labels=classes).tolist(),
        n_features=n_feat,
    ))

    # Transformer (treats each cell as a single-token sequence for classification).
    # Built on top of the regression-ready Transformer but with a softmax head
    # via cross-entropy loss; reuse the encoder body with seq_len=1.
    cfg = TransformerConfig(
        n_features=n_feat, output_dim=n_stages, seq_len=1, seed=seed,
    )
    res = _fit_predict_classifier_transformer(
        Xtr, ytr, Xva, yva, Xte, cfg, n_classes=n_stages,
    )
    pred_tx = res["predictions"]
    out.append(StageClassificationResult(
        model_name="Transformer", condition=condition,
        macro_f1=float(f1_score(yte, pred_tx, average="macro", labels=classes, zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(yte, pred_tx)),
        confusion=confusion_matrix(yte, pred_tx, labels=classes).tolist(),
        n_features=n_feat,
    ))
    return out


def _fit_predict_classifier_transformer(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, cfg: TransformerConfig, n_classes: int,
    epochs: int = 100, batch_size: int = 32, patience: int = 10,
) -> dict:
    """Cross-entropy variant of the Transformer for Task A.

    Inputs are (n, n_features); we reshape to (n, 1, n_features) so the
    encoder sees a single-token sequence (consistent with the model body's
    seq_len=1 contract).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from .transformer import _build_model

    cfg = TransformerConfig(**{**asdict(cfg), "output_dim": n_classes, "seq_len": 1})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    Xtr = torch.from_numpy(X_train).float().unsqueeze(1).to(device)
    ytr = torch.from_numpy(y_train).long().to(device)
    Xva = torch.from_numpy(X_val).float().unsqueeze(1).to(device)
    yva = torch.from_numpy(y_val).long().to(device)
    Xs  = torch.from_numpy(X_test).float().unsqueeze(1).to(device)

    best_val = float("inf"); bad = 0
    best_state = None
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    n = Xtr.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = model(Xtr[idx])
            loss = F.cross_entropy(logits, ytr[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(Xva)
            val_loss = float(F.cross_entropy(val_logits, yva).item())
        if val_loss < best_val - 1e-6:
            best_val = val_loss; bad = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(Xs).argmax(dim=-1).cpu().numpy()
    return {"predictions": preds, "best_val_loss": best_val}


def run_stage_classification(
    state: np.ndarray,
    stage_labels: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    feature_labels: tuple[str, ...] = ("P", "D", "Pr"),
    seed: int = 42,
    run_permutation_null: bool = True,
    run_pr_only: bool = False,
) -> list[StageClassificationResult]:
    """Run Task A across 2D, 3D, and (if requested) Pr-permuted conditions.

    Permutation conditions:
      - "3D_perm_within": Pr shuffled within each stage (cell-level association
        destroyed; stage-Pr relationship preserved). Tests cell-level Pr signal.
      - "3D_perm_across": Pr shuffled globally across all cell-stages
        (stage-Pr relationship destroyed). Tests stage-discriminating Pr signal.
        This is the **right null for Task A**.

    `run_pr_only=True` adds a "Pr_only" condition that trains on the single
    Pr column. Used to make explicit where the Task A signal lives: if Pr-only
    matches 3D performance, then for *classification* the multi-omics framing
    is doing no more than a single monotonic feature — the multidimensional
    value of the 3D state must come from Task B / orthogonality, not Task A
    capacity. Reporting this volunteers the decomposition a sharp reviewer
    would derive anyway.
    """
    out: list[StageClassificationResult] = []
    out += _stage_classification_one(
        "2D", _slice_features(state, (0, 1)),
        train_idx, val_idx, test_idx, condition="2D", seed=seed,
    )
    if state.shape[2] >= 3:
        s3 = _slice_features(state, (0, 1, 2))
        out += _stage_classification_one(
            "3D", s3,
            train_idx, val_idx, test_idx, condition="3D", seed=seed,
        )
        if run_pr_only:
            out += _stage_classification_one(
                "Pr_only", _slice_features(state, (2,)),
                train_idx, val_idx, test_idx, condition="Pr_only", seed=seed,
            )
        if run_permutation_null:
            out += _stage_classification_one(
                "3D_perm_within",
                _permute_pr_within_stage(s3, pr_dim=2, seed=seed),
                train_idx, val_idx, test_idx,
                condition="3D_perm_within", seed=seed,
            )
            out += _stage_classification_one(
                "3D_perm_across",
                _permute_pr_across_stages(s3, pr_dim=2, seed=seed),
                train_idx, val_idx, test_idx,
                condition="3D_perm_across", seed=seed,
            )
    if state.shape[2] >= 4:
        out += _stage_classification_one(
            "4D", _slice_features(state, (0, 1, 2, 3)),
            train_idx, val_idx, test_idx, condition="4D", seed=seed,
        )
    return out


# ---------------------------------------------------------------------------
# Task B: trajectory prediction (Ridge / RF / Transformer DT).
# ---------------------------------------------------------------------------

def _trajectory_prediction_one(
    state: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    condition: str,
    feature_labels: tuple[str, ...],
    seed: int = 42,
) -> tuple[list[TrajectoryPredictionResult], np.ndarray]:
    """Train Ridge / RF / Transformer to predict s_t3 from [s_t1, s_t2].

    Returns the model results AND the Transformer's per-test-trajectory
    prediction array (used as input to the hero figure).
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    n_traj, n_stages, n_feat = state.shape
    if n_stages != 3:
        raise ValueError("Task B assumes 3 stages: D11, D30, D52")

    def assemble(idx):
        s = state[idx]                          # (k, 3, D)
        X = s[:, :2, :].reshape(len(idx), -1)   # (k, 2*D)
        y = s[:, 2, :]                          # (k, D)
        return X.astype(np.float32), y.astype(np.float32)

    Xtr, ytr = assemble(train_idx)
    Xva, yva = assemble(val_idx)
    Xte, yte = assemble(test_idx)

    out: list[TrajectoryPredictionResult] = []
    feat_keys = feature_labels[:n_feat]

    # Ridge
    ridge = Ridge(alpha=1.0, random_state=seed)
    ridge.fit(Xtr, ytr)
    pred_ridge = ridge.predict(Xte)
    out.append(_score_pred(pred_ridge, yte, "Ridge", condition, feat_keys, n_feat))

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    rf.fit(Xtr, ytr)
    pred_rf = rf.predict(Xte)
    out.append(_score_pred(pred_rf, yte, "RandomForest", condition, feat_keys, n_feat))

    # Transformer DT (sequence input)
    Xtr_seq = state[train_idx][:, :2, :].astype(np.float32)
    Xva_seq = state[val_idx][:, :2, :].astype(np.float32)
    Xte_seq = state[test_idx][:, :2, :].astype(np.float32)
    cfg = TransformerConfig(n_features=n_feat, output_dim=n_feat, seq_len=2, seed=seed)
    tx_res = fit_predict_transformer(
        Xtr_seq, ytr, Xva_seq, yva, Xte_seq, cfg,
        epochs=100, batch_size=32, patience=10,
    )
    pred_tx = tx_res["predictions"]
    out.append(_score_pred(pred_tx, yte, "Transformer", condition, feat_keys, n_feat))

    return out, pred_tx


def _score_pred(
    pred: np.ndarray, y: np.ndarray, name: str, condition: str,
    feat_keys: tuple[str, ...], n_feat: int,
) -> TrajectoryPredictionResult:
    err = np.abs(pred - y)
    return TrajectoryPredictionResult(
        model_name=name, condition=condition,
        mae_overall=float(err.mean()),
        mae_per_dim={feat_keys[d]: float(err[:, d].mean()) for d in range(n_feat)},
        n_features=n_feat,
    )


def run_trajectory_prediction(
    state: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    feature_labels: tuple[str, ...] = ("P", "D", "Pr"),
    seed: int = 42,
    run_permutation_null: bool = True,
) -> tuple[list[TrajectoryPredictionResult], dict[str, np.ndarray]]:
    """Run Task B across 2D, 3D, 3D-permuted conditions.

    Returns (results, predictions_by_condition) where predictions are
    Transformer outputs at D52 — the hero figure expects this shape.
    """
    results: list[TrajectoryPredictionResult] = []
    preds: dict[str, np.ndarray] = {}

    res, p = _trajectory_prediction_one(
        _slice_features(state, (0, 1)),
        train_idx, val_idx, test_idx, "2D",
        feature_labels=feature_labels, seed=seed,
    )
    results += res; preds["2D"] = p

    if state.shape[2] >= 3:
        res, p = _trajectory_prediction_one(
            _slice_features(state, (0, 1, 2)),
            train_idx, val_idx, test_idx, "3D",
            feature_labels=feature_labels, seed=seed,
        )
        results += res; preds["3D"] = p

        if run_permutation_null:
            perm_state = _permute_pr_within_stage(
                _slice_features(state, (0, 1, 2)), pr_dim=2, seed=seed,
            )
            res, p = _trajectory_prediction_one(
                perm_state, train_idx, val_idx, test_idx, "3D_perm",
                feature_labels=feature_labels, seed=seed,
            )
            results += res; preds["3D_perm"] = p

    if state.shape[2] >= 4:
        res, p = _trajectory_prediction_one(
            _slice_features(state, (0, 1, 2, 3)),
            train_idx, val_idx, test_idx, "4D",
            feature_labels=feature_labels, seed=seed,
        )
        results += res; preds["4D"] = p

    return results, preds


# ---------------------------------------------------------------------------
# Inference profiling (single-sample latency, model size).
# ---------------------------------------------------------------------------

def profile_inference_latency(predict_fn, sample, n_runs: int = 100) -> float:
    """Median wall-clock latency over n_runs single-sample predictions, in ms."""
    timings: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predict_fn(sample)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(timings))
