"""Architecture-matched Transformer DT for Paper 1 benchmarks.

The configuration intentionally matches Kanu et al. ICUFN 2026 so 2D-vs-3D-vs-4D
comparisons isolate the effect of state dimensionality from architecture:

    encoder layers : 4
    attention heads: 4
    d_model        : 128
    feedforward dim: 512
    dropout        : 0.1
    optimizer      : Adam, lr=1e-3
    early stop     : on val loss

The only thing that changes between conditions is the input feature width
(2 → 3 → 4); the rest of the model is identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TransformerConfig:
    n_features: int = 3                       # 2D / 3D / 4D set by caller
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    output_dim: int | None = None             # None = same as n_features (Task B)
    seq_len: int = 2                          # input is [s_t1, s_t2]
    seed: int = 42


def _build_model(cfg: TransformerConfig):
    """Build the encoder-only Transformer DT in PyTorch.

    Lazy imports torch so the rest of the package imports without it.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    out_dim = cfg.output_dim if cfg.output_dim is not None else cfg.n_features

    class TransformerDT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_proj = nn.Linear(cfg.n_features, cfg.d_model)
            self.pos_emb = nn.Parameter(
                torch.zeros(1, cfg.seq_len, cfg.d_model)
            )
            nn.init.normal_(self.pos_emb, std=0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.ffn_dim,
                dropout=cfg.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=cfg.n_layers
            )
            # Pool: take the last token's representation as the next-state context.
            self.head = nn.Linear(cfg.d_model, out_dim)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore
            # x: (B, seq_len, n_features)
            h = self.token_proj(x) + self.pos_emb
            h = self.encoder(h)
            return self.head(h[:, -1, :])

    return TransformerDT()


def fit_predict_transformer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    cfg: TransformerConfig,
    *,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
    verbose: bool = False,
) -> dict:
    """Train Transformer DT with early stopping on val loss; predict on test.

    Inputs
    ------
    X_train : (n_train, seq_len, n_features) — context [s_t1, s_t2]
    y_train : (n_train, output_dim) — target s_t3
    X_val, y_val, X_test : analogous

    Returns
    -------
    dict with:
        "predictions": (n_test, output_dim) np.ndarray
        "best_val_loss": float
        "best_epoch": int
        "history": {"train_loss": [...], "val_loss": [...]}
        "n_params": int
    """
    import torch
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    Xt = torch.from_numpy(X_train).float().to(device)
    yt = torch.from_numpy(y_train).float().to(device)
    Xv = torch.from_numpy(X_val).float().to(device)
    yv = torch.from_numpy(y_val).float().to(device)
    Xs = torch.from_numpy(X_test).float().to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    bad = 0

    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    n = Xt.shape[0]

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        total = 0.0
        seen = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xt[idx], yt[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.shape[0]
            seen += xb.shape[0]
        train_loss = total / max(seen, 1)

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)
            val_loss = float(F.mse_loss(val_pred, yv).item())
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(val_loss)
        if verbose:
            print(f"  epoch {ep:3d}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val - 1e-6:
            best_val, best_epoch, bad = val_loss, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(Xs).cpu().numpy()

    return {
        "predictions": preds,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "history": history,
        "n_params": n_params,
    }
