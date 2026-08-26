"""K_t kinase-pathway activity inference from transcriptomic data.

Methodology cross-reference: §1.3 (PSP-curated G_Ph) + §2.4 (revised Path A.1
"phospho-pathway activity score" instead of measured phosphoproteomics).

Approach
--------
Given a per-cell transcriptome (Jerber 2021 scRNA-seq), we estimate the
*activity* of each kinase upstream of a G_Ph phosphosite by scoring the
expression of that kinase's known transcriptional targets. Active kinases
drive characteristic transcriptional programs that are visible in the
transcriptome even when phosphorylation itself is not measured.

Two backends are supported, in priority order:

1. **decoupler-py + PROGENy** -- canonical pathway-activity inference.
   PROGENy provides a pre-trained model of 14 signaling pathways
   (MAPK/ERK, PI3K/AKT, etc.) calibrated on perturbation data. We map each
   G_Ph phosphosite to its upstream pathway and use the PROGENy score.

2. **Z-score over a curated kinase-target gene set** -- fallback when
   `decoupler` isn't installed. Each kinase is associated with a small
   target-gene set; per-cell activity is the mean z-score of those genes
   in the per-cell transcript distribution.

Output
------
For each Jerber cell, returns a length-|G_Ph| vector of pathway-activity
scores. Aggregating across cells per stage reproduces the per-stage K_t
the methodology asks for (one float per stage per phosphosite).

CRITICAL framing for the manuscript: K_t is a TRANSCRIPT-DERIVED phospho-
pathway-activity feature, NOT measured phosphoproteomics. §2.4 must say so
explicitly when the methodology doc is updated.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .markers import G_Ph


# ---------------------------------------------------------------------------
# G_Ph phosphosite → upstream kinase / PROGENy pathway mapping.
# ---------------------------------------------------------------------------

# (gene, residue) → (canonical kinase HGNC, PROGENy pathway label)
# PROGENy pathways: "Androgen", "EGFR", "Estrogen", "Hypoxia", "JAK-STAT",
# "MAPK", "NFkB", "p53", "PI3K", "TGFb", "TNFa", "Trail", "VEGF", "WNT"
PHOSPHOSITE_TO_PATHWAY: dict[tuple[str, str], tuple[str, str]] = {
    # TH-pSer40 — phosphorylated by PKA, PKC, ERK1/2 → MAPK / PI3K integrators.
    # Use MAPK as the dominant DA-relevant upstream signal in iPSC-DA cultures.
    ("TH",      "S40"):  ("MAPK1", "MAPK"),
    # ERK1/2 TEY-motif autophosphorylation: directly the MAPK pathway.
    ("MAPK3",   "T202"): ("MAPK3", "MAPK"),
    ("MAPK3",   "Y204"): ("MAPK3", "MAPK"),
    ("MAPK1",   "T185"): ("MAPK1", "MAPK"),
    ("MAPK1",   "Y187"): ("MAPK1", "MAPK"),
    # AKT-pS473 phosphorylation by mTORC2 → PI3K pathway downstream signal.
    ("AKT1",    "S473"): ("AKT1",  "PI3K"),
    # GSK3B-pS9 inhibitory phosphorylation by AKT → also PI3K-pathway readout.
    ("GSK3B",   "S9"):   ("AKT1",  "PI3K"),
}


# ---------------------------------------------------------------------------
# Fallback backend: kinase-target gene sets (canonical, small, curated).
# Used when decoupler-py isn't installed. Each gene set is short by design
# so unit tests can validate behaviour without huge dependencies.
# ---------------------------------------------------------------------------

KINASE_TARGET_GENESETS: dict[str, list[str]] = {
    # MAPK1/3 (ERK1/2) target genes — IEGs activated by ERK signaling.
    "MAPK": ["FOS", "JUN", "EGR1", "DUSP1", "DUSP6", "MYC", "CCND1"],
    # PI3K/AKT pathway targets (proxy: glucose metabolism + survival).
    "PI3K": ["FOXO3", "BCL2", "MDM2", "CDKN1A", "GSK3B", "TSC2"],
}


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------

@dataclass
class KineticActivityScores:
    """Per-cell kinase-pathway activity, one column per G_Ph site."""

    cell_ids: list[str]
    site_labels: list[str]               # "TH_S40", "MAPK1_T185", ...
    pathway_labels: list[str]            # PROGENy pathway per site
    activity: np.ndarray                 # (n_cells, n_sites)
    backend: str                         # "decoupler-progeny" | "geneset-zscore"


def infer_k_activity(
    expr: pd.DataFrame,
    sites: tuple[tuple[str, str], ...] = G_Ph,
    *,
    use_decoupler: bool | None = None,
) -> KineticActivityScores:
    """Compute per-cell pathway-activity scores for each phosphosite in `sites`.

    Parameters
    ----------
    expr : pd.DataFrame
        Per-cell transcriptome, rows = cells, columns = HGNC gene symbols,
        values = log-normalized expression (Scanpy log1p convention).
    sites : sequence of (gene, residue) tuples
        Default: the full curated G_Ph set from `markers.G_Ph`.
    use_decoupler : bool or None
        True  → require decoupler+PROGENy; raise if unavailable.
        False → force the gene-set fallback.
        None  → try decoupler first, fall back if missing (default).

    Returns
    -------
    KineticActivityScores

    Notes
    -----
    The PROGENy pathway labels are stored alongside per-site activity; sites
    that share an upstream pathway (e.g. all four ERK TEY-motif residues
    map to "MAPK") will receive identical activity scores by design — that
    is the correct biology. Aggregating duplicates is left to the caller.
    """
    site_labels = [f"{g}_{r}" for g, r in sites]
    mappings = []
    missing_in_table: list[str] = []
    for s in sites:
        m = PHOSPHOSITE_TO_PATHWAY.get(s)
        if m is None:
            missing_in_table.append(f"{s[0]}_{s[1]}")
        mappings.append(m)
    if missing_in_table:
        warnings.warn(
            f"phosphosites with no PHOSPHOSITE_TO_PATHWAY entry: {missing_in_table}; "
            "their activity will be NaN."
        )
    pathway_labels = [m[1] if m else "" for m in mappings]

    if use_decoupler is False:
        return _infer_via_geneset(expr, site_labels, pathway_labels)
    try:
        return _infer_via_decoupler(expr, site_labels, pathway_labels)
    except (ImportError, RuntimeError) as e:
        if use_decoupler is True:
            raise
        warnings.warn(
            f"decoupler/PROGENy unavailable ({e}); falling back to gene-set z-score."
        )
        return _infer_via_geneset(expr, site_labels, pathway_labels)


def _infer_via_decoupler(
    expr: pd.DataFrame, site_labels: list[str], pathway_labels: list[str],
) -> KineticActivityScores:
    """PROGENy pathway-activity scoring via decoupler-py."""
    try:
        import decoupler as dc
    except ImportError as e:
        raise ImportError(
            "decoupler-py not installed. `pip install decoupler-py` to use "
            "PROGENy-based K_t inference."
        ) from e

    # PROGENy weights model: per-gene per-pathway weights from large
    # perturbation studies, returned as a long-format DataFrame.
    progeny = dc.get_progeny(organism="human", top=300)

    # decoupler.run_mlm runs a multivariate linear model per cell, scoring
    # each pathway against the transcriptome; returns a (cells × pathways)
    # DataFrame.
    acts, _ = dc.dense_run(
        dc.run_mlm, mat=expr, net=progeny,
        source="source", target="target", weight="weight",
        verbose=False,
    )

    n_cells = expr.shape[0]
    n_sites = len(site_labels)
    activity = np.full((n_cells, n_sites), np.nan, dtype=np.float32)
    for col, pw in enumerate(pathway_labels):
        if pw and pw in acts.columns:
            activity[:, col] = acts[pw].to_numpy().astype(np.float32)

    return KineticActivityScores(
        cell_ids=list(expr.index),
        site_labels=site_labels,
        pathway_labels=pathway_labels,
        activity=activity,
        backend="decoupler-progeny",
    )


def _infer_via_geneset(
    expr: pd.DataFrame, site_labels: list[str], pathway_labels: list[str],
) -> KineticActivityScores:
    """Fallback: per-pathway z-score over a small kinase-target gene set."""
    n_cells = expr.shape[0]
    n_sites = len(site_labels)
    activity = np.full((n_cells, n_sites), np.nan, dtype=np.float32)

    # Pre-z-score each gene across cells once for all pathways.
    z = (expr - expr.mean(axis=0)) / expr.std(axis=0).replace(0, 1.0)

    cache: dict[str, np.ndarray] = {}
    for col, pw in enumerate(pathway_labels):
        if not pw:
            continue
        if pw in cache:
            activity[:, col] = cache[pw]
            continue
        gene_set = KINASE_TARGET_GENESETS.get(pw, [])
        present = [g for g in gene_set if g in z.columns]
        if not present:
            continue
        s = z[present].mean(axis=1).to_numpy().astype(np.float32)
        cache[pw] = s
        activity[:, col] = s

    return KineticActivityScores(
        cell_ids=list(expr.index),
        site_labels=site_labels,
        pathway_labels=pathway_labels,
        activity=activity,
        backend="geneset-zscore",
    )


# ---------------------------------------------------------------------------
# Aggregation: per-cell K → per-stage K_t (single scalar like Pr_t).
# ---------------------------------------------------------------------------

def aggregate_to_k_t(
    activity: np.ndarray, cell_stages: list[str],
    stages: tuple[str, ...] = ("D11", "D30", "D52"),
) -> dict[str, float]:
    """Mean kinase-activity across all G_Ph sites and all cells of a stage.

    Returns {stage: K_t scalar}; matches the shape of `Pr_t` so K can be
    appended as a 4th feature to the state tensor for the ablation.
    """
    stages_arr = np.asarray(cell_stages)
    out: dict[str, float] = {}
    for stage in stages:
        mask = stages_arr == stage
        if not mask.any():
            out[stage] = float("nan")
            continue
        vals = activity[mask]                # (n_stage_cells, n_sites)
        out[stage] = float(np.nanmean(vals))
    return out
