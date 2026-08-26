"""Cross-modal harmonization: ID mapping, stage proxy alignment, normalization.

Methodology cross-reference: §1.4 Steps 1–3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Step 1 — Gene → UniProt mapping (§1.4 Step 1).
# ---------------------------------------------------------------------------

@dataclass
class IdMapping:
    """Resolved ID mapping with provenance for unmapped markers."""

    gene_to_uniprot: dict[str, str]
    unmapped: list[str]


def map_to_uniprot(symbols: list[str], cache_path: str | None = None) -> IdMapping:
    """Resolve HGNC symbols → reviewed Swiss-Prot UniProt accessions.

    Calls the UniProt ID-mapping REST API (https://rest.uniprot.org/idmapping/).
    Ambiguous mappings are resolved by selecting the reviewed (Swiss-Prot)
    entry; unmapped symbols are returned in `IdMapping.unmapped` so the
    pipeline can drop them from the marker set with an audit log.

    A `cache_path` can be passed to memoize results to disk and avoid
    re-querying UniProt across runs (helps with provenance reproducibility).
    """
    raise NotImplementedError("UniProt mapping not yet implemented")


# ---------------------------------------------------------------------------
# Step 2 — Temporal proxy alignment (§1.4 Step 2).
# ---------------------------------------------------------------------------
#
# M-A resolution paths (see project_open_methodology_issues.md):
#
#   (1) `assign_stage_by_pd_anchor` — methodology-as-written. CIRCULAR for
#       Pr-vs-D correlation analyses; do not use for §4.1 results.
#
#   (2) `assign_stage_by_source` — DEFAULT. Stage labels come from each
#       source paper's own sample annotations (e.g. Virdi 2022 D25 → D30,
#       Brenes 2018 hiPSC → D11). Independent of (P, D), no circularity.
#       Replaces the HPA + DepMap projection approach; HPA's single-cell-type
#       taxonomy was found to lack the granularity (no progenitor / immature /
#       midbrain-DA categories — only adult cortical neurons).
#
#   (3) `as_stage_conditional_prior` — null baseline. Pr/Ph become
#       stage-independent reference priors (Pr_t constant across t); the
#       4D model's lift over 2D should disappear under this prior, validating
#       that any lift seen with path (2) is real signal, not capacity.

# Mapping: source-paper sample annotation → master's-thesis stage label.
# Each entry is (source_paper_id, sample_label) → stage in {D11, D30, D52}.
#
# TODO(M-A): populate after dataset access is confirmed. Provisional based on
# the M-A resolution research:
#   - Brenes 2018 hiPSC samples → D11 (pluripotent baseline)
#   - Virdi 2022 D25 / D35 → D30   (early-mid DA)
#   - Virdi 2022 D65       → D52   (mature DA)
# All PXD accession IDs and exact timepoint labels MUST be verified against
# PRIDE / the source paper before citation; the research agent's web access
# was denied so accession numbers are from prior knowledge only.
SOURCE_STAGE_MAPPING: dict[tuple[str, str], str] = {}


def register_source_stage(source: str, sample_label: str, stage: str) -> None:
    """Add a source-paper → stage entry to SOURCE_STAGE_MAPPING.

    Use this from a config-loading step rather than mutating the dict
    directly, so the mapping is auditable in the provenance manifest.
    """
    if stage not in {"D11", "D30", "D52"}:
        raise ValueError(f"Unknown stage label: {stage!r}")
    SOURCE_STAGE_MAPPING[(source, sample_label)] = stage


def assign_stage_by_pd_anchor(
    cell_pd_scores: pd.DataFrame,
    stage_iqr_bounds: dict[str, tuple[np.ndarray, np.ndarray]],
) -> pd.Series:
    """Assign each external cell to {D11, D30, D52} by IQR membership in (P,D).

    DEPRECATED for §4.1 results — see M-A. Kept for the optional
    sensitivity analysis comparing (1) vs (2).

    `stage_iqr_bounds[stage] = (lower_bounds, upper_bounds)` are 2-vectors
    over (P, D) computed from the master's-thesis trajectory data.
    """
    raise NotImplementedError("PD-anchor stage assignment not yet implemented")


def assign_stage_by_source(
    samples: pd.DataFrame,
    source_col: str = "source",
    label_col: str = "sample_label",
    mapping: dict[tuple[str, str], str] | None = None,
) -> pd.Series:
    """M-A path 2 — label samples by source-paper annotation, not (P,D) projection.

    `samples` is a per-sample DataFrame with columns identifying the source
    paper and the in-paper sample label (e.g. "Virdi2022", "D25"). Stages
    are looked up from `SOURCE_STAGE_MAPPING`; samples without a mapping
    entry get NaN so the caller can audit the drop rate per source.

    This avoids the circularity of the original HPA-projection design: the
    stage label comes from the same paper that produced the protein
    measurements, so correlations between Pr_t and D_t in §4.1 cannot be
    tautological.
    """
    table = mapping if mapping is not None else SOURCE_STAGE_MAPPING
    if not table:
        raise ValueError(
            "Empty SOURCE_STAGE_MAPPING. Populate via `register_source_stage(...)` "
            "or pass an explicit `mapping`. See M-A in project memory."
        )
    keys = list(zip(samples[source_col], samples[label_col]))
    return pd.Series([table.get(k) for k in keys], index=samples.index)


def as_stage_conditional_prior(
    cell_protein_matrix: pd.DataFrame,
    stages: tuple[str, ...] = ("D11", "D30", "D52"),
) -> pd.DataFrame:
    """M-A path 3 — single proteomic distribution applied to all stages.

    Returns a (n_stages × n_proteins) DataFrame whose rows are *identical*:
    each stage receives the same population-mean protein abundance across
    the full external dataset. Useful as a "Pr/Ph carries no stage signal"
    baseline; the 4D model's lift over 2D should disappear under this
    prior, validating that any lift seen with paths 1/2 is real.
    """
    pop_mean = cell_protein_matrix.mean(axis=0)
    return pd.DataFrame(
        np.tile(pop_mean.to_numpy(), (len(stages), 1)),
        index=list(stages),
        columns=cell_protein_matrix.columns,
    )


# ---------------------------------------------------------------------------
# Step 3 — Modality-specific normalization (§1.4 Step 3).
# ---------------------------------------------------------------------------

def normalize_transcriptomics(adata) -> None:
    """Scanpy log1p + normalize_total. Mutates `adata.X` in place."""
    import scanpy as sc
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)


def normalize_hpa(matrix: pd.DataFrame) -> pd.DataFrame:
    """log2(1 + x), then z-score per protein (row-wise)."""
    log = np.log2(1.0 + matrix)
    mu = log.mean(axis=1)
    sd = log.std(axis=1).replace(0, 1.0)
    return log.sub(mu, axis=0).div(sd, axis=0)


def normalize_depmap(matrix: pd.DataFrame) -> pd.DataFrame:
    """Median-center log2 TMT intensities per sample (column-wise)."""
    return matrix.sub(matrix.median(axis=0), axis=1)


def normalize_phospho(
    phospho: pd.DataFrame, total_protein: pd.DataFrame
) -> pd.DataFrame:
    """log2(1 + phospho), normalized to total protein abundance per sample.

    `phospho` and `total_protein` share the sample index. For each phosphosite
    "{gene}_{residue}", the corresponding total protein column is `gene`.
    """
    raise NotImplementedError("Phospho normalization not yet implemented")
