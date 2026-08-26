"""Dataset loaders.

Methodology cross-reference: §1.1 (Jerber), §1.2 (proteomics), §1.3
(phosphoproteomics).

M-A status: HPA + DepMap have been replaced as the proteomic source.
HPA's single-cell-type taxonomy lacks DA-relevant cell types (no
progenitor / immature-neuron / midbrain-DA categories), and the original
"project HPA into (P,D) space" approach was circular for §4.1 correlations.
The current design uses real iPSC-DA proteomic time courses (e.g. Virdi
2022, Brenes 2018) loaded through `load_proteomic_timecourse`. HPA / DepMap
loaders are retained for sensitivity analyses only.

All loaders return raw, *un*-normalized data; modality-specific normalization
lives in `harmonize.py` (§1.4 Step 3).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def load_jerber_2021(path: str | Path):
    """Load the Jerber et al. (2021) scRNA-seq AnnData (§1.1).

    Expected file: a single .h5ad with `obs.day` ∈ {D11, D30, D52} and
    `obs.treatment == 'NONE'` already filtered. The master's thesis
    preprocessing produces this artifact.
    """
    import anndata as ad
    return ad.read_h5ad(str(path))


# ---------------------------------------------------------------------------
# ICUFN backbone bridge — re-use Kanu et al. 2026 published trajectory artifacts.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ICUFNBaseline:
    """Published 2D `(P, D)` trajectory tensor + bounds metadata.

    Attributes
    ----------
    state : (200, 3, 2) float32
        Already-normalized `(P, D) ∈ [0, 1]` per trajectory per stage.
        Stage order along axis 1 is fixed: index 0=D11, 1=D30, 2=D52.
    bounds : dict {'P_min','P_max','D_min','D_max'}
        Population min-max used to produce `state` in
        `build_timepoint_trajectories.py`. Stored as documentation only —
        already applied to `state`. Re-used at integration time so the
        new Pr/K dimensions can adopt the same kind of fixed-bounds
        normalization for fair comparison.
    """

    state: np.ndarray
    bounds: dict[str, float]

    @property
    def stage_order(self) -> tuple[str, ...]:
        return ("D11", "D30", "D52")


def load_icufn_baseline(
    repo_path: str | Path,
    trajectory_filename: str = "dopaminergic_trajectories_pseudotime.pkl",
) -> ICUFNBaseline:
    """Load the published 2D state tensor + bounds from the ICUFN paper repo.

    The canonical file is `dopaminergic_trajectories_pseudotime.pkl`
    (the name is a historical artifact — `build_timepoint_trajectories.py`
    overwrote the older pseudotime file in place with ground-truth-timepoint
    trajectories). Each entry is a list of 3 [P_norm, D_norm] pairs in [0,1],
    in fixed stage order D11 → D30 → D52.

    Bounds are loaded from `normalization_bounds.pkl` for documentation; they
    have already been applied to the state tensor.
    """
    repo = Path(repo_path)
    traj_path = repo / "data" / "processed" / trajectory_filename
    bounds_path = repo / "data" / "processed" / "normalization_bounds.pkl"

    with open(traj_path, "rb") as f:
        traj_list = pickle.load(f)
    arr = np.array(traj_list, dtype=np.float32)
    if arr.shape != (200, 3, 2):
        raise ValueError(
            f"Expected (200, 3, 2) trajectory tensor from {traj_path.name}, "
            f"got {arr.shape}. If you intended the legacy 3-feature file with "
            "day labels, pass trajectory_filename='dopaminergic_trajectories.pkl' "
            "and slice [..., :2] yourself."
        )

    with open(bounds_path, "rb") as f:
        bounds = pickle.load(f)
    required = {"P_min", "P_max", "D_min", "D_max"}
    if not required.issubset(bounds):
        raise KeyError(f"Bounds missing keys: {required - set(bounds)}")

    return ICUFNBaseline(
        state=arr,
        bounds={k: float(bounds[k]) for k in required},
    )


def load_icufn_population_dynamics(repo_path: str | Path) -> pd.DataFrame:
    """Load the per-stage population mean/median CSV from the ICUFN repo.

    Used as a sanity-check anchor: applying the saved bounds to the saved
    trajectory tensor should reproduce these per-stage statistics.
    """
    csv = Path(repo_path) / "data" / "processed" / "dopaminergic_population_dynamics.csv"
    return pd.read_csv(csv)


# ---------------------------------------------------------------------------
# M-A primary path: real iPSC-DA proteomic time courses.
# ---------------------------------------------------------------------------

@dataclass
class ProteomicTimeCourse:
    """Bundle of protein abundance + sample annotations from a single source paper.

    Attributes
    ----------
    abundance : DataFrame
        Wide format: rows = HGNC symbols, columns = sample IDs, values =
        log2 abundance (loader-specific normalization, see paper).
    samples : DataFrame
        Per-sample metadata, indexed by sample ID. Must contain at least
        ``source`` and ``sample_label`` columns so `harmonize.assign_stage_by_source`
        can resolve stage labels via :data:`harmonize.SOURCE_STAGE_MAPPING`.
    pxd_accession : str
        ProteomeXchange / PRIDE accession (e.g. ``"PXD009273"``); recorded
        in the provenance manifest.
    citation : str
        First-author + year + journal string.
    """

    abundance: pd.DataFrame
    samples: pd.DataFrame
    pxd_accession: str
    citation: str


def load_proteomic_timecourse(
    path: str | Path, source: str
) -> ProteomicTimeCourse:
    """Load a per-paper iPSC-DA proteomic time course.

    Each supported `source` has its own loader implementation since column
    layouts differ across papers (TMT channels, sample naming, etc.).
    """
    if source == "Novak2022":
        return _load_novak2022(path)
    raise NotImplementedError(
        f"Proteomic time-course loader for source={source!r} not yet implemented"
    )


# ---------------------------------------------------------------------------
# Novak et al. 2022 (PRIDE PXD028283) — MaxQuant proteinGroups.txt-format
# loader. The PRIDE file is named `selectedProteins.txt` but has identical
# columns to a standard MaxQuant `proteinGroups.txt`. The loader works on
# either.
# ---------------------------------------------------------------------------

# Sample-label → ICUFN-stage mapping. Two source files supported in parallel:
#
#  (a) selectedProteins.txt (PRIDE PXD028283, 127-protein curation):
#      Named CTRL columns are mapped; PINK1 ignored. 0/6 G_Pr markers in this
#      file → Pr is degenerate. Original M-A path 2 design.
#
#  (b) proteinGroups.txt (Path 3, re-processed locally via MaxQuant 2.8):
#      Sample identities are NOT in the PRIDE submission, so we use a
#      TH-stratified pragmatic grouping. All 8 raw files have TH log2 > 23.9
#      → all are DA-stage (no iPSC). Splitting at the TH-abundance median:
#          Lower-TH 4 (Skupin11–14, log2 TH 23.94–24.66) → D30
#          Higher-TH 4 (Skupin15–18, log2 TH 24.78–25.71) → D52
#      CTRL/PINK1 pooled within each stage (n=4 per stage); PINK1 only
#      perturbs TH ~10–20%, so stage-level Pr remains dominated by the
#      differentiation signal. Will be replaced by Dittmar's official
#      mapping when he replies (cross-validation; pipeline rerun is one
#      command). 2/6 G_Pr markers (TH, ALDH1A1) present → Pr is real.
NOVAK2022_CTRL_SAMPLES: dict[str, str] = {
    # (a) selectedProteins.txt CTRL columns
    "cont IPS prot":     "D11",
    "cont IPSC":         "D11",
    "cont prot IPSC":    "D11",
    "D25 C7":            "D30",
    "D25 C8":            "D30",
    "D40 C 7":           "D52",
    "D40 C 8":           "D52",
    # (b) Path 3 re-processed proteinGroups.txt — TH-stratified
    "Skupin11":          "D30",   # log2 TH = 24.66
    "Skupin12":          "D30",   # log2 TH = 24.46
    "Skupin13":          "D30",   # log2 TH = 23.94
    "Skupin14":          "D30",   # log2 TH = 24.04
    "Skupin15":          "D52",   # log2 TH = 25.71
    "Skupin16":          "D52",   # log2 TH = 25.28
    "Skupin17":          "D52",   # log2 TH = 24.74
    "Skupin18":          "D52",   # log2 TH = 24.78
}


def _load_novak2022(path: str | Path) -> ProteomicTimeCourse:
    """Load Novak 2022 selectedProteins.txt (PRIDE PXD028283).

    Returns
    -------
    ProteomicTimeCourse with:
        abundance : DataFrame (HGNC gene symbols × sample IDs), values are
                    raw LFQ intensities (pre-log).
        samples   : DataFrame indexed by sample ID with columns
                    `source`, `sample_label`, `condition`, `stage`.

    Multi-gene protein groups (rows whose 'Gene names' is e.g. "TAF9;AK6")
    are exploded — one output row per gene, sharing the row's intensity
    values. If the same gene appears in multiple protein groups, the entries
    are summed (LFQ intensities are additive on the linear scale).
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)

    # Standard MaxQuant filtering: drop reverse hits, contaminants,
    # site-only identifications.
    for flag in ("Reverse", "Potential contaminant", "Only identified by site"):
        if flag in df.columns:
            df = df[df[flag].fillna("") != "+"]

    if "Gene names" not in df.columns:
        raise ValueError(
            f"{path}: no 'Gene names' column — not a MaxQuant proteinGroups format"
        )

    # All LFQ intensity columns
    lfq_cols = [c for c in df.columns if c.startswith("LFQ intensity ")]
    if not lfq_cols:
        raise ValueError(
            f"{path}: no 'LFQ intensity *' columns — not a MaxQuant proteinGroups format"
        )

    # Map sample-id → matching column name (handles the awkward extra spaces).
    sample_to_col: dict[str, str] = {}
    for c in lfq_cols:
        suffix = c[len("LFQ intensity "):]
        sample_to_col[suffix] = c

    # Restrict to CTRL samples we care about; warn on any missing.
    selected_samples: dict[str, str] = {}    # sample_id → column name
    missing: list[str] = []
    for sample_id in NOVAK2022_CTRL_SAMPLES:
        if sample_id in sample_to_col:
            selected_samples[sample_id] = sample_to_col[sample_id]
        else:
            missing.append(sample_id)
    if missing:
        # Not fatal — the file may have been re-shared in a slimmer form.
        print(f"[novak] CTRL samples missing from file: {missing}")

    # Explode semicolon-separated 'Gene names' into one row per gene.
    long_rows = []
    for _, row in df[["Gene names", *selected_samples.values()]].iterrows():
        names = str(row["Gene names"]).strip()
        if not names or names.lower() == "nan":
            continue
        for g in names.split(";"):
            g = g.strip()
            if not g:
                continue
            long_rows.append({"gene": g, **{
                sid: float(row[col]) for sid, col in selected_samples.items()
            }})
    if not long_rows:
        raise ValueError(f"{path}: filtering produced zero rows")

    long_df = pd.DataFrame(long_rows)
    # Sum LFQ across redundant protein groups for the same gene.
    abundance = long_df.groupby("gene", as_index=True).sum()

    # Per-sample metadata
    samples = pd.DataFrame(
        [
            {
                "sample_id": sid,
                "source": "Novak2022",
                "sample_label": sid,
                "condition": "CTRL",
                "stage": NOVAK2022_CTRL_SAMPLES[sid],
            }
            for sid in selected_samples
        ]
    ).set_index("sample_id")

    return ProteomicTimeCourse(
        abundance=abundance,
        samples=samples,
        pxd_accession="PXD028283",
        citation=(
            "Novak G, Kyriakis D, Grzyb K, Bernini M, Rodius S, Dittmar G, "
            "Finkbeiner S, Skupin A. Single cell transcriptomics of human "
            "iPSC differentiation dynamics reveal a core network of "
            "Parkinson's disease. Commun Biol 5:49 (2022). PMID 35027645."
        ),
    )


def load_phospho_timecourse(
    path: str | Path, source: str
) -> ProteomicTimeCourse:
    """Load a per-paper phosphoproteomic time course.

    Returns a `ProteomicTimeCourse` whose `abundance` rows are keyed
    ``"{gene}_{residue}"`` to match `scores.phospho_score` expectations.

    Only Virdi 2022 currently has a clean iPSC-DA phospho time course at
    multiple stages per the M-A research; other sources are partial.
    """
    raise NotImplementedError(
        f"Phospho time-course loader for source={source!r} not yet implemented"
    )


# ---------------------------------------------------------------------------
# Sensitivity-analysis path: HPA / DepMap loaders (DO NOT use as primary
# Pr source — see §1.4 M-A note above).
# ---------------------------------------------------------------------------

def load_hpa_proteomics(path: str | Path) -> pd.DataFrame:
    """HPA single-cell-type protein expression (§1.2.1, sensitivity-only).

    Returns
    -------
    DataFrame indexed by HGNC gene symbol, columns = HPA cell types,
    values = normalized protein expression.
    """
    raise NotImplementedError("HPA loader not yet implemented")


def load_depmap_proteomics(path: str | Path) -> pd.DataFrame:
    """DepMap CCLE proteomics (§1.2.2, sensitivity-only).

    Returns
    -------
    DataFrame, rows = HGNC symbols, columns = cell line IDs (DepMap ID),
    values = log2 TMT intensity (median-centered per sample).
    """
    raise NotImplementedError("DepMap loader not yet implemented")


def load_phosphositeplus(
    path: str | Path, sites: list[tuple[str, str]]
) -> pd.DataFrame:
    """Curated phosphosite quantitative entries from PhosphoSitePlus (§1.3).

    NOTE(M-D): PhosphoSitePlus is primarily an annotation database; quantitative
    coverage of the chosen sites in DA differentiation is sparse. Prefer
    `load_phospho_timecourse(..., source="Virdi2022")` for primary §2.4 scores.
    """
    raise NotImplementedError("PhosphoSitePlus loader not yet implemented")


def load_replogle_perturbseq(path: str | Path):
    """Load Replogle 2022 K562 Perturb-seq AnnData (§1.2.3).

    NOTE(M-C): K562 + transcriptomic-only. Used for gene-level co-regulation
    evidence in marker-set rationale, *not* for protein validation.
    """
    import anndata as ad
    return ad.read_h5ad(str(path))
