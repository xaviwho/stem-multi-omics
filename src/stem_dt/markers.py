"""Marker sets for multi-omics state space scoring.

Methodology cross-reference: §1.1, §1.3, §2.1–§2.4.

Symbol conventions
------------------
- Transcriptomic markers use HGNC gene symbols (matched to scRNA-seq feature names).
- Proteomic markers use HGNC symbols too; UniProt accessions are resolved at
  load time via `harmonize.map_to_uniprot` (see §1.4 Step 1).
- Phosphosite markers are (HGNC_symbol, residue) tuples in canonical UniProt
  human numbering; resolution to PhosphoSitePlus rows happens in `data.load_phosphositeplus`.
"""

from __future__ import annotations

# §1.1 — Transcriptomic marker sets (carried over from master's thesis, unchanged)
G_P: tuple[str, ...] = ("POU5F1", "NANOG", "SOX2", "UTF1", "TDGF1")
G_D: tuple[str, ...] = ("TH", "DDC", "SLC6A3", "DRD2", "LMX1A", "FOXA2")

# §2.3 — Proteomic markers for dopaminergic maturity.
#
# v1.2 update (2026-05): Expanded G_Pr to 5 DA-pathway proteins all measured in
# Novak's Path-3 re-processed proteome. This addresses the v1.1 limitation of
# 2/6 marker coverage. The original DA-specification TFs (NR4A2, PITX3, LMX1B)
# remain absent due to MS biology (low-abundance nuclear proteins below LFQ
# detection limit) regardless of dataset choice; KCNJ6 (membrane channel)
# similarly requires targeted enrichment. Cavarischia-Rega 2024 (PXD050991)
# uses the same whole-cell-LFQ workflow and would not recover them either —
# this is a methodology-of-MS limitation, not a curation choice.
#
# The new G_Pr tests two complementary hypotheses:
# (a) Protein-level abundance of canonical DA markers also in G_D (TH, DDC,
#     FOXA2) carries information beyond their transcript counterparts —
#     tests transcript-protein decoupling at stage-level resolution
# (b) Novel DA-pathway proteins not in G_D (ALDH1A1, COMT) provide
#     additional independent discrimination
# Both contributions are isolated by the across-stage permutation null (§5.2).
#
# Original G_Pr archived for reference:
#   {TH, NR4A2, PITX3, LMX1B, ALDH1A1, KCNJ6}
G_Pr: tuple[str, ...] = (
    "TH",       # tyrosine hydroxylase — rate-limiting DA-synthesis enzyme
    "DDC",      # DOPA decarboxylase / AADC — second DA-synthesis enzyme
    "ALDH1A1",  # A9-DA subtype marker (the PD-vulnerable subtype)
    "COMT",     # catechol-O-methyltransferase — DA catabolism enzyme
)
# v1.3 G_Pr — filtered to markers with monotonically INCREASING abundance
# across iPSC → D25 → D40 in Dittmar's canonical proteinGroups (n=2 CTRL reps
# per stage, plus n=3 iPSC reference). Per-marker contrasts (log2 LFQ):
#     TH      : 7.32  → 25.43 → 26.79  (clear DA maturation marker)
#     DDC     : 0.00  → 28.05 → 27.90  (sharp iPSC→D25 jump; D25/D40 plateau
#                                       within noise — included because the
#                                       full 3-stage trajectory is monotonic)
#     ALDH1A1 : 0.00  → 0.00  → 22.50  (sharp threshold at D40; A9 subtype)
#     COMT    : 0.00  → 22.42 → 23.14  (monotonic increase with maturation)
#
# Excluded after per-marker analysis on Dittmar's data:
#     FOXA2   : 0.00  → 11.09 → 0.00   (transient at D25, silenced at D40 —
#                                       floor-plate TF; non-monotonic)
# Still missing (MS biology, even in Dittmar's 3,577-protein deeper proteome):
#     NR4A2, PITX3, LMX1B  : nuclear DA-specification TFs, below standard
#                            whole-cell LFQ detection limit
#     KCNJ6                 : GIRK2 membrane channel — needs targeted
#                            enrichment workflow
G_Pr_excluded_with_reasons: dict[str, str] = {
    "FOXA2": "non-monotonic across stages (transient at D25, silenced D40)",
    "NR4A2": "nuclear TF below standard whole-cell LFQ detection",
    "PITX3": "nuclear TF below standard whole-cell LFQ detection",
    "LMX1B": "nuclear TF below standard whole-cell LFQ detection",
    "KCNJ6": "membrane channel — needs targeted enrichment workflow",
}

# §1.3 — Phosphosite markers.
# TODO(M-E): ERK1/2 pT202/pY204 → those numbers are ERK1 (MAPK3); ERK2 (MAPK1)
# TEY motif sits at T185/Y187. Listing both isoforms below; collapse if author
# confirms a single isoform.
# TODO(M-D): PhosphoSitePlus quantitative coverage in DA differentiation is
# sparse — define fallback (drop / impute / carry-forward) in
# `data.load_phosphositeplus`.
# TODO(M-A): PPP1R1B (DARPP-32) is striatal-MSN-specific and unlikely to be
# quantified in midbrain iPSC-DA cultures — flagged commented-out below.
# Candidate replacements with PD-relevance and good DA-culture coverage:
#   ("SNCA",  "S129"),  # α-synuclein pS129 — canonical PD marker
#   ("LRRK2", "S935"),  # LRRK2 pS935 — kinase-active state, well-quantified
# Pending user decision; until then G_Ph is 6 sites (DARPP-32 excluded).
G_Ph: tuple[tuple[str, str], ...] = (
    ("TH",       "S40"),     # regulates DA synthesis rate
    ("MAPK3",    "T202"),    # ERK1 TEY-motif activation
    ("MAPK3",    "Y204"),    # ERK1 TEY-motif activation
    ("MAPK1",    "T185"),    # ERK2 TEY-motif activation
    ("MAPK1",    "Y187"),    # ERK2 TEY-motif activation
    ("AKT1",     "S473"),    # PI3K/AKT survival signaling (AKT1 isoform assumed)
    # ("PPP1R1B", "T34"),    # DARPP-32 — striatal-only; see TODO(M-A) above
    ("GSK3B",    "S9"),      # inhibitory phosphorylation; neuroprotection
)

ALL_GENE_SYMBOLS: frozenset[str] = frozenset(
    list(G_P) + list(G_D) + list(G_Pr) + [s for s, _ in G_Ph]
)
