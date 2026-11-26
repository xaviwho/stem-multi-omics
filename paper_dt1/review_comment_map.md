# Reviewer comment numbering (CMPB-D-26-05530)

The decision letter numbers comments per question (Q1–Q9), so the same number
repeats. `revision.tex` uses one sequential number per reviewer instead. This
table maps each tag to the letter. Duplicate comments share one tag.
`AC` = author correction found while revising (not requested by a reviewer).

## Reviewer 1

| Tag | Letter location | Comment (short) | Where addressed |
|---|---|---|---|
| R1.1 | Q1.2, final #1 | Clarify main novelty: Pr axis vs prediction vs blockchain | Abstract; Intro "Objectives"; contribution list |
| R1.2 | Q2.2 | How the 200 trajectories were sampled; how Pr was assigned | §3.3 Eq. (prsample), "Assigning Pr", trajectory assembly |
| R1.3 | Q2.3 | LORO is a consistency check, not external validation | §3.5 LORO; §5.2 validation list |
| R1.4 | Q3.2 | CIs for the MAE reduction; do seeds change the split? | Abstract; §3.3; §3.5 multi-seed; Table 7 CI rows; §4.3.2 |
| R1.5 | Q3.3, Q7.2, final #2 | Small CTRL replicate count | Abstract; §4.3.1; Limitation 1; Conclusion |
| R1.6 | Q4.2 | Fig. 1 labels too small | Fig. 1 printed at 0.85 width; caption |
| R1.7 | Q4.3 | Figs 2/4: distinguish 2D, 3D, Pr-only, null conditions | §4.3 condition definitions; Table 6; Fig. 3 and Fig. 5 captions |
| R1.8 | Q5.2, Q7.3, final #3 | Task A = separability, not model superiority | Abstract; Table 6; §4.3.1; Limitation 4; Conclusion |
| R1.9 | Q5.3, final #4 | MAE gain comes from Pr; P/D unchanged | Abstract; §4.3.2; Conclusion |
| R1.10 | Q6.2 | Emphasise controls as the main strength | §5.2 "What the validation experiments establish" |
| R1.11 | Q6.3, final #5 | Explain blockchain's practical value simply | §3.4 "What the provenance layer does in practice" |
| R1.12 | Q7.4 | Missing phosphoproteomics as future direction | Limitation 6; Future directions |
| R1.13 | Q8.2 | Link each table/figure to its takeaway | "Takeaway" sentences throughout §4 |
| R1.14 | Q8.3 | Make Discussion more concise | §5.1, blockchain discussion, Conclusion rewritten shorter |
| R1.15 | Q9 | Language editing | Revised text; see note below |
| R1.16 | final #6 | Cite Pandey & Kumar (IEEE TIE 2026) | §2.1; `newref.bib` |

## Reviewer 2

| Tag | Letter location | Comment (short) | Where addressed |
|---|---|---|---|
| R2.1 | Q1.1 | Primary objective: stage ID or trajectory prediction? | Abstract; Intro "Objectives" |
| R2.2 | Q1.2 | Rationale for TH, DDC, ALDH1A1, COMT; are they adequate? | §3.3 "Rationale and adequacy" |
| R2.3 | Q1.3 | Justify matching D25→D30, D40→D52 | §3.2.3 cross-dataset correspondence |
| R2.4 | Q2.1 | Proteomics processing: missing values, selection, aggregation | §3.2 processing steps (i)–(v); Table 1 caption |
| R2.5 | Q2.2 | Table linking file names → samples → stages | Table 3 (per-replicate) and text |
| R2.6 | Q2.3 | How stage μ/σ are used; how leakage was avoided | §3.1; §3.3 Pr assignment, normalisation, "Leakage control" |
| R2.7 | Q2.4 | Make code public | "Code and data availability" (repo is PRIVATE — see TODO) |
| R2.8 | Q3.1 | Small n effect on estimates and stability | Abstract; Limitation 1 |
| R2.9 | Q3.2 | Seed independence; why paired Wilcoxon | §3.5 "Independence of seeds and choice of test" |
| R2.10 | Q4.1 | Stage-mapping table with sources and replicate counts | Table 2 |
| R2.11 | Q4.2 | Plot of individual replicates | Fig. 2 (`images/fig_pr_replicates.png`) |
| R2.12 | Q5.1 | Acknowledge Pr-only = 3D | Abstract; §4.3.1; Limitation 4 |
| R2.13 | Q5.2 | Interpret MAE carefully per dimension | Abstract; Table 7 caption; §4.3.2 |
| R2.14 | Q6.1 | How Pr complements transcriptomics; marker relevance | §3.3 rationale; §5.1 |
| R2.15 | Q6.2 | Summarise validation experiments | §5.2 |
| R2.16 | Q6.3 | Practical role of blockchain provenance | §3.4; blockchain discussion |
| R2.17 | Q7.1 | Cross-study batch effects | Limitation 2; Future directions |
| R2.18 | Q7.2 | Few replicates at D30/D52 | Limitation 1 |
| R2.19 | Q7.3 | No external validation | Limitation 3; Future directions |
| R2.20 | Q7.4 | Drug screening not validated | Abstract; Limitation 7; Future directions |
| R2.21 | Q8.1 | Objectives at end of Introduction | Intro "Objectives" |
| R2.22 | Q8.2 | Systematic matching description in Methods | §3.2.3 |
| R2.23 | Q8.3 | Separate classification from prediction in Results | §4.3 intro; §4.3.1 / §4.3.2 headings |
| R2.24 | Q8.4 | Shorten / move blockchain details | Solidity settings + 5 table rows → `supplementary.tex` (Table S1) |
| R2.25 | Q9 | Language editing | Revised text; see note below |

## Author corrections (AC)

Code–paper mismatches found while answering the comments. All are described
truthfully in the text; no results changed.

- D11 Pr is drawn from N(0, σ_D11/2) (detection floor), not N(1.83, 3.17) as originally written.
- P/D are scaled with fixed bounds over all 161,584 cells, not training-split bounds; only Pr is fitted on the training split.
- Eq. (MAE) now divides by the number of dimensions k (the code averages over dimensions).
- Random Forest uses 200 trees for both tasks (text said 100 for Task A).
- Task A's linear model is logistic regression (Fig. 1 caption said Ridge).
- D11 is the floor-plate progenitor stage, not "pluripotent".
- Within-stage orthogonality of Pr is expected by construction (Pr is drawn independently of P, D); the abstract and §4.2 claims were toned down accordingly.
- Limitation "4 of 6 markers" corrected to 4 of the 9 candidates in Table 1; "Paper 2" wording removed.

## Language editing note (R1.15, R2.25)

New and rewritten text was edited for clarity. Unchanged original text has not
had a full language pass; a professional editing service is advisable before
resubmission.
