# stem-multi-omics

Paper 1 implementation: **Multi-Omics State Space Construction for Stem Cell-Based Digital Twins**.

Extends the 2D `(P, D)` transcriptomic state from the master's thesis to a 4D
`(P, D, Pr, Ph)` state by integrating proteomics (HPA + DepMap CCLE) and
phosphoproteomics (PhosphoSitePlus), with a blockchain provenance layer.

See [methodology.md](methodology.md) for the full protocol.

## Repo layout

```
src/stem_dt/
  markers.py        marker sets G_P, G_D, G_Pr, G_Ph
  scores.py         P_t, D_t, Pr_t, Ph_t score functions + min-max scaling
  data.py           dataset loaders (Jerber, HPA, DepMap, PhosphoSitePlus, Replogle)
  harmonize.py      UniProt ID mapping, stage proxy alignment, normalization
  provenance.py     SHA-256 hashing, JSON manifest, optional on-chain registration
  validation.py     correlation, MOFA+, trajectory coherence
  benchmark.py      stage classification (Task A) + trajectory prediction (Task B)
scripts/
  run_pipeline.py   end-to-end driver
tests/
  test_scores.py
configs/
  default.yaml
```

## Status

Scaffold only — modules contain typed stubs and TODOs tied to open
methodology questions (see `methodology.md` and the inline `TODO(M-x)`
markers, where `x` matches the critique items from the design review).
