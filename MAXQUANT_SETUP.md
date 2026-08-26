# MaxQuant Setup Guide — Path 3 (re-process Novak 2022 raw files)

This guide walks through running MaxQuant on the 8 `.raw` files downloaded
from PRIDE PXD028283 to produce a `proteinGroups.txt` that includes the
canonical DA markers (TH, ALDH1A1, NR4A2, etc.) which Novak's public
`selectedProteins.txt` curation excluded.

## Prerequisites

1. **MaxQuant** — Windows, free download from <https://www.maxquant.org>.
   Latest stable as of writing is 2.6.x. Match the version Novak used
   (MS_Software field in the raw header was `2.8-280502/2.8.1.2806` —
   that is the **firmware** of the Q Exactive Plus, not MaxQuant).
   Any MaxQuant ≥ 2.0 will work fine.
2. **Search database** — UniProt human reviewed (Swiss-Prot) FASTA.
   ```
   curl -L 'https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta&query=(reviewed:true)+AND+(organism_id:9606)' \
     -o data/raw/uniprot_human_reviewed.fasta.gz
   gunzip data/raw/uniprot_human_reviewed.fasta.gz
   ```
   ~13 MB compressed, ~80 MB unzipped, ~20,000 sequences.
3. **Disk space** — MaxQuant writes intermediate files; reserve ~10 GB
   on the same drive as the raw files.

## MaxQuant GUI configuration

Launch `MaxQuant.exe`. The GUI has a tabbed layout. Configure each tab
in order:

### Tab 1 — Raw files

1. Click **Load** → select all 8 `Skupin*.raw` files from
   `data/raw/novak2022_raw/`.
2. Click **Set experiment** → enter a unique experiment name per file.
   Use descriptive names that you will recognize later in the
   `proteinGroups.txt` columns. Recommended convention until Dittmar
   confirms the vial mapping:

   | File | Experiment name | Notes |
   |---|---|---|
   | Skupin11.raw | `Skupin11` | (rename later when Dittmar replies) |
   | Skupin12.raw | `Skupin12` | |
   | Skupin13.raw | `Skupin13` | |
   | Skupin14.raw | `Skupin14` | |
   | Skupin15.raw | `Skupin15` | |
   | Skupin16.raw | `Skupin16` | |
   | Skupin17.raw | `Skupin17` | |
   | Skupin18.raw | `Skupin18` | |

   These names become the column suffixes in `LFQ intensity SkupinXX`.
   Our post-hoc matcher converts these to `D25 C7 / D40 C 8 / …` after
   the run finishes.
3. Click **Set parameter group** → leave all rows in group 0
   (single experiment design). MBR groups all samples together so that
   identifications can be transferred between any pair of runs.

### Tab 2 — Group-specific parameters (Group 0)

Apply these settings to group 0:

- **Type:** `Standard`
- **Multiplicity:** `1` (label-free)
- **Variable modifications:** `Oxidation (M)`, `Acetyl (Protein N-term)`
- **Fixed modifications:** `Carbamidomethyl (C)`
- **Digestion mode:** `Specific`
- **Enzyme:** `Trypsin/P`
- **Max missed cleavages:** `2`
- **Label-free quantification (LFQ):** **ON**
- **LFQ min ratio count:** `2`
- **Min number of neighbors:** `3` (default)
- **Average number of neighbors:** `6` (default)
- **iBAQ:** OFF (not needed)

### Tab 3 — Sequences

- Click **Add file** → select `data/raw/uniprot_human_reviewed.fasta`
- Identifier rule: `>.*\|(.*)\|` (UniProt accession)
- Description rule: `>(.*)` (full FASTA header)
- Taxonomy: `Homo sapiens`

### Tab 4 — Identification

- **PSM FDR:** `0.01`
- **Protein FDR:** `0.01`
- **Site decoy fraction:** `0.01`
- **Min peptides:** `1`
- **Min razor + unique peptides:** `1`
- **Min unique peptides:** `0`

### Tab 5 — Quantification

- **Match between runs:** **ON** ← critical
- **Match time window:** `0.7` min (default)
- **Alignment time window:** `20` min (default)

### Tab 6 — Global parameters

- **Number of threads:** match your CPU core count (typically 8–16)
- **Output folder:** leave default (`combined/` next to raw files)
- **Min peptide length:** `7`
- **Max peptide length for unspecific search:** `25`

## Save and run

1. Click **Save parameters** at the top → save as `mqpar.xml` next to the
   raw files (`data/raw/novak2022_raw/mqpar.xml`).
2. Click **Start** to begin the run.

Expected runtime: **4–12 hours** depending on CPU. The run logs progress
in real time; you can leave it overnight. CPU-bound, no user input needed.

## Headless / scripted run (optional)

If you'd rather run from the command line:

```
cd data/raw/novak2022_raw
"C:/Path/To/MaxQuant/bin/MaxQuantCmd.exe" mqpar.xml
```

`MaxQuantCmd.exe` reads the same `mqpar.xml` produced by the GUI.

## After it finishes

Output lands at `data/raw/novak2022_raw/combined/txt/`. The file we want is
`proteinGroups.txt`. Then:

```
# Standardize MQ output and run post-hoc sample-mapping
python scripts/postprocess_maxquant.py \
  --proteingroups data/raw/novak2022_raw/combined/txt/proteinGroups.txt \
  --reference     data/raw/novak2022_selectedProteins.txt \
  --out           data/raw/novak2022_proteinGroups.txt

# Update .env and re-run pipeline
# NOVAK2022_PROTEINGROUPS_PATH=data/raw/novak2022_proteinGroups.txt
python scripts/preview_novak.py     # confirm G_Pr markers found
python scripts/run_benchmark.py     # real numbers
```

The post-processor uses Pearson correlation across the 127 proteins in
`selectedProteins.txt` to identify which of our `Skupin11..Skupin18` columns
maps to which of Novak's named samples (`D25 C7`, `D40 C 8`, etc.). Once
Dittmar replies with the official mapping, we cross-validate against his
table; in practice the post-hoc matching converges with correlation > 0.95
per pair when applied to the same raw data re-processed.

## Troubleshooting

- **"Cannot find Andromeda"** — install MaxQuant from the official .NET
  installer; it bundles Andromeda. Standalone unzip lacks the search engine.
- **Low protein count (~500)** — usually means MBR was off, or the FASTA
  is small. Check tabs 2 + 3.
- **Identifications inconsistent across runs** — verify all 8 raw files
  are in the same parameter group (group 0).
- **OOM at the FDR step** — drop the parallel-search depth in tab 6 to
  half your core count.
