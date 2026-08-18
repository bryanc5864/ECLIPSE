# ECLIPSE

Extrachromosomal DNA (ecDNA) is present in roughly a third of cancers, drives oncogene
amplification, and segregates randomly at mitosis, which makes it a nasty driver of
resistance. ECLIPSE is a set of three independently trained models that attack different
parts of that problem:

- ecDNA-Former predicts whether a cell line carries ecDNA from copy number, expression and
  Hi-C derived features.
- CircularODE models ecDNA copy number dynamics under treatment as a physics-informed
  neural SDE, with a simpler GRU variant for next-step prediction.
- VulnCausal looks for ecDNA-specific therapeutic vulnerabilities in DepMap CRISPR screens
  using a VAE + IRM + NOTEARS pipeline, alongside a plain differential test.

The three modules share no representations and are trained on different sample
intersections. Composing them (formation probability -> dynamics -> vulnerabilities) is
post-hoc, not end-to-end.

## Install

```bash
git clone https://github.com/bryanc5864/ECLIPSE.git
cd ECLIPSE
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

A GPU helps but is not required. Tested on A100/V100 with Python 3.10.

## Data

Most of the inputs download themselves:

```bash
python main.py download --data-dir data --skip-large
```

Everything else, and where it has to end up:

| What | Where it goes | How to get it |
|---|---|---|
| DepMap CRISPR / expression / copy number / model table (~1.6 GB) | `data/depmap/{crispr,expression,copy_number,cell_line_info}.csv` | `python main.py download`; resolved through the DepMap portal API (`https://depmap.org/portal/download/api/downloads`, files `CRISPRGeneEffect.csv`, `OmicsExpressionProteinCodingGenesTPMLogp1.csv`, `OmicsCNGene.csv`, `Model.csv`). The release is whatever the API serves, so it is not pinned. |
| CytoCellDB supplementary file 1 — FISH ecDNA labels, 1,859 cell lines | `data/cytocell_db/CytoCellDB_Supp_File1.xlsx` (ships in this repo) | Fessler et al., NAR Cancer 2024, DOI `10.1093/narcan/zcae035`, <https://academic.oup.com/narcancer/article/6/3/zcae035> supplementary data. |
| Kim et al. 2020 ecDNA calls (AmpliconArchitect), used for the cross-source label check | `data/amplicon_repository/41588_2020_678_MOESM2_ESM.xlsx` (ships in this repo) | Nature Genetics 2020, <https://www.nature.com/articles/s41588-020-0678-2>, Supplementary Table 2. `scripts/extract_features.py` (legacy path) expects the same workbook at `data/ecdna_labels/kim2020_supplementary_tables.xlsx`. |
| GDSC2 dose response (Module 3 validation) | `data/gdsc/GDSC2_fitted_dose_response.xlsx` | <https://www.cancerrxgene.org/downloads/bulk_download> (release 8.5). |
| Hi-C, GM12878 `.mcool` at 50 kb — the only Hi-C input the feature code reads | `data/hic/GM12878.mcool` | 4D Nucleome, file `4DNFIXP4QG5B`; `python main.py download` (without `--skip-large`, ~30 GB) or <https://data.4dnucleome.org>. `scripts/download_hic.sh` is a different, optional route: it pulls K562/IMR90/HeLa `.hic` from Aiden Lab (Rao et al. 2014, GSE63525) and is *not* what `scripts/extract_hic_features.py` reads. |
| HumCFS fragile sites, COSMIC gene list | `data/supplementary/` (ships in this repo) | <https://webs.iiitd.edu.in/raghava/humcfs/humcfs.txt>. |
| ecSimulator + hg38 reference, for Module 2 trajectories | `ecSimulator/`, `data/reference/hg38.fa` | <https://github.com/AmpliconSuite/ecSimulator> (v0.7.1) and any GRCh38/hg38 FASTA. Both are git-ignored and neither ships here. |

Raw downloads are not tracked in git. What is tracked, and what every reported number was
computed from, is the derived material: feature matrices (`data/features/`), result CSVs
(`data/validation/`), gene lists (`data/vulnerabilities/`) and the two label workbooks
above. `data/features/hic_features.npz` is the precomputed Hi-C summary, so Module 1 can be
rebuilt without ever downloading a contact map.

Two gaps worth knowing about before you try to reproduce something:

- Module 2's 500 synthetic trajectories (`data/ecdna_trajectories/`) are not kept. Regenerating
  them needs ecSimulator plus an hg38 FASTA, and `scripts/generate_trajectories.py` passes no
  seed, so a regenerated set will not be identical to the one behind the Module 2 numbers.
- The feature matrices in `data/features/` were last regenerated on 2026-02-01, after the
  single-split Module 1 run and the 5-fold CV. The ablation, lineage-LOOCV, MLP-baseline and
  no-dosage numbers were produced from the files as they now stand; the 0.801 single-split
  figure and the 0.729 CV figure were not, and re-running `scripts/compute_significance.py`
  against the current split will not land on 0.801 (that run's validation split had 10
  positives, the current one has 17).

## Running things

```bash
# features for module 1 (112 non-leaky features; needs data/depmap/ + the CytoCellDB
# workbook + data/features/hic_features.npz)
python scripts/extract_nonleaky_features.py

# module 1
python main.py train --module former --data-dir data --epochs 200 --patience 30

# module 2 (needs ecSimulator/ and data/reference/hg38.fa)
python scripts/generate_trajectories.py
python scripts/train_circularode_full.py --epochs 100 --batch_size 32 --lr 5e-4

# module 3
python scripts/train_vulncausal_full.py --epochs 50 --batch_size 16 --lr 5e-4 --irm_warmup 10
```

Validation and analysis live in `scripts/validate_*.py`, `scripts/run_*.py` and
`scripts/analyze_*.py`; each writes a CSV into `data/validation/`. `scripts/eclipse_demo.py`
runs the three modules end to end on a few synthetic patients. Model weights go to
`checkpoints/`, which is git-ignored; only the best checkpoint per run is kept, but the
per-epoch validation metrics for every run survive as `checkpoints/**/logs/validation_log_*.csv`,
which is what the ablation, cross-validation and LOOCV scripts actually read back.
`scripts/extract_features.py` is the older TCGA/Kim-based extractor and does not produce the
112-feature matrices used here.

## Results

Module 1 reaches AUROC 0.801 on the single 85/15 split (1,383 cell lines, 8.9% ecDNA+), but
that number is optimistic: 5-fold stratified CV gives 0.729 +/- 0.042, and leave-one-lineage-out
CV averages about 0.66 with a huge spread (0.94 on blood, 0.45 on soft tissue). The advantage
over a random forest is not significant (bootstrap p = 0.075) with only 17 positives in the
validation set. Feature ablation says the 45 Hi-C features are dead weight -- removing them
improves AUROC (0.787 -> 0.796), because cnv_hic_X is just cnv_X scaled by a reference-genome
constant. Dropping the 9 dosage features helps too (retrained: 0.812).

Module 2 fits its own simulator well (correlation 0.993 for the next-step GRU, 0.615 for the
full 50-step SDE) and reproduces the published ecDNA-versus-HSR asymmetry from Lange et al.
2022 within error bars. All training trajectories are synthetic, so treat this as a sanity
check rather than validation. The physics ablation shows the constraints act as mild
regularization and are not the source of the fit.

Module 3 recovers a coherent set of hits -- CDK1, KIF11, NDC80, NCAPD2, SGO1, ORC6, MCM2,
PSMD7, SNRPF, DDX3X, BCL2L1 -- enriched for mitotic division and cell cycle (GSEA FDR 2e-4)
and 38x enriched for literature-supported genes over random gene sets. No individual gene
survives FDR correction across 17,453 tests, and no drug in GDSC2 shows significant ecDNA+
selectivity (Navitoclax comes closest at 1.24x, p = 0.066). The full causal model did not
converge: the NOTEARS acyclicity penalty stays around 11,600 and its top 100 genes overlap
the differential analysis by zero. The simpler differential and learned rankings are the ones
worth using.

A few numbers from earlier rounds are not in any results file, so they are recorded here.
Generation 2 of Module 1 (the same architecture on 67 DepMap-only features, no Hi-C) reached
AUROC 0.736 / AUPRC 0.419 / recall 65.2% on its own split; the random forest on those 67
features got 0.651, and on all 112 features 0.695. Those training logs are gone, so 0.736 is
not re-derivable without retraining. The cross-source label check in
`scripts/validate_ecdna_former.py` found 76.2% concordance (16/21 overlapping cell lines)
between CytoCellDB FISH calls and Kim et al. 2020 AmpliconArchitect calls, and the GBM39
isogenic pair came out essentially flat (EC 0.068 vs HSR 0.067) because those feature vectors
are hand-built; both of those are recomputable from the two workbooks that ship here.

One caveat that belongs next to the Module 1 numbers: 839 of the 1,383 cell lines have no
ecDNA call in CytoCellDB and are treated as negatives, so some fraction of the negative class
is unlabeled rather than known-negative (`data/validation/label_noise_*.csv`).

## Citation

```bibtex
@software{eclipse2026,
  title={ECLIPSE: Extrachromosomal Circular DNA Learning for Integrated
         Prediction of Synthetic-lethality and Expression},
  author={Cheng, Bryan and Zhang, Jasper},
  year={2026},
  url={https://github.com/bryanc5864/ECLIPSE}
}
```

MIT licensed. See LICENSE.
