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
bash scripts/download_hic.sh   # optional, ~3 GB of Hi-C
```

Three things need manual download because they sit behind publisher or portal pages:
CytoCellDB supplementary file 1 (NAR Cancer 2024) into `data/cytocell_db/`, the Kim et al.
2020 supplementary tables (Nature Genetics) into `data/ecdna_labels/`, and the GDSC2
dose-response table into `data/gdsc/`. Raw downloads are not tracked in git; only the
derived feature matrices (`data/features/`), result CSVs (`data/validation/`) and gene
lists (`data/vulnerabilities/`) are.

## Running things

```bash
# features for module 1 (112 non-leaky features)
python scripts/extract_features.py --data-dir data --output data/features

# module 1
python main.py train --module former --data-dir data --epochs 200 --patience 30

# module 2
python scripts/generate_trajectories.py
python scripts/train_circularode_full.py --epochs 100 --batch_size 32 --lr 5e-4

# module 3
python scripts/train_vulncausal_full.py --epochs 50 --batch_size 16 --lr 5e-4 --irm_warmup 10
```

Validation and analysis live in `scripts/validate_*.py`, `scripts/run_*.py` and
`scripts/analyze_*.py`; each writes a CSV into `data/validation/`. `scripts/eclipse_demo.py`
runs the three modules end to end on a few synthetic patients. Model weights go to
`checkpoints/`, which is git-ignored.

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
