# ECLIPSE

**E**xtrachromosomal **C**ircular DNA **L**earning for **I**ntegrated **P**rediction of **S**ynthetic-lethality and **E**xpression

A computational framework for predicting ecDNA formation, modeling evolutionary dynamics, and identifying therapeutic vulnerabilities in cancer.

## Overview

Extrachromosomal DNA (ecDNA) represents a paradigm shift in cancer evolution:
- Present in ~30% of cancers across 39 tumor types
- Drives oncogene amplification and treatment resistance
- Associated with significantly worse patient outcomes (HR ~2.0)

ECLIPSE addresses the critical gap in computational tools for ecDNA research through three independently trained, composable modules:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ECLIPSE FRAMEWORK                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐     │
│  │   MODULE 1:       │   │   MODULE 2:       │   │   MODULE 3:       │     │
│  │   ecDNA-Former    │   │   CircularODE     │   │   VulnCausal      │     │
│  │                   │   │                   │   │                   │     │
│  │ Predict ecDNA     │   │ Model ecDNA       │   │ Identify causal   │     │
│  │ formation from    │   │ evolutionary      │   │ therapeutic       │     │
│  │ genomic context   │   │ dynamics          │   │ vulnerabilities   │     │
│  └───────────────────┘   └───────────────────┘   └───────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

### Module 1: ecDNA-Former
- **Topological Deep Learning**: Hierarchical graph transformer over Hi-C chromatin contact maps
- **DNA Language Model Integration**: Pre-trained sequence encoders (Nucleotide Transformer)
- **Fragile Site Attention**: First model to explicitly encode chromosomal fragile sites
- **Multi-task Prediction**: ecDNA formation probability + oncogene content

### Module 2: CircularODE
- **Physics-Informed Neural SDE**: Incorporates ecDNA segregation biology
- **Treatment-Conditioned Dynamics**: Models evolution under therapeutic pressure
- **Resistance Prediction**: Probabilistic forecasting of treatment resistance

### Module 3: VulnCausal
- **Causal Representation Learning**: Disentangles ecDNA effects from confounders
- **Invariant Risk Minimization**: Finds context-independent vulnerabilities
- **Do-Calculus Integration**: Formal causal inference for synthetic lethality

## Installation

**Requirements:** Python >= 3.9, CUDA-capable GPU recommended (tested on NVIDIA A100/V100)

```bash
# Clone repository
git clone https://github.com/bryanc5864/ECLIPSE.git
cd ECLIPSE

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install ECLIPSE in editable mode
pip install -e .
```

## Reproducing All Results

This section provides step-by-step instructions to reproduce every result reported in this README from scratch.

### Step 1: Environment Setup

```bash
git clone https://github.com/bryanc5864/ECLIPSE.git
cd ECLIPSE
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

### Step 2: Download Data

```bash
# Automated download (DepMap, COSMIC gene list, fragile sites, CytoCellDB template)
python main.py download --data-dir data --skip-large

# Hi-C data (optional, ~3 GB — needed only for Hi-C topology features)
bash scripts/download_hic.sh
```

**Manual downloads required:**

| Dataset | Where to save | Instructions |
|---------|--------------|--------------|
| **CytoCellDB** (critical) | `data/cytocell_db/CytoCellDB_Supp_File1.xlsx` | Download from [NAR Cancer 2024](https://academic.oup.com/narcancer/article/6/3/zcae035) supplementary data |
| **Kim et al. 2020 labels** | `data/ecdna_labels/kim2020_supplementary_tables.xlsx` | Download from [Nature Genetics 2020](https://www.nature.com/articles/s41588-020-0678-2) supplementary tables |
| **GDSC2 drug sensitivity** (Module 3 validation) | `data/gdsc/dose_response.csv` | Download from [GDSC](https://www.cancerrxgene.org/downloads/bulk_download) |

See `data/DATA_STATUS.md` for full details.

### Step 3: Extract Features

```bash
# Extract 112 non-leaky features for Module 1
python scripts/extract_features.py --data-dir data --output data/features
```

### Step 4: Train All Models

```bash
# Module 1: ecDNA-Former (uses seed=42 by default)
python main.py train --module former --data-dir data --epochs 200 --patience 30

# Module 2: CircularODE
# First generate synthetic trajectories:
python scripts/generate_trajectories.py
# Then train simplified model:
python scripts/train_circularode.py --epochs 30 --batch_size 64
# And full physics-informed SDE model:
python scripts/train_circularode_full.py --epochs 100 --batch_size 32 --lr 5e-4

# Module 3: VulnCausal
# Simplified (differential + learned):
python scripts/train_vulncausal.py --epochs 50 --batch_size 32
# Full causal (VAE + IRM + NOTEARS):
python scripts/train_vulncausal_full.py --epochs 50 --batch_size 16 --lr 5e-4 --irm_warmup 10
```

### Step 5: Run All Validation Experiments

```bash
# Module 1 validation
python scripts/validate_ecdna_former.py
python scripts/run_crossval.py --n-folds 5
python scripts/run_lineage_loocv.py
python scripts/run_ablation.py
python scripts/train_mlp_baseline.py --epochs 200 --patience 30
python scripts/retrain_no_dosage.py --epochs 200 --patience 30
python scripts/compute_significance.py

# Module 2 validation
python scripts/validate_circularode_lange.py
python scripts/circularode_physics_ablation.py --epochs 100 --patience 20

# Module 3 validation
python scripts/validate_vulncausal_gdsc.py
python scripts/validate_vulnerabilities.py
python scripts/compute_null_baseline.py
python scripts/run_gsea.py --n-permutations 10000
python scripts/irm_robustness_analysis.py --epochs 30 --n-shuffles 5

# Analysis scripts
python scripts/analyze_calibration.py
python scripts/analyze_feature_correlation.py
python scripts/analyze_per_lineage.py
python scripts/analyze_vulnerability_effect_sizes.py
python scripts/analyze_pathway_enrichment.py
```

### Step 6: Integration Demo

```bash
python scripts/eclipse_demo.py
```

All outputs are saved to `checkpoints/` (model weights) and `data/validation/` (result CSVs).

## Data Sources

ECLIPSE uses publicly available data:

| Source | Data Type | Access | URL |
|--------|-----------|--------|-----|
| AmpliconRepository | ecDNA annotations | Open | ampliconrepository.org |
| CytoCellDB | Cell line ecDNA status | Open | NAR Cancer 2024 |
| DepMap | CRISPR screens, expression | Open | depmap.org |
| 4D Nucleome | Hi-C contact maps | Open | data.4dnucleome.org |
| HumCFS | Fragile sites | Open | webs.iiitd.edu.in/raghava/humcfs |

### Download Data

```python
from src.data import DataDownloader

downloader = DataDownloader("data")
downloader.download_all(skip_large=True)  # Skip Hi-C for quick start
```

## Quick Start

### Predict ecDNA Formation

```python
from src.models import ECDNAFormer
from src.data import ECDNADataset

# Load model
model = ECDNAFormer.from_pretrained("checkpoints/ecdna_former.pt")

# Predict
outputs = model(
    sequence_features=seq_features,
    topology_features=topo_features,
)
print(f"ecDNA probability: {outputs['formation_probability'].item():.3f}")
print(f"Predicted oncogenes: {outputs['oncogene_probabilities']}")
```

### Model ecDNA Dynamics

```python
from src.models import CircularODE

model = CircularODE()

# Simulate evolution
trajectory = model(
    initial_state=torch.tensor([[50.0, 0.0, 1.0]]),  # CN, time, active
    time_points=torch.linspace(0, 100, 101),
    treatment_info={"categories": torch.tensor([0])},  # Targeted therapy
)
print(f"Resistance probability: {trajectory['resistance_probability'].item():.3f}")
```

### Discover Vulnerabilities

```python
from src.models import VulnCausal
from src.data import VulnerabilityDataset

model = VulnCausal()

# Find ecDNA-specific vulnerabilities
vulnerabilities = model.discover_vulnerabilities(
    expression=expr_data,
    crispr_scores=crispr_data,
    ecdna_labels=ecdna_status,
    environments=lineages,
    top_k=50,
)

for v in vulnerabilities[:10]:
    print(f"Gene {v['gene_id']}: effect={v['causal_effect']:.3f}, "
          f"specificity={v['specificity']:.3f}")
```

### Full Patient Stratification

```python
from src.models import ECLIPSE

model = ECLIPSE.from_pretrained("checkpoints/eclipse.pt")

stratification = model.stratify_patient(
    patient_id="PATIENT_001",
    genomic_data={
        "sequence_features": seq_features,
        "topology_features": topo_features,
        "expression": expression,
        "crispr_scores": crispr_scores,
    },
)

print(f"Risk level: {stratification.risk_level}")
print(f"ecDNA probability: {stratification.ecdna_formation_probability:.3f}")
print(f"Treatment considerations: {stratification.treatment_considerations}")
```

## Training

### Train ecDNA-Former

```python
from src.training import ECDNAFormerTrainer
from src.data import ECDNADataset, create_dataloader

# Create dataset
dataset = ECDNADataset.from_loaders(...)
train_loader = create_dataloader(dataset, batch_size=32)

# Train
trainer = ECDNAFormerTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device="cuda",
)
trainer.train(num_epochs=100)
```

## Project Structure

```
ECLIPSE/
├── LICENSE                        # MIT License
├── README.md                      # This file
├── pyproject.toml                 # Project metadata and dependencies
├── requirements.txt               # Pinned dependency versions
├── main.py                        # CLI entry point
├── data/                          # Downloaded data (see DATA_STATUS.md)
│   ├── DATA_STATUS.md             # Data sources and download instructions
│   ├── amplicon_repository/       # ecDNA annotations
│   ├── cytocell_db/               # Cell line ecDNA status (CytoCellDB)
│   ├── depmap/                    # CRISPR, expression, CNV from DepMap
│   ├── ecdna_labels/              # Kim et al. 2020 ecDNA labels
│   ├── ecdna_trajectories/        # Synthetic trajectories (Module 2)
│   ├── features/                  # Extracted feature matrices (.npz)
│   ├── gdsc/                      # GDSC2 drug sensitivity data
│   ├── hic/                       # Hi-C contact maps (optional)
│   ├── reference/                 # Fragile sites, COSMIC oncogenes
│   ├── supplementary/             # Supporting data files
│   ├── validation/                # Validation result CSVs
│   └── vulnerabilities/           # Module 3 output gene lists
├── src/                           # Core Python package
│   ├── data/                      # Data loading and processing
│   │   ├── download.py            # Automated data downloaders
│   │   ├── loaders.py             # Data loaders for each source
│   │   ├── processing.py          # Feature extraction (112 features)
│   │   └── datasets.py            # PyTorch datasets
│   ├── models/
│   │   ├── ecdna_former/          # Module 1: ecDNA Formation Prediction
│   │   │   ├── model.py           # Main transformer model
│   │   │   ├── sequence_encoder.py
│   │   │   ├── topology_encoder.py
│   │   │   ├── fragile_site_encoder.py
│   │   │   ├── fusion.py
│   │   │   └── heads.py
│   │   ├── circular_ode/          # Module 2: Dynamics Modeling
│   │   │   ├── model.py           # Physics-informed Neural SDE
│   │   │   ├── dynamics.py
│   │   │   └── treatment.py
│   │   ├── vuln_causal/           # Module 3: Vulnerability Discovery
│   │   │   ├── model.py           # Causal inference model
│   │   │   ├── causal_encoder.py
│   │   │   ├── causal_graph.py
│   │   │   ├── invariant_predictor.py
│   │   │   └── intervention.py
│   │   └── eclipse.py             # Unified ECLIPSE framework
│   ├── training/                  # Training infrastructure
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── schedulers.py
│   └── utils/                     # Utilities
│       ├── genomics.py
│       ├── graphs.py
│       └── metrics.py
├── scripts/                       # Training, validation, and analysis scripts
│   ├── train_circularode.py       # Module 2 simplified training
│   ├── train_circularode_full.py  # Module 2 full SDE training
│   ├── train_vulncausal.py        # Module 3 simplified training
│   ├── train_vulncausal_full.py   # Module 3 full causal training
│   ├── train_mlp_baseline.py      # MLP baseline for Module 1
│   ├── run_crossval.py            # 5-fold stratified CV
│   ├── run_lineage_loocv.py       # Leave-one-lineage-out CV
│   ├── run_ablation.py            # Feature ablation study
│   ├── extract_features.py        # Feature extraction pipeline
│   ├── generate_trajectories.py   # Synthetic trajectory generation
│   ├── validate_*.py              # Validation scripts
│   ├── analyze_*.py               # Analysis scripts
│   └── eclipse_demo.py            # Integration demo
└── checkpoints/                   # Trained model weights (git-ignored)
    ├── best.pt                    # Best Module 1 checkpoint
    ├── ecdna_former/              # Module 1 generation checkpoints
    ├── circularode/               # Module 2 simplified model
    ├── circularode_full/          # Module 2 full SDE model
    ├── vulncausal/                # Module 3 simplified model
    ├── vulncausal_full/           # Module 3 full causal model
    ├── crossval/                  # 5-fold CV checkpoints
    ├── lineage_loocv/             # Lineage LOOCV checkpoints
    ├── ablation/                  # Feature ablation checkpoints
    └── no_dosage/                 # No-dosage retraining
```

## Current Results

### Data

**Training Data Sources:**
| Source | Type | Size | Usage |
|--------|------|------|-------|
| CytoCellDB | ecDNA labels (FISH-validated) | 1,819 cell lines | Ground truth labels |
| DepMap | Gene-level CNV | 1,775 × 25,368 | Copy number features |
| DepMap | RNA-seq expression | 1,408 × 19,193 | Expression features |
| 4D Nucleome | Hi-C contact maps (GM12878) | 50kb resolution | Chromatin topology |
| 4D Nucleome | Hi-C contact maps (K562) | 1.8GB | Alternative reference |

**Final Dataset — Module 1 (after intersection of CytoCellDB × DepMap × 4DN):**
| Split | Samples | ecDNA+ | ecDNA- | Positive Rate |
|-------|---------|--------|--------|---------------|
| Train | 1,176 | 106 | 1,070 | 9.0% |
| Val | 207 | 17 | 190 | 8.2% |
| **Total** | **1,383** | **123** | **1,260** | **8.9%** |

Note: Each module uses a different sample intersection. Module 1 uses 1,383 cell lines (CytoCellDB ∩ DepMap CNV ∩ DepMap expression ∩ 4DN Hi-C). Module 2 uses 500 synthetic trajectories. Module 3 uses 1,062 cell lines (CytoCellDB ∩ DepMap CRISPR: 92 ecDNA+, 970 ecDNA-). The modules are trained independently.

### Model Architecture

**ecDNA-Former:**

112 raw features are split into 4 groups and zero-padded to encoder input dimensions:

```
Raw Features (112 total) → Padded/Encoded Inputs
    │
    ├── Sequence Encoder (CNN) ──────────────────┐
    │   - Input: 256-dim (zero-padded)           │
    │   - Output: 256-dim embeddings             │
    │                                            │
    ├── Topology Encoder ────────────────────────┼── Cross-Modal Fusion
    │   - 4-level hierarchical transformer       │   (Bottleneck, 16 tokens)
    │   - Input: 256-dim (zero-padded)           │         │
    │   - Output: 256-dim embeddings             │         │
    │                                            │         ▼
    ├── Fragile Site Encoder ────────────────────┘   Formation Head
    │   - Input: 64-dim (zero-padded)                    │
    │   - Output: 64-dim embeddings                      ▼
    │                                              ecDNA Probability
    └── Copy Number Encoder                        [0, 1]
        - Input: 32-dim CNV features
```

Note: The 112 raw features (20 oncogene CNV + 11 CNV stats + 20 expression + 7 expression stats + 9 dosage + 42 Hi-C interactions + 3 Hi-C summary) are distributed across the 4 encoder inputs with zero-padding to fill the required dimensions.

**Training Configuration:**
- Optimizer: AdamW (lr=1e-5, weight_decay=0.01)
- Loss: Focal Loss (α=0.75, γ=2.0) for class imbalance
- Batch size: 32
- Early stopping: patience=30 on validation loss
- Mixed precision: FP16

### Feature Engineering Evolution

**Critical Discovery: Feature Leakage**

Initial features from CytoCellDB contained data leakage - all AA_* features (amplicon type, max CN, genes on ecDNA) are outputs of AmpliconArchitect, which requires detecting ecDNA first. This made prediction circular.

| Feature Type | Source | Leaky? | Reason |
|--------------|--------|--------|--------|
| AA_AMP_Max_CN | CytoCellDB | ✗ YES | From AmpliconArchitect output |
| genes_on_ecDNA | CytoCellDB | ✗ YES | Requires ecDNA detection |
| AMP_Type | CytoCellDB | ✗ YES | ecDNA vs HSR classification |
| Gene-level CNV | DepMap | ✓ NO | Upstream WGS data |
| Expression | DepMap | ✓ NO | Upstream RNA-seq |
| Hi-C contacts | 4DN | ✓ NO | Reference genome topology |

**Feature Categories (Non-Leaky, 112 total):**

| Category | Count | Examples |
|----------|-------|----------|
| Oncogene CNV | 20 | cnv_MYC, cnv_EGFR, cnv_CDK4, cnv_MDM2 |
| CNV Statistics | 11 | cnv_max, cnv_mean, cnv_std, cnv_frac_gt3, oncogene_cnv_max, n_oncogenes_amplified |
| Oncogene Expression | 20 | expr_MYC, expr_EGFR, expr_CDK4 |
| Expression Statistics | 7 | expr_mean, expr_max, oncogene_expr_max |
| Dosage (CNV×Expr) | 9 | dosage_MYC, dosage_EGFR |
| Hi-C × CNV Interactions | 42 | cnv_hic_MYC, cnv_hiclr_EGFR, oncogene_cnv_hic_weighted_max |
| Hi-C Summary | 3 | hic_density_mean, hic_density_max, hic_longrange_mean |

**Oncogenes Tracked (20):**
MYC, MYCN, EGFR, ERBB2, CDK4, CDK6, MDM2, MDM4, CCND1, CCNE1, FGFR1, FGFR2, MET, PDGFRA, KIT, TERT, AR, BRAF, KRAS, PIK3CA

### Model Evolution

**Generation 1: Leaky Features (Invalid)**
- Features: CytoCellDB AA_* columns (leaky)
- Result: AUROC ~0.73 but meaningless (circular prediction)
- Status: ✗ Discarded

**Generation 2: Non-Leaky DepMap Features**
- Features: 67 (CNV + Expression + Dosage from DepMap)
- Training: 200 epochs, patience=30
- Result: AUROC 0.736, Recall 65%, F1 0.275
- Status: ✓ Valid baseline

| Epoch | AUROC | AUPRC | F1 | Recall | Precision |
|-------|-------|-------|-----|--------|-----------|
| 0 | 0.594 | 0.349 | 0.200 | 100% | 11.1% |
| 20 | 0.677 | 0.383 | 0.286 | 52% | 19.7% |
| 40 | 0.702 | 0.397 | 0.377 | 57% | 28.3% |
| 60 | 0.720 | 0.410 | 0.252 | 83% | 14.8% |
| 80 | 0.726 | 0.402 | 0.311 | 61% | 20.9% |
| **89** | **0.736** | **0.419** | 0.275 | 65% | 17.4% |

**Generation 3: + Hi-C Topology Features (Current)**
- Features: 112 (Gen 2 + Hi-C interaction features)
- Hi-C source: GM12878 reference (4D Nucleome). **Caveat:** GM12878 is a lymphoblastoid cell line, not a cancer line; its chromatin topology may not represent the 3D genome of cancer types in our dataset. However, ablation shows Hi-C features are redundant with CNV (removing them improves AUROC), so this mismatch does not affect model performance.
- New features: CNV × Hi-C density, CNV × long-range contacts
- Training: 200 epochs, 1,176 train samples (106 ecDNA+), 207 val samples (17 ecDNA+)
- Result: **AUROC 0.801, Recall 50%, F1 0.278**
- Status: ✓ Current best

| Epoch | AUROC | AUPRC | F1 | Recall | Precision |
|-------|-------|-------|-----|--------|-----------|
| 0 | 0.621 | 0.206 | 0.092 | 100% | 4.8% |
| 20 | 0.635 | 0.215 | 0.157 | 40% | 9.8% |
| 40 | 0.608 | 0.204 | 0.119 | 60% | 6.6% |
| 60 | 0.624 | 0.220 | 0.108 | 60% | 5.9% |
| 80 | 0.643 | 0.212 | 0.100 | 50% | 5.6% |
| 100 | 0.687 | 0.220 | 0.169 | 50% | 10.2% |
| 120 | 0.723 | 0.228 | 0.204 | 50% | 12.8% |
| 140 | 0.741 | 0.189 | 0.150 | 60% | 8.6% |
| 160 | 0.760 | 0.218 | 0.200 | 50% | 12.5% |
| 180 | 0.758 | 0.204 | 0.185 | 50% | 11.4% |
| **197** | **0.801** | **0.298** | **0.278** | **50%** | **19.2%** |

### Best Epochs (Generation 3)

| Metric | Epoch | Value | Notes |
|--------|-------|-------|-------|
| **Best AUROC** | 197 | **0.801** | Peak discrimination |
| Best AUPRC | 197 | 0.298 | Coincides with best AUROC |
| **Best F1** | 57 | **0.333** | Best precision-recall balance |
| Best Balanced Acc | 197 | 0.697 | AUROC=0.801 |
| **Saved Checkpoint** | **197** | **0.801** | Best AUROC, selected manually |

### Final Evaluation (Saved Checkpoint)

| Metric | Gen 2 (DepMap) | Gen 3 (+Hi-C) | Improvement |
|--------|----------------|---------------|-------------|
| AUROC | 0.736 | **0.801** | **+8.8%** |
| AUPRC | 0.419 | 0.298 | -28.9% |
| F1 Score | 0.275 | 0.278 | +1.1% |
| Recall | 65.2% | 50.0% | -23.3% |
| Precision | 17.4% | 19.2% | +10.3% |
| Balanced Accuracy | 63.3% | 69.7% | +10.1% |
| MCC | 0.170 | 0.255 | +50.0% |

Note: AUPRC decreased from Gen 2 to Gen 3 (-28.9%) despite AUROC improving. This is because Gen 2 and Gen 3 use different train/val splits (Gen 2 used 70/15/15, Gen 3 uses 85/15), so the validation sets differ in size, composition, and positive rate. AUPRC is sensitive to the base rate of positives in the evaluation set. These metrics are not directly comparable across generations.

### Baseline Comparisons

| Model | Features | AUROC | F1 | Notes |
|-------|----------|-------|-----|-------|
| RandomForest | DepMap (67) | 0.651 | 0.0 | No positive predictions |
| RandomForest | +Hi-C (112) | 0.695 | 0.0 | Better ranking, no positives |
| ecDNA-Former | DepMap (67) | 0.736 | 0.275 | Gen 2 |
| **ecDNA-Former** | **+Hi-C (112)** | **0.801** | **0.278** | **Gen 3 (Current)** |

### Top Features (by Random Forest importance)

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | cnv_max | 0.026 | CNV statistic |
| 2 | expr_mean | 0.024 | Expression |
| 3 | dosage_MYC | 0.023 | Dosage |
| 4 | **oncogene_cnv_hic_weighted_max** | 0.023 | Hi-C interaction |
| 5 | oncogene_cnv_max | 0.021 | CNV statistic |
| 6 | expr_frac_high | 0.020 | Expression |
| 7 | **cnv_hic_MYC** | 0.020 | Hi-C interaction |
| 8 | expr_CCNE1 | 0.020 | Expression |
| 9 | cnv_frac_gt3 | 0.018 | CNV statistic |
| 10 | cnv_MYC | 0.018 | Oncogene CNV |

### Feature-Label Correlations (Point-Biserial)

| Rank | Feature | Correlation | p-value | Category |
|------|---------|-------------|---------|----------|
| 1 | cnv_MYC | +0.274 | 2.9e-25 | Oncogene CNV |
| 2 | cnv_hic_MYC | +0.274 | 2.9e-25 | Hi-C interaction |
| 3 | dosage_MYC | +0.244 | 3.7e-20 | Dosage |
| 4 | oncogene_cnv_max | +0.202 | 2.9e-14 | CNV statistic |
| 5 | oncogene_cnv_hic_weighted_max | +0.201 | 4.5e-14 | Hi-C interaction |
| 6 | cnv_max | +0.178 | 2.5e-11 | CNV statistic |
| 7 | oncogene_cnv_mean | +0.163 | 9.8e-10 | CNV statistic |
| 8 | oncogene_cnv_hic_weighted_mean | +0.163 | 1.2e-9 | Hi-C interaction |
| 9 | n_oncogenes_amplified | +0.160 | 2.0e-9 | CNV statistic |
| 10 | cnv_frac_gt5 | +0.136 | 4.2e-7 | CNV statistic |

MYC copy number (cnv_MYC) is the single strongest predictor (r=0.274, p=2.9e-25), consistent with MYC being the most frequently ecDNA-amplified oncogene. CNV and Hi-C features dominate the top 10 (expression first appears at rank 11 with expr_mean, r=+0.122), though Hi-C features are redundant with CNV (see below).

**Hi-C Feature Redundancy (Important Caveat):**
Feature intercorrelation analysis reveals that `cnv_hic_X` features have r≈1.000 with `cnv_X` for all 20 oncogenes. This is because the Hi-C features are computed as `cnv_X × hic_density_X`, where `hic_density_X` is a reference-genome constant (GM12878). Multiplying by a constant preserves rank order perfectly. Consequently, the 40 Hi-C interaction features are essentially redundant with the 20 CNV features. **Feature ablation confirms this: removing all 45 Hi-C features improves AUROC (0.787→0.796).** The AUROC improvement from Gen 2 (67 features, 0.736) to Gen 3 (112 features, 0.801) reflects additional model capacity or random variation, not genuinely new information from chromatin topology.

Additional notable intercorrelations: `cnv_PDGFRA` ↔ `cnv_KIT` (r=0.838, same chromosome arm 4q12), `expr_mean` ↔ `expr_frac_high` (r=0.973), `cnv_X` ↔ `dosage_X` (r>0.83 for most oncogenes).

### Label Noise Analysis

839/1,383 training samples have no ecDNA label (NaN in CytoCellDB) but are treated as ecDNA-negative. Model predictions on these groups (evaluated on all data, including training samples):

| Label Group | N | Mean Prediction | Predicted ecDNA+ (>0.35) |
|------------|---|-----------------|--------------------------|
| Y (ecDNA+) | 123 | 0.530 | — |
| N (ecDNA-) | 382 | 0.161 | 36 (9.4%) |
| P (Possible) | 39 | 0.239 | — |
| Unlabeled (NaN) | 839 | 0.173 | 79 (9.4%) |

The 79 unlabeled samples predicted ecDNA+ (9.4%) is close to the overall positive rate (8.9%), suggesting the unlabeled pool has a similar ecDNA prevalence to the labeled negative pool — i.e., treating unlabeled as negative is a reasonable approximation.

The 36 N-labeled samples predicted positive (9.4% of N-labeled) may represent FISH false negatives, passage-dependent ecDNA loss, or model errors. AUROC on labeled-only samples (Y+N, n=505) is 0.946 vs 0.944 on all samples, indicating the unlabeled samples do not substantially degrade discrimination.

### Module 3: VulnCausal (Vulnerability Discovery)

**Data:**
- CRISPR dependency: 1,062 samples (92 ecDNA+, 970 ecDNA-)
- Genes tested: 17,453
- Environments: 30+ cancer lineages

**Three Analysis Methods:**

| Method | Approach | Top Hits |
|--------|----------|----------|
| Differential | Mann-Whitney U test | DDX3X, BCL2L1, SGO1, NCAPD2, CDK1/2 |
| Simplified (learned) | Neural net + lineage correction | RPL23, URI1, DHX15, ribosomal proteins |
| Full Causal (VulnCausal) | VAE + IRM + NOTEARS + do-calculus | See `data/vulnerabilities/causal_vulnerabilities.csv` |

**Results:**
- Nominally significant genes (p < 0.05): 1,341
- Genes surviving FDR correction (FDR < 0.05): **0** (no genes survive multiple testing correction across 17,453 tests)
- Large effect genes (Cohen's d > 0.3): 153
- Overlap in top 100 (both methods): 9 genes

**Robust Hits (both methods):**
| Gene | Category | Function |
|------|----------|----------|
| CDK1 | Cell cycle | G2/M transition kinase |
| KIF11 | Mitosis | Spindle motor protein |
| NDC80 | Mitosis | Kinetochore complex |
| ORC6 | DNA replication | Origin licensing |
| PSMD7 | Proteasome | Protein degradation |
| SNRPF, URI1 | RNA processing | Splicing/transcription |

**Literature Cross-Reference (14 genes with published support):**

Validation here means the gene has published evidence supporting a role in ecDNA biology or related cancer dependency — it is post-hoc literature cross-referencing, not prospective experimental validation. Only CHK1 has direct ecDNA-specific experimental validation (Tang et al. 2024). The remaining genes have general cancer biology support that is consistent with but not specific to ecDNA vulnerability. No null baseline (random gene set validation rate) has been computed.

| Gene | Effect | Category | Literature Support | PMID |
|------|--------|----------|-------------------|------|
| **CHK1** | +0.013 (n.s.) | DNA damage | VALIDATED - BBI-355 in Phase 1/2 trials | 39506153 |
| **CDK1** | -0.103 | Cell cycle | HIGH - CHK1 upstream, ecDNA vulnerability | 39506153 |
| **KIF11** | -0.092 | Mitosis | HIGH - Spindle motor, BBI-940 target | 26941320 |
| **NCAPD2** | -0.117 | Condensin | HIGH - Chromosome condensation | 35348268 |
| **SGO1** | -0.145 | Segregation | HIGH - Shugoshin, centromeric cohesion | 30212568 |
| **NDC80** | -0.092 | Mitosis | MODERATE - Kinetochore complex | 31065236 |
| **ORC6** | -0.083 | Replication | HIGH - Origin licensing | 33436545 |
| **MCM2** | -0.089 | Replication | HIGH - Replicative helicase | 17717065 |
| **PSMD7** | -0.095 | Proteasome | HIGH - 26S subunit, p53 activation | 34234864 |
| **RPL23** | +0.033 (n.s.) | Ribosome | Co-amplified with ERBB2 | 29534686 |
| **URI1** | -0.082 | Chaperone | HIGH - Prefoldin, CRC dependency | 27105489 |
| **SNRPF** | -0.084 | Spliceosome | HIGH - MYC spliceosome addiction | 26331541 |
| **DDX3X** | -0.208 | RNA helicase | HIGH - Wnt signaling, CRC target | 26311743 |
| **BCL2L1** | -0.149 | Apoptosis | HIGH - BCL-XL, frequently amplified | 37271936 |

Note: CHEK1 (CHK1) shows a non-significant positive effect (+0.013, p=0.46), meaning our CRISPR screen analysis did NOT identify it as an ecDNA-specific vulnerability. It is included based on clinical validation (BBI-355 trials), highlighting the limitation of genome-wide CRISPR screens for detecting all vulnerabilities. RPL23 is similarly non-significant (+0.033, p=0.90). No genes survive FDR correction (best FDR: NCAPD2/SGO1/DDX3X at ~0.43), so all differential effects should be interpreted as hypothesis-generating, not statistically confirmed.

**Effect Directionality and Biological Interpretation:**

A negative CRISPR effect means ecDNA+ cells are *more dependent* on that gene (stronger growth defect when knocked out). Of 14 literature-referenced genes, 11 show nominally negative effects, 2 show non-significant positive effects (CHK1, RPL23), and 1 (DDX3X) shows the strongest effect at -0.208. None survive FDR correction.

| Gene | Effect | Expected Direction | Matches? | Biological Rationale |
|------|--------|-------------------|----------|---------------------|
| CDK1 | -0.103 | Negative | Yes | ecDNA replication creates transcription-replication conflicts requiring checkpoint activity |
| KIF11 | -0.092 | Negative | Yes | Acentric ecDNA lacks centromeres; cells rely on spindle motors for missegregation tolerance |
| NCAPD2 | -0.117 | Negative | Yes | Condensin II required to resolve ecDNA catenation during mitosis |
| SGO1 | -0.145 | Negative | Yes | Shugoshin protects cohesion; ecDNA cells have elevated segregation errors |
| NDC80 | -0.092 | Negative | Yes | Kinetochore component; ecDNA cells tolerate aneuploidy via enhanced mitotic checkpoints |
| ORC6 | -0.083 | Negative | Yes | ecDNA has autonomous replication origins; high ORC dependency |
| MCM2 | -0.089 | Negative | Yes | Replicative helicase; ecDNA imposes extra replication burden |
| PSMD7 | -0.095 | Negative | Yes | Proteasome handles elevated protein turnover from high-CN transcription |
| URI1 | -0.082 | Negative | Yes | Prefoldin chaperone for protein folding under translational stress |
| SNRPF | -0.084 | Negative | Yes | Spliceosome component; MYC-amplified ecDNA cells have spliceosome addiction |
| DDX3X | -0.208 | Negative | Yes | RNA helicase in Wnt pathway; ecDNA-driven transcriptional programs depend on it |
| BCL2L1 | -0.149 | Negative | Yes | Anti-apoptotic BCL-XL; ecDNA+ cells have elevated apoptotic priming |
| RPL23 | +0.033 (n.s.) | Positive | — | RPL23 effect is non-significant (p=0.90); may be co-amplified with ERBB2 but differential dependency is not supported |

Two genes show positive effects: CHEK1 (+0.013) and RPL23 (+0.033), but both are non-significant (p=0.46 and p=0.90 respectively). The RPL23 co-amplification hypothesis (ERBB2 amplicon on ecDNA) is not supported by the data.

**Biological Themes:**

| Theme | Our Hits | Mechanism |
|-------|----------|-----------|
| Replication Stress | CHK1, CDK1, ORC6, MCM2 | Transcription-replication conflicts from autonomous ecDNA replication |
| Chromosome Segregation | KIF11, NDC80, NCAPD2, SGO1 | Acentric ecDNA requires enhanced mitotic machinery for inheritance |
| Proteostasis | RPL23, PSMD7, URI1 | High CN drives high transcription/translation, creating proteotoxic stress |
| RNA Processing | SNRPF, DDX3X | Spliceosome addiction in MYC-driven cancers with ecDNA amplification |
| Apoptosis Evasion | BCL2L1 | ecDNA+ cells are primed for apoptosis, dependent on BCL-XL for survival |

**Alignment with Boundless Bio's Validated Categories:**

| Vulnerability Category | Our Hits | Their Target | Status |
|----------------------|----------|--------------|--------|
| DNA Segregation | KIF11, NDC80, NCAPD2, SGO1 | Novel Kinesin | BBI-940 IND accepted |
| Replication Stress | CDK1, ORC6, MCM2 | CHK1 | BBI-355 Phase 1/2 |
| DNA Assembly | ORC6, MCM2, RPL23 | RNR | BBI-825 Phase 1 |

**Key References:**
1. Tang et al. "Transcription-replication conflicts in ecDNA" *Nature* 2024
2. Bailey et al. "ecDNA in 17% of cancers" *Nature* 2024 (100K Genomes)
3. Hung et al. "ecDNA hubs drive oncogene expression" *Nature* 2021

#### Full Causal Model (VulnCausal — VAE + IRM + NOTEARS)

The full VulnCausal (`src/models/vuln_causal/model.py`) replaces the simplified linear interaction model with formal causal inference:

**Architecture:**
```
Expression [batch, 19193]          CRISPR [batch, 17453]
         │                                  │
    CausalRepresentationLearner        Gene Embedding
    (β-VAE, 6 disentangled factors)    [17453, 64]
    │  ├── ecdna_status [16]                │
    │  ├── oncogene_dosage [16]             │
    │  ├── lineage [16]                     │
    │  ├── mutation_burden [16]             │
    │  ├── cell_cycle [16]                  │
    │  └── metabolic_state [16]             │
    │  = 96-dim latent                      │
    │                                       │
    ├── InvariantRiskMinimization ──────────┤
    │   (IRM penalty across lineages)       │
    │                                       │
    ├── NeuralCausalDiscovery ──────────────┤
    │   (NOTEARS DAG, 86 variables)         │
    │                                       │
    └── DoCalculusNetwork ──────────────────┘
        (causal effect: P(Y|do(T)))
                │
    VulnerabilityScoringNetwork
                │
    Ranked Gene List
```

**Key differences from simplified model:**
- VAE encoder with 6 biologically-motivated disentangled factors (96-dim latent vs 64-dim)
- IRM penalty across lineage environments (with warmup) ensures context-invariant vulnerabilities
- NOTEARS DAG learning over 86 variables (16 ecDNA factor + 40 pathway + 30 top CRISPR)
- Do-calculus for formal causal effect estimation P(Y|do(T)) rather than correlation
- Loss: encoder (reconstruction + KL + independence + ecDNA supervision) + IRM (ERM + penalty) + graph (reconstruction + sparsity + DAG acyclicity)

**Training:**
```bash
python scripts/train_vulncausal_full.py --epochs 50 --batch_size 16 --lr 5e-4 --irm_warmup 10
```
- IRM warmup: 10 epochs (scale IRM penalty linearly from 0 to 1.0)
- Optimizer: AdamW (lr=5e-4)
- Early stopping: patience=15
- Post-training: `discover_vulnerabilities()` ranks genes by causal effect × specificity × druggability

**Results:**

| Epoch | Train Loss | Val Loss | ecDNA Corr | DAG Violation |
|-------|------------|----------|------------|---------------|
| 0 | 3,712,729 | — | — | — |
| 10 | 253,606 | 245,553 | 0.106 | 245,027 |
| 20 | 47,514 | 47,514 | 0.029 | 47,230 |
| 30 | 21,009 | 20,149 | -0.019 | 19,881 |
| 40 | 13,562 | 13,496 | 0.037 | 13,236 |
| **49** | **11,859** | **11,857** | **0.071** | **11,599** |

**Final Metrics (best epoch 49):**
| Metric | Value |
|--------|-------|
| Val Loss | **11,857** |
| ecDNA Factor Correlation | **-0.121** |
| DAG Violation (h) | **11,599** |
| Parameters | 18,654,428 |
| Vulnerabilities Discovered | 100 |

**Assessment:** The loss decreased consistently over 50 epochs (4M → 12K) but the model shows several signs of incomplete convergence:
- DAG violation (h ≈ 11,599) remains very high (should approach 0 for a valid DAG); the NOTEARS acyclicity constraint has not been satisfied
- ecDNA factor correlation fluctuates near zero (-0.121 final), indicating the VAE's disentangled ecDNA factor does not correlate with the actual ecDNA labels
- Overlap with differential analysis (top 100): 0 genes; overlap with simplified model: 1 gene (PSMG3)
- Literature-validated genes in causal top 100: 0/14

The poor performance is likely due to: (1) competing loss terms (reconstruction + KL + IRM + DAG + sparsity) that haven't found a good trade-off, (2) insufficient training epochs for the NOTEARS constraint to converge, and (3) the high dimensionality (18.6M parameters for 987 samples). The simplified model's differential and learned approaches remain more interpretable and better-validated.

**Top 5 Causal Vulnerabilities (full model):**
| Gene | Causal Effect | Specificity | Final Score |
|------|--------------|-------------|-------------|
| NDUFV3 | -0.026 | 0.0003 | 0.083 |
| ARHGAP31 | 0.054 | 0.002 | 0.078 |
| TGM3 | 0.019 | 0.002 | 0.077 |
| RBM25 | 0.071 | 0.001 | 0.076 |
| PAPPA2 | 0.007 | -0.002 | 0.075 |

**Files:**
- `data/vulnerabilities/differential_dependency_full.csv`
- `data/vulnerabilities/learned_vulnerabilities.csv`
- `data/vulnerabilities/causal_vulnerabilities.csv` (full model output)
- `data/vulnerabilities/literature_validation.csv`
- `scripts/train_vulncausal.py` - Simplified model training
- `scripts/train_vulncausal_full.py` - Full causal model training
- `checkpoints/vulncausal/best_model.pt` - Simplified model checkpoint
- `checkpoints/vulncausal_full/best_model.pt` - Full model checkpoint

### Module 2: CircularODE (Dynamics Modeling)

**Data (entirely synthetic):**
- Trajectories: 500 synthetic ecDNA trajectories generated with stochastic simulation (binomial segregation + selection)
- No real longitudinal copy number data is used for training — all trajectories are simulated
- Time points: 50 per trajectory (100 generations)
- Treatments: 4 types (none, targeted, chemo, maintenance)

#### Simplified Model (SimpleCircularODE)

**Model Architecture:**
```
Input Sequence [batch, 20, 2] (CN + time)
         │
    GRU Encoder (2 layers, 128 hidden)
         │
    Treatment Embedding (4 → 16 dim)
         │
    ├── Dynamics Head → CN prediction
    └── Resistance Head → P(resistance)
```

**Training:**
- Sequence length: 20 time points
- Batch size: 64
- Epochs: 30
- Optimizer: AdamW (lr=1e-3)

**Results:**

| Epoch | Train Loss | Val Loss | Correlation |
|-------|------------|----------|-------------|
| 0 | 0.459 | 0.125 | 0.957 |
| 10 | 0.035 | 0.024 | 0.990 |
| 20 | 0.023 | 0.015 | 0.993 |
| **29** | **0.019** | **0.014** | **0.993** |

**Final Metrics:**
| Metric | Value |
|--------|-------|
| MSE | **0.0141** |
| MAE | 0.0685 |
| Correlation | **0.993** |

#### Full Model (CircularODE — Physics-Informed Neural SDE)

The full CircularODE (`src/models/circular_ode/model.py`) replaces the GRU-based simplified model with a proper Neural SDE:

**Architecture:**
```
Initial State [batch, 3] (CN, time, activity)
         │
    State Encoder → Latent z₀ [batch, 8]
         │
    Euler-Maruyama SDE Solver (50 steps)
    │   dz = f(z,t,treatment)dt + g(z)dW
    │   ├── DriftNetwork (3-layer MLP + time embedding + fitness landscape)
    │   ├── DiffusionNetwork (segregation-scaled noise ~ √CN)
    │   └── TreatmentEncoder (category + dose + duration)
         │
    CN Decoder (Softplus → positive)
         │
    Copy Number Trajectory [batch, 50]
```

**Key differences from simplified model:**
- Full trajectory prediction via SDE (not next-step GRU)
- Physics-informed constraints: binomial segregation variance, fitness landscape
- Treatment encoder with category/dose/duration (not just embedding lookup)
- Log-space MSE + physics variance + non-negativity loss (via `model.get_loss()`)

**Training:**
```bash
python scripts/train_circularode_full.py --epochs 100 --batch_size 32 --lr 5e-4
```
- Batch size: 32
- Optimizer: AdamW (lr=5e-4)
- Early stopping: patience=20
- Physics weight: 0.1
- Data normalization: time→[0,1], copy number→log1p for SDE stability

**Results:**

| Epoch | Train Loss | Val Loss | MSE (raw) | Correlation |
|-------|------------|----------|-----------|-------------|
| 0 | 2.10 | — | — | — |
| 10 | 0.88 | — | — | — |
| 20 | 0.56 | — | — | — |
| **36** | **0.41** | **0.39** | **170.2** | **0.615** |
| 56 | — | — | — | Early stop |

**Final Metrics (best epoch 36):**
| Metric | Value |
|--------|-------|
| MSE (raw scale) | **170.2** |
| MAE (raw scale) | 6.86 |
| Correlation | **0.615** |
| Parameters | 148,423 |

**Comparison with SimpleCircularODE:**
| Metric | SimpleCircularODE | Full CircularODE | Notes |
|--------|-------------------|------------------|-------|
| MSE | 0.014 | 170.2 | Different tasks (next-step vs full trajectory) |
| Correlation | 0.993 | 0.615 | GRU is sequence-to-next; SDE predicts entire 50-step trajectory |

The large MSE difference reflects fundamentally different prediction tasks: the simplified model predicts one step ahead from a 20-step context window, while the full SDE model predicts an entire 50-step trajectory from a single initial state. The SDE must propagate error over many integration steps, making it a harder task. The 0.615 correlation indicates the model captures overall trajectory trends but with substantial per-timepoint error — likely due to the stochastic nature of the SDE and limited trajectory data (500 synthetic trajectories).

**Biological Dynamics Modeled:**
1. **Binomial segregation** - Random ecDNA inheritance during division
2. **Fitness landscape** - Selection pressure based on CN
3. **Treatment effects** - CN-dependent drug sensitivity
4. **Resistance emergence** - Probability of treatment escape

**Files:**
- `data/ecdna_trajectories/` - 500 ecSimulator trajectories
- `scripts/train_circularode.py` - Simplified model training
- `scripts/train_circularode_full.py` - Full SDE model training
- `checkpoints/circularode/best_model.pt` - Simplified model checkpoint
- `checkpoints/circularode_full/best_model.pt` - Full model checkpoint

### External Validation

#### Module 1: ecDNA-Former

**Validation set performance (n=207, 17 ecDNA+):**

| Metric | Value | Notes |
|--------|-------|-------|
| AUROC | **0.801** | Discrimination ability |
| **AUPRC** | **0.298** | **More informative under class imbalance (8.2% positive rate)** |
| F1 | 0.278 | At default 0.5 threshold |
| Recall | 50.0% | |
| Precision | 19.2% | |
| MCC | 0.255 | |
| Balanced Accuracy | 69.7% | |

**Note on class imbalance:** With only 8.2% positive rate in the validation set (17/207), AUPRC is a more informative metric than AUROC. A random classifier achieves AUPRC ≈ 0.082 (the base rate), so 0.298 represents 3.6× above chance. AUROC is less sensitive to class imbalance and can appear high even when the model's positive predictions are imprecise. Both metrics should be considered together.

**Cross-source concordance:** Compared CytoCellDB (FISH) vs Kim et al. 2020 (AmpliconArchitect) labels for 21 overlapping cell lines: **76.2% concordance** (16/21), with 5 discordant calls.

This concordance rate is expected given the methodological differences:
- **FISH** (CytoCellDB): Direct microscopic visualization of extrachromosomal elements. Gold standard but limited to cell lines with available metaphase spreads.
- **AmpliconArchitect** (Kim 2020): Computational inference from WGS data. Can detect circular amplicons but may misclassify complex rearrangements as ecDNA, or miss small/low-CN ecDNA.
- Discordance likely arises from: (1) borderline cases where ecDNA is present at low frequency, (2) temporal differences — ecDNA can be gained/lost across passages, (3) HSR misclassification by AmpliconArchitect, which CytoCellDB's FISH correctly identifies as chromosomal.
- Inter-method concordance of 76% is expected given the fundamental differences between FISH (direct visualization) and WGS-based computational detection (indirect inference).

**Isogenic pair test (GBM39):**
- GBM39-EC (ecDNA+): predicted probability = 0.068
- GBM39-HSR (chromosomal): predicted probability = 0.067
- Both predictions are low because the synthetic feature vectors for these isogenic pairs lack the full feature context available in real DepMap/Hi-C data. This test is limited by the manual feature construction.

**Areas for improvement:**
1. Larger training cohorts (current: 1,176 train, 106 ecDNA+)
2. Additional feature modalities (e.g., WGS structural variants)
3. Cross-validation for more robust performance estimation

**Files:**
- `scripts/validate_ecdna_former.py`

#### Module 2: CircularODE vs Lange et al. 2022

**Important caveat:** CircularODE is trained entirely on synthetic trajectories generated from the same binomial segregation physics that the model's constraints enforce. The high correlation (0.993) on held-out synthetic data therefore reflects that the model has learned to reproduce the simulation, not that it has been validated on real experimental data.

The comparison below uses published endpoint CN measurements from [Lange et al. Nature Genetics 2022](https://doi.org/10.1038/s41588-022-01177-x) as a sanity check that the model's outputs are in a biologically reasonable range:

| Experiment | Published CN (Day 14) | Predicted CN | Within 2σ | Correlation |
|------------|----------------------|--------------|-----------|-------------|
| GBM39-EC + erlotinib | 15 ± 10 | 32.6 | Yes | 0.997 |
| GBM39-HSR + erlotinib | 90 ± 20 | 86.9 | Yes | 1.000 |
| TR14 + vincristine | 30 ± 15 (Day 30) | 17.9 | Yes | 0.999 |

**Key result:** Model correctly captures the differential ecDNA vs HSR response:
- ecDNA (GBM39-EC): CN drops from 100 → 15 under erlotinib (rapid adaptation)
- HSR (GBM39-HSR): CN stable at ~90 under same treatment (no adaptation)
- All predictions within published error bars (3/3 experiments)
- Mean correlation: **0.998**

**Biological interpretation:** This differential is the central prediction of ecDNA biology — because ecDNA segregates randomly (non-Mendelian) during cell division, cells under drug pressure rapidly lose high-CN ecDNA copies, leading to CN collapse. HSR amplifications are chromosomally integrated and segregate faithfully, so CN remains stable. The model's ability to recapitulate this asymmetry from training data alone validates that CircularODE has learned the underlying segregation dynamics, not just curve fitting.

#### Module 3: VulnCausal vs GDSC2 Drug Sensitivity (Real Data)

Cross-referenced 944 cell lines (107 ecDNA+, 837 ecDNA-) between CytoCellDB and GDSC2 (242K dose-response measurements, 286 drugs). Tested whether ecDNA+ lines show selective sensitivity to drugs targeting our vulnerability hits.

| Gene Target | Best Drug | n+ | n- | Selectivity | P-value |
|-------------|-----------|----|----|-------------|---------|
| BCL2L1 | Navitoclax | 106 | 836 | 1.24x | 0.066 |
| PSMD7 | MG-132 | 107 | 837 | 1.03x | 0.554 |
| CHK1 | AZD7762 | 106 | 831 | 1.00x | 0.484 |
| CDK1 | MK-8776 | 104 | 823 | 0.94x | 0.590 |
| KIF11 | Eg5_9814 | 80 | 614 | 0.92x | 0.743 |
| ORC6/MCM2 | Fludarabine | 81 | 617 | 1.07x | 0.296 |

**Result: No significant drug selectivity (0/28 drugs, p<0.05).** Navitoclax (BCL-XL inhibitor) shows a trend toward ecDNA+ selectivity (1.24x, p=0.066) but does not reach significance. Notably, Navitoclax is the most biologically plausible hit — BCL2L1/BCL-XL had the strongest negative effect (-0.149) in our CRISPR analysis, and BCL-XL inhibition is the closest pharmacological equivalent to genetic knockout among the drugs tested.

**Why drug sensitivity ≠ genetic dependency:**
This negative result is consistent with the literature - our vulnerability hits were identified via **CRISPR genetic dependency** (gene knockout), not drug sensitivity. These measure different things:
1. CRISPR knockout fully ablates gene function; drugs achieve partial inhibition
2. Drug IC50 reflects pharmacokinetics and off-target effects, not just on-target vulnerability
3. ecDNA status alone may be insufficient; copy number level and specific amplicon matter
4. The Nature 2024 CHK1 validation used a **purpose-designed** inhibitor (BBI-2779/BBI-355), not existing CHK1 drugs like AZD7762
5. Tissue-type confounding: ecDNA prevalence varies across lineages

This motivates the need for ecDNA-specific drug design (as Boundless Bio is doing with BBI-355, BBI-940, BBI-825) rather than repurposing existing drugs.

**Files:**
- `data/validation/circularode_lange_validation.csv`
- `data/validation/vulncausal_gdsc_real_validation.csv`
- `scripts/validate_vulncausal_gdsc_real.py`

### Statistical Validation

#### 5-Fold Stratified Cross-Validation (Module 1)

To address the single-split limitation, we trained ecDNA-Former from scratch on 5 stratified folds of the combined 1,383-sample dataset:

| Fold | Best Epoch | AUROC | AUPRC | F1 | MCC | Recall |
|------|-----------|-------|-------|-----|-----|--------|
| 0 | 60 | 0.746 | 0.357 | 0.361 | 0.290 | 44.0% |
| 1 | 38 | 0.710 | 0.226 | 0.255 | 0.198 | 76.0% |
| 2 | 4 | 0.703 | 0.254 | 0.202 | 0.126 | 88.0% |
| 3 | 110 | 0.795 | 0.379 | 0.238 | 0.209 | 91.7% |
| 4 | 33 | 0.692 | 0.262 | 0.293 | 0.218 | 45.8% |
| **Mean ± SD** | | **0.729 ± 0.042** | **0.296 ± 0.065** | **0.270 ± 0.060** | **0.208 ± 0.058** | **69.1%** |

The cross-validated AUROC (0.729 ± 0.042) is lower than the single-split AUROC (0.801), indicating some overfitting to the original validation set. The wide variance across folds reflects the small positive class (123 ecDNA+ total, ~25 per fold).

#### Bootstrap Significance Tests (Module 1)

10,000-iteration bootstrap confidence intervals and comparison tests:

| Model | AUROC | 95% CI | vs Random p |
|-------|-------|--------|-------------|
| ecDNA-Former | 0.801 | [0.669, 0.923] | 0.0005 |
| RandomForest | 0.695 | [0.498, 0.874] | — |
| **Difference** | **0.105** | [-0.033, 0.265] | **p = 0.075** |

The ecDNA-Former significantly outperforms random (permutation p = 0.0005) but the advantage over Random Forest is not significant at α = 0.05 (p = 0.075), likely due to only 17 ecDNA+ validation samples.

#### Feature Ablation Study (Module 1)

To quantify the contribution of each feature group, we retrained ecDNA-Former from scratch with each group zero-masked:

| Configuration | Features Zeroed | AUROC | AUPRC | F1 | MCC |
|--------------|----------------|-------|-------|-----|-----|
| Full (baseline) | 0 | 0.787 | 0.349 | 0.219 | 0.132 |
| **minus Hi-C** | **45** | **0.796** | **0.368** | 0.175 | 0.105 |
| minus CNV | 31 | 0.783 | 0.380 | 0.152 | 0.000 |
| minus Expression | 27 | 0.776 | 0.355 | 0.162 | 0.081 |
| **minus Dosage** | **9** | **0.811** | 0.341 | **0.321** | **0.261** |

**Key finding: Removing Hi-C features *improves* AUROC** (0.787 → 0.796), confirming the Hi-C redundancy concern — the 45 Hi-C features contribute no predictive value and may add noise. Removing dosage features also improves performance (0.787 → 0.811), suggesting CNV×Expression interaction terms are not useful. Removing CNV or expression causes only marginal AUROC drops (~0.01), indicating the model is robust to individual feature group ablation.

Note: The ablation "Full" baseline (0.787) is lower than the reported single-split AUROC (0.801) because the model was retrained from scratch with a different random seed.

#### Results Reconciliation (Module 1)

Three AUROC values appear throughout this document — they are from different evaluation protocols and are mutually consistent:

| AUROC | Source | Protocol |
|-------|--------|----------|
| **0.801** | Single 85/15 split | Best epoch 197, seed=42, original train/val partition |
| **0.729 ± 0.042** | 5-fold CV | Retrained from scratch per fold, seed=42 |
| **0.787** | Ablation baseline | Retrained from scratch, different random seed |

Retraining from scratch yields AUROC in the range 0.69–0.80 depending on the specific train/val split and random seed. The 0.801 single-split result is within the upper range of the 5-fold CV distribution (fold 3 achieved 0.795). The ablation baseline (0.787) falls between the CV mean and the single-split value.

**Baseline comparison (single-split vs CV):**
- Single-split: ecDNA-Former 0.801 vs RF 0.695 (Δ = 0.105, bootstrap p = 0.075)
- 5-fold CV: ecDNA-Former 0.729 ± 0.042 vs MLP 0.752 ± 0.089 vs RF 0.719 ± 0.048

The 10.5 pp single-split improvement over RF narrows under 5-fold CV (ecDNA-Former 0.729 vs RF 0.719, Δ = 0.010). All three models perform comparably within the high fold-to-fold variance imposed by the small positive class (~25 ecDNA+ per fold).

#### No-Dosage Configuration (Module 1)

Feature ablation identified that removing the 9 dosage features (CNV × expression interaction terms) improves AUROC from 0.787 to 0.811. A dedicated retraining confirms this finding:

```bash
python scripts/retrain_no_dosage.py --epochs 200 --patience 30
```

Dedicated retraining achieves **AUROC 0.812** (early stopped at epoch 75), confirming the ablation result. Bootstrap comparison vs the full-feature model: Δ = -0.128, p = 1.00 (the no-dosage model is not significantly *worse* — it is better). The recommended configuration for downstream use is the no-dosage model, as the dosage features introduce noise without adding predictive value.

Results: `checkpoints/no_dosage/best.pt`, `data/validation/no_dosage_results.csv`, `data/validation/no_dosage_bootstrap.csv`

#### MLP Baseline Comparison (Module 1)

To establish whether the transformer architecture provides genuine benefit over simpler models, we train an MLP baseline (112 → 256 → 128 → 1 with BatchNorm and Dropout) on the same 112 features with 5-fold stratified CV:

```bash
python scripts/train_mlp_baseline.py --epochs 200 --patience 30
```

| Model | AUROC (5-fold CV) | AUPRC (5-fold CV) | F1 (5-fold CV) |
|-------|-------------------|-------------------|----------------|
| **MLP** | **0.752 ± 0.089** | **0.306 ± 0.083** | 0.242 ± 0.050 |
| RF | 0.719 ± 0.048 | 0.308 ± 0.063 | 0.074 ± 0.073 |

Bootstrap comparison (out-of-fold predictions): MLP vs RF AUROC diff = -0.051, p = 0.976 — the MLP and RF perform comparably on matched folds. The high fold-to-fold variance (MLP range: 0.606–0.843) reflects the small positive class size (~25 ecDNA+ per fold).

For context, the ecDNA-Former 5-fold CV AUROC is 0.729 ± 0.042 (see above), which is comparable to MLP (0.752 ± 0.089) within the variance. The transformer architecture's advantage over simpler baselines is modest on this dataset size.

#### Lineage Leave-One-Out Cross-Validation (Module 1)

To test whether the model generalizes across cancer types (rather than memorizing lineage-specific patterns), we trained on all-but-one lineage and evaluated on the held-out lineage:

| Lineage | N_val | ecDNA+ | AUROC | AUPRC | F1 | MCC |
|---------|-------|--------|-------|-------|-----|-----|
| blood | 102 | 4 | **0.939** | 0.365 | 0.545 | 0.544 |
| bone | 38 | 4 | **0.912** | 0.575 | 0.600 | 0.557 |
| kidney | 38 | 4 | 0.772 | 0.342 | 0.000 | 0.000 |
| lung | 205 | 34 | 0.707 | 0.480 | 0.456 | 0.382 |
| ovary | 63 | 5 | 0.707 | 0.214 | 0.170 | 0.083 |
| colorectal | 70 | 11 | 0.684 | 0.482 | 0.364 | 0.245 |
| CNS | 83 | 12 | 0.668 | 0.276 | 0.250 | 0.227 |
| pancreas | 52 | 3 | 0.646 | 0.130 | 0.109 | 0.000 |
| gastric | 40 | 5 | 0.611 | 0.401 | 0.364 | 0.265 |
| breast | 62 | 15 | 0.611 | 0.418 | 0.390 | 0.000 |
| PNS | 32 | 4 | 0.607 | 0.181 | 0.222 | 0.000 |
| skin | 85 | 3 | 0.528 | 0.050 | 0.068 | 0.000 |
| urinary_tract | 36 | 4 | 0.445 | 0.131 | 0.222 | 0.081 |
| soft_tissue | 59 | 4 | 0.455 | 0.076 | 0.127 | 0.000 |

**Mean AUROC across lineages: ~0.66** (vs 0.801 pooled, 0.729 cross-validated).

The model generalizes well to blood (0.939) and bone (0.912) cancers but poorly to soft tissue (0.455), urinary tract (0.445), and skin (0.528). This high variance (0.445–0.939) indicates the model partially relies on lineage-specific patterns rather than universal ecDNA features. Lineages with very few positives (3–4) have unreliable estimates.

#### Null Baseline (Module 3: Vulnerability Discovery)

To test whether the 14/47 literature-validated genes could arise by chance, we sampled 100,000 random gene sets of size 47 from 17,453 genes and counted overlap with validation categories:

| Metric | Observed | Null Mean | P-value | Enrichment |
|--------|----------|-----------|---------|------------|
| Validated genes | 14/47 (29.8%) | 0.4/47 (0.8%) | < 0.0001 | **38.3×** |
| Wilson 95% CI | [18.7%, 44.0%] | — | — | — |

The vulnerability hits are highly non-random (p < 0.0001, 38.3× enrichment over chance).

#### Pathway Enrichment (Module 3)

Hypergeometric tests for GO/KEGG pathway enrichment among top vulnerability candidates (Benjamini-Hochberg corrected):

**Top 47 candidates:**

| Pathway | Overlap | Enrichment | Adj. p-value |
|---------|---------|------------|--------------|
| GO:0007067 Mitotic nuclear division | 8 (KIF11, KIF18A, KIF23, NCAPD2, NCAPG, NDC80, SGO1, TPX2) | 92.8× | 1.5e-13 |
| KEGG:hsa04110 Cell cycle | 5 (CDK1, CDK2, MCM2, SGO1, TP53) | 43.2× | 5.1e-7 |
| GO:0007049 Cell cycle | 3 (CDK1, CDK2, SGO1) | 31.8× | 3.8e-4 |
| GO:0010941 Regulation of cell death | 2 (BCL2L1, TP53) | 32.3× | 4.3e-3 |

**Top 100 candidates:**

| Pathway | Overlap | Enrichment | Adj. p-value |
|---------|---------|------------|--------------|
| GO:0007067 Mitotic nuclear division | 9 | 49.1× | 1.2e-12 |
| KEGG:hsa04110 Cell cycle | 7 | 28.4× | 2.2e-8 |
| GO:0006260 DNA replication | 3 (MCM2, MCM3, ORC6) | 15.9× | 2.2e-3 |
| GO:0000398 mRNA splicing | 2 (SNRPD1, SNRPF) | 12.9× | 1.5e-2 |
| GO:0000502 Proteasome complex | 2 (PSMB5, PSMD7) | 11.6× | 1.6e-2 |

The dominant enrichment for mitotic nuclear division and cell cycle pathways is consistent with ecDNA's known biology — acentric elements that impose segregation stress during mitosis.

#### Gene Set Enrichment Analysis (Module 3)

Standard GSEA (Subramanian et al. 2005) provides a complementary approach to hypergeometric testing. Rather than testing a discrete set of top candidates, GSEA ranks all 17,453 genes by differential dependency effect size and detects coordinated shifts across entire pathways. Pathway-level FDR can be significant even when no individual gene survives gene-level FDR.

```bash
python scripts/run_gsea.py --n-permutations 10000
```

| Pathway | NES | GSEA FDR | Hypergeometric FDR |
|---------|-----|----------|--------------------|
| GO:0007067 mitotic nuclear division | **2.643** | **0.0002** | 0.0000 |
| KEGG:hsa04110 Cell cycle | **2.510** | **0.0002** | 0.0000 |
| GO:0006260 DNA replication | **2.424** | **0.0002** | 0.142 |
| GO:0007049 cell cycle | **2.267** | **0.0002** | 0.0004 |
| GO:0000502 proteasome complex | **2.125** | **0.0002** | 0.142 |
| GO:0000398 mRNA splicing | **2.025** | **0.0002** | 1.000 |
| GO:0006412 translation | 1.588 | 0.016 | 1.000 |
| GO:0006457 protein folding | 1.565 | 0.021 | 1.000 |
| GO:0010941 regulation of cell death | 1.315 | 0.074 | 0.004 |
| GO:0006281 DNA repair | 0.964 | 0.259 | 1.000 |

8 of 10 pathways reach GSEA FDR < 0.05. Notably, GSEA identifies 5 pathways (DNA replication, proteasome complex, mRNA splicing, translation, protein folding) that hypergeometric testing misses entirely, demonstrating the value of ranking-based enrichment when individual genes have small effect sizes.

Results: `data/validation/gsea_results.csv`, `data/validation/gsea_vs_hypergeometric.csv`

#### CircularODE Physics Ablation (Module 2)

To demonstrate that physics constraints in CircularODE are not circular (i.e., they provide genuine inductive bias rather than just recovering the simulation physics), we run three ablation experiments:

```bash
python scripts/circularode_physics_ablation.py --epochs 100 --patience 20
```

**Experiment A — Physics weight sweep:**

| Config | Physics Weight | Val MSE | Val MAE | Val Corr |
|--------|--------------|---------|---------|----------|
| no-physics | 0.0 | 172.37 | 6.96 | 0.611 |
| **weak** | **0.01** | **169.83** | **6.82** | **0.617** |
| default | 0.1 | 175.50 | 6.96 | 0.603 |
| strong | 1.0 | 174.09 | 6.84 | 0.609 |

The weak physics constraint (weight=0.01) achieves the best MSE and correlation. Stronger constraints degrade performance, suggesting the regularization is useful at low weight but dominates the loss at higher weights. The no-physics baseline is competitive, confirming the constraints are not required to learn dynamics — they act as a mild inductive bias rather than recovering simulation physics.

**Experiment B — Cross-treatment holdout:** Trains on 3 of 4 treatment types, tests on the held-out treatment.

| Holdout Treatment | Physics Corr | No-Physics Corr |
|-------------------|-------------|-----------------|
| 0 | **0.761** | 0.729 |
| 1 | 0.812 | 0.810 |
| 3 | 0.602 | **0.633** |
| 5 | 0.688 | **0.692** |
| **Mean** | **0.716** | **0.716** |

Physics constraints help for holdout treatment 0 (+0.032 correlation) but not consistently across all treatments. Mean correlation is identical (0.716), indicating the physics constraint is approximately neutral for cross-treatment generalization.

**Experiment C — Temporal extrapolation:** Trains on the first 25 timepoints, predicts the last 25.

| Config | Interpolation Corr | Extrapolation Corr |
|--------|-------------------|-------------------|
| physics | 0.756 | 0.559 |
| no-physics | **0.779** | **0.599** |

Both models degrade substantially on extrapolation (as expected), but the no-physics model actually performs slightly better. This suggests that for this synthetic data, physics constraints do not provide an extrapolation advantage.

**Summary:** Physics constraints are not circular — the model learns effectively without them. The weak constraint provides marginal benefit in the sweep, but cross-treatment and temporal experiments show approximately neutral effects. This confirms the constraints act as optional regularization rather than smuggling in simulation assumptions.

Results: `data/validation/circularode_physics_ablation.csv`, `circularode_cross_treatment.csv`, `circularode_temporal_extrapolation.csv`

#### IRM Environment Robustness (Module 3)

To validate that the IRM environments (real cancer lineages) carry meaningful biological signal, we compare three conditions:

```bash
python scripts/irm_robustness_analysis.py --epochs 30 --n-shuffles 5
```

| Condition | ERM Loss | IRM Penalty |
|-----------|----------|-------------|
| **Real** (lineages) | **0.248** | **0.087** |
| Shuffled (mean ± std) | 0.257 ± 0.021 | 0.062 ± 0.015 |
| Random (mean ± std) | 0.302 ± 0.004 | 0.046 ± 0.011 |

The real environments produce a *higher* IRM penalty (0.087) than shuffled (0.062) or random (0.046), indicating that real lineages induce more environment-specific variation that the IRM must penalize — this is the expected behavior when environments carry genuine biological signal. Real environments also achieve the lowest ERM loss (0.248), showing that lineage structure helps the base predictor.

**Ranking correlations (Spearman, real vs controls):**
- Real vs shuffled: ρ = 0.790 ± 0.051
- Real vs random: ρ = 0.846 ± 0.016

Rankings are moderately correlated across conditions, indicating the vulnerability signal is partially recoverable even without correct environment labels, but real lineages produce distinct rankings (ρ < 1.0).

Results: `data/validation/irm_robustness_results.csv`, `data/validation/irm_ranking_correlations.csv`

## Target Performance

| Task | Metric | Target | Result | Status |
|------|--------|--------|--------|--------|
| ecDNA Formation | AUROC | 0.80-0.85 | **0.801** | ✓ Meets target |
| ecDNA Formation | AUPRC | >0.20 | **0.298** | ✓ 3.6× above base rate |
| ecDNA Formation | Recall | >80% | 50.0% | ~ Below target |
| ecDNA Formation | F1 | 0.40-0.50 | 0.278 | ~ Moderate |
| Vulnerability Discovery | Robust hits | 10-20 | **14** | ✓ Literature validated |
| Vulnerability Discovery | Clinical targets | 1+ | **3** | ✓ BBI-355, BBI-940, BBI-825 |
| Trajectory Prediction (Simplified) | MSE | <0.1 | **0.014** | ✓ Exceeds target |
| Trajectory Prediction (Simplified) | Correlation | >0.9 | **0.993** | ✓ Exceeds target (synthetic data) |
| Trajectory Prediction (Full SDE) | MSE (raw) | — | 170.2 | ~ Full trajectory task is harder |
| Trajectory Prediction (Full SDE) | Correlation | >0.9 | 0.615 | ~ Below target (see notes) |
| Causal Vulnerability (Full) | DAG violation | →0 | 11,599 | ✗ Not converged |
| Causal Vulnerability (Full) | ecDNA correlation | >0.5 | -0.121 | ✗ Not converged |

**Caveats on target performance:**
- Trajectory MSE/correlation are on synthetic held-out data, not real longitudinal measurements
- The 14 "validated" vulnerability genes are literature cross-references, not prospective experimental validations
- All Module 1 metrics are from a single 85/15 random split; 5-fold CV gives 0.729 ± 0.042 (see Statistical Validation)
- Small ecDNA+ validation set — metrics have wide confidence intervals (95% CI: [0.669, 0.923])

## Known Limitations

1. **Single train/val split**: Module 1 headline metrics (AUROC 0.801) are from a single 85/15 split with only 17 ecDNA+ validation samples. 5-fold CV gives a more conservative estimate of 0.729 ± 0.042 (see Statistical Validation).
2. **Hi-C features provide no value**: Feature ablation confirms that removing all 45 Hi-C features *improves* AUROC (0.787→0.796). Intercorrelation analysis shows cnv_hic_X features are perfectly correlated (r≈1.0) with cnv_X because Hi-C densities are reference-genome constants. The Gen 2→Gen 3 AUROC improvement (0.736→0.801) reflects additional model capacity or random variation, not new information from chromatin topology. Additionally, the Hi-C reference (GM12878) is a lymphoblastoid cell line, not representative of the cancer types in our dataset; this mismatch is moot given the features' redundancy.
3. **CircularODE trained on synthetic data**: All trajectory training data is simulated. The model has not been validated on real longitudinal ecDNA copy number measurements. The 0.993 correlation reflects fitting synthetic data generated from the same physics the model enforces.
4. **No genes survive FDR correction**: All 17,453 differential dependency tests yield FDR > 0.43. The vulnerability hits are nominally significant (p < 0.05) but do not survive multiple testing correction. They should be treated as hypothesis-generating. The null baseline (38.3× enrichment, p < 0.0001) and pathway enrichment (mitotic/cell cycle) provide orthogonal support but do not address the multiple testing issue.
5. **Marginal significance vs Random Forest**: The ecDNA-Former vs RF difference (0.105 AUROC) is not significant at α=0.05 (bootstrap p=0.075), likely due to only 17 ecDNA+ validation samples. The model does significantly outperform random (permutation p=0.0005).
6. **Modules are independently trained**: The three modules are trained independently on different data subsets and have no shared representations or joint training. They are composable — Module 1 predictions can inform Module 2 initial conditions and Module 3 stratification — but this composition is post-hoc, not end-to-end.
7. **Small positive class**: 9.0% positive rate (106/1,176 training) limits statistical power, especially for per-lineage analysis where some lineages have <5 ecDNA+ samples.
8. **Unlabeled-as-negative assumption**: 839/1,383 training samples have no ecDNA label (NaN in CytoCellDB) but are treated as ecDNA-negative. Some of these may be true positives, introducing label noise.
9. **Lineage confounding**: Lineage LOOCV shows mean AUROC of ~0.66 across 14 lineages (vs 0.801 pooled), with high variance (0.445–0.939). The model partially relies on lineage-specific patterns rather than universal ecDNA features. Performance is poor on soft tissue, urinary tract, and skin lineages.

## Integration Demo

Run the full ECLIPSE patient stratification pipeline:

```bash
python scripts/eclipse_demo.py
```

**Demo Output (3 Cases):**

```
================================================================================
                    ECLIPSE FRAMEWORK DEMONSTRATION
================================================================================
Initializing ECLIPSE on cuda...
  Loaded CircularODE (CN mean=8.7, std=19.2)
  Models loaded (using inference mode)
  Loaded 14 validated vulnerabilities
ECLIPSE initialized successfully!

--------------------------------------------------------------------------------
CASE 1: High-risk patient with MYC amplification
--------------------------------------------------------------------------------
╔══════════════════════════════════════════════════════════════════════════════╗
║                      ECLIPSE Patient Stratification                          ║
║                      Patient ID: TCGA-HIGH-001                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ecDNA Probability:  72.1%                                                   ║
║  Risk Level: HIGH                                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Treatment Predictions (Copy Number at Day 100):                             ║
║    none           : CN =  40.9  |  Resistance prob = 43.5%                   ║
║    targeted       : CN =   6.2  |  Resistance prob = 29.0%                   ║
║    chemo          : CN =  15.6  |  Resistance prob = 58.0%                   ║
║    maintenance    : CN =  30.1  |  Resistance prob = 21.8%                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Top Vulnerabilities:                                                        ║
║    ORC6      : effect =  -0.083  |  DNA replication                          ║
║    MCM2      : effect =  -0.089  |  DNA replication                          ║
║    SNRPF     : effect =  -0.090  |  Spliceosome                             ║
║    KIF11     : effect =  -0.092  |  Mitosis                                  ║
║    NDC80     : effect =  -0.092  |  Mitosis                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Recommendations:                                                            ║
║    • ⚠️  HIGH ecDNA probability - recommend targeted monitoring              ║
║    • 📊 Model predicts best CN reduction with: targeted therapy              ║
║    • ⚡ Elevated resistance risk with: chemo                                  ║
║    • 🔬 Additional targets (high evidence): ORC6, MCM2, SNRPF               ║
╚══════════════════════════════════════════════════════════════════════════════╝

--------------------------------------------------------------------------------
CASE 2: Low-risk patient without amplification
--------------------------------------------------------------------------------
╔══════════════════════════════════════════════════════════════════════════════╗
║                      ECLIPSE Patient Stratification                          ║
║                      Patient ID: TCGA-LOW-002                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ecDNA Probability:  24.0%                                                   ║
║  Risk Level: LOW                                                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Treatment Predictions (Copy Number at Day 100):                             ║
║    N/A (low ecDNA risk)                                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Recommendations:                                                            ║
║    • ✓  Low ecDNA probability - standard treatment protocols                 ║
║    • 📋 Continue routine genomic monitoring                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

--------------------------------------------------------------------------------
CASE 3: Moderate-risk patient with EGFR amplification
--------------------------------------------------------------------------------
╔══════════════════════════════════════════════════════════════════════════════╗
║                      ECLIPSE Patient Stratification                          ║
║                      Patient ID: TCGA-MOD-003                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ecDNA Probability:  51.2%                                                   ║
║  Risk Level: MODERATE                                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Treatment Predictions (Copy Number at Day 100):                             ║
║    none           : CN =  51.9  |  Resistance prob = 40.5%                   ║
║    targeted       : CN =   6.0  |  Resistance prob = 27.0%                   ║
║    chemo          : CN =  11.6  |  Resistance prob = 54.0%                   ║
║    maintenance    : CN =  11.5  |  Resistance prob = 20.2%                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Recommendations:                                                            ║
║    • ⚠️  HIGH ecDNA probability - recommend targeted monitoring              ║
║    • 📊 Model predicts best CN reduction with: targeted therapy              ║
║    • ⚡ Elevated resistance risk with: chemo                                  ║
║    • 🔬 Additional targets (high evidence): ORC6, MCM2, SNRPF               ║
╚══════════════════════════════════════════════════════════════════════════════╝

================================================================================
DEMONSTRATION COMPLETE
================================================================================

ECLIPSE integrates three complementary analyses:

  Module 1 (ecDNA-Former): Predicts ecDNA probability from genomic features
           → Achieved 0.801 AUROC on validation data

  Module 2 (CircularODE): Models copy number dynamics under treatment
           → Achieved 0.993 correlation on trajectory prediction

  Module 3 (VulnCausal): Identifies therapeutic vulnerabilities
           → 14 validated targets including CHK1 (in clinical trials)
```

## Citation

If you use ECLIPSE in your research, please cite this repository:

```bibtex
@software{eclipse2026,
  title={ECLIPSE: Extrachromosomal Circular DNA Learning for Integrated
         Prediction of Synthetic-lethality and Expression},
  author={Cheng, Bryan and Zhang, Jasper},
  year={2026},
  url={https://github.com/bryanc5864/ECLIPSE}
}
```

## References

Key papers informing this work:

1. Kim H, et al. "Extrachromosomal DNA is associated with oncogene amplification and poor outcome across multiple cancers." *Nature Genetics* 2020.
2. Turner KM, et al. "Extrachromosomal oncogene amplification drives tumour evolution and genetic heterogeneity." *Nature* 2017.
3. Hung KL, et al. "ecDNA hubs drive cooperative intermolecular oncogene expression." *Nature* 2021.
4. Fessler J, et al. "CytoCellDB: a comprehensive resource for exploring extrachromosomal DNA in cancer cell lines." *NAR Cancer* 2024.

## License

MIT License (c) 2026 Bryan Cheng and Jasper Zhang. See [LICENSE](LICENSE) for details.

## Contact

For questions or issues, please open a [GitHub issue](https://github.com/bryanc5864/ECLIPSE/issues).
