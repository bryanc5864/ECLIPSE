# ECLIPSE Data Status

## Downloaded (1.6 GB)

| Dataset | Size | Status |
|---------|------|--------|
| DepMap CRISPR | 369 MB | ✓ Ready |
| DepMap Expression | 410 MB | ✓ Ready |
| DepMap Copy Number | 805 MB | ✓ Ready |
| DepMap Cell Lines | 457 KB | ✓ Ready |
| COSMIC Genes | 394 B | ✓ Ready (curated list) |
| Fragile Sites (hg38) | 2.5 KB | ✓ Ready (55 CFS) |
| **CytoCellDB** | 1.2 MB | ✓ Ready (1,859 cell lines) |

## Feature Extraction Status

Features extracted and saved to `data/features/`:

| Split | Samples | ecDNA+ | ecDNA- | File |
|-------|---------|--------|--------|------|
| Train | 1,301 | 103 (7.9%) | 1,198 | `module1_features_train.npz` |
| Val | 278 | 19 (6.8%) | 259 | `module1_features_val.npz` |
| Test | 280 | 19 (6.8%) | 261 | `module1_features_test.npz` |

**Features included (27 total):**
- Copy number (7): `max_copy_number`, `mean_copy_number`, `log_max_cn`, `cn_gt_10`, `cn_gt_20`, `cn_gt_50`, `num_amplicons`
- Oncogenes (14): `has_MYC`, `has_MYCL`, `has_MYCN`, `has_EGFR`, `has_ERBB2`, `has_CDK4`, `has_CDK6`, `has_MDM2`, `has_CCND1`, `has_CCNE1`, `has_FGFR1`, `has_MET`, `has_PDGFRA`, `num_oncogenes`, `total_genes_on_ecdna`
- Structural (5): `has_ecdna_type`, `has_bfb_type`, `has_linear_type`, `has_complex_type`, `has_hsr`

## Manual Download Required

### 1. ecDNA Labels - Kim et al. 2020 (CRITICAL)
**3,212 tumor samples with ecDNA classifications**

1. Go to: https://www.nature.com/articles/s41588-020-0678-2
2. Click "Supplementary Information"
3. Download "Supplementary Tables" (Excel file)
4. Save to: `data/ecdna_labels/kim2020_supplementary_tables.xlsx`

### 2. CytoCellDB - Cell Line ecDNA Status
**577 DepMap cell lines (139 ecDNA+, 438 ecDNA-)**

1. Go to: https://cytocelldb.unc.edu/
2. Click "Browse Cell Lines"
3. Export all data (look for download/export button)
4. Save to: `data/cytocell_db/cytocell_annotations.csv`

### 3. AmpliconRepository (Optional - more samples)
**4,500+ samples with ecDNA classifications**

1. Create account: https://ampliconrepository.org/accounts/signup/
2. Browse public projects
3. Download sample classifications
4. Save to: `data/amplicon_repository/classifications.csv`

### 4. Hi-C Data (Optional - for topology encoder)
**~3 GB per cell line (selective download)**

1. Go to: https://data.4dnucleome.org/
2. Create free account
3. Search: Cell Line = "GM12878", File Type = "mcool"
4. Download .mcool file (~3 GB)
5. Save to: `data/hic/GM12878.mcool`

Recommended cell lines: GM12878 (reference), K562 (CML), IMR90 (normal)

## After Manual Downloads

Once you have the Kim et al. labels, run:
```bash
python -c "
import pandas as pd
df = pd.read_excel('data/ecdna_labels/kim2020_supplementary_tables.xlsx', sheet_name=0)
print(f'Loaded {len(df)} samples')
print(df.columns.tolist())
"
```

## Data Priority

| Priority | Dataset | Impact |
|----------|---------|--------|
| 🔴 Critical | Kim 2020 labels | Can't train without ecDNA labels |
| 🟠 High | CytoCellDB | Cell line validation set |
| 🟡 Medium | Hi-C data | Topology encoder (can use synthetic) |
| 🟢 Low | AmpliconRepository | Extra training samples |
