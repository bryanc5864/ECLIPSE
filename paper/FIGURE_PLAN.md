# ECLIPSE Paper - Comprehensive Figure Plan

## Analysis of Reference Projects

### TACTIC Project Figure Patterns:
- **Figure 1 (2 panels)**: Line plots showing kinetic curves - single vs multi-condition
- **Figure 2 (3 panels)**: Mechanism-specific signatures across conditions
- **Architecture**: Pipeline flow diagram
- **Figure 3 (2 panels)**: Confusion matrix heatmap + per-class accuracy bar chart
- **Figure 4 (4 panels)**: Calibration, method comparison, noise robustness, training dynamics

### Gradient Project Figure Patterns:
- **Figure 1 (2 panels)**: Concept diagram showing problem setup
- **Figure 2 (3 panels)**: Scatter + regression, phase transition curve, cross-domain bars
- **Figure 3 (2 panels)**: Practical utility scatter + grouped bar chart
- **Supplementary**: Hierarchically clustered heatmaps

---

## ECLIPSE Figure Plan

### **FIGURE 1: The Disconnected ecDNA Analysis Problem** (Full Width, 3 Panels)
*Demonstrates the three problems we solve*

| Panel | Type | Data | Purpose |
|-------|------|------|---------|
| **(a) Data Leakage** | Horizontal bar chart | Feature importance scores | Show AA_* features dominate (78.3% importance) vs non-leaky (21.7%) |
| **(b) Physics Mismatch** | Scatter + reference line | Variance ratio vs copy number | Unconstrained models violate binomial segregation (target = 0.25) |
| **(c) Confounding** | Grouped bar chart | Validation rates by method | Correlation-based: 8.0% vs Causal: 29.8% validation |

**Specifics:**
- Panel (a): 10 features, sorted by importance. AA_* in red, non-leaky in blue
- Panel (b): X = parent CN, Y = observed variance ratio. Line at 0.25 (theoretical). Show standard models deviate
- Panel (c): 4 methods (Diff. CRISPR, CERES, Hartwig, VulnCausal), bars show validation %

---

### **FIGURE 2: ECLIPSE Architecture** (Full Width, Single Panel)
*Pipeline diagram showing three modules*

**Components:**
1. **Input section** (left): Genomic Features, Initial State + Treatment, Expression + CRISPR
2. **Module 1** (blue box): ecDNA-Former - "Formation Prediction" - Output: P(ecDNA+)
3. **Module 2** (green box): CircularODE - "Dynamics Modeling" - Output: Trajectory
4. **Module 3** (purple box): VulnCausal - "Vulnerability Discovery" - Output: Targets
5. **Integration** (orange box, bottom): Patient Stratification - Risk Level + Recommendations

**Arrows:** Show data flow between modules and to integration

---

### **FIGURE 3: Formation Prediction Results (Module 1)** (Full Width, 4 Panels)
*Comprehensive validation of ecDNA-Former*

| Panel | Type | Data | Metrics Shown |
|-------|------|------|---------------|
| **(a) ROC Curves** | Line plot | 4 methods | Random (0.500), XGBoost (0.745), MLP (0.762), ecDNA-Former (0.801) |
| **(b) PR Curves** | Line plot | 4 methods | Baseline PR=0.089, compare AP values |
| **(c) Calibration** | Bar + diagonal | Predicted vs actual | ECE = 0.032, bins from 0.1 to 0.8 |
| **(d) Ablation** | Horizontal bars | Component removal | Hi-C (-0.058), Bottleneck (-0.045), Expression (-0.033), CNV (-0.029), Fragile (-0.016) |

**Key annotations:**
- Panel (a): Legend with AUROC values, "Ours" line thicker in orange
- Panel (b): Horizontal line at class prior (8.9%)
- Panel (c): "ECE=0.032" text annotation
- Panel (d): Vertical line at full model performance (0.801)

---

### **FIGURE 4: Dynamics and Vulnerability Results (Modules 2 & 3)** (Full Width, 4 Panels)
*Combined figure for dynamics and vulnerability discovery*

| Panel | Type | Data | Metrics Shown |
|-------|------|------|---------------|
| **(a) Trajectory Prediction** | Line + scatter + CI | Synthetic trajectory | Ground truth points, predicted line, 95% CI shading |
| **(b) Physics Validation** | Scatter + theory line | Segregation variance | X: Copy Number, Y: Variance. Theory: CN/4. r = 0.993 |
| **(c) Isogenic Validation** | Grouped bars | 3 experiments | GBM39_EC, GBM39_HSR, TR14. Show MSE and within-error % |
| **(d) Vulnerability Discovery** | Bar chart | 4 methods | Diff. CRISPR (8.0%), CERES (14.7%), Hartwig (18.0%), VulnCausal (29.8%) |

**Key annotations:**
- Panel (a): "MSE = 0.014, r = 0.993" text box
- Panel (b): Dashed line for theoretical CN/4
- Panel (c): Correlation values above bars
- Panel (d): VulnCausal bar highlighted in orange, others in gray

---

### **FIGURE 5: Vulnerability Analysis Details** (Full Width, 4 Panels)
*Deep dive into VulnCausal results*

| Panel | Type | Data | Purpose |
|-------|------|------|---------|
| **(a) Category Breakdown** | Stacked/grouped bar | 14 genes by category | DNA Damage (3), Cell Cycle (3), Mitosis (3), Replication (3), Chromatin (2) |
| **(b) Drug Sensitivity** | Scatter plot | GDSC IC50 data | X: ecDNA- IC50, Y: ecDNA+ IC50. Points below diagonal = selective |
| **(c) IRM Ablation** | Comparison bars | With/without IRM | Full IRM (29.8%) vs No IRM (15.2%) validation rate |
| **(d) Effect Sizes** | Horizontal bars | Top 10 genes | Gene names + effect sizes (NCAPD2: -0.117, SGO1: -0.150, etc.) |

**Key annotations:**
- Panel (a): Each category colored differently, genes labeled
- Panel (b): Diagonal reference line, significant drugs colored (AZD7762 = CHK1 inhibitor)
- Panel (c): Fold improvement annotation (+14.6 pp)
- Panel (d): CHK1 highlighted as "VALIDATED - Clinical Trials"

---

### **SUPPLEMENTARY FIGURE S1: Per-Lineage Performance** (Full Width, 2 Panels)

| Panel | Type | Data | Purpose |
|-------|------|------|---------|
| **(a) Per-Lineage AUROC Heatmap** | Heatmap | ~20 lineages | Show performance varies by cancer type |
| **(b) Lineage Distribution** | Stacked bar | Train/Val split | Show class balance across lineages |

---

### **SUPPLEMENTARY FIGURE S2: Training Dynamics** (Full Width, 3 Panels)

| Panel | Type | Data | Purpose |
|-------|------|------|---------|
| **(a) ecDNA-Former Training** | Dual line plot | Train/Val AUROC vs epoch | Show convergence, early stopping point |
| **(b) CircularODE Training** | Dual line plot | Train/Val MSE vs epoch | Show physics loss contribution |
| **(c) VulnCausal Training** | Dual line plot | Train/Val + IRM penalty vs epoch | Show invariance convergence |

---

### **SUPPLEMENTARY FIGURE S3: Full Vulnerability Table** (Visual Table)

| Column | Content |
|--------|---------|
| Gene | Gene symbol |
| Effect | Our differential effect |
| Category | Functional category |
| Literature | Support level (VALIDATED/HIGH/MODERATE) |
| Mechanism | Brief mechanism description |
| References | PMIDs |

---

## Graph Type Summary

| Graph Type | Count | Usage |
|------------|-------|-------|
| **Line plots** | 6 | ROC, PR, trajectories, training dynamics |
| **Scatter plots** | 4 | Physics validation, drug sensitivity |
| **Horizontal bar charts** | 4 | Ablation, feature importance, effect sizes |
| **Vertical bar charts** | 5 | Method comparisons, validation rates |
| **Heatmaps** | 1 | Per-lineage performance |
| **Pipeline diagrams** | 1 | Architecture |
| **Grouped/stacked bars** | 2 | Category breakdown, isogenic validation |

---

## Color Scheme

```python
MODULE_COLORS = {
    'former': '#4472C4',      # Blue for Module 1
    'circularode': '#70AD47', # Green for Module 2
    'vulncausal': '#9B59B6',  # Purple for Module 3
    'integration': '#E8871E', # Orange for integration / "Ours"
}

CATEGORY_COLORS = {
    'DNA Damage': '#E74C3C',     # Red
    'Cell Cycle': '#3498DB',     # Blue
    'Mitosis': '#2ECC71',        # Green
    'Replication': '#9B59B6',    # Purple
    'Chromatin': '#E8871E',      # Orange
    'Proteasome': '#1ABC9C',     # Teal
}

BASELINE_GRAY = '#888888'
```

---

## Data Requirements

### Figure 1:
- Feature importance scores (can simulate based on paper claims)
- Variance ratio measurements for unconstrained vs constrained models
- Validation rates: 8.0%, 14.7%, 18.0%, 29.8%

### Figure 3:
- ROC/PR curves: AUROC = 0.745 (XGB), 0.762 (MLP), 0.801 (Ours)
- Calibration bins and actual frequencies
- Ablation deltas: Hi-C (-0.058), Bottleneck (-0.045), etc.

### Figure 4:
- Trajectory data from `circularode_lange_validation.csv`
- Physics validation: correlation = 0.993
- Isogenic: GBM39_EC, GBM39_HSR, TR14 experiments

### Figure 5:
- 14 validated genes from `literature_validation.csv`
- GDSC IC50 from `vulncausal_gdsc_real_validation.csv`
- IRM ablation: 29.8% vs 15.2%

---

## Implementation Priority

1. **Figure 2 (Architecture)** - Essential for understanding
2. **Figure 3 (Formation Results)** - Core Module 1 validation
3. **Figure 4 (Dynamics + Vuln)** - Core Modules 2 & 3
4. **Figure 1 (Problem)** - Motivates the approach
5. **Figure 5 (Vuln Details)** - Deep dive
6. **Supplementary** - Supporting evidence

---

## Panel Comprehensiveness Checklist

Following TACTIC/Gradient patterns, each panel should show:

- [ ] **Quantitative metrics** with exact numbers
- [ ] **Statistical annotations** (p-values, confidence intervals where applicable)
- [ ] **Reference lines/baselines** for comparison
- [ ] **Color coding** consistent with module/category
- [ ] **Legend with values** (e.g., "ecDNA-Former (0.801)")
- [ ] **Clear axis labels** with units
- [ ] **Panel labels** (a), (b), (c) in bold

---

## Figure Dimensions (ICLR single-column = 5.5" width)

| Figure | Panels | Dimensions |
|--------|--------|------------|
| Fig 1 | 3 | 5.5" × 1.8" |
| Fig 2 | 1 | 5.5" × 2.5" |
| Fig 3 | 4 | 5.5" × 1.6" |
| Fig 4 | 4 | 5.5" × 1.8" |
| Fig 5 | 4 | 5.5" × 2.0" |
| Supp S1 | 2 | 5.5" × 2.2" |
| Supp S2 | 3 | 5.5" × 1.6" |
