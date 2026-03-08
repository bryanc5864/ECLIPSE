"""
Data loaders for ECLIPSE.

Provides unified interfaces for loading:
- AmpliconRepository (ecDNA annotations)
- CytoCellDB (cell line ecDNA status)
- DepMap (CRISPR, expression, copy number)
- Hi-C (chromatin contact maps)
- Fragile sites
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import h5py
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """Base class for data loaders."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load the data."""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate the loaded data."""
        pass


class AmpliconRepositoryLoader(BaseLoader):
    """
    Load ecDNA annotations from AmpliconRepository.

    AmpliconRepository contains AmpliconArchitect outputs for ~4,500 tumor samples
    with classifications: Circular (ecDNA), BFB, Linear.

    Attributes:
        samples: DataFrame of sample metadata
        classifications: DataFrame of amplicon classifications
        amplicons: DataFrame of amplicon details
    """

    def __init__(self, data_dir: str = "data/amplicon_repository"):
        super().__init__(data_dir)
        self.samples = None
        self.classifications = None
        self.amplicons = None

    def load(self) -> pd.DataFrame:
        """
        Load AmpliconRepository data.

        Returns:
            DataFrame with sample-level ecDNA annotations
        """
        # Load sample metadata
        samples_path = self.data_dir / "samples.csv"
        if samples_path.exists():
            self.samples = pd.read_csv(samples_path)
        else:
            logger.warning("Samples file not found, creating mock data")
            self.samples = self._create_mock_samples()

        # Load classifications
        class_path = self.data_dir / "classifications.csv"
        if class_path.exists():
            self.classifications = pd.read_csv(class_path)
        else:
            logger.warning("Classifications file not found, creating mock data")
            self.classifications = self._create_mock_classifications()

        # Merge to create unified view
        return self._merge_data()

    def _merge_data(self) -> pd.DataFrame:
        """Merge samples with classifications."""
        if self.samples is None or self.classifications is None:
            raise ValueError("Data not loaded")

        # Create sample-level summary
        merged = self.samples.copy()

        # Add ecDNA status based on classifications
        if "sample_id" in self.classifications.columns:
            ecdna_samples = self.classifications[
                self.classifications["classification"] == "Circular"
            ]["sample_id"].unique()

            merged["ecdna_positive"] = merged["sample_id"].isin(ecdna_samples)

        return merged

    def get_ecdna_samples(self) -> pd.DataFrame:
        """Get only ecDNA-positive samples."""
        data = self.load()
        return data[data["ecdna_positive"]]

    def get_amplicons_for_sample(self, sample_id: str) -> pd.DataFrame:
        """Get amplicon details for a specific sample."""
        if self.classifications is None:
            self.load()

        return self.classifications[
            self.classifications["sample_id"] == sample_id
        ]

    def _create_mock_samples(self) -> pd.DataFrame:
        """Create mock samples for testing."""
        np.random.seed(42)
        n_samples = 4500

        cancer_types = [
            "GBM", "BRCA", "LUAD", "COAD", "SKCM", "OV", "LUSC",
            "HNSC", "KIRC", "LIHC", "PRAD", "STAD", "THCA", "BLCA"
        ]

        return pd.DataFrame({
            "sample_id": [f"TCGA-{i:04d}" for i in range(n_samples)],
            "cancer_type": np.random.choice(cancer_types, n_samples),
            "source": np.random.choice(["TCGA", "PCAWG", "CCLE"], n_samples,
                                       p=[0.6, 0.3, 0.1]),
        })

    def _create_mock_classifications(self) -> pd.DataFrame:
        """Create mock classifications for testing."""
        np.random.seed(42)
        n_samples = 4500
        n_amplicons = 6000

        # ~30% ecDNA positive rate
        sample_ids = [f"TCGA-{i:04d}" for i in range(n_samples)]
        ecdna_samples = np.random.choice(sample_ids, int(n_samples * 0.30), replace=False)

        records = []
        for i in range(n_amplicons):
            sample = sample_ids[i % n_samples]
            if sample in ecdna_samples:
                classification = np.random.choice(
                    ["Circular", "BFB", "Linear"],
                    p=[0.5, 0.3, 0.2]
                )
            else:
                classification = np.random.choice(
                    ["BFB", "Linear"],
                    p=[0.4, 0.6]
                )

            records.append({
                "sample_id": sample,
                "amplicon_id": f"amp_{i}",
                "classification": classification,
                "chromosome": f"chr{np.random.randint(1, 23)}",
                "copy_number": np.random.uniform(5, 100),
            })

        return pd.DataFrame(records)

    def validate(self) -> bool:
        """Validate loaded data."""
        if self.samples is None:
            return False

        required_cols = ["sample_id"]
        return all(col in self.samples.columns for col in required_cols)


class CytoCellDBLoader(BaseLoader):
    """
    Load cell line ecDNA annotations from CytoCellDB.

    CytoCellDB contains 577 cell lines with cytogenetically-validated
    ecDNA status, linked to DepMap identifiers.
    """

    def __init__(self, data_dir: str = "data/cytocell_db"):
        super().__init__(data_dir)
        self.annotations = None

    def load(self) -> pd.DataFrame:
        """Load CytoCellDB annotations."""
        # Try to load from file
        for filename in ["cytocell_annotations.csv", "cytocell_template.csv"]:
            filepath = self.data_dir / filename
            if filepath.exists():
                self.annotations = pd.read_csv(filepath)
                break

        if self.annotations is None:
            logger.warning("CytoCellDB not found, creating mock data")
            self.annotations = self._create_mock_annotations()

        return self.annotations

    def get_ecdna_positive_lines(self) -> List[str]:
        """Get list of ecDNA-positive cell line DepMap IDs."""
        if self.annotations is None:
            self.load()

        return self.annotations[
            self.annotations["ecdna_status"] == "positive"
        ]["depmap_id"].tolist()

    def get_ecdna_negative_lines(self) -> List[str]:
        """Get list of ecDNA-negative cell line DepMap IDs."""
        if self.annotations is None:
            self.load()

        return self.annotations[
            self.annotations["ecdna_status"] == "negative"
        ]["depmap_id"].tolist()

    def _create_mock_annotations(self) -> pd.DataFrame:
        """Create mock CytoCellDB data for testing."""
        np.random.seed(42)

        # 577 cell lines, 139 ecDNA+, 438 ecDNA-
        n_positive = 139
        n_negative = 438
        n_total = n_positive + n_negative

        cell_lines = [f"CellLine_{i}" for i in range(n_total)]
        depmap_ids = [f"ACH-{i:06d}" for i in range(n_total)]

        ecdna_genes_options = [
            "MYC", "MYCN", "EGFR", "CDK4", "MDM2", "ERBB2",
            "MYC,EGFR", "CDK4,MDM2", "MYCN,CDK4"
        ]

        records = []
        for i in range(n_total):
            is_positive = i < n_positive
            records.append({
                "cell_line": cell_lines[i],
                "depmap_id": depmap_ids[i],
                "ecdna_status": "positive" if is_positive else "negative",
                "ecdna_genes": np.random.choice(ecdna_genes_options) if is_positive else "",
                "validation_method": "FISH" if is_positive else "metaphase",
                "source": "CytoCellDB",
            })

        return pd.DataFrame(records)

    def validate(self) -> bool:
        """Validate loaded data."""
        if self.annotations is None:
            return False

        required_cols = ["cell_line", "depmap_id", "ecdna_status"]
        return all(col in self.annotations.columns for col in required_cols)


class DepMapLoader(BaseLoader):
    """
    Load DepMap data (CRISPR screens, expression, etc.).

    DepMap provides genome-wide CRISPR knockout screens across 1000+ cell lines,
    along with multi-omics data (expression, copy number, mutations).
    """

    def __init__(self, data_dir: str = "data/depmap"):
        super().__init__(data_dir)
        self._crispr = None
        self._expression = None
        self._copy_number = None
        self._cell_lines = None
        self._mutations = None
        self._drug_sensitivity = None

    @property
    def crispr(self) -> pd.DataFrame:
        """Load CRISPR gene effect scores (Chronos)."""
        if self._crispr is None:
            path = self.data_dir / "crispr.csv"
            if path.exists():
                self._crispr = pd.read_csv(path, index_col=0)
            else:
                self._crispr = self._create_mock_crispr()
        return self._crispr

    @property
    def expression(self) -> pd.DataFrame:
        """Load expression data (TPM, log-transformed)."""
        if self._expression is None:
            path = self.data_dir / "expression.csv"
            if path.exists():
                self._expression = pd.read_csv(path, index_col=0)
            else:
                self._expression = self._create_mock_expression()
        return self._expression

    @property
    def copy_number(self) -> pd.DataFrame:
        """Load copy number data."""
        if self._copy_number is None:
            path = self.data_dir / "copy_number.csv"
            if path.exists():
                self._copy_number = pd.read_csv(path, index_col=0)
            else:
                self._copy_number = self._create_mock_copy_number()
        return self._copy_number

    @property
    def cell_lines(self) -> pd.DataFrame:
        """Load cell line metadata."""
        if self._cell_lines is None:
            path = self.data_dir / "cell_line_info.csv"
            if path.exists():
                self._cell_lines = pd.read_csv(path)
            else:
                self._cell_lines = self._create_mock_cell_lines()
        return self._cell_lines

    def load(self) -> Dict[str, pd.DataFrame]:
        """Load all DepMap data."""
        return {
            "crispr": self.crispr,
            "expression": self.expression,
            "copy_number": self.copy_number,
            "cell_lines": self.cell_lines,
        }

    def get_dependency_scores(
        self,
        cell_lines: Optional[List[str]] = None,
        genes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get CRISPR dependency scores for specified cell lines and genes.

        Args:
            cell_lines: List of DepMap IDs (e.g., ACH-000001)
            genes: List of gene symbols

        Returns:
            DataFrame of dependency scores
        """
        df = self.crispr.copy()

        if cell_lines is not None:
            df = df.loc[df.index.intersection(cell_lines)]

        if genes is not None:
            df = df[[g for g in genes if g in df.columns]]

        return df

    def get_differential_dependencies(
        self,
        group1_ids: List[str],
        group2_ids: List[str],
        method: str = "ttest"
    ) -> pd.DataFrame:
        """
        Compute differential dependencies between two groups.

        Args:
            group1_ids: DepMap IDs for group 1 (e.g., ecDNA+)
            group2_ids: DepMap IDs for group 2 (e.g., ecDNA-)
            method: Statistical test ("ttest" or "mannwhitneyu")

        Returns:
            DataFrame with differential dependency statistics
        """
        from scipy import stats

        crispr = self.crispr

        # Filter to available cell lines
        group1_ids = [x for x in group1_ids if x in crispr.index]
        group2_ids = [x for x in group2_ids if x in crispr.index]

        results = []
        for gene in crispr.columns:
            g1_scores = crispr.loc[group1_ids, gene].dropna()
            g2_scores = crispr.loc[group2_ids, gene].dropna()

            if len(g1_scores) < 3 or len(g2_scores) < 3:
                continue

            if method == "ttest":
                stat, pval = stats.ttest_ind(g1_scores, g2_scores)
            else:
                stat, pval = stats.mannwhitneyu(g1_scores, g2_scores)

            results.append({
                "gene": gene,
                "mean_group1": g1_scores.mean(),
                "mean_group2": g2_scores.mean(),
                "diff": g1_scores.mean() - g2_scores.mean(),
                "statistic": stat,
                "pvalue": pval,
            })

        df = pd.DataFrame(results)
        df["fdr"] = self._compute_fdr(df["pvalue"])

        return df.sort_values("pvalue")

    def _compute_fdr(self, pvalues: pd.Series) -> pd.Series:
        """Compute Benjamini-Hochberg FDR."""
        from scipy.stats import rankdata

        n = len(pvalues)
        ranked = rankdata(pvalues)
        fdr = pvalues * n / ranked
        fdr = np.minimum.accumulate(fdr.values[::-1])[::-1]
        return pd.Series(np.minimum(fdr, 1.0), index=pvalues.index)

    def _create_mock_crispr(self) -> pd.DataFrame:
        """Create mock CRISPR data."""
        np.random.seed(42)
        n_lines = 1000
        n_genes = 18000

        cell_lines = [f"ACH-{i:06d}" for i in range(n_lines)]
        genes = [f"GENE{i}" for i in range(n_genes)]

        # Add known cancer genes
        cancer_genes = ["MYC", "EGFR", "TP53", "KRAS", "BRCA1", "CDK4", "MDM2"]
        genes[:len(cancer_genes)] = cancer_genes

        data = np.random.normal(0, 0.5, (n_lines, n_genes))
        return pd.DataFrame(data, index=cell_lines, columns=genes)

    def _create_mock_expression(self) -> pd.DataFrame:
        """Create mock expression data."""
        np.random.seed(42)
        n_lines = 1000
        n_genes = 20000

        cell_lines = [f"ACH-{i:06d}" for i in range(n_lines)]
        genes = [f"GENE{i}" for i in range(n_genes)]

        data = np.random.lognormal(2, 1, (n_lines, n_genes))
        data = np.log2(data + 1)
        return pd.DataFrame(data, index=cell_lines, columns=genes)

    def _create_mock_copy_number(self) -> pd.DataFrame:
        """Create mock copy number data."""
        np.random.seed(42)
        n_lines = 1000
        n_genes = 20000

        cell_lines = [f"ACH-{i:06d}" for i in range(n_lines)]
        genes = [f"GENE{i}" for i in range(n_genes)]

        # Most genes diploid (2), with some amplifications
        data = np.full((n_lines, n_genes), 2.0)
        amp_mask = np.random.random((n_lines, n_genes)) < 0.05
        data[amp_mask] = np.random.uniform(3, 20, amp_mask.sum())

        return pd.DataFrame(data, index=cell_lines, columns=genes)

    def _create_mock_cell_lines(self) -> pd.DataFrame:
        """Create mock cell line metadata."""
        np.random.seed(42)
        n_lines = 1000

        lineages = ["lung", "breast", "colon", "brain", "skin", "blood", "liver"]

        return pd.DataFrame({
            "DepMap_ID": [f"ACH-{i:06d}" for i in range(n_lines)],
            "cell_line_name": [f"CellLine_{i}" for i in range(n_lines)],
            "lineage": np.random.choice(lineages, n_lines),
            "primary_disease": np.random.choice(
                ["cancer", "normal"], n_lines, p=[0.9, 0.1]
            ),
        })

    def validate(self) -> bool:
        """Validate loaded data."""
        return self._crispr is not None or (self.data_dir / "crispr.csv").exists()


class HiCLoader(BaseLoader):
    """
    Load Hi-C chromatin contact data.

    Supports loading from:
    - Cooler format (.cool, .mcool)
    - 4DN data portal format
    """

    def __init__(self, data_dir: str = "data/hic"):
        super().__init__(data_dir)
        self.contact_matrices = {}

    def load(self, cell_line: str = None) -> Dict[str, np.ndarray]:
        """Load Hi-C contact matrices."""
        if cell_line:
            return self._load_single(cell_line)

        # Load all available
        for path in self.data_dir.glob("*.mcool"):
            name = path.stem
            self.contact_matrices[name] = self._load_mcool(path)

        for path in self.data_dir.glob("*.cool"):
            name = path.stem
            self.contact_matrices[name] = self._load_cool(path)

        return self.contact_matrices

    def _load_single(self, cell_line: str) -> np.ndarray:
        """Load Hi-C for a single cell line."""
        mcool_path = self.data_dir / f"{cell_line}.mcool"
        cool_path = self.data_dir / f"{cell_line}.cool"

        if mcool_path.exists():
            return self._load_mcool(mcool_path)
        elif cool_path.exists():
            return self._load_cool(cool_path)
        else:
            logger.warning(f"No Hi-C data found for {cell_line}, returning mock")
            return self._create_mock_hic()

    def _load_mcool(self, path: Path, resolution: int = 50000) -> np.ndarray:
        """Load multi-resolution cooler file."""
        try:
            import cooler
            clr = cooler.Cooler(f"{path}::resolutions/{resolution}")
            return clr.matrix(balance=True)[:]
        except ImportError:
            logger.warning("cooler not installed, returning mock data")
            return self._create_mock_hic()
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return self._create_mock_hic()

    def _load_cool(self, path: Path) -> np.ndarray:
        """Load cooler file."""
        try:
            import cooler
            clr = cooler.Cooler(str(path))
            return clr.matrix(balance=True)[:]
        except ImportError:
            logger.warning("cooler not installed, returning mock data")
            return self._create_mock_hic()
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return self._create_mock_hic()

    def get_contact_graph(
        self,
        cell_line: str,
        chromosome: str = None,
        threshold: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert Hi-C matrix to a graph representation.

        Args:
            cell_line: Cell line name
            chromosome: Chromosome to extract (optional)
            threshold: Minimum contact frequency for edge

        Returns:
            Tuple of (edge_index, edge_weights)
        """
        matrix = self._load_single(cell_line)

        # Threshold to create sparse graph
        edges = np.where(matrix > threshold)
        edge_index = np.array([edges[0], edges[1]])
        edge_weights = matrix[edges]

        return edge_index, edge_weights

    def _create_mock_hic(self, size: int = 1000) -> np.ndarray:
        """Create mock Hi-C contact matrix."""
        np.random.seed(42)

        # Create distance-dependent contact decay
        i, j = np.meshgrid(np.arange(size), np.arange(size))
        distance = np.abs(i - j)

        # Exponential decay with distance
        contacts = np.exp(-distance / 50)

        # Add TAD structure (block diagonal)
        tad_size = 50
        for start in range(0, size, tad_size):
            end = min(start + tad_size, size)
            contacts[start:end, start:end] *= 2

        # Symmetrize and normalize
        contacts = (contacts + contacts.T) / 2
        contacts = contacts / contacts.max()

        return contacts

    def validate(self) -> bool:
        """Validate data availability."""
        return len(list(self.data_dir.glob("*.mcool"))) > 0 or \
               len(list(self.data_dir.glob("*.cool"))) > 0


class FragileSiteLoader(BaseLoader):
    """Load chromosomal fragile site annotations."""

    def __init__(self, data_dir: str = "data/supplementary"):
        super().__init__(data_dir)
        self.sites = None

    def load(self) -> pd.DataFrame:
        """Load fragile site annotations."""
        path = self.data_dir / "fragile_sites.csv"
        if path.exists():
            self.sites = pd.read_csv(path)
        else:
            logger.warning("Fragile sites not found, creating default list")
            self.sites = self._create_default_sites()

        return self.sites

    def get_distance_to_nearest(
        self,
        chromosome: str,
        position: int
    ) -> Tuple[float, str]:
        """
        Get distance to nearest fragile site.

        Args:
            chromosome: Chromosome (e.g., "chr3")
            position: Genomic position

        Returns:
            Tuple of (distance, site_id)
        """
        if self.sites is None:
            self.load()

        chrom_sites = self.sites[self.sites["chromosome"] == chromosome]

        if len(chrom_sites) == 0:
            return float('inf'), None

        # Calculate distances to midpoints
        midpoints = (chrom_sites["start"] + chrom_sites["end"]) / 2
        distances = np.abs(midpoints - position)

        idx = distances.idxmin()
        return distances[idx], chrom_sites.loc[idx, "site_id"]

    def _create_default_sites(self) -> pd.DataFrame:
        """Create default fragile site list."""
        # Well-established common fragile sites
        sites = [
            {"site_id": "FRA3B", "chromosome": "chr3", "start": 60400000, "end": 63000000, "type": "CFS", "gene": "FHIT"},
            {"site_id": "FRA16D", "chromosome": "chr16", "start": 78400000, "end": 79000000, "type": "CFS", "gene": "WWOX"},
            {"site_id": "FRA7G", "chromosome": "chr7", "start": 116000000, "end": 117000000, "type": "CFS", "gene": "CAV1"},
            {"site_id": "FRA6E", "chromosome": "chr6", "start": 162000000, "end": 163500000, "type": "CFS", "gene": "PARK2"},
            {"site_id": "FRA4F", "chromosome": "chr4", "start": 87000000, "end": 88500000, "type": "CFS", "gene": "GRID2"},
            {"site_id": "FRAXB", "chromosome": "chrX", "start": 6500000, "end": 7500000, "type": "CFS", "gene": ""},
            {"site_id": "FRA2G", "chromosome": "chr2", "start": 168000000, "end": 169000000, "type": "CFS", "gene": ""},
            {"site_id": "FRA7H", "chromosome": "chr7", "start": 130000000, "end": 131000000, "type": "CFS", "gene": ""},
            {"site_id": "FRA1H", "chromosome": "chr1", "start": 61000000, "end": 62000000, "type": "CFS", "gene": ""},
        ]
        return pd.DataFrame(sites)

    def validate(self) -> bool:
        """Validate loaded data."""
        if self.sites is None:
            return False
        required_cols = ["site_id", "chromosome", "start", "end"]
        return all(col in self.sites.columns for col in required_cols)
