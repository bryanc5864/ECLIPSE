"""
data processing pipelines for ECLIPSE.

Handles:
- Feature extraction from raw genomic data
- Sample/identifier harmonization across datasets
- Train/validation split generation
- Data augmentation for ecDNA
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import defaultdict

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    main data processor for ECLIPSE.

    Harmonizes data from multiple sources and creates unified feature matrices.
    """

    def __init__(
        self,
        amplicon_loader,
        cytocell_loader,
        depmap_loader,
        hic_loader=None,
        fragile_loader=None,
    ):
        self.amplicon = amplicon_loader
        self.cytocell = cytocell_loader
        self.depmap = depmap_loader
        self.hic = hic_loader
        self.fragile = fragile_loader

        self._unified_data = None

    def process(self) -> pd.DataFrame:
        """process and harmonize all data sources."""
        logger.info("Processing ECLIPSE data...")

        amplicon_data = self.amplicon.load()
        cytocell_data = self.cytocell.load()
        depmap_data = self.depmap.load()

        unified = self._harmonize_identifiers(
            amplicon_data, cytocell_data, depmap_data
        )

        unified = self._add_ecdna_labels(unified, amplicon_data, cytocell_data)

        unified = self._add_genomic_features(unified, depmap_data)

        self._unified_data = unified
        logger.info(f"Processed {len(unified)} samples")

        return unified

    def _harmonize_identifiers(
        self,
        amplicon_data: pd.DataFrame,
        cytocell_data: pd.DataFrame,
        depmap_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Harmonize sample identifiers across datasets."""
        # get all unique identifiers
        all_ids = set()

        # from CytoCellDB (has DepMap IDs directly)
        if "depmap_id" in cytocell_data.columns:
            all_ids.update(cytocell_data["depmap_id"].dropna())

        if "cell_lines" in depmap_data:
            all_ids.update(depmap_data["cell_lines"]["DepMap_ID"])

        # from AmpliconRepository (may need mapping)
        # TCGA/PCAWG samples use different IDs

        # create unified sample table
        unified = pd.DataFrame({"sample_id": list(all_ids)})

        # add source information
        unified["in_cytocell"] = unified["sample_id"].isin(
            cytocell_data.get("depmap_id", [])
        )
        unified["in_depmap"] = unified["sample_id"].isin(
            depmap_data.get("cell_lines", pd.DataFrame())
            .get("DepMap_ID", [])
        )

        return unified

    def _add_ecdna_labels(
        self,
        unified: pd.DataFrame,
        amplicon_data: pd.DataFrame,
        cytocell_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Add ecDNA status labels."""
        # map ecDNA status from CytoCellDB
        cytocell_ecdna = dict(zip(
            cytocell_data.get("depmap_id", []),
            cytocell_data.get("ecdna_status", [])
        ))

        unified["ecdna_status"] = unified["sample_id"].map(cytocell_ecdna)
        unified["ecdna_positive"] = unified["ecdna_status"] == "positive"

        # add ecDNA genes if available
        cytocell_genes = dict(zip(
            cytocell_data.get("depmap_id", []),
            cytocell_data.get("ecdna_genes", [])
        ))
        unified["ecdna_genes"] = unified["sample_id"].map(cytocell_genes)

        return unified

    def _add_genomic_features(
        self,
        unified: pd.DataFrame,
        depmap_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Add genomic features from DepMap."""
        # get cell line metadata
        if "cell_lines" in depmap_data:
            cell_info = depmap_data["cell_lines"].set_index("DepMap_ID")
            unified = unified.join(
                cell_info[["lineage", "primary_disease"]],
                on="sample_id"
            )

        return unified

    def get_split_data(
        self,
        val_size: float = 0.15,
        stratify_by: str = "ecdna_positive",
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """split data into train/validation sets."""
        if self._unified_data is None:
            self.process()

        data = self._unified_data.dropna(subset=[stratify_by])

        train, val = train_test_split(
            data,
            test_size=val_size,
            stratify=data[stratify_by],
            random_state=random_state
        )

        logger.info(f"Split: train={len(train)}, val={len(val)}")

        return train, val


class FeatureExtractor:
    """
    extract features for ECLIPSE modules.

    Generates feature matrices for:
    - Module 1: Sequence + topology + fragile site features
    - Module 2: Time-series copy number features
    - Module 3: CRISPR + expression features
    """

    def __init__(
        self,
        depmap_loader,
        hic_loader=None,
        fragile_loader=None,
        sequence_context: int = 500000,  # 500kb
    ):
        self.depmap = depmap_loader
        self.hic = hic_loader
        self.fragile = fragile_loader
        self.sequence_context = sequence_context

    def extract_module1_features(
        self,
        sample_ids: List[str],
        genomic_regions: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """extract features for ecDNA-Former (Module 1)."""
        features = {
            "sequence_features": [],
            "topology_features": [],
            "fragile_site_features": [],
            "copy_number_features": [],
        }

        for _, row in genomic_regions.iterrows():
            # extract sequence context features
            seq_feat = self._extract_sequence_features(
                row["chrom"], row["start"], row["end"]
            )
            features["sequence_features"].append(seq_feat)

            # extract Hi-C topology features
            if self.hic is not None:
                topo_feat = self._extract_topology_features(
                    row["sample_id"], row["chrom"], row["start"], row["end"]
                )
            else:
                topo_feat = np.zeros(128)  # placeholder
            features["topology_features"].append(topo_feat)

            # extract fragile site proximity features
            if self.fragile is not None:
                frag_feat = self._extract_fragile_features(
                    row["chrom"], (row["start"] + row["end"]) // 2
                )
            else:
                frag_feat = np.zeros(16)  # placeholder
            features["fragile_site_features"].append(frag_feat)

            cn_feat = self._extract_copy_number_features(
                row["sample_id"], row["chrom"], row["start"], row["end"]
            )
            features["copy_number_features"].append(cn_feat)

        # convert to arrays
        for key in features:
            features[key] = np.array(features[key])

        return features

    def extract_module3_features(
        self,
        sample_ids: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """extract features for VulnCausal (Module 3)."""
        crispr = self.depmap.crispr
        expression = self.depmap.expression
        copy_number = self.depmap.copy_number

        # filter to requested samples
        valid_ids = [s for s in sample_ids if s in crispr.index]

        return {
            "crispr": crispr.loc[valid_ids],
            "expression": expression.loc[
                expression.index.intersection(valid_ids)
            ] if expression is not None else None,
            "copy_number": copy_number.loc[
                copy_number.index.intersection(valid_ids)
            ] if copy_number is not None else None,
        }

    def _extract_sequence_features(
        self,
        chrom: str,
        start: int,
        end: int
    ) -> np.ndarray:
        """Extract sequence-based features."""
        # in production, this would use a DNA language model
        # for now, return placeholder features
        region_size = end - start
        return np.random.randn(256)  # placeholder embedding

    def _extract_topology_features(
        self,
        sample_id: str,
        chrom: str,
        start: int,
        end: int
    ) -> np.ndarray:
        """Extract Hi-C topology features."""
        try:
            edge_index, edge_weights = self.hic.get_contact_graph(sample_id)
            # convert to fixed-size feature vector
            # in production, this would use graph neural network embeddings
            return np.random.randn(128)  # placeholder
        except Exception:
            return np.zeros(128)

    def _extract_fragile_features(
        self,
        chrom: str,
        position: int
    ) -> np.ndarray:
        """Extract fragile site proximity features."""
        self.fragile.load()
        distance, site_id = self.fragile.get_distance_to_nearest(chrom, position)

        features = np.zeros(16)
        features[0] = np.log1p(distance) / 20  # normalized log distance
        features[1] = 1.0 if distance < 1e6 else 0.0  # within 1Mb
        features[2] = 1.0 if distance < 5e6 else 0.0  # within 5Mb

        return features

    def _extract_copy_number_features(
        self,
        sample_id: str,
        chrom: str,
        start: int,
        end: int
    ) -> np.ndarray:
        """Extract copy number features."""
        cn = self.depmap.copy_number

        if sample_id not in cn.index:
            return np.zeros(32)

        # get copy number values for sample
        cn_values = cn.loc[sample_id]

        features = np.zeros(32)
        features[0] = cn_values.mean()
        features[1] = cn_values.std()
        features[2] = cn_values.max()
        features[3] = (cn_values > 4).sum() / len(cn_values)  # amplified fraction

        return features


class SplitGenerator:
    """generate train/validation splits with various strategies."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def stratified_split(
        self,
        data: pd.DataFrame,
        stratify_col: str,
        train_size: float = 0.85,
        val_size: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """create stratified splits."""
        assert abs(train_size + val_size - 1.0) < 0.001

        # remove samples with missing stratification column
        data = data.dropna(subset=[stratify_col])

        train, val = train_test_split(
            data,
            train_size=train_size,
            stratify=data[stratify_col],
            random_state=self.random_state
        )

        return train, val

    def cross_validation_splits(
        self,
        data: pd.DataFrame,
        stratify_col: str,
        n_folds: int = 5
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """generate cross-validation splits."""
        data = data.dropna(subset=[stratify_col])

        skf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=self.random_state
        )

        splits = []
        for train_idx, val_idx in skf.split(data, data[stratify_col]):
            splits.append((train_idx, val_idx))

        return splits

    def leave_one_cancer_out(
        self,
        data: pd.DataFrame,
        cancer_col: str = "lineage"
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Leave-one-cancer-out cross-validation.

        Useful for testing generalization across cancer types.
        """
        cancer_types = data[cancer_col].unique()

        splits = []
        for cancer in cancer_types:
            test_mask = data[cancer_col] == cancer
            train_df = data[~test_mask]
            test_df = data[test_mask]
            splits.append((train_df, test_df))

        return splits


def create_ecdna_dataset_split(
    cytocell_loader,
    depmap_loader,
    val_size: float = 0.15,
) -> Dict[str, List[str]]:
    """create standard train/val splits for ecDNA prediction."""
    cytocell_data = cytocell_loader.load()
    depmap_data = depmap_loader.cell_lines

    # get cell lines with both ecDNA annotations and DepMap data
    valid_ids = set(cytocell_data["depmap_id"]) & set(depmap_data["DepMap_ID"])
    valid_data = cytocell_data[cytocell_data["depmap_id"].isin(valid_ids)]

    # stratified split
    generator = SplitGenerator()
    train, val = generator.stratified_split(
        valid_data,
        stratify_col="ecdna_status",
        train_size=1 - val_size,
        val_size=val_size,
    )

    return {
        "train": train["depmap_id"].tolist(),
        "val": val["depmap_id"].tolist(),
    }
