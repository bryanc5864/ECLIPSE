#!/usr/bin/env python3
"""
Extract real features for ECLIPSE Module 1 (ecDNA-Former).

Extracts:
1. Sequence features from reference genome (GC content, k-mer frequencies)
2. Fragile site proximity features
3. Copy number features from DepMap
4. Topology features from Hi-C (if available)
"""

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReferenceGenome:
    """Load and query reference genome."""

    def __init__(self, fasta_path: str):
        self.fasta_path = Path(fasta_path)
        self.sequences = {}
        self._load_genome()

    def _load_genome(self):
        """Load reference genome into memory."""
        logger.info(f"Loading reference genome from {self.fasta_path}")

        current_chrom = None
        current_seq = []

        with open(self.fasta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_chrom is not None:
                        self.sequences[current_chrom] = ''.join(current_seq)
                    # Parse chromosome name (e.g., >chr1 or >1)
                    current_chrom = line[1:].split()[0]
                    if not current_chrom.startswith('chr'):
                        current_chrom = 'chr' + current_chrom
                    current_seq = []
                else:
                    current_seq.append(line.upper())

        if current_chrom is not None:
            self.sequences[current_chrom] = ''.join(current_seq)

        logger.info(f"Loaded {len(self.sequences)} chromosomes")

    def get_sequence(self, chrom: str, start: int, end: int) -> str:
        """Get sequence for a region."""
        if not chrom.startswith('chr'):
            chrom = 'chr' + chrom

        if chrom not in self.sequences:
            return ''

        seq = self.sequences[chrom]
        start = max(0, start)
        end = min(len(seq), end)
        return seq[start:end]


class SequenceFeatureExtractor:
    """Extract sequence-based features."""

    def __init__(self, genome: ReferenceGenome):
        self.genome = genome
        self.kmer_size = 4

    def extract(self, chrom: str, start: int, end: int, output_dim: int = 256) -> np.ndarray:
        """Extract sequence features for a region."""
        # For large regions, sample 50kb windows at start, center, and end
        region_size = end - start
        max_window = 50000  # 50kb max for efficiency

        if region_size > max_window * 3:
            # Sample three windows
            center = (start + end) // 2
            seqs = [
                self.genome.get_sequence(chrom, start, start + max_window),
                self.genome.get_sequence(chrom, center - max_window//2, center + max_window//2),
                self.genome.get_sequence(chrom, end - max_window, end),
            ]
            seq = ''.join(seqs)
        else:
            seq = self.genome.get_sequence(chrom, start, end)

        if len(seq) < 100:
            return np.zeros(output_dim, dtype=np.float32)

        features = []

        # 1. GC content (1 feature)
        gc_count = seq.count('G') + seq.count('C')
        gc_content = gc_count / len(seq) if len(seq) > 0 else 0.5
        features.append(gc_content)

        # 2. Nucleotide frequencies (4 features)
        for nuc in 'ACGT':
            freq = seq.count(nuc) / len(seq) if len(seq) > 0 else 0.25
            features.append(freq)

        # 3. Dinucleotide frequencies (16 features)
        dinucs = [a+b for a in 'ACGT' for b in 'ACGT']
        dinuc_counts = Counter(seq[i:i+2] for i in range(len(seq)-1))
        total_dinucs = sum(dinuc_counts.values())
        for dinuc in dinucs:
            freq = dinuc_counts.get(dinuc, 0) / total_dinucs if total_dinucs > 0 else 0
            features.append(freq)

        # 4. K-mer frequencies (use k=3 for speed: 64 features)
        kmers = self._count_kmers(seq, 3)  # k=3 instead of k=4
        total_kmers = sum(kmers.values())
        all_kmers = self._generate_all_kmers(3)
        for kmer in all_kmers:
            freq = kmers.get(kmer, 0) / total_kmers if total_kmers > 0 else 0
            features.append(freq)

        # 5. Region size features (4 features)
        features.append(np.log1p(region_size) / 20)  # Log-scaled size
        features.append(1.0 if region_size < 1e5 else 0.0)  # <100kb
        features.append(1.0 if 1e5 <= region_size < 1e6 else 0.0)  # 100kb-1Mb
        features.append(1.0 if region_size >= 1e6 else 0.0)  # >1Mb

        features = np.array(features[:output_dim])

        # Pad if needed
        if len(features) < output_dim:
            features = np.pad(features, (0, output_dim - len(features)))

        return features.astype(np.float32)

    def _count_kmers(self, seq: str, k: int) -> Dict[str, int]:
        """Count k-mers in sequence."""
        counts = Counter()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if 'N' not in kmer:
                counts[kmer] += 1
        return counts

    def _generate_all_kmers(self, k: int) -> List[str]:
        """Generate all possible k-mers."""
        if k == 1:
            return list('ACGT')
        smaller = self._generate_all_kmers(k - 1)
        return [nuc + kmer for nuc in 'ACGT' for kmer in smaller]


class FragileSiteFeatureExtractor:
    """Extract fragile site proximity features."""

    def __init__(self, fragile_sites_path: str):
        self.sites = self._load_fragile_sites(fragile_sites_path)

    def _load_fragile_sites(self, path: str) -> pd.DataFrame:
        """Load fragile sites from BED file."""
        logger.info(f"Loading fragile sites from {path}")

        # Read file, skipping comment lines
        rows = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 4:
                    rows.append({
                        'chrom': parts[0],
                        'start': int(parts[1]),
                        'end': int(parts[2]),
                        'name': parts[3],
                    })

        df = pd.DataFrame(rows)
        logger.info(f"Loaded {len(df)} fragile sites")
        return df

    def extract(self, chrom: str, start: int, end: int, output_dim: int = 64) -> np.ndarray:
        """Extract fragile site features for a region."""
        if not chrom.startswith('chr'):
            chrom = 'chr' + chrom

        center = (start + end) // 2
        features = []

        # Filter to same chromosome
        chrom_sites = self.sites[self.sites['chrom'] == chrom]

        if len(chrom_sites) == 0:
            return np.zeros(output_dim, dtype=np.float32)

        # Calculate distances to all fragile sites on this chromosome
        site_centers = (chrom_sites['start'] + chrom_sites['end']) // 2
        distances = np.abs(site_centers.values - center)

        # 1. Distance to nearest fragile site (log-scaled)
        min_dist = distances.min()
        features.append(np.log1p(min_dist) / 20)

        # 2. Number of fragile sites within various distances
        for threshold in [1e5, 5e5, 1e6, 5e6, 1e7]:
            count = (distances < threshold).sum()
            features.append(count / len(chrom_sites))

        # 3. Binary indicators for proximity
        for threshold in [1e5, 5e5, 1e6, 5e6]:
            features.append(1.0 if min_dist < threshold else 0.0)

        # 4. Distribution of nearby fragile sites
        nearby_mask = distances < 5e6
        if nearby_mask.sum() > 0:
            nearby_dists = distances[nearby_mask]
            features.append(np.mean(nearby_dists) / 5e6)
            features.append(np.std(nearby_dists) / 5e6 if len(nearby_dists) > 1 else 0)
        else:
            features.extend([1.0, 0.0])

        # 5. Is region inside a fragile site?
        inside = ((chrom_sites['start'] <= center) & (chrom_sites['end'] >= center)).any()
        features.append(1.0 if inside else 0.0)

        # 6. Overlaps with region
        overlaps = ((chrom_sites['start'] <= end) & (chrom_sites['end'] >= start)).sum()
        features.append(overlaps / max(len(chrom_sites), 1))

        features = np.array(features[:output_dim])

        # Pad if needed
        if len(features) < output_dim:
            features = np.pad(features, (0, output_dim - len(features)))

        return features.astype(np.float32)


class CopyNumberFeatureExtractor:
    """Extract copy number features from DepMap."""

    def __init__(self, copy_number_path: str, gene_info_path: Optional[str] = None):
        logger.info(f"Loading copy number data from {copy_number_path}")
        self.cn_data = pd.read_csv(copy_number_path, index_col=0)
        logger.info(f"Loaded copy number for {len(self.cn_data)} samples, {len(self.cn_data.columns)} genes")

        # Parse gene locations from column names (format: GENE (ENTREZ_ID))
        self.gene_info = self._parse_gene_info()

    def _parse_gene_info(self) -> Dict[str, dict]:
        """Parse gene names from column headers."""
        gene_info = {}
        for col in self.cn_data.columns:
            match = re.match(r'(\w+)\s*\((\d+)\)', col)
            if match:
                gene_name, entrez_id = match.groups()
                gene_info[col] = {'name': gene_name, 'entrez_id': entrez_id}
        return gene_info

    def extract(self, sample_id: str, chrom: str, start: int, end: int,
                output_dim: int = 32) -> np.ndarray:
        """Extract copy number features."""
        features = []

        if sample_id not in self.cn_data.index:
            return np.zeros(output_dim, dtype=np.float32)

        cn_values = self.cn_data.loc[sample_id].values

        # Global statistics
        features.append(np.mean(cn_values))
        features.append(np.std(cn_values))
        features.append(np.median(cn_values))
        features.append(np.max(cn_values))
        features.append(np.min(cn_values))

        # Amplification statistics
        amp_threshold = 4
        features.append((cn_values > amp_threshold).mean())  # Fraction amplified
        features.append((cn_values > 6).mean())  # High amplification
        features.append((cn_values > 8).mean())  # Very high amplification

        # Deletion statistics
        features.append((cn_values < 1).mean())  # Hemizygous deletions
        features.append((cn_values < 0.5).mean())  # Homozygous deletions

        # Distribution features
        features.append(np.percentile(cn_values, 25))
        features.append(np.percentile(cn_values, 75))
        features.append(np.percentile(cn_values, 90))
        features.append(np.percentile(cn_values, 95))
        features.append(np.percentile(cn_values, 99))

        # Variability
        features.append(np.var(cn_values))
        iqr = np.percentile(cn_values, 75) - np.percentile(cn_values, 25)
        features.append(iqr)

        features = np.array(features[:output_dim])

        # Pad if needed
        if len(features) < output_dim:
            features = np.pad(features, (0, output_dim - len(features)))

        return features.astype(np.float32)


class TopologyFeatureExtractor:
    """Extract Hi-C topology features."""

    def __init__(self, hic_path: Optional[str] = None):
        self.hic_path = hic_path
        self.cooler = None

        if hic_path and Path(hic_path).exists():
            try:
                import cooler
                self.cooler = cooler.Cooler(f"{hic_path}::/resolutions/50000")
                logger.info(f"Loaded Hi-C data from {hic_path}")
            except Exception as e:
                logger.warning(f"Could not load Hi-C data: {e}")

    def extract(self, chrom: str, start: int, end: int, output_dim: int = 256) -> np.ndarray:
        """Extract topology features from Hi-C."""
        if self.cooler is None:
            # Return basic positional features if no Hi-C available
            return self._extract_positional_features(chrom, start, end, output_dim)

        try:
            if not chrom.startswith('chr'):
                chrom = 'chr' + chrom

            # Get Hi-C matrix for region
            region = f"{chrom}:{start}-{end}"
            matrix = self.cooler.matrix(balance=True).fetch(region)

            if matrix.size == 0:
                return self._extract_positional_features(chrom, start, end, output_dim)

            features = []

            # Matrix statistics
            valid_values = matrix[~np.isnan(matrix)]
            if len(valid_values) > 0:
                features.append(np.mean(valid_values))
                features.append(np.std(valid_values))
                features.append(np.max(valid_values))
                features.append(np.median(valid_values))
            else:
                features.extend([0, 0, 0, 0])

            # Diagonal strengths (TAD-like features)
            for offset in range(min(10, matrix.shape[0])):
                diag = np.diag(matrix, offset)
                valid_diag = diag[~np.isnan(diag)]
                if len(valid_diag) > 0:
                    features.append(np.mean(valid_diag))
                else:
                    features.append(0)

            # Flatten and use as features
            flat = matrix.flatten()
            flat = flat[~np.isnan(flat)]
            if len(flat) > output_dim - len(features):
                flat = flat[:output_dim - len(features)]
            features.extend(flat.tolist())

            features = np.array(features[:output_dim])
            if len(features) < output_dim:
                features = np.pad(features, (0, output_dim - len(features)))

            return features.astype(np.float32)

        except Exception as e:
            logger.debug(f"Hi-C extraction failed for {chrom}:{start}-{end}: {e}")
            return self._extract_positional_features(chrom, start, end, output_dim)

    def _extract_positional_features(self, chrom: str, start: int, end: int,
                                     output_dim: int) -> np.ndarray:
        """Extract basic positional features when Hi-C unavailable."""
        features = []

        # Chromosome encoding (one-hot for chr1-22, X, Y)
        chrom_num = chrom.replace('chr', '')
        for i in range(1, 23):
            features.append(1.0 if chrom_num == str(i) else 0.0)
        features.append(1.0 if chrom_num == 'X' else 0.0)
        features.append(1.0 if chrom_num == 'Y' else 0.0)

        # Position features (normalized)
        features.append(start / 3e8)  # Normalized start
        features.append(end / 3e8)    # Normalized end
        features.append((end - start) / 1e7)  # Normalized size

        # Region size categories
        size = end - start
        features.append(1.0 if size < 1e5 else 0.0)  # <100kb
        features.append(1.0 if 1e5 <= size < 1e6 else 0.0)  # 100kb-1Mb
        features.append(1.0 if 1e6 <= size < 1e7 else 0.0)  # 1-10Mb
        features.append(1.0 if size >= 1e7 else 0.0)  # >10Mb

        features = np.array(features[:output_dim])
        if len(features) < output_dim:
            features = np.pad(features, (0, output_dim - len(features)))

        return features.astype(np.float32)


def parse_interval(interval_str: str) -> List[Tuple[str, int, int]]:
    """Parse interval string like 'chr1:1000-2000,chr2:3000-4000'."""
    regions = []
    for part in interval_str.split(','):
        part = part.strip()
        if ':' in part and '-' in part:
            try:
                chrom, coords = part.split(':')
                start, end = coords.split('-')
                regions.append((chrom, int(start), int(end)))
            except:
                continue
    return regions


def main():
    parser = argparse.ArgumentParser(description="Extract features for ECLIPSE")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="data/features", help="Output directory")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "all"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Kim 2020 labels
    logger.info("Loading Kim 2020 labels...")
    labels_df = pd.read_excel(data_dir / "ecdna_labels" / "kim2020_supplementary_tables.xlsx", sheet_name=0)
    labels_df["ecdna_positive"] = (labels_df["amplicon_classification"] == "Circular").astype(int)

    # Get unique samples with their labels and intervals
    sample_data = []
    for _, row in labels_df.iterrows():
        regions = parse_interval(row['amplicon_intervals'])
        if regions:
            # Use first region as representative
            chrom, start, end = regions[0]
            sample_data.append({
                'sample_id': row['sample_barcode'],
                'amplicon_idx': row['amplicon_index'],
                'ecdna_positive': row['ecdna_positive'],
                'chrom': chrom,
                'start': start,
                'end': end,
                'all_intervals': row['amplicon_intervals'],
            })

    samples_df = pd.DataFrame(sample_data)
    logger.info(f"Parsed {len(samples_df)} samples with genomic regions")
    logger.info(f"  ecDNA+: {samples_df['ecdna_positive'].sum()}")
    logger.info(f"  ecDNA-: {len(samples_df) - samples_df['ecdna_positive'].sum()}")

    # Initialize feature extractors
    logger.info("Initializing feature extractors...")

    # Sequence features (from reference genome)
    ref_path = data_dir / "reference" / "hg38.fa"
    if ref_path.exists():
        genome = ReferenceGenome(str(ref_path))
        seq_extractor = SequenceFeatureExtractor(genome)
    else:
        logger.warning(f"Reference genome not found at {ref_path}")
        seq_extractor = None

    # Fragile site features
    fragile_path = data_dir / "supplementary" / "fragile_sites_hg38.bed"
    if fragile_path.exists():
        fragile_extractor = FragileSiteFeatureExtractor(str(fragile_path))
    else:
        fragile_path = data_dir / "supplementary" / "major_fragile_sites_hg38.bed"
        if fragile_path.exists():
            fragile_extractor = FragileSiteFeatureExtractor(str(fragile_path))
        else:
            logger.warning("Fragile sites file not found")
            fragile_extractor = None

    # Copy number features
    cn_path = data_dir / "depmap" / "copy_number.csv"
    if cn_path.exists():
        cn_extractor = CopyNumberFeatureExtractor(str(cn_path))
    else:
        logger.warning(f"Copy number data not found at {cn_path}")
        cn_extractor = None

    # Hi-C topology features (skip for now - too slow, use positional features instead)
    # hic_path = data_dir / "hic" / "K562_RCMC.mcool"
    # if not hic_path.exists():
    #     hic_path = data_dir / "hic" / "GM12878.mcool"
    topo_extractor = TopologyFeatureExtractor(None)  # Use positional features only
    logger.info("Using positional features for topology (Hi-C disabled for speed)")

    # Extract features for all samples
    logger.info("Extracting features...")

    all_features = {
        'sequence_features': [],
        'topology_features': [],
        'fragile_site_features': [],
        'copy_number_features': [],
    }
    all_labels = []
    all_sample_ids = []

    for idx, row in samples_df.iterrows():
        if idx % 100 == 0:
            logger.info(f"  Processing sample {idx}/{len(samples_df)}")

        chrom = row['chrom']
        start = row['start']
        end = row['end']
        sample_id = row['sample_id']

        # Sequence features
        if seq_extractor:
            seq_feat = seq_extractor.extract(chrom, start, end, output_dim=256)
        else:
            seq_feat = np.zeros(256, dtype=np.float32)
        all_features['sequence_features'].append(seq_feat)

        # Topology features
        topo_feat = topo_extractor.extract(chrom, start, end, output_dim=256)
        all_features['topology_features'].append(topo_feat)

        # Fragile site features
        if fragile_extractor:
            frag_feat = fragile_extractor.extract(chrom, start, end, output_dim=64)
        else:
            frag_feat = np.zeros(64, dtype=np.float32)
        all_features['fragile_site_features'].append(frag_feat)

        # Copy number features
        if cn_extractor:
            cn_feat = cn_extractor.extract(sample_id, chrom, start, end, output_dim=32)
        else:
            cn_feat = np.zeros(32, dtype=np.float32)
        all_features['copy_number_features'].append(cn_feat)

        all_labels.append(row['ecdna_positive'])
        all_sample_ids.append(sample_id)

    # Convert to arrays
    for key in all_features:
        all_features[key] = np.array(all_features[key])
        logger.info(f"  {key}: shape {all_features[key].shape}")

    all_labels = np.array(all_labels)

    # Split data
    np.random.seed(42)
    n_samples = len(all_sample_ids)
    indices = np.random.permutation(n_samples)

    n_val = int(n_samples * 0.15)
    n_train = n_samples - n_val

    splits = {
        'train': indices[:n_train],
        'val': indices[n_train:],
    }

    # Save features for each split
    for split_name, split_indices in splits.items():
        if args.split != 'all' and args.split != split_name:
            continue

        split_features = {k: v[split_indices] for k, v in all_features.items()}
        split_labels = all_labels[split_indices]
        split_ids = [all_sample_ids[i] for i in split_indices]

        output_file = output_dir / f"module1_features_{split_name}.npz"
        np.savez(
            output_file,
            **split_features,
            labels=split_labels,
            sample_ids=split_ids,
        )

        logger.info(f"Saved {split_name} features to {output_file}")
        logger.info(f"  Samples: {len(split_indices)}")
        logger.info(f"  ecDNA+: {split_labels.sum()}")

    logger.info("Feature extraction complete!")


if __name__ == "__main__":
    main()
