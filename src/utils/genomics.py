"""
Genomic Utilities for ECLIPSE.

Handles:
- Genomic coordinate parsing and manipulation
- Sequence processing
- Chromosome handling
- BED file parsing
"""

import re
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class GenomicCoordinates:
    """Genomic coordinate representation."""
    chromosome: str
    start: int
    end: int
    strand: str = "+"
    name: Optional[str] = None

    def __post_init__(self):
        # Normalize chromosome name
        if not self.chromosome.startswith("chr"):
            self.chromosome = f"chr{self.chromosome}"

    @property
    def length(self) -> int:
        """Get region length."""
        return self.end - self.start

    @property
    def midpoint(self) -> int:
        """Get region midpoint."""
        return (self.start + self.end) // 2

    def overlaps(self, other: "GenomicCoordinates") -> bool:
        """Check if regions overlap."""
        if self.chromosome != other.chromosome:
            return False
        return self.start < other.end and other.start < self.end

    def distance_to(self, other: "GenomicCoordinates") -> int:
        """Get distance to another region (0 if overlapping)."""
        if self.chromosome != other.chromosome:
            return float('inf')
        if self.overlaps(other):
            return 0
        return min(abs(self.start - other.end), abs(other.start - self.end))

    def expand(self, bp: int) -> "GenomicCoordinates":
        """Expand region by bp on each side."""
        return GenomicCoordinates(
            chromosome=self.chromosome,
            start=max(0, self.start - bp),
            end=self.end + bp,
            strand=self.strand,
            name=self.name,
        )

    def to_string(self) -> str:
        """Convert to UCSC-style string."""
        return f"{self.chromosome}:{self.start}-{self.end}"

    @classmethod
    def from_string(cls, coord_string: str) -> "GenomicCoordinates":
        """Parse from string like 'chr1:1000-2000'."""
        match = re.match(r"(chr\w+):(\d+)-(\d+)", coord_string)
        if not match:
            raise ValueError(f"Invalid coordinate string: {coord_string}")
        return cls(
            chromosome=match.group(1),
            start=int(match.group(2)),
            end=int(match.group(3)),
        )


class SequenceProcessor:
    """Process DNA sequences for model input."""

    # Nucleotide encoding
    NUCLEOTIDE_TO_IDX = {
        'A': 0, 'a': 0,
        'C': 1, 'c': 1,
        'G': 2, 'g': 2,
        'T': 3, 't': 3,
        'N': 4, 'n': 4,
    }

    # Complement mapping
    COMPLEMENT = {
        'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N',
        'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'n': 'n',
    }

    def __init__(self, max_length: int = 6000):
        """
        Initialize sequence processor.

        Args:
            max_length: Maximum sequence length
        """
        self.max_length = max_length

    def encode(self, sequence: str) -> np.ndarray:
        """
        Encode sequence to integer array.

        Args:
            sequence: DNA sequence string

        Returns:
            Integer array of nucleotide indices
        """
        encoded = np.array([
            self.NUCLEOTIDE_TO_IDX.get(nt, 4)
            for nt in sequence
        ])

        # Pad or truncate
        if len(encoded) > self.max_length:
            # Take center
            start = (len(encoded) - self.max_length) // 2
            encoded = encoded[start:start + self.max_length]
        elif len(encoded) < self.max_length:
            # Pad with N (4)
            padding = np.full(self.max_length - len(encoded), 4)
            encoded = np.concatenate([encoded, padding])

        return encoded

    def one_hot_encode(self, sequence: str) -> np.ndarray:
        """
        One-hot encode sequence.

        Args:
            sequence: DNA sequence string

        Returns:
            One-hot encoded array [length, 5]
        """
        encoded = self.encode(sequence)
        one_hot = np.zeros((len(encoded), 5))
        one_hot[np.arange(len(encoded)), encoded] = 1
        return one_hot

    def reverse_complement(self, sequence: str) -> str:
        """Get reverse complement of sequence."""
        return ''.join(self.COMPLEMENT.get(nt, 'N') for nt in reversed(sequence))

    def gc_content(self, sequence: str) -> float:
        """Calculate GC content."""
        gc = sum(1 for nt in sequence.upper() if nt in 'GC')
        total = sum(1 for nt in sequence.upper() if nt in 'ACGT')
        return gc / total if total > 0 else 0.0

    def kmer_frequencies(self, sequence: str, k: int = 3) -> Dict[str, float]:
        """
        Calculate k-mer frequencies.

        Args:
            sequence: DNA sequence
            k: k-mer size

        Returns:
            Dictionary of k-mer frequencies
        """
        sequence = sequence.upper()
        kmers = {}
        total = 0

        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            if 'N' not in kmer:
                kmers[kmer] = kmers.get(kmer, 0) + 1
                total += 1

        return {kmer: count / total for kmer, count in kmers.items()} if total > 0 else {}


def parse_bed_file(filepath: str) -> List[GenomicCoordinates]:
    """
    Parse BED file into list of coordinates.

    Args:
        filepath: Path to BED file

    Returns:
        List of GenomicCoordinates
    """
    coordinates = []

    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('track') or not line.strip():
                continue

            parts = line.strip().split('\t')
            if len(parts) >= 3:
                coord = GenomicCoordinates(
                    chromosome=parts[0],
                    start=int(parts[1]),
                    end=int(parts[2]),
                    name=parts[3] if len(parts) > 3 else None,
                    strand=parts[5] if len(parts) > 5 else "+",
                )
                coordinates.append(coord)

    return coordinates


def liftover_coordinates(
    coordinates: List[GenomicCoordinates],
    from_assembly: str = "hg19",
    to_assembly: str = "hg38",
) -> List[Optional[GenomicCoordinates]]:
    """
    Lift over coordinates between genome assemblies.

    Note: Requires pyliftover package.

    Args:
        coordinates: List of coordinates to lift over
        from_assembly: Source assembly
        to_assembly: Target assembly

    Returns:
        List of lifted coordinates (None if failed)
    """
    try:
        from pyliftover import LiftOver
        lo = LiftOver(from_assembly, to_assembly)
    except ImportError:
        raise ImportError("pyliftover required for liftover. Install with: pip install pyliftover")

    lifted = []
    for coord in coordinates:
        # LiftOver uses 0-based coordinates
        result = lo.convert_coordinate(coord.chromosome, coord.start)
        if result and len(result) > 0:
            new_chrom, new_pos, strand, _ = result[0]
            # Approximate end position
            new_end = new_pos + coord.length
            lifted.append(GenomicCoordinates(
                chromosome=new_chrom,
                start=new_pos,
                end=new_end,
                strand=strand,
                name=coord.name,
            ))
        else:
            lifted.append(None)

    return lifted


# Chromosome size reference (hg38)
CHROMOSOME_SIZES_HG38 = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559,
    "chr4": 190214555, "chr5": 181538259, "chr6": 170805979,
    "chr7": 159345973, "chr8": 145138636, "chr9": 138394717,
    "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
    "chr16": 90338345, "chr17": 83257441, "chr18": 80373285,
    "chr19": 58617616, "chr20": 64444167, "chr21": 46709983,
    "chr22": 50818468, "chrX": 156040895, "chrY": 57227415,
}


def get_chromosome_bins(
    chromosome: str,
    bin_size: int = 50000,
    assembly: str = "hg38",
) -> List[Tuple[int, int]]:
    """
    Get genomic bins for a chromosome.

    Args:
        chromosome: Chromosome name
        bin_size: Bin size in bp
        assembly: Genome assembly

    Returns:
        List of (start, end) tuples
    """
    if assembly == "hg38":
        chrom_size = CHROMOSOME_SIZES_HG38.get(chromosome, 0)
    else:
        raise ValueError(f"Unknown assembly: {assembly}")

    bins = []
    for start in range(0, chrom_size, bin_size):
        end = min(start + bin_size, chrom_size)
        bins.append((start, end))

    return bins
