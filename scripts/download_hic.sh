#!/bin/bash
# Download Hi-C data from Aiden Lab (Rao et al. 2014)
# These are .hic files - can convert to mcool with hic2cool if needed
# Source: https://www.aidenlab.org/data.html

set -e

HIC_DIR="/home/bcheng/eclipse/data/hic"
mkdir -p "$HIC_DIR"
cd "$HIC_DIR"

echo "=== Downloading Hi-C Data from Aiden Lab ==="
echo "Target directory: $HIC_DIR"
echo "Source: Rao et al. 2014 (Cell) - GSE63525"
echo ""

# K562 - CML cell line with known ecDNA (~500MB)
echo "[1/3] Downloading K562 Hi-C (~500MB)..."
if [ ! -f "K562.hic" ]; then
    wget -q --show-progress \
        "https://hicfiles.s3.amazonaws.com/hiseq/k562/in-situ/combined.hic" \
        -O K562.hic.tmp && mv K562.hic.tmp K562.hic
    echo "K562.hic downloaded successfully"
else
    echo "K562.hic already exists, skipping"
fi

# IMR90 - Normal fibroblast control (~400MB)
echo "[2/3] Downloading IMR90 Hi-C (~400MB)..."
if [ ! -f "IMR90.hic" ]; then
    wget -q --show-progress \
        "https://hicfiles.s3.amazonaws.com/hiseq/imr90/in-situ/combined.hic" \
        -O IMR90.hic.tmp && mv IMR90.hic.tmp IMR90.hic
    echo "IMR90.hic downloaded successfully"
else
    echo "IMR90.hic already exists, skipping"
fi

# HeLa - Has ecDNA, commonly studied (~300MB)
echo "[3/3] Downloading HeLa Hi-C (~300MB)..."
if [ ! -f "HeLa.hic" ]; then
    wget -q --show-progress \
        "https://hicfiles.s3.amazonaws.com/hiseq/hela/in-situ/combined.hic" \
        -O HeLa.hic.tmp && mv HeLa.hic.tmp HeLa.hic
    echo "HeLa.hic downloaded successfully"
else
    echo "HeLa.hic already exists, skipping"
fi

echo ""
echo "=== Download Complete ==="
echo "Files downloaded:"
ls -lh "$HIC_DIR"/*.hic 2>/dev/null || echo "No .hic files found"
ls -lh "$HIC_DIR"/*.mcool 2>/dev/null || echo "No .mcool files found"
echo ""
echo "Total Hi-C data size:"
du -sh "$HIC_DIR"
echo ""
echo "NOTE: .hic files can be used directly with hic-straw or converted to mcool with hic2cool"
echo "      pip install hic2cool && hic2cool convert K562.hic K562.mcool"
