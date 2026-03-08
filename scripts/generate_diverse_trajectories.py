#!/usr/bin/env python3
"""
Generate diverse ecDNA trajectories for CircularODE training.
Each trajectory uses a unique random seed for structural variation.
"""

import subprocess
import os
import sys
import time
import random
import yaml
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
NUM_TRAJECTORIES = 500
MAX_WORKERS = 4
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REF_FASTA = PROJECT_DIR / "data" / "reference" / "hg38.fa"
ECSIMULATOR_DIR = PROJECT_DIR / "ecSimulator"
OUTPUT_DIR = PROJECT_DIR / "data" / "ecdna_trajectories_v2"
CONFIG_DIR = OUTPUT_DIR / "configs"

# Variation parameters for diverse trajectories
ORIGINS = ["episome", "chromothripsis", "two-foldback"]
TARGET_SIZES = [500000, 1000000, 1500000, 2000000, 3000000, 5000000]  # 0.5-5 Mb
NUM_INTERVALS = [1, 2, 3, 4]


def create_config(traj_id: int) -> Path:
    """Create a unique config file for each trajectory."""
    config = {
        "random_seed": traj_id + random.randint(1, 1000000),  # Unique seed
        "target_size": random.choice(TARGET_SIZES),
        "origin": random.choice(ORIGINS),
        "mean_segment_size": random.randint(50000, 300000),
        "min_segment_size": 1000,
        "num_breakpoints": "auto",
        "num_intervals": random.choice(NUM_INTERVALS),
        "same_chromosome": random.choice([True, False]),
        "allow_interval_reuse": True,
        "overlap_bed": "",
        "viral_insertion": False,
        "viral_strain": "hpv16.fasta",
        "sv_probs": {
            "del": random.uniform(0.3, 0.8),
            "dup": random.uniform(0.3, 0.8),
            "inv": random.uniform(0.2, 0.6),
            "trans": random.uniform(0.2, 0.6),
            "fback": random.uniform(0.02, 0.15),
        }
    }

    config_path = CONFIG_DIR / f"config_{traj_id:04d}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    return config_path


def generate_single_trajectory(traj_id: int) -> dict:
    """Generate a single ecDNA trajectory with unique config."""
    output_prefix = OUTPUT_DIR / f"traj_{traj_id:04d}"

    # Skip if already exists
    cycles_file = Path(f"{output_prefix}_amplicon1_cycles.txt")
    if cycles_file.exists():
        return {"id": traj_id, "status": "skipped", "message": "Already exists"}

    # Create unique config
    config_path = create_config(traj_id)

    cmd = [
        sys.executable,
        str(ECSIMULATOR_DIR / "src" / "ecSimulator.py"),
        "--ref_name", "GRCh38",
        "--ref_fasta", str(REF_FASTA),
        "--config_file", str(config_path),
        "-o", str(output_prefix),
        "-n", "1"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,  # 3 min timeout
            cwd=str(ECSIMULATOR_DIR)
        )

        if result.returncode == 0:
            return {"id": traj_id, "status": "success", "message": "Generated"}
        else:
            return {"id": traj_id, "status": "error", "message": result.stderr[:200]}
    except subprocess.TimeoutExpired:
        return {"id": traj_id, "status": "timeout", "message": "Timeout after 180s"}
    except Exception as e:
        return {"id": traj_id, "status": "error", "message": str(e)[:200]}


def main():
    print(f"Generating {NUM_TRAJECTORIES} DIVERSE ecDNA trajectories")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Using {MAX_WORKERS} parallel workers")
    print(f"Origins: {ORIGINS}")
    print(f"Target sizes: {TARGET_SIZES}")
    print("-" * 60)

    # Create directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "intermediate_structures").mkdir(exist_ok=True)

    # Check reference
    if not REF_FASTA.exists():
        print(f"ERROR: Reference fasta not found: {REF_FASTA}")
        sys.exit(1)

    start_time = time.time()
    success_count = 0
    skip_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(generate_single_trajectory, i): i
                   for i in range(NUM_TRAJECTORIES)}

        for future in as_completed(futures):
            result = future.result()
            traj_id = result["id"]
            status = result["status"]

            if status == "success":
                success_count += 1
                if success_count % 10 == 0:
                    print(f"[{success_count + skip_count + error_count}/{NUM_TRAJECTORIES}] "
                          f"{success_count} generated, {error_count} errors")
            elif status == "skipped":
                skip_count += 1
            else:
                error_count += 1
                print(f"Trajectory {traj_id:04d}: {status.upper()} - {result['message'][:50]}")

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Completed in {elapsed/60:.1f} minutes")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Errors:  {error_count}")


if __name__ == "__main__":
    main()
