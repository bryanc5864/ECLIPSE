#!/usr/bin/env python3
"""
Generate ecDNA trajectories for CircularODE training.
Runs ecSimulator in batch mode to create ~500 trajectories.
"""

import subprocess
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
NUM_TRAJECTORIES = 500
MAX_WORKERS = 4  # Parallel processes
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REF_FASTA = PROJECT_DIR / "data" / "reference" / "hg38.fa"
ECSIMULATOR_DIR = PROJECT_DIR / "ecSimulator"
OUTPUT_DIR = PROJECT_DIR / "data" / "ecdna_trajectories"

def generate_single_trajectory(traj_id: int) -> dict:
    """Generate a single ecDNA trajectory."""
    output_prefix = OUTPUT_DIR / f"traj_{traj_id:04d}"

    # Skip if already exists
    cycles_file = Path(f"{output_prefix}_amplicon1_cycles.txt")
    if cycles_file.exists():
        return {"id": traj_id, "status": "skipped", "message": "Already exists"}

    cmd = [
        sys.executable,
        str(ECSIMULATOR_DIR / "src" / "ecSimulator.py"),
        "--ref_name", "GRCh38",
        "--ref_fasta", str(REF_FASTA),
        "-o", str(output_prefix),
        "-n", "1"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 min timeout per trajectory
            cwd=str(ECSIMULATOR_DIR)
        )

        if result.returncode == 0:
            return {"id": traj_id, "status": "success", "message": "Generated"}
        else:
            return {"id": traj_id, "status": "error", "message": result.stderr[:200]}
    except subprocess.TimeoutExpired:
        return {"id": traj_id, "status": "timeout", "message": "Timeout after 120s"}
    except Exception as e:
        return {"id": traj_id, "status": "error", "message": str(e)[:200]}

def main():
    print(f"Generating {NUM_TRAJECTORIES} ecDNA trajectories")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Using {MAX_WORKERS} parallel workers")
    print("-" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check reference exists
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
                print(f"[{success_count + skip_count + error_count}/{NUM_TRAJECTORIES}] "
                      f"Trajectory {traj_id:04d}: SUCCESS")
            elif status == "skipped":
                skip_count += 1
            else:
                error_count += 1
                print(f"[{success_count + skip_count + error_count}/{NUM_TRAJECTORIES}] "
                      f"Trajectory {traj_id:04d}: {status.upper()} - {result['message'][:50]}")

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Completed in {elapsed/60:.1f} minutes")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Errors:  {error_count}")
    print(f"  Total:   {success_count + skip_count}")

if __name__ == "__main__":
    main()
