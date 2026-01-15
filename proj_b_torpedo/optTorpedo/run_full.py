#!/usr/bin/env python3
"""
Full optimization pipeline: Stage 1 + Stage 2

Usage:
    python3 run_full.py <instance.json> <output.sol> <time_limit_seconds>

Example:
    python3 run_full.py ../data/inst_config3_300_200.json best.sol 1800
"""

import subprocess
import sys
import os

def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <instance.json> <output.sol> <time_limit>")
        sys.exit(1)

    instance = sys.argv[1]
    output = sys.argv[2]
    time_limit = sys.argv[3]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    stage1_sol = os.path.join(script_dir, "stage1_temp.sol")

    print("=" * 60)
    print(f"FULL OPTIMIZATION PIPELINE")
    print(f"Instance: {instance}")
    print(f"Time limit: {time_limit}s")
    print(f"Output: {output}")
    print("=" * 60)

    # Stage 1
    print("\n" + "=" * 60)
    print("STAGE 1: Minimize missed + minimize max concurrent trips")
    print("=" * 60 + "\n")

    result = subprocess.run([
        sys.executable, os.path.join(script_dir, "solver.py"),
        instance, stage1_sol, time_limit
    ])

    if result.returncode != 0:
        print("Stage 1 failed!")
        sys.exit(1)

    # Stage 2
    print("\n" + "=" * 60)
    print("STAGE 2: Optimize torpedo assignment (interval coloring)")
    print("=" * 60 + "\n")

    result = subprocess.run([
        sys.executable, os.path.join(script_dir, "optimizer_stage2.py"),
        instance, stage1_sol, output
    ])

    if result.returncode != 0:
        print("Stage 2 failed!")
        sys.exit(1)

    # Validate
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60 + "\n")

    result = subprocess.run([
        sys.executable, os.path.join(script_dir, "validate_solution.py"),
        instance, output
    ])

    # Cleanup temp file
    if os.path.exists(stage1_sol):
        os.remove(stage1_sol)

    print("\n" + "=" * 60)
    print(f"DONE! Solution saved to: {output}")
    print("=" * 60)

if __name__ == "__main__":
    main()
