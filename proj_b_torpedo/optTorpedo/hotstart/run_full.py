#!/usr/bin/env python3
"""
Full optimization pipeline with Hot Start Support

Usage:
    python3 run_full.py <instance.json> <output.sol> <time_limit> [--hot-start <prev.sol>] [--checkpoint-interval <sec>]

Examples:
    # Fresh start, 30 minutes
    python3 run_full.py ../data/inst.json solution.sol 1800

    # Hot start from previous run
    python3 run_full.py ../data/inst.json solution.sol 1800 --hot-start previous.sol

    # Continue optimizing from existing solution (2 hours with checkpoints every 5 min)
    python3 run_full.py ../data/inst.json solution.sol 7200 --hot-start solution.sol --checkpoint-interval 300
"""

import subprocess
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Full Torpedo Scheduling Pipeline with Hot Start',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('instance', help='Input JSON instance file')
    parser.add_argument('output', help='Output solution file')
    parser.add_argument('time_limit', type=int, help='Time limit in seconds')
    parser.add_argument('--hot-start', dest='hot_start', help='Previous solution for hot start')
    parser.add_argument('--checkpoint-interval', dest='checkpoint_interval', type=int, default=300,
                        help='Checkpoint interval in seconds (default: 300)')

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    stage1_sol = os.path.join(script_dir, "stage1_temp.sol")

    print("=" * 60)
    print("FULL OPTIMIZATION PIPELINE (with Hot Start)")
    print(f"Instance: {args.instance}")
    print(f"Time limit: {args.time_limit}s")
    print(f"Output: {args.output}")
    if args.hot_start:
        print(f"Hot start: {args.hot_start}")
    print(f"Checkpoint interval: {args.checkpoint_interval}s")
    print("=" * 60)

    # Stage 1
    print("\n" + "=" * 60)
    print("STAGE 1: Minimize missed + minimize max concurrent trips")
    print("=" * 60 + "\n")

    cmd = [
        sys.executable, os.path.join(script_dir, "solver.py"),
        args.instance, stage1_sol, str(args.time_limit),
        "--checkpoint-interval", str(args.checkpoint_interval)
    ]
    if args.hot_start:
        cmd.extend(["--hot-start", args.hot_start])

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("Stage 1 failed!")
        sys.exit(1)

    # Stage 2
    print("\n" + "=" * 60)
    print("STAGE 2: Optimize torpedo assignment (interval coloring)")
    print("=" * 60 + "\n")

    result = subprocess.run([
        sys.executable, os.path.join(script_dir, "optimizer_stage2.py"),
        args.instance, stage1_sol, args.output
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
        args.instance, args.output
    ])

    # Cleanup temp file
    if os.path.exists(stage1_sol):
        os.remove(stage1_sol)

    print("\n" + "=" * 60)
    print(f"DONE! Solution saved to: {args.output}")
    print("=" * 60)

if __name__ == "__main__":
    main()
