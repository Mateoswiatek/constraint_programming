#!/usr/bin/env python3
"""
Stage 2 Optimizer: Minimize Torpedoes (Torpedo-Only Optimization)

This optimizer takes a valid solution with 0 missed conversions and optimizes
ONLY the torpedo assignment using interval graph coloring.

It does NOT modify any timing - just reassigns torpedo IDs optimally.

Usage:
    python optimizer_stage2.py <instance.json> <input.sol> <output.sol> <time_limit>

Note: time_limit is accepted for interface compatibility but mostly ignored
since interval coloring is fast.
"""

import sys
import re
from typing import List, Dict, Tuple


def parse_solution(sol_path: str) -> Dict:
    """Parse solution file."""
    with open(sol_path, 'r') as f:
        content = f.read()

    sol = {}
    for name in ['assignedTorpedo', 'assignedConversion', 'startTrip', 'startFurnace',
                 'endFurnace', 'startFullBuffer', 'endFullBuffer', 'startDesulf',
                 'endDesulf', 'startConverter', 'endConverter', 'endTrip']:
        match = re.search(rf'{name}\s*=\s*\[(.*?)\];', content, re.DOTALL)
        if match:
            values = match.group(1).replace('\n', '').replace(' ', '')
            sol[name] = [int(x) for x in values.split(',') if x]

    for name in ['numMissedConversions', 'numUsedTorpedoes', 'totalDesulfTime']:
        match = re.search(rf'{name}\s*=\s*(\d+);', content)
        if match:
            sol[name] = int(match.group(1))

    return sol


def optimize_torpedo_assignment(start_trip: List[int], end_trip: List[int]) -> Tuple[List[int], int]:
    """
    Find minimum torpedo assignment using greedy interval coloring.

    This is the classic interval graph coloring problem:
    - Each pouring is an interval [start_trip, end_trip]
    - Overlapping intervals need different colors (torpedoes)
    - Greedy approach: sort by start time, assign to first available torpedo

    This is OPTIMAL for interval graphs!

    Returns:
        (torpedo_assignment, num_torpedoes)
    """
    n = len(start_trip)

    # Sort pourings by start time
    sorted_pourings = sorted(range(n), key=lambda i: start_trip[i])

    # Greedy assignment
    torpedo_assignment = [0] * n
    torpedo_end_times = []  # End time of each torpedo's last assignment

    for i in sorted_pourings:
        s, e = start_trip[i], end_trip[i]

        # Find first available torpedo (one that finished before this starts)
        assigned = False
        for t in range(len(torpedo_end_times)):
            if torpedo_end_times[t] <= s:
                # Torpedo t is free - assign and update its end time
                torpedo_assignment[i] = t + 1  # 1-indexed
                torpedo_end_times[t] = e
                assigned = True
                break

        if not assigned:
            # Need new torpedo
            torpedo_assignment[i] = len(torpedo_end_times) + 1
            torpedo_end_times.append(e)

    return torpedo_assignment, len(torpedo_end_times)


def write_solution(sol: Dict, path: str):
    """Write solution to file."""
    with open(path, 'w') as f:
        for name in ['assignedTorpedo', 'assignedConversion', 'startTrip', 'startFurnace',
                     'endFurnace', 'startFullBuffer', 'endFullBuffer', 'startDesulf',
                     'endDesulf', 'startConverter', 'endConverter', 'endTrip']:
            values = ', '.join(str(x) for x in sol[name])
            f.write(f"{name} = [{values}];\n")

        f.write(f"numMissedConversions = {sol['numMissedConversions']};\n")
        f.write(f"numUsedTorpedoes = {sol['numUsedTorpedoes']};\n")
        f.write(f"totalDesulfTime = {sol['totalDesulfTime']};\n")


def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <instance.json> <input.sol> <output.sol> [time_limit]")
        sys.exit(1)

    instance_path = sys.argv[1]
    input_sol_path = sys.argv[2]
    output_sol_path = sys.argv[3]
    # time_limit accepted but not used (interval coloring is instant)

    print(f"Loading solution: {input_sol_path}")
    sol = parse_solution(input_sol_path)

    n = len(sol['startTrip'])
    print(f"Problem: {n} pourings")
    print(f"Input: {sol.get('numMissedConversions', 'N/A')} missed, "
          f"{sol.get('numUsedTorpedoes', 'N/A')} torpedoes, "
          f"{sol.get('totalDesulfTime', 'N/A')} desulf")

    # Optimize torpedo assignment using interval coloring
    print(f"\nOptimizing torpedo assignment...")
    new_assignment, num_torpedoes = optimize_torpedo_assignment(
        sol['startTrip'], sol['endTrip']
    )

    old_torpedoes = sol.get('numUsedTorpedoes', max(sol['assignedTorpedo']))
    print(f"Torpedo optimization: {old_torpedoes} -> {num_torpedoes}")

    # Update solution - ONLY change torpedo assignment
    sol['assignedTorpedo'] = new_assignment
    sol['numUsedTorpedoes'] = num_torpedoes

    # Write output
    write_solution(sol, output_sol_path)
    print(f"\nOutput: {sol['numMissedConversions']} missed, "
          f"{sol['numUsedTorpedoes']} torpedoes, "
          f"{sol['totalDesulfTime']} desulf")
    print(f"Written to: {output_sol_path}")


if __name__ == '__main__':
    main()
