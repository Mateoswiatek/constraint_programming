Torpedo Scheduling Solver - Two-Stage Optimization
==================================================

python3 run_full.py ../data/inst_config3_300_200.json best.sol 1800

Files:
- solver.py              - Stage 1: Minimize missed conversions (Simulated Annealing)
- optimizer_stage2.py    - Stage 2: Optimize torpedo count (interval coloring)
- validate_solution.py   - Solution validator
- solution.sol           - Best Stage 1 solution (0 missed, 6 torpedoes)
- optimized.sol          - Stage 2 output (0 missed, 6 torpedoes - optimal)

Usage:
------

1. Stage 1 (minimize missed):
   python3 solver.py <instance.json> <output.sol> <time_limit>

   Example:
   python3 solver.py ../data/inst_config3_300_200.json solution.sol 120

2. Stage 2 (optimize torpedoes on existing solution):
   python3 optimizer_stage2.py <instance.json> <input.sol> <output.sol> [time_limit]

   Example:
   python3 optimizer_stage2.py ../data/inst_config3_300_200.json solution.sol optimized.sol

3. Validate solution:
   python3 validate_solution.py <instance.json> <solution.sol>

Optimization Strategy:
---------------------
Stage 1: Simulated Annealing to find solution with 0 missed conversions
         - Primary objective: minimize missed conversions
         - Uses constraint-aware scheduling

Stage 2: Interval Graph Coloring for optimal torpedo assignment
         - DOES NOT modify timing or conversion assignments
         - Only re-assigns torpedo IDs optimally
         - Finds minimum number of torpedoes for given schedule
         - Fast (polynomial time) and OPTIMAL

Key Insight:
-----------
Torpedo can ARRIVE at converter parking (startConverter) BEFORE the converter
opens (converterOpensAt). Actual conversion starts at:
  actual_conv_start = max(startConverter, converterOpensAt)

Torpedo Optimization:
--------------------
The number of torpedoes needed = chromatic number of interval graph
                               = maximum clique size (overlapping trips)

For solution.sol: 6 torpedoes is OPTIMAL for the current schedule.
To reduce further would require changing conversion assignments (different
pouring->conversion mapping) which could increase missed conversions.

Results on inst_config3_300_200.json:
------------------------------------
- Stage 1: 0 missed, 5 torpedoes, 2430 desulf
- Stage 2: 0 missed, 5 torpedoes, 2430 desulf (optimal for this schedule)

Note: Solver now prioritizes minimizing max concurrent trips (= min torpedoes)
over minimizing desulf time. This results in fewer torpedoes but potentially
higher desulf time.
