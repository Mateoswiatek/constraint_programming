# Course Scheduling Optimization - Research Notes

## Problem Overview
University Course Timetabling Problem (UCTP) - NP-complete problem assigning students to class groups with constraints:
- Hard constraints: capacity, conflicts, exclusions
- Soft constraints: preferences, break minimization
- Objective: minimize sum of squared disappointments

## Best Approaches from Literature

### 1. Hybrid CP/CSP + Simulated Annealing
**Source:** [Solving the course scheduling problem by constraint programming and simulated annealing](https://www.academia.edu/3316012/Solving_the_course_scheduling_problem_by_constraint_programming_and_simulated_annealing)

- Use Constraint Programming (CP-SAT) for initial feasible solution
- Apply Simulated Annealing for optimization
- This is exactly what our hybrid solver does

### 2. Adaptive Cooling with Reheating
**Source:** [A comparison of annealing techniques for academic course scheduling](https://link.springer.com/chapter/10.1007/BFb0055883)

Key findings:
- **Adaptive cooling**: Adjust cooling rate based on acceptance ratio
- **Reheating**: When stuck in local minimum, increase temperature
- **Rule-based preprocessor**: Start from good initial solution (CP-SAT does this)

Implementation:
```python
# Adaptive cooling
if acceptance_ratio < 0.1:
    alpha = 0.99  # Cool faster when few accepts
elif acceptance_ratio > 0.5:
    alpha = 0.999  # Cool slower when many accepts

# Reheating
if no_improvement_count > threshold:
    temp = temp * reheat_factor  # e.g., 2.0
    no_improvement_count = 0
```

### 3. Multiple Neighborhood Structures
**Source:** [Investigation of simulated annealing components](https://www.academia.edu/93077133/Investigation_of_simulated_annealing_components_to_solve_the_university_course_timetabling_problem)

Use Dual-sequence SA with Round Robin neighborhood selection:
1. **Single move**: Change one student's group for one class
2. **Swap move**: Swap groups between two students for same class
3. **Chain move**: Move student A to group G1, move student B (from G1) to G2

### 4. Constraint Types
**Source:** [University Course Timetable using Constraint Satisfaction and Optimization](https://www.researchgate.net/publication/296805727_University_Course_Timetable_using_Constraint_Satisfaction_and_Optimization)

Hard constraints (must satisfy):
- One group per class per student
- Capacity limits
- No time conflicts

Soft constraints (minimize violations):
- Student preferences
- Minimize gaps between classes
- Avoid early morning/late classes

### 5. CP-SAT Best Practices
**Source:** [CP-SAT Primer](https://github.com/d-krupke/cpsat-primer)

- Use `AddHint()` to warm-start from previous solution
- Use `num_search_workers` for parallelism
- For scheduling: use interval variables and `AddNoOverlap()`

## Implementation Ideas for Our Solver

### Priority 1: Adaptive Cooling with Reheating
```python
def simulated_annealing_adaptive(state, time_limit):
    temp = initial_temp
    best_obj = current_obj
    no_improve_count = 0
    accept_count = 0
    total_count = 0

    while time.time() - start < time_limit:
        move = get_random_move()
        delta = try_move(move)

        if delta is not None:
            total_count += 1
            if delta < 0 or random.random() < exp(-delta/temp):
                apply_move(move)
                accept_count += 1
                if current_obj < best_obj:
                    best_obj = current_obj
                    no_improve_count = 0
                else:
                    no_improve_count += 1

        # Adaptive cooling every N iterations
        if total_count % 1000 == 0:
            ratio = accept_count / max(1, total_count)
            if ratio < 0.1:
                alpha = 0.995  # Cool faster
            elif ratio > 0.4:
                alpha = 0.9999  # Cool slower
            else:
                alpha = 0.9995  # Normal
            accept_count = 0
            total_count = 0

        # Reheating when stuck
        if no_improve_count > 50000:
            temp = temp * 2.0  # Reheat
            no_improve_count = 0

        temp = max(temp_min, temp * alpha)
```

### Priority 2: Swap Moves
```python
def get_swap_move(self):
    """Swap groups between two students for same class."""
    c = random.randrange(self.n_classes)
    # Find two students with different groups for class c
    candidates = []
    for s in range(self.n_students):
        g = self.student_class_group[s][c]
        if g is not None and len(self.allowed[s][c]) > 1:
            candidates.append((s, g))

    if len(candidates) < 2:
        return None

    (s1, g1), (s2, g2) = random.sample(candidates, 2)
    if g1 == g2:
        return None

    # Check if swap is valid (both can have each other's group)
    if g2 not in self.allowed[s1][c] or g1 not in self.allowed[s2][c]:
        return None

    return ('swap', s1, s2, c, g1, g2)
```

### Priority 3: Multiple Restarts
Run SA multiple times from different starting points (perturbed CP-SAT solutions).

## Current Best Result
- **Objective: 39444** (validated)
- Method: CP-SAT (30s) + SA with adaptive cooling/reheating/swap moves (570s)
- File: solution_hybrid.sol

## Improvement History

| Objective | Improvement | Time | Method | Key Changes |
|-----------|-------------|------|--------|-------------|
| 42992 | baseline | 2min | CP-SAT only | Initial feasible solution |
| 40520 | -2472 (5.7%) | 2min | CP-SAT + SA | Added swap moves, adaptive cooling, reheating |
| 39444 | -1076 (2.7%) | 10min | CP-SAT + SA | Longer SA time (570s vs 100s) |

**Total improvement: 42992 → 39444 = -3548 points (8.3%)**

### Move Type Analysis (10-min run)
From 168 improvements during SA phase:
- **Swap moves**: ~90% of improvements (majority of best moves)
- **Single moves**: ~10% of improvements
- **Reheating**: Triggered 17+ times, enabled escaping local minima

### Key Insights
1. **Swap moves are crucial** - Allow simultaneous changes that would be infeasible as separate single moves
2. **Reheating works** - Multiple reheats at temp=0.02 led to continued improvements
3. **Longer runtime helps** - More time = more chances to find better solutions


Elementy adaptive reheat
```python
    # Reheating parameters - more aggressive for long runs
    no_improve_count = 0
    reheat_threshold = 100000  # Reheat after this many iterations without improvement (~7s)
    reheat_count = 0  # Track number of reheats
    base_reheat_temp = 10.0  # Base temperature to reheat to
    max_reheat_temp = 100.0  # Max reheat temperature


        # Reheating when stuck in local minimum
        if no_improve_count >= reheat_threshold:
            reheat_count += 1
            # Progressive reheating: increase temperature more with each reheat
            # This helps escape deeper local minima
            temp = min(max_reheat_temp, base_reheat_temp * (1 + reheat_count * 0.5))
            no_improve_count = 0
            if verbose:
                elapsed = time.time() - start_time
                print(f"  [{elapsed:.1f}s] Reheating #{reheat_count} to temp={temp:.2f}")

        temp = max(temp_min, temp * alpha)
        iterations += 1
```
## References

1. Aycan, E., & Ayav, T. (2009). "Solving the Course Scheduling Problem Using Simulated Annealing." IEEE.
   https://ieeexplore.ieee.org/document/4809055/

2. Abramson, D., Krishnamoorthy, M., & Dang, H. (1996). "A comparison of annealing techniques for academic course scheduling." Springer.
   https://link.springer.com/chapter/10.1007/BFb0055883

3. Ceschia, S., Di Gaspero, L., & Schaerf, A. (2012). "Design, engineering, and experimental analysis of a simulated annealing approach to the post-enrolment course timetabling problem." Computers & Operations Research.

4. Krupke, D. (2024). "CP-SAT Primer: Using and Understanding Google OR-Tools' CP-SAT Solver."
   https://github.com/d-krupke/cpsat-primer

5. Lemos, A., et al. (2019). "Investigation of simulated annealing components to solve the university course timetabling problem." Academia.edu.
