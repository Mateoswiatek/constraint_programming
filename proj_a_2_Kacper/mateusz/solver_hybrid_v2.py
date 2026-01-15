#!/usr/bin/env python3
"""
Hybrid enrollment solver v2: OR-Tools CP-SAT + Enhanced Local Search.
Improvements over v1:
- Adaptive reheat threshold (increases over time)
- Varied reheat temperatures
- Chain swap moves (3 students)
- Periodic restarts from best solution
"""

from __future__ import annotations

import argparse
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ortools.sat.python import cp_model


@dataclass
class Instance:
    n_students: int
    n_days: int
    n_classes: int
    n_groups: int
    n_time_units_in_hour: int
    student_break_importance: List[int]
    student_prefers: List[List[int]]
    class_duration: List[int]
    class_size: List[int]
    group_class: List[int]
    group_start: List[int]
    group_day: List[int]
    groups_conflicts: List[List[bool]]


def parse_dzn(path: Path) -> Instance:
    text = path.read_text()

    def parse_int(name: str) -> int:
        m = re.search(rf"{name}\s*=\s*([-\d]+)\s*;", text)
        if not m:
            raise ValueError(f"Missing integer parameter {name}")
        return int(m.group(1))

    def parse_array(name: str) -> List[int]:
        m = re.search(rf"{name}\s*=\s*\[([^\]]+)\]\s*;", text, re.DOTALL)
        if not m:
            raise ValueError(f"Missing array parameter {name}")
        raw = m.group(1)
        values = []
        for tok in re.split(r",", raw):
            tok = tok.strip()
            if tok:
                values.append(int(tok))
        return values

    def parse_array2d(name: str, n1: int, n2: int, convert=int) -> List[List]:
        m = re.search(
            rf"{name}\s*=\s*array2d\([^\[]*,\s*\[([^\]]+)\]\s*\);", text, re.DOTALL
        )
        if not m:
            raise ValueError(f"Missing array2d parameter {name}")
        raw = m.group(1)
        values = []
        for tok in re.split(r",", raw):
            tok = tok.strip()
            if tok:
                if convert == bool:
                    values.append(tok.lower() == "true")
                else:
                    values.append(convert(tok))
        matrix = []
        idx = 0
        for _ in range(n1):
            row = values[idx : idx + n2]
            matrix.append(row)
            idx += n2
        return matrix

    n_students = parse_int("n_students")
    n_days = parse_int("n_days")
    n_classes = parse_int("n_classes")
    n_groups = parse_int("n_groups")
    n_time_units_in_hour = parse_int("n_time_units_in_hour")

    return Instance(
        n_students=n_students,
        n_days=n_days,
        n_classes=n_classes,
        n_groups=n_groups,
        n_time_units_in_hour=n_time_units_in_hour,
        student_break_importance=parse_array("student_break_importance"),
        student_prefers=parse_array2d("student_prefers", n_students, n_groups, int),
        class_duration=parse_array("class_duration"),
        class_size=parse_array("class_size"),
        group_class=parse_array("group_class"),
        group_start=parse_array("group_start"),
        group_day=parse_array("group_day"),
        groups_conflicts=parse_array2d("groups_conflicts", n_groups, n_groups, bool),
    )


class SolutionState:
    """Mutable solution state for local search."""

    def __init__(self, inst: Instance, assignment: List[List[int]]):
        self.inst = inst
        self.n_students = inst.n_students
        self.n_groups = inst.n_groups
        self.n_classes = inst.n_classes
        self.n_days = inst.n_days
        self.units = inst.n_time_units_in_hour

        # 0-based indexing
        self.group_class = [c - 1 for c in inst.group_class]
        self.group_day = [d - 1 for d in inst.group_day]
        self.group_start = inst.group_start
        self.group_duration = [
            inst.class_duration[self.group_class[g]] for g in range(self.n_groups)
        ]

        # Groups by class/day
        self.groups_by_class = [[] for _ in range(self.n_classes)]
        for g in range(self.n_groups):
            self.groups_by_class[self.group_class[g]].append(g)

        self.groups_by_day = [[] for _ in range(self.n_days)]
        for g in range(self.n_groups):
            self.groups_by_day[self.group_day[g]].append(g)

        # Allowed groups per student per class
        self.allowed = [
            [[] for _ in range(self.n_classes)] for _ in range(self.n_students)
        ]
        for s in range(self.n_students):
            for c in range(self.n_classes):
                self.allowed[s][c] = [
                    g
                    for g in self.groups_by_class[c]
                    if inst.student_prefers[s][g] != -1
                ]

        # Current assignment (0-based groups)
        self.assignment = [[g - 1 for g in groups] for groups in assignment]

        # Group counts
        self.group_counts = [0] * self.n_groups
        for s in range(self.n_students):
            for g in self.assignment[s]:
                self.group_counts[g] += 1

        # Build student-class mapping for fast lookup
        self.student_class_group = [
            [None for _ in range(self.n_classes)] for _ in range(self.n_students)
        ]
        for s in range(self.n_students):
            for g in self.assignment[s]:
                self.student_class_group[s][self.group_class[g]] = g

        # Build list of (student, class) pairs that have multiple options
        self.moveable_pairs = []
        for s in range(self.n_students):
            for c in range(self.n_classes):
                if len(self.allowed[s][c]) > 1:
                    self.moveable_pairs.append((s, c))

        # Build index of students by group (for chain swaps)
        self.students_by_group = [[] for _ in range(self.n_groups)]
        for s in range(self.n_students):
            for g in self.assignment[s]:
                self.students_by_group[g].append(s)

    def get_random_move(self) -> Optional[Tuple[int, int, int, int]]:
        """Generate a random single move dynamically."""
        if not self.moveable_pairs:
            return None
        s, c = random.choice(self.moveable_pairs)
        current_g = self.student_class_group[s][c]
        if current_g is None:
            return None
        allowed = self.allowed[s][c]
        if len(allowed) <= 1:
            return None
        new_g = random.choice(allowed)
        while new_g == current_g:
            new_g = random.choice(allowed)
        return (s, c, current_g, new_g)

    def get_swap_move(self) -> Optional[Tuple[str, int, int, int, int, int]]:
        """Generate a swap move between two students for the same class."""
        if self.n_classes == 0:
            return None
        c = random.randrange(self.n_classes)

        candidates = []
        for s in range(self.n_students):
            g = self.student_class_group[s][c]
            if g is not None and len(self.allowed[s][c]) > 1:
                candidates.append((s, g))

        if len(candidates) < 2:
            return None

        idx1, idx2 = random.sample(range(len(candidates)), 2)
        s1, g1 = candidates[idx1]
        s2, g2 = candidates[idx2]

        if g1 == g2:
            return None

        if g2 not in self.allowed[s1][c] or g1 not in self.allowed[s2][c]:
            return None

        return ("swap", s1, s2, c, g1, g2)

    def get_chain_swap_move(self) -> Optional[Tuple]:
        """Generate a chain swap move: s1->g2, s2->g3, s3->g1 (cyclic)."""
        if self.n_classes == 0:
            return None
        c = random.randrange(self.n_classes)

        # Find students grouped by their current group for this class
        group_to_students = {}
        for s in range(self.n_students):
            g = self.student_class_group[s][c]
            if g is not None and len(self.allowed[s][c]) > 1:
                if g not in group_to_students:
                    group_to_students[g] = []
                group_to_students[g].append(s)

        groups = list(group_to_students.keys())
        if len(groups) < 3:
            return None

        # Pick 3 different groups
        try:
            g1, g2, g3 = random.sample(groups, 3)
        except ValueError:
            return None

        # Pick one student from each group
        if (
            not group_to_students[g1]
            or not group_to_students[g2]
            or not group_to_students[g3]
        ):
            return None

        s1 = random.choice(group_to_students[g1])
        s2 = random.choice(group_to_students[g2])
        s3 = random.choice(group_to_students[g3])

        # Check if cyclic swap is valid: s1->g2, s2->g3, s3->g1
        if g2 not in self.allowed[s1][c]:
            return None
        if g3 not in self.allowed[s2][c]:
            return None
        if g1 not in self.allowed[s3][c]:
            return None

        return ("chain", s1, s2, s3, c, g1, g2, g3)

    def try_chain_swap_move(self, move: Tuple) -> Optional[int]:
        """Try a chain swap move and return delta objective if valid."""
        _, s1, s2, s3, c, g1, g2, g3 = move

        # Check conflicts for s1 getting g2
        for g in self.assignment[s1]:
            if g != g1:
                if (
                    self.inst.groups_conflicts[g2][g]
                    or self.inst.groups_conflicts[g][g2]
                ):
                    return None

        # Check conflicts for s2 getting g3
        for g in self.assignment[s2]:
            if g != g2:
                if (
                    self.inst.groups_conflicts[g3][g]
                    or self.inst.groups_conflicts[g][g3]
                ):
                    return None

        # Check conflicts for s3 getting g1
        for g in self.assignment[s3]:
            if g != g3:
                if (
                    self.inst.groups_conflicts[g1][g]
                    or self.inst.groups_conflicts[g][g1]
                ):
                    return None

        # Compute delta
        old_contrib = (
            self._student_objective_contribution(s1)
            + self._student_objective_contribution(s2)
            + self._student_objective_contribution(s3)
        )

        # Temporarily apply chain swap
        idx1 = self.assignment[s1].index(g1)
        idx2 = self.assignment[s2].index(g2)
        idx3 = self.assignment[s3].index(g3)
        self.assignment[s1][idx1] = g2
        self.assignment[s2][idx2] = g3
        self.assignment[s3][idx3] = g1

        new_contrib = (
            self._student_objective_contribution(s1)
            + self._student_objective_contribution(s2)
            + self._student_objective_contribution(s3)
        )

        # Revert
        self.assignment[s1][idx1] = g1
        self.assignment[s2][idx2] = g2
        self.assignment[s3][idx3] = g3

        return new_contrib - old_contrib

    def apply_chain_swap_move(self, move: Tuple):
        """Apply a chain swap move."""
        _, s1, s2, s3, c, g1, g2, g3 = move
        idx1 = self.assignment[s1].index(g1)
        idx2 = self.assignment[s2].index(g2)
        idx3 = self.assignment[s3].index(g3)
        self.assignment[s1][idx1] = g2
        self.assignment[s2][idx2] = g3
        self.assignment[s3][idx3] = g1
        self.student_class_group[s1][c] = g2
        self.student_class_group[s2][c] = g3
        self.student_class_group[s3][c] = g1

    def try_swap_move(self, move: Tuple) -> Optional[int]:
        """Try a swap move and return delta objective if valid."""
        _, s1, s2, c, g1, g2 = move

        for g in self.assignment[s1]:
            if g != g1:
                if (
                    self.inst.groups_conflicts[g2][g]
                    or self.inst.groups_conflicts[g][g2]
                ):
                    return None

        for g in self.assignment[s2]:
            if g != g2:
                if (
                    self.inst.groups_conflicts[g1][g]
                    or self.inst.groups_conflicts[g][g1]
                ):
                    return None

        old_contrib = self._student_objective_contribution(
            s1
        ) + self._student_objective_contribution(s2)

        idx1 = self.assignment[s1].index(g1)
        idx2 = self.assignment[s2].index(g2)
        self.assignment[s1][idx1] = g2
        self.assignment[s2][idx2] = g1

        new_contrib = self._student_objective_contribution(
            s1
        ) + self._student_objective_contribution(s2)

        self.assignment[s1][idx1] = g1
        self.assignment[s2][idx2] = g2

        return new_contrib - old_contrib

    def apply_swap_move(self, move: Tuple):
        """Apply a swap move."""
        _, s1, s2, c, g1, g2 = move
        idx1 = self.assignment[s1].index(g1)
        idx2 = self.assignment[s2].index(g2)
        self.assignment[s1][idx1] = g2
        self.assignment[s2][idx2] = g1
        self.student_class_group[s1][c] = g2
        self.student_class_group[s2][c] = g1

    def compute_objective(self) -> Tuple[int, int, int]:
        """Compute full objective."""
        total_pref = 0
        total_break = 0
        total_obj = 0

        for s in range(self.n_students):
            pref_d, break_d = self._student_disappointment(s)
            total_pref += pref_d
            total_break += break_d

            w = self.inst.student_break_importance[s]
            numerator = w * break_d + (10 - w) * pref_d
            td = (numerator + 9) // 10
            total_obj += td * td

        return total_obj, total_break, total_pref

    def _student_disappointment(self, s: int) -> Tuple[int, int]:
        """Compute preference and break disappointment for student."""
        pref_d = 0
        for c in range(self.n_classes):
            allowed = self.allowed[s][c]
            if not allowed:
                continue
            best_pref = max(self.inst.student_prefers[s][g] for g in allowed)
            for g in self.assignment[s]:
                if self.group_class[g] == c:
                    pref_d += best_pref - self.inst.student_prefers[s][g]
                    break

        day_gaps = []
        for d in range(self.n_days):
            day_groups = [g for g in self.assignment[s] if self.group_day[g] == d]
            if not day_groups:
                day_gaps.append(0)
                continue
            starts = [self.group_start[g] for g in day_groups]
            ends = [self.group_start[g] + self.group_duration[g] for g in day_groups]
            total_duration = sum(self.group_duration[g] for g in day_groups)
            gap = max(ends) - min(starts) - total_duration
            day_gaps.append(gap)

        total_gap = max(0, sum(day_gaps))
        break_d = (total_gap + self.units - 1) // self.units

        return pref_d, break_d

    def _student_objective_contribution(self, s: int) -> int:
        """Compute objective contribution of single student."""
        pref_d, break_d = self._student_disappointment(s)
        w = self.inst.student_break_importance[s]
        numerator = w * break_d + (10 - w) * pref_d
        td = (numerator + 9) // 10
        return td * td

    def try_move(self, move: Tuple[int, int, int, int]) -> Optional[int]:
        """Try a move and return delta objective if valid."""
        s, c, old_g, new_g = move

        cap = self.inst.class_size[self.group_class[new_g]]
        if self.group_counts[new_g] >= cap:
            return None

        for g in self.assignment[s]:
            if g != old_g:
                if (
                    self.inst.groups_conflicts[new_g][g]
                    or self.inst.groups_conflicts[g][new_g]
                ):
                    return None

        old_contrib = self._student_objective_contribution(s)

        idx = self.assignment[s].index(old_g)
        self.assignment[s][idx] = new_g
        new_contrib = self._student_objective_contribution(s)

        self.assignment[s][idx] = old_g

        return new_contrib - old_contrib

    def apply_move(self, move: Tuple[int, int, int, int]):
        """Apply a move."""
        s, c, old_g, new_g = move
        idx = self.assignment[s].index(old_g)
        self.assignment[s][idx] = new_g
        self.student_class_group[s][c] = new_g
        self.group_counts[old_g] -= 1
        self.group_counts[new_g] += 1

    def get_assignment_1based(self) -> List[List[int]]:
        """Return assignment with 1-based group indices."""
        return [[g + 1 for g in groups] for groups in self.assignment]

    def restore_from(self, assignment: List[List[int]], class_group: List[List[int]]):
        """Restore state from saved assignment."""
        self.assignment = [list(groups) for groups in assignment]
        self.student_class_group = [[g for g in row] for row in class_group]
        # Rebuild group counts
        self.group_counts = [0] * self.n_groups
        for s in range(self.n_students):
            for g in self.assignment[s]:
                self.group_counts[g] += 1


def solve_cpsat(inst: Instance, time_limit: float) -> List[List[int]]:
    """Solve with CP-SAT and return assignment."""
    model = cp_model.CpModel()
    n_students, n_groups = inst.n_students, inst.n_groups
    n_classes = inst.n_classes

    group_class = [c - 1 for c in inst.group_class]
    groups_by_class = [[] for _ in range(n_classes)]
    for g, c in enumerate(group_class):
        groups_by_class[c].append(g)

    x = []
    for s in range(n_students):
        row = []
        for g in range(n_groups):
            var = model.NewBoolVar(f"x_{s}_{g}")
            if inst.student_prefers[s][g] == -1:
                model.Add(var == 0)
            row.append(var)
        x.append(row)

    for s in range(n_students):
        for c in range(n_classes):
            allowed = [
                g for g in groups_by_class[c] if inst.student_prefers[s][g] != -1
            ]
            if not allowed:
                for g in groups_by_class[c]:
                    model.Add(x[s][g] == 0)
                continue
            model.Add(sum(x[s][g] for g in allowed) == 1)

    for g in range(n_groups):
        cap = inst.class_size[group_class[g]]
        model.Add(sum(x[s][g] for s in range(n_students)) <= cap)

    for g1 in range(n_groups):
        for g2 in range(g1 + 1, n_groups):
            if inst.groups_conflicts[g1][g2] or inst.groups_conflicts[g2][g1]:
                for s in range(n_students):
                    model.Add(x[s][g1] + x[s][g2] <= 1)

    obj_terms = []
    for s in range(n_students):
        for g in range(n_groups):
            if inst.student_prefers[s][g] > 0:
                obj_terms.append(x[s][g] * inst.student_prefers[s][g])
    model.Maximize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible solution found")

    assignment = []
    for s in range(n_students):
        chosen = [g + 1 for g in range(n_groups) if solver.BooleanValue(x[s][g])]
        assignment.append(chosen)

    return assignment


def simulated_annealing(
    state: SolutionState, time_limit: float, verbose: bool = False
) -> int:
    """
    Enhanced simulated annealing with:
    - Adaptive reheat threshold (increases over time)
    - Varied reheat temperatures
    - Chain swap moves (3 students)
    - Periodic restarts from best solution
    """
    if not state.moveable_pairs:
        return state.compute_objective()[0]

    start_time = time.time()
    current_obj = state.compute_objective()[0]
    best_obj = current_obj
    best_assignment = [list(groups) for groups in state.assignment]
    best_class_group = [[g for g in row] for row in state.student_class_group]

    # SA parameters
    temp = 100.0
    temp_min = 0.01
    alpha = 0.9995

    # Adaptive parameters
    accept_count = 0
    total_count = 0
    adaptive_window = 1000

    # Reheating parameters - NOW ADAPTIVE
    no_improve_count = 0
    base_reheat_threshold = 30000
    reheat_count = 0
    last_improvement_time = start_time

    # Restart parameters
    restart_threshold = 300.0  # Restart from best every 5 minutes without improvement
    last_restart_time = start_time

    iterations = 0
    improvements = 0
    move_type = 0  # 0=single, 1=swap, 2=chain

    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time

        # Adaptive reheat threshold: increases logarithmically with time
        # Start at 30k, grow to ~200k over hours
        time_factor = 1.0 + math.log1p(elapsed / 60.0)  # log(1 + minutes)
        current_reheat_threshold = int(base_reheat_threshold * time_factor)

        # Periodic restart from best solution if stuck for too long
        if elapsed - (last_improvement_time - start_time) > restart_threshold:
            if elapsed - (last_restart_time - start_time) > restart_threshold:
                state.restore_from(best_assignment, best_class_group)
                current_obj = best_obj
                last_restart_time = time.time()
                temp = 50.0  # Moderate temperature after restart
                if verbose:
                    print(f"  [{elapsed:.1f}s] Restarting from best solution")

        # Round-robin neighborhood selection with chain swaps
        if move_type == 0:
            move = state.get_random_move()
            is_swap = False
            is_chain = False
        elif move_type == 1:
            move = state.get_swap_move()
            is_swap = True
            is_chain = False
        else:
            move = state.get_chain_swap_move()
            is_swap = False
            is_chain = True

        move_type = (move_type + 1) % 3

        if move is None:
            iterations += 1
            continue

        # Try move
        if is_chain:
            delta = state.try_chain_swap_move(move)
        elif is_swap:
            delta = state.try_swap_move(move)
        else:
            delta = state.try_move(move)

        if delta is None:
            iterations += 1
            continue

        total_count += 1

        # Metropolis criterion
        accept = False
        if delta < 0:
            accept = True
        elif temp > temp_min:
            prob = math.exp(-delta / temp)
            if random.random() < prob:
                accept = True

        if accept:
            if is_chain:
                state.apply_chain_swap_move(move)
            elif is_swap:
                state.apply_swap_move(move)
            else:
                state.apply_move(move)
            current_obj += delta
            accept_count += 1

            if current_obj < best_obj:
                best_obj = current_obj
                best_assignment = [list(groups) for groups in state.assignment]
                best_class_group = [
                    [g for g in row] for row in state.student_class_group
                ]
                improvements += 1
                no_improve_count = 0
                last_improvement_time = time.time()
                reheat_count = 0  # Reset reheat counter on improvement
                if verbose:
                    move_str = (
                        "chain" if is_chain else ("swap" if is_swap else "single")
                    )
                    print(f"  [{elapsed:.1f}s] New best: {best_obj} ({move_str})")
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        # Adaptive cooling adjustment
        if total_count % adaptive_window == 0 and total_count > 0:
            ratio = accept_count / adaptive_window
            if ratio < 0.1:
                alpha = 0.995
            elif ratio > 0.4:
                alpha = 0.9999
            else:
                alpha = 0.9995
            accept_count = 0

        # Reheating with varied temperatures
        if no_improve_count >= current_reheat_threshold:
            reheat_count += 1
            # Vary reheat temperature: cycle through different levels
            reheat_temps = [5.0, 10.0, 20.0, 50.0, 100.0]
            reheat_temp = reheat_temps[reheat_count % len(reheat_temps)]
            temp = max(temp, reheat_temp)
            no_improve_count = 0
            if verbose:
                print(
                    f"  [{elapsed:.1f}s] Reheating to temp={temp:.2f} (threshold={current_reheat_threshold})"
                )

        temp = max(temp_min, temp * alpha)
        iterations += 1

    # Restore best
    state.assignment = best_assignment
    state.student_class_group = best_class_group

    if verbose:
        print(f"SA completed: {iterations} iterations, {improvements} improvements")

    return best_obj


def format_assignment(assignment: Sequence[Sequence[int]]) -> str:
    parts = []
    for groups in assignment:
        inner = ",".join(str(g) for g in sorted(groups))
        parts.append("{" + inner + "}")
    return "[" + ",".join(parts) + "]"


def main():
    parser = argparse.ArgumentParser(description="Hybrid enrollment solver v2")
    parser.add_argument("instance", type=Path, help="Path to .dzn instance")
    parser.add_argument(
        "--time-limit", "-t", type=float, default=60.0, help="Total time limit"
    )
    parser.add_argument(
        "--cpsat-time",
        type=float,
        default=None,
        help="CP-SAT time (default: 20% of total)",
    )
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output file")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    inst = parse_dzn(args.instance)

    if args.verbose:
        print(
            f"Problem: {inst.n_students} students, {inst.n_classes} classes, {inst.n_groups} groups"
        )

    # Time allocation
    cpsat_time = args.cpsat_time or args.time_limit * 0.2
    sa_time = args.time_limit - cpsat_time

    # Phase 1: CP-SAT
    if args.verbose:
        print(f"Phase 1: CP-SAT ({cpsat_time:.1f}s)...")
    start = time.time()
    assignment = solve_cpsat(inst, cpsat_time)
    cpsat_elapsed = time.time() - start

    state = SolutionState(inst, assignment)
    obj, break_d, pref_d = state.compute_objective()
    if args.verbose:
        print(f"  CP-SAT objective: {obj} (time: {cpsat_elapsed:.1f}s)")

    # Phase 2: Simulated Annealing
    if args.verbose:
        print(f"Phase 2: Simulated Annealing ({sa_time:.1f}s)...")
    final_obj = simulated_annealing(state, sa_time, verbose=args.verbose)

    # Get final results
    obj, break_d, pref_d = state.compute_objective()
    assignment = state.get_assignment_1based()

    # Output
    sol_text = "\n".join(
        [
            f"assignment = {format_assignment(assignment)};",
            f"total_break_disappointment = {break_d};",
            f"total_preference_disappointment = {pref_d};",
            f"objective = {obj};",
        ]
    )

    print(sol_text)

    if args.output:
        args.output.write_text(sol_text + "\n")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
