#!/usr/bin/env python3
"""
Enrollment solver v3 - Minimal changes from working baseline.
Key insight: CP-SAT's default search is already very good.
Only add: warm-start hints, forced assignment detection, better parameters.
DO NOT override the search strategy.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

    @property
    def day_span(self) -> int:
        return self.n_time_units_in_hour * 24

    @property
    def horizon(self) -> int:
        return self.day_span * self.n_days


def _parse_int(text: str, name: str) -> int:
    m = re.search(rf"{name}\s*=\s*([-\d]+)\s*;", text)
    if not m:
        raise ValueError(f"Missing integer parameter {name}")
    return int(m.group(1))


def _parse_array(text: str, name: str, convert=str) -> List:
    m = re.search(rf"{name}\s*=\s*\[([^\]]+)\]\s*;", text, re.DOTALL)
    if not m:
        raise ValueError(f"Missing array parameter {name}")
    raw = m.group(1)
    values: List = []
    for tok in re.split(r",", raw):
        tok = tok.strip()
        if tok == "":
            continue
        values.append(convert(tok))
    return values


def _parse_array2d(text: str, name: str, n1: int, n2: int, convert=str) -> List[List]:
    m = re.search(
        rf"{name}\s*=\s*array2d\([^\[]*,\s*\[([^\]]+)\]\s*\);", text, re.DOTALL
    )
    if not m:
        raise ValueError(f"Missing array2d parameter {name}")
    raw = m.group(1)
    values: List = []
    for tok in re.split(r",", raw):
        tok = tok.strip()
        if tok == "":
            continue
        values.append(convert(tok))
    expected = n1 * n2
    if len(values) != expected:
        raise ValueError(
            f"array2d {name} length {len(values)} does not match expected {expected}"
        )
    matrix = []
    idx = 0
    for _ in range(n1):
        row = values[idx : idx + n2]
        matrix.append(row)
        idx += n2
    return matrix


def parse_dzn(path: Path) -> Instance:
    text = path.read_text()

    def conv_bool(tok: str) -> bool:
        t = tok.lower()
        if t == "true":
            return True
        if t == "false":
            return False
        raise ValueError(f"Invalid boolean token {tok}")

    n_students = _parse_int(text, "n_students")
    n_days = _parse_int(text, "n_days")
    n_classes = _parse_int(text, "n_classes")
    n_groups = _parse_int(text, "n_groups")
    n_time_units_in_hour = _parse_int(text, "n_time_units_in_hour")

    student_break_importance = [
        int(v) for v in _parse_array(text, "student_break_importance")
    ]
    class_duration = [int(v) for v in _parse_array(text, "class_duration")]
    class_size = [int(v) for v in _parse_array(text, "class_size")]
    group_class = [int(v) for v in _parse_array(text, "group_class")]
    group_start = [int(v) for v in _parse_array(text, "group_start")]
    group_day = [int(v) for v in _parse_array(text, "group_day")]

    student_prefers = _parse_array2d(
        text, "student_prefers", n_students, n_groups, convert=int
    )
    groups_conflicts = _parse_array2d(
        text, "groups_conflicts", n_groups, n_groups, convert=conv_bool
    )

    return Instance(
        n_students=n_students,
        n_days=n_days,
        n_classes=n_classes,
        n_groups=n_groups,
        n_time_units_in_hour=n_time_units_in_hour,
        student_break_importance=student_break_importance,
        student_prefers=student_prefers,
        class_duration=class_duration,
        class_size=class_size,
        group_class=group_class,
        group_start=group_start,
        group_day=group_day,
        groups_conflicts=groups_conflicts,
    )


def parse_solution(path: Path, n_students: int) -> Optional[List[List[int]]]:
    """Parse an existing solution file for warm-starting."""
    text = path.read_text()
    m = re.search(r"assignment\s*=\s*\[([^\]]+)\];", text, re.DOTALL)
    if not m:
        return None
    raw = m.group(1)
    assignment = []
    for set_match in re.finditer(r"\{([^}]*)\}", raw):
        inner = set_match.group(1).strip()
        if inner:
            groups = [int(g.strip()) for g in inner.split(",")]
        else:
            groups = []
        assignment.append(groups)
    if len(assignment) != n_students:
        return None
    return assignment


def build_model(
    inst: Instance, warm_start: Optional[List[List[int]]] = None
) -> Tuple[cp_model.CpModel, Dict]:
    model = cp_model.CpModel()
    var_ub = lambda v: v.Proto().domain[-1]
    n_students, n_groups = inst.n_students, inst.n_groups
    n_classes, n_days = inst.n_classes, inst.n_days
    units = inst.n_time_units_in_hour
    horizon = inst.horizon
    big_m = horizon + max(inst.class_duration) + 10

    # Precompute per-group values (0-based indexing in Python, MiniZinc is 1-based).
    group_class = [c - 1 for c in inst.group_class]
    group_day = [d - 1 for d in inst.group_day]
    group_start = inst.group_start
    group_duration = [inst.class_duration[group_class[g]] for g in range(n_groups)]
    day_span = inst.day_span
    group_start_global = [
        group_start[g] + group_day[g] * day_span for g in range(n_groups)
    ]
    group_end_global = [
        group_start_global[g] + group_duration[g] for g in range(n_groups)
    ]

    groups_by_day: List[List[int]] = [[] for _ in range(n_days)]
    for g in range(n_groups):
        groups_by_day[group_day[g]].append(g)

    groups_by_class: List[List[int]] = [[] for _ in range(n_classes)]
    for g, c in enumerate(group_class):
        groups_by_class[c].append(g)

    # Decision variables: x[s][g] = 1 if student s attends group g.
    x: List[List[cp_model.IntVar]] = []
    for s in range(n_students):
        row = []
        for g in range(n_groups):
            var = model.NewBoolVar(f"x_{s}_{g}")
            if inst.student_prefers[s][g] == -1:
                model.Add(var == 0)
            row.append(var)
        x.append(row)

    # Warm-start hints from existing solution (SAFE - just hints, doesn't constrain)
    if warm_start:
        for s in range(n_students):
            for g in range(n_groups):
                # Solution uses 1-based indexing
                if (g + 1) in warm_start[s]:
                    model.AddHint(x[s][g], 1)
                else:
                    model.AddHint(x[s][g], 0)

    # Each student attends at most one group per class (unless all groups excluded, then zero).
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
            # Disallow excluded groups explicitly.
            for g in groups_by_class[c]:
                if inst.student_prefers[s][g] == -1:
                    model.Add(x[s][g] == 0)

    # Group capacity.
    for g in range(n_groups):
        cap = inst.class_size[group_class[g]]
        model.Add(sum(x[s][g] for s in range(n_students)) <= cap)

    # Schedule conflicts (precomputed matrix).
    for g1 in range(n_groups):
        for g2 in range(g1 + 1, n_groups):
            if inst.groups_conflicts[g1][g2] or inst.groups_conflicts[g2][g1]:
                for s in range(n_students):
                    model.Add(x[s][g1] + x[s][g2] <= 1)

    # Preference disappointment per student.
    pref_disappointment_per_student: List[cp_model.IntVar] = []
    for s in range(n_students):
        per_class_diffs: List[cp_model.IntVar] = []
        for c in range(n_classes):
            allowed = [
                g for g in groups_by_class[c] if inst.student_prefers[s][g] != -1
            ]
            if not allowed:
                diff = model.NewIntVar(0, 0, f"pref_diff_{s}_{c}")
                per_class_diffs.append(diff)
                continue
            best_pref = max(inst.student_prefers[s][g] for g in allowed)
            # Actual preference of selected group in this class.
            min_pref = min(inst.student_prefers[s][g] for g in allowed)
            max_pref = best_pref
            actual_pref = model.NewIntVar(min_pref, max_pref, f"actual_pref_{s}_{c}")
            model.Add(
                actual_pref
                == sum(x[s][g] * inst.student_prefers[s][g] for g in allowed)
            )
            diff = model.NewIntVar(0, best_pref - min_pref, f"pref_diff_{s}_{c}")
            model.Add(actual_pref + diff == best_pref)
            per_class_diffs.append(diff)
        total_pref_ub = sum(var_ub(d) for d in per_class_diffs)
        total_pref = model.NewIntVar(0, total_pref_ub, f"pref_total_{s}")
        model.Add(total_pref == sum(per_class_diffs))
        pref_disappointment_per_student.append(total_pref)

    # Break disappointment per student.
    break_disappointment_per_student: List[cp_model.IntVar] = []
    for s in range(n_students):
        day_breaks: List[cp_model.IntVar] = []
        for d in range(n_days):
            groups_d = groups_by_day[d]
            if not groups_d:
                # No groups that day at all.
                day_break = model.NewIntVar(0, 0, f"break_day_{s}_{d}")
                day_breaks.append(day_break)
                continue
            sum_day = model.NewIntVar(0, len(groups_d), f"sum_day_{s}_{d}")
            model.Add(sum_day == sum(x[s][g] for g in groups_d))
            day_has = model.NewBoolVar(f"day_has_{s}_{d}")
            model.Add(sum_day >= 1).OnlyEnforceIf(day_has)
            model.Add(sum_day == 0).OnlyEnforceIf(day_has.Not())

            start_choices: List[cp_model.IntVar] = []
            end_choices: List[cp_model.IntVar] = []
            for g in groups_d:
                sc = model.NewIntVar(0, horizon + big_m, f"start_choice_{s}_{g}")
                ec = model.NewIntVar(-big_m, horizon + big_m, f"end_choice_{s}_{g}")
                model.Add(sc == group_start_global[g] + (1 - x[s][g]) * big_m)
                model.Add(ec == group_end_global[g] - (1 - x[s][g]) * big_m)
                start_choices.append(sc)
                end_choices.append(ec)

            start_min = model.NewIntVar(0, horizon + big_m, f"start_min_{s}_{d}")
            end_max = model.NewIntVar(-big_m, horizon + big_m, f"end_max_{s}_{d}")
            model.AddMinEquality(start_min, start_choices)
            model.AddMaxEquality(end_max, end_choices)

            duration_sum = model.NewIntVar(
                0, sum(group_duration[g] for g in groups_d), f"dur_sum_{s}_{d}"
            )
            model.Add(
                duration_sum == sum(x[s][g] * group_duration[g] for g in groups_d)
            )

            span_gap = model.NewIntVar(-big_m, horizon + big_m, f"span_gap_{s}_{d}")
            model.Add(span_gap == end_max - start_min - duration_sum).OnlyEnforceIf(
                day_has
            )
            model.Add(span_gap == 0).OnlyEnforceIf(day_has.Not())
            day_breaks.append(span_gap)

        # Sum all daily gaps first, THEN apply max(0, ...) as per README:
        total_break_sum = model.NewIntVar(
            -big_m * n_days, big_m * n_days, f"break_sum_{s}"
        )
        model.Add(total_break_sum == sum(day_breaks))
        zero_const = model.NewIntVar(0, 0, f"zero_{s}")
        total_break_raw = model.NewIntVar(0, big_m * n_days, f"break_raw_{s}")
        model.AddMaxEquality(total_break_raw, [total_break_sum, zero_const])
        # ceil_div(total_break_raw, units)
        max_break_norm = math.ceil(big_m * n_days / units)
        break_norm = model.NewIntVar(0, max_break_norm, f"break_norm_{s}")
        model.Add(total_break_raw <= break_norm * units)
        model.Add(total_break_raw >= break_norm * units - (units - 1))
        break_disappointment_per_student.append(break_norm)

    # Total disappointment per student.
    total_disappointments: List[cp_model.IntVar] = []
    total_disappointments_sq: List[cp_model.IntVar] = []
    for s in range(n_students):
        w = inst.student_break_importance[s]
        numerator_max = 10 * max(
            var_ub(pref_disappointment_per_student[s]),
            var_ub(break_disappointment_per_student[s]),
        )
        numerator = model.NewIntVar(0, numerator_max, f"numerator_{s}")
        model.Add(
            numerator
            == w * break_disappointment_per_student[s]
            + (10 - w) * pref_disappointment_per_student[s]
        )
        td = model.NewIntVar(0, numerator_max, f"total_d_{s}")
        model.Add(numerator <= td * 10)
        model.Add(numerator >= td * 10 - 9)
        total_disappointments.append(td)

        # Square via table mapping.
        max_td = var_ub(td)
        td_sq = model.NewIntVar(0, max_td * max_td, f"total_d_sq_{s}")
        table = [(k, k * k) for k in range(max_td + 1)]
        model.AddAllowedAssignments([td, td_sq], table)
        total_disappointments_sq.append(td_sq)

    total_break_disappointment = model.NewIntVar(
        0, sum(var_ub(v) for v in break_disappointment_per_student), "total_break"
    )
    model.Add(total_break_disappointment == sum(break_disappointment_per_student))
    total_pref_disappointment = model.NewIntVar(
        0, sum(var_ub(v) for v in pref_disappointment_per_student), "total_pref"
    )
    model.Add(total_pref_disappointment == sum(pref_disappointment_per_student))

    objective = model.NewIntVar(
        0, sum(var_ub(v) for v in total_disappointments_sq), "objective"
    )
    model.Add(objective == sum(total_disappointments_sq))
    model.Minimize(objective)

    # NO custom search strategy - let CP-SAT use its excellent defaults!

    aux = dict(
        x=x,
        objective=objective,
        total_break=total_break_disappointment,
        total_pref=total_pref_disappointment,
        total_d=total_disappointments,
    )
    return model, aux


def solve_instance(
    inst: Instance,
    time_limit: float,
    workers: int,
    warm_start: Optional[List[List[int]]] = None,
) -> Dict:
    model, aux = build_model(inst, warm_start)
    solver = cp_model.CpSolver()

    if time_limit:
        solver.parameters.max_time_in_seconds = time_limit
    if workers:
        solver.parameters.num_search_workers = workers

    # Keep original parameter that was working well
    solver.parameters.linearization_level = 0

    # Log progress
    solver.parameters.log_search_progress = True

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(
            f"No feasible solution found (status={solver.StatusName(status)})"
        )

    # Extract assignment.
    assignment: List[List[int]] = []
    x = aux["x"]
    for s in range(inst.n_students):
        chosen = [g + 1 for g in range(inst.n_groups) if solver.BooleanValue(x[s][g])]
        assignment.append(chosen)

    result = {
        "status": solver.StatusName(status),
        "objective": int(round(solver.Value(aux["objective"]))),
        "total_break_disappointment": int(round(solver.Value(aux["total_break"]))),
        "total_preference_disappointment": int(round(solver.Value(aux["total_pref"]))),
        "assignment": assignment,
        "solver_time": solver.WallTime(),
    }
    return result


def format_assignment(assignment: Sequence[Sequence[int]]) -> str:
    parts = []
    for groups in assignment:
        inner = ",".join(str(g) for g in sorted(groups))
        parts.append("{" + inner + "}")
    return "[" + ",".join(parts) + "]"


def format_solution(result: Dict) -> str:
    return "\n".join(
        [
            f"assignment = {format_assignment(result['assignment'])};",
            f"total_break_disappointment = {result['total_break_disappointment']};",
            f"total_preference_disappointment = {result['total_preference_disappointment']};",
            f"objective = {result['objective']};",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrollment solver v3 - minimal changes, warm-start only"
    )
    parser.add_argument("instance", type=Path, help="Path to .dzn instance")
    parser.add_argument(
        "--time-limit", type=float, default=300.0, help="Solver time limit in seconds"
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="Number of parallel workers (0=auto)"
    )
    parser.add_argument(
        "--warm-start", type=Path, default=None, help="Path to existing solution"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Path to save solution"
    )
    args = parser.parse_args()

    inst = parse_dzn(args.instance)

    # Try to load warm-start
    warm_start = None
    if args.warm_start and args.warm_start.exists():
        print(f"Loading warm-start from {args.warm_start}")
        warm_start = parse_solution(args.warm_start, inst.n_students)
        if warm_start:
            print(f"  Loaded {len(warm_start)} student assignments as hints")
    elif args.warm_start is None:
        # Auto-detect competition.sol
        default_sol = Path("competition.sol")
        if default_sol.exists() and "competition" in args.instance.stem:
            print(f"Auto-loading warm-start from {default_sol}")
            warm_start = parse_solution(default_sol, inst.n_students)

    result = solve_instance(
        inst, time_limit=args.time_limit, workers=args.workers, warm_start=warm_start
    )

    sol_text = format_solution(result)
    print(sol_text)
    print(f"status = {result['status']};")
    print(f"solver_time = {result['solver_time']:.3f}s;")

    target = args.output
    if target is None and "competition" in args.instance.stem:
        target = Path("competition.sol")
    if target is not None:
        target.write_text(sol_text + "\n")
        print(f"Saved solution to {target}")


if __name__ == "__main__":
    main()
