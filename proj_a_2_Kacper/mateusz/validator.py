#!/usr/bin/env python3
"""
Validator for enrollment solutions.
Validates solution against constraints and recalculates objective.
Solution format is like competition.sol.
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple


def parse_dzn(path: Path) -> Dict:
    """Parse .dzn instance file."""
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
            row = values[idx:idx + n2]
            matrix.append(row)
            idx += n2
        return matrix

    n_students = parse_int("n_students")
    n_days = parse_int("n_days")
    n_classes = parse_int("n_classes")
    n_groups = parse_int("n_groups")
    n_time_units_in_hour = parse_int("n_time_units_in_hour")

    return {
        "n_students": n_students,
        "n_days": n_days,
        "n_classes": n_classes,
        "n_groups": n_groups,
        "n_time_units_in_hour": n_time_units_in_hour,
        "student_break_importance": parse_array("student_break_importance"),
        "class_duration": parse_array("class_duration"),
        "class_size": parse_array("class_size"),
        "group_class": parse_array("group_class"),
        "group_start": parse_array("group_start"),
        "group_day": parse_array("group_day"),
        "student_prefers": parse_array2d("student_prefers", n_students, n_groups, int),
        "groups_conflicts": parse_array2d("groups_conflicts", n_groups, n_groups, bool),
    }


def parse_solution(path: Path) -> Dict:
    """Parse solution file in competition.sol format."""
    text = path.read_text()

    # Parse assignment
    m = re.search(r"assignment\s*=\s*\[([^\]]+)\];", text, re.DOTALL)
    if not m:
        raise ValueError("Missing assignment in solution")

    raw = m.group(1)
    assignment = []
    # Match each {g1,g2,...} set
    for set_match in re.finditer(r"\{([^}]*)\}", raw):
        groups_str = set_match.group(1).strip()
        if groups_str:
            groups = [int(g.strip()) for g in groups_str.split(",")]
        else:
            groups = []
        assignment.append(groups)

    # Parse reported values
    def parse_val(name: str) -> int:
        m = re.search(rf"{name}\s*=\s*(\d+)\s*;", text)
        return int(m.group(1)) if m else None

    return {
        "assignment": assignment,
        "total_break_disappointment": parse_val("total_break_disappointment"),
        "total_preference_disappointment": parse_val("total_preference_disappointment"),
        "objective": parse_val("objective"),
    }


def validate_and_compute(inst: Dict, sol: Dict, verbose: bool = True) -> Tuple[bool, Dict]:
    """Validate solution and compute objective."""
    assignment = sol["assignment"]
    n_students = inst["n_students"]
    n_groups = inst["n_groups"]
    n_classes = inst["n_classes"]
    n_days = inst["n_days"]
    units = inst["n_time_units_in_hour"]
    day_span = units * 24

    # Convert to 0-based indexing
    group_class = [c - 1 for c in inst["group_class"]]
    group_day = [d - 1 for d in inst["group_day"]]
    group_start = inst["group_start"]
    group_duration = [inst["class_duration"][group_class[g]] for g in range(n_groups)]

    # Groups by class
    groups_by_class = [[] for _ in range(n_classes)]
    for g in range(n_groups):
        groups_by_class[group_class[g]].append(g)

    # Groups by day
    groups_by_day = [[] for _ in range(n_days)]
    for g in range(n_groups):
        groups_by_day[group_day[g]].append(g)

    errors = []

    # Check assignment length
    if len(assignment) != n_students:
        errors.append(f"Assignment has {len(assignment)} students, expected {n_students}")
        return False, {"errors": errors}

    # Validate each student
    group_counts = [0] * n_groups

    for s in range(n_students):
        groups = assignment[s]

        # Convert to 0-based
        groups_0based = [g - 1 for g in groups]

        # Check exactly one group per class
        class_assignments = {}
        for g in groups_0based:
            if g < 0 or g >= n_groups:
                errors.append(f"Student {s+1}: invalid group {g+1}")
                continue
            c = group_class[g]
            if c in class_assignments:
                errors.append(f"Student {s+1}: multiple groups for class {c+1} (groups {class_assignments[c]+1} and {g+1})")
            class_assignments[c] = g

        # Check all classes are covered (unless excluded)
        for c in range(n_classes):
            allowed = [g for g in groups_by_class[c] if inst["student_prefers"][s][g] != -1]
            if allowed:
                if c not in class_assignments:
                    errors.append(f"Student {s+1}: no group assigned for class {c+1}")
            else:
                # All groups excluded for this class
                if c in class_assignments:
                    errors.append(f"Student {s+1}: assigned to excluded class {c+1}")

        # Check exclusions
        for g in groups_0based:
            if g >= 0 and g < n_groups:
                if inst["student_prefers"][s][g] == -1:
                    errors.append(f"Student {s+1}: assigned to excluded group {g+1}")

        # Check conflicts
        for i, g1 in enumerate(groups_0based):
            for g2 in groups_0based[i+1:]:
                if g1 >= 0 and g2 >= 0 and g1 < n_groups and g2 < n_groups:
                    if inst["groups_conflicts"][g1][g2] or inst["groups_conflicts"][g2][g1]:
                        errors.append(f"Student {s+1}: conflicting groups {g1+1} and {g2+1}")

        # Count group usage
        for g in groups_0based:
            if 0 <= g < n_groups:
                group_counts[g] += 1

    # Check capacities
    for g in range(n_groups):
        cap = inst["class_size"][group_class[g]]
        if group_counts[g] > cap:
            errors.append(f"Group {g+1}: {group_counts[g]} students exceeds capacity {cap}")

    if errors:
        if verbose:
            print("VALIDATION ERRORS:")
            for e in errors[:20]:
                print(f"  - {e}")
            if len(errors) > 20:
                print(f"  ... and {len(errors) - 20} more errors")
        return False, {"errors": errors}

    # Compute objective
    total_pref_disappointment = 0
    total_break_disappointment = 0
    total_disappointments = []

    for s in range(n_students):
        groups_0based = [g - 1 for g in assignment[s]]

        # Preference disappointment
        pref_disappointment = 0
        for c in range(n_classes):
            allowed = [g for g in groups_by_class[c] if inst["student_prefers"][s][g] != -1]
            if not allowed:
                continue
            best_pref = max(inst["student_prefers"][s][g] for g in allowed)
            # Find assigned group for this class
            assigned_g = None
            for g in groups_0based:
                if 0 <= g < n_groups and group_class[g] == c:
                    assigned_g = g
                    break
            if assigned_g is not None:
                actual_pref = inst["student_prefers"][s][assigned_g]
                pref_disappointment += best_pref - actual_pref

        total_pref_disappointment += pref_disappointment

        # Break disappointment
        day_gaps = []
        for d in range(n_days):
            day_groups = [g for g in groups_0based if 0 <= g < n_groups and group_day[g] == d]
            if not day_groups:
                day_gaps.append(0)
                continue

            # Compute span - duration for this day
            starts = [group_start[g] for g in day_groups]
            ends = [group_start[g] + group_duration[g] for g in day_groups]
            total_duration = sum(group_duration[g] for g in day_groups)
            span = max(ends) - min(starts)
            gap = span - total_duration
            day_gaps.append(gap)

        total_gap = sum(day_gaps)
        total_gap = max(0, total_gap)  # Ensure non-negative
        break_norm = (total_gap + units - 1) // units  # Ceiling division
        total_break_disappointment += break_norm

        # Total disappointment per student (ceil_div per README)
        w = inst["student_break_importance"][s]
        numerator = w * break_norm + (10 - w) * pref_disappointment
        td = (numerator + 9) // 10  # Ceiling division: ceil_div(numerator, 10)
        total_disappointments.append(td)

    # Objective is sum of squared disappointments
    objective = sum(td * td for td in total_disappointments)

    result = {
        "valid": True,
        "total_break_disappointment": total_break_disappointment,
        "total_preference_disappointment": total_pref_disappointment,
        "objective": objective,
        "total_disappointments": total_disappointments,
    }

    if verbose:
        print("VALIDATION: PASSED")
        print(f"  Total break disappointment: {total_break_disappointment}")
        print(f"  Total preference disappointment: {total_pref_disappointment}")
        print(f"  Objective (sum of squared): {objective}")

        if sol["objective"] is not None:
            if sol["objective"] != objective:
                print(f"  WARNING: Reported objective {sol['objective']} != computed {objective}")
            else:
                print(f"  Objective matches reported value")

    return True, result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate enrollment solution")
    parser.add_argument("solution", type=Path, help="Path to .sol solution file")
    parser.add_argument("--instance", "-i", type=Path,
                        default=Path("data/competition.dzn"),
                        help="Path to .dzn instance file")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress output, just return exit code")
    args = parser.parse_args()

    if not args.instance.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent.parent
        alt_path = script_dir / "data" / "competition.dzn"
        if alt_path.exists():
            args.instance = alt_path
        else:
            print(f"Error: Instance file not found: {args.instance}")
            sys.exit(1)

    if not args.solution.exists():
        print(f"Error: Solution file not found: {args.solution}")
        sys.exit(1)

    inst = parse_dzn(args.instance)
    sol = parse_solution(args.solution)

    valid, result = validate_and_compute(inst, sol, verbose=not args.quiet)

    if not valid:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
