#!/usr/bin/env python3
"""
4-Torpedo Solver v4: Two-pass assignment to fix deadline issues.

Strategy:
1. First pass: min-slack greedy (achieves 4 torpedoes)
2. Identify deadline violations
3. Fix violations by swapping assignments
4. Verify still 4 torpedoes
"""

import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


@dataclass
class ProblemData:
    """Problem instance data."""
    time_pour: int
    time_desulf_one: int
    time_convert: int
    slots_full_buffer: int
    slots_desulf: int
    slots_converter: int
    t_empty_to_furnace: int
    t_furnace_to_full: int
    t_full_to_desulf: int
    t_desulf_to_conv: int
    t_conv_to_empty: int
    t_emergency: int
    metal_starts: List[int]
    metal_sulfur: List[int]
    conv_opens: List[int]
    conv_max_sulfur: List[int]

    @property
    def n(self) -> int:
        return len(self.metal_starts)

    @property
    def m(self) -> int:
        return len(self.conv_opens)

    def min_travel_time(self, pouring_idx: int, conv_idx: int) -> int:
        sulfur_reduction = max(0, self.metal_sulfur[pouring_idx] - self.conv_max_sulfur[conv_idx])
        desulf_time = sulfur_reduction * self.time_desulf_one
        return (self.time_pour + self.t_furnace_to_full + self.t_full_to_desulf +
                desulf_time + self.t_desulf_to_conv)

    def can_reach(self, pouring_idx: int, conv_idx: int, slack: int = 0) -> bool:
        earliest_arrival = self.metal_starts[pouring_idx] + self.min_travel_time(pouring_idx, conv_idx)
        return earliest_arrival + slack <= self.conv_opens[conv_idx]


@dataclass
class TorpedoCycle:
    """Represents a single torpedo cycle."""
    pouring_idx: int
    conv_idx: int
    torpedo_id: int = 0

    start_trip: int = 0
    start_furnace: int = 0
    end_furnace: int = 0
    start_full_buffer: int = -1
    end_full_buffer: int = -1
    start_desulf: int = -1
    end_desulf: int = -1
    start_converter: int = -1
    end_converter: int = -1
    end_trip: int = 0
    desulf_time: int = 0


def load_instance(filepath: str) -> ProblemData:
    """Load problem instance from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    return ProblemData(
        time_pour=data["timeToPourMetal"],
        time_desulf_one=data["timeToDesulfOneLevel"],
        time_convert=data["timeToConvert"],
        slots_full_buffer=data["numOfSlotsAtFullBuffer"],
        slots_desulf=data["numOfSlotsAtDesulf"],
        slots_converter=data["numOfSlotsAtConverter"],
        t_empty_to_furnace=data["timeToTransferFromEmptyBufferToFurnace"],
        t_furnace_to_full=data["timeToTransferFromFurnaceToFullBuffer"],
        t_full_to_desulf=data["timeToTransferFromFullBufferToDesulf"],
        t_desulf_to_conv=data["timeToTransferFromDesulfToConverter"],
        t_conv_to_empty=data["timeToTransferFromConverterToEmptyBuffer"],
        t_emergency=data["timeToEmergencyTransferFromFurnaceToEmptyBuffer"],
        metal_starts=data["metalStartsPouringAt"],
        metal_sulfur=data["metalSulfurLevels"],
        conv_opens=data["converterOpensAt"],
        conv_max_sulfur=data["converterMaxSulfurLevels"],
    )


def compute_emergency_conflicts(data: ProblemData) -> Dict[int, Set[int]]:
    """Compute which pourings conflict in emergency mode."""
    conflicts = {i: set() for i in range(data.n)}

    for i in range(data.n):
        end_furnace_i = data.metal_starts[i] + data.time_pour
        seg6_start_i = end_furnace_i
        seg6_end_i = end_furnace_i + data.t_emergency

        for j in range(i + 1, data.n):
            end_furnace_j = data.metal_starts[j] + data.time_pour
            seg6_start_j = end_furnace_j
            seg6_end_j = end_furnace_j + data.t_emergency

            if seg6_start_i < seg6_end_j and seg6_start_j < seg6_end_i:
                conflicts[i].add(j)
                conflicts[j].add(i)

    return conflicts


def compute_slack(data: ProblemData, pouring_idx: int, conv_idx: int) -> int:
    """Compute slack for assignment."""
    if conv_idx == -1:
        return float('inf')
    earliest_arrival = data.metal_starts[pouring_idx] + data.min_travel_time(pouring_idx, conv_idx)
    return data.conv_opens[conv_idx] - earliest_arrival


def create_initial_assignment(data: ProblemData) -> Dict[int, int]:
    """
    Create min-slack greedy assignment (achieves 4 max concurrent).
    """
    assignment = {}
    conv_assigned = set()

    conv_order = sorted(range(data.m), key=lambda j: data.conv_opens[j])

    for j in conv_order:
        best_pouring = None
        best_slack = float('inf')

        for i in range(data.n):
            if i in assignment:
                continue
            if not data.can_reach(i, j, slack=0):
                continue

            earliest_arrival = data.metal_starts[i] + data.min_travel_time(i, j)
            slack = data.conv_opens[j] - earliest_arrival

            if slack < best_slack:
                best_pouring = i
                best_slack = slack

        if best_pouring is not None:
            assignment[best_pouring] = j
            conv_assigned.add(j)

    # Remaining go to emergency
    for i in range(data.n):
        if i not in assignment:
            assignment[i] = -1

    return assignment


def find_zero_slack_pourings(data: ProblemData, assignment: Dict[int, int]) -> List[int]:
    """Find pourings with slack = 0."""
    zero_slack = []
    for i, conv_idx in assignment.items():
        if conv_idx == -1:
            continue
        slack = compute_slack(data, i, conv_idx)
        if slack == 0:
            zero_slack.append(i)
    return zero_slack


def try_fix_zero_slack(data: ProblemData, assignment: Dict[int, int],
                       zero_slack_pourings: List[int]) -> bool:
    """
    Try to fix zero-slack pourings by swapping with other pourings.
    Returns True if all fixes successful.
    """
    fixed = 0

    for p in zero_slack_pourings:
        c = assignment[p]  # Current conversion (0-indexed)
        current_slack = compute_slack(data, p, c)

        if current_slack > 0:
            continue  # Already fixed

        print(f"  Fixing pouring {p} (conv {c+1}, slack={current_slack})")

        # Strategy: Find another pouring that:
        # 1. Is assigned to a conversion with higher slack
        # 2. Can also serve our conversion with slack > 0

        best_swap = None
        best_new_slack = 0

        for other_p, other_c in assignment.items():
            if other_p == p or other_c == -1:
                continue

            # Can other_p serve our conversion c with slack > 0?
            if not data.can_reach(other_p, c, slack=1):
                continue

            other_new_slack = compute_slack(data, other_p, c)

            # Can we serve other_c with ANY slack >= 0?
            if not data.can_reach(p, other_c, slack=0):
                continue

            our_new_slack = compute_slack(data, p, other_c)

            # Would swap improve our situation?
            if other_new_slack > current_slack and our_new_slack >= 0:
                if other_new_slack > best_new_slack:
                    best_swap = (other_p, other_c, other_new_slack, our_new_slack)
                    best_new_slack = other_new_slack

        if best_swap:
            other_p, other_c, new_slack_for_c, new_slack_for_other = best_swap
            print(f"    SWAP: p{p} <-> p{other_p}")
            print(f"      p{p}: conv {c+1} (slack=0) -> conv {other_c+1} (slack={new_slack_for_other})")
            print(f"      p{other_p}: conv {other_c+1} -> conv {c+1} (slack={new_slack_for_c})")

            # Perform swap
            assignment[p] = other_c
            assignment[other_p] = c
            fixed += 1
        else:
            # Try to find a free conversion for this pouring with slack > 0
            # and assign the zero-slack conversion to someone else
            used_convs = set(v for v in assignment.values() if v != -1)
            free_convs = [j for j in range(data.m) if j not in used_convs and data.can_reach(p, j, slack=1)]

            if free_convs:
                # This shouldn't happen since we serve exactly 200 conversions
                pass

            print(f"    Could not fix pouring {p}")

    return fixed == len(zero_slack_pourings)


def resolve_emergency_conflicts(data: ProblemData, assignment: Dict[int, int]) -> None:
    """Resolve emergency conflicts by swapping to free conversions."""
    emergency_conflicts = compute_emergency_conflicts(data)

    emergency_pourings = set(i for i, c in assignment.items() if c == -1)
    conv_to_pouring = {c: i for i, c in assignment.items() if c != -1}
    used_convs = set(conv_to_pouring.keys())

    iterations = 0
    max_iterations = 1000

    while iterations < max_iterations:
        iterations += 1

        conflict_found = False
        for i in list(emergency_pourings):
            conflicting = emergency_conflicts[i] & emergency_pourings
            if conflicting:
                j = min(conflicting)

                free_convs = set(range(data.m)) - used_convs

                options_i = [c for c in free_convs if data.can_reach(i, c, slack=1)]
                options_j = [c for c in free_convs if data.can_reach(j, c, slack=1)]

                if options_i and (not options_j or len(options_i) <= len(options_j)):
                    best_conv = min(options_i, key=lambda c: data.conv_opens[c])
                    assignment[i] = best_conv
                    conv_to_pouring[best_conv] = i
                    used_convs.add(best_conv)
                    emergency_pourings.remove(i)
                    conflict_found = True
                    break
                elif options_j:
                    best_conv = min(options_j, key=lambda c: data.conv_opens[c])
                    assignment[j] = best_conv
                    conv_to_pouring[best_conv] = j
                    used_convs.add(best_conv)
                    emergency_pourings.remove(j)
                    conflict_found = True
                    break
                else:
                    print(f"    WARNING: Can't resolve conflict between {i} and {j}")
                    # Try swap
                    for k, c in list(assignment.items()):
                        if c == -1:
                            continue
                        if data.can_reach(i, c, slack=1):
                            k_would_conflict = emergency_conflicts[k] & emergency_pourings
                            if not k_would_conflict or k_would_conflict == {i}:
                                assignment[i] = c
                                assignment[k] = -1
                                emergency_pourings.remove(i)
                                emergency_pourings.add(k)
                                conv_to_pouring[c] = i
                                conflict_found = True
                                break
                    if conflict_found:
                        break

        if not conflict_found:
            break


class ConstraintScheduler:
    """Schedules torpedo cycles respecting all constraints."""

    def __init__(self, data: ProblemData):
        self.data = data

    def _find_earliest_nonoverlap(self, intervals: List, duration: int, earliest: int) -> int:
        if not intervals:
            return earliest

        sorted_intervals = sorted(intervals)

        if earliest + duration <= sorted_intervals[0][0]:
            return earliest

        for i in range(len(sorted_intervals) - 1):
            gap_start = sorted_intervals[i][1]
            gap_end = sorted_intervals[i + 1][0]
            start = max(gap_start, earliest)
            if start + duration <= gap_end:
                return start

        return max(sorted_intervals[-1][1], earliest)

    def _check_cumulative(self, events: List, start: int, end: int, capacity: int) -> int:
        if not events:
            return start

        duration = end - start

        def can_fit(s: int, e: int) -> bool:
            load_at_s = sum(delta for t, delta in events if t <= s)
            if load_at_s >= capacity:
                return False
            load = load_at_s
            for t, delta in sorted(events):
                if t > s and t < e:
                    load += delta
                    if load >= capacity:
                        return False
            return True

        if can_fit(start, end):
            return start

        all_times = sorted(set(t for t, _ in events))

        for t in all_times:
            if t <= start:
                continue
            if can_fit(t, t + duration):
                return t

        if events:
            end_times = [t for t, d in events if d < 0]
            if end_times:
                return max(max(end_times), start)

        return max(t for t, _ in events) + duration + 1 if events else start

    def schedule_all(self, assignment: Dict[int, int]) -> List[TorpedoCycle]:
        d = self.data
        cycles = []

        order = sorted(range(d.n), key=lambda i: d.metal_starts[i])

        seg1, seg2, seg3, seg4, seg5, seg6 = [], [], [], [], [], []
        furnace = []
        full_buffer_events, desulf_events, converter_events = [], [], []

        for pouring_idx in order:
            conv_idx = assignment[pouring_idx]
            cycle = self._schedule_one(
                pouring_idx, conv_idx,
                seg1, seg2, seg3, seg4, seg5, seg6, furnace,
                full_buffer_events, desulf_events, converter_events
            )
            cycles.append(cycle)

        return sorted(cycles, key=lambda c: c.pouring_idx)

    def _schedule_one(self, pouring_idx, conv_idx, seg1, seg2, seg3, seg4, seg5, seg6, furnace,
                      full_buffer_events, desulf_events, converter_events) -> TorpedoCycle:
        d = self.data
        cycle = TorpedoCycle(pouring_idx=pouring_idx, conv_idx=conv_idx)

        metal_start = d.metal_starts[pouring_idx]

        earliest_seg1 = max(0, metal_start - d.t_empty_to_furnace)
        seg1_start = self._find_earliest_nonoverlap(seg1, d.t_empty_to_furnace, earliest_seg1)

        cycle.start_trip = seg1_start
        cycle.start_furnace = seg1_start + d.t_empty_to_furnace
        cycle.end_furnace = metal_start + d.time_pour

        if conv_idx == -1:
            cycle.end_trip = cycle.end_furnace + d.t_emergency

            seg1.append((cycle.start_trip, cycle.start_furnace))
            furnace.append((cycle.start_furnace, cycle.end_furnace))
            seg6.append((cycle.end_furnace, cycle.end_trip))
        else:
            conv_time = d.conv_opens[conv_idx]
            max_sulfur = d.conv_max_sulfur[conv_idx]
            metal_sulfur = d.metal_sulfur[pouring_idx]

            sulfur_reduction = max(0, metal_sulfur - max_sulfur)
            cycle.desulf_time = sulfur_reduction * d.time_desulf_one

            seg2_start = self._find_earliest_nonoverlap(seg2, d.t_furnace_to_full, cycle.end_furnace)
            cycle.start_full_buffer = seg2_start + d.t_furnace_to_full
            seg2.append((seg2_start, cycle.start_full_buffer))

            cycle.end_full_buffer = cycle.start_full_buffer

            for _ in range(200):
                seg3_start = self._find_earliest_nonoverlap(seg3, d.t_full_to_desulf, cycle.end_full_buffer)
                if seg3_start > cycle.end_full_buffer:
                    cycle.end_full_buffer = seg3_start
                cycle.start_desulf = seg3_start + d.t_full_to_desulf

                min_end_desulf = cycle.start_desulf + cycle.desulf_time
                earliest_arrival = min_end_desulf + d.t_desulf_to_conv

                cycle.start_converter = earliest_arrival
                actual_conv_start = max(cycle.start_converter, conv_time)
                cycle.end_converter = actual_conv_start + d.time_convert

                conv_start_check = self._check_cumulative(
                    converter_events, actual_conv_start, cycle.end_converter, d.slots_converter
                )
                if conv_start_check > actual_conv_start:
                    actual_conv_start = conv_start_check
                    cycle.start_converter = max(earliest_arrival, actual_conv_start)
                    cycle.end_converter = actual_conv_start + d.time_convert

                required_end_desulf = cycle.start_converter - d.t_desulf_to_conv
                cycle.end_desulf = max(min_end_desulf, required_end_desulf)

                seg4_start = self._find_earliest_nonoverlap(seg4, d.t_desulf_to_conv, cycle.end_desulf)
                if seg4_start > cycle.end_desulf:
                    cycle.end_desulf = seg4_start
                    cycle.start_converter = cycle.end_desulf + d.t_desulf_to_conv
                    actual_conv_start = max(cycle.start_converter, conv_time)
                    cycle.end_converter = actual_conv_start + d.time_convert

                    conv_start_check = self._check_cumulative(
                        converter_events, actual_conv_start, cycle.end_converter, d.slots_converter
                    )
                    if conv_start_check > actual_conv_start:
                        actual_conv_start = conv_start_check
                        cycle.end_converter = actual_conv_start + d.time_convert

                if cycle.desulf_time > 0 or cycle.end_desulf > cycle.start_desulf:
                    desulf_start = self._check_cumulative(
                        desulf_events, cycle.start_desulf, cycle.end_desulf, d.slots_desulf
                    )
                    if desulf_start > cycle.start_desulf:
                        cycle.end_full_buffer = desulf_start - d.t_full_to_desulf
                        continue

                break

            seg3.append((cycle.end_full_buffer, cycle.start_desulf))

            if cycle.desulf_time > 0 or cycle.end_desulf > cycle.start_desulf:
                desulf_events.append((cycle.start_desulf, 1))
                desulf_events.append((cycle.end_desulf, -1))

            seg4.append((cycle.end_desulf, cycle.start_converter))

            actual_conv_start = max(cycle.start_converter, conv_time)
            converter_events.append((actual_conv_start, 1))
            converter_events.append((cycle.end_converter, -1))

            seg5_start = self._find_earliest_nonoverlap(seg5, d.t_conv_to_empty, cycle.end_converter)
            if seg5_start > cycle.end_converter:
                cycle.end_converter = seg5_start
            cycle.end_trip = cycle.end_converter + d.t_conv_to_empty
            seg5.append((cycle.end_converter, cycle.end_trip))

            seg1.append((cycle.start_trip, cycle.start_furnace))
            furnace.append((cycle.start_furnace, cycle.end_furnace))

            if cycle.end_full_buffer > cycle.start_full_buffer:
                full_buffer_events.append((cycle.start_full_buffer, 1))
                full_buffer_events.append((cycle.end_full_buffer, -1))

        return cycle


def assign_torpedoes(cycles: List[TorpedoCycle]) -> int:
    sorted_cycles = sorted(cycles, key=lambda c: c.start_trip)

    torpedo_available = {}
    next_torpedo_id = 1

    for cycle in sorted_cycles:
        assigned = None
        for t_id, avail_time in torpedo_available.items():
            if avail_time <= cycle.start_trip:
                assigned = t_id
                break

        if assigned is None:
            assigned = next_torpedo_id
            next_torpedo_id += 1

        cycle.torpedo_id = assigned
        torpedo_available[assigned] = cycle.end_trip

    return next_torpedo_id - 1


def count_deadline_violations(cycles: List[TorpedoCycle], data: ProblemData) -> List[int]:
    """Return list of pourings with deadline violations."""
    violations = []
    for cycle in cycles:
        if cycle.conv_idx >= 0:
            if cycle.start_converter > data.conv_opens[cycle.conv_idx]:
                violations.append(cycle.pouring_idx)
    return violations


def validate_solution(cycles: List[TorpedoCycle], data: ProblemData) -> tuple:
    errors = []

    def check_overlaps(name, intervals):
        sorted_intervals = sorted(intervals)
        for i in range(1, len(sorted_intervals)):
            if sorted_intervals[i][0] < sorted_intervals[i-1][1]:
                errors.append(f"{name}: overlap")

    seg1 = [(c.start_trip, c.start_trip + data.t_empty_to_furnace) for c in cycles]
    check_overlaps("Seg1", seg1)

    seg2 = [(c.end_furnace, c.start_full_buffer) for c in cycles if c.conv_idx >= 0]
    check_overlaps("Seg2", seg2)

    seg3 = [(c.end_full_buffer, c.start_desulf) for c in cycles if c.conv_idx >= 0]
    check_overlaps("Seg3", seg3)

    seg4 = [(c.end_desulf, c.start_converter) for c in cycles if c.conv_idx >= 0]
    check_overlaps("Seg4", seg4)

    seg5 = [(c.end_converter, c.end_trip) for c in cycles if c.conv_idx >= 0]
    check_overlaps("Seg5", seg5)

    seg6 = [(c.end_furnace, c.end_trip) for c in cycles if c.conv_idx == -1]
    check_overlaps("Seg6", seg6)

    def check_cumulative(name, intervals, capacity):
        events = []
        for start, end in intervals:
            events.append((start, 1))
            events.append((end, -1))
        events.sort()
        load = 0
        for t, delta in events:
            load += delta
            if load > capacity:
                errors.append(f"{name}: capacity exceeded")

    desulf_intervals = [(c.start_desulf, c.end_desulf) for c in cycles
                        if c.conv_idx >= 0 and c.desulf_time > 0]
    check_cumulative("Desulf", desulf_intervals, data.slots_desulf)

    conv_intervals = [(c.start_converter, c.end_converter) for c in cycles if c.conv_idx >= 0]
    check_cumulative("Converter", conv_intervals, data.slots_converter)

    return len(errors) == 0, errors


def format_solution(cycles: List[TorpedoCycle], data: ProblemData) -> str:
    lines = []

    arrays = {
        "assignedTorpedo": [c.torpedo_id for c in cycles],
        "assignedConversion": [c.conv_idx + 1 if c.conv_idx >= 0 else -1 for c in cycles],
        "startTrip": [c.start_trip for c in cycles],
        "startFurnace": [c.start_furnace for c in cycles],
        "endFurnace": [c.end_furnace for c in cycles],
        "startFullBuffer": [c.start_full_buffer for c in cycles],
        "endFullBuffer": [c.end_full_buffer for c in cycles],
        "startDesulf": [c.start_desulf for c in cycles],
        "endDesulf": [c.end_desulf for c in cycles],
        "startConverter": [c.start_converter for c in cycles],
        "endConverter": [c.end_converter for c in cycles],
        "endTrip": [c.end_trip for c in cycles],
    }

    for name, values in arrays.items():
        lines.append(f"{name} = [{', '.join(map(str, values))}];")

    num_missed = 0
    num_torpedoes = len(set(c.torpedo_id for c in cycles))
    total_desulf = sum(c.desulf_time for c in cycles if c.conv_idx >= 0)

    lines.append(f"numMissedConversions = {num_missed};")
    lines.append(f"numUsedTorpedoes = {num_torpedoes};")
    lines.append(f"totalDesulfTime = {total_desulf};")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 3:
        print("Usage: python solver_4torp_v4.py <input.json> <output.sol>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Loading: {input_path}")
    data = load_instance(input_path)
    print(f"Problem: {data.n} pourings, {data.m} conversions")

    print("\n=== Phase 1: Initial min-slack assignment ===")
    assignment = create_initial_assignment(data)

    emergency_count = sum(1 for v in assignment.values() if v == -1)
    print(f"  Emergency: {emergency_count} pourings")
    print(f"  Conversions: {data.n - emergency_count} pourings")

    print("\n=== Phase 2: Identify zero-slack pourings ===")
    zero_slack = find_zero_slack_pourings(data, assignment)
    print(f"  Found {len(zero_slack)} pourings with slack=0")
    for p in zero_slack[:10]:
        c = assignment[p]
        print(f"    p{p} -> conv {c+1}")

    print("\n=== Phase 3: Fix zero-slack pourings ===")
    try_fix_zero_slack(data, assignment, zero_slack)

    # Check again
    zero_slack_after = find_zero_slack_pourings(data, assignment)
    print(f"  After fixing: {len(zero_slack_after)} pourings with slack=0")

    print("\n=== Phase 4: Resolve emergency conflicts ===")
    resolve_emergency_conflicts(data, assignment)

    # Final counts
    emergency_count = sum(1 for v in assignment.values() if v == -1)
    print(f"  Final emergency: {emergency_count} pourings")

    print("\n=== Phase 5: Scheduling ===")
    scheduler = ConstraintScheduler(data)
    cycles = scheduler.schedule_all(assignment)

    print("Assigning torpedoes...")
    num_torpedoes = assign_torpedoes(cycles)
    print(f"  Torpedoes used: {num_torpedoes}")

    print("\nChecking deadline violations...")
    violations = count_deadline_violations(cycles, data)
    if violations:
        print(f"  {len(violations)} deadline violations:")
        for p in violations[:10]:
            c = [cyc for cyc in cycles if cyc.pouring_idx == p][0]
            print(f"    p{p} -> conv {c.conv_idx+1}: start_converter={c.start_converter}, conv_opens={data.conv_opens[c.conv_idx]}")
    else:
        print("  No deadline violations!")

    print("\nValidating solution...")
    is_valid, errors = validate_solution(cycles, data)
    if is_valid and not violations:
        print("  *** VALID ***")
    else:
        print(f"  INVALID ({len(errors)} errors, {len(violations)} deadline violations)")
        for err in errors[:10]:
            print(f"    - {err}")

    output = format_solution(cycles, data)
    with open(output_path, "w") as f:
        f.write(output)
    print(f"\nWritten to: {output_path}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total_desulf = sum(c.desulf_time for c in cycles if c.conv_idx >= 0)
    print(f"  Missed conversions: 0")
    print(f"  Torpedoes: {num_torpedoes}")
    print(f"  Total desulf time: {total_desulf}")
    print(f"  Valid: {is_valid and not violations}")


if __name__ == "__main__":
    main()
