#!/usr/bin/env python3
"""
Stage 1 Solver: Minimize Missed Conversions for Torpedo Scheduling Problem

This solver focuses ONLY on minimizing the number of missed conversions.
It uses Simulated Annealing with proper constraint-aware scheduling.

Usage:
    python solver_stage1_missed.py [input.json] [output.sol] [time_limit_seconds]
"""

import json
import math
import random
import sys
import time
from bisect import insort
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
        """Number of pourings (blast furnace events)."""
        return len(self.metal_starts)

    @property
    def m(self) -> int:
        """Number of conversions."""
        return len(self.conv_opens)

    def min_travel_time(self, pouring_idx: int, conv_idx: int) -> int:
        """Minimum time from pouring start to converter arrival."""
        sulfur_reduction = max(
            0, self.metal_sulfur[pouring_idx] - self.conv_max_sulfur[conv_idx]
        )
        desulf_time = sulfur_reduction * self.time_desulf_one
        return (
            self.time_pour
            + self.t_furnace_to_full
            + self.t_full_to_desulf
            + desulf_time
            + self.t_desulf_to_conv
        )

    def can_reach(self, pouring_idx: int, conv_idx: int, slack: int = 0) -> bool:
        """Check if pouring can reach conversion on time with optional slack margin."""
        earliest_arrival = self.metal_starts[pouring_idx] + self.min_travel_time(
            pouring_idx, conv_idx
        )
        return earliest_arrival + slack <= self.conv_opens[conv_idx]


@dataclass
class TorpedoCycle:
    """Represents a single torpedo cycle from empty buffer and back."""

    pouring_idx: int  # Which pouring this cycle handles
    conv_idx: int  # Which conversion (-1 for emergency)
    torpedo_id: int  # Which torpedo handles this cycle

    # Timing variables (computed based on assignment)
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


class IntervalTracker:
    """Tracks non-overlapping intervals for a single-capacity resource (railway segment)."""

    def __init__(self):
        self.intervals: List[
            Tuple[int, int]
        ] = []  # List of (start, end) sorted by start

    def clear(self):
        self.intervals = []

    def find_earliest_start(self, duration: int, earliest: int) -> int:
        """Find earliest time >= earliest where interval of given duration fits without overlap."""
        if not self.intervals:
            return earliest

        # Try to fit before first interval
        if earliest + duration <= self.intervals[0][0]:
            return earliest

        # Try to fit between intervals
        for i in range(len(self.intervals) - 1):
            gap_start = self.intervals[i][1]
            gap_end = self.intervals[i + 1][0]
            start = max(gap_start, earliest)
            if start + duration <= gap_end:
                return start

        # Fit after last interval
        return max(self.intervals[-1][1], earliest)

    def add_interval(self, start: int, end: int):
        """Add an interval to the tracker."""
        # Insert maintaining sorted order
        insort(self.intervals, (start, end))

    def has_overlap(self, start: int, end: int) -> bool:
        """Check if interval [start, end) overlaps with any existing interval."""
        for s, e in self.intervals:
            if start < e and end > s:
                return True
        return False


class CumulativeTracker:
    """Tracks cumulative resource usage with capacity constraint."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.events: List[Tuple[int, int]] = []  # (time, delta) where delta is +1 or -1

    def clear(self):
        self.events = []

    def add_interval(self, start: int, end: int):
        """Add usage interval [start, end)."""
        self.events.append((start, 1))
        self.events.append((end, -1))

    def find_earliest_start(self, start: int, end: int) -> int:
        """
        Find earliest time >= start where adding interval [start, end)
        doesn't exceed capacity.
        """
        duration = end - start

        # Build sorted events
        all_events = sorted(self.events + [(start, 0)])  # Add dummy to check at start

        # Sweep through time
        current_load = 0
        prev_time = 0

        # First check if we can start at the requested time
        for t, delta in sorted(self.events):
            if t <= start:
                current_load += delta

        if current_load < self.capacity:
            # Check if capacity is maintained throughout [start, end)
            ok = True
            for t, delta in sorted(self.events):
                if t > start and t < end:
                    current_load += delta
                    if current_load >= self.capacity:
                        ok = False
                        break
            if ok:
                return start

        # Need to find a later start time
        # Find all time points where load decreases
        times_sorted = sorted(set(t for t, d in self.events))

        for candidate_start in times_sorted:
            if candidate_start < start:
                continue

            # Check load at candidate_start
            load_at_start = 0
            for t, delta in self.events:
                if t <= candidate_start:
                    load_at_start += delta

            if load_at_start >= self.capacity:
                continue

            # Check if load stays below capacity throughout interval
            candidate_end = candidate_start + duration
            ok = True
            load = load_at_start

            for t, delta in sorted(self.events):
                if t > candidate_start and t < candidate_end:
                    load += delta
                    if load >= self.capacity:
                        ok = False
                        break

            if ok:
                return candidate_start

        # Return after all events
        if times_sorted:
            return max(times_sorted[-1], start)
        return start

    def is_valid(self) -> bool:
        """Check if capacity is never exceeded."""
        if not self.events:
            return True

        events_sorted = sorted(self.events)
        current_load = 0

        for t, delta in events_sorted:
            current_load += delta
            if current_load > self.capacity:
                return False

        return True

    def get_max_load(self) -> int:
        """Get maximum load at any point."""
        if not self.events:
            return 0

        events_sorted = sorted(self.events)
        current_load = 0
        max_load = 0

        for t, delta in events_sorted:
            current_load += delta
            max_load = max(max_load, current_load)

        return max_load


class ConstraintScheduler:
    """
    Schedules torpedo cycles respecting all constraints.

    Key insight: We need to schedule by DEADLINE priority, not by metal_start time.
    Conversions with earlier deadlines should get priority for railway segments.
    """

    def __init__(self, data: ProblemData):
        self.data = data

    def schedule_all_cycles(
        self, pouring_to_conv: Dict[int, int]
    ) -> List[TorpedoCycle]:
        """
        Schedule all cycles respecting constraints.
        Uses CHRONOLOGICAL scheduling (by metal_start) to properly track cumulative resources.
        Deadline constraints are checked but we prioritize chronological order for resource allocation.
        """
        d = self.data
        cycles = []

        # Build list of all cycles with their info
        cycle_info = []  # (pouring_idx, conv_idx, metal_start)

        for i in range(d.n):
            conv_idx = pouring_to_conv.get(i, -1)
            cycle_info.append((i, conv_idx, d.metal_starts[i]))

        # Sort by metal_start (chronological order) - this ensures proper cumulative tracking
        cycle_info.sort(key=lambda x: x[2])

        # Now schedule in deadline order, tracking segment usage
        seg1_intervals = []  # (start, end)
        seg2_intervals = []
        seg3_intervals = []
        seg4_intervals = []
        seg5_intervals = []
        seg6_intervals = []
        furnace_intervals = []

        # Cumulative tracking (simplified - just track events)
        full_buffer_events = []  # (time, delta)
        desulf_events = []
        converter_events = []

        scheduled_cycles = {}  # pouring_idx -> cycle

        for pouring_idx, conv_idx, metal_start in cycle_info:
            cycle = self._schedule_with_segments(
                pouring_idx,
                conv_idx,
                seg1_intervals,
                seg2_intervals,
                seg3_intervals,
                seg4_intervals,
                seg5_intervals,
                seg6_intervals,
                furnace_intervals,
                full_buffer_events,
                desulf_events,
                converter_events,
            )
            scheduled_cycles[pouring_idx] = cycle

        # Return cycles sorted by pouring index
        return [scheduled_cycles[i] for i in range(d.n)]

    def _compute_ideal_timing(self, pouring_idx: int, conv_idx: int) -> TorpedoCycle:
        """Compute ideal timing without considering segment conflicts."""
        d = self.data
        cycle = TorpedoCycle(pouring_idx=pouring_idx, conv_idx=conv_idx, torpedo_id=0)

        metal_start = d.metal_starts[pouring_idx]

        cycle.start_trip = max(0, metal_start - d.t_empty_to_furnace)
        cycle.start_furnace = cycle.start_trip + d.t_empty_to_furnace
        cycle.end_furnace = metal_start + d.time_pour

        if conv_idx == -1:
            cycle.end_trip = cycle.end_furnace + d.t_emergency
        else:
            conv_time = d.conv_opens[conv_idx]
            max_sulfur = d.conv_max_sulfur[conv_idx]
            metal_sulfur = d.metal_sulfur[pouring_idx]

            sulfur_reduction = max(0, metal_sulfur - max_sulfur)
            cycle.desulf_time = sulfur_reduction * d.time_desulf_one

            cycle.start_full_buffer = cycle.end_furnace + d.t_furnace_to_full

            min_arrival = (
                cycle.start_full_buffer
                + d.t_full_to_desulf
                + cycle.desulf_time
                + d.t_desulf_to_conv
            )

            if min_arrival <= conv_time:
                slack = conv_time - min_arrival
                cycle.end_full_buffer = cycle.start_full_buffer + slack
            else:
                cycle.end_full_buffer = cycle.start_full_buffer

            cycle.start_desulf = cycle.end_full_buffer + d.t_full_to_desulf
            cycle.end_desulf = cycle.start_desulf + cycle.desulf_time
            cycle.start_converter = cycle.end_desulf + d.t_desulf_to_conv
            cycle.end_converter = conv_time + d.time_convert
            cycle.end_trip = cycle.end_converter + d.t_conv_to_empty

        return cycle

    def _find_earliest_nonoverlap(
        self, intervals: List[Tuple[int, int]], duration: int, earliest: int
    ) -> int:
        """Find earliest time >= earliest where interval fits without overlap."""
        if not intervals:
            return earliest

        sorted_intervals = sorted(intervals)

        # Try before first
        if earliest + duration <= sorted_intervals[0][0]:
            return earliest

        # Try between intervals
        for i in range(len(sorted_intervals) - 1):
            gap_start = sorted_intervals[i][1]
            gap_end = sorted_intervals[i + 1][0]
            start = max(gap_start, earliest)
            if start + duration <= gap_end:
                return start

        # After last
        return max(sorted_intervals[-1][1], earliest)

    def _check_cumulative(
        self, events: List[Tuple[int, int]], start: int, end: int, capacity: int
    ) -> int:
        """
        Check if adding interval [start, end) would exceed capacity.
        Returns the earliest valid start time >= start.

        IMPORTANT: We need to find a time slot where the current load + 1 <= capacity
        throughout the entire interval. So we need load < capacity.
        """
        if not events:
            return start

        duration = end - start

        # Helper: check if interval [s, e) can fit without exceeding capacity
        def can_fit(s: int, e: int) -> bool:
            # Calculate load at each point in [s, e)
            # After adding our interval, load becomes load+1, must be <= capacity
            # So we need: load < capacity throughout [s, e)
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

        # First, try the requested start time
        if can_fit(start, end):
            return start

        # Collect all event times and also end-of-intervals (where load might drop)
        all_times = set()
        for t, delta in events:
            all_times.add(t)
        times = sorted(all_times)

        # Search for valid slot starting at each event time
        for t in times:
            if t <= start:
                continue

            candidate_start = t
            candidate_end = candidate_start + duration

            if can_fit(candidate_start, candidate_end):
                return candidate_start

        # If no good slot found, find after the last event
        if events:
            # Find the latest end time (where load decreases)
            end_times = [t for t, d in events if d < 0]
            if end_times:
                latest_end = max(end_times)
                candidate_start = max(latest_end, start)
                candidate_end = candidate_start + duration
                if can_fit(candidate_start, candidate_end):
                    return candidate_start
                # Try right after latest_end
                candidate_start = latest_end
                candidate_end = candidate_start + duration
                if can_fit(candidate_start, candidate_end):
                    return candidate_start

        # Final fallback: find ANY time in far future where it fits
        # This is a safety net - compute when load reaches 0
        final_load = sum(delta for _, delta in events)
        if final_load == 0:  # Balanced events
            if end_times:
                far_future = max(end_times) + 1
                return max(far_future, start)

        # Should not happen with balanced events, but return a very late time
        return max(t for t, _ in events) + duration + 1

    def _schedule_with_segments(
        self,
        pouring_idx: int,
        conv_idx: int,
        seg1: List,
        seg2: List,
        seg3: List,
        seg4: List,
        seg5: List,
        seg6: List,
        furnace: List,
        full_buffer_events: List,
        desulf_events: List,
        converter_events: List,
    ) -> TorpedoCycle:
        """
        Schedule a cycle respecting all segment and cumulative constraints.

        IMPORTANT: We compute all times first, THEN register segments/resources
        to avoid inconsistencies when adjustments are made.
        """
        d = self.data
        cycle = TorpedoCycle(pouring_idx=pouring_idx, conv_idx=conv_idx, torpedo_id=0)

        metal_start = d.metal_starts[pouring_idx]

        # Segment 1: Empty -> Furnace
        # Must arrive at furnace by metal_start
        earliest_seg1 = max(0, metal_start - d.t_empty_to_furnace)
        seg1_start = self._find_earliest_nonoverlap(
            seg1, d.t_empty_to_furnace, earliest_seg1
        )

        cycle.start_trip = seg1_start
        cycle.start_furnace = seg1_start + d.t_empty_to_furnace

        # Furnace timing is fixed by problem
        cycle.end_furnace = metal_start + d.time_pour

        if conv_idx == -1:
            # Emergency route - CRITICAL: seg6 MUST start exactly at end_furnace
            # This is a hard constraint - emergency transport starts immediately
            # end_trip = end_furnace + t_emergency (fixed, no flexibility)
            cycle.end_trip = cycle.end_furnace + d.t_emergency

            # Now register all segments
            seg1.append((cycle.start_trip, cycle.start_furnace))
            furnace.append((cycle.start_furnace, cycle.end_furnace))
            # Register segment from end_furnace to end_trip
            # NOTE: If two emergency pourings overlap here, SA must avoid this assignment!
            seg6.append((cycle.end_furnace, cycle.end_trip))
        else:
            conv_time = d.conv_opens[conv_idx]
            max_sulfur = d.conv_max_sulfur[conv_idx]
            metal_sulfur = d.metal_sulfur[pouring_idx]

            sulfur_reduction = max(0, metal_sulfur - max_sulfur)
            cycle.desulf_time = sulfur_reduction * d.time_desulf_one

            # Segment 2: Furnace -> Full Buffer
            seg2_start = self._find_earliest_nonoverlap(
                seg2, d.t_furnace_to_full, cycle.end_furnace
            )
            cycle.start_full_buffer = seg2_start + d.t_furnace_to_full
            seg2.append((seg2_start, cycle.start_full_buffer))

            # Initialize - no wait at full buffer initially
            cycle.end_full_buffer = cycle.start_full_buffer

            # Combined scheduling for Seg3 + Desulf + Seg4 + Converter
            # MUST satisfy:
            # 1. seg3 NoOverlap
            # 2. desulf cumulative (for the FULL duration: start_desulf to end_desulf)
            # 3. seg4 NoOverlap
            # 4. converter cumulative
            # 5. start_converter >= conv_time (converter opens constraint)
            #
            # KEY INSIGHT: If we need to wait for converter, we wait in FULL BUFFER (before desulf),
            # not at desulf! Otherwise we overflow desulf cumulative.

            max_iterations = 200
            found_valid = False

            for _ in range(max_iterations):
                # Step 1: Find seg3 slot
                seg3_start = self._find_earliest_nonoverlap(
                    seg3, d.t_full_to_desulf, cycle.end_full_buffer
                )
                if seg3_start > cycle.end_full_buffer:
                    cycle.end_full_buffer = seg3_start
                cycle.start_desulf = seg3_start + d.t_full_to_desulf

                # Step 2: Calculate minimum end_desulf (just processing time)
                min_end_desulf = cycle.start_desulf + cycle.desulf_time

                # Step 3: Torpedo can ARRIVE at converter parking before it opens
                # arrival = end_desulf + t_desulf_to_conv (travel time)
                # actual_conversion_start = max(arrival, conv_time)
                earliest_arrival = min_end_desulf + d.t_desulf_to_conv

                # Step 4: Set arrival and calculate actual conversion timing
                cycle.start_converter = earliest_arrival  # This is ARRIVAL time
                actual_conv_start = max(cycle.start_converter, conv_time)
                cycle.end_converter = actual_conv_start + d.time_convert

                # Check converter cumulative (based on actual conversion, not arrival)
                conv_start_check = self._check_cumulative(
                    converter_events,
                    actual_conv_start,
                    cycle.end_converter,
                    d.slots_converter,
                )
                if conv_start_check > actual_conv_start:
                    # Need later slot - adjust arrival so actual_conv_start = conv_start_check
                    actual_conv_start = conv_start_check
                    cycle.start_converter = max(
                        earliest_arrival, actual_conv_start
                    )  # Can't arrive before earliest
                    cycle.end_converter = actual_conv_start + d.time_convert

                # Step 5: Calculate end_desulf that is compatible with start_converter
                # end_desulf = start_converter - t_desulf_to_conv, but at least min_end_desulf
                required_end_desulf = cycle.start_converter - d.t_desulf_to_conv
                cycle.end_desulf = max(min_end_desulf, required_end_desulf)

                # Step 6: Now find valid seg4 slot
                # seg4 is from end_desulf to start_converter (duration = t_desulf_to_conv)
                # We need to ensure this doesn't overlap with existing seg4
                seg4_start = self._find_earliest_nonoverlap(
                    seg4, d.t_desulf_to_conv, cycle.end_desulf
                )
                if seg4_start > cycle.end_desulf:
                    # Need to delay - adjust end_desulf and arrival
                    cycle.end_desulf = seg4_start
                    cycle.start_converter = cycle.end_desulf + d.t_desulf_to_conv
                    actual_conv_start = max(cycle.start_converter, conv_time)
                    cycle.end_converter = actual_conv_start + d.time_convert

                    # Re-check converter cumulative (based on actual conversion time)
                    conv_start_check = self._check_cumulative(
                        converter_events,
                        actual_conv_start,
                        cycle.end_converter,
                        d.slots_converter,
                    )
                    if conv_start_check > actual_conv_start:
                        actual_conv_start = conv_start_check
                        cycle.end_converter = actual_conv_start + d.time_convert
                        # We can arrive earlier, but conversion starts later
                        # Keep arrival time as is if possible

                # Step 7: Check desulf cumulative for the ACTUAL duration
                if cycle.desulf_time > 0 or cycle.end_desulf > cycle.start_desulf:
                    # We occupy desulf from start_desulf to end_desulf
                    desulf_start = self._check_cumulative(
                        desulf_events,
                        cycle.start_desulf,
                        cycle.end_desulf,
                        d.slots_desulf,
                    )
                    if desulf_start > cycle.start_desulf:
                        # Need to delay start - wait longer in full buffer
                        cycle.end_full_buffer = desulf_start - d.t_full_to_desulf
                        continue

                # All constraints satisfied!
                found_valid = True
                break

            # Register segments and resources
            seg3.append((cycle.end_full_buffer, cycle.start_desulf))

            if cycle.desulf_time > 0 or cycle.end_desulf > cycle.start_desulf:
                desulf_events.append((cycle.start_desulf, 1))
                desulf_events.append((cycle.end_desulf, -1))

            seg4.append((cycle.end_desulf, cycle.start_converter))

            # Converter cumulative: based on ACTUAL conversion time (when converter opens)
            # Torpedo can arrive earlier and wait at parking
            actual_conv_start = max(cycle.start_converter, conv_time)
            converter_events.append((actual_conv_start, 1))
            converter_events.append((cycle.end_converter, -1))

            # Segment 5: Converter -> Empty (IS NoOverlap per documentation)
            seg5_start = self._find_earliest_nonoverlap(
                seg5, d.t_conv_to_empty, cycle.end_converter
            )
            if seg5_start > cycle.end_converter:
                # Need to wait at converter - extend end_converter
                cycle.end_converter = seg5_start
            cycle.end_trip = cycle.end_converter + d.t_conv_to_empty
            seg5.append((cycle.end_converter, cycle.end_trip))

            # Register remaining segments/resources
            seg1.append((cycle.start_trip, cycle.start_furnace))
            furnace.append((cycle.start_furnace, cycle.end_furnace))

            if cycle.end_full_buffer > cycle.start_full_buffer:
                full_buffer_events.append((cycle.start_full_buffer, 1))
                full_buffer_events.append((cycle.end_full_buffer, -1))

        return cycle


class SolutionValidator:
    """Validates a complete solution against all constraints."""

    def __init__(self, data: ProblemData):
        self.data = data
        self.errors = []

    def validate(
        self, cycles: List[TorpedoCycle], pouring_to_conv: Dict[int, int]
    ) -> bool:
        """Validate complete solution. Returns True if valid."""
        self.errors = []
        d = self.data

        # 1. Check timing consistency for each cycle
        for cycle in cycles:
            p = cycle.pouring_idx

            # Check furnace timing
            if cycle.start_furnace > d.metal_starts[p]:
                self.errors.append(
                    f"Pouring {p}: start_furnace ({cycle.start_furnace}) > metal_start ({d.metal_starts[p]})"
                )

            if cycle.end_furnace != d.metal_starts[p] + d.time_pour:
                self.errors.append(
                    f"Pouring {p}: end_furnace ({cycle.end_furnace}) != metal_start + time_pour ({d.metal_starts[p] + d.time_pour})"
                )

            if cycle.conv_idx >= 0:
                # Non-emergency route
                c = cycle.conv_idx

                # NOTE: start_converter is ARRIVAL time - torpedo can arrive early and wait at parking
                # The actual conversion starts at max(start_converter, conv_opens)
                # No constraint check needed here - cumulative handles the actual conversion timing

                # Check desulf time
                sulfur_needed = max(0, d.metal_sulfur[p] - d.conv_max_sulfur[c])
                min_desulf = sulfur_needed * d.time_desulf_one
                actual_desulf = cycle.end_desulf - cycle.start_desulf
                if actual_desulf < min_desulf:
                    self.errors.append(
                        f"Pouring {p}: desulf_time ({actual_desulf}) < required ({min_desulf})"
                    )

        # 2. Check segment overlaps (NoOverlap constraints)
        self._check_segment_overlaps(
            cycles,
            "Seg1 (Empty->Furnace)",
            lambda c: (c.start_trip, c.start_trip + d.t_empty_to_furnace)
            if True
            else None,
        )

        self._check_segment_overlaps(
            cycles, "Furnace", lambda c: (c.start_furnace, c.end_furnace)
        )

        self._check_segment_overlaps(
            cycles,
            "Seg2 (Furnace->Full)",
            lambda c: (c.end_furnace, c.start_full_buffer) if c.conv_idx >= 0 else None,
        )

        self._check_segment_overlaps(
            cycles,
            "Seg3 (Full->Desulf)",
            lambda c: (c.end_full_buffer, c.start_desulf) if c.conv_idx >= 0 else None,
        )

        self._check_segment_overlaps(
            cycles,
            "Seg4 (Desulf->Conv)",
            lambda c: (c.end_desulf, c.start_converter) if c.conv_idx >= 0 else None,
        )

        self._check_segment_overlaps(
            cycles,
            "Seg5 (Conv->Empty)",
            lambda c: (c.end_converter, c.end_trip) if c.conv_idx >= 0 else None,
        )

        self._check_segment_overlaps(
            cycles,
            "Seg6 (Emergency)",
            lambda c: (c.end_furnace, c.end_trip) if c.conv_idx == -1 else None,
        )

        # 3. Check cumulative constraints
        self._check_cumulative(
            cycles,
            "Full Buffer",
            d.slots_full_buffer,
            lambda c: (c.start_full_buffer, c.end_full_buffer)
            if c.conv_idx >= 0 and c.end_full_buffer > c.start_full_buffer
            else None,
        )

        self._check_cumulative(
            cycles,
            "Desulf",
            d.slots_desulf,
            lambda c: (c.start_desulf, c.end_desulf)
            if c.conv_idx >= 0 and c.desulf_time > 0
            else None,
        )

        self._check_cumulative(
            cycles,
            "Converter",
            d.slots_converter,
            lambda c: (c.start_converter, c.end_converter) if c.conv_idx >= 0 else None,
        )

        # 4. Check each conversion is assigned at most once
        conv_assignments = {}
        for c in cycles:
            if c.conv_idx >= 0:
                if c.conv_idx in conv_assignments:
                    self.errors.append(
                        f"Conversion {c.conv_idx} assigned to multiple pourings: {conv_assignments[c.conv_idx]} and {c.pouring_idx}"
                    )
                else:
                    conv_assignments[c.conv_idx] = c.pouring_idx

        return len(self.errors) == 0

    def _check_segment_overlaps(self, cycles, name, get_interval):
        """Check for overlaps in a NoOverlap segment."""
        intervals = []
        for c in cycles:
            interval = get_interval(c)
            if interval is not None:
                intervals.append((interval[0], interval[1], c.pouring_idx))

        intervals.sort()

        for i in range(1, len(intervals)):
            prev_end = intervals[i - 1][1]
            curr_start = intervals[i][0]
            if curr_start < prev_end:
                self.errors.append(
                    f"{name}: overlap between pouring {intervals[i - 1][2]} (ends {prev_end}) and {intervals[i][2]} (starts {curr_start})"
                )

    def _check_cumulative(self, cycles, name, capacity, get_interval):
        """Check cumulative resource constraint."""
        events = []
        for c in cycles:
            interval = get_interval(c)
            if interval is not None:
                events.append((interval[0], 1, c.pouring_idx))
                events.append((interval[1], -1, c.pouring_idx))

        events.sort()
        current_load = 0
        max_load = 0

        for t, delta, p in events:
            current_load += delta
            if current_load > capacity:
                self.errors.append(
                    f"{name}: capacity exceeded ({current_load} > {capacity}) at time {t}"
                )
            max_load = max(max_load, current_load)


class TorpedoScheduler:
    """
    Simulated Annealing solver for Torpedo Scheduling.
    Stage 1: Focus on minimizing missed conversions.
    """

    def __init__(self, data: ProblemData, max_torpedoes: int = 10, seed: int = None):
        self.data = data
        self.max_torpedoes = max_torpedoes
        if seed is not None:
            random.seed(seed)

        # Precompute feasibility matrix
        self._precompute_feasibility()

        # Current solution
        self.cycles: List[TorpedoCycle] = []
        self.conv_to_pouring: Dict[int, int] = {}  # conv_idx -> pouring_idx
        self.pouring_to_conv: Dict[
            int, int
        ] = {}  # pouring_idx -> conv_idx (-1 for emergency)

        # Scheduler for constraint-aware timing
        self.scheduler = ConstraintScheduler(data)

        # Validator
        self.validator = SolutionValidator(data)

    def _precompute_feasibility(self):
        """Precompute which pourings can serve which conversions."""
        self.feasible_convs: Dict[
            int, List[int]
        ] = {}  # pouring_idx -> list of feasible conv_idx
        self.feasible_pourings: Dict[
            int, List[int]
        ] = {}  # conv_idx -> list of feasible pouring_idx

        # Also precompute which pourings would overlap in emergency
        self.emergency_conflicts: Dict[
            int, Set[int]
        ] = {}  # pouring_idx -> set of conflicting pouring_idx

        # Use slack margin to avoid borderline feasibility that fails due to segment delays
        # Reduced from 15 to 5 to allow more conversions to be feasible
        SLACK_MARGIN = 5

        for i in range(self.data.n):
            self.feasible_convs[i] = []
            for j in range(self.data.m):
                if self.data.can_reach(i, j, slack=SLACK_MARGIN):
                    self.feasible_convs[i].append(j)

        for j in range(self.data.m):
            self.feasible_pourings[j] = []
            for i in range(self.data.n):
                if self.data.can_reach(i, j, slack=SLACK_MARGIN):
                    self.feasible_pourings[j].append(i)

        # Precompute emergency conflicts
        # Two pourings conflict in emergency if their end_furnace times are within t_emergency of each other
        for i in range(self.data.n):
            self.emergency_conflicts[i] = set()
            end_furnace_i = self.data.metal_starts[i] + self.data.time_pour
            for j in range(self.data.n):
                if i == j:
                    continue
                end_furnace_j = self.data.metal_starts[j] + self.data.time_pour
                # Check if intervals [end_furnace_i, end_furnace_i + t_emergency) and
                # [end_furnace_j, end_furnace_j + t_emergency) overlap
                if (
                    end_furnace_i < end_furnace_j + self.data.t_emergency
                    and end_furnace_j < end_furnace_i + self.data.t_emergency
                ):
                    self.emergency_conflicts[i].add(j)

    def _rebuild_cycles_with_scheduling(self):
        """Rebuild cycles with proper constraint-aware scheduling."""
        self.cycles = self.scheduler.schedule_all_cycles(self.pouring_to_conv)

    def _create_initial_solution(self):
        """Create initial solution using greedy assignment."""
        self.pouring_to_conv = {}
        self.conv_to_pouring = {}

        # Sort conversions by opening time
        conv_order = sorted(range(self.data.m), key=lambda j: self.data.conv_opens[j])

        # For each conversion, find the best available pouring
        assigned_pourings = set()

        for j in conv_order:
            best_pouring = None
            best_slack = float("inf")

            for i in self.feasible_pourings[j]:
                if i in assigned_pourings:
                    continue

                # Calculate slack (how much time before deadline)
                earliest_arrival = self.data.metal_starts[
                    i
                ] + self.data.min_travel_time(i, j)
                slack = self.data.conv_opens[j] - earliest_arrival

                if slack >= 0 and slack < best_slack:
                    best_slack = slack
                    best_pouring = i

            if best_pouring is not None:
                self.pouring_to_conv[best_pouring] = j
                self.conv_to_pouring[j] = best_pouring
                assigned_pourings.add(best_pouring)

        # All unassigned pourings go to emergency
        for i in range(self.data.n):
            if i not in self.pouring_to_conv:
                self.pouring_to_conv[i] = -1

        # Build cycles with scheduling
        self._rebuild_cycles_with_scheduling()

    def _count_missed_conversions(self) -> int:
        """Count number of missed conversions."""
        return self.data.m - len(self.conv_to_pouring)

    def _count_deadline_violations(self) -> int:
        """Count number of deadline violations."""
        violations = 0
        for cycle in self.cycles:
            if cycle.conv_idx >= 0:
                if cycle.start_converter > self.data.conv_opens[cycle.conv_idx]:
                    violations += 1
        return violations

    def _count_constraint_violations(self) -> int:
        """Count total constraint violations (segment overlaps, cumulative capacity)."""
        # Quick validation without storing error messages
        violations = 0
        d = self.data

        # Check segment overlaps
        def check_segment_overlaps(intervals):
            sorted_intervals = sorted(intervals)
            count = 0
            for i in range(1, len(sorted_intervals)):
                if sorted_intervals[i][0] < sorted_intervals[i - 1][1]:
                    count += 1
            return count

        # Seg6 (Emergency) - CRITICAL: emergency segments MUST start at end_furnace
        # Count emergency conflicts using precomputed data
        emergency_pourings = [c.pouring_idx for c in self.cycles if c.conv_idx == -1]
        emergency_set = set(emergency_pourings)

        for p in emergency_pourings:
            conflicts_in_emergency = self.emergency_conflicts[p] & emergency_set
            violations += len(conflicts_in_emergency)
        violations //= 2  # Each conflict is counted twice

        # Check cumulative constraints
        def check_cumulative(intervals, capacity):
            if not intervals:
                return 0
            events = []
            for start, end in intervals:
                events.append((start, 1))
                events.append((end, -1))
            events.sort()
            load = 0
            count = 0
            for t, delta in events:
                load += delta
                if load > capacity:
                    count += 1
            return count

        # Desulf cumulative
        desulf_intervals = [
            (c.start_desulf, c.end_desulf)
            for c in self.cycles
            if c.conv_idx >= 0 and c.desulf_time > 0
        ]
        violations += check_cumulative(desulf_intervals, d.slots_desulf)

        # Converter cumulative
        conv_intervals = [
            (c.start_converter, c.end_converter) for c in self.cycles if c.conv_idx >= 0
        ]
        violations += check_cumulative(conv_intervals, d.slots_converter)

        return violations

    def _evaluate_solution(self) -> float:
        """
        Evaluate current solution quality.
        Returns a score (lower is better).
        """
        missed = self._count_missed_conversions()
        deadline_violations = self._count_deadline_violations()
        constraint_violations = self._count_constraint_violations()
        total_desulf = sum(c.desulf_time for c in self.cycles if c.conv_idx >= 0)

        # Large weights for lexicographic ordering
        MISSED_WEIGHT = 10**12
        DEADLINE_WEIGHT = 10**9
        CONSTRAINT_WEIGHT = (
            10**8
        )  # Penalize other constraint violations (including emergency overlaps!)
        DESULF_WEIGHT = 1

        return (
            missed * MISSED_WEIGHT
            + deadline_violations * DEADLINE_WEIGHT
            + constraint_violations * CONSTRAINT_WEIGHT
            + total_desulf * DESULF_WEIGHT
        )

    def _move_swap_conversions(self) -> bool:
        """Move: Swap conversion assignments between two pourings."""
        serving_pourings = [i for i, j in self.pouring_to_conv.items() if j >= 0]
        if len(serving_pourings) < 2:
            return False

        p1 = random.choice(serving_pourings)
        j1 = self.pouring_to_conv[p1]

        candidates = []

        # Other serving pourings within range (use slack for safety)
        SLACK = 5
        for p2 in serving_pourings:
            if p2 == p1:
                continue
            j2 = self.pouring_to_conv[p2]
            if self.data.can_reach(p1, j2, slack=SLACK) and self.data.can_reach(
                p2, j1, slack=SLACK
            ):
                candidates.append((p2, j2))

        # Emergency pourings that can serve j1
        for p2, j2 in self.pouring_to_conv.items():
            if j2 == -1 and self.data.can_reach(p2, j1, slack=SLACK):
                candidates.append((p2, -1))

        if not candidates:
            return False

        p2, j2 = random.choice(candidates)

        if j2 == -1:
            self.pouring_to_conv[p1] = -1
            self.pouring_to_conv[p2] = j1
            del self.conv_to_pouring[j1]
            self.conv_to_pouring[j1] = p2
        else:
            self.pouring_to_conv[p1] = j2
            self.pouring_to_conv[p2] = j1
            self.conv_to_pouring[j1] = p2
            self.conv_to_pouring[j2] = p1

        self._rebuild_cycles_with_scheduling()
        return True

    def _move_reassign_missed(self) -> bool:
        """Move: Try to assign a missed conversion to an emergency pouring."""
        missed_convs = [j for j in range(self.data.m) if j not in self.conv_to_pouring]
        if not missed_convs:
            return False

        target_conv = random.choice(missed_convs)
        SLACK_MARGIN = 5

        candidates = []
        for i, j in self.pouring_to_conv.items():
            if j == -1 and self.data.can_reach(i, target_conv, slack=SLACK_MARGIN):
                candidates.append(i)

        if not candidates:
            return False

        best_i = None
        best_slack = -float("inf")

        for i in candidates:
            earliest_arrival = self.data.metal_starts[i] + self.data.min_travel_time(
                i, target_conv
            )
            slack = self.data.conv_opens[target_conv] - earliest_arrival
            if slack > best_slack:
                best_slack = slack
                best_i = i

        if best_i is None or best_slack < SLACK_MARGIN:
            return False

        self.pouring_to_conv[best_i] = target_conv
        self.conv_to_pouring[target_conv] = best_i
        self._rebuild_cycles_with_scheduling()
        return True

    def _move_unassign_conversion(self) -> bool:
        """Move: Unassign a conversion (send pouring to emergency)."""
        serving_pourings = [i for i, j in self.pouring_to_conv.items() if j >= 0]
        if not serving_pourings:
            return False

        i = random.choice(serving_pourings)
        j = self.pouring_to_conv[i]

        self.pouring_to_conv[i] = -1
        del self.conv_to_pouring[j]
        self._rebuild_cycles_with_scheduling()
        return True

    def _move_resolve_emergency_conflict(self) -> bool:
        """Move: Try to resolve emergency conflict by assigning one pouring to a conversion."""
        # Find all emergency pourings that have conflicts
        emergency_pourings = [i for i, j in self.pouring_to_conv.items() if j == -1]
        emergency_set = set(emergency_pourings)

        conflicting_pairs = []
        for p in emergency_pourings:
            conflicts = self.emergency_conflicts[p] & emergency_set
            for q in conflicts:
                if p < q:  # Avoid duplicates
                    conflicting_pairs.append((p, q))

        if not conflicting_pairs:
            return False

        # Pick a random conflicting pair
        p1, p2 = random.choice(conflicting_pairs)

        # Try to assign one of them to a conversion
        # Prefer the one with more feasible conversions
        feasible_p1 = [
            j for j in self.feasible_convs[p1] if j not in self.conv_to_pouring
        ]
        feasible_p2 = [
            j for j in self.feasible_convs[p2] if j not in self.conv_to_pouring
        ]

        candidates = []
        if feasible_p1:
            candidates.append((p1, feasible_p1))
        if feasible_p2:
            candidates.append((p2, feasible_p2))

        if not candidates:
            return False

        # Pick the candidate with more options
        p, feasible = max(candidates, key=lambda x: len(x[1]))

        # Pick a random feasible conversion
        target_conv = random.choice(feasible)

        self.pouring_to_conv[p] = target_conv
        self.conv_to_pouring[target_conv] = p
        self._rebuild_cycles_with_scheduling()
        return True

    def _move_chain_reassign(self) -> bool:
        """Move: Chain reassignment."""
        if not self.conv_to_pouring:
            return False

        start_j = random.choice(list(self.conv_to_pouring.keys()))
        start_i = self.conv_to_pouring[start_j]

        chain = [(start_i, start_j)]
        visited_convs = {start_j}
        visited_pourings = {start_i}

        current_i = start_i

        for _ in range(5):
            candidates = [
                j
                for j in self.feasible_convs[current_i]
                if j not in visited_convs and j in self.conv_to_pouring
            ]

            if not candidates:
                break

            next_j = random.choice(candidates)
            next_i = self.conv_to_pouring[next_j]

            if next_i in visited_pourings:
                break

            chain.append((next_i, next_j))
            visited_convs.add(next_j)
            visited_pourings.add(next_i)
            current_i = next_i

        if len(chain) < 2:
            return False

        SLACK = 5
        for idx in range(len(chain)):
            i, _ = chain[idx]
            _, new_j = chain[(idx + 1) % len(chain)]
            if not self.data.can_reach(i, new_j, slack=SLACK):
                return False

        for idx in range(len(chain)):
            i, old_j = chain[idx]
            _, new_j = chain[(idx + 1) % len(chain)]

            self.pouring_to_conv[i] = new_j
            self.conv_to_pouring[new_j] = i

        self._rebuild_cycles_with_scheduling()
        return True

    def _resolve_emergency_conflicts_greedy(self):
        """
        Post-processing: greedily resolve emergency conflicts by assigning
        one pouring from each conflicting pair to an available conversion.

        If no free conversions are available, try to swap with a non-conflicting
        pouring that currently holds a feasible conversion.
        """
        max_iterations = 1000
        for _ in range(max_iterations):
            # Find emergency conflicts
            emergency_pourings = [i for i, j in self.pouring_to_conv.items() if j == -1]
            emergency_set = set(emergency_pourings)

            conflicting_pairs = []
            for p in emergency_pourings:
                conflicts = self.emergency_conflicts[p] & emergency_set
                for q in conflicts:
                    if p < q:
                        conflicting_pairs.append((p, q))

            if not conflicting_pairs:
                self._rebuild_cycles_with_scheduling()
                return  # No conflicts, done

            # Try to resolve conflicts one by one
            resolved_any = False
            for p1, p2 in conflicting_pairs:
                # First, try to assign to a free conversion
                feasible_p1 = [
                    j for j in self.feasible_convs[p1] if j not in self.conv_to_pouring
                ]
                feasible_p2 = [
                    j for j in self.feasible_convs[p2] if j not in self.conv_to_pouring
                ]

                best_p = None
                best_conv = None

                if feasible_p1 and (
                    not feasible_p2 or len(feasible_p1) >= len(feasible_p2)
                ):
                    best_p = p1
                    best_conv = max(
                        feasible_p1,
                        key=lambda j: self.data.conv_opens[j]
                        - self.data.metal_starts[p1]
                        - self.data.min_travel_time(p1, j),
                    )
                elif feasible_p2:
                    best_p = p2
                    best_conv = max(
                        feasible_p2,
                        key=lambda j: self.data.conv_opens[j]
                        - self.data.metal_starts[p2]
                        - self.data.min_travel_time(p2, j),
                    )

                if best_p is not None and best_conv is not None:
                    self.pouring_to_conv[best_p] = best_conv
                    self.conv_to_pouring[best_conv] = best_p
                    resolved_any = True
                    break

                # If no free conversion, try to SWAP with a non-conflicting pouring
                # Find all conversions that one of our pourings can serve, currently held by others
                for p_conflict in [p1, p2]:
                    for j in self.feasible_convs[p_conflict]:
                        if j in self.conv_to_pouring:
                            other_p = self.conv_to_pouring[j]
                            # Check if other_p can go to emergency without creating new conflicts
                            would_conflict = False
                            other_end_furnace = (
                                self.data.metal_starts[other_p] + self.data.time_pour
                            )
                            for em_p in emergency_pourings:
                                if em_p == p_conflict:
                                    continue  # This one will leave emergency
                                em_end = (
                                    self.data.metal_starts[em_p] + self.data.time_pour
                                )
                                if (
                                    abs(em_end - other_end_furnace)
                                    < self.data.t_emergency
                                ):
                                    would_conflict = True
                                    break

                            if not would_conflict:
                                # Swap: p_conflict takes j, other_p goes to emergency
                                self.pouring_to_conv[p_conflict] = j
                                self.conv_to_pouring[j] = p_conflict
                                self.pouring_to_conv[other_p] = -1
                                resolved_any = True
                                break
                    if resolved_any:
                        break
                if resolved_any:
                    break

            if not resolved_any:
                # Last resort: try reducing slack margin for feasibility check
                # and look for borderline conversions
                self._rebuild_cycles_with_scheduling()
                return

        self._rebuild_cycles_with_scheduling()

    def solve(self, time_limit: float = 300.0, verbose: bool = True) -> dict:
        """Run Simulated Annealing to minimize missed conversions."""
        start_time = time.time()

        self._create_initial_solution()

        current_score = self._evaluate_solution()
        best_score = current_score
        best_assignment = self.pouring_to_conv.copy()

        initial_missed = self._count_missed_conversions()
        initial_violations = self._count_deadline_violations()
        if verbose:
            print(
                f"Initial: {initial_missed} missed, {initial_violations} deadline violations"
            )

        temperature = current_score / 10.0 if current_score > 0 else 1000000
        cooling_rate = 0.9995
        min_temperature = 1.0

        iteration = 0
        improvements = 0
        no_improve_count = 0

        # Move probabilities: swap, reassign_missed, unassign, chain, resolve_emergency
        move_probs = [0.30, 0.30, 0.05, 0.15, 0.20]

        while time.time() - start_time < time_limit:
            old_pouring_to_conv = self.pouring_to_conv.copy()
            old_conv_to_pouring = self.conv_to_pouring.copy()

            r = random.random()
            move_made = False

            cumsum = 0
            if r < (cumsum := cumsum + move_probs[0]):
                move_made = self._move_swap_conversions()
            elif r < (cumsum := cumsum + move_probs[1]):
                move_made = self._move_reassign_missed()
            elif r < (cumsum := cumsum + move_probs[2]):
                move_made = self._move_unassign_conversion()
            elif r < (cumsum := cumsum + move_probs[3]):
                move_made = self._move_chain_reassign()
            else:
                move_made = self._move_resolve_emergency_conflict()

            if not move_made:
                continue

            new_score = self._evaluate_solution()
            delta = new_score - current_score

            if delta < 0 or random.random() < math.exp(-delta / max(temperature, 1)):
                current_score = new_score

                if new_score < best_score:
                    best_score = new_score
                    best_assignment = self.pouring_to_conv.copy()
                    improvements += 1
                    no_improve_count = 0

                    if verbose and improvements % 10 == 0:
                        missed = self._count_missed_conversions()
                        violations = self._count_deadline_violations()
                        elapsed = time.time() - start_time
                        print(
                            f"[{elapsed:.1f}s] #{improvements}: {missed} missed, {violations} violations"
                        )
                else:
                    no_improve_count += 1
            else:
                self.pouring_to_conv = old_pouring_to_conv
                self.conv_to_pouring = old_conv_to_pouring
                self._rebuild_cycles_with_scheduling()
                no_improve_count += 1

            temperature *= cooling_rate
            if temperature < min_temperature:
                temperature = min_temperature

            if no_improve_count > 10000:
                temperature = max(
                    temperature * 10, current_score / 100 if current_score > 0 else 1000
                )
                no_improve_count = 0
                if verbose:
                    print(f"Reheating to {temperature:.0f}")

            iteration += 1

        # Restore best solution
        self.pouring_to_conv = best_assignment
        self.conv_to_pouring = {j: i for i, j in best_assignment.items() if j >= 0}
        self._rebuild_cycles_with_scheduling()

        # Post-processing: try to resolve emergency conflicts
        self._resolve_emergency_conflicts_greedy()

        # Validate solution
        is_valid = self.validator.validate(self.cycles, self.pouring_to_conv)

        final_missed = self._count_missed_conversions()
        final_violations = self._count_deadline_violations()
        elapsed = time.time() - start_time

        if verbose:
            print(f"\nFinal after {elapsed:.1f}s:")
            print(f"  Missed: {final_missed}")
            print(f"  Deadline violations: {final_violations}")
            print(f"  Valid: {is_valid}")
            if not is_valid:
                print(f"  Errors ({len(self.validator.errors)}):")
                for err in self.validator.errors[:10]:
                    print(f"    - {err}")

        return self._build_solution()

    def _assign_torpedoes(self):
        """Assign torpedo IDs to cycles."""
        pouring_order = sorted(
            range(self.data.n), key=lambda i: self.data.metal_starts[i]
        )

        torpedo_available = {}
        next_torpedo_id = 1

        for i in pouring_order:
            cycle = self.cycles[i]
            latest_start = self.data.metal_starts[i] - self.data.t_empty_to_furnace

            assigned_torpedo = None
            for t_id, avail_time in torpedo_available.items():
                if avail_time <= latest_start:
                    assigned_torpedo = t_id
                    break

            if assigned_torpedo is None:
                assigned_torpedo = next_torpedo_id
                next_torpedo_id += 1

            cycle.torpedo_id = assigned_torpedo
            torpedo_available[assigned_torpedo] = cycle.end_trip

    def _build_solution(self) -> dict:
        """Build solution dictionary for output."""
        self._assign_torpedoes()

        solution = {
            "assignedTorpedo": [],
            "assignedConversion": [],
            "startTrip": [],
            "startFurnace": [],
            "endFurnace": [],
            "startFullBuffer": [],
            "endFullBuffer": [],
            "startDesulf": [],
            "endDesulf": [],
            "startConverter": [],
            "endConverter": [],
            "endTrip": [],
        }

        for i in range(self.data.n):
            cycle = self.cycles[i]
            solution["assignedTorpedo"].append(cycle.torpedo_id)
            solution["assignedConversion"].append(
                cycle.conv_idx + 1 if cycle.conv_idx >= 0 else -1
            )
            solution["startTrip"].append(cycle.start_trip)
            solution["startFurnace"].append(cycle.start_furnace)
            solution["endFurnace"].append(cycle.end_furnace)
            solution["startFullBuffer"].append(cycle.start_full_buffer)
            solution["endFullBuffer"].append(cycle.end_full_buffer)
            solution["startDesulf"].append(cycle.start_desulf)
            solution["endDesulf"].append(cycle.end_desulf)
            solution["startConverter"].append(cycle.start_converter)
            solution["endConverter"].append(cycle.end_converter)
            solution["endTrip"].append(cycle.end_trip)

        solution["numMissedConversions"] = self._count_missed_conversions()
        solution["numUsedTorpedoes"] = len(set(c.torpedo_id for c in self.cycles))
        solution["totalDesulfTime"] = sum(
            c.desulf_time for c in self.cycles if c.conv_idx >= 0
        )

        return solution


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


def format_solution(solution: dict) -> str:
    """Format solution for output file."""
    lines = []

    def format_array(name, arr):
        return f"{name} = [{', '.join(map(str, arr))}];"

    lines.append(format_array("assignedTorpedo", solution["assignedTorpedo"]))
    lines.append(format_array("assignedConversion", solution["assignedConversion"]))
    lines.append(format_array("startTrip", solution["startTrip"]))
    lines.append(format_array("startFurnace", solution["startFurnace"]))
    lines.append(format_array("endFurnace", solution["endFurnace"]))
    lines.append(format_array("startFullBuffer", solution["startFullBuffer"]))
    lines.append(format_array("endFullBuffer", solution["endFullBuffer"]))
    lines.append(format_array("startDesulf", solution["startDesulf"]))
    lines.append(format_array("endDesulf", solution["endDesulf"]))
    lines.append(format_array("startConverter", solution["startConverter"]))
    lines.append(format_array("endConverter", solution["endConverter"]))
    lines.append(format_array("endTrip", solution["endTrip"]))
    lines.append(f"numMissedConversions = {solution['numMissedConversions']};")
    lines.append(f"numUsedTorpedoes = {solution['numUsedTorpedoes']};")
    lines.append(f"totalDesulfTime = {solution['totalDesulfTime']};")

    return "\n".join(lines)


def main():
    instance_file = "../data/inst_config3_300_200.json"
    output_file = "competition_easy_v2.sol"
    time_limit = 300

    if len(sys.argv) > 1:
        instance_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    if len(sys.argv) > 3:
        time_limit = int(sys.argv[3])

    print(f"Loading: {instance_file}")
    data = load_instance(instance_file)

    print(f"Problem: {data.n} pourings, {data.m} conversions")
    print(f"Time limit: {time_limit}s")

    solver = TorpedoScheduler(data, max_torpedoes=50)
    solution = solver.solve(time_limit=time_limit, verbose=True)

    output = format_solution(solution)
    print(f"\nSolution:\n{output}")

    with open(output_file, "w") as f:
        f.write(output)
    print(f"\nWritten to {output_file}")

    return solution


if __name__ == "__main__":
    main()
