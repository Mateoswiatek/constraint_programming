#!/usr/bin/env python3
"""
Torpedo Scheduling Solver using OR-Tools CP-SAT
Solves the ACP 2016 challenge variant for steel mill torpedo car scheduling.
"""

import json
import sys

from ortools.sat.python import cp_model


def load_instance(filepath):
    """Load problem instance from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def solve_torpedo_scheduling(data, max_torpedoes=None, time_limit=300):
    """
    Solve the torpedo scheduling problem using CP-SAT.

    Args:
        data: Problem instance dictionary
        max_torpedoes: Maximum number of torpedoes to use (None = use heuristic)
        time_limit: Time limit in seconds

    Returns:
        Solution dictionary or None if no solution found
    """
    # Extract problem parameters
    time_pour = data["timeToPourMetal"]
    time_desulf_one = data["timeToDesulfOneLevel"]
    time_convert = data["timeToConvert"]

    slots_full_buffer = data["numOfSlotsAtFullBuffer"]
    slots_desulf = data["numOfSlotsAtDesulf"]
    slots_converter = data["numOfSlotsAtConverter"]

    t_empty_to_furnace = data["timeToTransferFromEmptyBufferToFurnace"]
    t_furnace_to_full = data["timeToTransferFromFurnaceToFullBuffer"]
    t_full_to_desulf = data["timeToTransferFromFullBufferToDesulf"]
    t_desulf_to_conv = data["timeToTransferFromDesulfToConverter"]
    t_conv_to_empty = data["timeToTransferFromConverterToEmptyBuffer"]
    t_emergency = data["timeToEmergencyTransferFromFurnaceToEmptyBuffer"]

    metal_starts = data["metalStartsPouringAt"]
    metal_sulfur = data["metalSulfurLevels"]
    conv_opens = data["converterOpensAt"]
    conv_max_sulfur = data["converterMaxSulfurLevels"]

    num_pourings = len(metal_starts)
    num_conversions = len(conv_opens)

    # Estimate time horizon
    max_time = max(metal_starts) + 500  # Add buffer for last batch processing
    if conv_opens:
        max_time = max(max_time, max(conv_opens) + time_convert + t_conv_to_empty + 100)

    # Estimate max torpedoes needed if not specified
    if max_torpedoes is None:
        # Heuristic: consider round-trip time
        min_trip_time = (
            t_empty_to_furnace
            + time_pour
            + t_furnace_to_full
            + t_full_to_desulf
            + t_desulf_to_conv
            + time_convert
            + t_conv_to_empty
        )
        avg_gap = (
            (metal_starts[-1] - metal_starts[0]) / num_pourings
            if num_pourings > 1
            else min_trip_time
        )
        max_torpedoes = min(num_pourings, max(4, int(min_trip_time / avg_gap) + 2))

    print(f"Problem: {num_pourings} pourings, {num_conversions} conversions")
    print(f"Using max {max_torpedoes} torpedoes, time horizon {max_time}")

    model = cp_model.CpModel()

    # ==========================================================================
    # DECISION VARIABLES
    # ==========================================================================

    # For each pouring (metal batch): which torpedo handles it (1-indexed)
    torpedo_assignment = [
        model.NewIntVar(1, max_torpedoes, f"torpedo_{p}") for p in range(num_pourings)
    ]

    # For each pouring: which conversion it's assigned to (0-indexed), or -1 for emergency
    conversion_assignment = [
        model.NewIntVar(-1, num_conversions - 1, f"conv_{p}")
        for p in range(num_pourings)
    ]

    # Boolean: is this pouring going to emergency?
    is_emergency = [model.NewBoolVar(f"emergency_{p}") for p in range(num_pourings)]

    # Time variables for each pouring
    start_trip = [
        model.NewIntVar(0, max_time, f"start_trip_{p}") for p in range(num_pourings)
    ]
    start_furnace = [
        model.NewIntVar(0, max_time, f"start_furnace_{p}") for p in range(num_pourings)
    ]
    end_furnace = [
        model.NewIntVar(0, max_time, f"end_furnace_{p}") for p in range(num_pourings)
    ]
    start_full_buffer = [
        model.NewIntVar(-1, max_time, f"start_full_buffer_{p}")
        for p in range(num_pourings)
    ]
    end_full_buffer = [
        model.NewIntVar(-1, max_time, f"end_full_buffer_{p}")
        for p in range(num_pourings)
    ]
    start_desulf = [
        model.NewIntVar(-1, max_time, f"start_desulf_{p}") for p in range(num_pourings)
    ]
    end_desulf = [
        model.NewIntVar(-1, max_time, f"end_desulf_{p}") for p in range(num_pourings)
    ]
    start_converter = [
        model.NewIntVar(-1, max_time, f"start_converter_{p}")
        for p in range(num_pourings)
    ]
    end_converter = [
        model.NewIntVar(-1, max_time, f"end_converter_{p}") for p in range(num_pourings)
    ]
    end_trip = [
        model.NewIntVar(0, max_time, f"end_trip_{p}") for p in range(num_pourings)
    ]

    # Desulfurization time for each pouring
    desulf_time = [
        model.NewIntVar(0, 5 * time_desulf_one, f"desulf_time_{p}")
        for p in range(num_pourings)
    ]

    # ==========================================================================
    # CONSTRAINTS
    # ==========================================================================

    for p in range(num_pourings):
        # Link emergency boolean to conversion assignment
        model.Add(conversion_assignment[p] == -1).OnlyEnforceIf(is_emergency[p])
        model.Add(conversion_assignment[p] >= 0).OnlyEnforceIf(is_emergency[p].Not())

        # Timing: Empty Buffer -> Furnace
        model.Add(start_furnace[p] == start_trip[p] + t_empty_to_furnace)

        # Furnace must wait for metal to start pouring
        model.Add(start_furnace[p] <= metal_starts[p])

        # Furnace duration
        model.Add(end_furnace[p] == metal_starts[p] + time_pour)

        # ----- Normal route (non-emergency) -----
        # Furnace -> Full Buffer -> Desulf -> Converter -> Empty Buffer

        # Full buffer timing (for non-emergency)
        model.Add(
            start_full_buffer[p] == end_furnace[p] + t_furnace_to_full
        ).OnlyEnforceIf(is_emergency[p].Not())

        # Full buffer: can pass through instantly or wait
        model.Add(end_full_buffer[p] >= start_full_buffer[p]).OnlyEnforceIf(
            is_emergency[p].Not()
        )

        # Full buffer -> Desulf
        model.Add(
            start_desulf[p] == end_full_buffer[p] + t_full_to_desulf
        ).OnlyEnforceIf(is_emergency[p].Not())

        # Desulfurization time depends on sulfur level and assigned conversion
        # For now, we'll compute the required desulf level for each conversion
        for c in range(num_conversions):
            conv_assigned = model.NewBoolVar(f"conv_assigned_{p}_{c}")
            model.Add(conversion_assignment[p] == c).OnlyEnforceIf(conv_assigned)
            model.Add(conversion_assignment[p] != c).OnlyEnforceIf(conv_assigned.Not())

            # Required desulf levels to reduce sulfur
            sulfur_reduction_needed = max(0, metal_sulfur[p] - conv_max_sulfur[c])
            required_desulf_time = sulfur_reduction_needed * time_desulf_one

            model.Add(desulf_time[p] >= required_desulf_time).OnlyEnforceIf(
                conv_assigned
            )

        # End desulf
        model.Add(end_desulf[p] == start_desulf[p] + desulf_time[p]).OnlyEnforceIf(
            is_emergency[p].Not()
        )

        # Desulf -> Converter
        model.Add(start_converter[p] == end_desulf[p] + t_desulf_to_conv).OnlyEnforceIf(
            is_emergency[p].Not()
        )

        # Converter constraint: torpedo must arrive BY the due date (before converter opens)
        # Conversion starts when converter opens, ends after time_convert
        for c in range(num_conversions):
            conv_assigned = model.NewBoolVar(f"conv_open_{p}_{c}")
            model.Add(conversion_assignment[p] == c).OnlyEnforceIf(conv_assigned)
            model.Add(conversion_assignment[p] != c).OnlyEnforceIf(conv_assigned.Not())

            # Torpedo must arrive BY the converter due date (arrive before or at opening)
            model.Add(start_converter[p] <= conv_opens[c]).OnlyEnforceIf(conv_assigned)
            # Conversion ends at due_date + convert_time
            model.Add(end_converter[p] == conv_opens[c] + time_convert).OnlyEnforceIf(
                conv_assigned
            )

        # For non-emergency routes, end_converter is set by the conversion constraints above
        # Converter -> Empty Buffer
        model.Add(end_trip[p] == end_converter[p] + t_conv_to_empty).OnlyEnforceIf(
            is_emergency[p].Not()
        )

        # ----- Emergency route -----
        # Set all intermediate times to -1 for emergency
        model.Add(start_full_buffer[p] == -1).OnlyEnforceIf(is_emergency[p])
        model.Add(end_full_buffer[p] == -1).OnlyEnforceIf(is_emergency[p])
        model.Add(start_desulf[p] == -1).OnlyEnforceIf(is_emergency[p])
        model.Add(end_desulf[p] == -1).OnlyEnforceIf(is_emergency[p])
        model.Add(start_converter[p] == -1).OnlyEnforceIf(is_emergency[p])
        model.Add(end_converter[p] == -1).OnlyEnforceIf(is_emergency[p])
        model.Add(desulf_time[p] == 0).OnlyEnforceIf(is_emergency[p])

        # Emergency route timing
        model.Add(end_trip[p] == end_furnace[p] + t_emergency).OnlyEnforceIf(
            is_emergency[p]
        )

    # ----- Each conversion can be assigned to at most one pouring -----
    for c in range(num_conversions):
        conv_vars = []
        for p in range(num_pourings):
            is_assigned = model.NewBoolVar(f"conv_{c}_to_pour_{p}")
            model.Add(conversion_assignment[p] == c).OnlyEnforceIf(is_assigned)
            model.Add(conversion_assignment[p] != c).OnlyEnforceIf(is_assigned.Not())
            conv_vars.append(is_assigned)
        model.Add(sum(conv_vars) <= 1)

    # ----- Furnace capacity (only one torpedo at a time) -----
    # Using interval variables for no-overlap constraint
    # Furnace interval spans from arrival to departure
    furnace_intervals = []
    furnace_sizes = []
    for p in range(num_pourings):
        # Size of furnace stay = end_furnace - start_furnace
        furn_size = model.NewIntVar(0, max_time, f"furn_size_{p}")
        model.Add(furn_size == end_furnace[p] - start_furnace[p])
        furnace_sizes.append(furn_size)

        interval = model.NewIntervalVar(
            start_furnace[p], furn_size, end_furnace[p], f"furnace_interval_{p}"
        )
        furnace_intervals.append(interval)
    model.AddNoOverlap(furnace_intervals)

    # ----- Railway segment constraints (one torpedo at a time per segment) -----
    # Segment 1: Empty Buffer -> Furnace
    seg1_intervals = []
    for p in range(num_pourings):
        seg1_intervals.append(
            model.NewIntervalVar(
                start_trip[p], t_empty_to_furnace, start_furnace[p], f"seg1_{p}"
            )
        )
    model.AddNoOverlap(seg1_intervals)

    # Segment 2: Furnace -> Full Buffer (non-emergency only)
    seg2_intervals = []
    for p in range(num_pourings):
        seg2_start = model.NewIntVar(0, max_time, f"seg2_start_{p}")
        seg2_end = model.NewIntVar(0, max_time, f"seg2_end_{p}")
        seg2_size = model.NewIntVar(0, t_furnace_to_full, f"seg2_size_{p}")

        # For non-emergency: normal segment
        model.Add(seg2_start == end_furnace[p]).OnlyEnforceIf(is_emergency[p].Not())
        model.Add(seg2_size == t_furnace_to_full).OnlyEnforceIf(is_emergency[p].Not())
        # For emergency: zero-length interval
        model.Add(seg2_size == 0).OnlyEnforceIf(is_emergency[p])
        model.Add(seg2_start == 0).OnlyEnforceIf(is_emergency[p])

        model.Add(seg2_end == seg2_start + seg2_size)

        interval = model.NewIntervalVar(
            seg2_start, seg2_size, seg2_end, f"seg2_interval_{p}"
        )
        seg2_intervals.append(interval)
    model.AddNoOverlap(seg2_intervals)

    # Segment 3: Full Buffer -> Desulf (non-emergency only)
    seg3_intervals = []
    for p in range(num_pourings):
        seg3_start = model.NewIntVar(0, max_time, f"seg3_start_{p}")
        seg3_end = model.NewIntVar(0, max_time, f"seg3_end_{p}")
        seg3_size = model.NewIntVar(0, t_full_to_desulf, f"seg3_size_{p}")

        model.Add(seg3_start == end_full_buffer[p]).OnlyEnforceIf(is_emergency[p].Not())
        model.Add(seg3_size == t_full_to_desulf).OnlyEnforceIf(is_emergency[p].Not())
        model.Add(seg3_size == 0).OnlyEnforceIf(is_emergency[p])
        model.Add(seg3_start == 0).OnlyEnforceIf(is_emergency[p])

        model.Add(seg3_end == seg3_start + seg3_size)

        interval = model.NewIntervalVar(
            seg3_start, seg3_size, seg3_end, f"seg3_interval_{p}"
        )
        seg3_intervals.append(interval)
    model.AddNoOverlap(seg3_intervals)

    # Segment 4: Desulf -> Converter (non-emergency only)
    seg4_intervals = []
    for p in range(num_pourings):
        seg4_start = model.NewIntVar(0, max_time, f"seg4_start_{p}")
        seg4_end = model.NewIntVar(0, max_time, f"seg4_end_{p}")
        seg4_size = model.NewIntVar(0, t_desulf_to_conv, f"seg4_size_{p}")

        model.Add(seg4_start == end_desulf[p]).OnlyEnforceIf(is_emergency[p].Not())
        model.Add(seg4_size == t_desulf_to_conv).OnlyEnforceIf(is_emergency[p].Not())
        model.Add(seg4_size == 0).OnlyEnforceIf(is_emergency[p])
        model.Add(seg4_start == 0).OnlyEnforceIf(is_emergency[p])

        model.Add(seg4_end == seg4_start + seg4_size)

        interval = model.NewIntervalVar(
            seg4_start, seg4_size, seg4_end, f"seg4_interval_{p}"
        )
        seg4_intervals.append(interval)
    model.AddNoOverlap(seg4_intervals)

    # Segment 5: Converter -> Empty Buffer (non-emergency only)
    seg5_intervals = []
    for p in range(num_pourings):
        seg5_start = model.NewIntVar(0, max_time, f"seg5_start_{p}")
        seg5_end = model.NewIntVar(0, max_time, f"seg5_end_{p}")
        seg5_size = model.NewIntVar(0, t_conv_to_empty, f"seg5_size_{p}")

        model.Add(seg5_start == end_converter[p]).OnlyEnforceIf(is_emergency[p].Not())
        model.Add(seg5_size == t_conv_to_empty).OnlyEnforceIf(is_emergency[p].Not())
        model.Add(seg5_size == 0).OnlyEnforceIf(is_emergency[p])
        model.Add(seg5_start == 0).OnlyEnforceIf(is_emergency[p])

        model.Add(seg5_end == seg5_start + seg5_size)

        interval = model.NewIntervalVar(
            seg5_start, seg5_size, seg5_end, f"seg5_interval_{p}"
        )
        seg5_intervals.append(interval)
    model.AddNoOverlap(seg5_intervals)

    # Segment 6: Emergency route (Furnace -> Empty Buffer)
    seg6_intervals = []
    for p in range(num_pourings):
        seg6_start = model.NewIntVar(0, max_time, f"seg6_start_{p}")
        seg6_end = model.NewIntVar(0, max_time, f"seg6_end_{p}")
        seg6_size = model.NewIntVar(0, t_emergency, f"seg6_size_{p}")

        model.Add(seg6_start == end_furnace[p]).OnlyEnforceIf(is_emergency[p])
        model.Add(seg6_size == t_emergency).OnlyEnforceIf(is_emergency[p])
        model.Add(seg6_size == 0).OnlyEnforceIf(is_emergency[p].Not())
        model.Add(seg6_start == 0).OnlyEnforceIf(is_emergency[p].Not())

        model.Add(seg6_end == seg6_start + seg6_size)

        interval = model.NewIntervalVar(
            seg6_start, seg6_size, seg6_end, f"seg6_interval_{p}"
        )
        seg6_intervals.append(interval)
    model.AddNoOverlap(seg6_intervals)

    # ----- Capacity constraints at facilities -----
    # Full Buffer cumulative constraint
    full_buffer_demands = []
    full_buffer_intervals = []
    for p in range(num_pourings):
        fb_start = model.NewIntVar(0, max_time, f"fb_cum_start_{p}")
        fb_end = model.NewIntVar(0, max_time, f"fb_cum_end_{p}")
        fb_size = model.NewIntVar(0, max_time, f"fb_cum_size_{p}")
        fb_demand = model.NewIntVar(0, 1, f"fb_demand_{p}")

        # Non-emergency: actual duration at full buffer
        model.Add(fb_start == start_full_buffer[p]).OnlyEnforceIf(is_emergency[p].Not())
        model.Add(fb_end == end_full_buffer[p]).OnlyEnforceIf(is_emergency[p].Not())
        model.Add(fb_demand == 1).OnlyEnforceIf(is_emergency[p].Not())

        # Emergency: zero demand
        model.Add(fb_demand == 0).OnlyEnforceIf(is_emergency[p])
        model.Add(fb_start == 0).OnlyEnforceIf(is_emergency[p])
        model.Add(fb_end == 0).OnlyEnforceIf(is_emergency[p])

        model.Add(fb_size == fb_end - fb_start)

        interval = model.NewIntervalVar(fb_start, fb_size, fb_end, f"fb_interval_{p}")
        full_buffer_intervals.append(interval)
        full_buffer_demands.append(fb_demand)

    model.AddCumulative(full_buffer_intervals, full_buffer_demands, slots_full_buffer)

    # Desulf cumulative constraint
    desulf_demands = []
    desulf_intervals = []
    for p in range(num_pourings):
        ds_start = model.NewIntVar(0, max_time, f"ds_cum_start_{p}")
        ds_end = model.NewIntVar(0, max_time, f"ds_cum_end_{p}")
        ds_size = model.NewIntVar(0, max_time, f"ds_cum_size_{p}")
        ds_demand = model.NewIntVar(0, 1, f"ds_demand_{p}")

        model.Add(ds_start == start_desulf[p]).OnlyEnforceIf(is_emergency[p].Not())
        model.Add(ds_end == end_desulf[p]).OnlyEnforceIf(is_emergency[p].Not())
        model.Add(ds_demand == 1).OnlyEnforceIf(is_emergency[p].Not())

        model.Add(ds_demand == 0).OnlyEnforceIf(is_emergency[p])
        model.Add(ds_start == 0).OnlyEnforceIf(is_emergency[p])
        model.Add(ds_end == 0).OnlyEnforceIf(is_emergency[p])

        model.Add(ds_size == ds_end - ds_start)

        interval = model.NewIntervalVar(ds_start, ds_size, ds_end, f"ds_interval_{p}")
        desulf_intervals.append(interval)
        desulf_demands.append(ds_demand)

    model.AddCumulative(desulf_intervals, desulf_demands, slots_desulf)

    # Converter cumulative constraint
    conv_demands = []
    conv_intervals = []
    for p in range(num_pourings):
        cv_start = model.NewIntVar(0, max_time, f"cv_cum_start_{p}")
        cv_end = model.NewIntVar(0, max_time, f"cv_cum_end_{p}")
        cv_size = model.NewIntVar(0, max_time, f"cv_cum_size_{p}")
        cv_demand = model.NewIntVar(0, 1, f"cv_demand_{p}")

        model.Add(cv_start == start_converter[p]).OnlyEnforceIf(is_emergency[p].Not())
        model.Add(cv_end == end_converter[p]).OnlyEnforceIf(is_emergency[p].Not())
        model.Add(cv_demand == 1).OnlyEnforceIf(is_emergency[p].Not())

        model.Add(cv_demand == 0).OnlyEnforceIf(is_emergency[p])
        model.Add(cv_start == 0).OnlyEnforceIf(is_emergency[p])
        model.Add(cv_end == 0).OnlyEnforceIf(is_emergency[p])

        model.Add(cv_size == cv_end - cv_start)

        interval = model.NewIntervalVar(cv_start, cv_size, cv_end, f"cv_interval_{p}")
        conv_intervals.append(interval)
        conv_demands.append(cv_demand)

    model.AddCumulative(conv_intervals, conv_demands, slots_converter)

    # ----- Torpedo reuse constraint -----
    # If the same torpedo handles two pourings, the second trip must start after the first ends
    for p1 in range(num_pourings):
        for p2 in range(p1 + 1, num_pourings):
            same_torpedo = model.NewBoolVar(f"same_torp_{p1}_{p2}")
            model.Add(torpedo_assignment[p1] == torpedo_assignment[p2]).OnlyEnforceIf(
                same_torpedo
            )
            model.Add(torpedo_assignment[p1] != torpedo_assignment[p2]).OnlyEnforceIf(
                same_torpedo.Not()
            )

            # If same torpedo, the later pouring's trip must start after the earlier one ends
            # Since metal_starts are sorted, p1 happens before p2
            model.Add(start_trip[p2] >= end_trip[p1]).OnlyEnforceIf(same_torpedo)

    # ==========================================================================
    # OBJECTIVE
    # ==========================================================================

    # Count missed conversions (conversions not assigned to any pouring)
    conversion_used = []
    for c in range(num_conversions):
        used = model.NewBoolVar(f"conv_used_{c}")
        conv_vars = []
        for p in range(num_pourings):
            is_assigned = model.NewBoolVar(f"obj_conv_{c}_to_{p}")
            model.Add(conversion_assignment[p] == c).OnlyEnforceIf(is_assigned)
            model.Add(conversion_assignment[p] != c).OnlyEnforceIf(is_assigned.Not())
            conv_vars.append(is_assigned)
        model.Add(sum(conv_vars) >= 1).OnlyEnforceIf(used)
        model.Add(sum(conv_vars) == 0).OnlyEnforceIf(used.Not())
        conversion_used.append(used)

    num_missed_conversions = num_conversions - sum(conversion_used)

    # Count used torpedoes
    torpedo_used = []
    for t in range(1, max_torpedoes + 1):
        used = model.NewBoolVar(f"torp_used_{t}")
        torp_vars = []
        for p in range(num_pourings):
            is_assigned = model.NewBoolVar(f"obj_torp_{t}_to_{p}")
            model.Add(torpedo_assignment[p] == t).OnlyEnforceIf(is_assigned)
            model.Add(torpedo_assignment[p] != t).OnlyEnforceIf(is_assigned.Not())
            torp_vars.append(is_assigned)
        model.Add(sum(torp_vars) >= 1).OnlyEnforceIf(used)
        model.Add(sum(torp_vars) == 0).OnlyEnforceIf(used.Not())
        torpedo_used.append(used)

        # ----- Symmetry breaking: torpedoes must be used in order -----
    # If torpedo t+1 is used, then torpedo t must also be used
    # This eliminates redundant permutations of torpedo labels
    for t in range(1, max_torpedoes):
        # torpedo_used[t] represents torpedo t+1 (0-indexed list)
        # If torpedo t+1 is used, torpedo t must be used
        model.AddImplication(torpedo_used[t], torpedo_used[t - 1])

    # ----- Optional hard constraint: no missed conversions -----
    # for c in range(num_conversions):
    #     model.Add(conversion_used[c] == 1)

    num_used_torpedoes = sum(torpedo_used)

    # Total desulfurization time
    total_desulf_time = sum(desulf_time)

    # Lexicographic objective: minimize missed_conversions, then torpedoes, then desulf_time
    # Use large fixed weights to guarantee proper ordering:
    # - Any improvement in missed conversions beats any torpedo/desulf combination
    # - Any improvement in torpedoes beats any desulf improvement
    #
    # Example: (0 missed, 1 torpedo, 1_000_000 desulf) must be better than (0 missed, 2 torpedoes, 0 desulf)
    DESULF_WEIGHT = 1
    TORPEDO_WEIGHT = 10**9  # 1 torpedo improvement > any desulf improvement
    MISSED_WEIGHT = 10**12  # 1 missed conversion improvement > any torpedo+desulf combo

    objective = (
        num_missed_conversions * MISSED_WEIGHT
        + num_used_torpedoes * TORPEDO_WEIGHT
        + total_desulf_time * DESULF_WEIGHT
    )

    model.Minimize(objective)

    # ==========================================================================
    # SOLVE
    # ==========================================================================

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8
    solver.parameters.log_search_progress = True

    print(f"\nSolving with time limit {time_limit}s...")
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(
            f"\nSolution found! Status: {'OPTIMAL' if status == cp_model.OPTIMAL else 'FEASIBLE'}"
        )

        # Extract solution
        solution = {
            "assignedTorpedo": [
                solver.Value(torpedo_assignment[p]) for p in range(num_pourings)
            ],
            "assignedConversion": [
                solver.Value(conversion_assignment[p]) + 1
                if solver.Value(conversion_assignment[p]) >= 0
                else -1
                for p in range(num_pourings)
            ],  # Convert to 1-indexed
            "startTrip": [solver.Value(start_trip[p]) for p in range(num_pourings)],
            "startFurnace": [
                solver.Value(start_furnace[p]) for p in range(num_pourings)
            ],
            "endFurnace": [solver.Value(end_furnace[p]) for p in range(num_pourings)],
            "startFullBuffer": [
                solver.Value(start_full_buffer[p]) for p in range(num_pourings)
            ],
            "endFullBuffer": [
                solver.Value(end_full_buffer[p]) for p in range(num_pourings)
            ],
            "startDesulf": [solver.Value(start_desulf[p]) for p in range(num_pourings)],
            "endDesulf": [solver.Value(end_desulf[p]) for p in range(num_pourings)],
            "startConverter": [
                solver.Value(start_converter[p]) for p in range(num_pourings)
            ],
            "endConverter": [
                solver.Value(end_converter[p]) for p in range(num_pourings)
            ],
            "endTrip": [solver.Value(end_trip[p]) for p in range(num_pourings)],
            "numMissedConversions": sum(
                1 for c in conversion_used if solver.Value(c) == 0
            ),
            "numUsedTorpedoes": sum(1 for t in torpedo_used if solver.Value(t) == 1),
            "totalDesulfTime": sum(
                solver.Value(desulf_time[p]) for p in range(num_pourings)
            ),
        }

        print("\nObjective values:")
        print(f"  Missed conversions: {solution['numMissedConversions']}")
        print(f"  Used torpedoes: {solution['numUsedTorpedoes']}")
        print(f"  Total desulf time: {solution['totalDesulfTime']}")

        return solution
    else:
        print(f"\nNo solution found. Status: {status}")
        return None


def format_solution(solution):
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
    # Default to easy competition instance
    instance_file = "data/inst_config3_300_200.json"
    output_file = "competition_easy.sol"
    time_limit = 300
    max_torpedoes = 6  # Start with slightly more than best known (4)

    if len(sys.argv) > 1:
        instance_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    if len(sys.argv) > 3:
        time_limit = int(sys.argv[3])
    if len(sys.argv) > 4:
        max_torpedoes = int(sys.argv[4])

    print(f"Loading instance: {instance_file}")
    data = load_instance(instance_file)

    solution = solve_torpedo_scheduling(
        data, max_torpedoes=max_torpedoes, time_limit=time_limit
    )

    if solution:
        output = format_solution(solution)
        print(f"\nSolution:\n{output}")

        with open(output_file, "w") as f:
            f.write(output)
        print(f"\nSolution written to {output_file}")
    else:
        print("\nFailed to find a solution!")
        sys.exit(1)


if __name__ == "__main__":
    main()
