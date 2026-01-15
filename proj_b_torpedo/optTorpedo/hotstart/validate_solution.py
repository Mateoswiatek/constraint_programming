#!/usr/bin/env python3
"""
Validator for torpedo scheduling solution.
Checks all constraints according to problem specification.
"""

import json
import sys
import re

def parse_solution(sol_path):
    """Parse solution file."""
    with open(sol_path, 'r') as f:
        content = f.read()

    sol = {}
    # Parse arrays
    for name in ['assignedTorpedo', 'assignedConversion', 'startTrip', 'startFurnace',
                 'endFurnace', 'startFullBuffer', 'endFullBuffer', 'startDesulf',
                 'endDesulf', 'startConverter', 'endConverter', 'endTrip']:
        match = re.search(rf'{name}\s*=\s*\[(.*?)\];', content, re.DOTALL)
        if match:
            values = match.group(1).replace('\n', '').replace(' ', '')
            sol[name] = [int(x) for x in values.split(',') if x]

    # Parse scalars
    for name in ['numMissedConversions', 'numUsedTorpedoes', 'totalDesulfTime']:
        match = re.search(rf'{name}\s*=\s*(\d+);', content)
        if match:
            sol[name] = int(match.group(1))

    return sol

def load_instance(inst_path):
    """Load instance data."""
    with open(inst_path, 'r') as f:
        return json.load(f)

def check_overlaps(intervals, name):
    """Check for overlapping intervals. Returns list of overlaps."""
    overlaps = []
    # Sort by start time
    sorted_intervals = sorted(enumerate(intervals), key=lambda x: x[1][0])
    for i in range(len(sorted_intervals) - 1):
        idx1, (s1, e1) = sorted_intervals[i]
        idx2, (s2, e2) = sorted_intervals[i + 1]
        if e1 > s2:  # Overlap!
            overlaps.append((idx1, idx2, s1, e1, s2, e2))
    return overlaps

def validate(inst_path, sol_path):
    """Validate solution against instance."""
    data = load_instance(inst_path)
    sol = parse_solution(sol_path)

    n = len(data['metalStartsPouringAt'])
    m = len(data['converterOpensAt'])

    # Extract parameters
    t_pour = data['timeToPourMetal']
    t_desulf = data['timeToDesulfOneLevel']
    t_convert = data['timeToConvert']
    t_seg1 = data['timeToTransferFromEmptyBufferToFurnace']
    t_seg2 = data['timeToTransferFromFurnaceToFullBuffer']
    t_seg3 = data['timeToTransferFromFullBufferToDesulf']
    t_seg4 = data['timeToTransferFromDesulfToConverter']
    t_seg5 = data['timeToTransferFromConverterToEmptyBuffer']
    t_emergency = data['timeToEmergencyTransferFromFurnaceToEmptyBuffer']

    cap_fullbuf = data['numOfSlotsAtFullBuffer']
    cap_desulf = data['numOfSlotsAtDesulf']
    cap_converter = data['numOfSlotsAtConverter']

    metal_starts = data['metalStartsPouringAt']
    sulfur_levels = data['metalSulfurLevels']
    conv_opens = data['converterOpensAt']
    conv_max_sulfur = data['converterMaxSulfurLevels']

    errors = []

    print(f"Instance: {n} pourings, {m} conversions")
    print(f"Timing: pour={t_pour}, desulf/level={t_desulf}, convert={t_convert}")
    print(f"Transport: seg1={t_seg1}, seg2={t_seg2}, seg3={t_seg3}, seg4={t_seg4}, seg5={t_seg5}, emergency={t_emergency}")
    print(f"Capacity: fullbuf={cap_fullbuf}, desulf={cap_desulf}, converter={cap_converter}")
    print()

    # Check array lengths
    for name, arr in sol.items():
        if isinstance(arr, list) and len(arr) != n:
            errors.append(f"Array {name} has length {len(arr)}, expected {n}")

    # Build segments for each pouring
    seg1 = []  # (start, end, pouring_idx)
    seg2_furnace = []  # Furnace occupation
    seg2 = []  # Furnace to full buffer
    seg3 = []  # Full buffer to desulf
    seg4 = []  # Desulf to converter
    seg5 = []  # Converter to empty buffer
    seg6 = []  # Emergency route

    full_buffer_usage = []  # (start, end, pouring_idx)
    desulf_usage = []
    converter_usage = []

    for i in range(n):
        conv_idx = sol['assignedConversion'][i]

        st = sol['startTrip'][i]
        sf = sol['startFurnace'][i]
        ef = sol['endFurnace'][i]
        et = sol['endTrip'][i]

        # Seg1: startTrip -> startFurnace
        seg1.append((st, sf, i))

        # Furnace: startFurnace -> endFurnace
        seg2_furnace.append((sf, ef, i))

        # Check furnace timing matches metal pour
        expected_sf = metal_starts[i]
        expected_ef = expected_sf + t_pour
        if sf != expected_sf:
            errors.append(f"Pouring {i}: startFurnace={sf}, expected {expected_sf}")
        if ef != expected_ef:
            errors.append(f"Pouring {i}: endFurnace={ef}, expected {expected_ef}")

        # Check seg1 timing
        expected_st = sf - t_seg1
        if st != expected_st:
            errors.append(f"Pouring {i}: startTrip={st}, expected {expected_st} (startFurnace - t_seg1)")

        if conv_idx == -1:
            # Emergency route
            # Seg6: endFurnace -> endTrip (should be endFurnace + t_emergency)
            expected_et = ef + t_emergency
            seg6.append((ef, et, i))
            if et != expected_et:
                errors.append(f"Pouring {i} (emergency): endTrip={et}, expected {expected_et}")
        else:
            # Normal route through conversion
            sfb = sol['startFullBuffer'][i]
            efb = sol['endFullBuffer'][i]
            sd = sol['startDesulf'][i]
            ed = sol['endDesulf'][i]
            sc = sol['startConverter'][i]
            ec = sol['endConverter'][i]

            # Check conversion index valid
            if conv_idx < 1 or conv_idx > m:
                errors.append(f"Pouring {i}: invalid conversion index {conv_idx}")
                continue

            # Check sulfur compatibility
            sulfur = sulfur_levels[i]
            max_sulfur = conv_max_sulfur[conv_idx - 1]  # 1-indexed
            desulf_needed = max(0, sulfur - max_sulfur)
            desulf_time = desulf_needed * t_desulf

            # Seg2: endFurnace -> startFullBuffer (should take t_seg2)
            expected_sfb = ef + t_seg2
            if sfb < expected_sfb:
                errors.append(f"Pouring {i}: startFullBuffer={sfb} < expected {expected_sfb}")

            # Full buffer waiting: startFullBuffer -> endFullBuffer (can be 0 or more)
            if efb < sfb:
                errors.append(f"Pouring {i}: endFullBuffer={efb} < startFullBuffer={sfb}")
            full_buffer_usage.append((sfb, efb, i))

            # Seg3: endFullBuffer -> startDesulf (should take t_seg3)
            expected_sd = efb + t_seg3
            if sd < expected_sd:
                errors.append(f"Pouring {i}: startDesulf={sd} < expected {expected_sd}")

            # Desulf: startDesulf -> endDesulf
            expected_ed = sd + desulf_time
            if ed < expected_ed:
                errors.append(f"Pouring {i}: endDesulf={ed} < expected {expected_ed} (desulf_time={desulf_time})")
            desulf_usage.append((sd, ed, i))

            # Seg4: endDesulf -> startConverter (should take t_seg4)
            expected_sc = ed + t_seg4
            if sc < expected_sc:
                errors.append(f"Pouring {i}: startConverter={sc} < expected {expected_sc}")

            # Converter timing: torpedo can ARRIVE (startConverter) before converter opens
            # Actual conversion starts at max(arrival, converterOpens)
            conv_opens_at = conv_opens[conv_idx - 1]
            actual_conv_start = max(sc, conv_opens_at)

            # Converter: actual_conv_start -> endConverter
            expected_ec = actual_conv_start + t_convert
            if ec < expected_ec:
                errors.append(f"Pouring {i}: endConverter={ec} < expected {expected_ec}")
            # Track converter cumulative from actual conversion start (not arrival)
            converter_usage.append((actual_conv_start, ec, i))

            # Seg5: endConverter -> endTrip (should take t_seg5)
            expected_et = ec + t_seg5
            if et < expected_et:
                errors.append(f"Pouring {i}: endTrip={et} < expected {expected_et}")

            # Build segment intervals for overlap check
            seg2.append((ef, sfb, i))  # Furnace to full buffer
            seg3.append((efb, sd, i))  # Full buffer exit to desulf entry
            seg4.append((ed, sc, i))   # Desulf exit to converter entry
            seg5.append((ec, et, i))   # Converter exit to empty buffer

    print("=== Checking NoOverlap constraints ===")

    # Check Seg1 overlaps
    overlaps = check_overlaps([(s, e) for s, e, _ in seg1], "Seg1")
    for idx1, idx2, s1, e1, s2, e2 in overlaps:
        errors.append(f"Seg1 overlap: pouring {idx1} ({s1}-{e1}) overlaps pouring {idx2} ({s2}-{e2})")
    print(f"Seg1: {len(overlaps)} overlaps")

    # Check Furnace overlaps
    overlaps = check_overlaps([(s, e) for s, e, _ in seg2_furnace], "Furnace")
    for idx1, idx2, s1, e1, s2, e2 in overlaps:
        errors.append(f"Furnace overlap: pouring {idx1} ({s1}-{e1}) overlaps pouring {idx2} ({s2}-{e2})")
    print(f"Furnace: {len(overlaps)} overlaps")

    # Check Seg2 overlaps (furnace to full buffer)
    overlaps = check_overlaps([(s, e) for s, e, _ in seg2], "Seg2")
    for idx1, idx2, s1, e1, s2, e2 in overlaps:
        errors.append(f"Seg2 overlap: pouring {idx1} ({s1}-{e1}) overlaps pouring {idx2} ({s2}-{e2})")
    print(f"Seg2: {len(overlaps)} overlaps")

    # Check Seg3 overlaps (full buffer to desulf)
    overlaps = check_overlaps([(s, e) for s, e, _ in seg3], "Seg3")
    for idx1, idx2, s1, e1, s2, e2 in overlaps:
        errors.append(f"Seg3 overlap: pouring {idx1} ({s1}-{e1}) overlaps pouring {idx2} ({s2}-{e2})")
    print(f"Seg3: {len(overlaps)} overlaps")

    # Check Seg4 overlaps (desulf to converter)
    overlaps = check_overlaps([(s, e) for s, e, _ in seg4], "Seg4")
    for idx1, idx2, s1, e1, s2, e2 in overlaps:
        errors.append(f"Seg4 overlap: pouring {idx1} ({s1}-{e1}) overlaps pouring {idx2} ({s2}-{e2})")
    print(f"Seg4: {len(overlaps)} overlaps")

    # Seg5 (converter to empty buffer) - IS NoOverlap per documentation
    overlaps = check_overlaps([(s, e) for s, e, _ in seg5], "Seg5")
    for idx1, idx2, s1, e1, s2, e2 in overlaps:
        errors.append(f"Seg5 overlap: pouring {idx1} ({s1}-{e1}) overlaps pouring {idx2} ({s2}-{e2})")
    print(f"Seg5: {len(overlaps)} overlaps")

    # Check Seg6 overlaps (emergency)
    overlaps = check_overlaps([(s, e) for s, e, _ in seg6], "Seg6")
    for idx1, idx2, s1, e1, s2, e2 in overlaps:
        p1 = seg6[idx1][2]
        p2 = seg6[idx2][2]
        errors.append(f"Seg6 overlap: pouring {p1} ({s1}-{e1}) overlaps pouring {p2} ({s2}-{e2})")
    print(f"Seg6 (emergency): {len(overlaps)} overlaps")

    print("\n=== Checking Cumulative constraints ===")

    # Check Full Buffer cumulative
    def check_cumulative(intervals, capacity, name):
        """Check cumulative constraint."""
        if not intervals:
            return []
        events = []
        for s, e, idx in intervals:
            events.append((s, 1, idx))
            events.append((e, -1, idx))
        events.sort(key=lambda x: (x[0], x[1]))  # Process ends before starts at same time

        violations = []
        current = 0
        active = set()
        for t, delta, idx in events:
            if delta == 1:
                active.add(idx)
            else:
                active.discard(idx)
            current += delta
            if current > capacity:
                violations.append((t, current, list(active)))
        return violations

    violations = check_cumulative(full_buffer_usage, cap_fullbuf, "FullBuffer")
    if violations:
        for t, count, active in violations[:5]:
            errors.append(f"FullBuffer capacity exceeded at t={t}: {count} > {cap_fullbuf}, pourings: {active}")
    print(f"FullBuffer: {len(violations)} capacity violations")

    violations = check_cumulative(desulf_usage, cap_desulf, "Desulf")
    if violations:
        for t, count, active in violations[:5]:
            errors.append(f"Desulf capacity exceeded at t={t}: {count} > {cap_desulf}, pourings: {active}")
    print(f"Desulf: {len(violations)} capacity violations")

    violations = check_cumulative(converter_usage, cap_converter, "Converter")
    if violations:
        for t, count, active in violations[:5]:
            errors.append(f"Converter capacity exceeded at t={t}: {count} > {cap_converter}, pourings: {active}")
    print(f"Converter: {len(violations)} capacity violations")

    print("\n=== Checking conversion assignments ===")

    # Check each conversion is used at most once
    conv_usage = {}
    for i, conv_idx in enumerate(sol['assignedConversion']):
        if conv_idx != -1:
            if conv_idx in conv_usage:
                errors.append(f"Conversion {conv_idx} assigned to multiple pourings: {conv_usage[conv_idx]} and {i}")
            else:
                conv_usage[conv_idx] = i

    used_conversions = len(conv_usage)
    missed = m - used_conversions
    print(f"Conversions used: {used_conversions}/{m}, missed: {missed}")

    if sol.get('numMissedConversions', 0) != missed:
        errors.append(f"numMissedConversions={sol.get('numMissedConversions', 0)} but calculated {missed}")

    print("\n=== Summary ===")
    print(f"Total errors: {len(errors)}")

    if errors:
        print("\nFirst 20 errors:")
        for err in errors[:20]:
            print(f"  - {err}")

    return len(errors) == 0

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <instance.json> <solution.sol>")
        sys.exit(1)

    valid = validate(sys.argv[1], sys.argv[2])
    sys.exit(0 if valid else 1)
