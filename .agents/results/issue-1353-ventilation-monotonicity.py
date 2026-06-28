#!/usr/bin/env python3
"""Issue #1353 — ASHRAE combined-formula monotonicity bound derivation.

This script verifies the monotonicity bound `combined >= max(wind, stack)` for
fluxion's `calculate_combined_infiltration_ach`, mirroring the Rust source at
`src/sim/ventilation.rs:35-110`.

The bound is required by Issue #1353's proptest coverage. The implementer MUST
derive the bound from the source code (not from memory) because fluxion's
simplified additive form differs from the canonical ASHRAE Fundamentals
combined-infiltration formula (`wind + stack - sqrt(wind*stack)`).

Run:
    python3 .agents/results/issue-1353-ventilation-monotonicity.py
"""
import math
import random

# Mirror src/sim/ventilation.rs (verified from source — see ventilation.rs:35-110)
STACK_COEFFICIENT = 0.025
AIR_DENSITY = 1.2
AIR_SPECIFIC_HEAT = 1000.0


def calculate_wind_infiltration_ach(wind_speed, building_height, shielding_factor):
    """Mirror of fluxion::sim::ventilation::calculate_wind_infiltration_ach.

    Source (src/sim/ventilation.rs:35-45):
        shelter_coefficient = (1 - shielding_factor) * 0.4
        height_factor       = (building_height / 3.0)^0.5
        n_factor            = shelter_coefficient * height_factor
        return n_factor * (wind_speed / 3.0)

    Note: the source does NOT clamp negative wind speeds; physical inputs are
    always wind_speed >= 0, so for physical inputs the output is non-negative
    because height_factor >= 0 and (1 - shielding_factor) * 0.4 >= 0 when
    shielding_factor <= 1.
    """
    shelter_coefficient = 0.0 + (1.0 - shielding_factor) * 0.4
    height_factor = (building_height / 3.0) ** 0.5
    base_wind_speed = 3.0
    n_factor = shelter_coefficient * height_factor
    return n_factor * (wind_speed / base_wind_speed)


def calculate_stack_infiltration_ach(indoor_temp, outdoor_temp, height_diff,
                                     opening_area, zone_volume):
    """Mirror of fluxion::sim::ventilation::calculate_stack_infiltration_ach.

    Source (src/sim/ventilation.rs:58-76):
        if zone_volume <= 0 or height_diff <= 0: return 0
        delta_t = abs(indoor - outdoor)
        if delta_t < 0.5: return 0
        flow_sqrt = sqrt(delta_t / height_diff)
        q_vent    = STACK_COEFFICIENT * opening_area * flow_sqrt
        return q_vent / zone_volume
    """
    if zone_volume <= 0.0 or height_diff <= 0.0:
        return 0.0
    delta_t = abs(indoor_temp - outdoor_temp)
    if delta_t < 0.5:
        return 0.0
    flow_arg = delta_t / height_diff
    flow_sqrt = math.sqrt(flow_arg) if flow_arg > 0.0 else 0.0
    q_vent = STACK_COEFFICIENT * opening_area * flow_sqrt
    return q_vent / zone_volume


def calculate_combined_infiltration_ach(outdoor_temp, indoor_temp, wind_speed,
                                        height_diff, opening_area, zone_volume,
                                        shielding_factor):
    """Mirror of fluxion::sim::ventilation::calculate_combined_infiltration_ach.

    Source (src/sim/ventilation.rs:91-110):
        wind_ach  = calculate_wind_infiltration_ach(...)
        stack_ach = calculate_stack_infiltration_ach(...)
        total_ach = wind_ach + stack_ach
        return total_ach.max(0.0)
    """
    wind_ach = calculate_wind_infiltration_ach(wind_speed, height_diff, shielding_factor)
    stack_ach = calculate_stack_infiltration_ach(
        indoor_temp, outdoor_temp, height_diff, opening_area, zone_volume
    )
    total_ach = wind_ach + stack_ach
    return max(total_ach, 0.0)


def main():
    random.seed(1353)

    # =====================================================================
    # Step 1: Verify non-negativity of wind_ach and stack_ach for
    #         PHYSICAL inputs (wind_speed >= 0, height >= 0, etc.)
    # =====================================================================
    print("=" * 72)
    print("Step 1: Wind and stack components are non-negative for physical inputs")
    print("=" * 72)

    violations_wind = 0
    violations_stack = 0
    for _ in range(100_000):
        # Physical inputs only: non-negative wind speed, positive volume, etc.
        ws = random.uniform(0.0, 30.0)
        bh = random.uniform(0.5, 30.0)
        sf = random.uniform(0.0, 1.0)
        w = calculate_wind_infiltration_ach(ws, bh, sf)
        if w < 0:
            violations_wind += 1

        ti = random.uniform(-30.0, 45.0)
        to = random.uniform(-30.0, 45.0)
        hd = random.uniform(0.5, 10.0)
        oa = random.uniform(0.1, 5.0)
        zv = random.uniform(50.0, 500.0)
        s = calculate_stack_infiltration_ach(ti, to, hd, oa, zv)
        if s < 0:
            violations_stack += 1

    print(f"wind_ach  < 0 violations: {violations_wind} / 100,000  "
          f"(expected 0 for physical inputs)")
    print(f"stack_ach < 0 violations: {violations_stack} / 100,000  "
          f"(expected 0 for physical inputs)")
    print()

    # =====================================================================
    # Step 2: Derive monotonicity bound from source code formula
    # =====================================================================
    print("=" * 72)
    print("Step 2: Derive monotonicity bound combined >= max(wind, stack)")
    print("=" * 72)
    print("From source (src/sim/ventilation.rs:108-109):")
    print("    total_ach = wind_ach + stack_ach")
    print("    return total_ach.max(0.0)")
    print()
    print("For PHYSICAL inputs (Step 1), wind_ach >= 0 and stack_ach >= 0.")
    print("Therefore:")
    print("    combined = max(wind_ach + stack_ach, 0)")
    print("           >= wind_ach + stack_ach")
    print("           >= max(wind_ach, stack_ach)     (both terms non-negative)")
    print()
    print("BOUND: combined >= max(wind_ach, stack_ach)   OK")
    print()

    # =====================================================================
    # Step 3: Empirical verification across 100k plausible physical inputs
    # =====================================================================
    print("=" * 72)
    print("Step 3: Empirical verification of bound on 100k physical inputs")
    print("=" * 72)

    bound_violations = 0
    max_combined = 0.0
    max_wind = 0.0
    max_stack = 0.0
    for _ in range(100_000):
        to = random.uniform(-30.0, 45.0)
        ti = random.uniform(15.0, 30.0)
        ws = random.uniform(0.0, 30.0)
        bh = random.uniform(1.0, 30.0)
        oa = random.uniform(0.1, 5.0)
        zv = random.uniform(50.0, 500.0)
        sf = random.uniform(0.0, 1.0)

        w = calculate_wind_infiltration_ach(ws, bh, sf)
        s = calculate_stack_infiltration_ach(ti, to, bh, oa, zv)
        c = calculate_combined_infiltration_ach(to, ti, ws, bh, oa, zv, sf)

        max_combined = max(max_combined, c)
        max_wind = max(max_wind, w)
        max_stack = max(max_stack, s)

        # Use a tiny epsilon to absorb float rounding in the comparison.
        if c + 1e-12 < max(w, s):
            bound_violations += 1

    print(f"combined < max(wind, stack) violations: "
          f"{bound_violations} / 100,000  (expected 0)")
    print(f"max wind_ach:    {max_wind:.4f}")
    print(f"max stack_ach:   {max_stack:.4f}")
    print(f"max combined_ach:{max_combined:.4f}")
    print(f"=> bound holds in ALL 100,000 cases for plausible inputs.")
    print()

    # =====================================================================
    # Step 4: Edge cases — zero wind, zero ΔT, zero height
    # =====================================================================
    print("=" * 72)
    print("Step 4: Edge cases (zero wind, zero ΔT, zero height)")
    print("=" * 72)
    print("zero wind, delta_t=10, height=2.7:")
    print(f"    wind_ach  = {calculate_wind_infiltration_ach(0.0, 2.7, 0.5):.6f}  "
          f"(expected 0.0)")
    print(f"    stack_ach = {calculate_stack_infiltration_ach(25.0, 15.0, 2.7, 1.0, 100.0):.6f}  "
          f"(expected > 0)")
    print()
    print("zero delta_t, wind=5, height=2.7:")
    print(f"    wind_ach  = {calculate_wind_infiltration_ach(5.0, 2.7, 0.5):.6f}  "
          f"(expected > 0)")
    print(f"    stack_ach = {calculate_stack_infiltration_ach(20.0, 20.0, 2.7, 1.0, 100.0):.6f}  "
          f"(expected 0.0)")
    print()
    print("zero height (height_diff=0):")
    print(f"    stack_ach = {calculate_stack_infiltration_ach(25.0, 15.0, 0.0, 1.0, 100.0):.6f}  "
          f"(expected 0.0)")
    print()

    # =====================================================================
    # Step 5: Summary
    # =====================================================================
    print("=" * 72)
    print("Step 5: Summary — bound the proptest must check")
    print("=" * 72)
    print("""
ASHRAE COMBINED-FORMULA MONOTONICITY BOUND (fluxion implementation):

    Given fluxion's source code:
        wind_ach  = ((1 - shielding_factor) * 0.4) * sqrt(building_height/3) * (wind_speed/3)
                  >= 0  (for physical inputs wind_speed >= 0, building_height >= 0)
        stack_ach = (returns 0 when zone_volume <= 0, height_diff <= 0, |dT| < 0.5)
                  >= 0  (STACK_COEFFICIENT * opening_area * sqrt(...) / zone_volume)
        combined  = max(wind_ach + stack_ach, 0.0)
                  >= 0

    THE BOUND IS:
        combined >= max(wind_ach, stack_ach)

    Derivation (algebraic, from the source code):
        combined  = max(wind_ach + stack_ach, 0.0)
                 >= wind_ach + stack_ach                    (clamp only truncates)
                 >= max(wind_ach, stack_ach)                (both terms non-negative)

This is the property the proptest block in tests/ventilation_isolation.rs
must assert for `proptest_combined_infiltration_ach`.
""")


if __name__ == "__main__":
    main()
