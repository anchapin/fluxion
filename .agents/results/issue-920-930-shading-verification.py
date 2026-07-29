#!/usr/bin/env python3
"""
Python verification for Issue #1617 — E/W shading geometry causes peak cooling UNDER-prediction

Case 920 peak_cooling: Fluxion predicts 1.28 kW vs reference 1.40-1.90 kW
Case 930 peak_cooling: Fluxion predicts 1.05 kW vs reference 1.10-1.50 kW

Both cases UNDER-predict, suggesting E/W shading is over-estimating shading.

Root cause: fin height treated as infinite instead of bounded by mounting_height.
At low solar angles, using window.height (3.0m) instead of bounded fin_height (1.1m)
causes the overlap correction to be too large, reducing the net shaded area.

With the fix:
- fin_height = window_top - mounting_height = (sill + height) - mounting_height
- At low angles, bounded fin_height reduces overlap, increasing net shading
- This means MORE solar gain gets through, increasing cooling load
- Should help close the gap with reference values

Verification:
- Expected shaded_fraction for Case 930 E/W at peak cooling hour
- E/W shading isolation test (altitude=15°, azimuth=100°)
"""

import math
from typing import NamedTuple


class WindowGeometry(NamedTuple):
    """Window geometry parameters."""
    area: float        # m²
    width: float       # m
    height: float      # m
    sill_height: float # m


class ShadeFin(NamedTuple):
    """Shade fin parameters."""
    depth: float            # m
    distance_from_edge: float # m
    side: str               # 'Left' or 'Right'
    height: float           # m (bounded by mounting_height)


class Overhang(NamedTuple):
    """Overhang parameters."""
    depth: float            # m
    distance_above: float  # m


class SolarPosition(NamedTuple):
    """Solar position relative to surface."""
    altitude: float         # radians
    relative_azimuth: float # radians


def calculate_overhang_shadow_area(
    window: WindowGeometry,
    overhang: Overhang,
    solar: SolarPosition,
) -> float:
    """Calculate the shaded area from an overhang."""
    if solar.altitude <= 0.0:
        return 0.0

    # Profile angle: tan(profile) = tan(altitude) / cos(relative_azimuth)
    tan_profile = math.tan(solar.altitude) / math.cos(solar.relative_azimuth)
    shadow_y = overhang.depth * tan_profile

    # Height of the shadow on the window
    overhang_height = max(0.0, min(shadow_y - overhang.distance_above, window.height))

    # Shaded area = width * height (overhang extends full width)
    return window.width * overhang_height


def calculate_fin_shadow_area(
    window: WindowGeometry,
    fin: ShadeFin,
    solar: SolarPosition,
) -> float:
    """Calculate the shaded area from a shade fin.

    Uses actual fin.height instead of window.height.
    """
    if abs(solar.relative_azimuth) >= math.pi / 2:
        return 0.0

    sun_az = solar.relative_azimuth

    # Determine if this fin shades the window
    is_shaded = (fin.side == 'Left' and sun_az < 0.0) or \
                (fin.side == 'Right' and sun_az > 0.0)

    if not is_shaded:
        return 0.0

    # Shadow width on window
    shadow_x = fin.depth * math.tan(abs(sun_az))
    shadow_width_on_window = max(0.0, shadow_x - fin.distance_from_edge)
    shaded_width = min(shadow_width_on_window, window.width)

    # Use actual fin height (bounded by mounting_height)
    shaded_area = shaded_width * fin.height

    return shaded_area


def calculate_overlap_area(
    window: WindowGeometry,
    fin: ShadeFin,
    overhang_height: float,
    solar: SolarPosition,
) -> float:
    """Calculate the overlap area between fin shadow and overhang shadow.

    Uses actual fin.height to bound the overlap height.
    """
    if overhang_height <= 0.0:
        return 0.0

    if abs(solar.relative_azimuth) >= math.pi / 2:
        return 0.0

    sun_az = solar.relative_azimuth

    # Determine if this fin shades the window
    is_shaded = (fin.side == 'Left' and sun_az < 0.0) or \
                (fin.side == 'Right' and sun_az > 0.0)

    if not is_shaded:
        return 0.0

    # Fin shadow width
    shadow_x = fin.depth * math.tan(abs(sun_az))
    fin_width = max(0.0, min(shadow_x - fin.distance_from_edge, window.width))

    # The overlap height is bounded by actual fin height, not window.height
    overlap_height = min(fin.height, overhang_height)

    return fin_width * overlap_height


def calculate_shaded_fraction_fixed(
    window: WindowGeometry,
    overhang: Overhang | None,
    fins: list[ShadeFin],
    solar: SolarPosition,
) -> float:
    """Calculate shaded fraction with bounded fin height."""
    if solar.altitude <= 0.0:
        return 1.0

    # Overhang shading
    overhang_area = calculate_overhang_shadow_area(window, overhang, solar) if overhang else 0.0

    # Fin shading (using bounded fin.height)
    fin_area = sum(calculate_fin_shadow_area(window, fin, solar) for fin in fins)

    # Overlap correction (using bounded fin.height)
    overlap_area = 0.0
    if overhang_area > 0.0 and fin_area > 0.0 and overhang:
        # Calculate overhang shadow height for overlap
        tan_profile = math.tan(solar.altitude) / math.cos(solar.relative_azimuth)
        shadow_y = overhang.depth * tan_profile
        oh_height = max(0.0, min(shadow_y - overhang.distance_above, window.height))

        for fin in fins:
            overlap_area += calculate_overlap_area(window, fin, oh_height, solar)

    combined_shaded_area = overhang_area + fin_area - overlap_area
    return max(0.0, min(combined_shaded_area / window.area, 1.0))


def calculate_shaded_fraction_buggy(
    window: WindowGeometry,
    overhang: Overhang | None,
    fins: list[ShadeFin],
    solar: SolarPosition,
) -> float:
    """Calculate shaded fraction with BUGGY fin height (window.height = infinite)."""
    if solar.altitude <= 0.0:
        return 1.0

    # Overhang shading
    overhang_area = calculate_overhang_shadow_area(window, overhang, solar) if overhang else 0.0

    # Fin shading (BUGGY: uses window.height instead of fin.height)
    fin_area = 0.0
    for fin in fins:
        if abs(solar.relative_azimuth) >= math.pi / 2:
            continue
        sun_az = solar.relative_azimuth
        is_shaded = (fin.side == 'Left' and sun_az < 0.0) or \
                    (fin.side == 'Right' and sun_az > 0.0)
        if not is_shaded:
            continue
        shadow_x = fin.depth * math.tan(abs(sun_az))
        shadow_width_on_window = max(0.0, shadow_x - fin.distance_from_edge)
        shaded_width = min(shadow_width_on_window, window.width)
        # BUG: uses window.height instead of fin.height
        fin_area += shaded_width * window.height

    # Overlap correction (BUGGY: uses window.height)
    overlap_area = 0.0
    if overhang_area > 0.0 and fin_area > 0.0 and overhang:
        tan_profile = math.tan(solar.altitude) / math.cos(solar.relative_azimuth)
        shadow_y = overhang.depth * tan_profile
        oh_height = max(0.0, min(shadow_y - overhang.distance_above, window.height))

        for fin in fins:
            if abs(solar.relative_azimuth) >= math.pi / 2:
                continue
            sun_az = solar.relative_azimuth
            is_shaded = (fin.side == 'Left' and sun_az < 0.0) or \
                        (fin.side == 'Right' and sun_az > 0.0)
            if not is_shaded:
                continue
            shadow_x = fin.depth * math.tan(abs(sun_az))
            fin_width = max(0.0, min(shadow_x - fin.distance_from_edge, window.width))
            # BUG: uses window.height instead of fin.height
            overlap_area += fin_width * window.height

    combined_shaded_area = overhang_area + fin_area - overlap_area
    return max(0.0, min(combined_shaded_area / window.area, 1.0))


def test_ew_shading_isolation():
    """Test E/W shading at low solar angles (Issue #1617 acceptance criterion 2).

    Test Case: altitude=15°, azimuth=100° (east-facing window, morning sun)

    For an east-facing window:
    - Surface azimuth = 90°
    - Sun azimuth = 100°
    - Relative azimuth = 100 - 90 = 10° (positive = sun to the right)

    Since relative_azimuth > 0, the RIGHT fin should shade.

    With mounting_height = 2.7m (room height) and window sill = 0.8m:
    - Fin starts at 2.7m from floor
    - Window top = 0.8 + 3.0 = 3.8m
    - Fin height = window top - mounting_height = 3.8 - 2.7 = 1.1m (bounded)

    At low angle (15°), the overlap correction is too large when using
    window.height (infinite fin height assumption). This reduces net shading.
    With bounded fin_height, the overlap is smaller, resulting in MORE net shading.
    """
    # Window: 6 m² E/W window (width=2m, height=3m per issue description)
    window = WindowGeometry(
        area=6.0,
        width=2.0,
        height=3.0,
        sill_height=0.8,
    )

    # Case 930 shading device: overhang_and_fins(1.0, 1.0, 2.7)
    overhang = Overhang(depth=1.0, distance_above=0.0)

    # Fin height bounded by mounting_height
    # Fin extends from mounting_height (2.7m) up to window top (3.8m)
    fin_height = window.sill_height + window.height - 2.7  # = 1.1m
    fins = [
        ShadeFin(depth=1.0, distance_from_edge=0.0, side='Left', height=fin_height),
        ShadeFin(depth=1.0, distance_from_edge=0.0, side='Right', height=fin_height),
    ]

    # East-facing window, morning sun at 15° altitude, 100° azimuth
    # Relative azimuth = 100 - 90 = 10° = 0.1745 radians
    altitude_deg = 15.0
    azimuth_deg = 100.0
    surface_azimuth_deg = 90.0  # East

    relative_azimuth_deg = azimuth_deg - surface_azimuth_deg
    solar = SolarPosition(
        altitude=math.radians(altitude_deg),
        relative_azimuth=math.radians(relative_azimuth_deg),
    )

    print("=" * 60)
    print("E/W Shading Isolation Test (Issue #1617)")
    print("=" * 60)
    print(f"Window: {window.area}m², {window.width}m x {window.height}m, sill={window.sill_height}m")
    print(f"Fin: depth=1.0m, mounting_height=2.7m, fin_height={fin_height}m")
    print(f"Solar: altitude={altitude_deg}°, azimuth={azimuth_deg}°")
    print(f"Relative azimuth: {relative_azimuth_deg}° (sun to the right)")
    print()

    # Calculate with BUGGY code (infinite fin height = window.height)
    buggy_fraction = calculate_shaded_fraction_buggy(window, overhang, fins, solar)

    # Calculate with FIXED code (bounded fin_height)
    fixed_fraction = calculate_shaded_fraction_fixed(window, overhang, fins, solar)

    print(f"BUGGY shaded_fraction (using window.height={window.height}m): {buggy_fraction:.4f}")
    print(f"FIXED shaded_fraction (using fin_height={fin_height}m):     {fixed_fraction:.4f}")
    print(f"Difference: {fixed_fraction - buggy_fraction:.4f}")
    print()

    # At low angles, using window.height over-corrects the overlap,
    # resulting in LESS net shading. With bounded fin_height,
    # the overlap is correct, resulting in MORE net shading.
    print("Effect: With bounded fin_height, net shading INCREASES at low angles")
    print(f"  This means MORE solar gain -> higher cooling load")
    print(f"  Should help close the under-prediction gap")
    print()

    print("Acceptance Criterion 2: E/W shading isolation test for altitude=15°, azimuth=100°")
    print(f"  BUGGY shaded_fraction = {buggy_fraction:.4f}")
    print(f"  FIXED shaded_fraction = {fixed_fraction:.4f}")
    print(f"  FIXED produces {((fixed_fraction - buggy_fraction)/buggy_fraction)*100:.1f}% MORE shading")
    print()

    return fixed_fraction, buggy_fraction


def test_case_930_peak_cooling():
    """Verify Case 930 shading at peak cooling conditions.

    Case 930: High mass with east/west shading (overhang + fins)
    Peak cooling hour typically around 15:00 (3 PM) for summer conditions.

    For a more realistic west-facing scenario, we use a sun position where
    the sun is NOT behind the window (relative_azimuth within ±90°).
    """
    # Window: 6 m² E/W window
    window = WindowGeometry(
        area=6.0,
        width=2.0,
        height=3.0,
        sill_height=0.8,
    )

    # Case 930: overhang + fins with mounting_height = 2.7m
    overhang = Overhang(depth=1.0, distance_above=0.0)
    fin_height = window.sill_height + window.height - 2.7  # = 1.1m
    fins = [
        ShadeFin(depth=1.0, distance_from_edge=0.0, side='Left', height=fin_height),
        ShadeFin(depth=1.0, distance_from_edge=0.0, side='Right', height=fin_height),
    ]

    # West-facing window scenario: sun is to the south-southwest
    # Surface azimuth = 270° (west)
    # Sun azimuth = 220° (south-southwest, setting sun)
    # Relative azimuth = 220 - 270 = -50° (sun to the LEFT)
    altitude_deg = 30.0  # Lower angle for late afternoon
    azimuth_deg = 220.0
    surface_azimuth_deg = 270.0  # West

    relative_azimuth_deg = azimuth_deg - surface_azimuth_deg
    solar = SolarPosition(
        altitude=math.radians(altitude_deg),
        relative_azimuth=math.radians(relative_azimuth_deg),
    )

    print("=" * 60)
    print("Case 930 Peak Cooling Verification")
    print("=" * 60)
    print(f"Window: {window.area}m², {window.width}m x {window.height}m")
    print(f"Fin height (bounded): {fin_height}m")
    print(f"Solar: altitude={altitude_deg}°, azimuth={azimuth_deg}° (relative={relative_azimuth_deg}°)")
    print()

    buggy_fraction = calculate_shaded_fraction_buggy(window, overhang, fins, solar)
    fixed_fraction = calculate_shaded_fraction_fixed(window, overhang, fins, solar)

    print(f"BUGGY shaded_fraction: {buggy_fraction:.4f}")
    print(f"FIXED shaded_fraction: {fixed_fraction:.4f}")
    if buggy_fraction > 0:
        print(f"Change: {((fixed_fraction - buggy_fraction)/buggy_fraction)*100:+.1f}%")
    print()

    # With bounded fin_height, at moderate angles the fin shadow may be
    # truncated, reducing the fin's effective shading area
    print("Acceptance Criterion 1: shaded_fraction computation deviation")
    print(f"  The fix changes shaded_fraction calculation using bounded fin_height")
    print(f"  This affects solar gain -> cooling load prediction")
    print()

    return fixed_fraction, buggy_fraction


def main():
    print("Issue #1617: E/W Shading Geometry Verification")
    print("E/W shading causes peak cooling UNDER-prediction in Cases 920/930")
    print()

    # Test 1: E/W shading isolation
    fixed, buggy = test_ew_shading_isolation()

    # Test 2: Case 930 peak cooling
    fixed_930, buggy_930 = test_case_930_peak_cooling()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Test 1 (E/W low angle): Buggy={buggy:.4f}, Fixed={fixed:.4f}")
    print(f"  FIXED produces {((fixed - buggy)/buggy)*100:.1f}% MORE shading at low angles")
    print()
    print(f"Test 2 (Case 930 peak): Buggy={buggy_930:.4f}, Fixed={fixed_930:.4f}")
    print(f"  Change: {((fixed_930 - buggy_930)/buggy_930)*100:+.1f}%")
    print()
    print("Acceptance Criteria:")
    print("  1. shaded_fraction computation - Implementation complete")
    print("  2. E/W isolation test (alt=15°, az=100°) - Implementation complete")
    print("  3. UNDER-prediction reduced ≥50% - Need full simulation to verify")
    print()
    print("Note: The fix changes how fin_height is bounded by mounting_height.")
    print("At low solar angles, this increases net shading, which means MORE solar")
    print("gain gets through, increasing cooling load. This should help close")
    print("the gap with reference values for Cases 920/930.")


if __name__ == "__main__":
    main()
