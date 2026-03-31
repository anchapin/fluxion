#!/usr/bin/env python3
"""
Analysis of heat transfer coefficient calculations for Case 600

Compares two different calculation methods used in the code.
"""

print("=" * 70)
print("HEAT TRANSFER COEFFICIENT CALCULATION COMPARISON")
print("=" * 70)

# Case 600 Specifications
floor_area = 48.0  # m²
ceiling_height = 2.7  # m
volume = floor_area * ceiling_height  # 129.6 m³
window_area = 12.0  # m²

# Geometry
aspect_ratio = 1.0  # For square 8m × 6m, this gives proper dimensions
# Actually: aspect_ratio = depth / width = 6 / 8 = 0.75
aspect_ratio = 6.0 / 8.0  # depth / width
width = (floor_area * aspect_ratio) ** 0.5  # sqrt(48 * 0.75) ≈ 6m
depth = floor_area / width  # 48 / 6 = 8m
perimeter = 2.0 * (width + depth)  # 28m
gross_wall_area = perimeter * ceiling_height  # 28 * 2.7 = 75.6 m²

# Using window_ratio to calculate window area from geometry
window_ratio = window_area / gross_wall_area  # 12 / 75.6 ≈ 0.159
# But spec says 25% window ratio

# Let's use the spec approach: window_ratio determines window_area
window_ratio_spec = 0.25  # ASHRAE 140 spec
gross_wall_area_from_spec = window_area / window_ratio_spec  # 12 / 0.25 = 48 m²

opaque_wall_area = gross_wall_area_from_spec - window_area  # 48 - 12 = 36 m²

print("\nGeometry (from ASHRAE 140 spec):")
print(f"  Floor area: {floor_area:.1f} m²")
print(f"  Ceiling height: {ceiling_height:.1f} m")
print(f"  Volume: {volume:.1f} m³")
print(f"  Window area: {window_area:.1f} m²")
print(f"  Window ratio: {window_ratio_spec:.2f}")
print(f"  Gross wall area: {gross_wall_area_from_spec:.1f} m²")
print(f"  Opaque wall area: {opaque_wall_area:.1f} m²")
print(f"  Roof area: {floor_area:.1f} m²")

# Construction U-values
wall_u = 0.514  # W/m²K (low-mass wall)
roof_u = 0.318  # W/m²K (low-mass roof)
window_u = 3.0  # W/m²K (double clear glass)

print("\nConstruction U-values:")
print(f"  Wall U-value: {wall_u:.3f} W/m²K")
print(f"  Roof U-value: {roof_u:.3f} W/m²K")
print(f"  Window U-value: {window_u:.1f} W/m²K")

# Method 1: Physics-based h_tr_em (used in from_spec, then overwritten)
print("\n" + "=" * 70)
print("METHOD 1: Physics-based h_tr_em (k * A / d)")
print("=" * 70)
k_envelope_low_mass = 0.7  # W/mK (for low-mass)
d_envelope_low_mass = 0.1  # m (thickness)
h_tr_em_method1 = k_envelope_low_mass * opaque_wall_area / d_envelope_low_mass
print(f"  k_envelope (low-mass): {k_envelope_low_mass:.1f} W/mK")
print(f"  d_envelope (thickness): {d_envelope_low_mass:.2f} m")
print(
    f"  h_tr_em = {k_envelope_low_mass:.1f} × {opaque_wall_area:.1f} / {d_envelope_low_mass:.2f}"
)
print(f"  h_tr_em = {h_tr_em_method1:.2f} W/K")

# Method 2: Construction U-value based (used in update_derived_parameters)
print("\n" + "=" * 70)
print("METHOD 2: Construction U-value based (wall * U + roof * U)")
print("=" * 70)
h_tr_wall = opaque_wall_area * wall_u
h_tr_roof = floor_area * roof_u
h_tr_em_method2 = h_tr_wall + h_tr_roof
print(
    f"  h_tr_wall = {opaque_wall_area:.1f} m² × {wall_u:.3f} W/m²K = {h_tr_wall:.2f} W/K"
)
print(f"  h_tr_roof = {floor_area:.1f} m² × {roof_u:.3f} W/m²K = {h_tr_roof:.2f} W/K")
print(f"  h_tr_em = {h_tr_em_method2:.2f} W/K")

# h_tr_w (window)
h_tr_w = window_area * window_u
print(f"\n  h_tr_w = {window_area:.1f} m² × {window_u:.1f} W/m²K = {h_tr_w:.2f} W/K")

# h_ve (ventilation)
infiltration_ach = 0.5  # ACH
air_density = 1.2  # kg/m³
cp_air = 1000.0  # J/kgK
# Using 1.225 and 1005.0 from the code
air_density_code = 1.225
cp_air_code = 1005.0
h_ve_code = (infiltration_ach * volume * air_density_code * cp_air_code) / 3600.0
h_ve_simple = (infiltration_ach * volume * air_density * cp_air) / 3600.0
print(f"\n  h_ve (air_density=1.225, cp=1005) = {h_ve_code:.2f} W/K")
print(f"  h_ve (air_density=1.200, cp=1000) = {h_ve_simple:.2f} W/K")

# h_tr_is (surface-to-interior)
h_si = 3.07  # W/m²K (ASHRAE 140 interior surface film coefficient)
interior_surface_area = (
    opaque_wall_area + floor_area
)  # Only walls + floor, not roof × 2
h_tr_is = h_si * interior_surface_area
print(
    f"\n  h_tr_is = {interior_surface_area:.1f} m² × {h_si:.2f} W/m²K = {h_tr_is:.2f} W/K"
)

# Total h_ext
h_ext_method1 = h_tr_em_method1 + h_tr_w + h_ve_code
h_ext_method2 = h_tr_em_method2 + h_tr_w + h_ve_code

print("\n" + "=" * 70)
print("TOTAL HEAT LOSS COEFFICIENT (h_ext)")
print("=" * 70)
print(
    f"Method 1: h_ext = {h_tr_em_method1:.2f} + {h_tr_w:.2f} + {h_ve_code:.2f} = {h_ext_method1:.2f} W/K"
)
print(
    f"Method 2: h_ext = {h_tr_em_method2:.2f} + {h_tr_w:.2f} + {h_ve_code:.2f} = {h_ext_method2:.2f} W/K"
)

print("\n" + "=" * 70)
print("COMPARISON WITH EXPECTED VALUES")
print("=" * 70)
h_tr_em_expected = 33.77  # From earlier calculation
print(f"Expected h_tr_em: {h_tr_em_expected:.2f} W/K")
print(
    f"Method 1 result: {h_tr_em_method1:.2f} W/K ({h_tr_em_method1 / h_tr_em_expected:.2f}x expected)"
)
print(
    f"Method 2 result: {h_tr_em_method2:.2f} W/K ({h_tr_em_method2 / h_tr_em_expected:.2f}x expected)"
)

# Sensitivity calculation
print("\n" + "=" * 70)
print("SENSITIVITY CALCULATION")
print("=" * 70)
h_tr_ms = 10.0  # Assumed value
term_rest_1 = h_tr_ms + h_tr_is
den1 = h_tr_ms * h_tr_is + term_rest_1 * h_ext_method1
den2 = h_tr_ms * h_tr_is + term_rest_1 * h_ext_method2
sens1 = term_rest_1 / den1
sens2 = term_rest_1 / den2
print(f"h_tr_ms = {h_tr_ms:.1f} W/K (assumed)")
print(f"term_rest_1 = {h_tr_ms:.1f} + {h_tr_is:.2f} = {term_rest_1:.2f} W/K")
print("\nUsing Method 1 h_ext:")
print(
    f"  den = {h_tr_ms:.1f} × {h_tr_is:.2f} + {term_rest_1:.2f} × {h_ext_method1:.2f}"
)
print(f"      = {h_tr_ms * h_tr_is:.2f} + {term_rest_1 * h_ext_method1:.2f}")
print(f"      = {den1:.2f} W²/K²")
print(f"  sensitivity = {term_rest_1:.2f} / {den1:.2f} = {sens1:.5f}")
print("\nUsing Method 2 h_ext:")
print(
    f"  den = {h_tr_ms:.1f} × {h_tr_is:.2f} + {term_rest_1:.2f} × {h_ext_method2:.2f}"
)
print(f"      = {h_tr_ms * h_tr_is:.2f} + {term_rest_1 * h_ext_method2:.2f}")
print(f"      = {den2:.2f} W²/K²")
print(f"  sensitivity = {term_rest_1:.2f} / {den2:.2f} = {sens2:.5f}")

print("\n" + "=" * 70)
print("KEY FINDINGS:")
print("=" * 70)
print("1. Method 2 (construction U-value) gives h_tr_em ≈ 33.8 W/K")
print("   This is close to expected 33.77 W/K ✓")
print("2. Method 1 (physics k/d) gives h_tr_em ≈ 252 W/K")
print("   This is 7.5x too high! ✗")
print("3. The code uses Method 2 in update_derived_parameters()")
print("   BUT Method 1 is calculated first in from_spec()")
print("   Need to verify which value actually gets used")
print("=" * 70)
