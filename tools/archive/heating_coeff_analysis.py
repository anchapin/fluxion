#!/usr/bin/env python3
"""
Heat Transfer Coefficient Analysis for ASHRAE 140 Case 600

This script calculates expected vs actual heat transfer coefficients
to investigate heating overprediction.
"""

# ASHRAE 140 Case 600 Specifications
floor_area = 48.0  # m²
ceiling_height = 2.7  # m
volume = floor_area * ceiling_height  # 129.6 m³

# Window specifications
window_area = 12.0  # m² (south-facing)
window_u_value = 3.0  # W/m²K (double clear glass)

# Construction U-values (from low_mass construction)
# These are approximate - actual values depend on layer R-values
wall_u_approx = 0.514  # W/m²K (expected for low-mass wall)
roof_u_approx = 0.318  # W/m²K (expected for low-mass roof)

# Geometry
aspect_ratio = 6.0 / 8.0  # depth / width
width = (floor_area * aspect_ratio) ** 0.5  # ~6.93 m
depth = floor_area / width  # ~6.93 m (actually width=depth for square)
perimeter = 2.0 * (width + depth)  # ~27.7 m
gross_wall_area = perimeter * ceiling_height  # ~74.8 m²
window_ratio = window_area / gross_wall_area  # ~0.16

# Using 25% window ratio as spec
window_ratio_spec = 0.25
gross_wall_area_spec = window_area / window_ratio_spec  # 48.0 m²
opaque_wall_area = gross_wall_area_spec - window_area  # 36.0 m²

print("=" * 60)
print("ASHRAE 140 Case 600 - Heat Transfer Coefficient Analysis")
print("=" * 60)

print("\nGeometry:")
print(f"  Floor area: {floor_area:.2f} m²")
print(f"  Ceiling height: {ceiling_height:.2f} m")
print(f"  Volume: {volume:.2f} m³")
print(f"  Window area: {window_area:.2f} m²")
print(f"  Gross wall area: {gross_wall_area_spec:.2f} m²")
print(f"  Opaque wall area: {opaque_wall_area:.2f} m²")
print(f"  Roof area: {floor_area:.2f} m²")

print("\nExpected Coefficients (ASHRAE 140):")

# h_tr_w = Window Area × Window U-value
h_tr_w_expected = window_area * window_u_value
print(f"  h_tr_w (window): {h_tr_w_expected:.2f} W/K")
print(f"    = {window_area:.1f} m² × {window_u_value:.1f} W/m²K")

# h_tr_em = Opaque Wall Area × Wall U + Roof Area × Roof U
h_tr_em_expected = opaque_wall_area * wall_u_approx + floor_area * roof_u_approx
print(f"  h_tr_em (opaque+roof): {h_tr_em_expected:.2f} W/K")
print(
    f"    = {opaque_wall_area:.1f} m² × {wall_u_approx:.3f} W/m²K + {floor_area:.1f} m² × {roof_u_approx:.3f} W/m²K"
)

# h_ve = Air Density × Cp × (ACH × Volume / 3600)
air_density = 1.2  # kg/m³
cp_air = 1000.0  # J/kg·K
infiltration_ach = 0.5  # ACH
q_vent = infiltration_ach * volume / 3600.0  # m³/s
h_ve_expected = air_density * cp_air * q_vent
print(f"  h_ve (ventilation): {h_ve_expected:.2f} W/K")
print(f"    = {air_density:.1f} kg/m³ × {cp_air:.0f} J/kg·K × {q_vent:.4f} m³/s")
print(
    f"    = {air_density * cp_air:.0f} J/m³K × {infiltration_ach:.1f} ACH × {volume:.1f} m³ / 3600"
)

print("\nTotal Heat Loss Coefficient:")
h_total_expected = h_tr_w_expected + h_tr_em_expected + h_ve_expected
print("  h_ext = h_tr_w + h_tr_em + h_ve")
print(
    f"  h_ext = {h_tr_w_expected:.2f} + {h_tr_em_expected:.2f} + {h_ve_expected:.2f} = {h_total_expected:.2f} W/K"
)

print("\n" + "=" * 60)
print("Sensitivity Analysis")
print("=" * 60)

# 5R1C model parameters
h_is = 3.45  # W/m²K (interior surface conductance)
h_tr_is = (opaque_wall_area + floor_area * 2.0) * h_is  # surface->interior
print(f"  h_tr_is = ({opaque_wall_area:.1f} + {floor_area:.1f} × 2) × {h_is} W/m²K")
print(f"  h_tr_is = {h_tr_is:.2f} W/K")

# Calculate approximate sensitivity
# sensitivity = h_tr_ms / (h_tr_ms * h_tr_is + h_tr_ms * h_ext + h_tr_is * h_ext)
# For simplified analysis, assume h_tr_ms is in series with h_tr_is
h_tr_ms_approx = 10.0  # W/K (typical value for mass->surface)
term_rest_1 = h_tr_ms_approx + h_tr_is
den = h_tr_ms_approx * h_tr_is + term_rest_1 * h_total_expected
sensitivity_approx = term_rest_1 / den

print(f"\n  h_tr_ms (mass->surface): {h_tr_ms_approx:.2f} W/K (assumed)")
print(
    f"  term_rest_1 = h_tr_ms + h_tr_is = {h_tr_ms_approx:.2f} + {h_tr_is:.2f} = {term_rest_1:.2f} W/K"
)
print("  den = h_ms*h_is + term_rest_1*h_ext")
print(
    f"       = {h_tr_ms_approx:.2f} × {h_tr_is:.2f} + {term_rest_1:.2f} × {h_total_expected:.2f}"
)
print(
    f"       = {h_tr_ms_approx * h_tr_is:.2f} + {term_rest_1 * h_total_expected:.2f} = {den:.2f} W²/K²"
)
print(
    f"  sensitivity = term_rest_1 / den = {term_rest_1:.2f} / {den:.2f} = {sensitivity_approx:.4f}"
)

print("\nHVAC Demand Calculation:")
print("  required_load = (setpoint - Ti_free) / sensitivity")
print("  If Ti_free = 15°C and heating setpoint = 20°C:")
print(f"    required_load = (20.0 - 15.0) / {sensitivity_approx:.4f}")
print(
    f"    required_load = 5.0 / {sensitivity_approx:.4f} = {5.0 / sensitivity_approx:.2f} W"
)
print(f"    required_load = {5.0 / sensitivity_approx / 1000:.3f} kW")

print("\n" + "=" * 60)
print("Key Findings:")
print("=" * 60)
print("1. If sensitivity is too small, heating demand will be overpredicted")
print("2. Sensitivity decreases as h_ext increases (more heat loss)")
print("3. Sensitivity decreases as h_tr_ms or h_tr_is increase")
print("4. Check actual vs expected h_tr_w, h_tr_em, h_ve values")
print("=" * 60)
