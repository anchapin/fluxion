#!/usr/bin/env python3
"""
Generate EnergyPlus reference data for conduction step response tests.

This script generates synthetic reference data for wall, roof, and floor
constructions based on EnergyPlus simulation methodology.

For actual E+ validation, run these constructions in EnergyPlus and export
the surface heat flux and temperature data.

Usage:
    python tests/generate_conduction_reference_data.py
"""

import csv
from pathlib import Path

# Weather data for Golden, CO (Jan 1-3) - extracted from TMY3
WEATHER_DATA = [
    # hour, T_outdoor
    (0.25, -6.0),
    (0.50, -5.0),
    (0.75, -4.0),
    (1.00, -3.0),
    (1.25, -3.0),
    (1.50, -3.0),
    (1.75, -3.0),
    (2.00, -3.0),
    (2.25, -3.25),
    (2.50, -3.5),
    (2.75, -3.75),
    (3.00, -4.0),
    (3.25, -4.25),
    (3.50, -4.5),
    (3.75, -4.75),
    (4.00, -5.0),
    (4.25, -5.25),
    (4.50, -5.5),
    (4.75, -5.75),
    (5.00, -6.0),
    (5.25, -6.25),
    (5.50, -6.5),
    (5.75, -6.75),
    (6.00, -7.0),
    (6.25, -6.75),
    (6.50, -6.5),
    (6.75, -6.25),
    (7.00, -6.0),
    (7.25, -5.5),
    (7.50, -5.0),
    (7.75, -4.5),
    (8.00, -4.0),
    (8.25, -3.25),
    (8.50, -2.5),
    (8.75, -1.75),
    (9.00, -1.0),
    (9.25, -0.25),
    (9.50, 0.5),
    (9.75, 1.25),
    (10.00, 2.0),
    (10.25, 2.75),
    (10.50, 3.5),
    (10.75, 4.25),
    (11.00, 5.0),
    (11.25, 5.75),
    (11.50, 6.5),
    (11.75, 7.25),
    (12.00, 8.0),
    (12.25, 8.5),
    (12.50, 9.0),
    (12.75, 9.5),
    (13.00, 10.0),
    (13.25, 10.5),
    (13.50, 11.0),
    (13.75, 11.5),
    (14.00, 12.0),
    (14.25, 12.25),
    (14.50, 12.5),
    (14.75, 12.75),
    (15.00, 13.0),
    (15.25, 13.0),
    (15.50, 13.0),
    (15.75, 13.0),
    (16.00, 13.0),
    (16.25, 12.75),
    (16.50, 12.5),
    (16.75, 12.25),
    (17.00, 12.0),
    (17.25, 11.5),
    (17.50, 11.0),
    (17.75, 10.5),
    (18.00, 10.0),
    (18.25, 9.25),
    (18.50, 8.5),
    (18.75, 7.75),
    (19.00, 7.0),
    (19.25, 6.25),
    (19.50, 5.5),
    (19.75, 4.75),
    (20.00, 4.0),
    (20.25, 3.5),
    (20.50, 3.0),
    (20.75, 2.5),
    (21.00, 2.0),
    (21.25, 1.5),
    (21.50, 1.0),
    (21.75, 0.5),
    (22.00, 0.0),
    (22.25, -0.5),
    (22.50, -1.0),
    (22.75, -1.5),
    (23.00, -2.0),
    (23.25, -2.5),
    (23.50, -3.0),
    (23.75, -3.5),
    (24.00, -4.0),
    (24.25, -4.25),
    (24.50, -4.5),
    (24.75, -4.75),
    (25.00, -5.0),
    (25.25, -5.25),
    (25.50, -5.5),
    (25.75, -5.75),
    (26.00, -6.0),
    (26.25, -6.25),
    (26.50, -6.5),
    (26.75, -6.75),
    (27.00, -7.0),
    (27.25, -7.25),
    (27.50, -7.5),
    (27.75, -7.75),
    (28.00, -8.0),
    (28.25, -8.0),
    (28.50, -8.0),
    (28.75, -8.0),
    (29.00, -8.0),
    (29.25, -7.75),
    (29.50, -7.5),
    (29.75, -7.25),
    (30.00, -7.0),
    (30.25, -6.5),
    (30.50, -6.0),
    (30.75, -5.5),
    (31.00, -5.0),
    (31.25, -4.25),
    (31.50, -3.5),
    (31.75, -2.75),
    (32.00, -2.0),
    (32.25, -1.25),
    (32.50, -0.5),
    (32.75, 0.25),
    (33.00, 1.0),
    (33.25, 1.75),
    (33.50, 2.5),
    (33.75, 3.25),
    (34.00, 4.0),
    (34.25, 4.75),
    (34.50, 5.5),
    (34.75, 6.25),
    (35.00, 7.0),
    (35.25, 7.5),
    (35.50, 8.0),
    (35.75, 8.5),
    (36.00, 9.0),
    (36.25, 9.5),
    (36.50, 10.0),
    (36.75, 10.5),
    (37.00, 11.0),
    (37.25, 11.25),
    (37.50, 11.5),
    (37.75, 11.75),
    (38.00, 12.0),
    (38.25, 12.0),
    (38.50, 12.0),
    (38.75, 12.0),
    (39.00, 12.0),
    (39.25, 11.75),
    (39.50, 11.5),
    (39.75, 11.25),
    (40.00, 11.0),
    (40.25, 10.5),
    (40.50, 10.0),
    (40.75, 9.5),
    (41.00, 9.0),
    (41.25, 8.25),
    (41.50, 7.5),
    (41.75, 6.75),
    (42.00, 6.0),
    (42.25, 5.25),
    (42.50, 4.5),
    (42.75, 3.75),
    (43.00, 3.0),
    (43.25, 2.5),
    (43.50, 2.0),
    (43.75, 1.5),
    (44.00, 1.0),
    (44.25, 0.5),
    (44.50, 0.0),
    (44.75, -0.5),
    (45.00, -1.0),
    (45.25, -1.5),
    (45.50, -2.0),
    (45.75, -2.5),
    (46.00, -3.0),
    (46.25, -3.25),
    (46.50, -3.5),
    (46.75, -3.75),
    (47.00, -4.0),
    (47.25, -4.25),
    (47.50, -4.5),
    (47.75, -4.75),
    (48.00, -5.0),
    (48.25, -5.25),
    (48.50, -5.5),
    (48.75, -5.75),
    (49.00, -6.0),
    (49.25, -6.25),
    (49.50, -6.5),
    (49.75, -6.75),
    (50.00, -7.0),
    (50.25, -7.0),
    (50.50, -7.0),
    (50.75, -7.0),
    (51.00, -7.0),
    (51.25, -6.75),
    (51.50, -6.5),
    (51.75, -6.25),
    (52.00, -6.0),
    (52.25, -5.5),
    (52.50, -5.0),
    (52.75, -4.5),
    (53.00, -4.0),
    (53.25, -3.25),
    (53.50, -2.5),
    (53.75, -1.75),
    (54.00, -1.0),
    (54.25, -0.25),
    (54.50, 0.5),
    (54.75, 1.25),
    (55.00, 2.0),
    (55.25, 2.75),
    (55.50, 3.5),
    (55.75, 4.25),
    (56.00, 5.0),
    (56.25, 5.75),
    (56.50, 6.5),
    (56.75, 7.25),
    (57.00, 8.0),
    (57.25, 8.5),
    (57.50, 9.0),
    (57.75, 9.5),
    (58.00, 10.0),
    (58.25, 10.5),
    (58.50, 11.0),
    (58.75, 11.5),
    (59.00, 12.0),
    (59.25, 12.25),
    (59.50, 12.5),
    (59.75, 12.75),
    (60.00, 13.0),
    (60.25, 13.0),
    (60.50, 13.0),
    (60.75, 13.0),
    (61.00, 13.0),
    (61.25, 12.75),
    (61.50, 12.5),
    (61.75, 12.25),
    (62.00, 12.0),
    (62.25, 11.5),
    (62.50, 11.0),
    (62.75, 10.5),
    (63.00, 10.0),
    (63.25, 9.25),
    (63.50, 8.5),
    (63.75, 7.75),
    (64.00, 7.0),
    (64.25, 6.25),
    (64.50, 5.5),
    (64.75, 4.75),
    (65.00, 4.0),
    (65.25, 3.5),
    (65.50, 3.0),
    (65.75, 2.5),
    (66.00, 2.0),
    (66.25, 1.5),
    (66.50, 1.0),
    (66.75, 0.5),
    (67.00, 0.0),
    (67.25, -0.5),
    (67.50, -1.0),
    (67.75, -1.5),
    (68.00, -2.0),
    (68.25, -2.5),
    (68.50, -3.0),
    (68.75, -3.5),
    (69.00, -4.0),
    (69.25, -4.25),
    (69.50, -4.5),
    (69.75, -4.75),
    (70.00, -5.0),
    (70.25, -5.25),
    (70.50, -5.5),
    (70.75, -5.75),
    (71.00, -6.0),
    (71.25, -6.25),
    (71.50, -6.5),
    (71.75, -6.75),
    (72.00, -7.0),
]


def generate_step_response(
    construction_name: str,
    layers: list,
    output_path: str,
    t_zone: float = 20.0,
    h_interior: float = 8.29,
    h_exterior: float = 29.3,
):
    """
    Generate step response data for a construction.

    This is a simplified model that estimates surface temperatures and heat flux
    based on the construction's thermal properties. For actual E+ validation,
    replace this with real E+ simulation output.

    Args:
        construction_name: Name of the construction
        layers: List of (thickness, conductivity, density, specific_heat) tuples
        output_path: Path to write CSV file
        t_zone: Zone temperature (°C)
        h_interior: Interior film coefficient (W/m²K)
        h_exterior: Exterior film coefficient (W/m²K)
    """
    # Calculate total R-value
    r_materials = sum(t / k for t, k, _, _ in layers)
    r_total = r_materials + 1 / h_interior + 1 / h_exterior
    u_value = 1 / r_total

    # Calculate total thermal mass
    c_total = sum(t * rho * cp for t, _, rho, cp in layers)

    rows = []

    for hour, t_outdoor in WEATHER_DATA:
        # Estimate surface temperatures using steady-state with thermal mass lag
        # This is a simplified model - actual E+ uses full implicit FD
        delta_t = t_zone - t_outdoor

        # Apply thermal mass effect (simplified)
        tau = c_total / (u_value * 3600)  # time constant in hours
        alpha = 1 / (1 + tau * 0.1)  # lag factor

        t_surface_inside = t_zone - delta_t * u_value / h_interior * alpha
        t_surface_outside = t_outdoor + delta_t * u_value / h_exterior * alpha

        # Heat flux (W/m²) - positive into zone
        q_inside = h_interior * (t_zone - t_surface_inside)
        q_outside = h_exterior * (t_surface_outside - t_outdoor)

        rows.append(
            (
                hour,
                t_outdoor,
                t_zone,
                t_surface_inside,
                t_surface_outside,
                q_inside,
                q_outside,
            )
        )

    # Write CSV
    with open(output_path, "w", newline="") as f:
        f.write(f"# EnergyPlus Reference Data: {construction_name}\n")
        f.write("# Generated: 2026-06-12 (synthetic for testing)\n")
        f.write("# Note: Replace with actual E+ output for validation\n")
        f.write(f"# U-value: {u_value:.4f} W/m²K\n")
        f.write(f"# Total R-value: {r_total:.4f} m²K/W\n")
        f.write(f"# Thermal mass: {c_total:.1f} J/m²K\n")
        f.write(
            "hour,T_outdoor,T_zone,T_surface_inside,T_surface_outside,q_inside_Wm2,q_outside_Wm2\n"
        )

        writer = csv.writer(f)
        for row in rows:
            writer.writerow(
                [
                    f"{row[0]:.4f}",
                    f"{row[1]:.6f}",
                    f"{row[2]:.6f}",
                    f"{row[3]:.6f}",
                    f"{row[4]:.6f}",
                    f"{row[5]:.6f}",
                    f"{row[6]:.6f}",
                ]
            )

    print(f"Generated: {output_path}")


def main():
    output_dir = Path("tests/reference_data/conduction")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 200mm Concrete (existing file - skip)
    # generate_step_response(
    #     "200mm Concrete",
    #     [(0.2, 1.73, 2243.0, 837.0)],
    #     output_dir / "step_response_concrete_200mm.csv",
    # )

    # Lightweight wall (steel stud + fiberglass + gypsum)
    generate_step_response(
        "Lightweight Steel Stud Wall",
        [
            (0.09, 45.0, 7800.0, 500.0),  # Steel stud
            (0.066, 0.04, 12.0, 840.0),  # Fiberglass
            (0.012, 0.16, 784.0, 840.0),  # Gypsum
        ],
        output_dir / "step_response_lightweight.csv",
    )

    # Composite wall (concrete + foam + concrete block)
    generate_step_response(
        "Composite Concrete Wall",
        [
            (0.100, 1.13, 1400.0, 1000.0),  # Concrete inner
            (0.0615, 0.04, 14.0, 1400.0),  # Foam insulation
            (0.100, 0.51, 1400.0, 840.0),  # Concrete block
        ],
        output_dir / "step_response_composite.csv",
    )

    # Roof (plasterboard + fiberglass + deck)
    generate_step_response(
        "Roof Assembly",
        [
            (0.010, 0.16, 784.0, 840.0),  # Plasterboard
            (0.1118, 0.04, 12.0, 840.0),  # Fiberglass
            (0.019, 0.14, 500.0, 1300.0),  # Roof deck
        ],
        output_dir / "step_response_roof.csv",
    )

    # Floor (timber + fiberglass)
    generate_step_response(
        "Insulated Floor",
        [
            (0.025, 0.14, 600.0, 1600.0),  # Timber
            (0.197, 0.04, 12.0, 840.0),  # Fiberglass
        ],
        output_dir / "step_response_floor.csv",
    )

    print("\nReference data generation complete.")
    print(
        "NOTE: These are synthetic data files. Replace with actual E+ output for validation."
    )


if __name__ == "__main__":
    main()
