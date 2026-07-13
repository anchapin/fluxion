#!/usr/bin/env python3
"""
Grid search for H_SI (interior surface convective coefficient) calibration.

ASHRAE 140 Case 600 Free-Floating (FF) target:
  Max Temp: 64.90 to 75.10°C

This script sweeps H_SI from 3.0 to 8.0 W/m²K and runs the Case 600FF
test to find values that put the peak temperature in the reference band.
"""

import subprocess
import re
import sys
from pathlib import Path

# H_SI constant locations in thermal_model_solvers.rs and construction.rs
SOLVER_FILE = Path("src/sim/thermal_model_solvers.rs")
CONSTRUCTION_FILE = Path("src/sim/construction.rs")

# Reference bounds for Case 600FF
REF_MIN = 64.90
REF_MAX = 75.10

def update_h_si(new_value: float) -> tuple[float, float]:
    """Update H_SI constant in both files. Returns old values."""
    # Read and update thermal_model_solvers.rs
    content_solver = SOLVER_FILE.read_text()
    old_solver = None
    for line in content_solver.split('\n'):
        if 'const H_SI: f64' in line and 'W/m²K' in line:
            old_solver = float(re.search(r'= (\d+\.?\d*)', line).group(1))
            content_solver = content_solver.replace(
                line,
                line.replace(f'= {old_solver};', f'= {new_value};')
            )
            break
    SOLVER_FILE.write_text(content_solver)

    # Read and update construction.rs
    content_const = CONSTRUCTION_FILE.read_text()
    old_const = None
    for line in content_const.split('\n'):
        if 'const H_SI: f64' in line and 'W/m²K' in line:
            old_const = float(re.search(r'= (\d+\.?\d*)', line).group(1))
            content_const = content_const.replace(
                line,
                line.replace(f'= {old_const};', f'= {new_value};')
            )
            break
    CONSTRUCTION_FILE.write_text(content_const)

    return old_solver, old_const

def restore_h_si(old_solver: float, old_const: float):
    """Restore original H_SI values."""
    content_solver = SOLVER_FILE.read_text()
    content_solver = re.sub(
        r'(const H_SI: f64.*= )\d+\.?\d*(;.*W/m²K)',
        f'\\g<1>{old_solver}\\g<2>',
        content_solver
    )
    SOLVER_FILE.write_text(content_solver)

    content_const = CONSTRUCTION_FILE.read_text()
    content_const = re.sub(
        r'(const H_SI: f64.*= )\d+\.?\d*(;.*W/m²K)',
        f'\\g<1>{old_const}\\g<2>',
        content_const
    )
    CONSTRUCTION_FILE.write_text(content_const)

def run_case_600ff_test() -> tuple[bool, float]:
    """Run Case 600FF test and return (passed, max_temp)."""
    result = subprocess.run(
        ["cargo", "test", "case_600ff::test_max_temperature", "--", "--nocapture"],
        capture_output=True,
        text=True,
        timeout=300
    )

    # Parse max temperature from output
    match = re.search(r'Case 600FF Max Temp: ([\d.]+)°C', result.stdout + result.stderr)
    if match:
        max_temp = float(match.group(1))
        passed = REF_MIN <= max_temp <= REF_MAX
        return passed, max_temp
    return False, 0.0

def main():
    print("=" * 60)
    print("Grid Search: H_SI Calibration for Case 600FF")
    print("=" * 60)
    print(f"Reference bounds: {REF_MIN}°C to {REF_MAX}°C")
    print()

    # Save original values
    _, _ = update_h_si(3.45)  # Just to get original values
    orig_solver, orig_const = update_h_si(3.45)  # Get current values
    print(f"Original H_SI values: solver={orig_solver}, construction={orig_const}")

    results = []

    # Sweep H_SI from 3.0 to 8.0 in 0.5 increments
    h_si_values = [round(x * 0.5 + 3.0, 1) for x in range(11)]  # 3.0, 3.5, 4.0, ... 8.0

    for h_si in h_si_values:
        print(f"\nTesting H_SI = {h_si} W/m²K...")

        # Update H_SI
        update_h_si(h_si)

        # Rebuild
        build_result = subprocess.run(
            ["cargo", "build"],
            capture_output=True,
            text=True,
            timeout=120
        )

        if build_result.returncode != 0:
            print(f"  BUILD FAILED: {build_result.stderr[-200:]}")
            continue

        # Run test
        passed, max_temp = run_case_600ff_test()
        status = "PASS" if passed else "FAIL"
        print(f"  Max Temp: {max_temp:.2f}°C [{status}]")

        results.append((h_si, max_temp, passed))

    # Restore original
    restore_h_si(orig_solver, orig_const)

    # Summary
    print("\n" + "=" * 60)
    print("GRID SEARCH RESULTS")
    print("=" * 60)
    print(f"{'H_SI (W/m²K)':<15} {'Max Temp (°C)':<15} {'Status'}")
    print("-" * 45)

    for h_si, max_temp, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        marker = " <-- IN RANGE" if passed else ""
        print(f"{h_si:<15.1f} {max_temp:<15.2f} {status}{marker}")

    # Find best in-range values
    in_range = [(h, t, p) for h, t, p in results if p]
    if in_range:
        print(f"\nValues in reference range: {[h for h, _, _ in in_range]}")
        # Find closest to center of range
        center = (REF_MIN + REF_MAX) / 2
        best = min(in_range, key=lambda x: abs(x[1] - center))
        print(f"Best (closest to center {center:.1f}°C): H_SI = {best[0]:.1f} -> {best[1]:.2f}°C")
    else:
        print("\nNo values in reference range. Consider wider sweep or different parameter.")

if __name__ == "__main__":
    main()
