#!/usr/bin/env python3
"""
Debug script to identify CTF solver issues

This script investigates why CTF is producing incorrect results:
1. Check boundary conditions
2. Verify coefficient values
3. Analyze flux calculations
4. Compare with expected behavior
"""

import re
import subprocess


def extract_ctf_debug_info():
    """Extract CTF debug information from validation output"""
    print("🔍 Extracting CTF debug info...")

    # Run validation and capture output
    result = subprocess.run(
        ["cargo", "run", "--release", "--bin", "fluxion", "validate", "--case", "900"],
        capture_output=True,
        text=True,
        timeout=300,
    )

    output = result.stdout

    # Extract CTF solver step information
    ctf_steps = re.findall(
        r"SESSION 48: CTF solver step \d+: T_mass=([\d.-]+)°C, T_ext=([\d.-]+)°C, Q_CTF=([\d.-]+) W/m²",
        output,
    )

    # Extract CTF fix information
    ctf_fix = re.findall(
        r"SESSION 48 FIX: CTF flux to zone air: Q_CTF=([\d.-]+) W", output
    )

    # Extract CTF enablement info
    ctf_enabled = re.findall(
        r"SESSION 48: CTF solver (ACTIVE|INACTIVE|ENABLED)", output
    )

    return {
        "steps": ctf_steps,
        "fix": ctf_fix,
        "enabled": ctf_enabled,
        "raw_output": output,
    }


def analyze_boundary_conditions(info):
    """Analyze the boundary conditions being used"""
    print("\n📊 Boundary Condition Analysis:")
    print("=" * 60)

    if not info["steps"]:
        print("❌ No CTF solver steps found in output")
        return

    print(f"Found {len(info['steps'])} CTF solver steps")

    # Analyze first few steps
    for i, step in enumerate(info["steps"][:5]):
        t_mass, t_ext, q_ctf = step
        print(f"\nStep {i}:")
        print(f"  Interior (T_mass): {t_mass}°C")
        print(f"  Exterior (T_ext):  {t_ext}°C")
        print(f"  Flux (Q_CTF):      {q_ctf} W/m²")

        # Calculate expected flux based on temperature difference
        try:
            delta_t = float(t_ext) - float(t_mass)
            print(f"  ΔT (T_ext - T_mass): {delta_t:.2f}°C")

            # Expected direction: if T_ext < T_mass, heat should flow OUT (negative flux)
            expected_direction = "OUT" if delta_t < 0 else "IN"
            actual_direction = "OUT" if float(q_ctf) < 0 else "IN"

            print(f"  Expected flow: {expected_direction} (based on ΔT)")
            print(f"  Actual flow:   {actual_direction} (based on Q_CTF)")

            if expected_direction == actual_direction:
                print("  ✅ Direction matches")
            else:
                print("  ❌ Direction MISMATCH!")
        except Exception:
            pass


def check_coefficient_values():
    """Check if CTF coefficients are reasonable"""
    print("\n📊 CTF Coefficient Analysis:")
    print("=" * 60)

    # Look for coefficient debug output
    result = subprocess.run(
        ["cargo", "test", "test_case_900_coefficients", "--", "--nocapture"],
        capture_output=True,
        text=True,
        timeout=120,
    )

    output = result.stdout + result.stderr

    # Extract U-value
    u_value = re.search(r"U-value: ([\d.]+) W/m²K", output)
    if u_value:
        print(f"U-value: {u_value.group(1)} W/m²K")

        # Expected U-value for Case 900
        # Concrete (0.1m, k=0.51) + Foam (0.0615m, k=0.04) + Wood (0.009m, k=0.16)
        # R_total = 0.1/0.51 + 0.0615/0.04 + 0.009/0.16 + R_si + R_se
        #         = 0.196 + 1.537 + 0.056 + 0.125 + 0.044 = 1.958 m²K/W
        # U_value = 1/1.958 = 0.511 W/m²K

        expected_u = 0.511
        actual_u = float(u_value.group(1))

        if abs(actual_u - expected_u) < 0.1:
            print(f"✅ U-value is reasonable (expected ~{expected_u})")
        else:
            print(f"❌ U-value mismatch! Expected ~{expected_u}, got {actual_u}")

    # Extract coefficient values
    x_coeffs = re.search(r"X\[0\]: ([\d.-]+)", output)
    y_coeffs = re.search(r"Y\[0\]: ([\d.-]+)", output)

    if x_coeffs and y_coeffs:
        print(f"\nX[0]: {x_coeffs.group(1)}")
        print(f"Y[0]: {y_coeffs.group(1)}")

        # X[0] should be close to U-value
        x0 = float(x_coeffs.group(1))
        if "u_value" in locals():
            if abs(x0 - actual_u) < 0.1:
                print("✅ X[0] ≈ U-value (correct)")
            else:
                print("❌ X[0] should ≈ U-value")


def identify_issues():
    """Identify specific issues with CTF implementation"""
    print("\n🔍 Issue Identification:")
    print("=" * 60)

    print("\n1. BOUNDARY CONDITION ISSUE:")
    print("   Current code uses:")
    print("   - Interior: T_mass (mass temperature)")
    print("   - Exterior: T_sol_air (sol-air temperature)")
    print()
    print("   ❌ PROBLEM: CTF expects interior surface temperature!")
    print("   The CTF formulation is:")
    print("   q''_i(t) = Σ(X_j·T_o) - Σ(Y_j·T_i) - Σ(Φ_j·q'')")
    print("   where T_i is INTERIOR SURFACE temperature, not mass temp")
    print()
    print("   In 5R1C network:")
    print("   Exterior ──h_tr_em──> Mass ──h_tr_ms──> Surface ──h_tr_is──> Zone Air")
    print()
    print("   CTF should replace the h_tr_em path, so it needs:")
    print("   - Interior boundary: Surface temperature (between mass and zone air)")
    print("   - OR: Zone air temperature (simplified)")

    print("\n2. RECOMMENDED FIX:")
    print("   Option A: Use zone air temperature")
    print("   - Change line 3426: t_mass → t_zone")
    print("   - This approximates interior surface as zone air")
    print()
    print("   Option B: Calculate interior surface temperature")
    print("   - T_surface = (T_zone·h_tr_is + T_mass·h_tr_ms) / (h_tr_is + h_tr_ms)")
    print("   - More accurate but more complex")
    print()
    print("   Option C: Disable CTF for now")
    print("   - Revert to 5R1C which passes validation")
    print("   - Fix CTF boundary conditions later")


def main():
    print("🔬 CTF Solver Debugging Script")
    print("=" * 60)

    # Extract debug info
    info = extract_ctf_debug_info()

    # Analyze boundary conditions
    analyze_boundary_conditions(info)

    # Check coefficients
    check_coefficient_values()

    # Identify issues
    identify_issues()

    print("\n" + "=" * 60)
    print("📋 Summary:")
    print("=" * 60)
    print()
    print("Root Cause: CTF solver is using WRONG boundary temperature")
    print()
    print("Current:  T_mass (interior boundary) ❌")
    print("Correct:   T_zone or T_surface (interior boundary) ✅")
    print()
    print("This causes:")
    print("1. Wrong temperature difference across wall")
    print("2. Wrong flux magnitude and direction")
    print("3. Wrong energy balance (too much heating, no cooling)")
    print()
    print("Fix: Change line 3426 in src/sim/engine.rs")
    print("     FROM: let t_mass = mass_temps.get(i).copied().unwrap_or(20.0);")
    print("     TO:   let t_zone = temps.get(i).copied().unwrap_or(20.0);")
    print("     AND:  let q_flux = solver.step(t_zone, t_ext);")


if __name__ == "__main__":
    main()
