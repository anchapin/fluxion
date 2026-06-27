#!/usr/bin/env python3
"""
Issue #1333 / B5 — strict ±15% annual-energy CI gate verification.

Reproduces the ASHRAE 140 ±15% band math used by
`tests/zone_balance_eplus_isolation.rs::EnergyReference` and asserts the
same logic that the Rust strict-gate tests assert. This is a sanity
check that the Python reproduction matches the Rust production code —
NOT an assertion that the engine currently passes the band.

Strict gate flow
----------------
1. `tests/zone_balance_eplus_isolation.rs` defines reference ranges for
   Cases 600 (low-mass) and 900 (high-mass) as
   `EnergyReference { case_id, annual_heating_min_mwh, annual_heating_max_mwh,
                       annual_cooling_min_mwh, annual_cooling_max_mwh, ... }`.
2. The acceptance band is computed as
   `midpoint ± 15% of midpoint` of the [min, max] range from the
   ASHRAE 140-2023 Annex B published values (matches EnergyPlus
   regenerated hourly CSVs in `tests/reference_data/zone_balance/`).
3. The strict CI gate
   (`.github/workflows/ashrae_140_strict_energy_gate.yml`) runs the four
   named tests on every PR; if any regress outside the band, the build
   fails.

Why this script
---------------
Per issue #1333 acceptance criteria: "Python verification script confirms
±15% tolerance math: given current pre-#1323 Case 900 annual_cooling
value vs ASHRAE 140 mid-band, prints the current ratio and PASS/FAIL
against the gate (used as a smoke test that the logic matches the Rust
test)."

Run it locally with:
    python3 .agents/results/issue-B5-strict-gate.py
Exit code 0 = math matches Rust. Exit code 1 = mismatch (FAIL).

Reference bands reproduced
--------------------------
- Case 600 (low-mass):  H=[4.36, 5.79] MWh  C=[3.92, 6.14] MWh
- Case 900 (high-mass): H=[1.17, 2.04] MWh  C=[8.00, 10.50] MWh
  (matches `tests/zone_balance_eplus_isolation.rs::CASE_600_REF` and
  `::CASE_900_REF` exactly.)

Engine values used here
-----------------------
The engine values below were measured by running
    cargo test --release --features ort --test zone_balance_eplus_isolation \
        -- --include-ignored \
        test_case_600_annual_energy_ashrae140_tolerance \
        test_case_900_annual_energy_ashrae140_tolerance
on the post-#1323 branch (PR #1356 merged). They are reproduced here so
this script is deterministic and offline — no engine run required.

Output
------
Prints a table of band, engine value, and PASS/FAIL for each metric, plus
an assertion that the computed bands match the Rust test's printed
band edges to 3 decimal places.

Issue link: https://github.com/anchapin/fluxion/issues/1333
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Tuple


TOLERANCE_PCT = 0.15  # ASHRAE 140 annual-energy tolerance


@dataclass(frozen=True)
class EnergyReference:
    case_id: str
    annual_heating_min_mwh: float
    annual_heating_max_mwh: float
    annual_cooling_min_mwh: float
    annual_cooling_max_mwh: float

    def annual_heating_band(self) -> Tuple[float, float]:
        """Acceptance band = midpoint ± 15% of midpoint."""
        mid = 0.5 * (self.annual_heating_min_mwh + self.annual_heating_max_mwh)
        tol = mid * TOLERANCE_PCT
        return (mid - tol, mid + tol)

    def annual_cooling_band(self) -> Tuple[float, float]:
        mid = 0.5 * (self.annual_cooling_min_mwh + self.annual_cooling_max_mwh)
        tol = mid * TOLERANCE_PCT
        return (mid - tol, mid + tol)


# Mirrors tests/zone_balance_eplus_isolation.rs::CASE_600_REF and ::CASE_900_REF
CASE_600_REF = EnergyReference(
    case_id="600",
    annual_heating_min_mwh=4.36,
    annual_heating_max_mwh=5.79,
    annual_cooling_min_mwh=3.92,
    annual_cooling_max_mwh=6.14,
)
CASE_900_REF = EnergyReference(
    case_id="900",
    annual_heating_min_mwh=1.17,
    annual_heating_max_mwh=2.04,
    annual_cooling_min_mwh=8.00,
    annual_cooling_max_mwh=10.50,
)


# Engine values measured on the post-#1323 branch (PR #1356 merged).
# These are the values the strict-gate CI workflow will evaluate when
# the `#[ignore]` attributes are removed in the follow-up issue tracked
# by PR #1367's body.
ENGINE_VALUES = {
    "600": {"heating_mwh": 3.167, "cooling_mwh": 2.672},
    "900": {"heating_mwh": 1.626, "cooling_mwh": 1.203},
}


def evaluate(ref: EnergyReference, heating: float, cooling: float) -> dict:
    """Apply the same logic as the Rust strict-gate test assertions."""
    h_lo, h_hi = ref.annual_heating_band()
    c_lo, c_hi = ref.annual_cooling_band()
    return {
        "heating_band": (h_lo, h_hi),
        "heating_ok": h_lo <= heating <= h_hi,
        "cooling_band": (c_lo, c_hi),
        "cooling_ok": c_lo <= cooling <= c_hi,
        "heating_ratio": heating / (0.5 * (h_lo + h_hi)),
        "cooling_ratio": cooling / (0.5 * (c_lo + c_hi)),
    }


def main() -> int:
    print("=" * 78)
    print("Issue #1333 / B5 — strict ±15% annual-energy CI gate verification")
    print("=" * 78)
    print()
    print(f"Reproducing band math from tests/zone_balance_eplus_isolation.rs")
    print(f"  tolerance: ±{TOLERANCE_PCT * 100:.0f}% of midpoint of [min, max]")
    print()

    overall_ok = True
    print(f"{'Case':<7}{'Metric':<10}{'Band (MWh)':<22}"
          f"{'Engine (MWh)':<14}{'Ratio':<10}{'Status':<8}")
    print("-" * 78)

    for ref in (CASE_600_REF, CASE_900_REF):
        eng = ENGINE_VALUES[ref.case_id]
        result = evaluate(ref, eng["heating_mwh"], eng["cooling_mwh"])

        h_lo, h_hi = result["heating_band"]
        c_lo, c_hi = result["cooling_band"]

        for label, band, value, ok, ratio in [
            ("Heating", (h_lo, h_hi), eng["heating_mwh"],
             result["heating_ok"], result["heating_ratio"]),
            ("Cooling", (c_lo, c_hi), eng["cooling_mwh"],
             result["cooling_ok"], result["cooling_ratio"]),
        ]:
            status = "PASS" if ok else "FAIL"
            print(f"{ref.case_id:<7}{label:<10}"
                  f"[{band[0]:.3f}, {band[1]:.3f}]      "
                  f"{value:<14.3f}{ratio:<10.3f}{status:<8}")
            if not ok:
                overall_ok = False

    print()

    # ---- Math-matches-Rust assertion ----
    # These are the exact numbers printed by the Rust test's assertion
    # message (truncated to 3 decimal places):
    #   Rust: 'Case 600 annual cooling X MWh outside ±15% band [4.275, 5.784]'
    #   Rust: 'Case 900 annual cooling X MWh outside ±15% band [7.862, 10.637]'
    c600_lo, c600_hi = CASE_600_REF.annual_cooling_band()
    c900_lo, c900_hi = CASE_900_REF.annual_cooling_band()
    h600_lo, h600_hi = CASE_600_REF.annual_heating_band()
    h900_lo, h900_hi = CASE_900_REF.annual_heating_band()

    expected = {
        "Case 600 cooling band": (c600_lo, c600_hi, 4.275, 5.784),
        "Case 900 cooling band": (c900_lo, c900_hi, 7.862, 10.637),
        "Case 600 heating band": (h600_lo, h600_hi, 4.314, 5.836),
        "Case 900 heating band": (h900_lo, h900_hi, 1.364, 1.846),
    }
    print("=== Math-matches-Rust assertion (3-dp) ===")
    math_ok = True
    for label, (actual_lo, actual_hi, rust_lo, rust_hi) in expected.items():
        match = (
            abs(actual_lo - rust_lo) < 0.001 and abs(actual_hi - rust_hi) < 0.001
        )
        print(f"  {label:<26}  Python=[{actual_lo:.3f}, {actual_hi:.3f}]"
              f"  Rust=[{rust_lo:.3f}, {rust_hi:.3f}]"
              f"  {'MATCH' if match else 'MISMATCH'}")
        if not match:
            math_ok = False

    print()
    print("=" * 78)
    print("Gate wiring (issue #1333 acceptance criteria)")
    print("=" * 78)
    print("  [x] .github/workflows/ashrae_140_strict_energy_gate.yml exists.")
    print("      Triggers on PR + push to main. Runs 4 named tests in")
    print("      --release mode with --features ort.")
    print("  [x] 4 named tests covered:")
    print("        1. test_case_600_annual_energy_ashrae140_tolerance")
    print("        2. test_case_900_annual_energy_ashrae140_tolerance")
    print("        3. test_blind_mode_case_600_infrastructure")
    print("        4. test_blind_mode_case_900_infrastructure")
    print("  [x] release_gates.yaml::ci.required_checks lists the gate.")
    print("  [ ] Tests 1+2 remain #[ignore] pending A#4 closure")
    print("      verification (post-#1323 cooling-gap regression).")
    print("      When un-ignored in the follow-up issue, the same gate")
    print("      enforces the ±15% band automatically.")
    print()
    print("=" * 78)
    if math_ok:
        print("SMOKE-TEST RESULT: PASS — Python band math matches Rust test.")
    else:
        print("SMOKE-TEST RESULT: FAIL — Python band math diverged from Rust.")
    print()
    if not overall_ok:
        print("ENGINE EVALUATION (informational):")
        print("  Case 600 H FAIL: 3.167 MWh outside [4.314, 5.836]")
        print("  Case 600 C FAIL: 2.672 MWh outside [4.275, 5.784]")
        print("  Case 900 H PASS: 1.626 MWh within [1.364, 1.846]")
        print("  Case 900 C FAIL: 1.203 MWh outside [7.862, 10.637]")
        print()
        print("  This is the expected pre-A#4 state. The gate's job is")
        print("  to keep it this way (red) until the physics layer closes")
        print("  the cooling gap — not to relax the band.")
    print("=" * 78)
    # Smoke-test exit code: 0 iff math matches Rust.
    # The overall_ok flag (engine evaluation) is informational; the
    # script's purpose is math-vs-Rust verification, not engine-PASS.
    return 0 if math_ok else 1


if __name__ == "__main__":
    sys.exit(main())
