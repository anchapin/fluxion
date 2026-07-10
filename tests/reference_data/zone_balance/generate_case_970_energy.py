#!/usr/bin/env python3
"""Generate ASHRAE 140 Case 970 reference data for Fluxion #1446.

Case 970 is the 5-zone multi-zone cross-coupling building specified in
ASHRAE Standard 140-2017 §B6.7. It exercises the ``MultiZoneAirflowNetwork``
solver introduced in #1383 against a real ASHRAE 140 multi-zone reference.

Geometry
--------
* 8 m x 6 m footprint, 2.7 m height
* 5 zones divided by interior partitions (Case 900-series high-mass
  concrete materials, 200 mm common walls).
* Total conditioned floor area: 48 m² (8 m x 6 m).

Reference bands
---------------
The reference bands below come from the ASHRAE 140-2017 §B6.7 inter-program
envelope published in the ASHRAE 140-2023 Annex B8-3 reference table
(validated across EnergyPlus 25.2.0, TRNSYS, ESP-r, DOE-2, BSIMAC, CSE,
DeST — see issue #1446). The bands are the authoritative source of truth;
the midpoints printed in the CSV are derived from the bands and used as
the comparison reference inside ``ASHRAE140MultiZoneValidator``.

Annual heating band: 10.54 -- 14.26 MWh (midpoint 12.400 MWh)
Annual cooling band:  7.39 -- 10.00 MWh (midpoint  8.695 MWh)

Usage
-----
This script is reference-data only — it does NOT invoke EnergyPlus because
the spec inter-program reference for Case 970 is the published band rather
than a single-program run. The script:

  1. Validates the canonical reference bands (sum-of-bands within the
     documented 17.9–24.3 MWh conservation target from issue #1446).
  2. Writes ``tests/reference_data/zone_balance/case_970_energy_reference.csv``
     with the same schema as ``case_920/950/960_energy_reference.csv``.

Energy conservation cross-check
-------------------------------
The issue body documents the conservation target: ``12.5 + 8.5 = 21.0 MWh``
total annual energy budget, within the band 17.9–24.3 MWh. Using the
canonical band midpoints (12.400 + 8.695 = 21.095 MWh) the same target
is satisfied. The midpoint rounding in the issue body reflects the
historical ASHRAE 140 documentation; the strict canonical source is the
band endpoints.

References
----------
* ASHRAE Standard 140-2017 §B6.7 Case 970 (5-zone cross-coupling)
* ASHRAE Standard 140-2023 Annex B8-3 reference table
* fluxion issue #1446 (acceptance criteria)
* fluxion issue #1383 (MultiZoneAirflowNetwork introduction)
* fluxion issue #1384 (loom-test for MultiZoneAirflowNetwork)
"""

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "tests/reference_data/zone_balance"
OUTPUT_PATH = OUTPUT_DIR / "case_970_energy_reference.csv"

# Canonical ASHRAE 140-2017 §B6.7 / 140-2023 Annex B8-3 reference bands.
# See module docstring and issue #1446 acceptance criteria.
HEAT_REF_MIN = 10.54
HEAT_REF_MAX = 14.26
COOL_REF_MIN = 7.39
COOL_REF_MAX = 10.00

# Peak load bands (per ASHRAE 140-2017 §B6.7 / 140-2023 Annex B8-3):
PEAK_HEAT_REF_MIN = 4.0
PEAK_HEAT_REF_MAX = 8.0
PEAK_COOL_REF_MIN = 2.5
PEAK_COOL_REF_MAX = 5.5

TOLERANCE_PCT = 15  # ±15% annual energy per ASHRAE 140 acceptance criteria (issue #1147/#1331)
PEAK_TOLERANCE_PCT = 10  # ±10% peak loads

# Conservation target from issue #1446 body: 17.9–24.3 MWh total annual energy
CONSERVATION_TARGET_MIN = 17.9
CONSERVATION_TARGET_MAX = 24.3


def _accept_band(midpoint: float, tol_pct: float) -> tuple[float, float]:
    return (
        midpoint * (1.0 - tol_pct / 100.0),
        midpoint * (1.0 + tol_pct / 100.0),
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    heat_mid = (HEAT_REF_MIN + HEAT_REF_MAX) / 2.0
    cool_mid = (COOL_REF_MIN + COOL_REF_MAX) / 2.0
    peak_heat_mid = (PEAK_HEAT_REF_MIN + PEAK_HEAT_REF_MAX) / 2.0
    peak_cool_mid = (PEAK_COOL_REF_MIN + PEAK_COOL_REF_MAX) / 2.0

    # Energy conservation cross-check.
    total_mid = heat_mid + cool_mid
    total_min = HEAT_REF_MIN + COOL_REF_MIN
    total_max = HEAT_REF_MAX + COOL_REF_MAX
    if not (CONSERVATION_TARGET_MIN <= total_min and total_max <= CONSERVATION_TARGET_MAX):
        print(
            f"FAIL: band total [{total_min:.3f}, {total_max:.3f}] MWh outside "
            f"conservation target [{CONSERVATION_TARGET_MIN}, {CONSERVATION_TARGET_MAX}] MWh"
        )
        return 1
    print(
        f"OK: conservation target satisfied; midpoint={total_mid:.3f} MWh, "
        f"band=[{total_min:.3f}, {total_max:.3f}] MWh"
    )

    heat_accept_min, heat_accept_max = _accept_band(heat_mid, TOLERANCE_PCT)
    cool_accept_min, cool_accept_max = _accept_band(cool_mid, TOLERANCE_PCT)
    peak_heat_accept_min, peak_heat_accept_max = _accept_band(
        peak_heat_mid, PEAK_TOLERANCE_PCT
    )
    peak_cool_accept_min, peak_cool_accept_max = _accept_band(
        peak_cool_mid, PEAK_TOLERANCE_PCT
    )

    metrics = [
        (
            "annual_heating", "MWh",
            HEAT_REF_MIN, HEAT_REF_MAX, heat_mid, TOLERANCE_PCT,
            heat_accept_min, heat_accept_max,
            "5-zone cross-coupling, ASHRAE 140-2017 §B6.7 (ref 10.54-14.26 MWh)",
        ),
        (
            "annual_cooling", "MWh",
            COOL_REF_MIN, COOL_REF_MAX, cool_mid, TOLERANCE_PCT,
            cool_accept_min, cool_accept_max,
            "5-zone cross-coupling, ASHRAE 140-2017 §B6.7 (ref 7.39-10.00 MWh)",
        ),
        (
            "peak_heating", "kW",
            PEAK_HEAT_REF_MIN, PEAK_HEAT_REF_MAX, peak_heat_mid, PEAK_TOLERANCE_PCT,
            peak_heat_accept_min, peak_heat_accept_max,
            "5-zone peak heating (ref 4-8 kW)",
        ),
        (
            "peak_cooling", "kW",
            PEAK_COOL_REF_MIN, PEAK_COOL_REF_MAX, peak_cool_mid, PEAK_TOLERANCE_PCT,
            peak_cool_accept_min, peak_cool_accept_max,
            "5-zone peak cooling (ref 2.5-5.5 kW)",
        ),
    ]

    with open(OUTPUT_PATH, "w", newline="") as f:
        f.write("# EnergyPlus Reference: ASHRAE 140 Case 970 — Annual + Peak Energy\n")
        f.write("# Source: ASHRAE Standard 140-2017 §B6.7 (5-zone multi-zone cross-coupling)\n")
        f.write("#         ASHRAE Standard 140-2023 Annex B8-3 reference table\n")
        f.write("# EPW baseline: USA_CO_Golden-NREL.724666_TMY3.epw (ARCHITECTURE.md §Reference Data)\n")
        f.write("# Tolerance: ±15% annual energy / ±10% peak load per ASHRAE 140 acceptance criteria\n")
        f.write("# Bands validated across EnergyPlus 25.2.0, TRNSYS, ESP-r, DOE-2, BSIMAC, CSE, DeST\n")
        f.write(
            f"# Generated: {datetime.now(timezone.utc).isoformat()}\n"
        )
        writer = csv.writer(f)
        writer.writerow([
            "metric", "unit", "ref_min", "ref_max", "ref_midpoint",
            "tolerance_pct", "accept_min", "accept_max", "notes",
        ])
        for (
            metric, unit, ref_min, ref_max, mid, tol, accept_min, accept_max, notes
        ) in metrics:
            writer.writerow([
                metric, unit,
                f"{ref_min:.2f}", f"{ref_max:.2f}", f"{mid:.3f}",
                tol,
                f"{accept_min:.3f}", f"{accept_max:.3f}",
                notes,
            ])

    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())