"""
Annual energy validation for ASHRAE 140 Cases 600 / 900 / 920 / 950 / 960 / 970
against the EnergyPlus reference envelope (issues #2678, #3572, #3593).

Honesty contract (RULES.md "no parameter tuning / hardcoding to match"):
  Every metric is checked against the ASHRAE 140 annual-energy band
  (``accept_min``/``accept_max`` = midpoint x (1 +/- 15 %) from the
  authoritative reference CSVs). There are NO percentage tolerances anywhere in
  this module -- and never again the 90-400 % bands that previously let a 5x
  deviation pass. Metrics the engine genuinely passes are asserted tightly;
  metrics with a KNOWN structural physics gap are marked
  ``pytest.mark.xfail(strict=True)`` with the SAME +/-15 % assertion inside, so
  the moment the structural fix lands (and the recorded baseline value re-enters
  its band) the xfail flips to XPASS and ``strict=True`` fails the suite --
  signalling it is time to remove the marker. This mirrors the strict-energy-gate
  pattern (AGENTS.md Validation Strategy; issues #1333 / #2506 / #3572 / #3593).

Data sources (single source of truth -- no magic numbers, no per-case tuning):
  * Published bands: ``tests/reference_data/zone_balance/
    case_{600,900,920,950,960,970}_energy_reference.csv`` -- reconciled with
    ``src/validation/benchmark.rs`` per issue #1421; cases 920/950/960/970 added
    per issue #3572; Python-side coverage extended per issue #3593.
  * Recorded engine values: ``tests/reference_data/zone_balance/
    strict_energy_gate_baseline.json`` (captured at ``develop @ 89b1763``
    2026-09-08, maintained by the ``zone_balance_eplus_isolation`` strict-energy
    cargo workflow). LIVE engine verification is that cargo workflow's job, not
    this module's; this module is the Python-side honesty guard that
    (a) tolerances stay tight and (b) known gaps stay explicitly documented
    instead of being hidden behind a 400 % tolerance.

Known structural gaps (do NOT "fix" by widening -- fix the physics):
  * Case 600 cooling -- engine 2.546 MWh vs band [4.275, 5.784] MWh
    (gap 34.38 % of band midpoint). docs/KNOWN_ISSUES.md Sec LIMIT-05 /
    Sec SOLAR-02 (issues #1457 / #2239).
  * Case 900 cooling -- engine 0.910 MWh vs band [2.465, 3.335] MWh
    (gap 53.62 % of band midpoint). docs/KNOWN_ISSUES.md Sec SOLAR-02 UPDATE
    (#2239) + Sec LIMIT-05 UPDATE (#2453): 900-series bidirectional
    annual-energy over-prediction routed to GaugeSolver (#1465 / #1462).
  * Case 920 heating + cooling -- engine 2.400 / 1.085 MWh vs bands
    [3.213, 4.347] / [2.189, 2.961] MWh (gaps 21.51 % / 42.87 %). Part of
    the LIMIT-05 UPDATE #2453 cohort (900-series bidirectional annual-energy
    over/under-prediction); routed to GaugeSolver (#1465 / #1462).
  * Case 950 cooling -- engine 0.028 MWh vs band [0.557, 0.753] MWh
    (gap 80.76 % of band midpoint). docs/KNOWN_ISSUES.md Sec LIMIT-24 / #3551
    (HVAC-mode annual cooling companion to Sec LIMIT-17 / #3058 free-floating
    winter-min); structural night-vent mass-coupling signature routed to
    GaugeSolver (#1465 / #1462). Case 950 heating is currently ``pass``
    (engine 0.000 MWh, band [0.000, 0.000] MWh -- heating is OFF by spec).
  * Case 960 heating + cooling -- engine 2.924 / 0.144 MWh vs bands
    [1.742, 2.357] / [1.840, 2.490] MWh (gaps 27.67 % / 78.34 %). Part of
    the LIMIT-14 / #3061 cohort (Case 960 sunspace inter-zone structural
    signature); routed to GaugeSolver (#1465 / #1462).
  * Case 970 heating + cooling -- engine 3.580 / 1.654 MWh vs bands
    [10.540, 14.260] / [7.391, 9.999] MWh (gaps 56.13 % / 65.98 %).
    docs/KNOWN_ISSUES.md Sec LIMIT-23 / #3552 (5-zone multi-zone cross-
    coupling bidirectional annual-energy over/under-prediction); routed to
    GaugeSolver (#1465 / #1462).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import pytest

# repo/tests/ashrae_140_output_validation/test_annual_energy.py -> repo/tests
REF_DIR = Path(__file__).resolve().parent.parent / "reference_data" / "zone_balance"


# Cases covered by this Python-side honesty guard. Mirrors the per-case CSV
# ``tests/reference_data/zone_balance/case_{N}_energy_reference.csv`` and the
# per-case ``{case}_{metric}`` entries in ``strict_energy_gate_baseline.json``.
# Adding a new case requires (1) the reference CSV, (2) a baseline entry, and
# (3) a structural LIMIT-* entry in ``docs/KNOWN_ISSUES.md`` (or a follow-up to
# add one) -- see issue #3572 for the acceptance template.
COVERED_CASES = ("600", "900", "920", "950", "960", "970")


def _load_band(csv_path: Path) -> Dict[str, dict]:
    """Load the ASHRAE 140 reference band rows for one case."""
    bands: Dict[str, dict] = {}
    with open(csv_path) as fh:
        rows = [ln for ln in fh if not ln.lstrip().startswith("#")]
    for r in csv.DictReader(rows):
        bands[r["metric"]] = {
            "unit": r["unit"],
            "ref_min": float(r["ref_min"]),
            "ref_max": float(r["ref_max"]),
            "midpoint": float(r["ref_midpoint"]),
            "tolerance_pct": float(r["tolerance_pct"]),
            "accept_min": float(r["accept_min"]),
            "accept_max": float(r["accept_max"]),
        }
    return bands


def _load_engine_values(json_path: Path) -> Dict[str, Dict[str, dict]]:
    """Load recorded engine values + canonical gap figures from the
    strict-energy-gate baseline for every COVERED_CASES."""
    metrics = json.loads(json_path.read_text())["metrics"]
    out: Dict[str, Dict[str, dict]] = {}
    for case in COVERED_CASES:
        out[case] = {
            "heating": metrics[f"case_{case}_heating"],
            "cooling": metrics[f"case_{case}_cooling"],
        }
    return out


BANDS: Dict[str, Dict[str, dict]] = {
    case: _load_band(REF_DIR / f"case_{case}_energy_reference.csv")
    for case in COVERED_CASES
}
ENGINE = _load_engine_values(REF_DIR / "strict_energy_gate_baseline.json")

# Known-issue references keyed by (case, metric) for the xfail reasons.
# Routes each gap to its docs/KNOWN_ISSUES.md section + upstream issue.
KNOWN_GAP_REFS = {
    ("600", "cooling"): (
        "docs/KNOWN_ISSUES.md Sec LIMIT-05 / Sec SOLAR-02 (issues #1457 / #2239)"
    ),
    ("900", "cooling"): (
        "docs/KNOWN_ISSUES.md Sec SOLAR-02 UPDATE (#2239) + "
        "Sec LIMIT-05 UPDATE (#2453): GaugeSolver #1465"
    ),
    ("920", "heating"): (
        "docs/KNOWN_ISSUES.md Sec LIMIT-05 UPDATE (#2453): "
        "900-series bidirectional annual-energy cohort; GaugeSolver #1465 / #1462"
    ),
    ("920", "cooling"): (
        "docs/KNOWN_ISSUES.md Sec LIMIT-05 UPDATE (#2453): "
        "900-series bidirectional annual-energy cohort; GaugeSolver #1465 / #1462"
    ),
    ("950", "cooling"): (
        "docs/KNOWN_ISSUES.md Sec LIMIT-24 / #3551 (HVAC-mode annual cooling) "
        "companion to Sec LIMIT-17 / #3058 (950FF night-vent free-floating); "
        "GaugeSolver #1465 / #1462"
    ),
    ("960", "heating"): (
        "docs/KNOWN_ISSUES.md Sec LIMIT-14 / #3061 (Case 960 sunspace "
        "inter-zone structural signature); GaugeSolver #1465 / #1462"
    ),
    ("960", "cooling"): (
        "docs/KNOWN_ISSUES.md Sec LIMIT-14 / #3061 (Case 960 sunspace "
        "inter-zone structural signature); GaugeSolver #1465 / #1462"
    ),
    ("970", "heating"): (
        "docs/KNOWN_ISSUES.md Sec LIMIT-23 / #3552 (5-zone multi-zone cross-"
        "coupling bidirectional annual-energy); GaugeSolver #1465 / #1462"
    ),
    ("970", "cooling"): (
        "docs/KNOWN_ISSUES.md Sec LIMIT-23 / #3552 (5-zone multi-zone cross-"
        "coupling bidirectional annual-energy); GaugeSolver #1465 / #1462"
    ),
}


def _annual_energy_params() -> List[pytest.param]:
    """Build parametrized cases. Tight-pass metrics carry no mark; known-gap
    metrics carry ``xfail(strict=True)`` so a future in-band value XPASS-fails
    the suite (signal to drop the marker) rather than silently going green."""
    params: List[pytest.param] = []
    for case in COVERED_CASES:
        for metric in ("heating", "cooling"):
            band = BANDS[case][f"annual_{metric}"]
            value = ENGINE[case][metric]["value_mwh"]
            in_band = band["accept_min"] <= value <= band["accept_max"]
            marks = ()
            if not in_band:
                gap = ENGINE[case][metric]["gap_pct_of_mid"]
                reason = (
                    f"known structural gap: engine {value:.3f} MWh vs "
                    f"+/-{band['tolerance_pct']:.0f}% band "
                    f"[{band['accept_min']:.3f}, {band['accept_max']:.3f}] MWh "
                    f"(gap {gap:.2f}% of band midpoint). "
                    f"{KNOWN_GAP_REFS[(case, metric)]}."
                )
                marks = (pytest.mark.xfail(strict=True, reason=reason),)
            params.append(
                pytest.param(
                    case,
                    metric,
                    band,
                    value,
                    marks=marks,
                    id=f"case-{case}-annual-{metric}",
                )
            )
    return params


class TestASHRAE140AnnualEnergy:
    """Annual HVAC energy for Cases 600/900/920/950/960/970 vs the ASHRAE 140 band.

    Passing metrics assert the real +/-15 % band (no wide tolerance).
    Known-gap metrics xfail(strict=True) on the SAME band assertion.
    """

    @pytest.mark.parametrize("case,metric,band,engine_value", _annual_energy_params())
    def test_annual_energy_within_ashrae140_band(
        self, case: str, metric: str, band: dict, engine_value: float
    ):
        """Engine annual energy must lie within the ASHRAE 140 +/-15 % band."""
        assert band["accept_min"] <= engine_value <= band["accept_max"], (
            f"Case {case} annual {metric}: engine {engine_value:.3f} MWh is "
            f"outside the ASHRAE 140 +/-{band['tolerance_pct']:.0f}% band "
            f"[{band['accept_min']:.3f}, {band['accept_max']:.3f}] MWh "
            f"(published range [{band['ref_min']:.2f}, {band['ref_max']:.2f}] "
            f"MWh, midpoint {band['midpoint']:.3f} MWh)."
        )


# NOTE on free-floating (600FF / 900FF / 950FF) energy:
#   A free-floating case has no HVAC system, so its annual HVAC energy is
#   structurally exactly 0.0 MWh -- a definition, not a comparison. The previous
#   ``TOLERANCES["free_floating"] = 100.0`` was dead config (never used by any
#   assertion) dressing up a tautology. Rather than assert ``0.0 == 0.0`` here,
#   the LIVE engine check that free-floating mode emits zero HVAC demand lives
#   in the Rust suite: ``tests/case_900ff_multinode_validation.rs::
#   test_case_900ff_multinode_free_floating_zero_hvac_demand`` (and free-floating
#   TEMPERATURE validation, KNOWN_ISSUES Sec FREE-01/02/03, lives in
#   ``tests/ashrae_140_free_floating.rs`` / ``tests/validation/free_floating_tests.rs``.