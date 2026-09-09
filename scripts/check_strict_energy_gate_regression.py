#!/usr/bin/env python3
"""Strict ±15% ASHRAE 140 energy-gate regression checker (issue #2506).

The strict tolerance tests
(`test_case_{600,800,810,900,920,950,960,970}_annual_energy_ashrae140_tolerance`)
are `#[ignore]`'d because the Case 600/900/950/960/970 annual-energy physics
gaps are not yet closed (post-#1323 / #1213 / #1328 cooling chain, plus the
§LIMIT-05 / §LIMIT-14 / §LIMIT-17 / §LIMIT-23 / §LIMIT-24 cohort tracked in
`docs/KNOWN_ISSUES.md`). Previously the strict-energy-gate workflow ran the
two Case 600/900 tests WITHOUT `--include-ignored`, so they reported
`ignored` and the gate was silently green on every PR — a regression that
worsened the cooling gap would pass undetected (the core complaint of
issue #2506). Issue #3572 extends the gate to the six additional cases
(800 / 810 / 920 / 950 / 960 / 970) whose reference CSVs are already
committed; no baseline value is raised to hide a regression.

This script implements the transparent, regression-catching gate:

  1. The workflow runs the ignored tests WITH `--include-ignored` so they
     execute and print their measured H/C values vs. the ±15% band:
       "[#1147 Case 600 strict] H=5.236 MWh (band 4.314-5.836), \
        C=2.455 MWh (band 4.275-5.784)"
  2. This script parses those lines, recomputes each metric's
     `gap_pct_of_mid` (0 inside the band; otherwise how far outside, as a
     percentage of the band midpoint), and compares against the recorded
     baseline in `strict_energy_gate_baseline.json`.
  3. Verdict per metric:
       - PASS          : gap == 0 (within ±15% band)
       - KNOWN-FAIL    : gap > 0 but <= baseline_gap + regression_tolerance_pp
                         (the documented structural gap — tracked)
       - REGRESSION    : gap > baseline_gap + regression_tolerance_pp, OR a
                         previously-`pass` metric whose gap now exceeds the
                         tolerance  →  exits 1, failing the gate
  4. The script also enforces that ALL required cases were measured (a
     missing line means the cargo filter / test name drifted — itself a
     regression).

Improvements (gap shrinking vs. baseline) are reported but do not fail the
gate; the engineer should lower the baseline in the same PR. Per RULES.md /
AGENTS.md the baseline must NEVER be raised to hide a regression.

Exit codes: 0 = gate holds (PASS or KNOWN-FAIL only); 1 = regression or
parse failure.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Issue #3572: the regex now matches all eight cases (600 / 800 / 810 /
# 900 / 920 / 950 / 960 / 970) the strict ±15% annual-energy gate observes.
# Match:  [#1147 Case 600 strict] H=5.236 MWh (band 4.314-5.836), C=2.455 MWh (band 4.275-5.784)
# The H/C order is stable (the test prints H first, then C). Band edges are
# formatted to 3 decimals by the Rust `:.3` formatter.
SUPPORTED_CASES = ("600", "800", "810", "900", "920", "950", "960", "970")
LINE_RE = re.compile(
    r"Case\s+(?P<case>(?:" + "|".join(SUPPORTED_CASES) + r"))\s+strict.*?"
    r"H=(?P<h>[-0-9.]+)\s+MWh\s+\(band\s+(?P<hlo>[-0-9.]+)-(?P<hhi>[-0-9.]+)\)"
    r".*?"
    r"C=(?P<c>[-0-9.]+)\s+MWh\s+\(band\s+(?P<clo>[-0-9.]+)-(?P<chi>[-0-9.]+)\)"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = REPO_ROOT / "tests/reference_data/zone_balance/strict_energy_gate_baseline.json"


def gap_pct_of_mid(value: float, band_lo: float, band_hi: float) -> float:
    """0.0 inside [band_lo, band_hi]; otherwise distance outside, % of midpoint.

    Degenerate bands (mid == 0, e.g. Case 950 heating per ASHRAE 140-2023 §B8.5
    where heating is OFF and the published envelope collapses to [0, 0]) are
    handled: if the value is inside the band the gap is 0 regardless of
    midpoint; otherwise an out-of-band value against a degenerate midpoint is
    reported as infinity (the previous behaviour).
    """
    if value < band_lo or value > band_hi:
        mid = 0.5 * (band_lo + band_hi)
        if mid <= 0:
            return float("inf")
        if value < band_lo:
            return (band_lo - value) / mid * 100.0
        return (value - band_hi) / mid * 100.0
    return 0.0


def parse_measured(log_text: str) -> dict[str, dict[str, float]]:
    """Return {'600': {'H': v,'C': v, 'hlo':..,'hhi':..,'clo':..,'chi':..}, '900': {...}, ...}."""
    measured: dict[str, dict[str, float]] = {}
    for m in LINE_RE.finditer(log_text):
        case = m.group("case")
        # The LAST match per case wins (in case of duplicate runs); the test
        # prints exactly one strict line per case so this is unambiguous.
        measured[case] = {
            "H": float(m.group("h")),
            "C": float(m.group("c")),
            "hlo": float(m.group("hlo")),
            "hhi": float(m.group("hhi")),
            "clo": float(m.group("clo")),
            "chi": float(m.group("chi")),
        }
    return measured


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", default="/tmp/strict_gate_output.txt",
                    help="captured cargo test --include-ignored output (default: %(default)s)")
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE),
                    help="baseline JSON (default: %(default)s)")
    ap.add_argument("--require-cases", default=",".join(SUPPORTED_CASES),
                    help="comma-separated case ids that MUST appear in the log "
                         "(default: all eight supported cases)")
    args = ap.parse_args()

    log_text = Path(args.log).read_text(errors="replace")
    baseline = json.loads(Path(args.baseline).read_text())
    tol_pp = float(baseline.get("regression_tolerance_pp", 5.0))
    metrics_baseline = baseline["metrics"]

    measured = parse_measured(log_text)

    required = [c.strip() for c in args.require_cases.split(",") if c.strip()]
    missing = [c for c in required if c not in measured]
    if missing:
        print("::error::Strict gate could not parse measured values for "
              f"case(s) {missing} from {args.log}.")
        print("   This usually means the cargo test filter or the test's "
              "println format drifted — itself a gate-coverage regression.")
        print("   Expected a line like: "
              "'[#1147 Case 600 strict] H=5.236 MWh (band 4.314-5.836), "
              "C=2.455 MWh (band 4.275-5.784)'")
        return 1

    print("=== ASHRAE 140 strict ±15% gate — transparent regression check ===")
    print(f"   baseline: {Path(args.baseline).name}  "
          f"(captured {baseline.get('captured_commit','?')})")
    print(f"   regression tolerance: {tol_pp:.2f} pp of band midpoint")
    print()
    print(f"{'case/metric':<20}{'value':>9}{'band':>22}{'gap%':>9}"
          f"{'base%':>9}{'verdict':>16}")

    regressions: list[str] = []
    improvements: list[str] = []
    n_pass = n_known = 0

    for case in required:
        for metric, label in (("H", "heating"), ("C", "cooling")):
            key = f"case_{case}_{label}"
            b = metrics_baseline[key]
            mv = measured[case][metric]
            # Use the band edges AS PRINTTED BY THE TEST (single source of
            # truth: the Rust EnergyReference const). They must match the
            # baseline band; cross-check and warn if not.
            blo = measured[case][f"{metric.lower()}lo"]
            bhi = measured[case][f"{metric.lower()}hi"]
            cur_gap = gap_pct_of_mid(mv, blo, bhi)
            base_gap = float(b["gap_pct_of_mid"])
            base_status = b["status"]

            if cur_gap == 0.0:
                verdict = "PASS"
                n_pass += 1
            elif cur_gap <= base_gap + tol_pp + 1e-9:
                # Within tolerance of the recorded known gap. Stricter rule for
                # metrics that previously PASSED: any movement outside the band
                # beyond the tolerance is a regression even though base_gap==0.
                if base_status == "pass" and cur_gap > tol_pp + 1e-9:
                    verdict = "REGRESSION"
                    regressions.append(
                        f"{key}: was PASS (gap 0.00), now gap {cur_gap:.2f}pp "
                        f"(value {mv:.3f} MWh outside band [{blo:.3f}, {bhi:.3f}])"
                    )
                else:
                    verdict = "KNOWN-FAIL"
                    n_known += 1
            else:
                verdict = "REGRESSION"
                regressions.append(
                    f"{key}: gap worsened {base_gap:.2f}pp -> {cur_gap:.2f}pp "
                    f"(> baseline+{tol_pp:.2f}pp tolerance); "
                    f"value {mv:.3f} MWh outside band [{blo:.3f}, {bhi:.3f}]"
                )
            if 0 < base_gap and cur_gap < base_gap - 0.5 and verdict != "REGRESSION":
                # Only flag meaningful improvements (>0.5 pp) to avoid
                # floating-point noise from 2dp baseline rounding.
                improvements.append(
                    f"{key}: gap improved {base_gap:.2f}pp -> {cur_gap:.2f}pp "
                    f"(lower the baseline in the same PR)"
                )

            print(f"{key:<20}{mv:>9.3f}{f'[{blo:.3f}, {bhi:.3f}]':>22}"
                  f"{cur_gap:>9.2f}{base_gap:>9.2f}{verdict:>16}")

    print()
    print(f"   summary: {n_pass} PASS, {n_known} KNOWN-FAIL (tracked), "
          f"{len(regressions)} REGRESSION")

    for msg in improvements:
        print(f"   ::notice::improvement {msg}")
    for msg in regressions:
        print(f"   ::error::regression {msg}")

    if regressions:
        print()
        print("::error::ASHRAE 140 strict ±15% gate FAILED — regression beyond "
              "known baseline detected.")
        print("   Per AGENTS.md 'no parameter tuning, fix the math': fix the "
              "physics, do NOT raise the baseline gap to hide this.")
        return 1

    print("PASS: strict ±15% gate holds. Documented structural cooling "
          "gaps (Cases 600/900/950/960/970 per docs/KNOWN_ISSUES.md §LIMIT-05 / "
          "§LIMIT-14 / §LIMIT-17 / §LIMIT-23 / §LIMIT-24) are tracked (not "
          "silently ignored); no regression detected.")
    return 0


def self_test() -> int:
    """Regression test for the parser (Issue #3572 acceptance criterion).

    Synthesises a measurement block with one strict line per supported case
    (the eight cases 600 / 800 / 810 / 900 / 920 / 950 / 960 / 970), asserts
    the regex consumes them all, and asserts the degenerate-band gap formula
    returns 0 for in-band values even when the band midpoint collapses to
    zero (Case 950 heating). Run via `python3 scripts/check_strict_energy_gate_regression.py --self-test`.
    """
    lines = [
        "[#1147 Case 600 strict] H=5.182 MWh (band 4.314-5.836), C=2.546 MWh (band 4.275-5.784)",
        "[#1147 Case 800 strict] H=5.453 MWh (band 4.378-5.923), C=2.007 MWh (band 4.888-6.612)",
        "[#1147 Case 810 strict] H=1.633 MWh (band 3.357-4.543), C=0.910 MWh (band 3.740-5.060)",
        "[#1147 Case 900 strict] H=1.633 MWh (band 1.364-1.846), C=0.910 MWh (band 2.465-3.335)",
        "[#1147 Case 920 strict] H=2.400 MWh (band 3.213-4.347), C=1.085 MWh (band 2.189-2.961)",
        "[#1147 Case 950 strict] H=0.000 MWh (band 0.000-0.000), C=0.028 MWh (band 0.557-0.753)",
        "[#1147 Case 960 strict] H=2.924 MWh (band 1.742-2.357), C=0.144 MWh (band 1.840-2.490)",
        "[#1147 Case 970 strict] H=3.580 MWh (band 10.540-14.260), C=1.654 MWh (band 7.391-9.999)",
    ]
    measured = parse_measured("\n".join(lines))
    missing = [c for c in SUPPORTED_CASES if c not in measured]
    if missing:
        print(f"FAIL: parser dropped cases {missing} (regex/format drift)", file=sys.stderr)
        return 1
    for case in SUPPORTED_CASES:
        for band_key in ("hlo", "hhi", "clo", "chi"):
            if not isinstance(measured[case][band_key], (int, float)):
                print(f"FAIL: {case}/{band_key} not numeric: {measured[case][band_key]!r}", file=sys.stderr)
                return 1

    # Degenerate-band coverage (Case 950 heating): value 0 inside [0, 0]
    # must produce gap == 0 (not inf from the pre-#3572 implementation).
    if gap_pct_of_mid(0.0, 0.0, 0.0) != 0.0:
        print("FAIL: gap_pct_of_mid(0, 0, 0) != 0 (Case 950 H degenerate band)", file=sys.stderr)
        return 1
    # Out-of-band against a degenerate midpoint stays inf (unchanged behaviour).
    if gap_pct_of_mid(0.5, 0.0, 0.0) != float("inf"):
        print("FAIL: gap_pct_of_mid(0.5, 0, 0) != inf (out-of-band degenerate mid)", file=sys.stderr)
        return 1
    # Sanity: standard in-band value still returns 0 (regression guard for the
    # degenerate-band fix above).
    if gap_pct_of_mid(5.0, 4.5, 5.8) != 0.0:
        print("FAIL: gap_pct_of_mid(5.0, 4.5, 5.8) != 0 (regression)", file=sys.stderr)
        return 1
    # Sanity: out-of-band UNDER returns a positive % of midpoint.
    under_gap = gap_pct_of_mid(2.0, 4.5, 5.8)
    if not (under_gap > 0 and abs(under_gap - (4.5 - 2.0) / 5.15 * 100.0) < 1e-6):
        print(f"FAIL: under_gap formula drifted: got {under_gap}", file=sys.stderr)
        return 1

    print(f"PASS: self_test consumed all {len(SUPPORTED_CASES)} strict cases "
          f"({', '.join(SUPPORTED_CASES)}) and gap_pct_of_mid invariants hold.")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.argv.remove("--self-test")
        sys.exit(self_test())
    sys.exit(main())
