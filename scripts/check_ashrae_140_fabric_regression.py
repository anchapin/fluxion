#!/usr/bin/env python3
"""PR-A+2 fabric harness regression gate (Refs #3986-A+2 / #4117).

The fabric harness
(`tests/all_tests/ashrae_140_fabric_multiselector.rs`) exercises
Cases 600 / 900 / 950 across the TwoROneC + NineRFourC selectors via
`run_blind_annual_energy` and prints parseable lines of the form:

    [#3986-A+2 Case 600 / 5r1c] H=4.901 MWh
    [#3986-A+2 Case 600 / 5r1c] C=2.620 MWh
    [#3986-A+2 Case 600 / 9r4c] H=4.558 MWh
    [#3986-A+2 Case 600 / 9r4c] C=4.858 MWh
    [#3986-A+2 Case 600 / parity] ratio_H=1.0753 ratio_C=0.5393

This script parses those lines, compares the per-component annual
energy and the per-case selector-parity ratios against the recorded
baseline at
`tests/reference_data/ashrae_140_fabric/baseline.json`, and FAILS on
regression beyond the recorded tolerance. It mirrors the contract of
`scripts/check_strict_energy_gate_regression.py` (issue #2506) but
operates on the fabric harness, not the strict ±15% ASHRAE band.

What this gate catches:

* **Selector wiring drift** — a regression that flips the dispatch
  path on one (case, selector) combination.
* **Selector-parity drift** — the `parity` row records the ratio
  (FiveROneC value) / (NineRFourC value) per case. A 10pp shift in
  the ratio means a selector silently changed paths.

What this gate does NOT catch (already covered by sibling gates):

* ASHRAE 140 ±15% band violations — `check_strict_energy_gate_regression.py`.
* Absolute-value physics regressions affecting both selectors
  identically — covered for Cases 600/800/810/900/920/950/960/970 by
  the strict-energy-gate workflow.

Exit codes: 0 = gate holds (PASS or KNOWN-FAIL); 1 = regression or
parse failure; 2 = argument / baseline error.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ---------- Constants ----------

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = (
    REPO_ROOT / "tests" / "reference_data" / "ashrae_140_fabric" / "baseline.json"
)

# The fabric harness covers Cases 600, 900, 950. Case 600 cooling is a
# known structural gap (#1147) — the gate STILL observes it because
# selector-parity drift on a known-fail metric is a real regression
# signal (the ratio must stay bounded even when both absolute values
# are out of band).
SUPPORTED_CASES: tuple[str, ...] = ("600", "900", "950")
SUPPORTED_SELECTORS: tuple[str, ...] = ("5r1c", "9r4c")
SUPPORTED_METRICS: tuple[str, ...] = ("H", "C")

# Lines are printed by `tests/all_tests/ashrae_140_fabric_multiselector.rs`.
# The metric-line regex matches the four per-component measurement lines.
# The parity-line regex matches the per-case selector-parity ratio lines.
_LINE_RE = re.compile(
    r"\[#3986-A\+2\s+Case\s+(?P<case>"
    + "|".join(SUPPORTED_CASES)
    + r")\s+/\s+(?P<selector>"
    + "|".join(SUPPORTED_SELECTORS)
    + r")\]\s+(?P<metric>H|C)=(?P<value>[-0-9.]+)\s+MWh"
)
_PARITY_RE = re.compile(
    r"\[#3986-A\+2\s+Case\s+(?P<case>"
    + "|".join(SUPPORTED_CASES)
    + r")\s+/\s+parity\]\s+ratio_H=(?P<ratio_h>[-0-9.]+)\s+ratio_C=(?P<ratio_c>[-0-9.]+)"
)


# ---------- Pure parsing functions ----------


def parse_measurements(
    log_text: str,
) -> tuple[dict[tuple[str, str, str], float], dict[str, dict[str, float]]]:
    """Parse a captured fabric-harness log.

    Returns:
        measurements: {(case, selector, metric): value_MWh} for the
                      per-component lines.
        parity:       {case: {'ratio_H': float, 'ratio_C': float}} for
                      the per-case parity lines.
    """
    measurements: dict[tuple[str, str, str], float] = {}
    for m in _LINE_RE.finditer(log_text):
        key = (m.group("case"), m.group("selector"), m.group("metric"))
        measurements[key] = float(m.group("value"))

    parity: dict[str, dict[str, float]] = {}
    for m in _PARITY_RE.finditer(log_text):
        parity[m.group("case")] = {
            "ratio_H": float(m.group("ratio_h")),
            "ratio_C": float(m.group("ratio_c")),
        }

    return measurements, parity


# ---------- Verdict ----------

PASS = "PASS"
REGRESSION = "REGRESSION"
KNOWN_FAIL = "KNOWN-FAIL"


def _verdict(
    cur: float,
    base: float,
    tol_rel_pct: float,
) -> str:
    """Per-measurement verdict against the recorded baseline.

    Degenerate cases:
    - `base == 0` (e.g. Case 950 heating): if `cur == 0`, PASS.
      Otherwise the drift is unbounded relative to a zero baseline;
      we treat any non-zero `cur` as REGRESSION (the recorded value
      says heating is OFF; if a future change makes heating ON, that
      is a structural shift).
    """
    if base == 0.0:
        if cur == 0.0:
            return PASS
        return REGRESSION

    delta_pct = abs(cur - base) / abs(base) * 100.0
    if delta_pct <= tol_rel_pct + 1e-9:
        return PASS
    return REGRESSION


def _ratio_verdict(
    cur: float,
    base: float,
    tol_abs: float,
) -> str:
    """Per-case selector-parity ratio verdict.

    Captures the structural relationship between selectors. Case 600
    shows non-unity ratios (5R1C vs 9R4C diverge on low-mass specs);
    Cases 900/950 show exactly 1.0 (high-mass auto-promotion collapses
    both selectors to the same network per ADR-0017). A 10pp shift in
    any case's ratio means a selector silently changed paths.
    """
    delta = abs(cur - base)
    if delta <= tol_abs + 1e-9:
        return PASS
    return REGRESSION


def aggregate(
    measurements: dict[tuple[str, str, str], float],
    parity: dict[str, dict[str, float]],
    baseline: dict,
) -> tuple[str, list[str]]:
    """Run the verdict. Returns (overall_verdict, log_lines)."""
    tol_rel = float(baseline["regression_tolerance_rel_pct"])
    tol_ratio_abs = float(baseline["regression_tolerance_ratio_abs"])
    metrics_baseline: dict = baseline["metrics"]
    # Parity entries are keyed `case_NNN` in the JSON; normalise to just
    # `NNN` so the loop's `case in parity_baseline` check works against
    # the same key shape as the live `parity` dict.
    parity_baseline_raw: dict = baseline["parity_ratios"]
    parity_baseline: dict[str, dict[str, float]] = {}
    for k, v in parity_baseline_raw.items():
        if not k.startswith("case_"):
            continue  # skip `_doc` and other non-data fields
        parity_baseline[k[len("case_"):]] = v

    log: list[str] = []
    log.append("=== ASHRAE 140 fabric harness — selector-parity drift gate ===")
    log.append(
        f"   baseline: tests/reference_data/ashrae_140_fabric/baseline.json "
        f"(captured {baseline.get('captured_commit', '?')})"
    )
    log.append(
        f"   per-measurement tolerance: {tol_rel:.2f}% relative; "
        f"per-ratio tolerance: ±{tol_ratio_abs:.4f} absolute"
    )
    log.append("")
    log.append(
        f"{'case/selector/metric':<30}{'value':>10}{'baseline':>10}"
        f"{'delta%':>10}{'verdict':>12}"
    )

    regressions: list[str] = []

    # Coverage check: every (case, selector, metric) we declared must
    # appear in the log. A missing line is itself a regression (filter
    # drift / cargo test name change).
    required_keys: list[tuple[str, str, str]] = []
    for case in SUPPORTED_CASES:
        for selector in SUPPORTED_SELECTORS:
            for metric in SUPPORTED_METRICS:
                required_keys.append((case, selector, metric))

    missing = [k for k in required_keys if k not in measurements]
    if missing:
        for key in missing:
            case, sel, met = key
            regressions.append(
                f"missing measurement: case={case} selector={sel} metric={met} "
                f"(cargo test filter / print format drifted)"
            )
            log.append(f"  {str(key):<30}{'MISSING':>10}")

    # Per-measurement verdict.
    for key in required_keys:
        if key not in measurements:
            continue
        case, selector, metric = key
        metric_key = f"case_{case}_{selector}"
        b = metrics_baseline.get(metric_key)
        if b is None:
            regressions.append(
                f"baseline missing entry: {metric_key} "
                f"(update tests/reference_data/ashrae_140_fabric/baseline.json)"
            )
            continue
        base_val = float(b["H_mwh"] if metric == "H" else b["C_mwh"])
        cur_val = measurements[key]
        verdict = _verdict(cur_val, base_val, tol_rel)

        if base_val != 0.0:
            delta_pct = abs(cur_val - base_val) / abs(base_val) * 100.0
            delta_str = f"{delta_pct:.2f}%"
        else:
            delta_str = "n/a (base=0)"
        log.append(
            f"  {metric_key:<30}{cur_val:>10.3f}{base_val:>10.3f}"
            f"{delta_str:>10}{verdict:>12}"
        )

        if verdict == REGRESSION:
            regressions.append(
                f"{metric_key} {metric}: value drifted {cur_val:.3f} MWh vs "
                f"baseline {base_val:.3f} MWh (> {tol_rel:.2f}% relative tolerance)"
            )

    log.append("")
    log.append("--- per-case selector-parity ratios ---")
    log.append(
        f"{'case':<10}{'ratio_H':>10}{'base_H':>10}{'delta_H':>12}"
        f"{'ratio_C':>10}{'base_C':>10}{'delta_C':>12}{'verdict':>12}"
    )

    for case in SUPPORTED_CASES:
        if case not in parity:
            regressions.append(
                f"missing parity line for case {case} (cargo test filter / print drift)"
            )
            log.append(f"  {case:<10}{'MISSING':>10}")
            continue
        if case not in parity_baseline:
            regressions.append(
                f"baseline missing parity for case {case}"
            )
            continue
        cur = parity[case]
        base = parity_baseline[case]
        base_h, base_c = float(base["ratio_H"]), float(base["ratio_C"])
        cur_h, cur_c = float(cur["ratio_H"]), float(cur["ratio_C"])

        v_h = _ratio_verdict(cur_h, base_h, tol_ratio_abs)
        v_c = _ratio_verdict(cur_c, base_c, tol_ratio_abs)
        verdict = REGRESSION if (v_h == REGRESSION or v_c == REGRESSION) else PASS

        d_h = abs(cur_h - base_h)
        d_c = abs(cur_c - base_c)
        log.append(
            f"  {case:<10}{cur_h:>10.4f}{base_h:>10.4f}{d_h:>12.4f}"
            f"{cur_c:>10.4f}{base_c:>10.4f}{d_c:>12.4f}{verdict:>12}"
        )

        if verdict == REGRESSION:
            if v_h == REGRESSION:
                regressions.append(
                    f"case {case} ratio_H drifted: {cur_h:.4f} vs baseline "
                    f"{base_h:.4f} (delta {d_h:.4f} > {tol_ratio_abs:.4f})"
                )
            if v_c == REGRESSION:
                regressions.append(
                    f"case {case} ratio_C drifted: {cur_c:.4f} vs baseline "
                    f"{base_c:.4f} (delta {d_c:.4f} > {tol_ratio_abs:.4f})"
                )

    log.append("")
    if regressions:
        log.append(
            f"FAIL: {len(regressions)} regression(s) detected against the recorded baseline."
        )
        for msg in regressions:
            log.append(f"  ::error::regression {msg}")
        log.append(
            "  Per AGENTS.md 'no parameter tuning, fix the math': fix the"
            " selector wiring / dispatch path, do NOT raise the baseline"
            " value to hide this."
        )
        return REGRESSION, log

    log.append("PASS: fabric harness holds. All (case × selector × metric)")
    log.append("       measurements and selector-parity ratios within tolerance.")
    log.append(
        "       Documented structural gaps (Case 600 cooling per #1147)"
        " are tracked, not silently ignored."
    )
    return PASS, log


# ---------- CLI ----------


def main(argv: list[str] | None = None) -> int:
    global SUPPORTED_CASES  # noqa: PLW0603
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "log",
        nargs="?",
        default="/tmp/fabric_output.txt",
        help="captured cargo test --nocapture output of the fabric harness "
             "(default: %(default)s)",
    )
    ap.add_argument(
        "--baseline",
        default=str(DEFAULT_BASELINE),
        help="baseline JSON (default: %(default)s)",
    )
    ap.add_argument(
        "--require-cases",
        default=",".join(SUPPORTED_CASES),
        help="comma-separated case ids that MUST appear in the log "
             "(default: all three supported cases)",
    )
    args = ap.parse_args(argv)

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"ERROR: log file does not exist: {log_path}", file=sys.stderr)
        return 2
    log_text = log_path.read_text(errors="replace")

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"ERROR: baseline file does not exist: {baseline_path}", file=sys.stderr)
        return 2
    try:
        baseline = json.loads(baseline_path.read_text())
    except json.JSONDecodeError as e:
        print(f"ERROR: baseline file is not valid JSON: {e}", file=sys.stderr)
        return 2

    # Filter to only the requested cases.
    requested = [c.strip() for c in args.require_cases.split(",") if c.strip()]
    SUPPORTED_CASES = tuple(requested)

    measurements, parity = parse_measurements(log_text)
    verdict, log_lines = aggregate(measurements, parity, baseline)
    for line in log_lines:
        print(line)

    return 0 if verdict == PASS else 1


def self_test() -> int:
    """Regression test for the parser.

    Synthesises a measurement block with all three (cases × selectors ×
    metrics) lines plus the parity line, asserts the regex consumes them
    all, and asserts the verdict invariants hold (in-band → PASS;
    drift beyond tolerance → REGRESSION; degenerate base==0 → REGRESSION
    on non-zero current).
    """
    synth_log = """
[#3986-A+2 Case 600 / 5r1c] H=4.901 MWh
[#3986-A+2 Case 600 / 5r1c] C=2.620 MWh
[#3986-A+2 Case 600 / 9r4c] H=4.558 MWh
[#3986-A+2 Case 600 / 9r4c] C=4.858 MWh
[#3986-A+2 Case 600 / parity] ratio_H=1.0753 ratio_C=0.5393
[#3986-A+2 Case 900 / 5r1c] H=1.633 MWh
[#3986-A+2 Case 900 / 5r1c] C=0.910 MWh
[#3986-A+2 Case 900 / 9r4c] H=1.633 MWh
[#3986-A+2 Case 900 / 9r4c] C=0.910 MWh
[#3986-A+2 Case 900 / parity] ratio_H=1.0000 ratio_C=1.0000
[#3986-A+2 Case 950 / 5r1c] H=0.000 MWh
[#3986-A+2 Case 950 / 5r1c] C=0.250 MWh
[#3986-A+2 Case 950 / 9r4c] H=0.000 MWh
[#3986-A+2 Case 950 / 9r4c] C=0.250 MWh
[#3986-A+2 Case 950 / parity] ratio_H=1.0000 ratio_C=1.0000
"""
    measurements, parity = parse_measurements(synth_log)
    missing = [
        (c, s, m)
        for c in SUPPORTED_CASES
        for s in SUPPORTED_SELECTORS
        for m in SUPPORTED_METRICS
        if (c, s, m) not in measurements
    ]
    if missing:
        print(f"FAIL: parser dropped measurements {missing}", file=sys.stderr)
        return 1
    missing_parity = [c for c in SUPPORTED_CASES if c not in parity]
    if missing_parity:
        print(f"FAIL: parser dropped parity lines for cases {missing_parity}", file=sys.stderr)
        return 1

    # Verdict invariants.
    if _verdict(4.901, 4.901, 10.0) != PASS:
        print("FAIL: _verdict exact match should be PASS", file=sys.stderr)
        return 1
    if _verdict(5.30, 4.901, 10.0) != PASS:
        print("FAIL: _verdict within tolerance should be PASS", file=sys.stderr)
        return 1
    if _verdict(6.0, 4.901, 10.0) != REGRESSION:
        print("FAIL: _verdict beyond tolerance should be REGRESSION", file=sys.stderr)
        return 1
    if _verdict(0.0, 0.0, 10.0) != PASS:
        print("FAIL: _verdict (0, 0) should be PASS (Case 950 H degenerate)", file=sys.stderr)
        return 1
    if _verdict(0.5, 0.0, 10.0) != REGRESSION:
        print(
            "FAIL: _verdict (cur>0, base=0) should be REGRESSION "
            "(structural shift on a degenerate-band metric)",
            file=sys.stderr,
        )
        return 1

    # Ratio verdict.
    if _ratio_verdict(1.0753, 1.0753, 0.10) != PASS:
        print("FAIL: _ratio_verdict exact should be PASS", file=sys.stderr)
        return 1
    if _ratio_verdict(1.15, 1.0753, 0.10) != PASS:
        print("FAIL: _ratio_verdict within tolerance should be PASS", file=sys.stderr)
        return 1
    if _ratio_verdict(1.20, 1.0753, 0.10) != REGRESSION:
        print("FAIL: _ratio_verdict beyond tolerance should be REGRESSION", file=sys.stderr)
        return 1

    print(
        f"PASS: self_test consumed all "
        f"{len(SUPPORTED_CASES) * len(SUPPORTED_SELECTORS) * len(SUPPORTED_METRICS)} "
        f"measurements and {len(SUPPORTED_CASES)} parity lines; "
        f"verdict invariants hold."
    )
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.argv.remove("--self-test")
        sys.exit(self_test())
    sys.exit(main())
