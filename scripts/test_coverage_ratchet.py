#!/usr/bin/env python3
"""Self-test for the per-critical-path coverage ratchet (#1932, #2533, #2710).

Builds synthetic LCOV traces in a tempdir and exercises the gate:

  1. Line-only regression is still caught (existing #1932 behaviour).
  2. Branch regression is caught even when line holds (#2533).
  3. Branch unenforced (baseline 0.0) does NOT trip the gate.
  4. Branch baseline set but current run has 0 branches -> measurement
     regression fails loud.
  5. Both dimensions pass -> empty failure list.
  6. coverage_baseline.py ratchets branch one-way (never down).
  7. Absolute branch floor (#2710): branch below min_branch_floor FAILS
     even when the regression ratchet would pass.
  8. min_branch_floor unenforced (0.0 / absent) -> no floor failure.
  9. v1_3_target_branch is REPORTED but never causes a failure.
 10. build_baseline_payload carries floor + target forward verbatim (#2710).

Exits 0 on success, 1 on any assertion failure.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from coverage_baseline import build_baseline_payload  # noqa: E402
from coverage_critical_paths import (  # noqa: E402
    bucket_coverage,
    evaluate_gate,
    parse_lcov,
)


def write_lcov(path: Path, *, lf: int, lh: int, brf: int, brh: int, sf: str = "src/sim/solar.rs") -> None:
    path.write_text(
        f"SF:{sf}\n"
        f"LF:{lf}\nLH:{lh}\n"
        f"BRF:{brf}\nBRH:{brh}\n"
        "end_of_record\n",
        encoding="utf-8",
    )


def reports_for(lcov: Path) -> dict:
    files = parse_lcov(lcov)
    # parse_lcov skips files with LF==0; that's fine for these fixtures.
    return bucket_coverage(files)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="cov-ratchet-test-"))

    # --- Fixture: weather_solar path, one file -----------------------
    lcov = tmp / "lcov.info"
    # Line 80% (8/10), branch 50% (5/10)
    write_lcov(lcov, lf=10, lh=8, brf=10, brh=5, sf="src/sim/solar.rs")
    reports = reports_for(lcov)
    rep = reports["weather_solar"]
    assert rep.line_pct == 80.0, f"line pct {rep.line_pct}"
    assert rep.branch_pct == 50.0, f"branch pct {rep.branch_pct}"

    # 1. Line-only regression caught (baseline line 90, branch 0=unenforced)
    base = {"paths": {"weather_solar": {"line": 90.0, "branch": 0.0}}}
    fails = evaluate_gate(reports, base, 0.01)
    assert any("line coverage" in f and "weather_solar" in f for f in fails), fails
    print("[1/10] OK: line-only regression caught")

    # 2. Branch regression caught even when line holds (#2533).
    #    baseline line 80 (= current, passes), branch 70 (current 50, fails)
    base = {"paths": {"weather_solar": {"line": 80.0, "branch": 70.0}}}
    fails = evaluate_gate(reports, base, 0.01)
    assert any("branch coverage" in f and "weather_solar" in f for f in fails), fails
    # And line should NOT be in failures (it passed)
    assert not any("line coverage" in f for f in fails), fails
    print("[2/10] OK: branch regression caught while line held")

    # 3. Branch unenforced (baseline branch 0.0) -> no branch failure,
    #    even though branch is low.  Line passes too.
    base = {"paths": {"weather_solar": {"line": 80.0, "branch": 0.0}}}
    fails = evaluate_gate(reports, base, 0.01)
    assert fails == [], fails
    print("[3/10] OK: branch-unenforced path does not trip gate")

    # 4. Branch baseline set but current run has 0 branches -> fail loud.
    write_lcov(lcov, lf=10, lh=8, brf=0, brh=0, sf="src/sim/solar.rs")
    reports = reports_for(lcov)
    base = {"paths": {"weather_solar": {"line": 80.0, "branch": 70.0}}}
    fails = evaluate_gate(reports, base, 0.01)
    assert any("instrumented 0 branches" in f for f in fails), fails
    print("[4/10] OK: branch-baseline-set-but-no-branches fails loud")

    # 5. Both dimensions pass.
    write_lcov(lcov, lf=10, lh=9, brf=10, brh=9, sf="src/sim/solar.rs")
    reports = reports_for(lcov)
    base = {"paths": {"weather_solar": {"line": 89.0, "branch": 89.0}}}
    fails = evaluate_gate(reports, base, 0.01)
    assert fails == [], fails
    print("[5/10] OK: both dimensions pass -> no failures")

    # 6. build_baseline_payload ratchets branch one-way (never down).
    #    Previous baseline branch = 70; current measured = 50.
    write_lcov(lcov, lf=10, lh=8, brf=10, brh=5, sf="src/sim/solar.rs")
    reports = reports_for(lcov)
    previous = {
        "paths": {
            "weather_solar": {"line": 80.0, "branch": 70.0},
            "weather_ventilation": {"line": 0.0, "branch": 0.0},
            "conduction_zone": {"line": 0.0, "branch": 0.0},
            "hvac_zone": {"line": 0.0, "branch": 0.0},
            "overall": {"line": 80.0, "branch": 70.0},
        }
    }
    payload = build_baseline_payload(reports, previous)
    ws = payload["paths"]["weather_solar"]
    assert ws["branch"] == 70.0, f"branch should ratchet to 70 (max of 70,50), got {ws['branch']}"
    assert ws["line"] == 80.0, f"line should ratchet to 80, got {ws['line']}"
    print("[6/10] OK: branch ratchets one-way (70 stayed at 70, not 50)")

    # 7. Absolute branch floor (#2710): branch below min_branch_floor FAILS
    #    even when the regression ratchet would pass.
    #    current: line 90%, branch 65%.  baseline: line 90 (passes), branch 60
    #    (ratchet floor 59.4, passes).  But min_branch_floor = 70 -> FAILS.
    write_lcov(lcov, lf=10, lh=9, brf=20, brh=13, sf="src/sim/solar.rs")  # 65% branch
    reports = reports_for(lcov)
    assert reports["weather_solar"].branch_pct == 65.0, reports["weather_solar"].branch_pct
    base = {"paths": {"weather_solar": {
        "line": 90.0, "branch": 60.0, "min_branch_floor": 70.0,
    }}}
    fails = evaluate_gate(reports, base, 0.01)
    assert any("absolute minimum floor" in f and "weather_solar" in f for f in fails), fails
    # And the ratchet-only check (branch 65 >= 59.4) should NOT be in failures
    assert not any("fell below ratchet floor" in f and "branch" in f for f in fails), fails
    print("[7/10] OK: absolute floor fails below 70% even when ratchet passes")

    # 8. min_branch_floor unenforced (absent) -> no floor failure.
    #    Same 65% branch, same baseline branch 60, but NO min_branch_floor.
    write_lcov(lcov, lf=10, lh=9, brf=20, brh=13, sf="src/sim/solar.rs")
    reports = reports_for(lcov)
    base = {"paths": {"weather_solar": {"line": 90.0, "branch": 60.0}}}
    fails = evaluate_gate(reports, base, 0.01)
    assert fails == [], fails
    print("[8/10] OK: no floor field -> no floor failure (unenforced)")

    # 9. v1_3_target_branch is REPORTED but never causes a failure,
    #    even when current is far below target.
    write_lcov(lcov, lf=10, lh=9, brf=20, brh=13, sf="src/sim/solar.rs")  # 65% branch
    reports = reports_for(lcov)
    base = {"paths": {"weather_solar": {
        "line": 90.0, "branch": 60.0, "v1_3_target_branch": 75.0,
    }}}
    fails = evaluate_gate(reports, base, 0.01)
    assert fails == [], fails  # target must not fail
    print("[9/10] OK: v1.3 target reported but does not fail gate")

    # 10. build_baseline_payload carries floor + target forward verbatim.
    #     Previous baseline had min_branch_floor=60, v1_3_target_branch=75.
    write_lcov(lcov, lf=10, lh=8, brf=10, brh=5, sf="src/sim/solar.rs")
    reports = reports_for(lcov)
    previous = {
        "paths": {
            "weather_solar": {
                "line": 80.0, "branch": 70.0,
                "min_branch_floor": 60.0, "v1_3_target_branch": 75.0,
            },
            "weather_ventilation": {"line": 0.0, "branch": 0.0},
            "conduction_zone": {"line": 0.0, "branch": 0.0},
            "hvac_zone": {"line": 0.0, "branch": 0.0},
            "overall": {"line": 80.0, "branch": 70.0},
        }
    }
    payload = build_baseline_payload(reports, previous)
    ws = payload["paths"]["weather_solar"]
    assert ws["min_branch_floor"] == 60.0, f"floor should carry forward, got {ws['min_branch_floor']}"
    assert ws["v1_3_target_branch"] == 75.0, f"target should carry forward, got {ws['v1_3_target_branch']}"
    # And a path WITHOUT policy fields should get 0.0 defaults, not KeyError
    assert payload["paths"]["overall"]["min_branch_floor"] == 0.0
    assert payload["paths"]["overall"]["v1_3_target_branch"] == 0.0
    print("[10/10] OK: floor + target carried forward verbatim; absent -> 0.0")

    print("\nAll ratchet self-tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
