#!/usr/bin/env python3
"""ASHRAE 140-2023 validation pipeline for fluxion.

Orchestrates the repo's existing ASHRAE 140 tooling into one pass/fail run,
with energy-balance results surfaced FIRST in the report:

  1. provenance   scripts/fetch_ashrae140.py verify  -- are the licensed suite
                  files on disk the normative ones ASHRAE published?
  2. energy-balance (sacrosanct, checked before anything else is reported):
     a. the strict ±15% annual-energy gate: cargo test --include-ignored over
        tests/all_tests/zone_balance_eplus_isolation.rs, parsed and judged by
        scripts/check_strict_energy_gate_regression.py against the recorded
        baseline (never raised to hide a regression);
     b. the Python-side honesty guard tests/ashrae_140_output_validation,
        which asserts every recorded engine value sits in the ASHRAE band or
        in a strict xfail for a documented structural gap.
  3. engine runs  scripts/ashrae_benchmark_harness.py --compare against the
        stored per-target baseline (all case series, timed, pass/fail counts).
  4. report       stage-by-stage verdicts, JSON artifact, aggregated exit code.

This script runs other repo scripts; it does not re-implement their physics,
tolerances, or parsing. Per RULES.md it never tunes outputs to pass — a
REGRESSION here is a failure, and a KNOWN-FAIL is reported as exactly that.

Usage
-----
  python3 scripts/ashrae_140_pipeline.py                        # full run
  python3 scripts/ashrae_140_pipeline.py --skip-engine          # no cargo build/test
  python3 scripts/ashrae_140_pipeline.py --output report.json   # JSON artifact
  python3 scripts/ashrae_140_pipeline.py --list                 # show stages

Exit codes: 0 all stages pass; 1 any stage failed; 2 a stage could not run
(missing tool, missing data) — distinct from a validation failure.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PY = sys.executable or "python3"

STRICT_GATE_SCRIPT = os.path.join(REPO, "scripts", "check_strict_energy_gate_regression.py")
BENCH_SCRIPT = os.path.join(REPO, "scripts", "ashrae_benchmark_harness.py")
BENCH_BASELINE = os.path.join(REPO, "benches", "baseline", "ashrae_benchmark_baseline.json")
PROVENANCE_SCRIPT = os.path.join(REPO, "scripts", "fetch_ashrae140.py")
HONESTY_TEST_DIR = os.path.join(REPO, "tests", "ashrae_140_output_validation")

# The ignored strict ±15% tests whose --nocapture lines the gate parser needs.
STRICT_TEST_BIN = "all_tests"  # tests/all_tests/ — one cargo target, many files
STRICT_TEST_MODULE = "zone_balance_eplus_isolation::"
STRICT_CASES = ["600", "800", "810", "900", "920", "950", "960", "970"]
STRICT_FILTER = "annual_energy_ashrae140_tolerance"


def run(cmd, timeout=None, cwd=REPO):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


class Stage:
    def __init__(self, name, critical):
        self.name = name
        self.critical = critical  # energy-balance stages come first in the report
        self.verdict = "NOT-RUN"  # PASS / FAIL / ERROR / SKIPPED
        self.detail = ""
        self.seconds = 0.0


def stage_provenance():
    s = Stage("provenance: suite files match ASHRAE publisher hashes", critical=False)
    r = run([PY, PROVENANCE_SCRIPT, "verify"], timeout=120)
    tail = [line for line in (r.stdout + r.stderr).strip().splitlines() if line.strip()]
    s.detail = tail[-1] if tail else "no output"
    s.verdict = "PASS" if r.returncode == 0 else "FAIL"
    return s


def stage_strict_energy_gate():
    """Run the ignored strict-energy tests, then the regression checker on them."""
    s = Stage("energy-balance: strict ±15% annual-energy gate (vs recorded baseline)", critical=True)
    filt = [f"{STRICT_TEST_MODULE}test_case_{c}_annual_energy_ashrae140_tolerance"
            for c in STRICT_CASES]
    # cargo takes one positional TESTNAME; libtest accepts multiple filters
    # after `--` (same as the repo's CI invocations in rust-tests.yml).
    r = run(["cargo", "test", "--test", STRICT_TEST_BIN, "--", *filt,
             "--include-ignored", "--nocapture"], timeout=7200)
    log = r.stdout + r.stderr
    # Feed the captured engine lines to the existing transparent gate checker.
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
        fh.write(log)
        log_path = fh.name
    r2 = run([PY, STRICT_GATE_SCRIPT, log_path], timeout=300)
    os.unlink(log_path)
    s.detail = (r2.stdout.strip().splitlines() or ["no gate output"])[-1]
    if r.returncode != 0 and "could not compile" in log.lower():
        s.verdict, s.detail = "ERROR", "cargo could not compile the strict-energy test target"
    else:
        s.verdict = "PASS" if r2.returncode == 0 else "FAIL"
    return s


def stage_honesty_guard():
    s = Stage("energy-balance: Python honesty guard (recorded values vs ASHRAE bands)", critical=True)
    r = run(["uv", "run", "--frozen", "pytest", os.path.relpath(HONESTY_TEST_DIR, REPO),
             "-q", "--no-header"], timeout=1800)
    lines = (r.stdout + r.stderr).strip().splitlines()
    summary = [line for line in lines if "passed" in line or "failed" in line or "error" in line.lower()]
    s.detail = summary[-1] if summary else (lines[-1] if lines else "no pytest output")
    if "No such file" in s.detail or "not found" in s.detail.lower():
        s.verdict, s.detail = "ERROR", "uv/pytest unavailable or test path wrong: " + s.detail
    else:
        s.verdict = "PASS" if r.returncode == 0 else "FAIL"
    return s


def stage_engine_benchmark():
    s = Stage("engine: ASHRAE 140 case series vs stored benchmark baseline", critical=False)
    cmd = [PY, BENCH_SCRIPT]
    if os.path.exists(BENCH_BASELINE):
        cmd += ["--compare", BENCH_BASELINE]
    r = run(cmd, timeout=7200)
    lines = (r.stdout + r.stderr).strip().splitlines()
    counts = [line for line in lines if "Rust tests (all)" in line or "REGRESSION" in line]
    s.detail = (counts[-1] if counts else (lines[-1] if lines else "no harness output"))
    if r.returncode == 2:
        s.verdict = "ERROR"
    else:
        s.verdict = "PASS" if r.returncode == 0 else "FAIL"
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-engine", action="store_true",
                    help="skip the cargo-based engine stages (build or run)")
    ap.add_argument("--skip-strict-gate", action="store_true",
                    help="skip the cargo strict-energy gate (e.g. under --skip-engine)")
    ap.add_argument("--output", help="write the JSON report here")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    args = ap.parse_args()

    if args.list:
        for name in ("provenance", "strict-energy gate", "honesty guard", "engine benchmark"):
            print(name)
        return 0

    stages = [stage_provenance()]
    if not (args.skip_engine and args.skip_strict_gate):
        stages.append(stage_strict_energy_gate())
    else:
        s = Stage("energy-balance: strict ±15% annual-energy gate (vs recorded baseline)", True)
        s.verdict = "SKIPPED"
        stages.append(s)
    s = stage_honesty_guard()
    if args.skip_engine and s.verdict == "ERROR":
        s.verdict = "SKIPPED"
    stages.append(s)
    if not args.skip_engine:
        stages.append(stage_engine_benchmark())
    else:
        s = Stage("engine: ASHRAE 140 case series vs stored benchmark baseline", False)
        s.verdict = "SKIPPED"
        stages.append(s)

    # Energy-balance stages are critical=True: they surface first regardless.
    ordered = sorted(stages, key=lambda s: (not s.critical,))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"\nASHRAE 140-2023 validation pipeline — {now}")
    print("=" * 72)
    for s in ordered:
        print(f"[{s.verdict:^8}] {s.name}")
        if s.detail:
            print(f"           {s.detail}")
    print("=" * 72)

    if args.output:
        with open(args.output, "w") as fh:
            json.dump({
                "generated": now,
                "stages": [vars(s) for s in ordered],
            }, fh, indent=2)
            fh.write("\n")

    verdicts = {s.verdict for s in stages}
    if "FAIL" in verdicts:
        return 1
    if "ERROR" in verdicts:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
