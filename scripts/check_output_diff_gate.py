#!/usr/bin/env python3
"""Release-over-release output-diff gate (issue #4281).

ASHRAE 140 defines no acceptance criteria (#4277), so the strongest
self-consistency signal available is: did any reported value move since the
last recorded run? This gate freezes a per-release baseline of every
reported value the strict battery prints and diffs a fresh run against it.
It does not replace the ±15% band reporting (diagnostic context) or the
absolute analytical anchors (#4277, §8 GC10a / §11 QASv2) — it adds
self-consistency everywhere else, per the BESTEST-GSR reference workflow's
"the diff is the thing we watch" model.

How it works
------------
1. A committed baseline CSV
   (``data/reference/ashrae140/output_diff_baseline.csv``) records every
   metric the strict battery prints: annual heating ``H`` and cooling ``C``
   (MWh) per annual case, and free-float ``Tmin``/``Tmax`` (°C) per FF case.
   Header comment lines carry dated provenance: the repo SHA, the UTC
   recording time, and the source. A committed baseline is never edited in
   place; intentional changes re-record it in the same PR via
   ``--update-baseline`` (which rewrites the whole file with fresh
   provenance).
2. On every run the strict-gate log is parsed (same grammar as
   ``check_strict_energy_gate_regression.py``) and each measured value is
   compared to the baseline with a relative epsilon (default 0.1%,
   ``--epsilon``).
3. Any movement beyond epsilon fails, and so does ANY change in the metric
   set: a metric missing from the log or absent from the baseline is a
   parse/coverage regression. The gate fails closed.
4. The delta table prints worst movement first so a reviewer sees at a
   glance what a change did to the whole suite.

Exit codes: 0 = every metric within epsilon and the metric set matches;
1 = drift beyond epsilon, metric-set change, parse failure, or missing
baseline.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import re
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from check_strict_energy_gate_regression import (  # noqa: E402
    FF_CASES,
    FF_LINE_RE,
    LINE_RE,
    SUPPORTED_CASES,
    parse_ff_measured,
    parse_measured,
)

REPO_ROOT = SCRIPTS_DIR.parent
DEFAULT_BASELINE = (
    REPO_ROOT / "data/reference/ashrae140/output_diff_baseline.csv"
)

BASELINE_HEADER_FIELDS = ("case", "metric", "value", "unit")


def collect_reported_values(log_text: str) -> dict[tuple[str, str], tuple[float, str]]:
    """Flatten the strict-gate log into {(case, metric): (value, unit)}.

    Raises ValueError with a named case list when the log is missing any
    required measurement line — same fail-closed coverage rule as the
    strict checker.
    """
    values: dict[tuple[str, str], tuple[float, str]] = {}
    for case, row in parse_measured(log_text).items():
        values[(case, "H")] = (row["H"], "MWh")
        values[(case, "C")] = (row["C"], "MWh")
    for case, row in parse_ff_measured(log_text).items():
        values[(case, "Tmin")] = (row["Tmin"], "C")
        values[(case, "Tmax")] = (row["Tmax"], "C")

    missing = sorted(
        (
            {(c, m) for c in SUPPORTED_CASES for m in ("H", "C")}
            | {(c, m) for c in FF_CASES for m in ("Tmin", "Tmax")}
        )
        - set(values)
    )
    if missing:
        raise ValueError(
            "strict-gate log is missing measurement lines for: "
            + ", ".join(f"{c}/{m}" for c, m in missing)
        )
    return values


def read_baseline(path: Path) -> tuple[list[str], dict[tuple[str, str], tuple[float, str]]]:
    """Return (provenance header lines, {(case, metric): (value, unit)})."""
    if not path.exists():
        raise ValueError(
            f"baseline not found at {path}. Record one with --update-baseline "
            "from a green strict-gate run and commit it."
        )
    provenance: list[str] = []
    baseline: dict[tuple[str, str], tuple[float, str]] = {}
    with open(path, newline="") as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("#"):
                provenance.append(line)
                continue
            if not line:
                continue
            row = next(csv.reader([line]))
            if row == list(BASELINE_HEADER_FIELDS):
                continue
            case, metric, value, unit = row
            baseline[(case, metric)] = (float(value), unit)
    if not baseline:
        raise ValueError(f"baseline at {path} has no data rows.")
    return provenance, baseline


def write_baseline(
    path: Path,
    values: dict[tuple[str, str], tuple[float, str]],
    source: str,
) -> None:
    """Rewrite the baseline with fresh dated provenance (#4281 step 5)."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        sha = "unknown"
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        fh.write(f"# repo_sha={sha}\n")
        fh.write(f"# recorded_utc={now}\n")
        fh.write(f"# source={source}\n")
        fh.write("# Epsilon policy: 0.1% relative (issue #4281); intentional physics\n")
        fh.write("# changes re-record this file in the same PR via --update-baseline.\n")
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(BASELINE_HEADER_FIELDS)
        for (case, metric), (value, unit) in sorted(values.items()):
            writer.writerow([case, metric, f"{value:.6f}", unit])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="strict-gate log (ANSI-stripped) to diff")
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--epsilon", type=float, default=0.001,
                    help="relative movement tolerance (default 0.001 = 0.1%%)")
    ap.add_argument("--update-baseline", action="store_true",
                    help="re-record the baseline from this run (fresh provenance) and exit 0")
    ap.add_argument("--source", default=None,
                    help="provenance source string written into the baseline header (default: log filename)")
    args = ap.parse_args(argv)

    log_text = Path(args.log).read_text(errors="replace")
    try:
        measured = collect_reported_values(log_text)
    except ValueError as exc:
        print(f"::error::output-diff gate: {exc}")
        return 1

    if args.update_baseline:
        source = args.source or f"strict-gate log {Path(args.log).name}"
        write_baseline(args.baseline, measured, source=source)
        print(f"output-diff gate: baseline re-recorded at {args.baseline} "
              f"({len(measured)} metrics, fresh dated provenance). Commit it in this PR.")
        return 0

    try:
        _, baseline = read_baseline(args.baseline)
    except ValueError as exc:
        print(f"::error::output-diff gate: {exc}")
        return 1
    missing_in_log = sorted(set(baseline) - set(measured))
    missing_in_baseline = sorted(set(measured) - set(baseline))
    if missing_in_log or missing_in_baseline:
        print("::error::output-diff gate: metric set changed — this is a coverage regression, not drift.")
        for key in missing_in_log:
            print(f"  in baseline but not measured: {key[0]}/{key[1]}")
        for key in missing_in_baseline:
            print(f"  measured but not in baseline: {key[0]}/{key[1]}")
        return 1

    unit_mismatches = [
        f"{k[0]}/{k[1]}: baseline {baseline[k][1]} vs log {measured[k][1]}"
        for k in sorted(baseline)
        if baseline[k][1] != measured[k][1]
    ]
    if unit_mismatches:
        print("::error::output-diff gate: unit mismatch — baseline and log disagree on units.")
        for line in unit_mismatches:
            print(f"  {line}")
        return 1

    deltas: list[tuple[float, str, float, float]] = []
    for key in sorted(baseline):
        base_val, _ = baseline[key]
        new_val, _ = measured[key]
        if base_val == 0.0:
            moved = 0.0 if new_val == 0.0 else float("inf")
        else:
            moved = abs(new_val - base_val) / abs(base_val)
        deltas.append((moved, f"{key[0]}/{key[1]}", base_val, new_val))

    deltas.sort(key=lambda d: d[0], reverse=True)
    print(f"=== Output-diff gate: {len(deltas)} metrics vs {args.baseline.name} (epsilon {args.epsilon:.3%}) ===")
    for moved, name, base_val, new_val in deltas:
        flag = "DRIFT" if moved > args.epsilon else "ok"
        if base_val == 0.0:
            print(f"  {flag:>5}  {name:<12} baseline 0 -> {new_val:.6f} (from-zero movement)")
        else:
            print(f"  {flag:>5}  {name:<12} {base_val:.6f} -> {new_val:.6f}  ({(new_val - base_val) / abs(base_val):+.4%})")

    failures = [d for d in deltas if d[0] > args.epsilon]
    if failures:
        print(f"::error::output-diff gate: {len(failures)} metric(s) moved beyond epsilon "
              f"({args.epsilon:.3%}). If this movement is an intentional, reviewed physics "
              "change, re-record the baseline in this PR with --update-baseline; "
              "never tune the engine to keep a baseline green (RULES.md).")
        return 1
    print(f"output-diff gate: PASS — all {len(deltas)} metrics within {args.epsilon:.3%} of baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
