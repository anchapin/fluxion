#!/usr/bin/env python3
"""
CI regression check for TDQS (Temporal Decision Quality Score).

Reads the current TDQS from the Criterion benchmark output JSON
and compares it against the stored baseline.

Fails with exit code 1 if TDQS drops > threshold on overall score
or on any individual decision type.

Usage (called from .github/workflows/tdqs_regression.yml):
    python3 scripts/check_tdqs_regression.py \
        --current  benches/orchestration_decisions/baselines/current_tdqs.json \
        --baseline benches/orchestration_decisions/baselines/rule_based_baseline.json \
        --threshold 0.05

Exit codes:
    0 — no regression
    1 — regression detected (fails CI)
    2 — missing input file (configuration error)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(2)
    with p.open() as f:
        return json.load(f)


DECISION_TYPES = [
    "solver_selection",
    "adaptive_timestep",
    "surrogate_routing",
    "constraint_warning",
    "hvac_horizon",
]


def extract_tdqs(data: dict) -> tuple[float, dict[str, float]]:
    """Return (overall, {type: tdqs}) from a TDQS JSON file."""
    overall = float(data.get("overall", 0.0))
    per_type: dict[str, float] = {}
    pt = data.get("per_type", {})
    for dt in DECISION_TYPES:
        entry = pt.get(dt, {})
        per_type[dt] = float(
            entry.get("tdqs", entry) if isinstance(entry, dict) else entry
        )
    return overall, per_type


def check_regression(
    current_path: str,
    baseline_path: str,
    threshold: float = 0.05,
    warn_only: bool = False,
) -> int:
    current_data = load_json(current_path)
    baseline_data = load_json(baseline_path)

    cur_overall, cur_pt = extract_tdqs(current_data)
    base_overall, base_pt = extract_tdqs(baseline_data)

    print(f"\n{'=' * 60}")
    print(f"TDQS Regression Check  (threshold = {threshold:.2f})")
    print(f"{'=' * 60}")
    print(
        f"  {'Metric':<28} {'Baseline':>10} {'Current':>10} {'Delta':>10} {'Status':>10}"
    )
    print(f"  {'-' * 28} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")

    regressions: list[str] = []

    # Overall
    delta_overall = cur_overall - base_overall
    sign = "+" if delta_overall >= 0 else ""
    status = "OK" if base_overall - cur_overall <= threshold else "REGRESSION"
    if status == "REGRESSION":
        regressions.append(f"overall ({delta_overall:+.4f})")
    print(
        f"  {'overall':<28} {base_overall:>10.4f} {cur_overall:>10.4f} {sign}{delta_overall:>9.4f} {status:>10}"
    )

    # Per-type
    for dt in DECISION_TYPES:
        base_score = base_pt.get(dt, 0.0)
        cur_score = cur_pt.get(dt, 0.0)
        delta = cur_score - base_score
        sign = "+" if delta >= 0 else ""
        status = "OK" if base_score - cur_score <= threshold else "REGRESSION"
        if status == "REGRESSION":
            regressions.append(f"{dt} ({delta:+.4f})")
        print(
            f"  {dt:<28} {base_score:>10.4f} {cur_score:>10.4f} {sign}{delta:>9.4f} {status:>10}"
        )

    print()

    # Write GitHub Actions annotations
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"tdqs_overall={cur_overall:.6f}\n")
            f.write(f"tdqs_regression={'true' if regressions else 'false'}\n")
            f.write(f"tdqs_regressions={','.join(regressions)}\n")

    # GitHub Step Summary
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        lines = [
            "## TDQS Regression Check\n",
            "| Metric | Baseline | Current | Delta | Status |",
            "|--------|----------|---------|-------|--------|",
        ]
        delta = cur_overall - base_overall
        sig = "+" if delta >= 0 else ""
        ok = "✅ OK" if base_overall - cur_overall <= threshold else "❌ REGRESSION"
        lines.append(
            f"| **overall** | {base_overall:.4f} | **{cur_overall:.4f}** | `{sig}{delta:.4f}` | {ok} |"
        )
        for dt in DECISION_TYPES:
            b = base_pt.get(dt, 0.0)
            c = cur_pt.get(dt, 0.0)
            d = c - b
            sig = "+" if d >= 0 else ""
            st = "✅ OK" if b - c <= threshold else "❌ REGRESSION"
            lines.append(f"| {dt} | {b:.4f} | {c:.4f} | `{sig}{d:.4f}` | {st} |")
        with open(summary_path, "a") as f:
            f.write("\n".join(lines) + "\n")

    if regressions:
        if warn_only:
            print(
                f"⚠️  WARNING: TDQS regressions (non-blocking): {', '.join(regressions)}"
            )
            return 0
        else:
            print(f"❌ REGRESSION DETECTED in: {', '.join(regressions)}")
            print(f"   TDQS dropped > {threshold:.0%} in one or more decision types.")
            print(
                "   To resolve: ensure physics fixes do not degrade decision quality."
            )
            return 1
    else:
        print(
            f"✅ No regressions detected. TDQS: {cur_overall:.4f} (baseline: {base_overall:.4f})"
        )
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--current",
        default="benches/orchestration_decisions/baselines/current_tdqs.json",
        help="Path to current TDQS JSON (output from Criterion bench)",
    )
    parser.add_argument(
        "--baseline",
        default="benches/orchestration_decisions/baselines/rule_based_baseline.json",
        help="Path to baseline TDQS JSON",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Regression threshold in TDQS points (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Warn but do not fail on regression (for PRs only)",
    )
    args = parser.parse_args()

    return check_regression(
        current_path=args.current,
        baseline_path=args.baseline,
        threshold=args.threshold,
        warn_only=args.warn_only,
    )


if __name__ == "__main__":
    sys.exit(main())
