#!/usr/bin/env python3
"""
Record the current coverage as the ratchet baseline for the Code Coverage
Gate (Issue #1932).

A maintainer runs this once after a clean ``develop`` CI run to flip the
gate from *unenforced* (baseline ``0.0``) to *enforced*.  Re-running it
bumps the baseline upward when coverage improves — the ratchet only
ever moves in the direction of the latest measurement, so coverage can
rise but never silently fall.

Usage
~~~~~
::

    # After a green CI run, download the lcov.info artifact then:
    python3 scripts/coverage_baseline.py --update \\
        --lcov target/llvm-cov/lcov.info \\
        --baseline validation/coverage_baseline.json

    # Dry run: print what would be written without touching the file.
    python3 scripts/coverage_baseline.py --lcov target/llvm-cov/lcov.info

Exit codes: 0 on success, 2 on script error.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Import the shared bucketing logic from the sibling script.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from coverage_critical_paths import (  # noqa: E402  (sys.path insert above)
    bucket_coverage,
    load_baseline,
    parse_lcov,
)


PATH_ORDER = ["overall", "weather_solar", "weather_ventilation", "conduction_zone", "hvac_zone"]


def build_baseline_payload(reports: dict, previous: dict) -> dict:
    """Compose the JSON payload that becomes the new baseline.

    Each path records ``line`` and ``branch`` percentages plus the raw
    counts so a future reader can audit the sample size.  Both
    dimensions are ratcheted one-way: once a non-zero floor has been
    recorded it never moves downward.  ``_ratchet`` documents the
    one-way tolerance the gate applies (Issue #2533 extended the
    ratchet from line-only to line + branch).
    """
    paths_section = previous.get("paths", {}) if isinstance(previous, dict) else {}
    new_paths: dict[str, dict] = {}
    for name in PATH_ORDER:
        rep = reports.get(name)
        prev_entry = paths_section.get(name, {}) if isinstance(paths_section, dict) else {}
        prev_line = float(prev_entry.get("line", 0.0)) if isinstance(prev_entry, dict) else 0.0
        prev_branch = (
            float(prev_entry.get("branch", 0.0)) if isinstance(prev_entry, dict) else 0.0
        )

        current_line = round(rep.line_pct, 4) if rep else 0.0
        current_branch = round(rep.branch_pct, 4) if rep else 0.0
        # Ratchet: never move the floor *down* once it has been set.  This
        # stops a bad merge from quietly lowering the bar; the floor only
        # rises (or stays flat) as coverage improves.  Applied to both
        # line and branch dimensions independently (#2533).
        ratcheted_line = max(current_line, prev_line) if prev_line > 0.0 else current_line
        ratcheted_branch = (
            max(current_branch, prev_branch) if prev_branch > 0.0 else current_branch
        )

        # Policy levers (#2710): min_branch_floor / v1_3_target_branch are
        # maintainer-set goals, not measured values, so re-baselining
        # neither invents nor erases them — they are carried forward
        # verbatim from the previous baseline (defaulting to 0.0 =
        # unenforced when absent, matching the existing convention).
        prev_min_floor = (
            float(prev_entry.get("min_branch_floor", 0.0))
            if isinstance(prev_entry, dict)
            else 0.0
        )
        prev_v1_3_target = (
            float(prev_entry.get("v1_3_target_branch", 0.0))
            if isinstance(prev_entry, dict)
            else 0.0
        )

        new_paths[name] = {
            "line": ratcheted_line,
            "branch": ratcheted_branch,
            "lines_hit": rep.lines_hit if rep else 0,
            "lines_found": rep.lines_found if rep else 0,
            "branches_hit": rep.branches_hit if rep else 0,
            "branches_found": rep.branches_found if rep else 0,
            "min_branch_floor": prev_min_floor,
            "v1_3_target_branch": prev_v1_3_target,
        }

    return {
        "_comment": (
            "Coverage baseline for the Code Coverage Gate (#1932). "
            "A value of 0.0 means the path/dimension is unenforced; the gate "
            "activates automatically once a real number is recorded here. "
            "Both line and branch coverage are ratcheted one-way (#2533). "
            "Per critical path, min_branch_floor is an absolute hard floor "
            "and v1_3_target_branch is the v1.3 release target (#2710); "
            "both default to 0.0 (unenforced) and are carried forward "
            "verbatim by this script. "
            "Regenerate with `python scripts/coverage_baseline.py --update "
            "--lcov target/llvm-cov/lcov.info` (requires `cargo llvm-cov "
            "--branch` upstream)."
        ),
        "_issue": 1932,
        "_updated": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "_ratchet": {
            "tolerance": 0.01,
            "description": (
                "Gate fails when a path's current line OR branch coverage "
                "drops below baseline × (1 − tolerance). Both dimensions' "
                "baselines never move downward. Branch coverage requires "
                "cargo llvm-cov --branch-coverage (#2533)."
            ),
        },
        "_policy": {
            "issue": 2710,
            "description": (
                "min_branch_floor (absolute hard floor; FAILS below it, "
                "independent of the ratchet) + v1_3_target_branch "
                "(aspirational v1.3 target; REPORTED but not yet failing). "
                "Both default to 0.0 = unenforced; carried forward verbatim."
            ),
        },
        "paths": new_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record coverage baseline for the #1932 ratchet gate"
    )
    parser.add_argument(
        "--lcov",
        type=Path,
        default=REPO_ROOT / "target" / "llvm-cov" / "lcov.info",
        help="Path to lcov.info from `cargo llvm-cov --lcov`",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=REPO_ROOT / "validation" / "coverage_baseline.json",
        help="Path to the baseline JSON to read/update",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Write the new baseline (without this flag the script is a dry run)",
    )
    args = parser.parse_args()

    try:
        files = parse_lcov(args.lcov)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    reports = bucket_coverage(files)
    previous = load_baseline(args.baseline)
    payload = build_baseline_payload(reports, previous)

    print("Baseline payload:")
    print(json.dumps(payload, indent=2))

    if args.update:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        with open(args.baseline, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        print(f"\n✅ Baseline written to {args.baseline}")
        print(
            "The Code Coverage Gate will now enforce these floors on the next "
            "CI run. Re-run this command after coverage improvements to bump "
            "the ratchet upward."
        )
    else:
        print(f"\nℹ️  Dry run — pass --update to write to {args.baseline}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover - defensive
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
