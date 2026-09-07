#!/usr/bin/env python3
"""
Per-critical-path coverage analysis + ratchet gate for the Code Coverage
Gate (Issue #1932).

Parses an LCOV ``lcov.info`` produced by ``cargo llvm-cov --lcov`` and
buckets the results into the four ARCHITECTURE.md critical physics paths:

    1. Weather  -> Solar          (fluxion-core/src/weather/**, src/sim/solar*)
    2. Weather  -> Ventilation    (fluxion-core/src/weather/**, src/sim/ventilation.rs)
    3. Conduction -> Zone Balance (src/physics/**, src/sim/thermal_model*)
    4. HVAC      -> Zone Balance   (src/sim/hvac/**, src/sim/*hvac*)

Plus an ``overall`` bucket that covers every instrumented source file.

Usage
-----
Report-only (prints a Markdown table to stdout / $GITHUB_STEP_SUMMARY)::

    python3 scripts/coverage_critical_paths.py \\
        --lcov target/llvm-cov/lcov.info

Gate mode (non-zero exit if any enforced path regresses beyond the
ratchet tolerance read from the baseline file)::

    python3 scripts/coverage_critical_paths.py \\
        --lcov target/llvm-cov/lcov.info \\
        --baseline validation/coverage_baseline.json \\
        --gate

The baseline stores one entry per critical path, with separate ``line``
and ``branch`` fields.  A baseline value of ``0.0`` means *unenforced*
for that dimension (collection pending) — the gate passes with a notice
for that path/dimension.  Once a maintainer records real numbers via
``scripts/coverage_baseline.py --update`` the ratchet activates
automatically on the next run.

Branch coverage (Issue #2533): the gate enforces *both* line and branch
coverage independently.  ``cargo llvm-cov`` must be invoked with
``--branch`` for the LCOV ``BRF:`` / ``BRH:`` records to appear;
otherwise branch coverage stays at 0 and the branch dimension remains
unenforced.

Exit codes
~~~~~~~~~~
  0  report-only, or gate passed
  1  gate failed (one or more enforced paths regressed)
  2  script error (missing file, malformed input)
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Critical path definitions.
#
# A path is a list of fnmatch globs matched against the *repo-relative* path
# found in the LCOV ``SF:`` records.  Files are allowed to contribute to more
# than one path (e.g. ``fluxion-core/src/weather/`` is on both the solar and
# ventilation paths) — this reflects the real data flow in ARCHITECTURE.md.
# ---------------------------------------------------------------------------
CRITICAL_PATHS: dict[str, list[str]] = {
    "weather_solar": [
        "fluxion-core/src/weather/**",
        "src/sim/solar.rs",
        "src/sim/solar_gain_distribution.rs",
    ],
    "weather_ventilation": [
        "fluxion-core/src/weather/**",
        "src/sim/ventilation.rs",
    ],
    "conduction_zone": [
        "src/physics/**",
        "src/sim/thermal_model.rs",
        "src/sim/thermal_model_core.rs",
        "src/sim/thermal_model_solvers.rs",
        "src/sim/thermal_model_iterative.rs",
        "src/sim/thermal_model_data/**",
        "src/sim/thermal_model_physics/**",
        "src/sim/per_surface_conduction.rs",
    ],
    "hvac_zone": [
        "src/sim/hvac/**",
        "src/sim/thermal_model_solvers.rs",
        "src/sim/hvac_controller.rs",
        "src/sim/multi_node_hvac_runner.rs",
    ],
}

# Human-readable labels for the step summary table.
PATH_LABELS: dict[str, str] = {
    "weather_solar": "Weather → Solar",
    "weather_ventilation": "Weather → Ventilation",
    "conduction_zone": "Conduction → Zone Balance",
    "hvac_zone": "HVAC → Zone Balance",
    "overall": "Overall (all instrumented)",
}

# Relative drop allowed before the ratchet trips.  A 1% relative drop means a
# path at 80.0% passes at 79.2% but fails at 79.1%.  Tuned to absorb noise
# from minor refactors while catching real regressions.
RATCHET_TOLERANCE = 0.01


@dataclass
class FileCoverage:
    """Line + branch coverage for one source file."""

    path: str
    lines_found: int = 0
    lines_hit: int = 0
    branches_found: int = 0
    branches_hit: int = 0

    @property
    def line_pct(self) -> float:
        return 100.0 * self.lines_hit / self.lines_found if self.lines_found else 0.0

    @property
    def branch_pct(self) -> float:
        return (
            100.0 * self.branches_hit / self.branches_found
            if self.branches_found
            else 0.0
        )


@dataclass
class PathReport:
    """Aggregated coverage for one critical path."""

    name: str
    files: list[FileCoverage] = field(default_factory=list)

    @property
    def lines_found(self) -> int:
        return sum(f.lines_found for f in self.files)

    @property
    def lines_hit(self) -> int:
        return sum(f.lines_hit for f in self.files)

    @property
    def branches_found(self) -> int:
        return sum(f.branches_found for f in self.files)

    @property
    def branches_hit(self) -> int:
        return sum(f.branches_hit for f in self.files)

    @property
    def line_pct(self) -> float:
        return 100.0 * self.lines_hit / self.lines_found if self.lines_found else 0.0

    @property
    def branch_pct(self) -> float:
        return (
            100.0 * self.branches_hit / self.branches_found
            if self.branches_found
            else 0.0
        )

    @property
    def file_count(self) -> int:
        return len(self.files)


# ---------------------------------------------------------------------------
# LCOV parsing
# ---------------------------------------------------------------------------
def parse_lcov(lcov_path: Path) -> list[FileCoverage]:
    """Parse an LCOV trace file into a list of per-file coverage records.

    Understands the DA / LF / LH / BRF / BRH records produced by
    ``cargo llvm-cov --lcov``.  Files with zero instrumented lines are
    skipped (they add noise without signal).
    """
    if not lcov_path.exists():
        raise FileNotFoundError(f"LCOV file not found: {lcov_path}")

    files: list[FileCoverage] = []
    current: Optional[FileCoverage] = None

    for raw in lcov_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("SF:"):
            # SF: may contain an absolute or relative path; normalise to
            # repo-relative so the fnmatch globs work either way.
            sf = line[3:]
            current = FileCoverage(path=_repo_relative(sf))
        elif line.startswith("LF:") and current is not None:
            current.lines_found = int(line[3:])
        elif line.startswith("LH:") and current is not None:
            current.lines_hit = int(line[3:])
        elif line.startswith("BRF:") and current is not None:
            current.branches_found = int(line[4:])
        elif line.startswith("BRH:") and current is not None:
            current.branches_hit = int(line[4:])
        elif line == "end_of_record" and current is not None:
            if current.lines_found > 0:
                files.append(current)
            current = None

    return files


def _repo_relative(path: str) -> str:
    """Strip absolute-prefix noise so fnmatch globs match consistently."""
    p = Path(path)
    try:
        rel = p.relative_to(REPO_ROOT)
        return rel.as_posix()
    except ValueError:
        pass
    # Already repo-relative? Return as-is. Covers ``cargo llvm-cov``
    # emitting workspace-relative ``SF:`` records (e.g.
    # ``fluxion-core/src/weather/x.rs``); the marker-stripping below
    # would otherwise drop the ``fluxion-core/`` crate prefix (latent
    # bug found by the #3427 unit tests).
    if path.startswith(("src/", "fluxion-core/src/")):
        return path
    # Fall back to stripping absolute prefixes that llvm-cov emits.
    # Longest marker first: ``/fluxion-core/src/`` must be tried BEFORE
    # ``/src/`` — with the old order, ``/src/`` matched first inside
    # ``.../fluxion-core/src/...`` and silently dropped the crate
    # prefix, un-matching every ``fluxion-core/...`` CRITICAL_PATHS glob
    # (making the ``/fluxion-core/src/`` marker dead code).
    for marker in ("/fluxion-core/src/", "/src/"):
        idx = path.find(marker)
        if idx != -1:
            return path[idx + 1 :]
    return path


def _matches_any(path: str, globs: list[str]) -> bool:
    """True if *path* matches any of the fnmatch globs.

    ``**`` is treated as a recursive wildcard (matches any number of path
    segments), mirroring .gitignore semantics.
    """
    for g in globs:
        if g.endswith("/**"):
            prefix = g[:-3]
            if path == prefix or path.startswith(prefix + "/"):
                return True
        elif fnmatch.fnmatch(path, g):
            return True
    return False


# ---------------------------------------------------------------------------
# Bucketing + reporting
# ---------------------------------------------------------------------------
def bucket_coverage(files: list[FileCoverage]) -> dict[str, PathReport]:
    """Assign every file to the critical paths it belongs to + an overall bucket."""
    reports: dict[str, PathReport] = {}
    for path_name in CRITICAL_PATHS:
        reports[path_name] = PathReport(name=path_name)
    reports["overall"] = PathReport(name="overall")

    for fc in files:
        reports["overall"].files.append(fc)
        for path_name, globs in CRITICAL_PATHS.items():
            if _matches_any(fc.path, globs):
                reports[path_name].files.append(fc)

    return reports


def render_markdown_table(reports: dict[str, PathReport]) -> str:
    """Render a GFM table for $GITHUB_STEP_SUMMARY."""
    rows = [
        "## Code Coverage by Critical Path (Issue #1932)",
        "",
        "| Critical path | Line coverage | Branch coverage | Files |",
        "|---------------|--------------:|----------------:|------:|",
    ]
    order = ["overall", "weather_solar", "weather_ventilation", "conduction_zone", "hvac_zone"]
    for name in order:
        rep = reports.get(name)
        if rep is None or rep.file_count == 0:
            rows.append(
                f"| {PATH_LABELS[name]} | — | — | 0 |"
            )
            continue
        rows.append(
            f"| {PATH_LABELS[name]} "
            f"| {rep.line_pct:.2f}% ({rep.lines_hit}/{rep.lines_found}) "
            f"| {rep.branch_pct:.2f}% ({rep.branches_hit}/{rep.branches_found}) "
            f"| {rep.file_count} |"
        )
    return "\n".join(rows)


def load_baseline(baseline_path: Optional[Path]) -> dict:
    """Load the baseline JSON, returning an empty dict when absent."""
    if baseline_path is None or not baseline_path.exists():
        return {}
    with open(baseline_path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_gate(
    reports: dict[str, PathReport],
    baseline: dict,
    tolerance: float,
) -> list[str]:
    """Return a list of human-readable failure lines (empty list = pass).

    Both the **line** and **branch** coverage dimensions are enforced
    independently (Issue #2533).  The gate trips when a path's current
    coverage in *either* dimension drops below
    ``baseline * (1 - tolerance)``.  Paths whose baseline value is
    ``0.0`` (or absent) for a dimension are *unenforced* on that
    dimension and never trip the gate; they emit a notice instead so the
    baseline-collection status is visible.

    Rationale for tracking branch separately: line coverage can hold at
    80% while branch coverage sits at 30%, and a hidden ``else`` branch
    can silently corrupt simulation behaviour.  Line coverage alone does
    not surface that risk.
    """
    failures: list[str] = []
    paths_section = baseline.get("paths", {}) if isinstance(baseline, dict) else {}
    order = ["overall", "weather_solar", "weather_ventilation", "conduction_zone", "hvac_zone"]
    for name in order:
        rep = reports.get(name)
        base_entry = paths_section.get(name, {}) if isinstance(paths_section, dict) else {}
        base_line = float(base_entry.get("line", 0.0)) if isinstance(base_entry, dict) else 0.0
        base_branch = float(base_entry.get("branch", 0.0)) if isinstance(base_entry, dict) else 0.0

        # If neither dimension has a baseline yet, the path is fully
        # unenforced — emit a single notice and move on.
        if base_line <= 0.0 and base_branch <= 0.0:
            if rep and (rep.lines_found or rep.branches_found):
                print(
                    f"   ℹ️  {name}: baseline not set (unenforced) — "
                    f"current line {rep.line_pct:.2f}%, "
                    f"branch {rep.branch_pct:.2f}%"
                )
            else:
                print(
                    f"   ℹ️  {name}: baseline not set (unenforced) — no instrumented files"
                )
            continue

        # A path with a baseline must still produce instrumented files;
        # losing all instrumentation is a measurement regression.
        if rep is None or (rep.lines_found == 0 and rep.branches_found == 0):
            failures.append(
                f"{name}: no instrumented files found "
                f"(baseline line {base_line:.2f}%, branch {base_branch:.2f}%)"
            )
            continue

        path_failed = False

        # --- Line coverage dimension -------------------------------------
        if base_line > 0.0:
            floor = base_line * (1.0 - tolerance)
            if rep.line_pct < floor:
                path_failed = True
                failures.append(
                    f"{name}: line coverage {rep.line_pct:.2f}% fell below "
                    f"ratchet floor {floor:.2f}% "
                    f"(baseline {base_line:.2f}% × (1 − {tolerance:.0%}))"
                )
            else:
                print(
                    f"   ✅ {name} line: {rep.line_pct:.2f}% ≥ floor {floor:.2f}% "
                    f"(baseline {base_line:.2f}%)"
                )
        else:
            print(
                f"   ℹ️  {name} line: unenforced (baseline 0.0), "
                f"current {rep.line_pct:.2f}%"
            )

        # --- Branch coverage dimension (#2533) ---------------------------
        if base_branch > 0.0:
            # A set branch baseline with no instrumented branches in the
            # current run means measurement regressed (e.g. CI forgot
            # ``--branch-coverage``) — fail loud rather than silently.
            if rep.branches_found == 0:
                path_failed = True
                failures.append(
                    f"{name}: branch baseline {base_branch:.2f}% is set but the "
                    f"current run instrumented 0 branches "
                    f"(did cargo llvm-cov get --branch-coverage?)"
                )
            else:
                floor = base_branch * (1.0 - tolerance)
                if rep.branch_pct < floor:
                    path_failed = True
                    failures.append(
                        f"{name}: branch coverage {rep.branch_pct:.2f}% fell below "
                        f"ratchet floor {floor:.2f}% "
                        f"(baseline {base_branch:.2f}% × (1 − {tolerance:.0%}))"
                    )
                else:
                    print(
                        f"   ✅ {name} branch: {rep.branch_pct:.2f}% ≥ floor {floor:.2f}% "
                        f"(baseline {base_branch:.2f}%)"
                    )
        else:
            cur = rep.branch_pct if rep.branches_found else 0.0
            print(
                f"   ℹ️  {name} branch: unenforced (baseline 0.0), current {cur:.2f}%"
            )

        # --- Absolute branch floor + v1.3 target (#2710) ----------------
        # The regression ratchet above only prevents coverage from
        # *dropping* relative to the recorded baseline — it never drives
        # coverage *up* toward a goal, so a 60-68% branch gap on the
        # critical physics paths could persist forever.  Issue #2710 adds
        # two independent per-path policy levers read from the baseline:
        #
        #   min_branch_floor    absolute hard floor; FAILS when current
        #                       branch coverage is below it.  Independent
        #                       of the regression ratchet so it still
        #                       bites even if the ratchet baseline were
        #                       lowered.  0.0 / absent = unenforced.
        #
        #   v1_3_target_branch  aspirational target for the v1.3 release;
        #                       REPORTED (gap printed every run) but not
        #                       yet failing — makes the remaining gap
        #                       visible so it cannot be ignored.  Becomes
        #                       a hard release gate once the metrics
        #                       approach it.
        min_branch_floor = (
            float(base_entry.get("min_branch_floor", 0.0))
            if isinstance(base_entry, dict)
            else 0.0
        )
        v1_3_target = (
            float(base_entry.get("v1_3_target_branch", 0.0))
            if isinstance(base_entry, dict)
            else 0.0
        )

        if rep.branches_found > 0:
            if min_branch_floor > 0.0:
                if rep.branch_pct < min_branch_floor:
                    path_failed = True
                    failures.append(
                        f"{name}: branch coverage {rep.branch_pct:.2f}% is below "
                        f"the absolute minimum floor {min_branch_floor:.2f}% "
                        f"(#2710 v1.3 critical-path bar)"
                    )
                else:
                    print(
                        f"   ✅ {name} branch floor: {rep.branch_pct:.2f}% ≥ "
                        f"min {min_branch_floor:.2f}% (#2710)"
                    )
            if v1_3_target > 0.0:
                gap = v1_3_target - rep.branch_pct
                if gap > 0.0:
                    print(
                        f"   🎯 {name} v1.3 target: {rep.branch_pct:.2f}% / "
                        f"{v1_3_target:.2f}% — {gap:.2f}pp to close (#2710)"
                    )
                else:
                    print(
                        f"   🎯 {name} v1.3 target: MET "
                        f"({rep.branch_pct:.2f}% ≥ {v1_3_target:.2f}%)"
                    )

        if not path_failed:
            # Aggregate confirmation line for paths that passed both
            # dimensions (keeps the single-path summary readable when
            # both dimensions are enforced).
            pass

    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-critical-path coverage analysis + ratchet gate (#1932)"
    )
    parser.add_argument(
        "--lcov",
        type=Path,
        default=REPO_ROOT / "target" / "llvm-cov" / "lcov.info",
        help="Path to the lcov.info produced by `cargo llvm-cov --lcov`",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=REPO_ROOT / "validation" / "coverage_baseline.json",
        help="Path to the committed coverage baseline JSON",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Fail (exit 1) when an enforced path regresses past the ratchet",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=RATCHET_TOLERANCE,
        help="Relative drop allowed before the ratchet trips (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the computed per-path metrics as JSON to stdout",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Append the Markdown table to this file (use $GITHUB_STEP_SUMMARY in CI)",
    )
    args = parser.parse_args()

    try:
        files = parse_lcov(args.lcov)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    reports = bucket_coverage(files)
    table = render_markdown_table(reports)

    if args.json:
        payload = {
            name: {
                "line_pct": round(rep.line_pct, 4),
                "branch_pct": round(rep.branch_pct, 4),
                "lines_hit": rep.lines_hit,
                "lines_found": rep.lines_found,
                "branches_hit": rep.branches_hit,
                "branches_found": rep.branches_found,
                "files": rep.file_count,
            }
            for name, rep in reports.items()
        }
        print(json.dumps(payload, indent=2))

    print(table)
    print()

    if args.summary_file is not None:
        with open(args.summary_file, "a", encoding="utf-8") as fh:
            fh.write(table)
            fh.write("\n")

    if args.gate:
        baseline = load_baseline(args.baseline)
        print("Ratchet gate evaluation:")
        failures = evaluate_gate(reports, baseline, args.tolerance)
        if failures:
            print()
            print(f"❌ {len(failures)} path(s) regressed past the ratchet:")
            for f in failures:
                print(f"   • {f}")
            return 1
        print()
        print("✅ All enforced critical paths held their ratchet floor.")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover - defensive
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
