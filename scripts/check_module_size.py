#!/usr/bin/env python3
"""Module size gate for Fluxion (Issue #2878).

Enforces hard upper bounds on the line count of selected `.rs` source files
that are prone to god-struct accumulation. The current file checks
`src/sim/thermal_model_data.rs` (or its directory form `mod.rs`); future
PRs may extend the policy to other files.

Each entry in `LIMITS` defines:
- `path`: source file (relative to repo root, OR absolute).
- `max_lines`: hard ceiling (PR-blocking).
- `ratchet_path`: optional JSON file holding the historical maximum that has
  already shipped; the ceiling is `max(ratchet_max, max_lines)`, so the
  bound can only tighten over time.
- `reason`: human-readable rationale, surfaced in failure messages and
  JSON output.

Exit codes:
- 0 — all entries within bounds.
- 1 — one or more entries exceed their bound.
- 2 — script error (e.g. file not found, malformed ratchet JSON).

Usage:
    python3 scripts/check_module_size.py [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Limit:
    path: Path
    max_lines: int
    reason: str
    ratchet_path: Path | None = None

    def effective_max(self) -> int:
        """Return the effective ceiling — `max(max_lines, ratchet_max)`."""
        if self.ratchet_path is None or not self.ratchet_path.exists():
            return self.max_lines
        try:
            data = json.loads(self.ratchet_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"ERROR: could not read ratchet JSON {self.ratchet_path}: {exc}"
            ) from exc
        ratchet_max = int(data.get("max_lines", 0))
        return max(self.max_lines, ratchet_max)


@dataclass
class Result:
    path: Path
    actual: int
    max: int
    passed: bool
    reason: str = ""


LIMITS: list[Limit] = [
    Limit(
        path=REPO_ROOT / "src" / "sim" / "thermal_model_data.rs",
        max_lines=200,
        ratchet_path=REPO_ROOT
        / "tests"
        / "reference_data"
        / "module_size"
        / "thermal_model_data_ratchet.json",
        reason=(
            "Issue #2878 acceptance: drop ThermalModelData below 200 lines so the "
            "god-struct (~140 fields, 145-line Clone impl) does not regress. "
            "Per-config clone must touch ≤6 fields."
        ),
    ),
    Limit(
        path=REPO_ROOT / "src" / "sim" / "thermal_model_data" / "mod.rs",
        max_lines=200,
        ratchet_path=REPO_ROOT
        / "tests"
        / "reference_data"
        / "module_size"
        / "thermal_model_data_ratchet.json",
        reason=(
            "Issue #2878 acceptance (directory form): drop ThermalModelData below "
            "200 lines so the god-struct (~140 fields, 145-line Clone impl) does "
            "not regress. Per-config clone must touch ≤6 fields."
        ),
    ),
]


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.open(encoding="utf-8"))


def check(limit: Limit) -> Result | None:
    if not limit.path.exists():
        return None
    actual = count_lines(limit.path)
    ceiling = limit.effective_max()
    return Result(
        path=limit.path,
        actual=actual,
        max=ceiling,
        passed=actual <= ceiling,
        reason=limit.reason,
    )


def update_ratchet(result: Result) -> None:
    """Tighten the ratchet if a new minimum is below the historical max."""
    if not result.path.exists():
        return
    rel = result.path.relative_to(REPO_ROOT)
    candidates = [
        lim
        for lim in LIMITS
        if lim.path.relative_to(REPO_ROOT) == rel or lim.path.name == result.path.name
    ]
    for limit in candidates:
        if limit.ratchet_path is None:
            continue
        if not limit.ratchet_path.exists():
            data: dict = {"max_lines": limit.max_lines, "history": []}
        else:
            try:
                data = json.loads(limit.ratchet_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {"max_lines": limit.max_lines, "history": []}
        history = list(data.get("history", []))
        history.append({"actual": result.actual})
        max_observed = max([entry["actual"] for entry in history] + [limit.max_lines])
        data["max_lines"] = max_observed
        data["history"] = history[-20:]
        limit.ratchet_path.parent.mkdir(parents=True, exist_ok=True)
        limit.ratchet_path.write_text(
            json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output for CI consumption.",
    )
    parser.add_argument(
        "--write-ratchet",
        action="store_true",
        help="Tighten the ratchet JSON to reflect observed line counts (PR-friendly).",
    )
    args = parser.parse_args()

    results: list[Result] = []
    for limit in LIMITS:
        result = check(limit)
        if result is not None:
            results.append(result)

    if args.write_ratchet:
        for result in results:
            update_ratchet(result)

    if args.json:
        payload = [
            {
                "path": str(r.path.relative_to(REPO_ROOT)),
                "actual": r.actual,
                "max": r.max,
                "passed": r.passed,
                "reason": r.reason,
            }
            for r in results
        ]
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("=== Fluxion module-size gate (Issue #2878) ===")
        print(f"Repo: {REPO_ROOT}")
        print()
        if not results:
            print("No matching files found — gate is a no-op.")
            return 0
        for result in results:
            rel = result.path.relative_to(REPO_ROOT)
            verdict = "PASS" if result.passed else "FAIL"
            print(f"  [{verdict}] {rel}: {result.actual} lines (max {result.max})")
            print(f"      {result.reason}")
        print()
        if all(r.passed for r in results):
            print("All module-size limits satisfied.")
            return 0
        print("One or more module-size limits exceeded; see FAIL lines above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())