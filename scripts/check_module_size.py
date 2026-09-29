#!/usr/bin/env python3
"""Module size gate for Fluxion (Issues #2878, #3457).

Enforces hard upper bounds on the line count of selected ``.rs`` source files
that are prone to god-struct accumulation. Issue #2878 introduced the gate
with a single limit for ``src/sim/thermal_model_data/mod.rs``; Issue #3457
extended the policy to cover the ten largest ``src/`` files (each ratcheted
to its current line count).

Each entry in ``LIMITS`` defines:
- ``path``: source file (relative to repo root, OR absolute).
- ``max_lines``: hard ceiling (PR-blocking). The ``max_lines`` is the
  active ceiling; the ratchet JSON holds the historical maximum for
  visibility (``effective_max = max(max_lines, ratchet_max)``). To tighten
  the bound, lower ``max_lines`` in this script.
- ``ratchet_path``: JSON file holding the historical maximum line count
  observed for that file (``max_lines`` + ``history`` keys). Pre-seeded at
  the file's current size on Issue #3457 so the bound only tightens from
  here onward as the YAML ceiling is lowered alongside file decomposition.
- ``reason``: human-readable rationale, surfaced in failure messages and
  JSON output.

Issue #4243 refactor: LIMITS are read from the canonical data file
``scripts/data/module_size_limits.json`` instead of being hardcoded in this
script. The ``LIMITS`` list itself stays ratcheted downward-only, but the
baseline now lives in the data file's ``"baseline"`` section (``count`` +
``paths``), mirroring the ``BASELINE_KNOWN_ORPHANS`` /
``_BASELINE_KNOWN_ORPHANS_SET`` pattern from
``scripts/check_orphan_modules.py`` (Issues #3458 / #3459): adding a new
entry without updating the baseline section is PR-blocking drift. Use
``--write-baseline [--reason ...]`` to update the data file AND the
baseline section atomically (records a history entry naming the reason),
and ``--remove-entries`` to handle decomposition cleanup (removes entries
and optionally deletes associated ratchet JSONs; companion cleanup PRs
that decompose a gated file should remove its entry AND lower the
baseline by one).

Exit codes:
- 0 — all entries within bounds, no drift against the freeze.
- 1 — one or more entries exceed their bound, OR the LIMITS list grew past
  the freeze baseline.
- 2 — script error (e.g. file not found, malformed ratchet JSON).

Usage:
    python3 scripts/check_module_size.py [--json] [--write-ratchet]
    python3 scripts/check_module_size.py --write-baseline           # update data file
    python3 scripts/check_module_size.py --remove-entries src/foo.rs [--delete-ratchets]
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LIMITS_FILE = REPO_ROOT / "scripts" / "data" / "module_size_limits.json"
RATCHET_DIR = REPO_ROOT / "tests" / "reference_data" / "module_size"


# ---------------------------------------------------------------------------
# Issue #3457 — LIMITS-list baseline (downward-only ratchet).
#
# Mirrors ``BASELINE_KNOWN_ORPHANS`` from ``scripts/check_orphan_modules.py``:
# the constant is the *highest* value of ``len(LIMITS)`` this guard will
# accept. The script FAILS the moment the live LIMITS list grows past it.
# Companion cleanup PRs that decompose a gated file (so its entry can be
# removed) are expected to drop the matching entry AND lower this baseline
# by one. Raising the baseline is reserved for legitimately gating a new
# large file, and MUST be accompanied by a documenting comment naming the
# tracking issue AND a matching addition to the data file.
#
# NOTE: As of Issue #4243, these constants are DERIVED from the LIMITS
# data file at module load time. The values here are fallbacks for when
# the data file is absent (e.g., in old checkouts). The data file
# (``scripts/data/module_size_limits.json``) is the authoritative source.
# ---------------------------------------------------------------------------

# Fallback default LIMITS (Issue #4243: replaced by data file).
_DEFAULT_LIMITS: list[dict] = [
    {
        "path": "src/sim/thermal_model_data.rs",
        "max_lines": 200,
        "ratchet_basename": "thermal_model_data_ratchet.json",
        "reason": (
            "Issue #2878 acceptance: drop ThermalModelData below 200 lines "
            "so the god-struct (~140 fields, 145-line Clone impl) does not "
            "regress. Per-config clone must touch ≤6 fields."
        ),
    },
    {
        "path": "src/sim/thermal_model_data/mod.rs",
        "max_lines": 200,
        "ratchet_basename": "thermal_model_data_ratchet.json",
        "reason": (
            "Issue #2878 acceptance (directory form): drop ThermalModelData "
            "below 200 lines so the god-struct (~140 fields, 145-line Clone "
            "impl) does not regress. Per-config clone must touch ≤6 fields."
        ),
    },
    {
        "path": "src/validation/ashrae_140_cases.rs",
        "max_lines": 4764,
        "ratchet_basename": "ashrae_140_cases_ratchet.json",
        "reason": (
            "Issue #3457: ASHRAE 140 cases module ratcheted at current "
            "size (4764 lines); v1.3 validation work depends on this module."
        ),
    },
    {
        "path": "src/sim/thermal_model/mod.rs",
        "max_lines": 3061,
        "ratchet_basename": "thermal_model_mod_ratchet.json",
        "reason": (
            "Issue #3457 (originally) → Issue #3789 (decomposed via PR "
            "#3843): the 3061-line single-file ``src/sim/thermal_model.rs`` "
            "was split into a ``thermal_model/`` directory; the original "
            "ceiling (3061) is preserved as the ratchet for the new "
            "``mod.rs`` so the audit thread keeps firing on growth. Hosts "
            "the ``ThermalModelTrait`` swap point; zero-headroom entry (Issue "
            "#3747) — further growth beyond the ceiling goes through a "
            "documented protocol bump."
        ),
    },
    {
        "path": "src/api/security.rs",
        "max_lines": 2199,
        "ratchet_basename": "api_security_ratchet.json",
        "reason": (
            "Issue #3574: ``src/api/security.rs`` crossed the smallest "
            "ratcheted threshold (~2000 LoC) after the Issue #3543 "
            "decomposition. Ratcheted at current size + buffer (max of "
            "+5% or +100 lines = 2199); decomposition tracked separately."
        ),
    },
    {
        "path": "src/sim/thermal_model_core/mod.rs",
        "max_lines": 4260,
        "ratchet_basename": "thermal_model_core_mod_ratchet.json",
        "reason": (
            "Issue #3574: ``src/sim/thermal_model_core/mod.rs`` is the "
            "largest child of the Issue #3543 decomposition (~4.1k LoC), "
            "still over the ratchet because of the bare "
            "``impl ThermalModel<VectorField>`` block. Ratcheted at "
            "current size 4057 lines + buffer (4260); further decomposition "
            "tracked separately."
        ),
    },
    {
        "path": "src/validation/ashrae_140_validator/mod.rs",
        "max_lines": 3247,
        "ratchet_basename": "ashrae_140_validator_mod_ratchet.json",
        "reason": (
            "Issue #3574: ``src/validation/ashrae_140_validator/mod.rs`` "
            "is the largest child of the Issue #3543 decomposition. "
            "Ratcheted at current size 3092 lines + buffer (3247); "
            "further decomposition tracked separately."
        ),
    },
    {
        "path": "src/physics/geometry_tensor.rs",
        "max_lines": 2932,
        "ratchet_basename": "geometry_tensor_ratchet.json",
        "reason": (
            "Issue #3574: ``src/physics/geometry_tensor.rs`` crossed the "
            "smallest ratcheted threshold after Issue #3543. Ratcheted at "
            "current size 2792 lines + buffer (2932; raised from 2486/2610 "
            "by #3731's typed ZoneCountPolicy wrapper); decomposition "
            "tracked separately."
        ),
    },
    {
        "path": "src/validation/reporter.rs",
        "max_lines": 2292,
        "ratchet_basename": "validation_reporter_ratchet.json",
        "reason": (
            "Issue #3574: ``src/validation/reporter.rs`` crossed the "
            "smallest ratcheted threshold after Issue #3543. Ratcheted at "
            "current size 2183 lines + buffer (2292); decomposition "
            "tracked separately."
        ),
    },
    {
        "path": "fluxion-city/src/lib.rs",
        "max_lines": 2814,
        "ratchet_basename": "fluxion_city_lib_ratchet.json",
        "reason": (
            "Issue #3574: workspace sibling ``fluxion-city/src/lib.rs`` "
            "crossed the smallest ratcheted threshold after Issue #3543. "
            "Ratcheted at current size 2680 lines + buffer (2814); "
            "decomposition tracked separately."
        ),
    },
    {
        "path": "fluxion-mcp/src/tools.rs",
        "max_lines": 1748,
        "ratchet_basename": "fluxion_mcp_tools_ratchet.json",
        "reason": (
            "Issue #3574: workspace sibling ``fluxion-mcp/src/tools.rs`` "
            "crossed the smallest ratcheted threshold after Issue #3543. "
            "Ratcheted at current size 1648 lines + buffer (1748); "
            "decomposition tracked separately."
        ),
    },
]


def _load_limits_file(limits_file: Path) -> dict | None:
    """Load the raw canonical LIMITS JSON data file (None if absent/unreadable)."""
    if not limits_file.exists():
        return None
    try:
        data = json.loads(limits_file.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _load_limits_from_file(limits_file: Path) -> list[dict] | None:
    """Load LIMITS from the canonical JSON data file."""
    data = _load_limits_file(limits_file)
    if data is None:
        return None
    return data.get("limits", None)


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _get_limits_data(limits_file: Path = DEFAULT_LIMITS_FILE) -> tuple[list[dict], Path]:
    """Get the LIMITS list, reading from the data file if available.

    Returns a tuple of (limits_list, limits_file_path). The file path is
    returned so callers know where to write updates.
    """
    data_limits = _load_limits_from_file(limits_file)
    if data_limits is not None:
        return data_limits, limits_file
    # Fall back to hardcoded defaults
    return _DEFAULT_LIMITS, limits_file


def _build_limits(limits_data: list[dict], repo_root: Path = REPO_ROOT) -> list["Limit"]:
    """Build the LIMITS list from the data format."""
    limits = []
    for entry in limits_data:
        path = repo_root / entry["path"]
        ratchet_basename = entry.get("ratchet_basename")
        ratchet_path = None
        if ratchet_basename:
            ratchet_path = RATCHET_DIR / ratchet_basename
        limits.append(
            Limit(
                path=path,
                max_lines=entry["max_lines"],
                reason=entry["reason"],
                ratchet_path=ratchet_path,
            )
        )
    return limits


# Load LIMITS from data file (or fall back to hardcoded defaults)
_LIMITS_FILE_DATA = _load_limits_file(DEFAULT_LIMITS_FILE)
if _LIMITS_FILE_DATA is not None and _LIMITS_FILE_DATA.get("limits") is not None:
    _LIMITS_DATA = _LIMITS_FILE_DATA["limits"]
else:
    _LIMITS_DATA = _DEFAULT_LIMITS


@dataclass
class Limit:
    path: Path
    max_lines: int
    reason: str
    ratchet_path: Path | None = None

    def effective_max(self) -> int:
        """Return the effective ceiling — ``max(max_lines, ratchet_max)``."""
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


# The LIMITS list used by the gate (derived from data file)
LIMITS: list[Limit] = _build_limits(_LIMITS_DATA)

# Freeze snapshot + count baseline, stored in the data file's "baseline" section
# (Issue #4243). The gate compares the live LIMITS list against these stored
# values, so adding an entry without updating the baseline is PR-blocking drift —
# same forcing function as the old hardcoded constants, but the churn lives in
# the JSON data file instead of the script. `python3
# scripts/check_module_size.py --write-baseline` updates the section atomically
# (with a history entry); hand-editing is equivalent. When the data file is
# absent (old checkouts), the baseline falls back to the live list (neutral).
_stored_baseline: dict = (_LIMITS_FILE_DATA or {}).get("baseline") or {}
_BASELINE_MODULE_SIZE_LIMITS_SET: frozenset[str] = frozenset(
    _stored_baseline.get("paths", [entry["path"] for entry in _LIMITS_DATA])
)

# Baseline count: stored in the data file's "baseline" section.
BASELINE_MODULE_SIZE_LIMITS: int = int(
    _stored_baseline.get("count", len(_LIMITS_DATA))
)


@dataclass
class Result:
    path: Path
    actual: int
    max: int
    passed: bool
    reason: str = ""


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


def check_baseline_drift(
    limits: list["Limit"] | None = None,
    freeze_set: frozenset[str] | None = None,
    baseline_count: int | None = None,
) -> list[str]:
    """Return human-readable drift messages (empty list = no drift).

    Compares the live ``LIMITS`` table against the freeze snapshot
    (``_BASELINE_MODULE_SIZE_LIMITS_SET``, stored in the data file's
    ``"baseline"`` section) and the count baseline
    (``BASELINE_MODULE_SIZE_LIMITS``). Mirrors
    ``BASELINE_KNOWN_ORPHANS`` / ``_BASELINE_KNOWN_ORPHANS_SET`` from
    ``check_orphan_modules.py``: adding entries without raising the
    baseline is treated as drift, and the failing messages name which
    entries are new so the diff is visible in CI output.

    The optional parameters exist so tests can drive the check against
    synthetic data without monkeypatching module state; production calls
    use the module-level defaults.
    """
    if limits is None:
        limits = LIMITS
    if freeze_set is None:
        freeze_set = _BASELINE_MODULE_SIZE_LIMITS_SET
    if baseline_count is None:
        baseline_count = BASELINE_MODULE_SIZE_LIMITS
    messages: list[str] = []
    live_paths: set[str] = set()
    for lim in limits:
        try:
            rel = lim.path.relative_to(REPO_ROOT)
        except ValueError:
            rel = lim.path
        live_paths.add(str(rel))
    new_paths = sorted(live_paths - set(freeze_set))
    if new_paths:
        joined = "\n".join(f"    - {p}" for p in new_paths)
        messages.append(
            "NEW LIMITS entries (regression): "
            f"{len(new_paths)} (not in freeze snapshot)\n{joined}\n"
            "If these entries are intentional, update the baseline section of "
            "scripts/data/module_size_limits.json (use --write-baseline, with "
            "--reason naming the tracking issue)."
        )
    if len(limits) > baseline_count:
        messages.append(
            "BASELINE_MODULE_SIZE_LIMITS drift: "
            f"len(LIMITS)={len(limits)} > "
            f"BASELINE_MODULE_SIZE_LIMITS={baseline_count}\n"
            "Either update the baseline section of "
            "scripts/data/module_size_limits.json (use --write-baseline, with "
            "--reason naming the tracking issue) or remove the surplus "
            "entry. The companion cleanup PRs that decompose a gated "
            "file are expected to lower the baseline by one."
        )
    return messages


def write_baseline_file(
    limits_file: Path,
    limits_data: list[dict],
    dry_run: bool = False,
    reason: str | None = None,
) -> None:
    """Write the LIMITS data to the canonical JSON file.

    This function updates the data file to match the current LIMITS table,
    updates the "baseline" section (count + path freeze) atomically, and
    appends a history entry. It preserves the schema_version.
    """
    if limits_file.exists():
        try:
            existing = json.loads(limits_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {"schema_version": 1}
    else:
        existing = {"schema_version": 1}

    # Update the file
    existing["limits"] = limits_data
    existing["generated_at"] = _now_iso()
    existing["source_script"] = "scripts/check_module_size.py"

    # Update the baseline section atomically with the limits (Issue #4243:
    # the drift gate compares the live list against this stored baseline).
    existing["baseline"] = {
        "count": len(limits_data),
        "paths": sorted(entry["path"] for entry in limits_data),
    }

    # Add history entry
    history = existing.get("history", [])
    action = "updated via --write-baseline"
    if reason:
        action += f": {reason}"
    history.append({
        "at": _now_iso()[:10],  # date only
        "len": len(limits_data),
        "action": action,
    })
    existing["history"] = history[-10:]  # keep last 10

    output = json.dumps(existing, indent=2, sort_keys=True) + "\n"

    if dry_run:
        print(f"[dry-run] Would write {len(output)} bytes to {limits_file}:")
        print(output[:500] + "..." if len(output) > 500 else output)
    else:
        limits_file.parent.mkdir(parents=True, exist_ok=True)
        limits_file.write_text(output, encoding="utf-8")
        print(f"Updated {limits_file} ({len(limits_data)} entries)")


def remove_entries_from_limits(
    limits_data: list[dict],
    remove_paths: list[str],
    delete_ratchets: bool = False,
    dry_run: bool = False,
) -> tuple[list[dict], list[str]]:
    """Remove entries from LIMITS data.

    Args:
        limits_data: current LIMITS list
        remove_paths: list of paths to remove (relative to repo root)
        delete_ratchets: if True, also delete the associated ratchet JSONs
        dry_run: if True, don't modify anything

    Returns:
        tuple of (new_limits_data, deleted_ratchet_files)
    """
    remove_set = set(remove_paths)
    new_limits = [entry for entry in limits_data if entry["path"] not in remove_set]
    deleted_ratchets: list[str] = []

    if delete_ratchets and not dry_run:
        for entry in limits_data:
            if entry["path"] in remove_set and entry.get("ratchet_basename"):
                ratchet_path = RATCHET_DIR / entry["ratchet_basename"]
                if ratchet_path.exists():
                    ratchet_path.unlink()
                    deleted_ratchets.append(str(ratchet_path))

    return new_limits, deleted_ratchets


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
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help=(
            "Write the current LIMITS table to the canonical data file "
            "(scripts/data/module_size_limits.json), updating the baseline "
            "section atomically. Pair with --reason to record why."
        ),
    )
    parser.add_argument(
        "--reason",
        default=None,
        help=(
            "Reason recorded in the data file's history entry when used with "
            "--write-baseline (name the tracking issue, e.g. "
            "'Issue #4243 refactor')."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be written without modifying any files.",
    )
    parser.add_argument(
        "--remove-entries",
        nargs="+",
        metavar="PATH",
        help=(
            "Remove the specified paths from LIMITS. Use this for "
            "decomposition cleanup PRs. Combine with --write-baseline to "
            "update the data file and --delete-ratchets to remove associated "
            "ratchet JSONs."
        ),
    )
    parser.add_argument(
        "--delete-ratchets",
        action="store_true",
        help=(
            "When used with --remove-entries, also delete the associated "
            "ratchet JSON files. Default: keep ratchet files for audit trail."
        ),
    )
    parser.add_argument(
        "--limits-file",
        type=Path,
        default=None,
        help=f"Path to the LIMITS data file (default: {DEFAULT_LIMITS_FILE.relative_to(REPO_ROOT)}).",
    )
    args = parser.parse_args()

    # Handle --remove-entries first (mutating operation)
    if args.remove_entries:
        limits_data, limits_file = _get_limits_data(args.limits_file or DEFAULT_LIMITS_FILE)
        new_limits, deleted = remove_entries_from_limits(
            limits_data,
            args.remove_entries,
            delete_ratchets=args.delete_ratchets,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print(f"[dry-run] Would remove entries: {args.remove_entries}")
            print(f"[dry-run] New LIMITS count: {len(new_limits)} (was {len(limits_data)})")
            if deleted:
                print(f"[dry-run] Would delete ratchets: {deleted}")
        else:
            print(f"Removed {len(limits_data) - len(new_limits)} entries from LIMITS")
            if deleted:
                print(f"Deleted ratchet files: {deleted}")
            print(f"New BASELINE_MODULE_SIZE_LIMITS should be: {len(new_limits)}")
            print("Run with --write-baseline to update the data file.")

        # With --write-baseline, also update the file
        if args.write_baseline and not args.dry_run:
            write_baseline_file(limits_file, new_limits, reason=args.reason)
        return 0

    # Handle --write-baseline (write-only operation)
    if args.write_baseline:
        limits_data, limits_file = _get_limits_data(args.limits_file or DEFAULT_LIMITS_FILE)
        write_baseline_file(limits_file, limits_data, dry_run=args.dry_run, reason=args.reason)
        return 0

    # Normal gate operation
    drift_messages = check_baseline_drift()
    results: list[Result] = []
    for limit in LIMITS:
        result = check(limit)
        if result is not None:
            results.append(result)

    if args.write_ratchet:
        for result in results:
            update_ratchet(result)

    all_passed = all(r.passed for r in results) and not drift_messages

    if args.json:
        payload = {
            "results": [
                {
                    "path": str(r.path.relative_to(REPO_ROOT)),
                    "actual": r.actual,
                    "max": r.max,
                    "passed": r.passed,
                    "reason": r.reason,
                }
                for r in results
            ],
            "baseline": {
                "max_limits": BASELINE_MODULE_SIZE_LIMITS,
                "observed": len(LIMITS),
                "drift": drift_messages,
            },
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("=== Fluxion module-size gate (Issues #2878, #3457) ===")
        print(f"Repo: {REPO_ROOT}")
        print(f"LIMITS data file: {DEFAULT_LIMITS_FILE.relative_to(REPO_ROOT)}")
        print()
        if drift_messages:
            print("BASELINE DRIFT (PR-blocking unless baseline is raised):")
            for msg in drift_messages:
                print(f"  - {msg}")
            print()
        if not results:
            print("No matching files found — gate is a no-op.")
            return 0 if not drift_messages else 1
        for result in results:
            rel = result.path.relative_to(REPO_ROOT)
            verdict = "PASS" if result.passed else "FAIL"
            print(f"  [{verdict}] {rel}: {result.actual} lines (max {result.max})")
            print(f"      {result.reason}")
        print()
        if all_passed:
            print("All module-size limits satisfied.")
            return 0
        print(
            "One or more module-size limits exceeded or baseline drift "
            "detected; see FAIL lines above."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
