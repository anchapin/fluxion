#!/usr/bin/env python3
"""
Downward Trend Guard for the ``sim <-> validation`` cycle (Issue #2768).

The existing ``scripts/check_ashrae_cases_cycle.py`` (#1441 / #2495) and
``scripts/check_physics_sim_cycle.py`` (#2463) enforce a *magnitude*
contract: the cycle edge count must stay at or below a grandfathered
baseline. They cannot detect two pathologies that this script catches:

1. **No downward progress** — the count stays at the baseline run after run,
   which is the "frozen, not broken" gap documented in issue #2768. Goal #3
   is a *directional* contract (coupling should shrink toward zero), not a
   magnitude contract.
2. **Net-flat edge swap** — a PR removes an edge in one file and adds a
   different edge in another, netting flat. The magnitude gate passes but
   the cycle's *shape* changed without authorisation. The new edge may be
   higher-criticality than the one it replaced (e.g. swapping a
   `validation::diagnostics` import for a fresh `CaseSpec` match arm).

This script reads ``scripts/cycle_baseline_history.json`` (an append-only
ledger of ``(timestamp, commit, totals, edge_signature)`` snapshots) and
enforces three directional rules:

* **R1 (no growth)**: ``current_total > last_total`` -> FAIL. Redundant
  with the magnitude gate but a second line of defence that fires when a
  cycle-removal PR has lowered the in-source baseline and a follow-up PR
  grows the count back (still under the old baseline, so the magnitude
  gate would pass).
* **R2 (downward progress)**: when run with ``--nightly``, FAIL if the
  last ``STALE_THRESHOLD`` snapshots (default 14) all have the same
  ``total``. Drives the architecture toward zero. Only enforced in the
  nightly cron job to avoid blocking ordinary PRs that do not touch the
  cycle.
* **R3 (no net-flat edge swap)**: ``current_total == last_total`` but the
  sorted multiset of ``(file, lineno, scanned-line)`` tuples has changed
  -> FAIL. Catches the swap path the magnitude gate cannot see.

Usage:
  python3 scripts/check_cycle_downward_trend.py             # per-PR (R1 + R3)
  python3 scripts/check_cycle_downward_trend.py --nightly   # nightly (R1 + R2 + R3)
  python3 scripts/check_cycle_downward_trend.py --update    # append a snapshot

Exit codes:
  0 -- trend healthy (no growth, no swap; nightly mode: progress within window)
  1 -- trend regression (growth / swap; nightly mode: stale)
  2 -- script error (corrupt history file, scan failure)

Reset policy: the history ledger is append-only. The ONLY authorised way
to extend it with a higher ``total`` is an architectural sign-off commit
that also updates the baselines in ``scripts/check_ashrae_cases_cycle.py``
and ``ARCHITECTURE.md`` Section "Cycle break (#1441)". Silently rewriting
the ledger to hide a regression defeats the purpose and is a blocking
review issue. See ``ARCHITECTURE.md`` Section "Downward trend guard
(Issue #2768)" for the full policy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORY_FILE = REPO_ROOT / "scripts" / "cycle_baseline_history.json"
SCHEMA_VERSION = 1
STALE_THRESHOLD_NIGHTS = 14

# Bucket labels (kept stable across history versions). Order matters: the
# ledger's ``buckets`` array mirrors this tuple so a reader can correlate
# the ``totals`` dict across snapshots.
BUCKETS = (
    "sim_to_validation",
    "validation_to_sim",
    "validation_to_physics",
    "validation_to_weather",
    "physics_to_sim",
    "sim_to_physics",
)


def _load_module(name: str, path: Path) -> Any:
    """Import a sibling script as a module without polluting ``sys.path``.

    Mirrors ``scripts/ci/test_check_physics_sim_cycle.py::_load_checker`` so
    the cycle-scan primitives stay the single source of truth: this guard
    never re-implements the scan logic, it only consumes the offender lists
    the existing scripts already produce.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_cycle_scripts() -> tuple[Any, Any]:
    """Load both existing cycle-check scripts as modules.

    The detection logic stays in those scripts; this guard only consumes
    their public scan functions. Issue #2768 explicitly forbids modifying
    the existing detection logic.
    """
    acc = _load_module(
        "check_ashrae_cases_cycle",
        REPO_ROOT / "scripts" / "check_ashrae_cases_cycle.py",
    )
    psc = _load_module(
        "check_physics_sim_cycle",
        REPO_ROOT / "scripts" / "check_physics_sim_cycle.py",
    )
    return acc, psc


def collect_current_edges(acc: Any, psc: Any) -> dict[str, Any]:
    """Run all six scan functions and return a structured snapshot.

    Returns a dict with:
      ``totals``    -- {bucket: count} for each label in ``BUCKETS``.
      ``total``     -- sum of ``totals.values()``.
      ``signature`` -- sha256 over the sorted offender strings
                       (``file:line: text``), so any change to the set of
                       edges -- even one that nets the total to flat -- is
                       detected by R3.
    """
    offenders: dict[str, list[str]] = {
        "sim_to_validation": acc.scan_sim_for_validation_deps(),
        "validation_to_sim": acc.scan_validation_for_sim_deps(),
        "validation_to_physics": acc.scan_validation_for_physics_deps(),
        "validation_to_weather": acc.scan_validation_for_weather_deps(),
        "physics_to_sim": psc.scan_physics_for_sim_deps(),
        "sim_to_physics": psc.scan_protected_sim_files_for_physics_deps(),
    }
    totals = {k: len(v) for k, v in offenders.items()}
    total = sum(totals.values())
    flat = sorted(off for parts in offenders.values() for off in parts)
    signature = hashlib.sha256("\n".join(flat).encode("utf-8")).hexdigest()
    return {"totals": totals, "total": total, "signature": signature}


def load_history(path: Path = HISTORY_FILE) -> dict[str, Any]:
    """Load the history ledger, or synthesise an empty cold-start ledger.

    A missing file is NOT an error: it signals a fresh install and the
    per-PR check passes warm (see ``evaluate_per_pr``). A present but
    corrupt or wrong-schema file IS an error and exits 2 so a botched
    manual edit cannot silently disable the guard.
    """
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "buckets": list(BUCKETS),
            "snapshots": [],
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"corrupt history file {path}: {e}") from e
    if data.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"history file {path} schema_version "
            f"{data.get('schema_version')!r} != expected {SCHEMA_VERSION}; "
            "manual migration required (see ARCHITECTURE.md Section "
            "'Downward trend guard (Issue #2768)')"
        )
    if "snapshots" not in data or not isinstance(data["snapshots"], list):
        raise RuntimeError(
            f"history file {path} missing 'snapshots' list; manual repair required"
        )
    return data


def save_history(history: dict[str, Any], path: Path = HISTORY_FILE) -> None:
    """Persist the ledger with stable formatting (deterministic diff)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(history, indent=2, sort_keys=False) + "\n"
    path.write_text(text, encoding="utf-8")


def _latest_snapshot(history: dict[str, Any]) -> dict[str, Any] | None:
    snaps = history.get("snapshots", [])
    return snaps[-1] if snaps else None


def evaluate_per_pr(
    current: dict[str, Any], last: dict[str, Any] | None
) -> tuple[int, list[str]]:
    """R1 + R3 -- no growth, no net-flat swap.

    Returns ``(exit_code, messages)``. ``last is None`` is the cold-start
    warmup path: per-PR check passes with a note telling the operator to
    seed the ledger via ``--update``.
    """
    msgs: list[str] = []
    if last is None:
        msgs.append(
            "WARMUP: history ledger has no prior snapshot; per-PR trend "
            "check deferred. Run `python3 scripts/check_cycle_downward_"
            "trend.py --update` once to seed the ledger, then commit the "
            "updated file as part of the cycle-removal workflow."
        )
        return 0, msgs

    if current["total"] > last["total"]:
        delta = current["total"] - last["total"]
        msgs.append(
            f"R1 FAIL: total cycle edges grew from {last['total']} to "
            f"{current['total']} (+{delta}). The sim<->validation cycle "
            "must trend toward zero; this guard rejects growth."
        )
        msgs.append("    Per-bucket delta:")
        for bucket in BUCKETS:
            was = last["totals"].get(bucket, 0)
            now = current["totals"].get(bucket, 0)
            if now > was:
                msgs.append(f"      {bucket}: {was} -> {now} (+{now - was})")
        return 1, msgs

    if current["total"] == last["total"]:
        if current["signature"] != last.get("edge_signature", last.get("signature")):
            msgs.append(
                f"R3 FAIL: total holds at {current['total']} but the set "
                "of edges changed (net-flat swap detected). Adding a new "
                "cycle edge while removing a different one is not allowed "
                "even when the net count is unchanged; the new edge may "
                "be higher criticality than the one it replaced."
            )
            msgs.append(
                f"    last signature:    {last.get('edge_signature', last.get('signature'))[:16]}"
            )
            msgs.append(f"    current signature: {current['signature'][:16]}")
            return 1, msgs
        msgs.append(
            f"OK: total holds at {current['total']} with unchanged edge "
            "signature (no growth, no swap)."
        )
        return 0, msgs

    # current_total < last_total -- downward progress.
    delta = last["total"] - current["total"]
    msgs.append(
        f"OK: downward progress -- total cycle edges fell from "
        f"{last['total']} to {current['total']} (-{delta}). When stable, "
        "update the baselines in scripts/check_ashrae_cases_cycle.py and "
        "ARCHITECTURE.md Section 'Cycle break (#1441)' via the "
        "cycle-removal workflow, then append a snapshot via "
        "`scripts/check_cycle_downward_trend.py --update`."
    )
    return 0, msgs


def evaluate_nightly(
    history: dict[str, Any], current: dict[str, Any]
) -> tuple[int, list[str]]:
    """R1 + R2 + R3 -- the nightly cron's full enforcement.

    R2 fails only when the ledger has at least ``STALE_THRESHOLD_NIGHTS``
    snapshots AND the trailing window is monotonic-flat at the current
    total. This drives the architecture toward zero without blocking PRs
    that do not touch the cycle (the per-PR check stays R1+R3 only).
    """
    last = _latest_snapshot(history)
    code, msgs = evaluate_per_pr(current, last)
    if code != 0:
        return code, msgs

    snaps = history.get("snapshots", [])
    if len(snaps) >= STALE_THRESHOLD_NIGHTS:
        tail = snaps[-STALE_THRESHOLD_NIGHTS:]
        if all(s["total"] == current["total"] for s in tail):
            msgs.append(
                f"R2 FAIL: total has been frozen at {current['total']} for "
                f"{STALE_THRESHOLD_NIGHTS} consecutive snapshots. Goal #3 "
                "requires the sim<->validation cycle to trend toward zero. "
                "Either land a cycle-removal PR (which lowers the total and "
                "resets the window) or document an explicit exception in "
                "ARCHITECTURE.md Section 'Downward trend guard (Issue #2768)'."
            )
            return 1, msgs
        msgs.append(
            f"R2 OK: total has moved within the last {STALE_THRESHOLD_NIGHTS} "
            "snapshots (downward progress within window)."
        )
    else:
        msgs.append(
            f"R2 deferred: ledger has {len(snaps)} snapshot(s), need "
            f"{STALE_THRESHOLD_NIGHTS} before enforcing the stale-window. "
            "Seed the ledger via the nightly cron's `--update` step."
        )
    return 0, msgs


def append_snapshot(
    history: dict[str, Any],
    current: dict[str, Any],
    commit: str | None,
    source: str,
) -> dict[str, Any]:
    """Append a snapshot to the ledger in-place; return ``history``."""
    snap = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit or "unknown",
        "source": source,
        "totals": current["totals"],
        "total": current["total"],
        "edge_signature": current["signature"],
    }
    history.setdefault("snapshots", []).append(snap)
    return history


def _git_commit_sha() -> str | None:
    """Best-effort short SHA for the snapshot's ``commit`` field.

    Returns ``None`` if git is unavailable (e.g. test sandbox); the
    snapshot records ``"unknown"`` instead. Called only from ``--update``.
    """
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=str(REPO_ROOT),
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Downward-trend guard for the sim<->validation cycle (#2768)."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--nightly",
        action="store_true",
        help="also enforce R2 (downward progress within last "
        f"{STALE_THRESHOLD_NIGHTS} snapshots); intended for the cron job",
    )
    mode.add_argument(
        "--update",
        action="store_true",
        help="append the current snapshot to the history ledger and exit "
        "(intended for the cycle-removal workflow operator; the nightly "
        "cron does NOT auto-commit)",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=HISTORY_FILE,
        help="path to the history ledger JSON (default: "
        "scripts/cycle_baseline_history.json)",
    )
    parser.add_argument(
        "--source",
        default="manual-update",
        help="label recorded in the snapshot's `source` field when using "
        "--update (default: manual-update)",
    )
    args = parser.parse_args(argv)

    print(
        f"Cycle downward-trend guard (issue #2768) -- repo: {REPO_ROOT}, "
        f"history: {args.history}"
    )
    print()

    try:
        acc, psc = _load_cycle_scripts()
        current = collect_current_edges(acc, psc)
    except Exception as e:
        print(f"ERROR: cycle scan failed: {e}", file=sys.stderr)
        return 2

    print("Current cycle edge scan:")
    for bucket in BUCKETS:
        print(f"    {bucket}: {current['totals'][bucket]}")
    print(f"    TOTAL: {current['total']}")
    print(f"    signature: {current['signature'][:16]}")
    print()

    try:
        history = load_history(args.history)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    last = _latest_snapshot(history)
    if last is not None:
        print(
            f"Last snapshot: total={last['total']} at "
            f"{last.get('timestamp')} ({last.get('source', 'unknown')})"
        )
    else:
        print("Last snapshot: <none> (cold-start)")
    print()

    if args.update:
        sha = _git_commit_sha()
        before = len(history.get("snapshots", []))
        append_snapshot(history, current, sha, source=args.source)
        save_history(history, args.history)
        try:
            rel = args.history.relative_to(REPO_ROOT)
        except ValueError:
            rel = args.history
        print(f"Appended snapshot ({before} -> {before + 1} entries) to {rel}")
        print(
            "Commit the updated ledger as part of the cycle-removal workflow; "
            "the nightly cron does NOT auto-commit."
        )
        return 0

    if args.nightly:
        code, msgs = evaluate_nightly(history, current)
    else:
        code, msgs = evaluate_per_pr(current, last)
    for m in msgs:
        print(m)
    print()
    if code == 0:
        print("Downward-trend guard PASSED.")
    else:
        print("Downward-trend guard FAILED.")
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
