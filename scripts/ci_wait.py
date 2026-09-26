#!/usr/bin/env python3
"""Wait for a PR's CI checks to finish, then report pass/fail — Issue #4067.

Polls ``gh pr view --json statusCheckRollup`` for a pull request until every
registered check reaches a terminal state, then exits 0 on success and
non-zero otherwise.

Why this exists
--------------
On 2026-09-26 GitHub twice dropped the ``pull_request`` ``synchronize``
dispatches for pushes to PR #4065/#4066: only the ``push``-event workflows
ran and the head ended up with **zero** registered checks. The previous
wait helper reported ``SUCCESS: All checks PASSED`` because "no failing
checks" was vacuously true — a merge-integrity hazard (a contributor could
merge believing CI was green while the required suite silently never ran).

This script is fail-closed about that case: an empty rollup — or a rollup
containing only *skipped* entries — for the current head is treated as
FAILURE, never as success. At least ``--min-checks`` (default 1)
completed, non-skipped checks must exist for the head before a SUCCESS
verdict is possible.

Recovering from dropped synchronize dispatches
----------------------------------------------
Amend the head and force-push; the new SHA re-dispatches the dropped
events (worked both times on 2026-09-26):

    git commit --amend --no-edit
    git push --force-with-lease

Usage
-----
    python3 scripts/ci_wait.py 4065
    python3 scripts/ci_wait.py 4065 --timeout 1800 --interval 20
    python3 scripts/ci_wait.py 4065 --min-checks 5 --once   # single evaluation

Exit codes: 0 = all checks passed; 1 = a check failed, the rollup was
empty/skipped-only, or fewer than --min-checks non-skipped checks were
registered; 2 = timed out waiting for pending checks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Rollup classification
# ---------------------------------------------------------------------------

# Conclusions / states that mean "this check will never run".
SKIPPED_CONCLUSIONS = {"SKIPPED"}

# CheckRun.status values that are not terminal.
PENDING_STATUSES = {"QUEUED", "IN_PROGRESS", "WAITING", "PENDING", "REQUESTED"}

# CheckRun.conclusion values that count as a hard failure.
FAILED_CONCLUSIONS = {
    "FAILURE",
    "TIMED_OUT",
    "CANCELLED",
    "ACTION_REQUIRED",
    "STARTUP_FAILURE",
}

# StatusContext.state values that count as a hard failure.
FAILED_STATES = {"FAILURE", "ERROR"}

# Conclusions / states that count as an acceptable terminal outcome.
OK_CONCLUSIONS = {"SUCCESS", "NEUTRAL", "SKIPPED"}
OK_STATES = {"SUCCESS"}


@dataclass
class Verdict:
    """Result of evaluating one snapshot of a PR's check rollup."""

    outcome: str  # "success" | "failure" | "pending" | "empty"
    message: str
    total: int = 0
    meaningful: int = 0  # non-skipped entries
    failed: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _entry_name(entry: dict) -> str:
    if entry.get("__typename") == "StatusContext":
        return str(entry.get("context", "<unknown status>"))
    return str(entry.get("name", "<unknown check>"))


def _is_skipped(entry: dict) -> bool:
    """A check that will never run (e.g. a skipped review bot)."""
    if entry.get("__typename") == "StatusContext":
        return False  # legacy statuses have no skipped state
    return entry.get("conclusion") in SKIPPED_CONCLUSIONS


def _is_failure(entry: dict) -> bool:
    if entry.get("__typename") == "StatusContext":
        return entry.get("state") in FAILED_STATES
    return entry.get("conclusion") in FAILED_CONCLUSIONS


def _is_complete(entry: dict) -> bool:
    if entry.get("__typename") == "StatusContext":
        return entry.get("state") not in {"EXPECTED", "PENDING"}
    return entry.get("status") not in PENDING_STATUSES


def evaluate_rollup(
    entries: list[dict],
    merge_state_status: str | None,
    head_sha: str,
    min_checks: int = 1,
) -> Verdict:
    """Classify one rollup snapshot. Pure function — no subprocess calls.

    ``entries`` is the ``statusCheckRollup`` list from
    ``gh pr view --json statusCheckRollup``. The empty/skipped-only case is
    deliberately a failure (Issue #4067): "no failing checks" must never be
    reported as success when GitHub registered nothing for the head.
    """
    warnings: list[str] = []
    if merge_state_status == "UNKNOWN":
        warnings.append(
            "mergeStateStatus is UNKNOWN: GitHub is still computing "
            "mergeability; the rollup below may be incomplete."
        )

    meaningful = [e for e in entries if not _is_skipped(e)]
    failed = [_entry_name(e) for e in meaningful if _is_failure(e)]
    pending = [_entry_name(e) for e in meaningful if _is_complete(e) is False]

    if failed:
        return Verdict(
            outcome="failure",
            message=(
                f"{len(failed)} check(s) failed for head {head_sha}: "
                + ", ".join(failed)
            ),
            total=len(entries),
            meaningful=len(meaningful),
            failed=failed,
            pending=pending,
            warnings=warnings,
        )

    if len(meaningful) < min_checks:
        # Fail-closed: an empty rollup (dropped synchronize dispatches) or a
        # rollup with only skipped entries must never read as SUCCESS.
        detail = (
            "no non-skipped checks registered"
            if not meaningful
            else f"only {len(meaningful)} non-skipped check(s) registered"
        )
        return Verdict(
            outcome="empty",
            message=(
                f"{detail} for head {head_sha} "
                f"(need at least {min_checks}). GitHub may have dropped the "
                "pull_request synchronize dispatches (issue #4067). Recover "
                "with: git commit --amend --no-edit && "
                "git push --force-with-lease"
            ),
            total=len(entries),
            meaningful=len(meaningful),
            failed=failed,
            pending=pending,
            warnings=warnings,
        )

    if pending:
        return Verdict(
            outcome="pending",
            message=(
                f"{len(pending)} check(s) still pending for head {head_sha}: "
                + ", ".join(pending)
            ),
            total=len(entries),
            meaningful=len(meaningful),
            failed=failed,
            pending=pending,
            warnings=warnings,
        )

    return Verdict(
        outcome="success",
        message=(
            f"All {len(meaningful)} non-skipped check(s) passed for head {head_sha}."
        ),
        total=len(entries),
        meaningful=len(meaningful),
        failed=failed,
        pending=pending,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# GitHub polling
# ---------------------------------------------------------------------------


def fetch_pr_data(pr: str) -> dict:
    """Return the parsed ``gh pr view`` JSON for a PR number or URL."""
    proc = subprocess.run(
        [
            "gh",
            "pr",
            "view",
            pr,
            "--json",
            "number,headRefOid,mergeStateStatus,statusCheckRollup",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gh pr view failed for {pr!r}: {proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"could not parse gh output for {pr!r}: {exc}") from exc


def wait_for_checks(
    pr: str,
    timeout: int = 3600,
    interval: int = 30,
    min_checks: int = 1,
    once: bool = False,
) -> tuple[Verdict, dict]:
    """Poll until the rollup is terminal; return (verdict, pr_data)."""
    start = time.monotonic()
    pr_data: dict = {}
    verdict = Verdict(outcome="pending", message="not evaluated yet")
    while True:
        pr_data = fetch_pr_data(pr)
        head_sha = str(pr_data.get("headRefOid", ""))[:12] or "<unknown>"
        entries = pr_data.get("statusCheckRollup") or []
        verdict = evaluate_rollup(
            entries,
            pr_data.get("mergeStateStatus"),
            head_sha,
            min_checks=min_checks,
        )
        elapsed = time.monotonic() - start
        if verdict.outcome in ("success", "failure", "empty") or once:
            return verdict, pr_data
        if elapsed >= timeout:
            verdict = Verdict(
                outcome="pending",
                message=(
                    f"timed out after {int(elapsed)}s waiting for "
                    f"{len(verdict.pending)} pending check(s) on head "
                    f"{head_sha}: " + ", ".join(verdict.pending)
                ),
                total=verdict.total,
                meaningful=verdict.meaningful,
                failed=verdict.failed,
                pending=verdict.pending,
                warnings=verdict.warnings,
            )
            return verdict, pr_data
        time.sleep(interval)


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for a PR's CI checks to finish and report pass/fail. "
            "An empty or skipped-only check rollup for the head is a "
            "FAILURE, never a success (issue #4067)."
        ),
        epilog=(
            "If ci-wait reports no checks registered for the head, GitHub "
            "likely dropped the pull_request synchronize dispatches. "
            "Recover with: git commit --amend --no-edit && "
            "git push --force-with-lease"
        ),
    )
    parser.add_argument("pr", help="PR number or URL to wait on")
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="max seconds to wait for pending checks (default: 3600)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="seconds between rollup polls (default: 30)",
    )
    parser.add_argument(
        "--min-checks",
        type=int,
        default=1,
        help="minimum number of completed non-skipped checks required "
        "for a SUCCESS verdict (default: 1)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="evaluate the current rollup once instead of polling",
    )
    args = parser.parse_args(argv)

    start = time.monotonic()
    try:
        verdict, pr_data = wait_for_checks(
            args.pr,
            timeout=args.timeout,
            interval=args.interval,
            min_checks=args.min_checks,
            once=args.once,
        )
    except RuntimeError as exc:
        print(f"[ci-wait] ERROR: {exc}", file=sys.stderr)
        return 1

    elapsed = _format_duration(time.monotonic() - start)
    number = pr_data.get("number", args.pr)
    for warning in verdict.warnings:
        print(f"[ci-wait] WARNING: {warning}")

    if verdict.outcome == "success":
        print(f"[ci-wait] SUCCESS: All checks PASSED in {elapsed} for PR {number}.")
        return 0
    if verdict.outcome == "empty":
        print(f"[ci-wait] FAILURE (no checks registered): {verdict.message}")
        return 1
    if verdict.outcome == "failure":
        print(f"[ci-wait] FAILURE: {verdict.message}")
        return 1
    if args.once and verdict.outcome == "pending":
        print(f"[ci-wait] PENDING: {verdict.message}")
        return 2
    print(f"[ci-wait] TIMEOUT: {verdict.message}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
