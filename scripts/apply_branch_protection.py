#!/usr/bin/env python3
"""Idempotent applier for branch-protection required-checks sync (Issue #3386).

The companion script ``scripts/check_required_checks_sync.py`` detects drift
between ``release_gates.yaml::ci.required_checks`` and the live GitHub branch
protection (when ``FLUXION_CHECK_LIVE_PROTECTION=1`` is set). It is by design
read-only — when drift is detected, ``scripts/check_required_checks_sync.py``
prints remediation guidance but does not modify the live branch protection.

This script is the apply side. It:

1. Reads ``release_gates.yaml::ci.required_checks`` (and the
   ``ci.workflow_index`` map) using the helpers from
   ``scripts/check_required_checks_sync.py`` — no duplicated YAML parsing.
2. ``gh api GET /repos/<repo>/branches/<branch>/protection`` to fetch the
   live state.
3. Computes a JSON PUT payload (the diff) that reconciles the live state to
   ``release_gates.yaml``. Default mode ``--dry-run`` only prints the payload
   and exits; ``--write`` actually performs the PUT and re-GETs to verify.
4. Supports ``--branch`` (default ``develop``; ``main`` also supported).

This script is fail-closed: it never proceeds with a write unless the
diff is well-formed, ``gh auth`` is configured, and the operator has
explicitly opted in via ``--write``. The default ``--dry-run`` is the safe
mode and is what every contributor and PR-blocking CI invocation should use.

Usage::

    # Dry-run (default): print the JSON PUT payload, exit 0
    python3 scripts/apply_branch_protection.py --branch develop --dry-run

    # Machine-readable dry-run
    python3 scripts/apply_branch_protection.py --branch develop --dry-run --json

    # Apply (destructive — requires operator-supervised ``--write`` flag)
    python3 scripts/apply_branch_protection.py --branch develop --write

Exit codes::

    0 — dry-run: diff computed and printed (or no diff, "already in sync").
        write: PUT succeeded AND re-GET verified the new state.
    1 — drift detected and ``--write`` was passed but the PUT failed, or the
        re-GET did not match the expected state.
    2 — script error (``release_gates.yaml`` missing, ``gh`` not on PATH,
        ``gh auth`` not configured, invalid arguments).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "anchapin/fluxion"
DEFAULT_BRANCH = "develop"

_REMOTE_URL_RE = re.compile(
    r"(?:https?://[^/]+/|git@[^:]+:)(?P<repo>[^/]+/.+?)(?:\.git)?$"
)


def resolve_default_repo(env: dict[str, str] | None = None) -> str:
    """Resolve the target ``owner/repo`` used when ``--repo`` is omitted.

    Resolution order (Issue #3429): ``$GITHUB_REPOSITORY`` (always set on
    GitHub Actions — correct on forks, mirrors, and overflow runners with
    a different remote) → the ``origin`` git remote (normalized from
    https or ssh form) → the ``DEFAULT_REPO`` constant fallback. Mirrors
    ``check_branch_protection_diff.resolve_default_repo``; the scripts are
    intentionally standalone so the helper is duplicated rather than
    shared. A hardcoded default made the *destructive* applier silently
    target upstream from any checkout whose remote differs.
    """
    if env is None:
        env = dict(os.environ)
    candidate = env.get("GITHUB_REPOSITORY", "").strip()
    if candidate:
        return candidate
    try:
        out = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return DEFAULT_REPO
    m = _REMOTE_URL_RE.match(out.stdout.strip())
    if not m:
        return DEFAULT_REPO
    return m.group("repo")

try:
    from scripts.check_required_checks_sync import (  # type: ignore[import-not-found]
        get_required_checks,
        get_workflow_index,
        load_release_gates,
    )
except ImportError:  # pragma: no cover - allow direct invocation from repo root
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.check_required_checks_sync import (  # type: ignore[no-redef]
        get_required_checks,
        get_workflow_index,
        load_release_gates,
    )


def _gh(*args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    """Invoke ``gh`` with the given args, returning the CompletedProcess.

    ``check=False`` is intentional — this wrapper surfaces non-zero exit
    codes to the caller so the diff/PUT path can decide whether to retry,
    log, or escalate. Never raises on non-zero exit (the wrapper that
    callers use decides the response).
    """
    return subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def fetch_live_protection(repo: str, branch: str) -> dict:
    """Fetch live branch protection via ``gh api``.

    Returns the parsed JSON payload. Raises ``RuntimeError`` on non-zero
    exit or non-JSON response.
    """
    proc = _gh("api", f"/repos/{repo}/branches/{branch}/protection")
    if proc.returncode != 0:
        raise RuntimeError(
            f"gh api GET failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()[:200] or '(no stderr)'}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"gh api response is not JSON: {exc}") from exc


def build_put_payload(
    required_checks: list[str],
    strict: bool = True,
) -> dict:
    """Build the JSON payload that PUT /protection expects.

    Only the four fields this script manages are included; everything else
    (e.g. ``restrictions``) is left untouched in the live payload via a
    separate ``required_status_checks`` patch.

    The shape matches the GitHub REST API documented at
    https://docs.github.com/en/rest/branches/branch-protection#update-branch-protection.
    """
    return {
        "required_status_checks": {
            "strict": strict,
            "contexts": list(required_checks),
        },
        "enforce_admins": True,
        "required_pull_request_reviews": {
            "required_approving_review_count": 1,
        },
        "restrictions": None,
    }


def compute_diff(live: dict, payload: dict) -> dict:
    """Compute a human-readable diff between live protection and the payload.

    Returns a dict with keys ``add``, ``remove``, ``already_present`` so the
    operator can review exactly what would change before re-running with
    ``--write``. The fields mirror the payload's structure (contexts list,
    enforce_admins.enabled, required_approving_review_count, strict).
    """
    live_rsc = live.get("required_status_checks") or {}
    live_contexts = set(live_rsc.get("contexts") or [])
    payload_contexts = set(payload["required_status_checks"]["contexts"])

    live_admins = (live.get("enforce_admins") or {}).get("enabled", False)
    payload_admins = payload.get("enforce_admins", True)

    live_rpr = live.get("required_pull_request_reviews") or {}
    live_approving = live_rpr.get("required_approving_review_count") or 0
    payload_approving = (payload.get("required_pull_request_reviews") or {}).get(
        "required_approving_review_count", 1
    )

    live_strict = live_rsc.get("strict", False)
    payload_strict = payload["required_status_checks"]["strict"]

    return {
        "contexts": {
            "add": sorted(payload_contexts - live_contexts),
            "remove": sorted(live_contexts - payload_contexts),
            "already_present": sorted(payload_contexts & live_contexts),
        },
        "enforce_admins": {
            "from": live_admins,
            "to": payload_admins,
            "would_change": live_admins != payload_admins,
        },
        "required_approving_review_count": {
            "from": live_approving,
            "to": payload_approving,
            "would_change": live_approving != payload_approving,
        },
        "strict": {
            "from": live_strict,
            "to": payload_strict,
            "would_change": live_strict != payload_strict,
        },
    }


def diff_has_changes(diff: dict) -> bool:
    """Return True iff any field in the diff would actually change."""
    if diff["contexts"]["add"] or diff["contexts"]["remove"]:
        return True
    if diff["enforce_admins"]["would_change"]:
        return True
    if diff["required_approving_review_count"]["would_change"]:
        return True
    return bool(diff["strict"]["would_change"])


def print_human_diff(diff: dict, branch: str) -> None:
    """Print the diff in human-readable form to stdout."""
    print(f"Branch: {branch}")
    print()
    print("Required status checks:")
    if diff["contexts"]["add"]:
        print(f"  + add:    {diff['contexts']['add']}")
    if diff["contexts"]["remove"]:
        print(f"  - remove: {diff['contexts']['remove']}")
    if diff["contexts"]["already_present"]:
        print(f"  = keep:   {diff['contexts']['already_present']}")
    if not (
        diff["contexts"]["add"]
        or diff["contexts"]["remove"]
        or diff["contexts"]["already_present"]
    ):
        print("  (none)")
    print()
    print(
        f"enforce_admins: {diff['enforce_admins']['from']} -> "
        f"{diff['enforce_admins']['to']}"
        + ("  (CHANGES)" if diff["enforce_admins"]["would_change"] else "")
    )
    print(
        f"required_approving_review_count: "
        f"{diff['required_approving_review_count']['from']} -> "
        f"{diff['required_approving_review_count']['to']}"
        + (
            "  (CHANGES)"
            if diff["required_approving_review_count"]["would_change"]
            else ""
        )
    )
    print(
        f"strict: {diff['strict']['from']} -> {diff['strict']['to']}"
        + ("  (CHANGES)" if diff["strict"]["would_change"] else "")
    )


def put_protection(
    repo: str,
    branch: str,
    payload: dict,
) -> tuple[bool, str]:
    """PUT the payload to the live branch protection.

    Returns (success, message). On failure, message contains the gh stderr
    (truncated to 200 chars) so the operator can diagnose.
    """
    proc = subprocess.run(
        [
            "gh",
            "api",
            "--method",
            "PUT",
            f"/repos/{repo}/branches/{branch}/protection",
            "--input",
            "-",
        ],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if proc.returncode != 0:
        return False, (
            f"PUT failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()[:200] or '(no stderr)'}"
        )
    return True, "PUT succeeded"


def verify_protection(
    repo: str,
    branch: str,
    expected_payload: dict,
) -> tuple[bool, str]:
    """Re-GET the live protection and verify it matches ``expected_payload``.

    Returns (success, message). On success, message is "verified"; on
    failure, message names which fields still diverge.
    """
    try:
        live = fetch_live_protection(repo, branch)
    except RuntimeError as exc:
        return False, f"verification GET failed: {exc}"

    diff = compute_diff(live, expected_payload)
    if diff_has_changes(diff):
        return False, f"verification mismatch: {json.dumps(diff, indent=2)}"
    return True, "verified"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Idempotent applier for branch-protection required-checks sync "
            "(Issue #3386). Default is --dry-run; --write is destructive."
        ),
    )
    parser.add_argument(
        "--branch",
        default=DEFAULT_BRANCH,
        help=f"branch to reconcile (default: {DEFAULT_BRANCH})",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help=(
            "target repo (default: $GITHUB_REPOSITORY, else the origin "
            "remote, else anchapin/fluxion)"
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "actually PUT the payload (destructive). Default is --dry-run, "
            "which only prints the diff."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the payload + diff as JSON (machine-readable).",
    )
    args = parser.parse_args()
    args.repo = args.repo or resolve_default_repo()

    try:
        gates = load_release_gates()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    required_checks = get_required_checks(gates)
    _ = get_workflow_index(gates)  # not used directly, but loaded to validate YAML
    payload = build_put_payload(required_checks)

    try:
        live = fetch_live_protection(args.repo, args.branch)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    diff = compute_diff(live, payload)

    if args.json:
        out = {
            "branch": args.branch,
            "repo": args.repo,
            "mode": "write" if args.write else "dry-run",
            "would_change": diff_has_changes(diff),
            "diff": diff,
            "payload": payload,
        }
        print(json.dumps(out, indent=2))
    else:
        mode_label = "WRITE" if args.write else "DRY-RUN"
        print(f"== {mode_label}: {args.repo}@{args.branch} ==")
        print()
        print_human_diff(diff, args.branch)
        print()
        if not diff_has_changes(diff):
            print("No changes required — live protection already matches.")
        elif args.write:
            print("Will apply the above changes.")
        else:
            print("Re-run with --write to apply.")

    if not diff_has_changes(diff):
        return 0

    if not args.write:
        return 0

    if not diff_has_changes(diff):
        return 0

    # --write path: PUT, then re-GET to verify.
    print()
    print(f"Applying PUT to {args.repo}@{args.branch} ...")
    ok, msg = put_protection(args.repo, args.branch, payload)
    if not ok:
        print(f"ERROR: {msg}", file=sys.stderr)
        return 1
    print(msg)

    print()
    print("Re-GET to verify ...")
    ok, msg = verify_protection(args.repo, args.branch, payload)
    if not ok:
        print(f"ERROR: {msg}", file=sys.stderr)
        return 1
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
