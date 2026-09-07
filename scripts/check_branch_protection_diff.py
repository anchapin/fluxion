#!/usr/bin/env python3
"""
Branch-protection diff diagnostic for the `develop` branch (Issue #3383).

Reads the canonical ``ci.required_checks`` list from ``release_gates.yaml``,
queries GitHub for the live ``develop`` branch-protection state, and prints
the diff (which required-checks entries are missing from live vs declared,
and the current ``enforce_admins.enabled`` value). On a non-empty diff,
prints the **desired** PUT payload that would reconcile the live state to
the canonical state — but does **NOT** execute the PUT itself.

This script is the diagnostic half of the reconciliation. The destructive
half (actually applying the PUT against the live GitHub branch-protection
endpoint) is intentionally NOT in scope here per the auto-improvement-loop
guardrail:

    "Destructive actions are SKIPPED, not paused. Specifically:
     ``scripts/apply_branch_protection.py`` may be implemented and tested
     but NOT run with ``--write`` against live GitHub. This is the
     're-executing settings-as-code that mutates GitHub branch protection'
     guardrail. File a tracked issue and surface in the final summary."

Tracking issues:

- #3383 — reconcile live develop branch-protection (this script's scope)
- #3386 — idempotent ``apply_branch_protection.py`` (the future destructive
  applier; gate on a manual operator run, not an automated cron)

The script's exit codes:

- 0 — no diff (live matches canonical, ``enforce_admins`` is true)
- 1 — diff present (printed; required-checks differ or ``enforce_admins``
       is false). The PUT payload is printed to stderr for the operator
       to review before invoking the destructive applier.
- 2 — script error (release_gates.yaml missing/unparseable, ``gh`` CLI
       unavailable, network/auth failure)

Usage::

    python3 scripts/check_branch_protection_diff.py
    python3 scripts/check_branch_protection_diff.py --repo anchapin/fluxion
    python3 scripts/check_branch_protection_diff.py --json   # machine-readable
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is a dev-dep everywhere
    print("ERROR: PyYAML required; install with `pip install pyyaml`.", file=sys.stderr)
    sys.exit(2)


REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE_GATES = REPO_ROOT / "release_gates.yaml"


def load_canonical_required_checks(path: Path) -> list[str]:
    """Return ``release_gates.yaml::ci.required_checks`` verbatim.

    Mirrors the contract enforced by ``scripts/check_required_checks_sync.py``:
    every entry must be the exact ``jobs.<id>.name`` string GitHub reports,
    no canonical/suffix tolerance (Issue #3116).
    """
    if not path.exists():
        print(f"ERROR: {path} missing", file=sys.stderr)
        sys.exit(2)

    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    try:
        return list(data["ci"]["required_checks"])
    except (KeyError, TypeError):
        print("ERROR: release_gates.yaml missing `ci.required_checks`", file=sys.stderr)
        sys.exit(2)


def fetch_live_protection(repo: str) -> dict:
    """Query GitHub for the live ``develop`` branch protection.

    Returns the raw JSON payload from
    ``/repos/{owner}/{repo}/branches/develop/protection``.
    Raises ``subprocess.CalledProcessError`` on auth/network failure.

    Strips ``GH_TOKEN`` from the subprocess environment because the project
    shell (``~/.zshrc``) exports an invalid token that overrides the working
    default-account token — see auto-improvement-loop guardrail.
    """
    cmd = [
        "gh", "api",
        f"/repos/{repo}/branches/develop/protection",
    ]
    env = {k: v for k, v in os.environ.items() if k != "GH_TOKEN"}
    out = subprocess.check_output(
        cmd, text=True, stderr=subprocess.PIPE, env=env
    )
    return json.loads(out)


def compute_diff(canonical: list[str], live_required: list[str]) -> dict:
    """Set-symmetric diff between the canonical and live required-checks.

    ``missing_from_live`` is the set of canonical checks that GitHub is
    NOT enforcing — the ones the desired PUT payload would add. ``extra_in_live``
    is the set GitHub IS enforcing that the canonical list does not call for
    (currently empty in practice but reported for completeness).
    """
    canonical_set = set(canonical)
    live_set = set(live_required)
    return {
        "missing_from_live": sorted(canonical_set - live_set),
        "extra_in_live": sorted(live_set - canonical_set),
        "in_both": sorted(canonical_set & live_set),
    }


def build_desired_put_payload(canonical: list[str], enforce_admins: bool) -> dict:
    """Build the JSON body for ``PUT .../protection/required_status_checks``.

    Per the GitHub REST API contract, the ``contexts`` array is the verbatim
    list of required check names; ``strict`` controls whether branches must
    be up-to-date before merging. The envelope for the PUT is documented at
    https://docs.github.com/en/rest/branches/branch-protection.

    The ``enforce_admins.enabled`` toggle is set on the protection endpoint
    itself (NOT the required_status_checks endpoint), so we surface a second
    payload that an operator would need to PUT to ``/protection`` to flip
    ``enforce_admins``. The script does not call either endpoint.
    """
    return {
        "_note": (
            "Diagnostic output from scripts/check_branch_protection_diff.py. "
            "This payload was NOT applied. See Issue #3386 for the destructive "
            "applier that would PUT this to GitHub (gated on operator review)."
        ),
        "protection_put_payload": {
            "url": f"/repos/<owner>/<repo>/branches/develop/protection",
            "body": {
                "enforce_admins": {"enabled": enforce_admins},
            },
        },
        "required_status_checks_put_payload": {
            "url": "/repos/<owner>/<repo>/branches/develop/protection/required_status_checks",
            "body": {
                "strict": True,
                "contexts": canonical,
            },
        },
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n",1)[0])
    parser.add_argument(
        "--repo",
        default="anchapin/fluxion",
        help="GitHub owner/repo to query (default: anchapin/fluxion)",
    )
    parser.add_argument(
        "--release-gates",
        type=Path,
        default=RELEASE_GATES,
        help=f"Path to release_gates.yaml (default: {RELEASE_GATES})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON (the diff + desired PUT payload)",
    )
    args = parser.parse_args(argv)

    canonical = load_canonical_required_checks(args.release_gates)
    try:
        live = fetch_live_protection(args.repo)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: gh api failed (exit {exc.returncode}): {exc.stderr}", file=sys.stderr)
        return 2
    except FileNotFoundError:
        print("ERROR: `gh` CLI not on PATH", file=sys.stderr)
        return 2

    live_required = list(
        (live.get("required_status_checks") or {}).get("contexts") or []
    )
    enforce_admins = bool(
        (live.get("enforce_admins") or {}).get("enabled")
    )

    diff = compute_diff(canonical, live_required)
    has_diff = bool(diff["missing_from_live"] or diff["extra_in_live"])
    has_enforce_drift = enforce_admins is False
    drift = has_diff or has_enforce_drift

    payload = build_desired_put_payload(canonical, enforce_admins=True)
    report = {
        "repo": args.repo,
        "canonical_count": len(canonical),
        "live_count": len(live_required),
        "enforce_admins": enforce_admins,
        "diff": diff,
        "desired_put": payload,
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1 if drift else 0

    print(f"Repo: {args.repo}")
    print(f"Canonical required_checks: {len(canonical)}")
    print(f"Live required_status_checks.contexts: {len(live_required)}")
    print(f"Live enforce_admins.enabled: {enforce_admins}")
    print()
    if not drift:
        print("OK: live state matches canonical (no diff).")
        return 0

    if diff["missing_from_live"]:
        print("MISSING FROM LIVE (canonical -> live gap):")
        for name in diff["missing_from_live"]:
            print(f"  - {name}")
        print()
    if diff["extra_in_live"]:
        print("EXTRA IN LIVE (live -> canonical gap, should be empty):")
        for name in diff["extra_in_live"]:
            print(f"  - {name}")
        print()
    if has_enforce_drift:
        print("DRIFT: enforce_admins.enabled is false (canonical: true).")
        print()
    print("DESIRED PUT PAYLOAD (NOT APPLIED — see Issue #3386 for the applier):")
    print(json.dumps(payload, indent=2))
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))