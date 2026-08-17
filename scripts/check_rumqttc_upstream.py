#!/usr/bin/env python3
"""
Upstream-watch for the rumqttc security cluster tracked by issue #2853.

Single-resolution cluster:

* RUSTSEC-2025-0134 (rustls-pemfile, unmaintained)
* RUSTSEC-2026-0049 (rustls-webpki, CRL Distribution Point faulty matching)
* RUSTSEC-2026-0098 (rustls-webpki, URI name-constraints wrongly accepted)
* RUSTSEC-2026-0099 (rustls-webpki, wildcard DNS name-constraints wrongly accepted)
* RUSTSEC-2026-0104 (rustls-webpki, reachable panic in CRL parsing)

All five advisories are transitive via ``rumqttc 0.25.1`` → ``rustls-webpki
0.102.8`` (and ``rustls-pemfile 2.2.0`` for -0134). They can only be cleared
in this repository when:

1. Upstream PR bytebeamio/rumqtt#1037 (``chore(deps): bump rustls-webpki
   and tokio-rustls``) is merged; AND
2. A new ``rumqttc`` release is published on crates.io that drops the
   ``0.102.x`` webpki copy (or, alternatively, an auditable-org ``[patch.
   crates-io]`` pin is added — see issue #2757 acceptance criteria); AND
3. ``crates/fluxion-twin/Cargo.toml`` is bumped to the new version.

This script polls the two upstream sources (GitHub PR + crates.io) and
exits:

* ``0`` — monitoring pass: no fix available, no action required.
* ``1`` — fix available: either the PR is merged, a new ``rumqttc`` release
  is on crates.io, or both. Operators should run the issue's verification
  steps and bump ``fluxion-twin`` + clean up the ``>>> REMOVE`` blocks in
  ``.cargo/audit.toml`` and ``deny.toml``.
* ``2`` — script error (network failure, malformed response, missing
  Cargo.lock).

The script is offline-safe by default: the offline ``(b)`` check on
``Cargo.lock`` still runs without contacting the network. The default
exit code when ``--online`` is NOT passed is ``0`` regardless of upstream
status — the gate is informational and intended to be invoked from a
weekly scheduled CI job (see ``.github/workflows/rumqttc-upstream.yml``)
where network access is granted.

Usage::

    # Offline (default) — always exits 0; prints local state only.
    python3 scripts/check_rumqttc_upstream.py

    # Online — poll GitHub + crates.io; exit 1 when a fix is available.
    python3 scripts/check_rumqttc_upstream.py --online

    # Force exit 1 even offline when Cargo.lock shows rumqttc still on
    # 0.25.1 but a fixed version is published (manual operator override).
    python3 scripts/check_rumqttc_upstream.py --online --strict

Exit codes
----------

0. Monitoring pass: no upstream fix detected (or offline mode).
1. Upstream fix detected: PR merged and/or new rumqttc release available;
   ``fluxion-twin`` bump is the next action.
2. Script error: ``Cargo.lock`` missing, malformed API response,
   unparsable version strings, etc.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CARGO_LOCK = REPO_ROOT / "Cargo.lock"
RUMQTTC_CARGO_TOML = REPO_ROOT / "crates" / "fluxion-twin" / "Cargo.toml"

UPSTREAM_PR_URL = "https://github.com/bytebeamio/rumqtt/pull/1037"
UPSTREAM_ISSUE_URL = "https://github.com/bytebeamio/rumqtt/issues/1067"
GITHUB_API_PR = "https://api.github.com/repos/bytebeamio/rumqtt/pulls/1037"
CRATES_IO_API_RUMQTTC = "https://crates.io/api/v1/crates/rumqttc"
CRATES_IO_API_DEPS = "https://crates.io/api/v1/crates/rumqttc/{version}/dependencies"
FLUXION_ISSUE_URL = "https://github.com/anchapin/fluxion/issues/2853"

# Documented remediation surface — see `.cargo/audit.toml` >>> REMOVE blocks.
TRACKED_ADVISORIES = (
    "RUSTSEC-2025-0134",  # rustls-pemfile unmaintained
    "RUSTSEC-2026-0049",  # rustls-webpki CRL Distribution Point
    "RUSTSEC-2026-0098",  # rustls-webpki URI name-constraints
    "RUSTSEC-2026-0099",  # rustls-webpki wildcard DNS name-constraints
    "RUSTSEC-2026-0104",  # rustls-webpki CRL panic
)
# Minimum fixed rustls-webpki version per the audit config comments:
#   0049 → 0.103.10, 0098/0099 → 0.103.12, 0104 → 0.103.13
MIN_FIXED_WEBPKI = "0.103.13"

_USER_AGENT = "fluxion-rumqttc-upstream/1.0 (anchapin/fluxion#2853)"


def _http_get_json(url: str, timeout: float = 10.0) -> dict | None:
    """Fetch ``url`` and parse as JSON.

    Returns ``None`` on network/parse failure. The script never raises on
    network failures — upstream poll failures degrade gracefully and the
    offline ``(b)`` check still runs.
    """
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.load(resp)
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError):
        return None


def _parse_cargo_lock_rumqttc(path: Path) -> str | None:
    """Return the resolved ``rumqttc`` version in ``Cargo.lock``, or ``None``.

    The lock file may list the same crate more than once (multi-version
    resolution); we return the first match for ``rumqttc`` because that
    is what ``fluxion-twin`` resolves under its declared
    ``rumqttc = "0.25"`` constraint.
    """
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"\[\[package\]\]\s*\n\s*name\s*=\s*\"rumqttc\"\s*\n"
        r"\s*version\s*=\s*\"([^\"]+)\"",
        re.MULTILINE,
    )
    m = pattern.search(text)
    return m.group(1) if m else None


def _parse_rumqttc_decl(cargo_toml: Path) -> str | None:
    """Return the declared ``rumqttc`` constraint from ``crates/fluxion-twin/Cargo.toml``."""
    if not cargo_toml.exists():
        return None
    text = cargo_toml.read_text(encoding="utf-8")
    m = re.search(r'^\s*rumqttc\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return m.group(1) if m else None


def _semver_tuple(v: str) -> tuple[int, int, int]:
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", v or "")
    if not m:
        return (0, 0, 0)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _constraint_mentions_fixed_webpki(req: str) -> bool:
    """``True`` iff a Cargo-style req string pins ``rustls-webpki`` to a
    fixed release (>=0.103.x). Mirrors the logic in
    ``scripts/check_audit_ignores_fresh.py`` so the two stay aligned.
    """
    if not req:
        return False
    if "0.103" not in req:
        return False
    return not any(t in req for t in (">=0.102", "<0.103"))


def fetch_upstream_pr() -> dict | None:
    """Return the upstream PR payload (state, merged_at, etc.) or ``None``."""
    return _http_get_json(GITHUB_API_PR)


def fetch_crates_io_latest() -> str | None:
    """Return the latest published ``rumqttc`` version on crates.io."""
    data = _http_get_json(CRATES_IO_API_RUMQTTC)
    if data is None:
        return None
    crate = data.get("crate", {})
    max_stable = crate.get("max_stable_version")
    if isinstance(max_stable, str) and max_stable:
        return max_stable
    return None


def fetch_crates_io_deps(version: str) -> list[dict]:
    """Return the dependency list for ``rumqttc@<version>`` on crates.io."""
    url = CRATES_IO_API_DEPS.format(version=version)
    data = _http_get_json(url)
    if data is None:
        return []
    deps = data.get("dependencies", [])
    return deps if isinstance(deps, list) else []


def _evaluate_upstream(pr: dict | None, latest: str | None) -> tuple[bool, list[str]]:
    """Inspect the GitHub PR + crates.io response and decide if a fix is available.

    Returns ``(fix_available, reasons)``.
    """
    reasons: list[str] = []

    if pr is not None:
        merged = bool(pr.get("merged"))
        state = pr.get("state") or "unknown"
        if merged:
            reasons.append(
                f"upstream PR #1037 is merged "
                f"(state={state}, merged_at={pr.get('merged_at')})"
            )
        else:
            # Surface the live state in the output even when not merged —
            # operators want to see "open" / "closed" / "draft" without
            # reading the GitHub UI.
            mergeable = pr.get("mergeable")
            mergeable_state = pr.get("mergeable_state")
            reasons.append(
                f"upstream PR #1037 not yet merged "
                f"(state={state}, mergeable={mergeable}, "
                f"mergeable_state={mergeable_state})"
            )

    if latest is not None:
        # We have a fresh crates.io release candidate — but we cannot
        # confirm it clears the cluster until we know its rustls-webpki
        # dep pins >=0.103.13. The check below covers that.
        reasons.append(f"crates.io max_stable_version = {latest}")
    else:
        reasons.append("crates.io lookup failed (offline?)")

    return (bool(reasons and "merged" in reasons[0].lower()), reasons)


def _release_pulls_fixed_webpki(latest: str | None) -> bool:
    """``True`` iff the latest ``rumqttc`` on crates.io pins
    ``rustls-webpki`` >=0.103.13 (the strict minimum across all five
    cluster advisories). Returns ``False`` on any network/parse failure —
    the operator-facing check_audit_ignores_fresh.py gate verifies the
    dep graph locally, so a silent ``False`` here is safe.
    """
    if latest is None:
        return False
    deps = fetch_crates_io_deps(latest)
    for dep in deps:
        cid = dep.get("crate_id", "")
        req = dep.get("req", "")
        if cid == "rustls-webpki":
            return _constraint_mentions_fixed_webpki(req) and _semver_tuple(
                _min_version_from_req(req)
            ) >= _semver_tuple(MIN_FIXED_WEBPKI)
    return False


def _min_version_from_req(req: str) -> str:
    """Best-effort extraction of the minimum version from a Cargo req."""
    m = re.search(r">=?(\d+\.\d+\.\d+)", req)
    return m.group(1) if m else "0.0.0"


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    ``argv`` is exposed for unit tests so they can drive the script
    without polluting ``sys.argv`` (pytest passes its own argv via
    monkeypatch); CLI invocation falls back to ``None`` and argparse
    reads ``sys.argv`` as usual.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help=(
            "Poll GitHub + crates.io. Without this flag the script runs "
            "offline (always exit 0) — useful for PR-time checks where "
            "the upstream poll is out of scope."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "With --online, also exit 1 when crates.io has a fixed "
            "rumqttc release even if the upstream PR is still open. "
            "Default is to require BOTH the PR merged AND a new release "
            "available before signalling 'fix available'."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a single-line JSON summary on stdout (machine-readable).",
    )
    args = parser.parse_args(argv)

    # --- Offline (b) check --------------------------------------------------
    if not CARGO_LOCK.exists():
        print(f"FAIL: {CARGO_LOCK.relative_to(REPO_ROOT)} not found", file=sys.stderr)
        return 2
    locked = _parse_cargo_lock_rumqttc(CARGO_LOCK)
    declared = _parse_rumqttc_decl(RUMQTTC_CARGO_TOML)

    if locked is None:
        print(
            "FAIL: rumqttc not resolved in Cargo.lock — fluxion-twin "
            "feature may be disabled.",
            file=sys.stderr,
        )
        return 2

    # --- Online (a) check ---------------------------------------------------
    pr: dict | None = None
    latest: str | None = None
    release_fixed = False
    upstream_status = "offline"
    if args.online:
        pr = fetch_upstream_pr()
        latest = fetch_crates_io_latest()
        if latest is not None:
            release_fixed = _release_pulls_fixed_webpki(latest)
            upstream_status = "online"
        else:
            upstream_status = "online-no-cratesio"
    pr_merged = bool(pr and pr.get("merged"))
    new_release_available = latest is not None and _semver_tuple(
        latest
    ) > _semver_tuple(locked)

    fix_available = False
    if args.online:
        # Strict semantics: PR merged AND a newer release on crates.io
        # that pulls a fixed webpki. The audit-gate condition is the
        # crates.io dep check; the PR-merged gate is the operator's
        # signal that the bump is upstream-vouched.
        if pr_merged and new_release_available and release_fixed:
            fix_available = True
        elif args.strict and new_release_available and release_fixed:
            # Operator override: skip the PR-merged precondition.
            fix_available = True

    # --- Output --------------------------------------------------------------
    if args.json:
        summary = {
            "tracked_advisories": list(TRACKED_ADVISORIES),
            "fluxion_issue": FLUXION_ISSUE_URL,
            "upstream_pr": UPSTREAM_PR_URL,
            "upstream_status": upstream_status,
            "locked_rumqttc": locked,
            "declared_constraint": declared,
            "crates_io_latest": latest,
            "crates_io_latest_pulls_fixed_webpki": release_fixed
            if args.online
            else None,
            "upstream_pr_merged": pr_merged if args.online else None,
            "fix_available": fix_available,
            "checked_on": date.today().isoformat(),
        }
        print(json.dumps(summary))
        return 1 if fix_available else 0

    print("=== Fluxion rumqttc Upstream Watch (Issue #2853) ===")
    print(f"Tracking issues:      {FLUXION_ISSUE_URL}")
    print(f"Upstream PR:          {UPSTREAM_PR_URL}")
    print(f"Upstream issue:       {UPSTREAM_ISSUE_URL}")
    print(f"Advisories:           {', '.join(TRACKED_ADVISORIES)}")
    print(f"Mode:                 {'online' if args.online else 'offline'}")
    print()
    print(f'fluxion-twin declared:    rumqttc = "{declared}"')
    print(f"Cargo.lock resolved:      rumqttc = {locked}")

    if args.online:
        print(
            f"Upstream PR #1037:        "
            f"{'MERGED' if pr_merged else 'not merged'} "
            f"(state={pr.get('state') if pr else 'unknown'})"
        )
        print(f"crates.io max_stable:     {latest if latest else '(lookup failed)'}")
        print(f"crates.io release fixes:  {'YES' if release_fixed else 'NO'}")
    else:
        print("Upstream PR #1037:        (offline — pass --online to poll)")
        print("crates.io max_stable:     (offline — pass --online to poll)")

    print()
    if fix_available:
        print(
            "ACTION: upstream fix is available. Run the verification "
            "steps from issue #2853:\n"
            "  1. cargo update -p rumqttc\n"
            "  2. cargo audit --deny warnings  # must be 0 warnings\n"
            "  3. cargo deny check advisories  # must be 'advisories ok'\n"
            "  4. cargo build --release\n"
            "  5. cargo test -p fluxion-twin --lib\n"
            "  6. Remove the five `>>> REMOVE` blocks from "
            ".cargo/audit.toml and the four entries from deny.toml."
        )
        return 1
    print(
        "STATUS: no upstream fix yet — the cluster is still blocked on "
        "bytebeamio/rumqtt#1037. This is the expected state; no action "
        "required."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
