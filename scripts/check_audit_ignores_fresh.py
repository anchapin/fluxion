#!/usr/bin/env python3
"""
Audit-Ignore Freshness Check for Fluxion.

Parses ``.cargo/audit.toml``, extracts each advisory ignore entry's
``>>> REMOVE`` block, and verifies the documented removal conditions
against the current repository state. Exits non-zero when at least one
condition has been met — i.e. when the ignore entry can be tightened.

Acceptance criteria (issue #2912):

    For each entry, the script treats the ``>>> REMOVE`` block as a
    removal contract and surfaces (exits non-zero) when at least one of:

        (a) Upstream PR/RFC is merged and released — checked via the
            public crates.io index for the version constraint declared
            in the REMOVE block. (Network-dependent; opt-in via
            ``--check-upstream``.)
        (b) ``Cargo.lock`` no longer resolves the vulnerable transitive
            — verified by scanning the lock file for the patched version
            constraint, or for the crate's absence when the advisory has
            "no patched versions".
        (c) Entry has a recorded removal-date that has passed — verified
            by parsing an ISO-8601 date from the REMOVE block.

The default invocation runs only the offline (b) and (c) checks.
Condition (a) is the canonical "upstream fix landed" path but requires
network access; it is implemented as a flag-gated check that is off by
default.

Historical pattern (#2749 stale deny.toml advisory-not-detected, #2750
cargo-audit gate leniency, #2681 unsound advisories papered over)
shows that documented-ignore lists decay into permanent ones unless
something actively checks them. When bytebeamio/rumqtt#1037 ships, the
four rustls-webpki entries and the rustls-pemfile entry should be
removed in the same PR that bumps ``crates/fluxion-twin/Cargo.toml``;
this gate is what surfaces that opportunity.

Usage::

    python3 scripts/check_audit_ignores_fresh.py
    python3 scripts/check_audit_ignores_fresh.py --check-upstream
    python3 scripts/check_audit_ignores_fresh.py --quiet

Exit codes:
    0 — All ``>>> REMOVE`` blocks have at least one blocking condition.
    1 — One or more REMOVE conditions are met; entries should be removed.
    2 — Script error (e.g. ``.cargo/audit.toml`` missing or unparseable).
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_TOML = REPO_ROOT / ".cargo" / "audit.toml"
DENY_TOML = REPO_ROOT / "deny.toml"
CARGO_LOCK = REPO_ROOT / "Cargo.lock"


def parse_cargo_lock_versions(path: Path) -> dict[str, list[str]]:
    """Parse ``Cargo.lock`` and return a mapping of crate name -> versions.

    A single crate may appear multiple times in the lock (multi-version
    resolution), so the value is a list. ``[[patch.*]]`` overlays are
    reflected in the same list because Cargo rewrites the package block
    in-place.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    text = path.read_text(encoding="utf-8")
    out: dict[str, list[str]] = {}
    pattern = re.compile(
        r"\[\[package\]\]\s*\n\s*name\s*=\s*\"([^\"]+)\"\s*\n"
        r"\s*version\s*=\s*\"([^\"]+)\"",
        re.MULTILINE,
    )
    for m in pattern.finditer(text):
        name, version = m.group(1), m.group(2)
        out.setdefault(name, []).append(version)
    return out


def _find_ignore_block_bounds(lines: list[str]) -> tuple[int, int] | None:
    """Return (start_index, end_index) of the ``ignore = [...]`` block.

    ``start_index`` is the line with ``ignore = [``; ``end_index`` is the
    matching closing ``]`` line (inclusive). Returns ``None`` if the
    block cannot be located.
    """
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*ignore\s*=\s*\[", line):
            start = i
            break
    if start is None:
        return None
    for j in range(start + 1, len(lines)):
        stripped = lines[j].strip()
        if stripped.startswith("]"):
            return (start, j)
    return (start, len(lines) - 1)


def parse_ignore_entries(text: str) -> list[dict]:
    """Parse the ignore list and return one record per advisory entry.

    Each record contains:

        - ``id``: the ``RUSTSEC-YYYY-NNNN`` identifier.
        - ``line``: 1-indexed line number where the entry appears.
        - ``remove_block``: list of comment-line strings belonging to the
          closest preceding ``>>> REMOVE`` block. Multiple consecutive
          entries share the same block until a new ``>>> REMOVE``
          directive appears.

    Lines outside the ignore block (the file header, the ``[advisories]``
    preamble) are intentionally skipped — they do not scope a single
    entry.
    """
    lines = text.splitlines()
    bounds = _find_ignore_block_bounds(lines)
    if bounds is None:
        return []
    start, end = bounds

    entries: list[dict] = []
    pending_remove_block: list[str] = []

    for k in range(start + 1, end):
        line = lines[k]
        stripped = line.strip()

        if stripped.startswith("#"):
            if ">>> REMOVE" in stripped:
                pending_remove_block = [stripped]
            elif pending_remove_block:
                pending_remove_block.append(stripped)
            continue

        m = re.match(r'^\s*"(RUSTSEC-\d{4}-\d{4})"', line)
        if m:
            entries.append(
                {
                    "id": m.group(1),
                    "line": k + 1,
                    "remove_block": list(pending_remove_block),
                }
            )

    return entries


_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def _version_tuple(v: str) -> tuple[int, int, int]:
    m = _VERSION_RE.search(v)
    if not m:
        return (0, 0, 0)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def version_gte(actual: str, required: str) -> bool:
    """Return True if ``actual >= required`` semver-style (dotted triple)."""
    return _version_tuple(actual) >= _version_tuple(required)


def extract_cargo_lock_constraint(
    remove_block: list[str],
) -> tuple[str, str] | None:
    """Extract a ``Cargo.lock`` version constraint from a REMOVE block.

    Recognises the explicit verification pattern documented in the
    rustls-webpki cluster REMOVE block:

        ``Cargo.lock resolves `<crate>` to >=<version>``

    Returns ``(crate_name, min_version)`` or ``None`` when the block has
    no parseable Cargo.lock constraint.
    """
    text = "\n".join(remove_block)
    m = re.search(
        r"Cargo\.lock\s+resolves\s+`?([\w-]+)`?\s+to\s+>=?\s*(\d+\.\d+\.\d+)",
        text,
    )
    if m:
        return (m.group(1), m.group(2))
    return None


def extract_absence_constraint(remove_block: list[str]) -> str | None:
    """Extract a "crate should be absent from Cargo.lock" constraint.

    When the REMOVE block (or its surrounding context) states ``no
    patched versions``, the only way to clear the advisory is to drop
    the crate from the dependency graph entirely. Returns the crate name
    to assert absence for, or ``None`` when no such statement is found.
    """
    text = "\n".join(remove_block).lower()
    if "no patched versions" not in text:
        return None
    for crate in ("rustls-pemfile", "paste", "ttf-parser"):
        if crate in text:
            return crate
    return None


def extract_removal_date(remove_block: list[str]) -> date | None:
    """Extract an ISO-8601 removal-date from a REMOVE block.

    Looks for any ``YYYY-MM-DD`` shape in the block. Returns the parsed
    date or ``None`` when no date is present.
    """
    text = "\n".join(remove_block)
    m = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if not m:
        return None
    try:
        return date.fromisoformat(m.group(1))
    except ValueError:
        return None


def evaluate_entry(
    entry: dict,
    lock_versions: dict[str, list[str]],
    today: date,
) -> tuple[bool, list[str]]:
    """Evaluate the three removal conditions for a single entry.

    Returns ``(any_met, reasons)``. ``any_met`` is ``True`` iff at least
    one condition currently triggers (the entry should be removed);
    ``reasons`` is the human-readable list of which conditions fired.
    """
    reasons: list[str] = []
    remove_block = entry["remove_block"]

    # (a) Upstream release check — network-dependent, skipped in default
    # mode. The flag-gated implementation lives outside this function.

    # (b) Cargo.lock constraint check.
    constraint = extract_cargo_lock_constraint(remove_block)
    if constraint is not None:
        crate, min_version = constraint
        versions = lock_versions.get(crate, [])
        if not versions:
            reasons.append(
                f"(b) Cargo.lock does not contain `{crate}` at all — "
                f"advisory is cleared from the workspace."
            )
        else:
            low = [v for v in versions if not version_gte(v, min_version)]
            if not low:
                reasons.append(
                    f"(b) Cargo.lock resolves `{crate}` entirely to "
                    f">= {min_version} (versions present: "
                    f"{', '.join(sorted(set(versions)))})."
                )
            # else: still vulnerable, keep ignore.

    absence = extract_absence_constraint(remove_block)
    if absence is not None:
        if absence not in lock_versions:
            reasons.append(
                f"(b) Cargo.lock no longer contains `{absence}` "
                f"(advisory has no patched version)."
            )
        # else: still in lock, keep ignore.

    # (c) Removal-date check. A future date does NOT trigger — it
    # documents a planned removal that has not yet come due. A past date
    # DOES trigger.
    removal_date = extract_removal_date(remove_block)
    if removal_date is not None and removal_date < today:
        reasons.append(
            f"(c) Recorded removal-date {removal_date.isoformat()} "
            f"has passed (today: {today.isoformat()})."
        )

    return (bool(reasons), reasons)


def fetch_crates_io_max_version(crate: str) -> str | None:
    """Fetch the max published version of ``crate`` from crates.io.

    Used by the opt-in condition (a) check. Returns the latest version
    string, or ``None`` on network/parse failure. This is intentionally
    a thin wrapper: the network call is only made when the operator
    passes ``--check-upstream``.
    """
    import json
    import urllib.error
    import urllib.request

    url = f"https://crates.io/api/v1/crates/{crate}"
    req = urllib.request.Request(url, headers={"User-Agent": "fluxion-audit-gate/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            data = json.load(resp)
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError):
        return None
    max_stable = data.get("crate", {}).get("max_stable_version")
    if isinstance(max_stable, str) and max_stable:
        return max_stable
    return None


def fetch_crates_io_dependencies(crate: str, version: str) -> list[dict]:
    """Fetch the dependency list for ``crate@version`` from crates.io.

    Returns the list of dependency dicts (each has ``crate_id`` and
    ``req`` keys), or an empty list on network/parse failure.
    """
    import json
    import urllib.error
    import urllib.request

    url = f"https://crates.io/api/v1/crates/{crate}/{version}/dependencies"
    req = urllib.request.Request(url, headers={"User-Agent": "fluxion-audit-gate/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            data = json.load(resp)
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError):
        return []
    deps = data.get("dependencies", [])
    return deps if isinstance(deps, list) else []


def _constraint_mentions_fixed_webpki(req: str) -> bool:
    """Return True if a Cargo-style version requirement mentions a fixed webpki.

    Recognises the documented remediation paths (>=0.103.10 / >=0.103.12 /
    >=0.103.13) plus the broad "0.103" prefix match. We deliberately do
    NOT match "0.102" because every 0.102.x release is vulnerable per
    the rustls-webpki advisories (#2757).
    """
    if not req:
        return False
    if "0.103" not in req:
        return False
    return not any(t in req for t in (">=0.102", "<0.103"))


def evaluate_upstream(
    remove_block: list[str],
    lock_versions: dict[str, list[str]],
) -> str | None:
    """Return a reason string if condition (a) is met, else ``None``.

    Condition (a) — upstream PR/RFC merged and released. The REMOVE
    block typically states "once a fixed ``<crate>`` is published on
    crates.io". We interpret "fixed" as:

      1. crates.io reports a published version that is strictly greater
         than every version of the crate currently resolved in
         Cargo.lock, OR
      2. the crate has been dropped from Cargo.lock entirely (in which
         case the entry is obsolete).

    We additionally verify the dependency tree of the new release to
    confirm it pulls a fixed ``rustls-webpki`` (>=0.103.x). If we cannot
    verify the dependency tree (network failure, ambiguous constraint),
    we return ``None`` to avoid false positives — the script then
    surfaces the entry only when condition (b) or (c) fires locally.
    """
    text = "\n".join(remove_block)
    m = re.search(r"(?:fixed|patched)\s+`?(\w[\w-]*)`?\s+is\s+published", text)
    if not m:
        return None
    crate = m.group(1)
    max_pub = fetch_crates_io_max_version(crate)
    if max_pub is None:
        return None

    resolved = lock_versions.get(crate, [])
    if not resolved:
        return (
            f"(a) crates.io has `{crate}@{max_pub}` published but the "
            f"crate is no longer in Cargo.lock — entry is obsolete."
        )

    max_pub_tuple = _version_tuple(max_pub)
    max_locked_tuple = max((_version_tuple(v) for v in resolved), default=(0, 0, 0))
    if max_pub_tuple <= max_locked_tuple:
        return None

    deps = fetch_crates_io_dependencies(crate, max_pub)
    if not deps:
        return None

    for dep in deps:
        cid = dep.get("crate_id", "")
        if cid in ("rustls-webpki", "webpki") and _constraint_mentions_fixed_webpki(
            dep.get("req", "")
        ):
            return (
                f"(a) crates.io has `{crate}@{max_pub}` published "
                f"(newer than locked {max_locked_tuple}); the release "
                f"declares `{cid} {dep.get('req', '')}` which is "
                f">=0.103.x — upstream fix is available; bump the "
                f"lock to clear the entry."
            )

    return None


def _check_file(
    path: Path,
    lock_versions: dict[str, list[str]],
    today: date,
    args,
) -> tuple[list[tuple[dict, list[str]]], list[str]]:
    """Check a single audit config file and return (surfacable, no_remove_block)."""
    surfacable: list[tuple[dict, list[str]]] = []
    no_remove_block: list[str] = []

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: cannot read {path}: {exc}", file=sys.stderr)
        return ([], [])

    try:
        entries = parse_ignore_entries(text)
    except Exception as exc:
        print(f"ERROR parsing {path}: {exc}", file=sys.stderr)
        return ([], [])

    print(f"Found {len(entries)} advisory ignore entries in {path.name}.")

    for entry in entries:
        rid = entry["id"]
        line = entry["line"]
        rb = entry["remove_block"]

        if not args.quiet:
            print(f"-- {rid} (line {line}) --")

        if not rb:
            no_remove_block.append(rid)
            if not args.quiet:
                print(
                    "  No `>>> REMOVE` block — gate not applicable "
                    "(advisory still tracked manually)."
                )
                print()
            continue

        if not args.quiet:
            print(f"  REMOVE: {rb[0].strip()}")

        any_met, reasons = evaluate_entry(entry, lock_versions, today)

        if args.check_upstream:
            upstream_reason = evaluate_upstream(rb, lock_versions)
            if upstream_reason is not None:
                reasons.append(upstream_reason)
                any_met = True

        if not args.quiet:
            if reasons:
                for r in reasons:
                    print(f"  TRIGGER: {r}")
            else:
                constraint = extract_cargo_lock_constraint(rb)
                absence = extract_absence_constraint(rb)
                rd = extract_removal_date(rb)
                if constraint is not None:
                    crate, mv = constraint
                    vs = lock_versions.get(crate, [])
                    print(
                        f"  (b) Cargo.lock has `{crate}` versions: "
                        f"{', '.join(sorted(set(vs))) or 'absent'} — "
                        f"not all >= {mv}, keep ignore."
                    )
                elif absence is not None:
                    print(
                        f"  (b) Cargo.lock still contains `{absence}` — "
                        f"keep ignore."
                    )
                else:
                    print(
                        "  (b) No automated Cargo.lock check available "
                        "for this entry (only upstream (a) or removal-"
                        "date (c) can fire."
                    )
                if rd is not None:
                    print(
                        f"  (c) Recorded removal-date: {rd.isoformat()} "
                        f"(in the future)."
                    )
                else:
                    print("  (c) No recorded removal-date.")
                if not args.check_upstream:
                    print(
                        "  (a) Upstream check disabled (offline mode). "
                        "Pass --check-upstream to poll crates.io."
                    )
                print("  STATUS: REMOVE conditions NOT met — keep ignore.")
            print()

        if any_met:
            surfacable.append((entry, reasons))

    return surfacable, no_remove_block


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check-upstream",
        action="store_true",
        help=(
            "Enable condition (a): poll crates.io for upstream releases "
            "of the crate named in the REMOVE block. Disabled by "
            "default to keep CI offline."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-entry details; print only the summary.",
    )
    args = parser.parse_args()

    today = date.today()

    print("=== Fluxion Audit-Ignore Freshness Check (Issue #2912, #3237) ===")
    print(f"Repo:           {REPO_ROOT}")
    print(f"Lock file:      {CARGO_LOCK.relative_to(REPO_ROOT)}")
    print(f"Today:          {today.isoformat()}")
    print(
        "Upstream check: "
        + ("ENABLED (will poll crates.io)" if args.check_upstream else "DISABLED (offline mode; conditions (b)+(c) only)")
    )
    print()

    # Check both .cargo/audit.toml and deny.toml (Issue #3237)
    files_to_check = [
        (AUDIT_TOML, ".cargo/audit.toml"),
        (DENY_TOML, "deny.toml"),
    ]

    for path, _label in files_to_check:
        if path.exists():
            print(f"Scanning {path.relative_to(REPO_ROOT)}...")
        else:
            print(f"Skipping {path.relative_to(REPO_ROOT)} (not found)")
            continue

    print()

    try:
        lock_versions = parse_cargo_lock_versions(CARGO_LOCK)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"Loaded {len(lock_versions)} packages from "
        f"{CARGO_LOCK.relative_to(REPO_ROOT)}."
    )
    print()

    all_surfacable: list[tuple[dict, list[str]]] = []
    all_no_remove_block: list[str] = []

    for path, _label in files_to_check:
        if not path.exists():
            continue

        surfacable, no_remove_block = _check_file(path, lock_versions, today, args)
        all_surfacable.extend(surfacable)
        all_no_remove_block.extend(no_remove_block)
        print()

    print("=" * 64)
    print()

    if all_no_remove_block:
        print(
            f"NOTE: {len(all_no_remove_block)} advisory entries have no "
            f"`>>> REMOVE` block (gate not enforced):"
        )
        for rid in all_no_remove_block:
            print(f"  - {rid}")
        print(
            "  These advisories are tracked manually or with prose "
            "removal guidance rather than the structured `>>> REMOVE` "
            "contract. Adding a `>>> REMOVE` block would put them "
            "under this gate; see issue #2912."
        )
        print()

    if all_surfacable:
        print(
            f"FAIL: {len(all_surfacable)} advisory ignore entries have met "
            f"removal conditions:"
        )
        for entry, reasons in all_surfacable:
            print(f"  - {entry['id']} (line {entry['line']})")
            for r in reasons:
                print(f"      {r}")
        print()
        print(
            "Action: remove these entries from `.cargo/audit.toml` and/or "
            "`deny.toml`. The removal conditions documented in the "
            "`>>> REMOVE` block are now satisfied — see issue #2912, #3237."
        )
        return 1

    print(
        "PASS: All `>>> REMOVE` blocks have at least one blocking "
        "condition. No advisory ignore entries can be tightened today."
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
