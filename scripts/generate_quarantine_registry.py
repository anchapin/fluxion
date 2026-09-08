#!/usr/bin/env python3
"""Quarantine registry synchroniser (Issue #3211, #3393).

Scans ``tests/**/*.rs`` for ``#[ignore]`` attributes and cross-references
them against the human-curated registry at ``tests/QUARANTINE.md``.

Purpose: close the 149-vs-78 gap documented by Issue #3393 — every
``#[ignore]`` attribute should have a corresponding entry in the registry
(or, if it's diagnostic-only, be located in ``tests/diagnostics/`` where
the registry is intentionally sparse). Without this audit, ``#[ignore]``
tests accumulate silently and the actual CI coverage is opaque.

The script's default mode is **informational**: print a per-category
report and exit 0 so the gate never breaks a green tree on first
adoption. The ``--strict`` flag flips the script into the ratchet the
issue asks for: any orphan ``#[ignore]`` (in code, not in registry)
makes the script exit 1. Wire ``--strict`` into CI once the
existing 71 orphans are triaged into the registry.

Output:

  === Fluxion quarantine registry audit (Issue #3211 / #3393) ===
  Tests directory: tests/
  Registry:        tests/QUARANTINE.md

  Scanned 149 #[ignore] attribute(s) across 47 file(s).
  Registered:     78 (in QUARANTINE.md)
  Orphan:         71 (in code, not in registry)
  Ghost:          0  (in registry, not in code)

  ...
  Exit 0.

Usage::

    python3 scripts/generate_quarantine_registry.py             # default (informational)
    python3 scripts/generate_quarantine_registry.py --strict    # fail on orphan
    python3 scripts/generate_quarantine_registry.py --json      # machine-readable
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
QUARANTINE_MD = REPO_ROOT / "tests" / "QUARANTINE.md"

# ---------------------------------------------------------------------------
# Downward-only ratchet for the quarantine registry (Issue #3443).
#
# Mirrors the `BASELINE_KNOWN_ORPHANS` pattern from
# `scripts/check_orphan_modules.py` (Issue #3459) and the
# `BASELINE_WIRED_BUT_DEAD` pattern (Issue #3458): the constants below
# record the *highest* `len(ORPHANED_IGNORES)` / `len(GHOST_ROWS)` the
# audit has ever accepted. The script FAILS (exit 1) the moment the live
# counts exceed those baselines — adding a new orphan or ghost without
# editing the baseline (with a documenting comment naming the tracking
# issue) is rejected. Lowering the baselines is the only authorised
# change; companion cleanup PRs that resolve an orphan or fix a ghost
# are expected to lower the corresponding baseline by one entry.
#
# History:
#   - 82 → seed (Issue #3443): initial triage of the 82 orphan
#     `#[ignore]` tests documented by the gap-issues audit. Each orphan
#     is now tracked in `tests/QUARANTINE.md` with a blocking-issue
#     reference, and the 23 ghost rows (registry entries with no
#     matching `#[ignore]`) were either re-pointed to the actual
#     function name, consolidated with duplicates, or removed entirely.
#     Companion cleanup PRs that resolve an orphan (e.g. by closing the
#     underlying tracking issue and un-ignoring the test) are expected
#     to drop the matching entry AND lower `BASELINE_ORPHANED_IGNORES`
#     by one.
# ---------------------------------------------------------------------------
BASELINE_ORPHANED_IGNORES = 0
BASELINE_GHOST_ROWS = 0

# Freeze snapshot of the orphan allowlist (Issue #3443 ratchet).
#
# Mirrors the `_BASELINE_KNOWN_ORPHANS_SET` / freeze-snapshot pattern
# from `scripts/check_orphan_modules.py`. At the moment the ratchet
# was introduced (Issue #3443) the orphan list is empty, so the
# snapshot is an empty frozenset. Future cleanup PRs that resolve an
# orphan will not need to touch this freeze set; only PRs that
# reintroduce an orphan (e.g. by re-adding a `#[ignore]` attribute
# without updating the registry) will fail the ratchet check, and
# such PRs are the only ones that need to add the new entry to this
# set AND raise `BASELINE_ORPHANED_IGNORES`.
_BASELINE_ORPHANED_IGNORES_SET: frozenset[tuple[str, str]] = frozenset()

# Freeze snapshot of the ghost rows (Issue #3443 ratchet). Same shape
# as the orphan freeze set: a `frozenset` of `(file, function)` pairs
# that mirrors the registry at freeze time. Editing this set is the
# "raise the ghost baseline" lever — any new ghost MUST be added here
# AND to the registry (and `BASELINE_GHOST_ROWS` must be raised to
# match), with a documenting comment naming the tracking issue.
_BASELINE_GHOST_ROWS_SET: frozenset[tuple[str, str]] = frozenset()

# `#[ignore]` and `#[ignore = "reason"]` (with optional reason string).
# Tolerant of whitespace and trailing comments.
_IGNORE_RE = re.compile(
    r"^\s*#\s*\[\s*ignore\s*(?:=\s*\"([^\"]*)\")?\s*\]\s*(?://.*)?$",
    re.MULTILINE,
)

# `#[cfg_attr(feature = "...", ignore = "reason")]` (gauge-build-only LIMIT-22 cohort).
# Also matches `#[cfg_attr(any(...), ignore)]` for unconditional cfg_attr gates.
# The first capture group is the feature-gate expr (informational only);
# the second capture group is the ignore reason string (optional).
_CFG_ATTR_IGNORE_RE = re.compile(
    r"^\s*#\s*\[\s*cfg_attr\s*\(\s*[^\n]+?,\s*ignore\s*(?:=\s*\"([^\"]*)\")?\s*\)\s*\]\s*(?://.*)?$",
    re.MULTILINE,
)

# Rust test function names: `fn test_xxx(...)` or `fn xxx(...)` inside
# `#[test]` blocks. We deliberately keep this loose so multi-line `fn`
# signatures with attribute blocks above still match.
_TEST_FN_RE = re.compile(
    r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
    re.MULTILINE,
)

# A QUARANTINE.md registry row is a markdown table row whose first cell
# is a backtick-wrapped file path. Example:
#   | `tests/foo.rs` | `test_foo` | #1234 | ... | `pending` |
_TABLE_ROW_RE = re.compile(
    r"^\|\s*`([^`]+)`\s*\|\s*`?([^|`]+)`?\s*\|",
    re.MULTILINE,
)


def _is_comment_only(line: str) -> bool:
    """A line that *talks about* `#[ignore]` in a doc-comment is not a real
    ``#[ignore]`` attribute. Detect Rust doc comments (``//!`` / ``///``)
    and line comments (``//``) so the audit doesn't double-count.
    """
    stripped = line.lstrip()
    return stripped.startswith(("//", "/*", "*"))


def scan_ignores(tests_dir: Path) -> list[dict]:
    """Scan every ``.rs`` file under ``tests_dir`` for ``#[ignore]`` attrs.

    Returns a list of dicts with keys ``file``, ``line``, ``function``,
    ``reason``, ``conditional`` (``True`` for ``cfg_attr(...ignore...)``,
    ``False`` for plain ``#[ignore]``). The ``function`` is the nearest
    ``fn test_xxx`` declaration either before *or* after the ``#[ignore]``
    attribute (best-effort: ``unknown`` if the attribute appears outside
    a test function, e.g. on a module). Rust's attribute syntax allows
    either ordering, so the script must support both.

    When a test has BOTH an unconditional ``#[ignore]`` AND a
    ``#[cfg_attr(..., ignore)]`` (e.g. the LIMIT-22 cohort with a
    fallback unconditional ignore), the unconditional wins for the
    audit — the test is always ignored, so reporting the conditional
    a second time would create a phantom duplicate. The dedup is keyed
    on ``(file, function)`` so different functions in the same file
    are still counted separately.
    """
    raw_results: list[dict] = []
    for path in sorted(tests_dir.rglob("*.rs")):
        text = path.read_text(encoding="utf-8")
        for match in _IGNORE_RE.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_text = text[line_start : text.find("\n", match.start())]
            if _is_comment_only(line_text):
                continue
            function = _nearest_fn(text, match.start(), match.end())
            raw_results.append(
                {
                    "file": str(path.relative_to(REPO_ROOT)),
                    "line": line_no,
                    "function": function,
                    "reason": match.group(1) or "",
                    "conditional": False,
                }
            )
        # Conditional ``cfg_attr(...ignore...)`` — only matches the
        # gauge-build-only LIMIT-22 cohort. These are detected in
        # addition to unconditional ignores.
        for match in _CFG_ATTR_IGNORE_RE.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_text = text[line_start : text.find("\n", match.start())]
            if _is_comment_only(line_text):
                continue
            function = _nearest_fn(text, match.start(), match.end())
            raw_results.append(
                {
                    "file": str(path.relative_to(REPO_ROOT)),
                    "line": line_no,
                    "function": function,
                    "reason": match.group(1) or "",
                    "conditional": True,
                }
            )

    # Dedup pass: when the same ``(file, function)`` pair appears with
    # both an unconditional and a conditional ``#[ignore]``, keep ONLY
    # the unconditional one. A test that is unconditionally ignored
    # does not gain anything from also listing the cfg_attr gate —
    # reporting both would inflate orphan / ghost counts and obscure
    # the actual quarantine state.
    unconditional_keys: set[tuple[str, str]] = {
        (r["file"], r["function"])
        for r in raw_results
        if not r["conditional"]
    }
    results: list[dict] = []
    for entry in raw_results:
        if entry["conditional"] and (
            (entry["file"], entry["function"]) in unconditional_keys
        ):
            continue
        results.append(entry)
    return results


def _nearest_fn(text: str, attr_start: int, attr_end: int) -> str:
    """Find the nearest ``fn name(`` to an attribute offset.

    Searches a ±2 KB window around the attribute and returns the
    closest ``fn`` declaration in either direction. Returns
    ``"unknown"`` if no test function is found within the window.
    """
    window_start = max(0, attr_start - 2000)
    window_end = min(len(text), attr_end + 2000)
    window = text[window_start:window_end]
    fn_matches = list(_TEST_FN_RE.finditer(window))
    if not fn_matches:
        return "unknown"
    ignore_offset = attr_start - window_start
    fn_match = min(
        fn_matches,
        key=lambda m: abs(m.start() - ignore_offset),
    )
    return fn_match.group(1)


def scan_registry(quarantine_md: Path) -> list[dict]:
    """Scan ``quarantine_md`` for table rows.

    Returns a list of dicts with keys ``file``, ``function`` (best-effort,
    the second column). The third column is the blocking-issue(s), but
    the audit script only consumes file + function for the orphan/ghost
    cross-check.
    """
    if not quarantine_md.exists():
        return []
    text = quarantine_md.read_text(encoding="utf-8")
    rows: list[dict] = []
    for match in _TABLE_ROW_RE.finditer(text):
        file_cell = match.group(1).strip()
        fn_cell = match.group(2).strip()
        if not file_cell.endswith(".rs"):
            continue  # skip category-header rows
        rows.append({"file": file_cell, "function": fn_cell})
    return rows


def classify_ignore(ignore: dict) -> str:
    """Bucket an ignore entry into one of the registry's categories.

    The categories mirror the QUARANTINE.md section headers; the
    classification is intentionally heuristic (the registry's human
    curator can override by editing the QUARANTINE.md row). Returns
    one of: ``diagnostic``, ``structural``, ``performance``, ``hardware``,
    ``calibration``, ``ci-broken``, ``manual-baseline``, ``other``.
    """
    reason = ignore["reason"].lower()
    path = ignore["file"].lower()
    if "/diagnostics/" in path or "diagnostic" in reason or "#2536" in reason:
        return "diagnostic"
    if "dhat" in path or "performance" in reason:
        return "performance"
    if "gpu" in reason or "cuda" in reason:
        return "hardware"
    if "#1577" in reason or "ci broken" in reason or "ci infra" in reason:
        return "ci-broken"
    if "calibration" in reason or "data" in reason:
        return "calibration"
    if "manual" in reason and "regener" in reason:
        return "manual-baseline"
    if (
        "limit-" in reason
        or "issue #" in reason
        or "structural" in reason
        or "physics gap" in reason
    ):
        return "structural"
    return "other"


def audit(ignores: list[dict], registry: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (orphans, ghosts).

    Orphan: a real ``#[ignore]`` in code that has no matching row in the
    registry. Match is by ``(file, function_substring)`` so multi-test
    rows (e.g. ``test_dhat_*``) match many actual functions.

    Ghost: a registry row whose file is real but whose function-substring
    does not appear in any scanned ignore. (Many ghost candidates are
    legitimate — they describe a cohort that's been closed via PR and
    the function moved out of quarantine without updating the registry.
    The audit reports them so a curator can decide.)
    """
    registry_keys: set[tuple[str, str]] = set()
    for row in registry:
        registry_keys.add((row["file"], row["function"]))

    orphans: list[dict] = []
    for ignore in ignores:
        # Match against any registry row whose file matches AND whose
        # function-substring appears in the actual function name (or
        # is a wildcard like `test_dhat_*`).
        matched = False
        for rf, rfn in registry_keys:
            if rf != ignore["file"]:
                continue
            if "*" in rfn:
                # Wildcard: `test_dhat_*` matches `test_dhat_anything`.
                prefix = rfn.replace("*", "")
                if prefix and ignore["function"].startswith(prefix.rstrip("_")):
                    matched = True
                    break
                if not prefix:
                    matched = True
                    break
            elif rfn and rfn in ignore["function"]:
                matched = True
                break
        if not matched:
            orphans.append(ignore)

    ignored_keys: set[tuple[str, str]] = set()
    for ignore in ignores:
        ignored_keys.add((ignore["file"], ignore["function"]))

    ghosts: list[dict] = []
    for row in registry:
        if "*" in row["function"]:
            continue  # wildcards are matched against the union, not a single function
        if not any(
            rf == row["file"] and rfn in ignore["function"]
            for ignore in ignores
            for (rf, rfn) in [(row["file"], row["function"])]
        ):
            ghosts.append(row)
    return orphans, ghosts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Quarantine registry audit (Issue #3211 / #3393 / #3443). Default mode "
            "is informational; --strict fails on orphan #[ignore] entries OR on "
            "growth above the BASELINE_ORPHANED_IGNORES / BASELINE_GHOST_ROWS ratchet."
        )
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail (exit 1) on any orphan #[ignore] not in QUARANTINE.md, "
            "on any ghost registry row, or on growth above the "
            "BASELINE_ORPHANED_IGNORES / BASELINE_GHOST_ROWS downward-only "
            "ratchet (Issue #3443)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output for CI consumption.",
    )
    args = parser.parse_args()

    ignores = scan_ignores(TESTS_DIR)
    registry = scan_registry(QUARANTINE_MD)
    orphans, ghosts = audit(ignores, registry)

    by_category: dict[str, int] = {}
    for ignore in ignores:
        cat = classify_ignore(ignore)
        by_category[cat] = by_category.get(cat, 0) + 1

    orphan_keys: set[tuple[str, str]] = {
        (o["file"], o["function"]) for o in orphans
    }
    ghost_keys: set[tuple[str, str]] = {
        (g["file"], g["function"]) for g in ghosts
    }

    if args.json:
        out = {
            "tests_dir": str(TESTS_DIR.relative_to(REPO_ROOT)),
            "registry": str(QUARANTINE_MD.relative_to(REPO_ROOT)),
            "total_ignores": len(ignores),
            "registered": len(registry),
            "orphans": [{k: v for k, v in o.items()} for o in orphans],
            "ghosts": [{k: v for k, v in g.items()} for g in ghosts],
            "by_category": by_category,
            "baseline_orphaned_ignores": BASELINE_ORPHANED_IGNORES,
            "baseline_ghost_rows": BASELINE_GHOST_ROWS,
            "strict": args.strict,
            "would_fail": (
                (len(orphans) > BASELINE_ORPHANED_IGNORES
                 or len(ghosts) > BASELINE_GHOST_ROWS
                 or len(orphans) > 0
                 or len(ghosts) > 0)
                and args.strict
            ),
        }
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print("=== Fluxion quarantine registry audit "
              "(Issue #3211 / #3393 / #3443) ===")
        print(f"Tests directory: {TESTS_DIR.relative_to(REPO_ROOT)}/")
        print(f"Registry:        {QUARANTINE_MD.relative_to(REPO_ROOT)}")
        print()
        print(
            f"Scanned {len(ignores)} #[ignore] attribute(s) across "
            f"{len({i['file'] for i in ignores})} file(s)."
        )
        print(f"Registered:      {len(registry)} (in QUARANTINE.md)")
        print(f"Orphan:          {len(orphans)} (in code, not in registry)")
        print(f"Ghost:           {len(ghosts)} (in registry, not in code)")
        print()
        print(
            f"Orphan ratchet baseline (BASELINE_ORPHANED_IGNORES): "
            f"{BASELINE_ORPHANED_IGNORES}"
        )
        print(
            f"Ghost  ratchet baseline (BASELINE_GHOST_ROWS): "
            f"{BASELINE_GHOST_ROWS}"
        )
        print()
        print("By category:")
        for cat in sorted(by_category):
            print(f"  {cat:18s}: {by_category[cat]}")
        print()
        if orphans:
            print(f"ORPHAN ENTRIES ({len(orphans)} — first 10):")
            for o in orphans[:10]:
                print(
                    f"  - {o['file']}:{o['line']} "
                    f"`{o['function']}` "
                    f"[{classify_ignore(o)}]"
                )
            print()
            print(
                "Fix: add a row to tests/QUARANTINE.md for each orphan, "
                "or relocate the test to tests/diagnostics/ if it's a "
                "diagnostic-only file. The triage should map the orphan's "
                "#[ignore] reason to one of the QUARANTINE.md categories "
                "above; structural orphans must cite a blocking-issue and "
                "LIMIT-* tag."
            )
        if ghosts:
            print(
                f"GHOST ENTRIES ({len(ghosts)} — registry rows with no "
                f"matching #[ignore]):"
            )
            for g in ghosts[:10]:
                print(f"  - {g['file']} :: {g['function']}")
            print()
            print(
                "Fix: the test may have been un-ignored without updating "
                "the registry, or the file path / function name has "
                "drifted. Update the registry row or remove it if the "
                "test is now live."
            )
        if not orphans and not ghosts:
            print("Registry in sync with code: no orphans, no ghosts.")

    if not args.strict:
        return 0

    # Downward-only ratchet (Issue #3443): reject growth above the
    # documented baseline, mirroring the BASELINE_KNOWN_ORPHANS /
    # BASELINE_WIRED_BUT_DEAD pattern from scripts/check_orphan_modules.py.
    if len(orphans) > BASELINE_ORPHANED_IGNORES:
        new_keys = sorted(orphan_keys - _BASELINE_ORPHANED_IGNORES_SET)
        print(
            "ORPHAN COUNT GREW ABOVE BASELINE (CI FAILURE — Issue #3443 "
            "downward-only ratchet):"
        )
        print(
            f"  len(orphans) = {len(orphans)} > "
            f"BASELINE_ORPHANED_IGNORES = {BASELINE_ORPHANED_IGNORES}"
        )
        if new_keys:
            print("  Newly added orphan entries (not in the freeze snapshot):")
            for file_cell, fn_cell in new_keys:
                print(f"    {file_cell} :: {fn_cell}")
        print(
            "\n"
            "Adding a new orphan to tests/QUARANTINE.md is allowed only\n"
            "when the new orphan is tracked by a documented issue AND the\n"
            "baseline constant is raised with a justifying comment naming\n"
            "the tracking issue. Otherwise the orphan count will silently\n"
            "grow back. Companion cleanup PRs that *resolve* an existing\n"
            "orphan (e.g. by closing the underlying tracking issue and\n"
            "un-ignoring the test) are expected to LOWER\n"
            "BASELINE_ORPHANED_IGNORES by one.\n"
        )
        return 1
    if len(ghosts) > BASELINE_GHOST_ROWS:
        new_keys = sorted(ghost_keys - _BASELINE_GHOST_ROWS_SET)
        print(
            "GHOST COUNT GREW ABOVE BASELINE (CI FAILURE — Issue #3443 "
            "downward-only ratchet):"
        )
        print(
            f"  len(ghosts) = {len(ghosts)} > "
            f"BASELINE_GHOST_ROWS = {BASELINE_GHOST_ROWS}"
        )
        if new_keys:
            print("  Newly added ghost entries (not in the freeze snapshot):")
            for file_cell, fn_cell in new_keys:
                print(f"    {file_cell} :: {fn_cell}")
        print(
            "\n"
            "Adding a new ghost to tests/QUARANTINE.md is allowed only\n"
            "when the stale registry row is tracked by a documented issue\n"
            "AND the baseline constant is raised with a justifying comment\n"
            "naming the tracking issue. Companion cleanup PRs that *fix* a\n"
            "ghost (e.g. by re-pointing the registry row to the actual\n"
            "function name or removing a duplicate) are expected to LOWER\n"
            "BASELINE_GHOST_ROWS by one.\n"
        )
        return 1
    if orphans or ghosts:
        # Either count is non-zero but within baseline — informational.
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
