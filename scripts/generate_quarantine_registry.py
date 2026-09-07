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

# `#[ignore]` and `#[ignore = "reason"]` (with optional reason string).
# Tolerant of whitespace and trailing comments.
_IGNORE_RE = re.compile(
    r"^\s*#\s*\[\s*ignore\s*(?:=\s*\"([^\"]*)\")?\s*\]\s*(?://.*)?$",
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
    ``reason``. The ``function`` is the nearest ``fn test_xxx``
    declaration either before *or* after the ``#[ignore]`` attribute
    (best-effort: ``unknown`` if the attribute appears outside a test
    function, e.g. on a module). Rust's attribute syntax allows either
    ordering, so the script must support both.
    """
    results: list[dict] = []
    for path in sorted(tests_dir.rglob("*.rs")):
        text = path.read_text(encoding="utf-8")
        for match in _IGNORE_RE.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_text = text[line_start : text.find("\n", match.start())]
            if _is_comment_only(line_text):
                continue
            # Search the surrounding ±2 KB for the nearest ``fn name(``,
            # preferring the closest match in either direction. This is
            # intentionally lenient so multi-line attribute blocks (e.g.
            # ``#[test]\n#[ignore = "..."]\nfn test_xxx() {}``) match.
            window_start = max(0, match.start() - 2000)
            window_end = min(len(text), match.end() + 2000)
            window = text[window_start:window_end]
            fn_matches = list(_TEST_FN_RE.finditer(window))
            fn_match = None
            if fn_matches:
                ignore_offset = match.start() - window_start
                fn_match = min(
                    fn_matches,
                    key=lambda m: abs(m.start() - ignore_offset),
                )
            function = fn_match.group(1) if fn_match else "unknown"
            results.append(
                {
                    "file": str(path.relative_to(REPO_ROOT)),
                    "line": line_no,
                    "function": function,
                    "reason": match.group(1) or "",
                }
            )
    return results


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
            "Quarantine registry audit (Issue #3211 / #3393). Default mode "
            "is informational; --strict fails on orphan #[ignore] entries."
        )
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail (exit 1) on any orphan #[ignore] not in QUARANTINE.md.",
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

    if args.json:
        out = {
            "tests_dir": str(TESTS_DIR.relative_to(REPO_ROOT)),
            "registry": str(QUARANTINE_MD.relative_to(REPO_ROOT)),
            "total_ignores": len(ignores),
            "registered": len(registry),
            "orphans": [{k: v for k, v in o.items()} for o in orphans],
            "ghosts": [{k: v for k, v in g.items()} for g in ghosts],
            "by_category": by_category,
            "strict": args.strict,
            "would_fail": len(orphans) > 0 and args.strict,
        }
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print("=== Fluxion quarantine registry audit (Issue #3211 / #3393) ===")
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

    if args.strict and orphans:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
