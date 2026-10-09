#!/usr/bin/env python3
"""Quarantine registry synchroniser (Issue #3211, #3393).

Scans ``tests/**/*.rs``, the root crate's ``src/**/*.rs`` and every
workspace member's ``src/**/*.rs`` and ``tests/**/*.rs`` for
``#[ignore]`` attributes
(override the scan roots per-invocation with ``--scan-root``) and
cross-references them against the human-curated registry at
``tests/QUARANTINE.md``.

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
  Scan roots: tests/ src/ fluxion-fluid/src/ ...
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
    python3 scripts/generate_quarantine_registry.py --scan-root tests  # narrower scan
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


def _workspace_members() -> list[str]:
    """Parse ``[workspace] members`` from the root ``Cargo.toml``.

    Minimal regex parse (no toml dependency): captures the quoted
    entries of the ``members = [...]`` array. Used to derive the
    default per-crate ``src/`` scan roots (Issue #4178).
    """
    cargo = REPO_ROOT / "Cargo.toml"
    try:
        text = cargo.read_text(encoding="utf-8")
    except OSError:
        return []
    match = re.search(r"members\s*=\s*\[(.*?)\]", text, re.DOTALL)
    if not match:
        return []
    return re.findall(r'"([^"]+)"', match.group(1))


def _default_scan_roots() -> list[Path]:
    """Default ``#[ignore]`` scan roots (Issue #4178).

    ``tests/``, the root crate's ``src/``, and every workspace
    member's ``src/`` and ``tests/`` that exist on disk. Override
    per-invocation with ``--scan-root`` (repeatable, relative to the
    repo root).
    """
    roots = [REPO_ROOT / "tests", REPO_ROOT / "src"]
    for member in _workspace_members():
        for sub in ("src", "tests"):
            candidate = REPO_ROOT / member / sub
            if candidate.is_dir() and candidate not in roots:
                roots.append(candidate)
    return roots


# ---------------------------------------------------------------------------
# Downward-only ratchet for the quarantine registry (Issue #3443,
# hardened by Issue #4179).
#
# As of Issue #4179 the ratchet is a KEY-MEMBERSHIP check, not a bare
# integer comparison: ``--strict`` fails when ANY orphan/ghost key is
# absent from the freeze snapshot below, regardless of the total
# count. The integer baselines are retained as documentation of the
# last accepted totals; the freeze SETS are authoritative.
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
#   - 0 → 10 (Issue #3443 reconciliation): commits bfc9c55 (#3572 —
#     strict-energy-gate baseline extension for cases 800/810/920/950/
#     960/970) and the #3585 Wave-5/8 + #3551/#3552 diagnostic-placeholder
#     commits added `#[ignore]`d tests under `tests/**` without registry
#     rows, silently tripping this ratchet on `develop`. Back-fill adds
#     the 10 rows to `tests/QUARANTINE.md` ("Strict-Energy-Gate &
#     Structural Diagnostics" section) and raises this baseline to 10
#     with the matching freeze-set entries below.
# ---------------------------------------------------------------------------
#   - 0 → 9 (Issue #3443 reconciliation, revised by PR improve/quarantine-burndown):
#     commits bfc9c55 (#3572 — strict-energy-gate baseline extension for
#     cases 800/810/920/950/960/970) and the #3585 Wave-5/8 +
#     #3551/#3552 diagnostic-placeholder commits added `#[ignore]`d tests
#     under `tests/**` without registry rows, silently tripping this
#     ratchet on `develop`. Back-fill lands the 9 rows (the 10th ignored
#     test, `test_case_970_validator_accepts_canonical_midpoints`, was
#     verified live and un-ignored by the burndown PR) in
#     `tests/QUARANTINE.md` ("Strict-energy-gate observation cohort" +
#     Diagnostic sections) and raises this baseline to 9 with the
#     matching freeze-set entries below.
# ---------------------------------------------------------------------------
#   - 11 → 0 (Issue #4179): the integer comparison had 11 slots of
#     unused headroom while the live audit reported 0 orphans, so up
#     to 11 brand-new unregistered `#[ignore]` attributes merged green.
#     The freeze snapshot is now authoritative (key-membership ratchet)
#     and, because every known orphan is registered, the snapshot is
#     EMPTY. Any new orphan fails `--strict`; registering it (with a
#     real QUARANTINE.md row) is the only way back to green. Lowering
#     the ratchet is the explicitly authorised direction.
# ---------------------------------------------------------------------------
BASELINE_ORPHANED_IGNORES = 0
BASELINE_GHOST_ROWS = 0

# Freeze snapshot of the orphan allowlist (Issue #3443 ratchet,
# hardened by Issue #4179).
#
# EMPTY as of Issue #4179: every `#[ignore]` known at the time is
# registered in `tests/QUARANTINE.md`, so there is nothing to
# grandfather. Under the key-membership ratchet, a PR that adds an
# `#[ignore]` without a registry row fails `--strict` with the new
# key named; the fix is to add the row (the freeze set must NOT be
# extended to launder it).
_BASELINE_ORPHANED_IGNORES_SET: frozenset[tuple[str, str]] = frozenset()

# Freeze snapshot of the ghost rows (Issue #3443 ratchet).
#
# EMPTY as of Issue #4179. (The 2026-09-11 entries for the 3 Phase-A8
# `src/` scratch_pool rows were "permanent ghosts" only because the
# scanner never left `tests/`; Issue #4178 widened the scan to `src/`
# and every workspace member's `src/`, so those rows now match real
# `#[ignore]` attributes and the grandfathering is obsolete.)
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

# Frozen registry schema (Issue #4179). QUARANTINE.md presents itself
# as a "Machine-readable registry"; these are the columns every row
# must carry, and the only legitimate Category values. `scan_registry`
# resolves column positions from the table header, so column ORDER is
# not significant — but every required column must be present.
_REQUIRED_COLUMNS: tuple[str, ...] = (
    "Test File",
    "Test Name",
    "Category",
    "Blocking Issue",
    "Owner",
    "Un-Ignore Criteria",
    "Status",
)

# The legitimate Category values (Issue #4179). These are the
# `classify_ignore` buckets as realised in the QUARANTINE.md
# `## Category:` sections — including `manual-baseline`, which the
# issue text omitted but the registry's "Manual Baseline Regeneration"
# section and the classifier both use.
_VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        "calibration",
        "ci-broken",
        "diagnostic",
        "hardware",
        "manual-baseline",
        "other",
        "performance",
        "structural",
    }
)


def _split_row(line: str) -> list[str]:
    """Split a markdown table row into raw cells."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _clean_cell(cell: str) -> str:
    """Strip whitespace and one layer of backtick wrapping."""
    cell = cell.strip()
    if len(cell) >= 2 and cell.startswith("`") and cell.endswith("`"):
        cell = cell[1:-1].strip()
    return cell


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
        (r["file"], r["function"]) for r in raw_results if not r["conditional"]
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

    Searches a ±2 KB window around the attribute. Rust attributes
    precede the item they decorate, so a ``fn`` AFTER the attribute is
    preferred: forward matches use their raw offset distance, backward
    matches are penalised 3x. Without the penalty, dense back-to-back
    single-statement tests tie on line distance and the previous test's
    ``fn`` (a few closing-brace characters behind) wins over the
    attribute's own ``fn`` a couple of lines ahead — mis-attributing
    the quarantine key. Returns ``"unknown"`` if no test function is
    found within the window.
    """
    window_start = max(0, attr_start - 2000)
    window_end = min(len(text), attr_end + 2000)
    window = text[window_start:window_end]
    fn_matches = list(_TEST_FN_RE.finditer(window))
    if not fn_matches:
        return "unknown"
    ignore_offset = attr_start - window_start

    def _distance(m: re.Match) -> int:
        d = m.start() - ignore_offset
        return d if d >= 0 else -d * 3

    fn_match = min(fn_matches, key=_distance)
    return fn_match.group(1)


def scan_registry(quarantine_md: Path) -> list[dict]:
    """Scan ``quarantine_md`` for registry rows.

    Returns a list of dicts with keys ``file``, ``function``,
    ``category``, ``issue``, ``owner``, ``unignore_criteria``,
    ``status`` and ``line`` (1-based source line, for diagnostics).
    Column positions are resolved from the table header, so column
    order is not significant. Rows whose file cell does not end in
    ``.rs`` (e.g. the ``## Summary`` table, category sub-header rows)
    are skipped.
    """
    if not quarantine_md.exists():
        return []
    lines = quarantine_md.read_text(encoding="utf-8").splitlines()
    col_index: dict[str, int] = {}
    rows: list[dict] = []
    for lineno, line in enumerate(lines, start=1):
        if not line.strip().startswith("|"):
            continue
        cells = [_clean_cell(c) for c in _split_row(line)]
        if not col_index:
            # First table row mentioning "Test File" is the header;
            # any earlier `|` table (there is none today) is ignored.
            lowered = [c.lower() for c in cells]
            if "test file" not in lowered:
                continue
            for required in _REQUIRED_COLUMNS:
                rl = required.lower()
                if rl in lowered:
                    col_index[required] = lowered.index(rl)
            continue  # the header row itself is never a data row
        if all((not c) or set(c) <= set("-:") for c in cells):
            continue  # markdown separator row

        def cell(name: str) -> str:
            idx = col_index.get(name, -1)
            return cells[idx] if 0 <= idx < len(cells) else ""

        file_cell = cell("Test File")
        if not file_cell.endswith(".rs"):
            continue  # category sub-header / summary-table row
        rows.append(
            {
                "file": file_cell,
                "function": cell("Test Name"),
                "category": cell("Category"),
                "issue": cell("Blocking Issue"),
                "owner": cell("Owner"),
                "unignore_criteria": cell("Un-Ignore Criteria"),
                "status": cell("Status"),
                "line": lineno,
            }
        )
    return rows


def validate_registry_schema(rows: list[dict]) -> list[str]:
    """Schema pass over parsed registry rows (Issue #4179).

    Returns a list of human-readable violation strings; empty means
    the registry is schema-clean. Fails on any empty Category /
    Blocking Issue / Owner / Un-Ignore Criteria / Status cell, and on
    any Category outside ``_VALID_CATEGORIES``. ``--strict`` turns any
    violation into exit 1.
    """
    violations: list[str] = []
    for row in rows:
        where = f"{row['file']} :: {row['function']} (line {row['line']})"
        for key, label in (
            ("category", "Category"),
            ("issue", "Blocking Issue"),
            ("owner", "Owner"),
            ("unignore_criteria", "Un-Ignore Criteria"),
            ("status", "Status"),
        ):
            if not row.get(key, "").strip():
                violations.append(f"{where}: empty {label}")
        cat = row.get("category", "").strip().lower()
        if cat and cat not in _VALID_CATEGORIES:
            violations.append(
                f"{where}: unknown Category {row['category']!r} "
                f"(expected one of {sorted(_VALID_CATEGORIES)})"
            )
    return violations


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
            "Quarantine registry audit (Issue #3211 / #3393 / #3443 / "
            "#4178 / #4179). Default mode is informational; --strict "
            "fails on any orphan #[ignore] entry absent from the freeze "
            "snapshot, on any ghost registry row absent from the freeze "
            "snapshot, or on any QUARANTINE.md row-schema violation."
        )
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail (exit 1) on any orphan #[ignore] absent from the "
            "QUARANTINE.md registry (key-membership ratchet, Issue "
            "#3443/#4179), on any ghost registry row, or on any "
            "registry row-schema violation (Issue #4179)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output for CI consumption.",
    )
    parser.add_argument(
        "--scan-root",
        action="append",
        default=None,
        metavar="DIR",
        help=(
            "Replacement #[ignore] scan root, relative to the repo root "
            "(repeatable). Defaults to tests/, src/ and every workspace "
            "member's src/ and tests/ (Issue #4178)."
        ),
    )
    args = parser.parse_args()

    if args.scan_root:
        scan_roots = [
            (REPO_ROOT / r) if not Path(r).is_absolute() else Path(r)
            for r in args.scan_root
        ]
    else:
        scan_roots = _default_scan_roots()
    ignores: list[dict] = []
    for scan_root in scan_roots:
        ignores.extend(scan_ignores(scan_root))
    registry = scan_registry(QUARANTINE_MD)
    schema_violations = validate_registry_schema(registry)
    orphans, ghosts = audit(ignores, registry)

    by_category: dict[str, int] = {}
    for ignore in ignores:
        cat = classify_ignore(ignore)
        by_category[cat] = by_category.get(cat, 0) + 1

    orphan_keys: set[tuple[str, str]] = {(o["file"], o["function"]) for o in orphans}
    ghost_keys: set[tuple[str, str]] = {(g["file"], g["function"]) for g in ghosts}

    # Key-membership ratchet (Issue #3443, hardened by Issue #4179):
    # any orphan/ghost key ABSENT from the freeze snapshot is new,
    # regardless of the total count. The integer baselines above are
    # documentation; these sets are authoritative.
    new_orphan_keys = sorted(orphan_keys - _BASELINE_ORPHANED_IGNORES_SET)
    new_ghost_keys = sorted(ghost_keys - _BASELINE_GHOST_ROWS_SET)

    if args.json:
        out = {
            "scan_roots": [str(r.relative_to(REPO_ROOT)) for r in scan_roots],
            "registry": str(QUARANTINE_MD.relative_to(REPO_ROOT)),
            "total_ignores": len(ignores),
            "registered": len(registry),
            "orphans": [{k: v for k, v in o.items()} for o in orphans],
            "ghosts": [{k: v for k, v in g.items()} for g in ghosts],
            "by_category": by_category,
            "schema_violations": schema_violations,
            "baseline_orphaned_ignores": BASELINE_ORPHANED_IGNORES,
            "baseline_ghost_rows": BASELINE_GHOST_ROWS,
            "strict": args.strict,
            "would_fail": (
                (
                    bool(new_orphan_keys)
                    or bool(new_ghost_keys)
                    or bool(schema_violations)
                )
                and args.strict
            ),
        }
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print("=== Fluxion quarantine registry audit (Issue #3211 / #3393 / #3443) ===")
        print(
            "Scan roots: "
            + ", ".join(str(r.relative_to(REPO_ROOT)) + "/" for r in scan_roots)
        )
        print(f"Registry:        {QUARANTINE_MD.relative_to(REPO_ROOT)}")
        print()
        print(
            f"Scanned {len(ignores)} #[ignore] attribute(s) across "
            f"{len({i['file'] for i in ignores})} file(s)."
        )
        print(f"Registered:      {len(registry)} (in QUARANTINE.md)")
        print(f"Orphan:          {len(orphans)} (in code, not in registry)")
        print(f"Ghost:           {len(ghosts)} (in registry, not in code)")
        print(f"Schema violations: {len(schema_violations)}")
        print()
        print(
            f"Orphan ratchet baseline (BASELINE_ORPHANED_IGNORES): "
            f"{BASELINE_ORPHANED_IGNORES}"
        )
        print(f"Ghost  ratchet baseline (BASELINE_GHOST_ROWS): {BASELINE_GHOST_ROWS}")
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

    # Key-membership ratchet (Issue #3443, hardened by Issue #4179):
    # --strict fails when ANY orphan/ghost key is absent from the
    # freeze snapshot, regardless of the total count. Registering the
    # new key with a real QUARANTINE.md row is the only way back to
    # green; extending the freeze set to launder it is forbidden.
    if new_orphan_keys:
        print(
            "NEW ORPHAN #[ignore] ENTRIES NOT IN THE FREEZE SNAPSHOT "
            "(CI FAILURE — Issue #3443/#4179 key-membership ratchet):"
        )
        for file_cell, fn_cell in new_orphan_keys:
            print(f"  - {file_cell} :: {fn_cell}")
        print(
            "\n"
            "Fix: add a row to tests/QUARANTINE.md for each new orphan "
            "(Category, Blocking Issue, Owner, Un-Ignore Criteria, "
            "Status). Do NOT add the key to "
            "_BASELINE_ORPHANED_IGNORES_SET to silence this check.\n"
        )
        return 1
    if new_ghost_keys:
        print(
            "NEW GHOST REGISTRY ROWS NOT IN THE FREEZE SNAPSHOT "
            "(CI FAILURE — Issue #3443/#4179 key-membership ratchet):"
        )
        for file_cell, fn_cell in new_ghost_keys:
            print(f"  - {file_cell} :: {fn_cell}")
        print(
            "\n"
            "Fix: the test may have been un-ignored without updating "
            "the registry, or the file path / function name has "
            "drifted. Update or remove the registry row.\n"
        )
        return 1
    if schema_violations:
        print("REGISTRY SCHEMA VIOLATIONS (CI FAILURE — Issue #4179 row-schema gate):")
        for violation in schema_violations:
            print(f"  - {violation}")
        print(
            "\n"
            "Fix: every QUARANTINE.md row must carry non-empty "
            "Category / Blocking Issue / Owner / Un-Ignore Criteria / "
            "Status cells, and Category must be one of "
            f"{sorted(_VALID_CATEGORIES)}.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
