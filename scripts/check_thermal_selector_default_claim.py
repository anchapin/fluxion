#!/usr/bin/env python3
"""
Anti-drift gate for ThermalSelector default claim (Issue #4160).

Scans root markdown docs for the pre-ADR-0017 false claim that
``ThermalSelector::default()`` resolves unconditionally to
``ZoneSolverKind::Gauge`` (the default in both feature states), or that
the ``gauge-solver`` cargo feature separately gates whether a fall-through
to legacy 5R1C/9R4C exists.

Per ADR-0017 (Issue #3978), the default selector is cfg-dependent and
explicit in every build:
  - ``ZoneSolverKind::Gauge`` with ``--features gauge-solver``
  - ``ZoneSolverKind::FiveROneC`` in default builds (HighMass specs
    still auto-promote to ``NineRFourC``)
  - An explicit ``Gauge`` selector in a default build panics loudly at
    construction
  - The silent fall-through was removed

The gate exits 0 when no stale claim is found, 1 when drift is detected.

Root docs scanned: README.md, AGENTS.md, RULES.md, CONTRIBUTING.md.

Usage::

    python3 scripts/check_thermal_selector_default_claim.py

Exit codes:
  0 — no stale claim detected
  1 — stale claim detected (root doc asserts the unconditional-Gauge
      default that ADR-0017 removed)
  2 — script error
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Root markdown docs to scan (names; resolved against REPO_ROOT at call time
# so hermetic tests can monkeypatch REPO_ROOT).
_ROOT_DOC_NAMES: tuple[str, ...] = (
    "README.md",
    "AGENTS.md",
    "RULES.md",
    "CONTRIBUTING.md",
)

# ---------------------------------------------------------------------------
# Patterns that assert the pre-ADR-0017 false claim.
# All must NOT appear in an affirmative (non-negated) sentence.
# ---------------------------------------------------------------------------

# Pattern 1: unconditional Gauge default — "Gauge is the unconditional default"
# or "the default is Gauge" (when not negated).
# Must NOT match negated forms like "is NOT the default", "is gone", etc.
# KEY: "ZoneSolverKind::Gauge" with "cfg-dependent" or "FiveROneC" context
# is the CORRECT posture; we only flag claims that say Gauge IS the
# unconditional/separate default without the cfg qualifier.
_UNCONDITIONAL_GAUGE_DEFAULT_RE = re.compile(
    # Pattern A: "Gauge [selector] is the unconditional default" or
    # "Gauge is the [zone solver] default" — "is the" must appear
    # within 15 chars of "Gauge" (tight; prevents matching across the
    # "panics loudly at construction. The unconditional default flip"
    # sentence that contains the CORRECT panic-description of Gauge).
    r"(?:ZoneSolverKind::)?Gauge\b.{0,15}?(?:unconditional\s+default|"
    r"is\s+the\s+(?:zone\s+solver\s+)?default\b)"
    r"|"
    # Pattern B: "the default is ZoneSolverKind::Gauge" (default BEFORE Gauge)
    r"(?:the\s+)?default\s+is\s+(?:ZoneSolverKind::)?Gauge\b"
    r"|"
    # Pattern C: "unconditional default" within ~15 chars BEFORE Gauge
    r"unconditional\s+default\b.{0,15}?(?:ZoneSolverKind::)?Gauge\b",
    re.IGNORECASE,
)

# Pattern 2: "default() resolves to Gauge in BOTH feature states"
_BOTH_FEATURE_STATES_RE = re.compile(
    r"(?:ThermalSelector::)?default\(\)[^\n]*?resolves?\s+to[^\n]*?Gauge\b[^\n]*?"
    r"(?:both|either|all)\s+(?:feature\s+)?states?\b"
    r"|"
    r"(?:both|either|all)\s+(?:feature\s+)?states?[^\n]*?"
    r"(?:ThermalSelector::)?default\(\)[^\n]*?resolves?\s+to[^\n]*?Gauge\b",
    re.IGNORECASE,
)

# Pattern 3: "gauge-solver separately gates ... falls through to legacy 5R1C/9R4C"
# i.e. claims the feature controls a fall-through rather than being the
# sole determinant of the default selector.
_SEPARATE_GATE_FALLTHROUGH_RE = re.compile(
    r"gauge-solver[^\n]*?separately\s+gates?[^\n]*?"
    r"(?:falls?\s+through|legacy|5R1C|9R4C)"
    r"|"
    r"(?:falls?\s+through|legacy|5R1C|9R4C)[^\n]*?"
    r"gauge-solver[^\n]*?separately\s+gates?",
    re.IGNORECASE,
)

# Negation patterns — if a match is negated, it is NOT a stale claim.
# Examples: "is NOT", "is gone", "was removed", "no longer", "silent fall-through was removed"
_NEGATION_RE = re.compile(
    r"\b(?:not|n't|never|no\s+longer|gone|removed|retired|eliminated|"
    r"was\s+removed|has\s+been\s+removed|is\s+removed|was\s+replaced|"
    r"is\s+superseded|supersedes|superseded\s+by|is\s+gone|is\s+no\s+longer|"
    r"no\s+silent|silent\s+was\s+removed|was\s+deleted)\b",
    re.IGNORECASE,
)

# Note: we intentionally do NOT skip lines that mention ADR-0017 — the
# negation detection below handles legitimate corrections. The ADR-0017
# marker approach was too broad (a false claim can appear after an
# "ADR-0017: ..." correct-posture intro).
# ADR-0017 CORRECT_MARKER_RE is removed per Issue #4160.


def _is_negated(match_start: int, line: str) -> bool:
    """Return True if the given match is inside a negation span in the line."""
    # Extract up to match_start chars and check if the last non-comment
    # word before the match is a negation.
    prefix = line[:match_start]
    # Strip trailing whitespace and take the last word.
    words = prefix.split()
    if not words:
        return False
    # Check if any negation word appears in the last few words.
    recent = " ".join(words[-5:])
    return bool(_NEGATION_RE.search(recent))


def _check_file(path: Path) -> list[tuple[int, str]]:
    """Scan a markdown file for stale thermal-selector default claims.

    Returns a list of (line_no, failure_message) for each stale claim found.
    """
    failures: list[tuple[int, str]] = []

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"    WARN: could not read {path}: {exc}")
        return failures

    in_fence = False
    for lineno, line in enumerate(content.splitlines(), start=1):
        # Fenced code blocks are excluded: claims quoted inside fences
        # (e.g. quoting the historical false claim for documentation) are
        # not re-drift. Toggle state on fence markers, skip fenced lines.
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        # Pattern 1: unconditional Gauge default claim.
        for m in _UNCONDITIONAL_GAUGE_DEFAULT_RE.finditer(line):
            if not _is_negated(m.start(), line):
                failures.append((
                    lineno,
                    "line asserts unconditional Gauge default "
                    "(pre-ADR-0017 false claim); per Issue #3978 "
                    "the default is cfg-dependent and explicit "
                    "(Gauge with gauge-solver feature, FiveROneC without; "
                    "no silent fall-through)",
                ))

        # Pattern 2: "default() resolves to Gauge in both feature states".
        for m in _BOTH_FEATURE_STATES_RE.finditer(line):
            if not _is_negated(m.start(), line):
                failures.append((
                    lineno,
                    "line asserts default() resolves to Gauge in both "
                    "feature states (pre-ADR-0017 false claim); per "
                    "Issue #3978 the default is cfg-dependent: "
                    "ZoneSolverKind::Gauge with gauge-solver feature, "
                    "ZoneSolverKind::FiveROneC in default builds",
                ))

        # Pattern 3: separate gate controlling fall-through.
        for m in _SEPARATE_GATE_FALLTHROUGH_RE.finditer(line):
            if not _is_negated(m.start(), line):
                failures.append((
                    lineno,
                    "line asserts gauge-solver feature separately gates a "
                    "fall-through to legacy 5R1C/9R4C (pre-ADR-0017 false "
                    "claim); per Issue #3978 the default selector is "
                    "cfg-dependent and explicit; no silent fall-through "
                    "exists",
                ))

    return failures


def main() -> int:
    print(
        "ThermalSelector default claim anti-drift gate "
        f"(Issue #4160 / ADR-0017, repo: {REPO_ROOT})"
    )
    print()

    all_failures: list[tuple[str, int, str]] = []

    for name in _ROOT_DOC_NAMES:
        doc = REPO_ROOT / name
        if not doc.exists():
            print(f"  SKIP: {doc.relative_to(REPO_ROOT)} not found")
            continue
        print(f"  Scanning {doc.relative_to(REPO_ROOT)} ...")
        failures = _check_file(doc)
        if failures:
            for lineno, msg in failures:
                all_failures.append((str(doc.relative_to(REPO_ROOT)), lineno, msg))
            print(f"    DRIFT: {len(failures)} stale claim(s)")
        else:
            print("    OK")

    print()
    if all_failures:
        print("THERMAL-SELECTOR DEFAULT CLAIM DRIFT DETECTED:")
        for rel_path, lineno, msg in all_failures:
            print(f"  {rel_path}:{lineno}: {msg}")
        print()
        print("Root docs assert the pre-ADR-0017 unconditional-Gauge default")
        print("that ADR-0017 (Issue #3978) removed. The correct posture is:")
        print("  - cfg-dependent explicit default: ZoneSolverKind::Gauge with")
        print("    --features gauge-solver, ZoneSolverKind::FiveROneC in default builds")
        print("  - No silent fall-through; explicit Gauge in default builds panics")
        print("  - Authority for the unconditional flip is #3986 teacher validation suite")
        print()
        print("Remediation: update the cited lines to match ADR-0017 posture.")
        return 1

    print("No stale ThermalSelector default claim detected.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
