#!/usr/bin/env python3
"""
CI guard: root docs must not claim the thermal selector's default is
unconditionally `ZoneSolverKind::Gauge` (Issue #4160).

Background: ADR-0017 (Issue #3978) superseded the Phase A8 two-solver
interim posture of Issue #3291. Under ADR-0017 the default is
**cfg-dependent and explicit in every build, with no silent cfg
fall-through anywhere**:

  * with ``--features gauge-solver``, ``ThermalSelector::default()`` is
    ``ZoneSolverKind::Gauge`` and the dispatcher's gauge arm is
    unconditional;
  * in the default build (feature off) it is the explicit legacy
    ``ZoneSolverKind::FiveROneC``, HighMass specs still auto-promoting to
    9R4C via ``from_spec_with_selector``;
  * an explicit ``Gauge`` selector in a default build panics loudly at
    construction, naming the feature flag.

``README.md`` documented the *deleted* posture instead: it claimed the
feature flag "separately gates whether the dispatcher's gauge arm runs
unconditionally or falls through to legacy 5R1C/9R4C" and that
``ThermalSelector::default()`` "resolves to ``ZoneSolverKind::Gauge`` in
both feature states". Both halves are false. README is the first document
a new contributor or agent reads, so a lie there scopes downstream physics
and benchmark work against a production path that does not exist -- and it
directly contradicted ``AGENTS.md``, the other root control document.

This check fails when a root doc asserts the unconditional-Gauge default,
so the claim cannot re-drift. It is deliberately narrow: it targets the
specific false assertions, not the phrase "unconditional" in general,
because ADR-0017's own wording legitimately uses "unconditional" to
describe the gauge arm *under the feature*. Each finding must name the
offending text so the author can rewrite rather than merely delete.

Scope: ``README.md``, ``AGENTS.md``, ``ARCHITECTURE.md``,
``CODEBASE_MAP.md``, ``CONTRIBUTING.md``. ADR files under ``docs/adr/``
are excluded -- they are dated historical records of what was decided
*then*, and rewriting an ADR body destroys its provenance.

Wired into the `Scripts Test Suite` job of
`.github/workflows/scripts-tests.yml` ("Reject unconditional-Gauge default
claims", Issue #4160). The matcher's invariants are additionally covered by
`scripts/ci/test_check_thermal_selector_doc_claims.py` against hermetic
`tmp_path` fixtures.

Exit codes:
    0 — no doc claims the unconditional-Gauge default.
    1 — one or more root docs assert the false posture.
    2 — script error (e.g. repo root missing).

Usage:
    python3 scripts/check_thermal_selector_doc_claims.py
    python3 scripts/check_thermal_selector_doc_claims.py --root /path/to/repo
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Root control documents, in the order findings are reported.
ROOT_DOCS = (
    "README.md",
    "AGENTS.md",
    "ARCHITECTURE.md",
    "CODEBASE_MAP.md",
    "CONTRIBUTING.md",
)

# Each entry is (name, pattern, qualifier). ``qualifier`` is an optional
# regex: when it matches the text *preceding* the claim on the same line,
# the claim is scoped and therefore legitimate, so it is not reported.
#
# The patterns are written to fire on the *assertion*, not on the words.
# "unconditional" alone must not trip the check: ADR-0017's own canonical
# wording says the gauge arm is "unconditional" under the feature, and
# that is correct. So an unqualified claim is required -- a line that
# scopes the claim to `--features gauge-solver` is exempt via `qualifier`.
_VIOLATIONS: tuple[tuple[str, re.Pattern[str], re.Pattern[str] | None], ...] = (
    (
        "default claimed to resolve to Gauge in both feature states",
        # "resolves to ZoneSolverKind::Gauge in both feature states"
        re.compile(
            r"ThermalSelector::default\(\)"
            r"[^\n]{0,160}?\bGauge\b"
            r"[^\n]{0,80}?\bboth\s+feature\s+states\b",
            re.IGNORECASE,
        ),
        None,
    ),
    (
        "unconditional default asserted without the gauge-solver feature",
        # "the Gauge selector is the unconditional default" with no
        # cfg-dependent qualifier earlier in the line.
        re.compile(
            # `Gauge selector` must be adjacent: a bare `\bGauge\b` also
            # matches the `gauge` inside `gauge-solver`, which would make
            # the qualifier lookup search the wrong prefix.
            r"\bGauge\s+selector\b[^\n]{0,80}?"
            r"\bis\s+(?:now\s+)?the\s+unconditional\s+default\b",
            re.IGNORECASE,
        ),
        # A claim scoped to the feature build is correct, not a lie.
        re.compile(r"--features\s+gauge-solver", re.IGNORECASE),
    ),
    (
        "silent fall-through to legacy 5R1C/9R4C claimed as current behavior",
        re.compile(
            r"\b(?:falls?|falling)\s+through\s+to\s+legacy\s+"
            r"(?:5R1C/9R4C|5R1C|9R4C)",
            re.IGNORECASE,
        ),
        None,
    ),
    (
        "feature flag claimed to gate the gauge arm vs a legacy fall-through",
        re.compile(
            r"gauge-solver[^\n]{0,120}?"
            r"(?:separately\s+)?gates\s+whether[^\n]{0,120}?"
            r"\bfalls?\s+through\b",
            re.IGNORECASE,
        ),
        None,
    ),
)

# Substrings that make an otherwise-matching line an *excuse* rather than a
# violation: the line is describing the superseded posture in the past
# tense, or explicitly negating it. Matched case-insensitively against the
# whole line.
_EXCUSES = (
    "superseded",
    "supersedes",
    "no silent",
    "removed",
    "not the unconditional",
    "was the unconditional",
    "has been removed",
    "interim posture",
    "rather than falling through",
    "deliberately deleted",
)


def _line_is_excused(line: str) -> bool:
    low = line.lower()
    return any(excuse in low for excuse in _EXCUSES)


def check_doc(path: Path, rel: str | None = None) -> list[str]:
    """Return claim findings for one document.

    Empty list means the document is clean. ``rel`` overrides the path
    used in finding text (tests pass a synthetic name).
    """
    rel = rel or path.name
    findings: list[str] = []
    text = path.read_text(encoding="utf-8")

    for lineno, line in enumerate(text.splitlines(), start=1):
        if _line_is_excused(line):
            continue
        for name, pattern, qualifier in _VIOLATIONS:
            m = pattern.search(line)
            if not m:
                continue
            if qualifier is not None and qualifier.search(line[: m.start()]):
                # The claim is scoped to a feature/qualifier earlier on the
                # same line, so it states current, correct behavior.
                continue
            snippet = m.group(0).strip()
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."
            findings.append(
                f"{rel}:{lineno}: {name}. Found: {snippet!r}. ADR-0017 "
                "(Issue #3978) makes the default cfg-dependent and explicit: "
                "`Gauge` with `--features gauge-solver`, explicit legacy "
                "`FiveROneC` without, and an explicit `Gauge` selector in a "
                "default build panics rather than falling through."
            )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root to scan (default: inferred from this script).",
    )
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"ERROR: {args.root} not found", file=sys.stderr)
        return 2

    findings: list[str] = []
    present = 0
    for name in ROOT_DOCS:
        path = args.root / name
        if not path.is_file():
            continue
        present += 1
        findings.extend(check_doc(path, rel=name))

    if not present:
        print(f"ERROR: none of {ROOT_DOCS} found under {args.root}", file=sys.stderr)
        return 2

    if findings:
        print("Unconditional-Gauge default claim in a root doc:", file=sys.stderr)
        for f in findings:
            print(f"  - {f}", file=sys.stderr)
        print(
            f"\n{len(findings)} finding(s) across {present} root doc(s). "
            "ADR-0017 (Issue #3978) is the authority; see "
            "`docs/adr/0017-equation-based-dae-teacher-architecture.md`.",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: {present} root doc(s) describe the cfg-dependent ADR-0017 "
        "thermal-selector default."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
