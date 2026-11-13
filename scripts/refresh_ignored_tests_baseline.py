#!/usr/bin/env python3
"""Refresh (and verify) the nightly ``#[ignore]`` census baseline.

Issue #4793 — the nightly gauge workflow's Criterion 7 ``#[ignore]`` census
compared the live census against a baseline that had been cut before the
``tests/all_tests/`` consolidation, so the delta signal was dead (every
consolidated entry looked like a "removal").

Mechanism (issue #4793):

* The census itself is **unchanged** — it is the workflow's exact grep:

      grep -rn "#\\[ignore" tests/ src/ --include="*.rs" 2>/dev/null | sort

  This script shells out to that very command (same arguments, same
  relative paths, same cwd) so the baseline can never drift away from the
  census definition.

* The **baseline is normalised**: the ``path:line-number:`` prefix of each
  census hit is reduced to ``path:`` and the attribute text is stripped of
  surrounding whitespace. Line numbers are the one part of a census hit
  that changes for free whenever an unrelated edit shifts a file, so they
  must not participate in the comparison: without this, a pure line-number
  drift registers as a removal (false positive) in the nightly's delta.

  Normalised baseline line::

      tests/all_tests/zone_balance_eplus_isolation.rs:#[ignore = "..."]

  The file is sorted in byte order (``sort`` / C collation semantics) and
  contains **data lines only** — no header, no comments — because the
  nightly's comparison is a plain ``comm -23`` over the file.

* Because the baseline stores normalised lines, any comparison must
  normalise the live census the same way before comparing. That is what
  ``--check`` does here, and it is what the shell equivalent below does:

      grep -rn "#\\[ignore" tests/ src/ --include="*.rs" 2>/dev/null \\
        | sed -E 's/^([^:]+):[0-9]+:[[:space:]]*/\\1:/; s/[[:space:]]+$//' \\
        | sort

  (see ``--print-sed`` for the exact one-liner).

Workflow interaction (issue #4793):

  The nightly gauge workflow's Criterion 7 step is *advisory* telemetry
  (it gates nothing).  It greps the census and ``comm -23``s the result
  against this baseline file, so the live side of that comparison has to
  be normalised the same way the baseline is.  The workflow file is out of
  scope for this issue, so the drift-robust comparison lives here
  (``--check``); a follow-up workflow change that normalises the live
  census before the ``comm`` would restore the nightly's own delta
  signal.  Until then, read the nightly's Criterion 7 ``removed`` count
  together with this script's ``--check`` output.

Usage::

    python3 scripts/refresh_ignored_tests_baseline.py            # re-cut the baseline
    python3 scripts/refresh_ignored_tests_baseline.py --check    # compare only
    python3 scripts/refresh_ignored_tests_baseline.py --raw      # raw census (workflow format)
    python3 scripts/refresh_ignored_tests_baseline.py --root /tmp/scratch --check

Exit codes:
    0 — baseline refreshed / comparison clean (no removals).
    1 — ``--check`` found removed census entries (a quarantine was lifted
        without refreshing the baseline).
    2 — script error (grep failed, baseline missing, ...).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_REL = Path("tests/reference_data/ignored_tests_baseline.txt")

# The workflow's exact census command (nightly-ashrae-140-gauge.yml,
# "Criterion 7 — #[ignore] census vs baseline").  The pattern is passed
# verbatim (BRE: `#\[ignore` matches a literal `#[ignore`).
CENSUS_GREP = [
    "grep",
    "-rn",
    r"#\[ignore",
    "tests/",
    "src/",
    "--include=*.rs",
]

# Equivalent sed pipeline for the live census → normalised lines.  Kept in
# sync with `normalize()` below; used by `--print-sed` so the shell one-liner
# used as evidence cannot drift away from the Python implementation.
SED_NORMALISE = (
    r"sed -E 's/^([^:]+):[0-9]+:[[:space:]]*/\1:/; s/[[:space:]]+$//'"
)


def run_census(root: Path) -> list[str]:
    """Run the workflow's exact census grep and return its sorted output."""
    proc = subprocess.run(
        CENSUS_GREP,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError(
            f"census grep failed with exit code {proc.returncode}: "
            f"{' '.join(CENSUS_GREP)}"
        )
    text = proc.stdout.decode("utf-8", errors="replace")
    return [line for line in text.splitlines() if line]


def normalize(raw_line: str) -> str:
    """Reduce a census hit ``path:line:text`` to ``path:text``.

    The line number is dropped (it drifts for free) and the attribute text
    is stripped of surrounding whitespace.
    """
    path, _, rest = raw_line.partition(":")
    _, _, text = rest.partition(":")
    return f"{path}:{text.strip()}"


def baseline_lines(root: Path) -> list[str]:
    """Return the normalised, byte-sorted census lines for ``root``."""
    lines = [normalize(raw) for raw in run_census(root)]
    return sorted(lines, key=lambda s: s.encode("utf-8"))


def render(lines: list[str]) -> str:
    return "".join(f"{line}\n" for line in lines)


def compare(baseline: list[str], live: list[str]) -> tuple[Counter, Counter]:
    """Return (removed, added) multisets between baseline and live census."""
    base_counts = Counter(baseline)
    live_counts = Counter(live)
    removed = base_counts - live_counts
    added = live_counts - base_counts
    return removed, added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Refresh / verify the nightly #[ignore] census baseline."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="repository root to census (default: this checkout)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the live census against the committed baseline "
        "(do not write); exit 1 when entries were removed",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="print the raw census output (workflow format) and exit",
    )
    parser.add_argument(
        "--print-sed",
        action="store_true",
        help="print the shell one-liner that normalises the live census",
    )
    parser.add_argument("--quiet", action="store_true", help="only print findings")
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()

    if args.print_sed:
        print(SED_NORMALISE)
        return 0

    if args.raw:
        for line in run_census(root):
            print(line)
        return 0

    live = baseline_lines(root)
    baseline_path = root / BASELINE_REL

    if args.check:
        if not baseline_path.exists():
            print(f"FAIL: baseline missing: {BASELINE_REL}", file=sys.stderr)
            return 2
        committed = [
            line for line in baseline_path.read_text(encoding="utf-8").splitlines() if line
        ]
        removed, added = compare(committed, live)
        if not args.quiet:
            print(f"census (normalised): now={len(live)} baseline={len(committed)}")
        if removed:
            print(f"removed={sum(removed.values())}")
            for line, n in sorted(removed.items()):
                print(f"  - {line}" + (f" (x{n})" if n > 1 else ""))
        else:
            print("removed=0")
        if added and not args.quiet:
            print(f"added={sum(added.values())} (new quarantines — refresh to record)")
            for line, n in sorted(added.items()):
                print(f"  + {line}" + (f" (x{n})" if n > 1 else ""))
        return 1 if removed else 0

    # Refresh: write the freshly cut baseline.
    payload = render(live)
    if baseline_path.exists():
        current = baseline_path.read_text(encoding="utf-8")
        if current == payload:
            if not args.quiet:
                print(f"baseline already current: {BASELINE_REL} ({len(live)} entries)")
            return 0
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(payload, encoding="utf-8")
    if not args.quiet:
        print(f"wrote {BASELINE_REL}: {len(live)} normalised census entries")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
