#!/usr/bin/env python3
"""
Criterion 2 cohort-drift detector — Issue #3744.

The ``scripts/check_beta_soak_gate.py::criterion_2_failures()`` function
is a hard-coded mirror of
``docs/agents/beta-soak-criterion-2-tracker.md`` — by construction it
cannot change as a matter of fresh test breakage. The β-soak nightly's
Criterion 2 step has been permanently red since 2026-09-01 (the §LIMIT-21
cohort is the documented blocker), so the workflow's job status carries
no marginal signal: a gauge-arm regression introduced by today's merge
would be indistinguishable from the known LIMIT-21 failures without a
human diffing the run log against the tracker doc.

This script closes that gap. It reads the Criterion 2 ``cargo test``
log, parses the *actual* failing-test set, and compares it against the
canonical §LIMIT-21 cohort listed by
``check_beta_soak_gate.criterion_2_failures()``. The script:

1. Extracts every failing test name from the log (``test <name> ...
   FAILED`` and the per-binary ``FAILED`` block).
2. Normalises the names to bare ``snake_case`` identifiers (strips the
   ``tests::all_tests::<module>::`` prefix so the canonical names match
   one-to-one).
3. Loads the canonical §LIMIT-21 cohort names from
   ``scripts/check_beta_soak_gate.py::criterion_2_failures()``
   (single source of truth — also the source the tracker doc and the
   ``beta-soak-criterion-2-failures.json`` artifact are mirrored from).
4. Computes three sets:

   * ``added`` — failures in the live run that are NOT in the
     canonical cohort (a fresh regression signal — the §LIMIT-21
     air-trajectory gap grew).
   * ``removed`` — failures in the canonical cohort that did NOT
     fire in the live run (either §LIMIT-21 is narrowing via the
     structural fix program, or the test was renamed/refactored and
     the tracker doc is now stale).
   * ``unchanged`` — failures common to both sets.

5. Emits a v1-schema JSON artifact
   (``beta-soak-criterion-2-drift.json``) with the diff and a
   boolean ``drift_detected`` flag.

6. Optionally emits a GitHub-issue markdown comment for posting on
   the tracking issue (``#3286``), distinct from the nightly result
   comment so a maintainer can spot drift at a glance.

The drift fires in **either direction** (per Issue #3744 acceptance):
a freshly-added failure (regression) and a removed canonical failure
(cohort is shrinking — potentially a recovery signal OR a stale
tracker doc) both raise the alarm.

Usage::

    # CI / nightly workflow (drift-detection step).
    python3 scripts/check_criterion_2_drift.py --log criterion-2-output.log \\
        --artifact beta-soak-criterion-2-drift.json

    # Render the GitHub-comment body for posting on #3286.
    python3 scripts/check_criterion_2_drift.py --log criterion-2-output.log \\
        --comment > drift-comment.md

    # Force the canonical cohort to a specific file (used by tests).
    python3 scripts/check_criterion_2_drift.py --log criterion-2-output.log \\
        --canonical path/to/beta-soak-criterion-2-failures.json

    # Emit the diff JSON to stdout instead of a file.
    python3 scripts/check_criterion_2_drift.py --log criterion-2-output.log --json

Exit codes:

  0 — no drift (live failing set ≡ canonical cohort).
  1 — drift detected (live ≠ canonical).
  2 — script error (missing log, malformed log, missing canonical,
      unparseable CLI arguments).

Cross-references:

* Issue #3744 — this drift detector.
* Issue #3359 — parent tracker issue / canonical list owner.
* Issue #3286 — β-soak gate tracking issue (the drift comment posts here).
* Issue #3297 — §LIMIT-21 cohort owner.
* ``scripts/check_beta_soak_gate.py`` — single source of truth for
  the canonical cohort (``criterion_2_failures()``).
* ``docs/agents/beta-soak-criterion-2-tracker.md`` — human-readable
  mirror of the canonical cohort.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BETA_SOAK_GATE = REPO_ROOT / "scripts" / "check_beta_soak_gate.py"
DEFAULT_ARTIFACT_NAME = "beta-soak-criterion-2-drift.json"

SCHEMA_VERSION = "1"
DRIFT_MARKER = "<!-- criterion-2-cohort-drift-marker -->"


# ---------------------------------------------------------------------------
# Live-failure parsing
# ---------------------------------------------------------------------------
#
# The Criterion 2 step is `cargo test --locked --features gauge-solver
# --test all_tests zone_balance_eplus_isolation::`. Cargo emits two
# failure surfaces we accept:
#
# 1. The per-test progress lines:
#      test zone_balance_eplus_isolation::test_xxx ... FAILED
#      test tests::all_tests::zone_balance_eplus_isolation::test_yyy ... FAILED
#
# 2. The per-binary `failures:` block (verbose-style):
#      failures:
#          zone_balance_eplus_isolation::test_xxx
#          zone_balance_eplus_isolation::test_yyy
#
# Both surfaces name the test in ``<module_path>::<test_fn>`` form. The
# consolidated runner (Issue #3764) uses the bare ``<module>::<fn>``
# form; older single-binary runners emit ``tests::all_tests::<module>::<fn>``
# — we strip every leading module path component and keep only the
# trailing snake_case identifier so the canonical cohort (which is
# bare ``snake_case``) matches one-to-one.

_PROGRESS_LINE_RE = re.compile(
    r"^test\s+(?P<path>[A-Za-z0-9_:{}\s]+?::)?(?P<name>[a-z][a-z0-9_]*)\s+\.\.\.\s+(?P<verdict>FAILED|ignored|ok)\b"
)
_FAILURES_BLOCK_LINE_RE = re.compile(
    r"^\s+(?P<path>[A-Za-z0-9_:{}\s]+?::)?(?P<name>[a-z][a-z0-9_]*)\s*$"
)
_FAILURES_HEADER_RE = re.compile(r"^failures:\s*$", re.IGNORECASE)
_TEST_RESULT_RE = re.compile(
    r"^test result:\s*(?P<verdict>FAILED|ok|ignored)\.\s*"
    r"(?P<passed>\d+)\s+passed;\s*"
    r"(?P<failed>\d+)\s+failed"
    r"(?:;\s*(?P<ignored>\d+)\s+ignored)?"
)

_BARE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass
class ParsedCriterion2Log:
    """Result of parsing a Criterion 2 ``cargo test`` log.

    The ``failures`` set is the *unique* set of failing test names
    normalised to bare ``snake_case`` identifiers; ``test_results`` is
    the list of per-binary ``test result:`` summary lines (one per
    test binary cargo executed).
    """

    failures: set[str] = field(default_factory=set)
    test_results: list[dict[str, Any]] = field(default_factory=list)
    parse_warnings: list[str] = field(default_factory=list)


def _normalise_test_name(raw: str) -> Optional[str]:
    """Strip the leading module path off a raw test name.

    Accepts forms like ``zone_balance_eplus_isolation::test_xxx``,
    ``tests::all_tests::zone_balance_eplus_isolation::test_yyy``, or
    bare ``test_xxx``. Returns the bare snake_case test function name,
    or ``None`` when the trailing segment is not a valid Rust test
    identifier (this rejects accidental non-test matches).
    """
    if not raw:
        return None
    # Take the last segment after the final `::`.
    name = raw.split("::")[-1].strip()
    if not name:
        return None
    # Cargo test names are snake_case; reject anything that doesn't look
    # like a Rust identifier. This filters out text accidentally matched
    # by the parser (e.g. inside a panic message).
    if not _BARE_NAME_RE.match(name):
        return None
    return name


def parse_criterion_2_log(text: str) -> ParsedCriterion2Log:
    """Parse a Criterion 2 ``cargo test`` log and return failing tests.

    The parser is intentionally tolerant: it accepts both the
    per-test progress lines (``test ... FAILED``) and the per-binary
    ``failures:`` block, deduplicates across both, and surfaces any
    parser-irregularities as ``parse_warnings`` rather than raising —
    the nightly may legitimately include unrelated stdout (e.g. prints
    from the simulation) that we want to ignore without aborting.
    """
    out = ParsedCriterion2Log()
    in_failures_block = False
    # ``failures:`` blocks in cargo output span a header line, a
    # blank line, the indented test list, and a closing blank line —
    # so the only unambiguous close is the next ``test result:``
    # summary line (or end-of-file). Blank lines INSIDE the block
    # (the pre-list and post-list delimiters) are skipped but do
    # not close the block.

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        # Track the failures: block delimiter.
        if _FAILURES_HEADER_RE.match(line):
            in_failures_block = True
            continue

        # Per-binary summary line — closes any active failures block
        # and falls through to the summary-line handler below.
        if line.startswith("test result:"):
            if in_failures_block:
                in_failures_block = False
            # fall through

        # Inside a ``failures:`` block, every indented line lists a
        # test; blank lines do NOT close the block (the canonical
        # cargo layout is ``failures:`` / blank / indented list /
        # blank / ``test result:``).
        if in_failures_block:
            if not line.strip():
                # Skip the blank line(s) that delimit the test list
                # but do NOT close the block — the closing token is
                # the ``test result:`` summary line above.
                continue
            failures_line = _FAILURES_BLOCK_LINE_RE.match(line)
            if failures_line:
                name = _normalise_test_name(
                    (failures_line.group("path") or "") + failures_line.group("name")
                )
                if name is None:
                    out.parse_warnings.append(
                        f"unparseable test name in failures block: {line!r}"
                    )
                    continue
                out.failures.add(name)
                continue
            # An unrecognised non-indented line closes the block.
            in_failures_block = False
            # fall through to other handlers (e.g. test result:)

        # Per-binary summary line.
        result_match = _TEST_RESULT_RE.match(line)
        if result_match:
            out.test_results.append(
                {
                    "verdict": result_match.group("verdict"),
                    "passed": int(result_match.group("passed")),
                    "failed": int(result_match.group("failed")),
                    "ignored": int(result_match.group("ignored") or 0),
                }
            )
            continue

        # Per-test progress line.
        progress = _PROGRESS_LINE_RE.match(line)
        if progress:
            verdict = progress.group("verdict")
            name = _normalise_test_name(
                (progress.group("path") or "") + progress.group("name")
            )
            if name is None:
                out.parse_warnings.append(
                    f"unparseable test name in progress line: {line!r}"
                )
                continue
            if verdict == "FAILED":
                out.failures.add(name)
            continue

    return out


def parse_criterion_2_log_file(path: Path) -> ParsedCriterion2Log:
    """Read a Criterion 2 ``cargo test`` log from ``path`` and parse it.

    Raises ``FileNotFoundError`` / ``OSError`` on IO errors; surfaces
    other parser problems as ``parse_warnings`` inside the result.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_criterion_2_log(text)


# ---------------------------------------------------------------------------
# Canonical cohort
# ---------------------------------------------------------------------------


def load_canonical_cohort(
    canonical_json: Optional[Path] = None,
) -> tuple[set[str], dict[str, Any]]:
    """Load the canonical §LIMIT-21 cohort names.

    Returns ``(names, payload)`` where ``names`` is the set of bare
    ``snake_case`` test names from the canonical
    ``check_beta_soak_gate.criterion_2_failures()`` payload, and
    ``payload`` is the full canonical JSON for downstream consumers
    (the artifact embeds it for traceability).

    The default source is the script's own subprocess invocation of
    ``scripts/check_beta_soak_gate.py --criterion-2-failures``. A
    pre-computed JSON artifact can be supplied via ``canonical_json``
    for hermetic testing without spawning Python.
    """
    if canonical_json is None:
        import subprocess

        proc = subprocess.run(
            [sys.executable, str(DEFAULT_BETA_SOAK_GATE), "--criterion-2-failures"],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(REPO_ROOT),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"failed to load canonical cohort from "
                f"{DEFAULT_BETA_SOAK_GATE} (exit {proc.returncode}): "
                f"{proc.stderr.strip()}"
            )
        payload = json.loads(proc.stdout)
    else:
        payload = json.loads(canonical_json.read_text(encoding="utf-8"))

    names: set[str] = set()
    for failure in payload.get("failures", []):
        name = failure.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names, payload


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


@dataclass
class CohortDrift:
    """The diff between the live Criterion 2 failures and the canonical
    §LIMIT-21 cohort.
    """

    live: set[str]
    canonical: set[str]
    added: set[str]
    removed: set[str]
    unchanged: set[str]

    @property
    def drift_detected(self) -> bool:
        return bool(self.added or self.removed)


def compute_drift(
    live: set[str], canonical: set[str]
) -> CohortDrift:
    """Compute the symmetric diff between ``live`` and ``canonical``."""
    return CohortDrift(
        live=set(live),
        canonical=set(canonical),
        added=set(live - canonical),
        removed=set(canonical - live),
        unchanged=set(live & canonical),
    )


# ---------------------------------------------------------------------------
# Artifact + comment
# ---------------------------------------------------------------------------


def render_drift_artifact(
    drift: CohortDrift,
    canonical_payload: dict[str, Any],
    parsed: ParsedCriterion2Log,
    *,
    run_id: Optional[str] = None,
    source_log: Optional[str] = None,
) -> dict[str, Any]:
    """Build the v1-schema JSON artifact payload."""
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tracker_doc": canonical_payload.get(
            "tracker_doc", "docs/agents/beta-soak-criterion-2-tracker.md"
        ),
        "limit_ref": canonical_payload.get("limit_ref", "§LIMIT-21"),
        "limit_owner_issue": canonical_payload.get("limit_owner_issue", "#3297"),
        "drift_detected": drift.drift_detected,
        "live_failing_count": len(drift.live),
        "canonical_count": len(drift.canonical),
        "added": sorted(drift.added),
        "removed": sorted(drift.removed),
        "unchanged": sorted(drift.unchanged),
        "parse_warnings": list(parsed.parse_warnings),
        "test_results": list(parsed.test_results),
    }
    if run_id is not None:
        artifact["run_id"] = run_id
    if source_log is not None:
        artifact["source_log"] = source_log
    return artifact


def render_drift_comment(
    drift: CohortDrift,
    artifact: dict[str, Any],
    run_url: Optional[str] = None,
) -> str:
    """Render a GitHub-issue markdown comment for posting on #3286.

    The comment is prefixed with a hidden HTML marker
    (``<!-- criterion-2-cohort-drift-marker -->``) so duplicate-post
    prevention can key off the marker if a future follow-up wires
    that in. The body intentionally distinguishes the two drift
    directions:

    * ``added`` (new failure not in canonical) — possible regression
      or §LIMIT-21 cohort expansion; needs triage.
    * ``removed`` (canonical failure no longer firing) — possibly the
      structural-fix program (#1465 / #1462 / #3059) closing in,
      OR the canonical list is stale (test renamed/refactored) and
      needs a docs update.
    """
    lines: list[str] = []
    lines.append(DRIFT_MARKER)
    lines.append("")

    if not drift.drift_detected:
        lines.append("✅ **Criterion 2 cohort drift check — clean**")
        lines.append("")
        lines.append(
            f"Live failing set ({len(drift.live)}) ≡ canonical §LIMIT-21 "
            f"cohort ({len(drift.canonical)}). No drift in either direction."
        )
        if run_url:
            lines.append("")
            lines.append(f"Run: {run_url}")
        return "\n".join(lines) + "\n"

    added = sorted(drift.added)
    removed = sorted(drift.removed)

    direction_label: str
    direction_emoji: str
    if added and not removed:
        direction_label = "Regression suspected — §LIMIT-21 cohort grew"
        direction_emoji = "🚨"
    elif removed and not added:
        direction_label = (
            "Canonical cohort shrank — tracker doc may be stale OR "
            "§LIMIT-21 is closing"
        )
        direction_emoji = "📉"
    else:
        direction_label = "Mixed drift — both added and removed"
        direction_emoji = "⚠️"

    lines.append(f"{direction_emoji} **Criterion 2 cohort drift detected — {direction_label}**")
    lines.append("")
    lines.append(
        f"Live failing set ({len(drift.live)}) ≠ canonical §LIMIT-21 "
        f"cohort ({len(drift.canonical)})."
    )
    if added:
        lines.append("")
        lines.append(f"**Added** ({len(added)}) — live failures NOT in canonical §LIMIT-21:")
        for name in added:
            lines.append(f"- `{name}` (new failure; needs triage)")
    if removed:
        lines.append("")
        lines.append(
            f"**Removed** ({len(removed)}) — canonical failures that did NOT "
            f"fire tonight:"
        )
        for name in removed:
            lines.append(
                f"- `{name}` (either §LIMIT-21 is narrowing or the tracker "
                f"doc is stale — investigate)"
            )
    lines.append("")
    lines.append(
        "See `beta-soak-criterion-2-drift` artifact for the structured "
        "v1-schema diff (Issue #3744)."
    )
    if run_url:
        lines.append("")
        lines.append(f"Run: {run_url}")

    lines.append("")
    lines.append(
        "Cross-references: Issue #3744 (this drift detector) · "
        "Issue #3359 (canonical cohort tracker) · "
        "Issue #3297 (§LIMIT-21 cohort owner) · "
        "Issue #3286 (β-soak gate)."
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Criterion 2 cohort-drift detector (Issue #3744). "
            "Parses the nightly Criterion 2 `cargo test` log and diffs "
            "the live failing-set against the canonical §LIMIT-21 cohort "
            "from scripts/check_beta_soak_gate.py::criterion_2_failures(). "
            "Exits 1 whenever the live set differs from the canonical set "
            "in either direction."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help=(
            "Path to the Criterion 2 `cargo test` log (stdout/stderr "
            "captured by the nightly workflow)."
        ),
    )
    parser.add_argument(
        "--canonical",
        type=Path,
        default=None,
        help=(
            "Optional path to a pre-computed canonical Criterion 2 "
            "failures JSON (e.g. `beta-soak-criterion-2-failures.json`). "
            "Defaults to invoking scripts/check_beta_soak_gate.py "
            "--criterion-2-failures."
        ),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help=(
            "Write the v1-schema drift JSON to PATH (default: "
            f"`{DEFAULT_ARTIFACT_NAME}` at the repository root when "
            "--json is not set; ignored when --json is set)."
        ),
    )
    parser.add_argument(
        "--comment",
        action="store_true",
        help=(
            "Emit a GitHub-issue markdown comment body for posting on "
            "the β-soak tracking issue (#3286), distinct from the "
            "nightly result comment. Use `--json` for the structured "
            "diff artifact."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit the v1-schema drift JSON to stdout instead of "
            "writing the artifact file. Mutually compatible with "
            "--comment only when used alone (the comment body is "
            "always emitted on stdout when --comment is set)."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Optional GitHub Actions run id (databaseId) to embed in "
            "the artifact and comment — used by the nightly workflow "
            "for cross-referencing the issue comment with the run log."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    log_path: Path = args.log
    if not log_path.exists():
        print(
            f"ERROR: Criterion 2 log not found: {log_path}",
            file=sys.stderr,
        )
        return 2

    try:
        parsed = parse_criterion_2_log_file(log_path)
    except (OSError, UnicodeDecodeError) as exc:
        print(
            f"ERROR: failed to read Criterion 2 log {log_path}: {exc}",
            file=sys.stderr,
        )
        return 2

    try:
        canonical_names, canonical_payload = load_canonical_cohort(
            canonical_json=args.canonical
        )
    except (RuntimeError, json.JSONDecodeError, OSError) as exc:
        print(
            f"ERROR: failed to load canonical §LIMIT-21 cohort: {exc}",
            file=sys.stderr,
        )
        return 2

    drift = compute_drift(parsed.failures, canonical_names)

    artifact = render_drift_artifact(
        drift,
        canonical_payload,
        parsed,
        run_id=args.run_id,
        source_log=str(log_path),
    )

    if args.json and not args.comment:
        print(json.dumps(artifact, indent=2))

    if args.comment:
        run_url: Optional[str] = None
        if args.run_id:
            run_url = (
                f"https://github.com/{_github_repo_or_unknown()}/"
                f"actions/runs/{args.run_id}"
            )
        print(render_drift_comment(drift, artifact, run_url=run_url), end="")

    if not args.json and not args.comment:
        artifact_path = args.artifact or (REPO_ROOT / DEFAULT_ARTIFACT_NAME)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(artifact, indent=2) + "\n", encoding="utf-8"
        )
        # Print a one-line human summary so the workflow log makes the
        # outcome obvious without parsing the artifact.
        verdict = "DRIFT" if drift.drift_detected else "clean"
        print(
            f"Criterion 2 cohort drift: {verdict} "
            f"(added={len(drift.added)}, removed={len(drift.removed)}, "
            f"unchanged={len(drift.unchanged)}). "
            f"Artifact: {artifact_path}"
        )

    return 1 if drift.drift_detected else 0


def _github_repo_or_unknown() -> str:
    """Best-effort `owner/repo` from ``GITHUB_REPOSITORY`` env var.

    Returns ``"unknown/unknown"`` when the env var is unset (e.g. in
    local CI). The comment URL is best-effort and the operator can
    reconstruct it from the embedded ``run_id``.
    """
    import os

    return os.environ.get("GITHUB_REPOSITORY", "unknown/unknown")


if __name__ == "__main__":
    sys.exit(main())