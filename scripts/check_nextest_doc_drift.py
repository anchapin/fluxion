#!/usr/bin/env python3
"""Fail when a documented nextest command does not appear in any workflow.

Issue #4177: the per-PR test gate ("Nextest Subset (GH)" in
``ci-gates.yml``) runs a CURATED package/test list, while AGENTS.md and
the docs presented ``cargo nextest run --workspace ...`` as the CI
command actually running in ``rust-tests.yml::test`` — which had not
been true since the phase-gating rollout. Documentation that names a
test command readers will assume CI runs must match a command CI
actually runs, or the doc is fiction.

The gate extracts ``cargo nextest run ...`` commands documented in
``AGENTS.md`` (the command block) and ``docs/**/*.md`` (inline
backtick-quoted commands) and requires each one to appear
(whitespace-normalized, ``\\``-continuations joined) in at least one
``.github/workflows/*.yml``.

Scope decisions (documented here so they are reviewable, not silent):

* ``docs/archive/**`` is excluded — archived snapshots are historical
  by definition.
* Fenced code blocks and indented shell transcripts are NOT scanned —
  they are usually session logs (e.g. the ADR-0014 audit's
  ``2>&1 | tee`` loops), not commands presented for reuse. Only inline
  backtick-quoted commands and the AGENTS.md command block are.
* Template placeholders (``${{ ... }}``, ``--features X``) and the bare
  ``cargo nextest run`` are skipped — they cannot appear verbatim.
* ``HISTORICAL_ALLOWLIST`` covers superseded commands that remain in
  prose for the record (ADR-0014 audit/adoption text, rollout-doc
  prose). Each entry carries its justification; adding a NEW
  documented command that is neither in a workflow nor justified here
  fails the gate.

Usage::

    python3 scripts/check_nextest_doc_drift.py
    python3 scripts/check_nextest_doc_drift.py --self-test

Exit codes:

    0 -- every documented command appears in a workflow (or is allowed)
    1 -- one or more documented commands are absent from all workflows
    2 -- script error
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
AGENTS_MD = REPO_ROOT / "AGENTS.md"
DOCS_DIR = REPO_ROOT / "docs"

# `cargo nextest run` + flags, inline backtick-quoted: `cargo nextest run ...`.
_BACKTICK_CMD_RE = re.compile(r"`(cargo nextest run[^`]*)`")

# Commands that are documented but intentionally NOT in any workflow.
# Each maps to the reason it is exempt; keep this list short and
# justified — it is the escape hatch, not the default.
HISTORICAL_ALLOWLIST = {
    # ADR-0014 audit command: the rollout's safety case ran this 5x on a
    # cold target. Historical record, not a current CI claim.
    "cargo nextest run --lib --test-threads=2":
        "ADR-0014 audit command (historical)",
    # ADR-0014 adoption prose ("Adopt ... as the test runner").
    # Superseded by the --no-fail-fast / --exclude fluxion-tauri
    # canonical form (Issue #4177).
    "cargo nextest run --workspace --all-targets":
        "ADR-0014 adoption prose (historical)",
    # Rollout-doc prose describing the wholesale switch. Superseded by
    # the canonical form above.
    "cargo nextest run --workspace --all-targets --test-threads=2":
        "nextest-rollout.md prose (historical)",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _documented_commands() -> list[tuple[str, str]]:
    """Return ``(command, location)`` for each documented nextest command."""
    found: list[tuple[str, str]] = []

    def add(cmd: str, location: str) -> None:
        cmd = _normalize(cmd)
        if cmd and not any(c == cmd for c, _ in found):
            found.append((cmd, location))

    # AGENTS.md command block: bare `cargo nextest run ...` lines;
    # the trailing `#` comment is not part of the command.
    if AGENTS_MD.is_file():
        for idx, line in enumerate(
            AGENTS_MD.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if stripped.startswith("cargo nextest run"):
                add(stripped.split("#")[0], f"AGENTS.md:{idx}")

    # Inline backtick-quoted commands in docs (excluding the archive).
    if DOCS_DIR.is_dir():
        for path in sorted(DOCS_DIR.rglob("*.md")):
            if "archive" in path.parts:
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            for idx, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                for m in _BACKTICK_CMD_RE.finditer(line):
                    add(m.group(1), f"{rel}:{idx}")

    return found


def _workflow_corpus() -> list[str]:
    """Normalized text of every workflow (``\\``-continuations joined)."""
    corpus = []
    for path in sorted(WORKFLOWS_DIR.glob("*.yml")):
        text = path.read_text(encoding="utf-8").replace("\\\n", " ")
        corpus.append(_normalize(text))
    return corpus


def _is_placeholder(cmd: str) -> bool:
    return (
        "${{" in cmd
        or "--features X" in cmd
        or cmd == "cargo nextest run"
    )


def check_commands(
    documented: list[tuple[str, str]], corpus: list[str]
) -> list[str]:
    """Return drift findings for documented commands absent from workflows."""
    findings = []
    for cmd, location in documented:
        if _is_placeholder(cmd):
            continue
        if cmd in HISTORICAL_ALLOWLIST:
            continue
        if not any(cmd in workflow_text for workflow_text in corpus):
            findings.append(
                f"{location}: documented `cargo nextest run` command "
                f"appears in no workflow: `{cmd}`"
            )
    return findings


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else sys.argv[1:]
    if "--self-test" in argv:
        return _self_test()

    if not WORKFLOWS_DIR.is_dir():
        print(f"ERROR: {WORKFLOWS_DIR} not found", file=sys.stderr)
        return 2

    documented = _documented_commands()
    corpus = _workflow_corpus()
    findings = check_commands(documented, corpus)

    if findings:
        print(f"FAIL: {len(findings)} documented nextest command(s) "
              f"absent from all workflows:")
        for f in findings:
            print(f"  {f}")
        return 1
    print(f"OK: {len(documented)} documented nextest command(s) checked "
          f"against {len(corpus)} workflow(s); no drift.")
    return 0


def _self_test() -> int:
    """Hermetic regression test for the detector."""
    import tempfile

    failures = 0

    def run_case(name: str, docs: dict[str, str], workflows: dict[str, str],
                 expect_findings: bool) -> None:
        nonlocal failures
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".github" / "workflows").mkdir(parents=True)
            (root / "docs").mkdir(parents=True)
            for fname, text in docs.items():
                p = root / fname
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text, encoding="utf-8")
            for fname, text in workflows.items():
                (root / ".github" / "workflows" / fname).write_text(
                    text, encoding="utf-8")
            global REPO_ROOT, WORKFLOWS_DIR, AGENTS_MD, DOCS_DIR  # noqa: PLW0603
            old = (REPO_ROOT, WORKFLOWS_DIR, AGENTS_MD, DOCS_DIR)
            REPO_ROOT = root
            WORKFLOWS_DIR = root / ".github" / "workflows"
            AGENTS_MD = root / "AGENTS.md"
            DOCS_DIR = root / "docs"
            try:
                findings = check_commands(
                    _documented_commands(), _workflow_corpus())
            finally:
                REPO_ROOT, WORKFLOWS_DIR, AGENTS_MD, DOCS_DIR = old
            if bool(findings) != expect_findings:
                print(f"FAIL: case {name!r}: expected "
                      f"findings={expect_findings}, got {findings}",
                      file=sys.stderr)
                failures += 1

    wf_with_cmd = (
        "jobs:\n  t:\n    steps:\n"
        "      - run: cargo nextest run --workspace --exclude foo "
        "--all-targets --test-threads=2 --no-fail-fast\n"
    )
    # 1. Documented command present in workflow -> clean.
    run_case(
        "documented command in workflow",
        {"AGENTS.md": "cargo nextest run --workspace --exclude foo "
                      "--all-targets --test-threads=2 --no-fail-fast "
                      "# canonical CI command\n"},
        {"w.yml": wf_with_cmd},
        expect_findings=False,
    )
    # 2. Documented command absent from workflow -> finding.
    run_case(
        "documented command missing",
        {"docs/g.md": "Run `cargo nextest run --workspace --no-such-flag`.\n"},
        {"w.yml": wf_with_cmd},
        expect_findings=True,
    )
    # 3. Placeholder commands are skipped.
    run_case(
        "placeholder skipped",
        {"docs/g.md": "Run `cargo nextest run --workspace --features X`.\n"},
        {"w.yml": "jobs: {}\n"},
        expect_findings=False,
    )
    # 4. Allowlisted historical command is skipped.
    run_case(
        "allowlist skipped",
        {"docs/g.md": "Audit `cargo nextest run --lib --test-threads=2` x5.\n"},
        {"w.yml": "jobs: {}\n"},
        expect_findings=False,
    )

    if failures:
        return 1
    print("PASS: self-test (4 cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
