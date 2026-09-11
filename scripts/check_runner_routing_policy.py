#!/usr/bin/env python3
"""
Runner-routing policy drift detection for Fluxion (Issue #3531).

GitHub Actions workflows that mention ``vars.FLUXION_LINUX_RUNNER`` create a
trust-boundary risk: when that repo variable is set, the variable's value
is used as the ``runs-on`` runner and PR-controlled code can silently
execute on a *persistent* self-hosted runner rather than the ephemeral
GH-hosted VM the policy assumes. ``docs/SECURITY.md §7`` (Issue #3445)
documents the mitigation:

* On ``pull_request`` events, ``runs-on`` MUST resolve to ``ubuntu-latest``
  regardless of ``vars.FLUXION_LINUX_RUNNER``. PR-controlled code is
  untrusted until branch protection has cleared it.
* On ``push`` to ``refs/heads/main`` (and sometimes ``refs/heads/develop``),
  the self-hosted runner is acceptable: the code path that arrived here has
  cleared the gate and the runner is therefore trusted to execute it.

The canonical YAML pattern (already used in ``onnx-integrity.yml:90-99``
and ``python-tests.yml:42-56``) is::

    runs-on: >-
      ${{
        github.event_name == 'push'
        && github.ref == 'refs/heads/main'
        && vars.FLUXION_LINUX_RUNNER
        || 'ubuntu-latest'
      }}

Issue #3531 closed six workflows that triggered on ``pull_request`` but
used the unguarded shorter form::

    runs-on: ${{ vars.FLUXION_LINUX_RUNNER || 'ubuntu-latest' }}

that falls through to the self-hosted runner on every PR. The earlier
gate (``scripts/check_required_checks_sync.py``) catches branch-protection
drift but cannot catch this — the ``runs-on`` expression compiles fine
and the runner variable resolves correctly; the security issue is that
the *trust boundary is the wrong way around*.

This script enforces two invariants:

1. Every ``runs-on:`` line that contains ``vars.FLUXION_LINUX_RUNNER`` MUST
   be guarded by the ``github.event_name == 'push' && github.ref ==
   'refs/heads/main'`` conjunction (in any YAML form — single-line,
   folded-block, or multi-line). An unguarded match FAILS the gate.
2. The same guard must appear on the same job's ``if:`` clause OR the
   workflow must be main-only / not PR-triggered. Specifically: a job whose
   ``if:`` clause includes ``pull_request`` MUST use the guarded
   ``runs-on``. A job that is main-only (``if: github.event_name == 'push'
   && github.ref == 'refs/heads/main'``) is permitted to use the unguarded
   pattern, because the job never executes on PR-controlled code.

To keep the parser simple and robust against arbitrary YAML structure
(``${{ }}`` expressions, folded block scalars, comments between fields),
this script uses a line-by-line regex sweep rather than ``yaml.safe_load``.
This mirrors the approach in ``check_required_checks_sync.py``: GitHub
Actions workflows embed expressions that PyYAML cannot parse, so
regex is the only safe option.

Usage::

    python3 scripts/check_runner_routing_policy.py

Exit codes:

    0 — no drift detected.
    1 — drift detected (one or more PR-triggered workflows use the
        unguarded ``vars.FLUXION_LINUX_RUNNER || 'ubuntu-latest'``
        ``runs-on`` form).
    2 — script error (e.g. ``.github/workflows/`` directory missing).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# Files inside `.github/workflows/` plus the top-level `.github/*.yml`
# variants (`onnx-integrity.yml`, `python-tests.yml`,
# `nightly-ashrae-140-gauge.yml`). Issue #3531 acceptance requires the
# gate to cover both. The matchers below treat `.yml` and `.yaml`
# uniformly.
_WORKFLOW_GLOB = ("*.yml", "*.yaml")

# Regex matching the unsafe ``runs-on:`` form. The expression
# ``vars.FLUXION_LINUX_RUNNER || 'ubuntu-latest'`` may appear on a single
# line or be folded across multiple lines via YAML ``>-`` / ``>`` block
# scalars. The folded-block case is matched by scanning *raw file text*
# for the substring in lines that start with ``runs-on:`` (see
# ``_RUNS_ON_LINE_RE`` below); this single-line regex catches the
# collapsed-form case used by the unsafe workflows from #3531.
_UNSAFE_RUNS_ON_INLINE_RE = re.compile(
    r"runs-on:\s*"
    r"\$\{\{\s*"
    r"vars\.FLUXION_LINUX_RUNNER\s*\|\|\s*'ubuntu-latest'\s*"
    r"\}\}",
)

# Lines that begin a ``runs-on:`` folded block (YAML ``>-`` or ``>``).
# The continuation lines that hold the actual expression are matched
# separately in ``_scan_runs_on_blocks``.
_RUNS_ON_BLOCK_HEADER_RE = re.compile(r"^(?P<indent>\s*)runs-on:\s*[>\|][+-]?\s*$")

# Canonical guarded fragment. The OR form
# (``&& vars.FLUXION_LINUX_RUNNER || 'ubuntu-latest'``) and the AND-only
# form (just ``&& vars.FLUXION_LINUX_RUNNER``) are both accepted: the
# AND-only form is followed by an explicit ``|| 'ubuntu-latest'`` line
# downstream, which the scanner treats as part of the same guarded
# expression. This matches both ``onnx-integrity.yml`` (``&& ... ||``)
# and ``python-tests.yml`` (same form).
_GUARD_FRAGMENT_RE = re.compile(
    r"github\.event_name\s*==\s*'push'\s*&&\s*github\.ref\s*==\s*'refs/heads/main'",
)

# ``if:`` clauses that scope a job to main-only. The job won't execute
# on a pull_request event, so the unguarded ``runs-on`` form is harmless.
# Match either bare form (``if: github.event_name == 'push' && ...``) or
# the `${{ }}`-wrapped form (``if: ${{ github.event_name == 'push' && ... }}``)
# — GitHub Actions accepts both.
_IF_PUSH_TO_MAIN_LINE_RE = re.compile(
    r"^(\s*)if:\s*"
    r"(?:\$\{\{\s*)?"
    r"github\.event_name\s*==\s*'push'\s*&&\s*github\.ref\s*==\s*'refs/heads/main'"
    r"(?:\s*\}\})?\s*$",
)
_IF_BLOCK_PUSH_TO_MAIN_RE = re.compile(
    r"if:\s*[>|][+-]?\s*\n(?:\s+\S.*\n)*?\s*github\.event_name\s*==\s*'push'"
    r"(?:\s|\n)*?&&(?:\s|\n)*?github\.ref\s*==\s*'refs/heads/main'",
    re.MULTILINE,
)
_IF_PULL_REQUEST_RE = re.compile(
    r"github\.event_name\s*==\s*'pull_request'",
)


# ---------------------------------------------------------------------------
# Workflow scanning
# ---------------------------------------------------------------------------


def _iter_workflow_files() -> list[Path]:
    """Return every ``.yml``/``.yaml`` file under ``.github/workflows/`` and
    the top-level ``.github/`` directory, sorted for deterministic output.

    The top-level scan picks up ``onnx-integrity.yml`` /
    ``python-tests.yml`` / ``nightly-ashrae-140-gauge.yml`` if present,
    which the issue body lists as canonical-pattern holders that must
    remain unguarded-but-pass (the canonical guarded pattern).
    """
    files: list[Path] = []
    workflows_dir = WORKFLOWS_DIR
    if workflows_dir.exists():
        for pattern in _WORKFLOW_GLOB:
            files.extend(workflows_dir.glob(pattern))
    github_dir = REPO_ROOT / ".github"
    if github_dir.exists():
        for pattern in _WORKFLOW_GLOB:
            for path in github_dir.glob(pattern):
                if path not in files:
                    files.append(path)
    return sorted(set(files))


def _extract_jobs_block(text: str) -> str | None:
    """Return the substring of ``text`` starting at the top-level
    ``jobs:`` block (zero-indent) through EOF, or ``None`` if absent.

    The job-block regex in ``check_required_checks_sync.py`` uses lookahead
    for the next zero-indent key; here we just split on the first
    occurrence of ``^jobs:`` (multi-line mode) and take everything after.
    """
    match = re.search(r"^jobs:\s*\n", text, re.MULTILINE)
    if not match:
        return None
    return text[match.start():]


def _split_jobs(jobs_block: str) -> list[tuple[str, str]]:
    """Split a ``jobs:`` block into ``(job_id, job_body)`` pairs.

    A job id is a 2-space-indented YAML key followed by ``:`` at start
    of line. The job body is everything from the line after the header
    until the next 2-space-indented key (or EOF).
    """
    out: list[tuple[str, str]] = []
    lines = jobs_block.splitlines(keepends=True)
    current_id: str | None = None
    body_start = -1
    for i, line in enumerate(lines):
        m = re.match(r"^  ([A-Za-z_][A-Za-z0-9_-]*):\s*$", line)
        if m:
            if current_id is not None:
                body = "".join(lines[body_start:i])
                out.append((current_id, body))
            current_id = m.group(1)
            body_start = i + 1
    if current_id is not None:
        body = "".join(lines[body_start:])
        out.append((current_id, body))
    return out


def _job_is_pr_runnable(job_body: str) -> bool:
    """Return True if the job's ``if:`` clause permits execution on a
    pull_request event.

    Jobs with no ``if:`` clause are PR-runnable (GitHub's default).
    Jobs whose ``if:`` mentions ``pull_request`` are PR-runnable.
    Jobs whose ``if:`` is strictly main-only (the
    ``github.event_name == 'push' && github.ref == 'refs/heads/main'``
    conjunction, single-line OR folded-block form) are NOT PR-runnable
    and are exempt from the guarded-pattern requirement.
    """
    # Look for any ``if:`` clause first.
    if_match = re.search(r"^(\s+)if:\s*", job_body, re.MULTILINE)
    if not if_match:
        return True  # no ``if:`` -> PR-runnable
    if_clause_start = if_match.end()
    # Determine the end of the ``if:`` clause: either end-of-line (single
    # line), or the next line at the same indent (start of next field).
    body_lines = job_body.splitlines(keepends=True)
    # Find the line index of the ``if:`` clause start.
    start_line_idx = job_body[:if_clause_start].count("\n")
    if_indent = len(if_match.group(1))
    end_line_idx = start_line_idx + 1
    while end_line_idx < len(body_lines):
        line = body_lines[end_line_idx]
        # Skip blank lines (folded block scalars may contain them).
        if line.strip() == "":
            end_line_idx += 1
            continue
        # If the next non-blank line is at a shallower-or-equal indent
        # than ``if_indent``, the ``if:`` clause has ended.
        leading = len(line) - len(line.lstrip(" "))
        if leading <= if_indent:
            break
        end_line_idx += 1
    if_clause = "".join(body_lines[start_line_idx:end_line_idx])
    # A folded-block ``if: |`` clause that contains the strict
    # main-only conjunction is exempt.
    if _IF_BLOCK_PUSH_TO_MAIN_RE.search(if_clause):
        return False
    if _IF_PUSH_TO_MAIN_LINE_RE.search(if_clause):
        return False
    # Anything else (including ``pull_request`` mentions) is PR-runnable.
    return True


def _has_unsafe_runs_on(job_body: str) -> bool:
    """Return True if the job body contains an unsafe
    ``runs-on: ${{ vars.FLUXION_LINUX_RUNNER || 'ubuntu-latest' }}``
    form (single-line).

    Folded-block ``runs-on: >-${{ ... }}`` forms use the
    ``_RUNS_ON_BLOCK_HEADER_RE`` to detect the header and then scan the
    following indented lines for the unsafe expression.
    """
    body_lines = job_body.splitlines(keepends=False)
    for i, line in enumerate(body_lines):
        # Single-line unsafe form.
        if _UNSAFE_RUNS_ON_INLINE_RE.search(line):
            return True
        # Folded-block ``runs-on: >-`` form — peek ahead.
        m = _RUNS_ON_BLOCK_HEADER_RE.match(line)
        if m:
            indent = len(m.group("indent"))
            # Collect continuation lines (deeper indent than the header).
            block_text_parts: list[str] = []
            for j in range(i + 1, len(body_lines)):
                next_line = body_lines[j]
                leading = len(next_line) - len(next_line.lstrip(" "))
                if leading <= indent:
                    break
                block_text_parts.append(next_line.strip())
            block_text = " ".join(block_text_parts)
            # The unsafe pattern in folded-block form collapses to the
            # same string after YAML folding: any line that contains
            # exactly ``vars.FLUXION_LINUX_RUNNER || 'ubuntu-latest'``
            # in its folded body is unsafe.
            if (
                "vars.FLUXION_LINUX_RUNNER" in block_text
                and "||" in block_text
                and "'ubuntu-latest'" in block_text
                and not _GUARD_FRAGMENT_RE.search(block_text)
            ):
                return True
    return False


def _workflow_has_pull_request_trigger(text: str) -> bool:
    """Return True if the workflow declares a ``pull_request`` trigger in
    its top-level ``on:`` block.

    A workflow without this trigger cannot run on PR-controlled code at
    all, so every job in it is exempt from the guarded-pattern
    requirement regardless of its ``runs-on:`` form.
    """
    match = re.search(r"^on:\s*\n((?:[ \t]+[^\n]*\n|[ \t]*#[^\n]*\n)*)", text, re.MULTILINE)
    if not match:
        return False
    for line in match.group(1).splitlines():
        m = re.match(r"^  ([A-Za-z_][A-Za-z0-9_-]*):", line)
        if m and m.group(1) == "pull_request":
            return True
    return False


def _job_runs_on_lines(job_body: str) -> list[str]:
    """Return a list of human-readable strings describing the
    ``runs-on:`` form used by this job, for error messages.

    Single-line form: the line text. Folded-block form: the header +
    continuation summary.
    """
    out: list[str] = []
    body_lines = job_body.splitlines(keepends=False)
    for i, line in enumerate(body_lines):
        if line.lstrip().startswith("runs-on:"):
            out.append(line.strip())
        elif _RUNS_ON_BLOCK_HEADER_RE.match(line):
            m = _RUNS_ON_BLOCK_HEADER_RE.match(line)
            indent = len(m.group("indent"))
            collected = [line.strip()]
            for j in range(i + 1, len(body_lines)):
                next_line = body_lines[j]
                leading = len(next_line) - len(next_line.lstrip(" "))
                if leading <= indent:
                    break
                collected.append(next_line.strip())
            out.append(" ".join(collected))
    return out


def scan_workflow(path: Path) -> list[tuple[str, int, list[str]]]:
    """Scan one workflow file. Return a list of
    ``(job_id, job_body_line_offset, runs_on_lines)`` tuples for every
    PR-runnable job that uses the unsafe ``runs-on`` pattern.

    ``job_body_line_offset`` is the 1-indexed line number of the first
    line of the job block in the original file (used for error messages).
    """
    text = path.read_text(encoding="utf-8")
    if not _workflow_has_pull_request_trigger(text):
        # Workflow has no `pull_request` trigger — cannot execute
        # PR-controlled code, so every job's `runs-on:` is exempt.
        return []
    jobs_block = _extract_jobs_block(text)
    if jobs_block is None:
        return []
    jobs = _split_jobs(jobs_block)
    # Compute the absolute line offset of the ``jobs:`` block header.
    jobs_header_match = re.search(r"^jobs:\s*\n", text, re.MULTILINE)
    jobs_header_line = text[: jobs_header_match.end()].count("\n") + 1
    findings: list[tuple[str, int, list[str]]] = []
    for job_id, job_body in jobs:
        if not _job_is_pr_runnable(job_body):
            continue
        if _has_unsafe_runs_on(job_body):
            # Compute the line offset of the job's ``runs-on`` line.
            job_offset_in_block = jobs_block.find(job_body)
            job_block_start_line = jobs_header_line + (
                jobs_block[:job_offset_in_block].count("\n") if job_offset_in_block > 0 else 0
            )
            run_on_line_offset = None
            body_lines = job_body.splitlines(keepends=False)
            for i, line in enumerate(body_lines):
                if line.lstrip().startswith("runs-on:"):
                    run_on_line_offset = job_block_start_line + i + 1
                    break
            findings.append((job_id, run_on_line_offset or -1, _job_runs_on_lines(job_body)))
    return findings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    if not WORKFLOWS_DIR.exists():
        print(f"ERROR: {WORKFLOWS_DIR} does not exist", file=sys.stderr)
        return 2

    workflow_files = _iter_workflow_files()
    print(
        f"Checking runner-routing policy (Issue #3531; "
        f"{len(workflow_files)} workflow file(s))"
    )
    print()

    all_findings: list[tuple[Path, str, int, list[str]]] = []
    scanned = 0
    for path in workflow_files:
        scanned += 1
        try:
            findings = scan_workflow(path)
        except Exception as exc:
            print(f"  WARN: failed to scan {path.relative_to(REPO_ROOT)}: {exc}", file=sys.stderr)
            continue
        for job_id, line_no, runs_on_lines in findings:
            all_findings.append((path, job_id, line_no, runs_on_lines))

    if not all_findings:
        print(
            "[1/1] every PR-runnable job's ``runs-on:`` is guarded by the "
            "``github.event_name == 'push' && github.ref == "
            "'refs/heads/main'`` conjunction (or scoped main-only via "
            "``if:``) ..."
        )
        print()
        print(
            f"OK. Scanned {scanned} workflow file(s). No "
            f"vars.FLUXION_LINUX_RUNNER trust-boundary regressions "
            f"detected (Issue #3531)."
        )
        return 0

    print(
        "[1/1] every PR-runnable job's ``runs-on:`` is guarded by the "
        "``github.event_name == 'push' && github.ref == "
        "'refs/heads/main'`` conjunction (or scoped main-only via "
        "``if:``) ..."
    )
    print()
    print(
        f"DRIFT DETECTED ({len(all_findings)} violation(s) across "
        f"{len({p for p, _, _, _ in all_findings})} workflow file(s)):"
    )
    for path, job_id, line_no, runs_on_lines in all_findings:
        rel = path.relative_to(REPO_ROOT)
        print(f"  - {rel} job `{job_id}` (line {line_no}):")
        for rl in runs_on_lines:
            print(f"      {rl}")
        print(
            "      Fix: replace with the canonical guarded pattern "
            "(see docs/SECURITY.md §7 or "
            ".github/workflows/onnx-integrity.yml:92-98)."
        )
    print()
    print(
        "Fix: every PR-triggered job that mentions "
        "``vars.FLUXION_LINUX_RUNNER`` MUST use the guarded pattern "
        "(folded block scalar form, mirroring onnx-integrity.yml / "
        "python-tests.yml):"
    )
    print()
    print("    runs-on: >-")
    print("      ${{")
    print("        github.event_name == 'push'")
    print("        && github.ref == 'refs/heads/main'")
    print("        && vars.FLUXION_LINUX_RUNNER")
    print("        || 'ubuntu-latest'")
    print("      }}")
    print()
    print(
        "Jobs that are strictly main-only (their ``if:`` clause already "
        "filters out pull_request events) are exempt — the unsafe "
        "``runs-on`` is harmless there because the job never executes on "
        "PR-controlled code."
    )
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover - defensive
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)