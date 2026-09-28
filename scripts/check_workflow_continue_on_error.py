#!/usr/bin/env python3
"""
CI guard: reject dead `steps.<id>.outcome` references to steps declared
`continue-on-error: true` (Issue #4159).

Background — the GitHub Actions result-context semantics that make this
a bug class rather than a style preference:

  * `steps.<id>.outcome`   = the step result *before* `continue-on-error`
    is applied.
  * `steps.<id>.conclusion` = the step result *after* `continue-on-error`
    is applied.

So for a step declared `continue-on-error: true`, `outcome` is pinned to
`'success'` and `conclusion` becomes `'failure'`. Every downstream
reference to that step's `.outcome` is therefore a constant:

  * `steps.x.outcome == 'failure'`  -> always false. A re-raise guard that
    can never fire. This is the #4159 shape: the "Fail if ... failed"
    step is dead code, and the gate it claims to enforce is never
    actually enforced.
  * `steps.x.outcome == 'success'`  -> always true. A *guard* that runs
    even when the step it guards actually failed.

Both are the same defect: the control asserts a fact about the run that
the expression cannot observe. The `continue-on-error: true` mask is
still legitimate on its own — it exists to let `if: always()` artifact
upload steps run — but the re-raise must read `.conclusion`.

The repository already shipped five instances of this across four
workflows, including the ASHRAE 140 validation gate (#4159) and the
ASHRAE 140 benchmark-harness regression gate. This check makes the class
structurally impossible to re-introduce.

Scope: a finding is reported only when a step in the *same job* is
declared `continue-on-error: true` AND some step in that job references
`steps.<that id>.outcome`. References to `.conclusion`, to non-masked
steps, or to step ids in other jobs are all correct and not reported.

Wired into the `Scripts Test Suite` job of
`.github/workflows/scripts-tests.yml` ("Reject dead continue-on-error
re-raises", Issue #4159). The matcher's invariants are additionally
covered by `scripts/ci/test_check_workflow_continue_on_error.py`
against hermetic `tmp_path` fixtures.

Exit codes:
    0 — no dead re-raise references.
    1 — one or more dead `steps.<id>.outcome` references.
    2 — script error (e.g. `.github/workflows/` missing).

Usage:
    python3 scripts/check_workflow_continue_on_error.py
    python3 scripts/check_workflow_continue_on_error.py --workflow ci.yml
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# `steps.<id>.outcome` / `steps.<id>.conclusion` in either an `if:`
# expression or a `${{ }}` interpolation inside a `run:` body.
_OUTCOME_REF_RE = re.compile(r"steps\.([A-Za-z0-9_-]+)\.outcome")


def _as_text(value: object) -> str:
    """Flatten a YAML scalar (or None) to text for regex scanning."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _masked_step_ids(steps: list[dict]) -> dict[str, str]:
    """Map ``id -> step name`` for steps with ``continue-on-error: true``."""
    masked: dict[str, str] = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        sid = step.get("id")
        if isinstance(sid, str) and sid and step.get("continue-on-error") is True:
            masked[sid] = str(step.get("name") or sid)
    return masked


def _reference_text(step: dict) -> str:
    """Concatenate the `if:` expression and `run:` body of a step.

    These are the only two fields that can observe a step result; the
    YAML scalar for a folded/literal `run:` block keeps its newlines, so
    the caller can report the offending line verbatim.
    """
    return _as_text(step.get("if")) + "\n" + _as_text(step.get("run"))


def check_workflow(path: Path, rel: str | None = None) -> list[str]:
    """Return dead-re-raise findings for one workflow.

    Empty list means the workflow is compliant. ``rel`` overrides the
    path used in finding text (tests pass a synthetic name).
    """
    if rel is None:
        try:
            rel = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            # Path lives outside the repo (e.g. a hermetic tmp_path
            # fixture). Report the bare filename rather than raising --
            # a checker must never crash on the input it is asked to scan.
            rel = path.name
    findings: list[str] = []

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return [f"{rel}: unparseable YAML ({exc})"]
    if not isinstance(data, dict):
        return findings

    jobs = data.get("jobs")
    if not isinstance(jobs, dict):
        return findings

    for job_name, job in jobs.items():
        if not isinstance(job, dict):
            continue
        steps = job.get("steps")
        if not isinstance(steps, list):
            continue

        masked = _masked_step_ids(steps)
        if not masked:
            continue

        for step in steps:
            if not isinstance(step, dict):
                continue
            haystack = _reference_text(step)
            for match in _OUTCOME_REF_RE.finditer(haystack):
                sid = match.group(1)
                if sid not in masked:
                    continue
                expr = _matched_line(haystack, match)
                findings.append(
                    f"{rel}::job {job_name}: step {step.get('name') or step.get('id')!r} "
                    f"reads `steps.{sid}.outcome`, but step {sid!r} "
                    f"({masked[sid]!r}) is declared `continue-on-error: true`, "
                    f"which pins `.outcome` to 'success'. Found: {expr!r}. "
                    f"Use `steps.{sid}.conclusion` instead."
                )
    return findings


def _matched_line(haystack: str, match: re.Match[str]) -> str:
    """Return the single source line containing ``match``, trimmed."""
    start = haystack.rfind("\n", 0, match.start()) + 1
    end = haystack.find("\n", match.end())
    if end == -1:
        end = len(haystack)
    return haystack[start:end].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        help="Restrict to a single workflow file (e.g. rust-tests.yml).",
    )
    args = parser.parse_args()

    if not WORKFLOWS_DIR.is_dir():
        print(f"ERROR: {WORKFLOWS_DIR} not found", file=sys.stderr)
        return 2

    files = sorted(WORKFLOWS_DIR.glob("*.yml"))
    if args.workflow:
        files = [f for f in files if f.name == args.workflow]
        if not files:
            print(f"ERROR: workflow {args.workflow} not found", file=sys.stderr)
            return 2

    findings: list[str] = []
    for path in files:
        findings.extend(check_workflow(path))

    if findings:
        print("Dead `steps.<id>.outcome` re-raise detected:", file=sys.stderr)
        for f in findings:
            print(f"  - {f}", file=sys.stderr)
        print(
            f"\n{len(findings)} finding(s) across {len(files)} workflow(s). "
            "A `continue-on-error: true` step has `outcome == 'success'` "
            "permanently; read `.conclusion` to observe its real result.",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: all {len(files)} workflow(s) reference `.conclusion` (not the "
        "`continue-on-error`-pinned `.outcome`) for masked steps."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
