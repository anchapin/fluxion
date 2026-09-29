#!/usr/bin/env python3
"""
CI guard: a Trivy step must scan the artifact it claims to scan
(Issue #4186).

Background: `docs/SECURITY.md` §7 states that the unpinned `apt-get
install` layers inside the `Dockerfile` are deliberately out of scope for
the base-image digest control, and names Trivy (``docker.yml::security``)
as the compensating control that "catches package-level CVEs".

That compensating control did not exist. The step ran
``scan-type: 'fs'`` with ``scan-ref: '.'`` — a filesystem scan of the
*repository source tree*. An ``fs`` scan reads the project's own
dependency manifests; it never inspects the OS packages installed into the
image, which are exactly the layers the doc defers to Trivy. A CVE in
``libssl3`` or ``libgomp1`` in ``debian:bookworm-slim`` shipped to
``ghcr.io`` with nothing reporting it.

This check fails when a ``trivy-action`` step uses ``scan-type: 'fs'``,
because that combination cannot report on the built image. A contributor
who genuinely needs a source-tree scan (e.g. scanning lockfiles for
license/SBOM policy, a different concern from image CVEs) may keep it by
adding an inline justification comment on the step; the check requires the
justification to be explicit rather than silent, so the next reader can
tell an intentional source scan from a drifted image scan.

A companion invariant is checked in the same pass: a ``trivy-action`` step
that is meant to gate on findings must set ``exit-code``. The action has no
default for that input, so without it ``TRIVY_EXIT_CODE`` is never
exported and Trivy's own default (``0``) applies — the step reports
success no matter what it finds.

Scope: ``.github/workflows/**.yml``. Only steps that reference
``aquasecurity/trivy-action`` are considered.

Wired into the `Scripts Test Suite` job of
`.github/workflows/scripts-tests.yml` ("Reject Trivy source-tree scans",
Issue #4186). The matcher's invariants are additionally covered by
`scripts/ci/test_check_trivy_scan_target.py` against hermetic
``tmp_path`` fixtures.

Exit codes:
    0 — every Trivy step scans a real artifact and can fail.
    1 — one or more Trivy steps are misconfigured.
    2 — script error (e.g. the workflows directory is missing).

Usage:
    python3 scripts/check_trivy_scan_target.py
    python3 scripts/check_trivy_scan_target.py --workflows-dir .github/workflows
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

TRIVY_ACTION = "aquasecurity/trivy-action"

# A `fs` scan inspects the repository tree, never the built image.
FS_SCAN_TYPES = {"fs"}


def _step_comment_is_justified(raw_lines: list[str], step_start: int) -> bool:
    """True when the lines directly above the step document a source scan.

    Walks upward over the contiguous comment block immediately preceding
    the step and looks for an explicit justification marker. Requiring an
    explicit phrase (rather than "any comment") is deliberate: a stale
    comment must not silently authorize an `fs` scan forever.
    """
    markers = (
        "justif",
        "intentional",
        "source-tree scan by design",
        "source scan by design",
        "not an image scan",
        "lockfile",
        "sbom",
        "license",
    )
    idx = step_start - 1
    while idx >= 0:
        line = raw_lines[idx].strip()
        if not line.startswith("#"):
            # A non-comment, non-blank line ends the preceding block. Allow
            # a blank line to be skipped so a comment separated by one blank
            # line from the step still counts.
            if line == "":
                idx -= 1
                continue
            return False
        low = line.lower()
        if any(marker in low for marker in markers):
            return True
        idx -= 1
    return False


def check_workflow(path: Path, rel: str | None = None) -> list[str]:
    """Return Trivy misconfiguration findings for one workflow."""
    rel = rel or path.name
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return [f"{rel}: unparseable YAML ({exc})"]
    if not isinstance(data, dict):
        return []
    jobs = data.get("jobs")
    if not isinstance(jobs, dict):
        return []

    findings: list[str] = []
    raw_lines = path.read_text(encoding="utf-8").splitlines()

    # PyYAML does not expose line numbers on parsed nodes, so the comment
    # block above a step is located by pairing each `trivy-action` `uses:`
    # line with the corresponding step in document order. Both the raw scan
    # and the step walk are ordered, so the k-th occurrence pairs with the
    # k-th step.
    trivy_use_lines = [
        idx
        for idx, line in enumerate(raw_lines)
        if line.strip().startswith("uses:") and TRIVY_ACTION in line
    ]
    trivy_idx = 0

    for job_name, job in jobs.items():
        if not isinstance(job, dict):
            continue
        for step in job.get("steps") or []:
            if not isinstance(step, dict):
                continue
            uses = step.get("uses")
            if not isinstance(uses, str) or TRIVY_ACTION not in uses:
                continue
            label = f"{rel}::job {job_name}::step {step.get('name') or step.get('id')}"
            with_ = step.get("with")
            if not isinstance(with_, dict):
                with_ = {}

            # A contributor may legitimately keep a source-tree scan (an
            # SBOM/license policy check is a different concern from image
            # CVEs) or an advisory SARIF-only report. Both require an
            # explicit justification comment, so the escape hatch is a
            # documented decision rather than a silent default.
            use_line = (
                trivy_use_lines[trivy_idx]
                if trivy_idx < len(trivy_use_lines)
                else len(raw_lines) - 1
            )
            trivy_idx += 1
            justified = _step_comment_is_justified(raw_lines, use_line)

            scan_type = str(with_.get("scan-type", "image")).lower()
            if scan_type in FS_SCAN_TYPES and not justified:
                findings.append(
                    f"{label}: `scan-type: {scan_type!r}` scans the "
                    "repository source tree, which cannot report CVEs in the "
                    "OS packages installed into the built image "
                    "(`ca-certificates`, `libssl3`, `libgomp1`, `curl`). "
                    "`docs/SECURITY.md` §7 names this step as the compensating "
                    "control for the unpinned `apt-get` layers. Use "
                    "`scan-type: 'image'` with `image-ref` (or `input:` a "
                    "`docker save` tarball), or add an explicit justification "
                    "comment above the step if a source scan is intended."
                )

            if "exit-code" not in with_ and not justified:
                findings.append(
                    f"{label}: no `exit-code:` input. The action has no default "
                    "for it, so `TRIVY_EXIT_CODE` is never exported and Trivy's "
                    "own default (0) applies — this step cannot fail on any "
                    "finding. Set `exit-code: '1'` (with `ignore-unfixed: true` "
                    "and a `.trivyignore` policy) to make it fail-closed."
                )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflows-dir",
        type=Path,
        default=WORKFLOWS_DIR,
        help="Workflows directory to scan (default: .github/workflows).",
    )
    args = parser.parse_args()

    if not args.workflows_dir.is_dir():
        print(f"ERROR: {args.workflows_dir} not found", file=sys.stderr)
        return 2

    files = sorted(args.workflows_dir.glob("*.yml"))
    findings: list[str] = []
    for path in files:
        findings.extend(check_workflow(path))

    if findings:
        print("Trivy step misconfiguration:", file=sys.stderr)
        for f in findings:
            print(f"  - {f}", file=sys.stderr)
        print(
            f"\n{len(findings)} finding(s) across {len(files)} workflow(s). "
            "A source-tree scan cannot see the image's OS packages; a Trivy "
            "step with no `exit-code` cannot fail.",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: every Trivy step in {len(files)} workflow(s) scans a real "
        "artifact and can fail on findings."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
