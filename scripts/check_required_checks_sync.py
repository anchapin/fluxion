#!/usr/bin/env python3
"""
Required-check / workflow-index drift detection for Fluxion (Issue #2866,
extended by #3116).

GitHub branch protection reads ``jobs.<id>.name`` from
``.github/workflows/*.yml`` directly when matching required status checks,
and the contexts array matches those names **verbatim** — there is no
canonical-vs-suffix tolerance, no regex/wildcard support, and no fallback
to ``workflow.name`` for single-job workflows. ``release_gates.yaml``
also declares a parallel ``ci.workflow_index`` map that names each
required check's owning workflow + job, and a ``ci.required_checks``
list whose entries must be the *exact* check name GitHub reports.

Issue #2866 documents that this mapping has been silently drifting: every
required check was added without a corresponding ``workflow_index`` entry,
and a few entries (``Physics-Sim-Cycle-Check``, ``Workspace Check``,
``MSRV Check (Issue #2934)``) had no exact job-name match in the workflow
file the index pointed at. This script was the regression guard, but its
original ``job_in_workflow`` implementation included a ``CANONICAL_NAME_SUFFIXES``
tolerance (lines that matched ``canonical`` + `` (GH)`` /
`` (Hetzner Overflow)`` as if branch protection did the same).

Issue #3116 documents that GitHub branch protection does no such
tolerance — the canonical-vs-suffix drift meant three required_checks
(``Workspace Check``, ``Physics-Sim-Cycle-Check``, ``Architecture Drift
Detection``) were matched only by this script's regex tolerance but never
by GitHub's actual contexts array, leaving develop with zero required
checks while every entry passed locally. The fix tightens the validation
below: **every required_check and every workflow_index entry must match an
actual ``jobs.<id>.name`` in the referenced workflow *exactly***. If a
workflow emits suffixed variants (e.g. ``Workspace Check (GH)`` and
``Workspace Check (Hetzner Overflow)``), the YAML must name the suffixed
job explicitly — never the bare canonical.

The script enforces four invariants:

1. Every ``workflow_index`` entry points at an existing
   ``.github/workflows/<name>.yml`` file.
2. Every ``workflow_index.job`` matches an actual ``jobs.<id>.name`` in
   that workflow *exactly*. **No canonical+suffix tolerance** — the
   tolerance that used to live here was the root cause of #3116 and has
   been removed. Workflows that emit multiple ``(GH)`` / ``(Hetzner
   Overflow)`` variants per listener must name the variant explicitly
   in ``workflow_index`` (or list it in ``required_checks``).
3. Every workflow referenced by ``workflow_index`` declares a
   ``pull_request`` or ``workflow_run`` trigger — scheduled-only workflows
   never produce a check run that can block a PR, so they cannot be a
   required status check.
4. Every ``ci.required_checks`` entry has a matching ``workflow_index``
   entry by exact job-string equality, so branch protection and the
   informational workflow index cannot silently diverge.

The script deliberately does NOT enforce the inverse (every
``workflow_index`` entry must also be in ``required_checks``) — the
``WASM Build Verification (Issue #2914)`` entry is intentionally not in
``required_checks`` (the YAML comment above it documents this; the size
assertion is PR-blocking but the cross-platform interface-stability check
`` is opt-in). Adding that as a hard fail would re-introduce the very
silence the gate was built to detect. Such entries are reported as
"informational" findings instead.

Live branch-protection verification (cron-mode)
-----------------------------------------------

Set ``FLUXION_CHECK_LIVE_PROTECTION=1`` (and have ``gh auth`` working)
to additionally ``gh api``-query ``develop``'s branch protection and
verify the live ``required_status_checks.contexts`` array matches the
``ci.required_checks`` list exactly (and that
``required_pull_request_reviews.required_approving_review_count`` ≥ 1
and ``enforce_admins.enabled`` is true). Designed to run as a scheduled
cron in ``.github/workflows/`` so #3116's "configuration has 0 required
checks" gap cannot recur silently. Always exits 0 in the default
(static-only) mode — the live check is opt-in to keep this script
network-free for the PR-blocking CI invocation.

Usage::

    python3 scripts/check_required_checks_sync.py             # static-only
    FLUXION_CHECK_LIVE_PROTECTION=1 python3 scripts/check_required_checks_sync.py
                                                            # also live API

Exit codes:

    0 — no drift detected (every check above holds; live protection
        matches when ``FLUXION_CHECK_LIVE_PROTECTION=1``).
    1 — drift detected (one or more required_checks / workflow_index
        entries are stale or missing; or live protection diverges).
    2 — script error (e.g. ``release_gates.yaml`` missing or unparseable).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is in scripts/requirements-test.txt
    sys.stderr.write(
        "ERROR: PyYAML is required for scripts/check_required_checks_sync.py. "
        "Install with `pip install pyyaml` (already in "
        "scripts/requirements-test.txt, used by the scripts-tests workflow).\n"
    )
    sys.exit(2)


REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE_GATES_YAML = REPO_ROOT / "release_gates.yaml"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# Triggers that produce a check run which GitHub branch protection can
# reference. `workflow_run` is included because some required checks are
# listener jobs that observe an upstream workflow's completion (e.g. the
# determinism / performance gates); the listener still creates a check run
# that blocks the PR.
BLOCKING_TRIGGERS = frozenset({"pull_request", "workflow_run"})

# Documented job-name suffixes. Several required checks share a single
# "canonical" listener name across multiple runner variants (GH runner +
# Hetzner overflow). The listener jobs in `.github/workflows/rust-tests.yml`
# use ``name: "<canonical> (GH)"`` and ``name: "<canonical> (Hetzner
# Overflow)"``; ``release_gates.yaml`` documents the suffixed names
# (``"Workspace Check (GH)"``, ``"Physics-Sim-Cycle-Check (GH)"``) as the
# branch-protection entries and the corresponding ``workflow_index.job``
# values. The pre-#3116 contract held the bare canonical name in YAML and
# relied on a ``CANONICAL_NAME_SUFFIXES`` regex tolerance in this script
# to "match" — but GitHub branch protection's contexts array does not
# strip suffixes, so the canonical entries never satisfied the gate while
# still passing this script. The contract was tightened in #3116: YAML
# must name the exact emitted ``jobs.<id>.name``. This constant is
# retained as a list of suffixes that, *if seen in the workflow*, indicate
# the YAML must include the suffixed name; the matching helper
# ``job_in_workflow`` no longer applies the tolerance itself.
CANONICAL_NAME_SUFFIXES = (
    " (GH)",
    " (Hetzner Overflow)",
)


# ---------------------------------------------------------------------------
# release_gates.yaml parsing
# ---------------------------------------------------------------------------


def load_release_gates(path: Path | None = None) -> dict:
    """Load ``release_gates.yaml`` and return the parsed structure.

    ``path`` defaults to the module-level ``RELEASE_GATES_YAML`` *at call
    time* (NOT at function-definition time) so tests can monkey-patch
    the constant. Using a default argument would freeze the original
    path at import time and silently bypass the test fixture.

    Raises ``FileNotFoundError`` if the file is missing. PyYAML is
    sufficient here because release_gates.yaml is pure YAML with no
    GitHub Actions-specific extensions (no ``${{ }}`` expressions).
    """
    if path is None:
        path = RELEASE_GATES_YAML
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level must be a mapping")
    return data


def get_required_checks(gates: dict) -> list[str]:
    """Return the ``ci.required_checks`` list as raw strings.

    Skips comment-only entries (lines that are entirely ``# ...``) which
    PyYAML may surface as ``None`` in the list.
    """
    ci = gates.get("ci") or {}
    raw = ci.get("required_checks") or []
    out: list[str] = []
    for entry in raw:
        if isinstance(entry, str):
            out.append(entry)
        # PyYAML may surface comment lines as None; drop them.
    return out


def get_workflow_index(gates: dict) -> list[dict]:
    """Return the ``ci.workflow_index`` list as a list of mappings.

    Each entry is expected to have at least ``job`` and ``workflow`` keys;
    entries missing one are returned as-is so the caller can flag them.
    """
    ci = gates.get("ci") or {}
    raw = ci.get("workflow_index") or []
    out: list[dict] = []
    for entry in raw:
        if isinstance(entry, dict):
            out.append(entry)
    return out


# ---------------------------------------------------------------------------
# .github/workflows/*.yml parsing (regex-based — PyYAML chokes on ${{ }})
# ---------------------------------------------------------------------------


# Match the YAML `on:` block at the top of a workflow. We capture the
# *first-level* keys (2-space indent) until the next 0-indent key or EOF.
# Comment-only lines (`# ...`) are tolerated because several workflows
# have block-comments inside the `on:` block (e.g. issue #1351 in
# ashrae_validation.yml).
_ON_BLOCK_RE = re.compile(
    r"^on:\s*\n((?:[ \t]+[^\n]*\n|[ \t]*#[^\n]*\n)*)",
    re.MULTILINE,
)
_TRIGGER_KEY_RE = re.compile(r"^  ([A-Za-z_][A-Za-z0-9_-]*):", re.MULTILINE)

# Match the YAML `jobs:` block at the top of a workflow. We split on the
# 0-indent `jobs:` token and capture everything until the next 0-indent
# key (or EOF).
_JOBS_BLOCK_RE = re.compile(
    r"^jobs:\s*\n(.*?)(?=^[A-Za-z_]|\Z)",
    re.MULTILINE | re.DOTALL,
)
# Each job id is at 2-space indent, then `name:` (if present) appears at
# 4-space indent *somewhere* within the first ~15 lines of the job block
# (allowing `runs-on:`, `permissions:`, etc. to interleave before
# `name:`). Quoted and unquoted values both supported.
_JOB_RE = re.compile(
    r"^  ([A-Za-z_][A-Za-z0-9_-]*):\n((?:    [^ #\n].*\n|    #[^\n]*\n){1,15})",
    re.MULTILINE,
)
_NAME_RE = re.compile(r'^    name:\s*"?(?P<v>[^"\n]+)"?', re.MULTILINE)
_WORKFLOW_NAME_RE = re.compile(r"^name:\s*\"?(?P<v>[^\"\n]+)\"?", re.MULTILINE)


def parse_workflow(path: Path) -> dict:
    """Parse one workflow file and return ``{"triggers": [...], "jobs": {...},
    "workflow_name": "..."}``.

    Uses regex (not PyYAML) because GitHub Actions workflows embed
    ``${{ }}`` expressions and other constructs PyYAML cannot parse.
    """
    text = path.read_text(encoding="utf-8")

    triggers: list[str] = []
    on_match = _ON_BLOCK_RE.search(text)
    if on_match:
        for line in on_match.group(1).splitlines():
            m = _TRIGGER_KEY_RE.match(line)
            if m:
                triggers.append(m.group(1))

    job_names: dict[str, str] = {}
    jobs_match = _JOBS_BLOCK_RE.search(text)
    if jobs_match:
        jobs_block = jobs_match.group(1)
        for jm in _JOB_RE.finditer(jobs_block):
            jid = jm.group(1)
            block = jm.group(2)
            nm = _NAME_RE.search(block)
            if nm:
                job_names[jid] = nm.group("v").strip()

    wn_match = _WORKFLOW_NAME_RE.search(text)
    workflow_name = wn_match.group("v").strip() if wn_match else None

    return {
        "triggers": triggers,
        "jobs": job_names,
        "workflow_name": workflow_name,
    }


def load_all_workflows() -> dict[str, dict]:
    """Parse every ``.github/workflows/*.yml`` and return ``{filename: parsed}``.

    Restricted to top-level ``*.yml`` files (skips ``scripts/`` and any
    nested ``reusable-*.yml`` that may appear in future).
    """
    out: dict[str, dict] = {}
    if not WORKFLOWS_DIR.exists():
        return out
    for path in sorted(WORKFLOWS_DIR.glob("*.yml")):
        # Key by the *relative path* used in workflow_index entries, e.g.
        # ".github/workflows/ashrae_140_strict_energy_gate.yml".
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            out[rel] = parse_workflow(path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  WARN: failed to parse {rel}: {exc}", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# Matching primitives
# ---------------------------------------------------------------------------


def job_in_workflow(workflow: dict, job_name: str) -> bool:
    """Return True if ``job_name`` matches a job in ``workflow`` (after
    YAML quote-stripping normalisation) **exactly**.

    GitHub branch protection's ``required_status_checks.contexts`` array
    matches the emitted ``jobs.<id>.name`` *verbatim* — there is no
    canonical-vs-suffix tolerance, no regex / wildcard support, and no
    fallback to the workflow's top-level ``name:`` field for single-job
    workflows. Issue #3116 closed the gap where this helper previously
    applied a ``CANONICAL_NAME_SUFFIXES`` regex tolerance and matched the
    bare canonical name (e.g. ``Workspace Check``) against a workflow job
    named ``Workspace Check (GH)`` — that tolerance made the local sync
    check pass while GitHub's actual gate stayed perpetually unsatisfied.

    Match rule: the YAML entry is considered to match iff the workflow
    declares a job whose ``name:`` equals ``job_name`` byte-for-byte.
    When the workflow has a single job and that job's ``name:`` is the
    same as the workflow's top-level ``name:``, this still holds because
    the YAML entry names the job (which happens to equal the workflow
    name). Workflows that emit multiple ``(GH)`` / ``(Hetzner Overflow)``
    variants per listener must list each variant explicitly in YAML.
    """
    for actual_name in workflow["jobs"].values():
        if actual_name == job_name:
            return True
    return False


def workflow_has_suffixed_variant(workflow: dict, job_name: str) -> bool:
    """Return True if ``workflow`` contains a job whose ``name:`` equals
    ``job_name + suffix`` for any entry in ``CANONICAL_NAME_SUFFIXES``.

    Used by ``collect_drift`` to surface the canonical-vs-suffix drift
    explicitly: when a YAML entry uses a bare canonical name but the
    workflow emits a suffixed variant, the entry passes ``job_in_workflow``
    only via the legacy tolerance and will never satisfy branch
    protection. Issue #3116's regression guard.
    """
    for actual_name in workflow["jobs"].values():
        for suffix in CANONICAL_NAME_SUFFIXES:
            if actual_name == job_name + suffix:
                return True
    return False


def has_blocking_trigger(workflow: dict) -> bool:
    """Return True if the workflow declares a pull_request or workflow_run
    trigger (the only two that produce PR-blockable check runs)."""
    return any(t in BLOCKING_TRIGGERS for t in workflow["triggers"])


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


def collect_drift(
    required_checks: list[str],
    workflow_index: list[dict],
    workflows: dict[str, dict],
) -> tuple[list[str], list[str]]:
    """Run all four invariants and return ``(failures, informational)``.

    ``failures`` is non-empty when the script must exit 1; ``informational``
    is for findings that are not blocking (e.g. ``workflow_index`` entries
    that intentionally live outside ``required_checks``).
    """
    failures: list[str] = []
    informational: list[str] = []

    workflow_index_jobs: dict[str, dict] = {}
    for entry in workflow_index:
        job = entry.get("job")
        wf_path = entry.get("workflow")
        if not job or not wf_path:
            failures.append(
                f"workflow_index entry missing required key(s): {entry}"
            )
            continue
        if job in workflow_index_jobs:
            failures.append(
                f"workflow_index has duplicate job entry: {job!r} "
                f"(first at {workflow_index_jobs[job].get('workflow')}, "
                f"second at {wf_path})"
            )
            continue
        workflow_index_jobs[job] = entry

        if wf_path not in workflows:
            failures.append(
                f"workflow_index.job {job!r}: workflow file not found: {wf_path}"
            )
            continue

        wf = workflows[wf_path]
        if not job_in_workflow(wf, job):
            job_names = sorted(wf["jobs"].values())
            failures.append(
                f"workflow_index.job {job!r} not found in {wf_path}. "
                f"Actual jobs: {job_names}"
            )

        if not has_blocking_trigger(wf):
            failures.append(
                f"workflow_index.job {job!r} ({wf_path}) has no "
                f"pull_request or workflow_run trigger "
                f"(found: {wf['triggers']})"
            )

    # Every required_check must have a workflow_index entry by exact
    # job-string equality. This is the user-facing branch-protection
    # check name and is what GitHub sees on each PR.
    for rc in required_checks:
        if rc not in workflow_index_jobs:
            # Try to suggest a workflow file that has the job, so the
            # failure message is actionable.
            suggestions = []
            for wf_path, wf in workflows.items():
                if rc in wf["jobs"].values():
                    suggestions.append(wf_path)
            hint = (
                f" — did you mean to add workflow_index entry pointing at "
                f"{suggestions[0]}?" if len(suggestions) == 1 else ""
            )
            failures.append(
                f"required_check {rc!r} has no workflow_index entry{hint}"
            )
        else:
            # Canonical-vs-suffix drift guard (Issue #3116). When a
            # required_check has a workflow_index entry and that workflow
            # contains a job named "<rc> (GH)" or "<rc> (Hetzner
            # Overflow)" but NO job with the bare <rc> name, the YAML
            # is using the legacy canonical form and would never satisfy
            # GitHub branch protection's exact-string contexts.
            wf_path = workflow_index_jobs[rc]["workflow"]
            wf = workflows.get(wf_path)
            if wf is not None and workflow_has_suffixed_variant(wf, rc):
                suffixed = sorted(
                    n for n in wf["jobs"].values()
                    if any(n == rc + s for s in CANONICAL_NAME_SUFFIXES)
                )
                failures.append(
                    f"required_check {rc!r} is in canonical form but "
                    f"{wf_path} only emits suffixed variants {suffixed!r}. "
                    f"GitHub branch protection matches the emitted job "
                    f"name verbatim (Issue #3116) — update this "
                    f"required_check to the suffixed name."
                )

    # Informational: workflow_index entries not referenced by any
    # required_check (the WASM Build entry is the canonical example —
    # intentionally opt-in). Surface them so the next drift isn't silent
    # but don't fail the gate.
    for job in workflow_index_jobs:
        if job not in required_checks:
            informational.append(
                f"workflow_index.job {job!r} is not referenced by any "
                f"required_check (intentionally opt-in or stale?)"
            )

    return failures, informational


# ---------------------------------------------------------------------------
# Live branch-protection verification (cron-mode, opt-in)
# ---------------------------------------------------------------------------


def check_live_branch_protection(
    required_checks: list[str],
    repo: str = "anchapin/fluxion",
    branch: str = "develop",
) -> list[str]:
    """Verify the live GitHub branch protection for ``repo:branch`` matches
    ``required_checks`` (Issue #3116 closure).

    Returns a list of human-readable failure messages; empty list means the
    live protection matches. Reads the configured branch protection via
    ``gh api`` and checks:

    * ``required_status_checks.contexts`` matches ``required_checks`` by
      symmetric set equality (the same set, in any order).
    * ``required_status_checks.strict`` is True.
    * ``required_pull_request_reviews.required_approving_review_count``
      is at least 1.
    * ``enforce_admins.enabled`` is True.

    Requires ``gh auth`` to be configured for the target repo. The cron-
    mode invocation (see module docstring) sets
    ``FLUXION_CHECK_LIVE_PROTECTION=1`` to enable this check; the default
    static-only mode never invokes ``gh api`` and is network-free for the
    PR-blocking CI invocation.

    The check is intentionally hard-coded to ``develop`` and the canonical
    upstream ``anchapin/fluxion`` so a misconfigured cron invocation
    cannot accidentally probe an unrelated repo's branch protection.
    """
    import json
    import subprocess

    failures: list[str] = []

    try:
        proc = subprocess.run(
            [
                "gh",
                "api",
                f"/repos/{repo}/branches/{branch}/protection",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return [f"gh api invocation failed: {exc}"]

    if proc.returncode != 0:
        return [
            f"gh api returned {proc.returncode}: "
            f"{proc.stderr.strip()[:200] or '(no stderr)'}"
        ]

    try:
        protection = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return [f"gh api response is not JSON: {exc}"]

    rsc = protection.get("required_status_checks") or {}
    contexts = set(rsc.get("contexts") or [])
    expected = set(required_checks)

    if contexts != expected:
        missing = sorted(expected - contexts)
        extra = sorted(contexts - expected)
        if missing:
            failures.append(
                f"develop branch protection is missing required check(s): "
                f"{missing}. Add via `gh api --method PUT` (see Issue #3116)."
            )
        if extra:
            failures.append(
                f"develop branch protection has stale check(s) not in "
                f"release_gates.yaml: {extra}. Either remove them or "
                f"re-add the corresponding required_check entry."
            )

    if not rsc.get("strict", False):
        failures.append(
            "develop branch protection has strict=false. Issue #3116 "
            "acceptance criterion requires strict=true so out-of-date "
            "branches are blocked."
        )

    rpr = protection.get("required_pull_request_reviews") or {}
    approving = rpr.get("required_approving_review_count") or 0
    if approving < 1:
        failures.append(
            f"develop branch protection has required_approving_review_count="
            f"{approving}. Issue #3116 acceptance criterion requires ≥1."
        )

    admins = protection.get("enforce_admins") or {}
    if not admins.get("enabled", False):
        failures.append(
            "develop branch protection has enforce_admins.enabled=false. "
            "Should be true so the gate applies to admins too."
        )

    return failures


def main() -> int:
    print(
        f"Checking release_gates.yaml <-> .github/workflows/ sync "
        f"(issue #2866; repo: {REPO_ROOT})"
    )
    print()

    try:
        gates = load_release_gates()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    required_checks = get_required_checks(gates)
    workflow_index = get_workflow_index(gates)
    workflows = load_all_workflows()

    print(
        f"Parsed {len(required_checks)} required_check(s), "
        f"{len(workflow_index)} workflow_index entr(ies), "
        f"{len(workflows)} workflow file(s)."
    )
    print()

    failures, informational = collect_drift(
        required_checks, workflow_index, workflows
    )

    print(
        "[1/5] every workflow_index entry references an existing "
        ".github/workflows/*.yml file ..."
    )
    print(
        "[2/5] every workflow_index.job matches a jobs.<id>.name in that "
        "workflow EXACTLY (no canonical+suffix tolerance — Issue #3116) ..."
    )
    print(
        "[3/5] every workflow_index workflow declares a pull_request or "
        "workflow_run trigger ..."
    )
    print(
        "[4/5] every required_check has a matching workflow_index entry "
        "(exact job-string equality) AND no canonical-vs-suffix drift ..."
    )
    print(
        "[5/5] when FLUXION_CHECK_LIVE_PROTECTION=1, the live "
        "develop branch protection matches release_gates.yaml ..."
    )
    print()

    if informational:
        print(f"INFORMATIONAL ({len(informational)} finding(s) — NOT a failure):")
        for msg in informational:
            print(f"  - {msg}")
        print()

    if failures:
        print(f"DRIFT DETECTED ({len(failures)} failure(s)):")
        for msg in failures:
            print(f"  - {msg}")
        print()
        print(
            "Fix: update release_gates.yaml::ci.workflow_index so each "
            "required_check maps to the actual jobs.<id>.name in the "
            "referenced workflow file. Job renames in "
            ".github/workflows/*.yml that don't update workflow_index "
            "silently desync branch protection — that's the gap this "
            "gate exists to prevent. Canonical-vs-suffix drift: when a "
            "workflow emits suffixed variants, the YAML must name the "
            "suffixed job explicitly (Issue #3116)."
        )
        return 1

    print(
        f"No drift. {len(required_checks)} required_check(s) and "
        f"{len(workflow_index)} workflow_index entr(ies) are in sync "
        f"with {len(workflows)} workflow file(s)."
    )

    # Optional: cron-mode live branch-protection verification. See module
    # docstring for the rationale. Only enabled when explicitly opted in
    # via FLUXION_CHECK_LIVE_PROTECTION=1 so the PR-blocking CI invocation
    # stays network-free.
    if os.environ.get("FLUXION_CHECK_LIVE_PROTECTION") == "1":
        print()
        print(
            "FLUXION_CHECK_LIVE_PROTECTION=1 — verifying live develop "
            "branch protection via `gh api` ..."
        )
        live_failures = check_live_branch_protection(required_checks)
        if live_failures:
            print(f"LIVE DRIFT DETECTED ({len(live_failures)} failure(s)):")
            for msg in live_failures:
                print(f"  - {msg}")
            print()
            print(
                "Fix: update develop branch protection so the live "
                "required_status_checks.contexts match release_gates.yaml "
                "ci.required_checks (Issue #3116 closure). See "
                "`scripts/check_required_checks_sync.py --help` for the "
                "`gh api` payload shape."
            )
            return 1
        print(
            "Live develop branch protection matches release_gates.yaml "
            "(contexts, strict, reviews, enforce_admins)."
        )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover - defensive
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)