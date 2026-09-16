#!/usr/bin/env python3
"""
Required-check / workflow-index drift detection for Fluxion (Issue #2866,
extended by #3116 and #3441).

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

The script enforces six invariants:

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
5. Every "NN checks" count literal in ``AGENTS.md`` and
   ``docs/ci/branch-protection-strict-mode.md`` matches the parsed
   ``ci.required_checks`` / ``ci.required_checks_workflow_only`` list
   lengths (and the "N path-filtered checks" arithmetic
   ``len(required) - len(workflow_only)``). Issue #3441 reconciled a
   31/26 drift where the docs still said 19/29/25 and "4 path-filtered
   checks" after Module Size (#2878) was wired in; this guard keeps the
   prose counts from drifting again.
6. Every ``ci.required_checks_workflow_only`` entry (Issue #3810) must
   either (a) point at a workflow with no ``pull_request.paths`` /
   ``pull_request.paths-ignore`` filter, OR (b) point at a workflow
   whose ``jobs`` block declares an additive listener job whose
   ``name:`` equals the required-check string exactly AND whose job
   carries ``if: always()``. This is the GH-listener pattern: a
   path-filtered workflow cannot emit a check run on a docs-only /
   scripts-only PR, so branch protection's exact-string context would
   be perpetually "expected" but never reported (#3805). The listener
   job is unconditional (``if: always()``) so it observes the upstream
   even when the upstream was ``skipped`` (path-filter neutral-success)
   and PASSes; a hard failure on the upstream propagates as a FAIL.
   Without this invariant, develop's branch-protection context list
   had to drop to the small set of path-filter-free required checks,
   leaving the rest silently unsatisfied (the #3810 / #3805 gap).

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
``ci.required_checks_workflow_only`` list exactly (and that
``required_pull_request_reviews.required_approving_review_count`` ≥ 1
and ``enforce_admins.enabled`` is true). The comparison list is the
**workflow-only set** (not the full ``ci.required_checks``) because
``scripts/apply_branch_protection.py`` — the source-of-truth applier —
restores ``develop`` branch protection to exactly this 18-entry set
(Issue #3810 design intent; see lines 382-388 of that script). The full
``ci.required_checks`` list intentionally contains the 5 path-filtered
checks (Docs Hygiene, Architecture Drift, Module Size, Crate Size,
MSRV) that never report on docs-only / scripts-only PRs; branch
protection cannot require what never emits, so comparing against the
full list produced false-positive drift (Issue #3831). Designed to run
as a scheduled cron in ``.github/workflows/`` so #3116's "configuration
has 0 required checks" gap cannot recur silently. Always exits 0 in the
default (static-only) mode — the live check is opt-in to keep this
script network-free for the PR-blocking CI invocation.

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


def get_workflow_only_checks(gates: dict) -> list[str]:
    """Return the ``ci.required_checks_workflow_only`` list as raw strings.

    Mirrors :func:`get_required_checks` (comment-only ``None`` entries
    are dropped). Added for the Issue #3441 doc-count sync guard.
    """
    ci = gates.get("ci") or {}
    raw = ci.get("required_checks_workflow_only") or []
    out: list[str] = []
    for entry in raw:
        if isinstance(entry, str):
            out.append(entry)
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

# Issue #3810: detect `paths:` and `paths-ignore:` keys at 4-space indent
# under `pull_request:` (the standard GitHub Actions path-filter shape).
# We capture the entire `pull_request:` sub-block (its lines, including
# nested `paths:` / `paths-ignore:`) so a later helper can decide
# whether the workflow is path-filtered. The block runs from the
# `pull_request:` line itself to the next 2-space-indent key (or EOF).
_PULL_REQUEST_BLOCK_RE = re.compile(
    r"^  pull_request:\s*\n((?:    [^\n]*\n|    #[^\n]*\n|[ \t]*#[^\n]*\n)*)",
    re.MULTILINE,
)
_PATH_FILTER_KEY_RE = re.compile(
    r"^    (paths|paths-ignore):\s*$", re.MULTILINE
)

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
# `name:`). Quoted and unquoted values both supported. The line
# pattern (`    +\S[^\n]*\n`) tolerates arbitrary continuation
# indentation (e.g. folded scalar `>-` continuations at 6+ spaces) so
# the regex window does NOT terminate prematurely on
# `runs-on: >-\n      ${{ ...` style blocks — that bug made the
# Issue #3810 6th invariant miss the existing
# `fast-math-gh`/`if: always()` listener in `fast_math_check.yml`.
_JOB_RE = re.compile(
    r"^  ([A-Za-z_][A-Za-z0-9_-]*):\n((?:    +\S[^\n]*\n){1,15})",
    re.MULTILINE,
)
_NAME_RE = re.compile(
    r"^    name:\s*"
    r'(?:"(?P<v>[^"\n]*)"|'  # double-quoted (literal; `#` is allowed inside)
    r"'(?P<v2>[^'\n]*)'|"     # single-quoted (literal; `#` is allowed inside)
    r"(?P<v3>[^\"'\n#]+))",   # unquoted: stop at `#` (YAML comment marker) too
    re.MULTILINE,
)
# Detect the `if: always()` directive at the top of a job block. The
# listener pattern (Issue #3810 / #3358) is exactly this: a job whose
# `name:` equals a required-check string and whose `if:` is `always()`.
# The regex anchors on column 4 (the standard job-key indent) and
# tolerates YAML quoting around the `always()` value.
_IF_ALWAYS_RE = re.compile(
    r"^    if:\s*[\"']?always\(\)[\"']?\s*$", re.MULTILINE
)
_WORKFLOW_NAME_RE = re.compile(r"^name:\s*\"?(?P<v>[^\"\n]+)\"?", re.MULTILINE)


def parse_workflow(path: Path) -> dict:
    """Parse one workflow file and return ``{"triggers": [...], "jobs": {...},
    "job_unconditional": {...}, "pull_request_path_filtered": bool,
    "workflow_name": "..."}``.

    Uses regex (not PyYAML) because GitHub Actions workflows embed
    ``${{ }}`` expressions and other constructs PyYAML cannot parse.

    The Issue #3810 extensions (``job_unconditional`` and
    ``pull_request_path_filtered``) are additive: existing callers
    that only read ``triggers`` / ``jobs`` / ``workflow_name`` are
    unaffected. The two new keys drive the 6th invariant (GH-listener
    pattern detection).
    """
    text = path.read_text(encoding="utf-8")

    triggers: list[str] = []
    on_match = _ON_BLOCK_RE.search(text)
    if on_match:
        for line in on_match.group(1).splitlines():
            m = _TRIGGER_KEY_RE.match(line)
            if m:
                triggers.append(m.group(1))

    # Issue #3810: does the `pull_request:` block declare `paths:` or
    # `paths-ignore:`? A workflow with no path filter runs on every PR
    # class (docs-only / scripts-only / physics); one with a filter is
    # path-gated and needs a listener job to emit its required-check
    # name unconditionally. The regex captures the entire
    # `pull_request:` sub-block; a single `paths:` / `paths-ignore:`
    # line anywhere in that block triggers the flag.
    pull_request_path_filtered = False
    pr_match = _PULL_REQUEST_BLOCK_RE.search(text)
    if pr_match:
        for line in pr_match.group(1).splitlines():
            if _PATH_FILTER_KEY_RE.match(line):
                pull_request_path_filtered = True
                break

    job_names: dict[str, str] = {}
    job_unconditional: dict[str, bool] = {}
    jobs_match = _JOBS_BLOCK_RE.search(text)
    if jobs_match:
        jobs_block = jobs_match.group(1)
        for jm in _JOB_RE.finditer(jobs_block):
            jid = jm.group(1)
            block = jm.group(2)
            nm = _NAME_RE.search(block)
            if nm:
                # The regex captures into named groups v (double-quoted),
                # v2 (single-quoted), or v3 (unquoted). All three carry
                # the same semantic — emit the value as the GitHub
                # check_run name — but only one will be set per match.
                emitted = nm.group("v") or nm.group("v2") or nm.group("v3")
                job_names[jid] = emitted.strip()
            # Issue #3810: `if: always()` at the job level is the
            # marker for the GH-listener pattern (the listener is
            # unconditional — fires even if the upstream was skipped
            # via path-filter). Per the canonical pattern
            # (AGENTS.md / #3358), the listener's classification step
            # then maps `success`/`skipped` to PASS and
            # `failure`/`cancelled`/`timed_out` to FAIL.
            job_unconditional[jid] = bool(_IF_ALWAYS_RE.search(block))

    wn_match = _WORKFLOW_NAME_RE.search(text)
    workflow_name = wn_match.group("v").strip() if wn_match else None

    return {
        "triggers": triggers,
        "jobs": job_names,
        "job_unconditional": job_unconditional,
        "pull_request_path_filtered": pull_request_path_filtered,
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


def workflow_has_pull_request_path_filter(workflow: dict) -> bool:
    """Return True if the workflow's ``pull_request`` block declares a
    ``paths:`` or ``paths-ignore:`` key (Issue #3810).

    A path-filtered ``pull_request`` trigger causes the workflow to be
    SKIPPED on PRs whose changed files are all matched by the filter
    (e.g. docs-only / scripts-only PRs hitting a workflow with
    ``paths-ignore: ['docs/**']``). When skipped, no check runs are
    emitted and GitHub branch protection's exact-string contexts array
    is left "expected" but never reported — the gap Issue #3810 closes
    with the GH-listener pattern.

    The parser tolerates block-comments and 6-space-indent nested
    continuation; the on-block capture in :data:`_ON_BLOCK_RE` is
    intentionally permissive. ``workflow_run:`` triggers are NOT
    path-filterable (GitHub ignores ``paths`` under ``workflow_run``),
    so this helper looks specifically at ``pull_request``.
    """
    return bool(workflow.get("pull_request_path_filtered", False))


def workflow_has_unconditional_listener(workflow: dict, job_name: str) -> bool:
    """Return True if ``workflow`` satisfies the Issue #3810 GH-listener
    pattern for ``job_name``.

    A workflow satisfies the listener pattern iff:

    * **(a) No path filter.** ``workflow_has_pull_request_path_filter``
      returns False — every PR class triggers the workflow and the
      existing job(s) emit unconditionally.
    * **(b) Has an unconditional listener.** The workflow's ``jobs``
      block contains a job whose ``name:`` equals ``job_name`` AND
      whose top-level ``if:`` is ``always()`` (so the listener fires
      even when the upstream was ``skipped`` by a path-filter
      elsewhere in the workflow — the GH-probe / Hetzner-overflow
      listener convention, AGENTS.md "Required checks sync discipline").

    Used by the 6th invariant in ``collect_drift`` to detect the
    exact gap #3805 documented: a path-filtered required check whose
    workflow has NO unconditional listener. Branch protection can name
    such a check, but it never reports on docs-only / scripts-only
    PRs — leaving develop with an unsatisfied context list that the
    wave orchestrator (not CI) has to monitor.
    """
    # (a) No path filter → the existing job emission covers every PR.
    if not workflow_has_pull_request_path_filter(workflow):
        return True

    # (b) An unconditional listener job must exist with the exact
    # emitted name AND carry `if: always()` at the job level.
    job_unconditional = workflow.get("job_unconditional") or {}
    for jid, jname in workflow.get("jobs", {}).items():
        if jname != job_name:
            continue
        if job_unconditional.get(jid, False):
            return True

    return False


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


def collect_drift(
    required_checks: list[str],
    workflow_index: list[dict],
    workflows: dict[str, dict],
    workflow_only_checks: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Run all six invariants and return ``(failures, informational)``.

    ``failures`` is non-empty when the script must exit 1; ``informational``
    is for findings that are not blocking (e.g. ``workflow_index`` entries
    that intentionally live outside ``required_checks``).

    The 6th invariant (Issue #3810) was added with the ``workflow_only_checks``
    parameter so existing callers do not have to pass it (defaulting to
    ``None`` runs the first five invariants and skips the new check).
    The ``main()`` invocation passes the parsed value.
    """
    failures: list[str] = []
    informational: list[str] = []
    workflow_only_checks = workflow_only_checks or []

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

    # Issue #3810 — 6th invariant: every `required_checks_workflow_only`
    # entry must satisfy the GH-listener pattern (Issue #3116 verbatim
    # match + the unconditional listener convention). The path-filtered
    # workflow's `pull_request:` block is skipped on docs-only /
    # scripts-only PRs — branch protection names the check but no
    # check run is emitted, leaving develop with an unsatisfied
    # required context list. The remediation is either:
    #
    #   (a) host the listener in an unfiltered workflow (the Option B
    #       `rust-tests-listeners.yml` pattern), OR
    #   (b) add an additive `name: <exact>` job to the path-filtered
    #       workflow with `if: always()` so the listener fires even
    #       when the upstream was skipped by the path filter.
    #
    # The invariant enforces BOTH the workflow_index mapping (so the
    # listener's owning workflow is recorded) AND the listener's
    # `if: always()` marker (so a future contributor cannot forget it
    # in a copy-paste from #3358's pattern).
    for rc in workflow_only_checks:
        entry = workflow_index_jobs.get(rc)
        if entry is None:
            # Invariant 4 above already reports a missing
            # workflow_index entry for `required_checks` — same gap
            # applies to `required_checks_workflow_only`, but only
            # when the workflow_only entry lacks a workflow_index row
            # entirely. The drift message is intentionally precise
            # so the contributor can fix either side.
            failures.append(
                f"required_checks_workflow_only {rc!r} has no "
                f"workflow_index entry — Issue #3810 requires a "
                f"workflow_index mapping so the listener's owning "
                f"workflow is recorded for the drift gate."
            )
            continue
        wf_path = entry.get("workflow") or ""
        wf = workflows.get(wf_path)
        if wf is None:
            # Invariant 1 above already reports a missing workflow
            # file — the GH-listener pattern check is moot without
            # the file on disk.
            continue
        if not workflow_has_unconditional_listener(wf, rc):
            # Actionable remediation: list the two pattern options so
            # the contributor can pick (a) or (b) without reading the
            # whole issue thread.
            is_path_filtered = workflow_has_pull_request_path_filter(wf)
            job_names = sorted(wf["jobs"].values())
            unconditional_jobs = sorted(
                jid
                for jid, is_unconditional in (
                    wf.get("job_unconditional") or {}
                ).items()
                if is_unconditional
            )
            failures.append(
                f"required_checks_workflow_only {rc!r} ({wf_path}) is "
                f"path-filtered and has no unconditional listener that "
                f"emits the required-check name (Issue #3810). "
                f"path-filtered={is_path_filtered}; unconditional jobs in "
                f"workflow: {unconditional_jobs}; emitted job names: "
                f"{job_names}. "
                f"Remediation: either (a) move the listener to an "
                f"unfiltered workflow (rust-tests-listeners.yml Option B "
                f"pattern), or (b) add an additive listener job to "
                f"{wf_path} with `name: {rc!r}` and `if: always()` "
                f"that classifies `success`/`skipped` as PASS and "
                f"`failure`/`cancelled`/`timed_out` as FAIL."
            )

    return failures, informational


# ---------------------------------------------------------------------------
# Doc count-literal sync (Issue #3441)
# ---------------------------------------------------------------------------

# Count-literal shapes validated in AGENTS.md and the branch-protection
# runbook. Each pattern captures the "NN" of an "NN checks" literal and
# binds it to a release_gates.yaml list length — or, for
# "path_filtered_delta", to len(required) - len(workflow_only), the
# number of path-filtered checks excluded from the workflow-only list.
#
# The same-line binding (``[^\n]*?``) is deliberate: the count literal
# must appear on the same line as the ``required_checks`` /
# ``required_checks_workflow_only`` token it describes, so unrelated
# numbers elsewhere in the prose (test counts, issue refs, the
# historical "14 of 23" incident narrative) can never be bound by
# accident.
DOC_COUNT_PATTERNS = (
    # "`release_gates.yaml -> ci.required_checks_workflow_only` (26 checks)"
    (
        re.compile(r"required_checks_workflow_only[^\n]*?(\d+)\s+checks"),
        "required_checks_workflow_only",
    ),
    # "**`required_checks`** — All checks ... (31 checks)" and
    # "Use `required_checks` (all 31 checks)". The negative lookahead
    # keeps the workflow-only token from matching the bare pattern.
    (
        re.compile(r"required_checks(?!_workflow_only)[^\n]*?(\d+)\s+checks"),
        "required_checks",
    ),
    # Runbook note: "The 26 always-run checks provide ..."
    (
        re.compile(r"(\d+)\s+always-run\s+checks"),
        "required_checks_workflow_only",
    ),
    # "This excludes the 5 path-filtered checks above" /
    # "it removes only the 5 path-filtered checks"
    (
        re.compile(r"(\d+)\s+path-filtered\s+checks"),
        "path_filtered_delta",
    ),
)

# Docs whose "NN checks" literals are validated against the parsed
# release_gates.yaml lists. Files that do not exist (e.g. the tmp_path
# mock repos used by scripts/ci tests) are skipped silently.
DOC_COUNT_DOCS = (
    "AGENTS.md",
    "docs/ci/branch-protection-strict-mode.md",
)

# Which count literals each doc MUST carry at least once — deleting the
# literal must fail the gate just like drifting it (Issue #3441).
DOC_COUNT_REQUIRED_KEYS = {
    "AGENTS.md": ("required_checks_workflow_only",),
    "docs/ci/branch-protection-strict-mode.md": (
        "required_checks",
        "required_checks_workflow_only",
        "path_filtered_delta",
    ),
}

_KEY_DISPLAY = {
    "required_checks": "ci.required_checks",
    "required_checks_workflow_only": "ci.required_checks_workflow_only",
    "path_filtered_delta": (
        "len(ci.required_checks) - len(ci.required_checks_workflow_only)"
    ),
}


def collect_doc_count_drift(
    required_checks: list[str],
    workflow_only_checks: list[str],
) -> list[str]:
    """Validate the "NN checks" count literals in AGENTS.md and the
    branch-protection runbook against the parsed release_gates.yaml
    lists (Issue #3441).

    Reads the docs from the module-level ``REPO_ROOT`` *at call time*
    (so tests can monkey-patch it at a synthetic ``tmp_path`` tree,
    mirroring :func:`load_release_gates`). Docs that do not exist under
    the current ``REPO_ROOT`` are skipped — the mock repos used by the
    ``scripts/ci`` test harness carry only ``release_gates.yaml`` and
    ``.github/workflows/``.

    Returns a list of failure messages; empty means every count literal
    agrees with the YAML (and each required literal is present).
    """
    expected = {
        "required_checks": len(required_checks),
        "required_checks_workflow_only": len(workflow_only_checks),
        "path_filtered_delta": len(required_checks)
        - len(workflow_only_checks),
    }
    failures: list[str] = []
    for rel in DOC_COUNT_DOCS:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        seen: set[str] = set()
        for pattern, key in DOC_COUNT_PATTERNS:
            for m in pattern.finditer(text):
                seen.add(key)
                literal = int(m.group(1))
                if literal != expected[key]:
                    failures.append(
                        f"{rel}: count literal ({literal} checks) is "
                        f"bound to {_KEY_DISPLAY[key]} but "
                        f"release_gates.yaml has {expected[key]}. "
                        f"Update the doc count (Issue #3441)."
                    )
        for key in DOC_COUNT_REQUIRED_KEYS.get(rel, ()):
            if key not in seen:
                failures.append(
                    f"{rel}: no 'NN checks' count literal found for "
                    f"{_KEY_DISPLAY[key]} (should say {expected[key]}). "
                    f"The count reference was deleted or moved off the "
                    f"token's line (Issue #3441)."
                )
    return failures


# ---------------------------------------------------------------------------
# Live branch-protection verification (cron-mode, opt-in)
# ---------------------------------------------------------------------------


def check_live_branch_protection(
    workflow_only_checks: list[str],
    repo: str = "anchapin/fluxion",
    branch: str = "develop",
) -> list[str]:
    """Verify the live GitHub branch protection for ``repo:branch`` matches
    ``workflow_only_checks`` (Issues #3116 closure + #3810 design intent).

    Returns a list of human-readable failure messages; empty list means the
    live protection matches. Reads the configured branch protection via
    ``gh api`` and checks:

    * ``required_status_checks.contexts`` matches ``workflow_only_checks``
      by symmetric set equality (the same set, in any order). This is the
      **always-run / workflow-only set** (the 18-entry list) — not the
      full ``ci.required_checks`` (23 entries). The applier
      ``scripts/apply_branch_protection.py`` writes exactly this set to
      ``develop`` branch protection (Issue #3810): the 5 path-filtered
      checks cannot be required at branch-protection level because they
      never report on docs-only / scripts-only PRs. Comparing against the
      full list previously produced false-positive drift (Issue #3831).
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
    expected = set(workflow_only_checks)

    if contexts != expected:
        missing = sorted(expected - contexts)
        extra = sorted(contexts - expected)
        if missing:
            failures.append(
                f"develop branch protection is missing required check(s): "
                f"{missing}. Add via `gh api --method PUT` (Issue #3116 "
                f"closure; the comparison list is the workflow-only set, "
                f"not the full required_checks — Issue #3831)."
            )
        if extra:
            failures.append(
                f"develop branch protection has stale check(s) not in "
                f"release_gates.yaml ci.required_checks_workflow_only: "
                f"{extra}. Either remove them or re-add the corresponding "
                f"required_checks_workflow_only entry."
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
    workflow_only_checks = get_workflow_only_checks(gates)
    workflow_index = get_workflow_index(gates)
    workflows = load_all_workflows()

    print(
        f"Parsed {len(required_checks)} required_check(s), "
        f"{len(workflow_only_checks)} required_checks_workflow_only "
        f"entr(ies), {len(workflow_index)} workflow_index entr(ies), "
        f"{len(workflows)} workflow file(s)."
    )
    print()

    failures, informational = collect_drift(
        required_checks, workflow_index, workflows, workflow_only_checks
    )
    failures.extend(
        collect_doc_count_drift(required_checks, workflow_only_checks)
    )

    print(
        "[1/7] every workflow_index entry references an existing "
        ".github/workflows/*.yml file ..."
    )
    print(
        "[2/7] every workflow_index.job matches a jobs.<id>.name in that "
        "workflow EXACTLY (no canonical+suffix tolerance — Issue #3116) ..."
    )
    print(
        "[3/7] every workflow_index workflow declares a pull_request or "
        "workflow_run trigger ..."
    )
    print(
        "[4/7] every required_check has a matching workflow_index entry "
        "(exact job-string equality) AND no canonical-vs-suffix drift ..."
    )
    print(
        "[5/7] the 'NN checks' count literals in AGENTS.md and "
        "docs/ci/branch-protection-strict-mode.md match the parsed "
        "release_gates.yaml list lengths (Issue #3441) ..."
    )
    print(
        "[6/7] every required_checks_workflow_only entry has an "
        "unconditional listener in its hosting workflow (Issue #3810 "
        "GH-listener pattern) ..."
    )
    print(
        "[7/7] when FLUXION_CHECK_LIVE_PROTECTION=1, the live "
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
        # Issue #3831: the live 7th invariant compares against the
        # workflow-only list (18 entries), not the full required_checks
        # list (23 entries). apply_branch_protection.py writes the
        # workflow-only set; comparing against the full set produced
        # false-positive drift for the 5 intentionally-excluded
        # path-filtered checks (Docs Hygiene, Architecture Drift,
        # Module Size, Crate Size, MSRV).
        live_failures = check_live_branch_protection(workflow_only_checks)
        if live_failures:
            print(f"LIVE DRIFT DETECTED ({len(live_failures)} failure(s)):")
            for msg in live_failures:
                print(f"  - {msg}")
            print()
            print(
                "Fix: update develop branch protection so the live "
                "required_status_checks.contexts match release_gates.yaml "
                "ci.required_checks_workflow_only (Issues #3116 closure + "
                "#3831 alignment with apply_branch_protection.py). See "
                "`scripts/apply_branch_protection.py --help` for the "
                "`gh api` payload shape."
            )
            return 1
        print(
            "Live develop branch protection matches release_gates.yaml "
            "ci.required_checks_workflow_only (contexts, strict, reviews, "
            "enforce_admins)."
        )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover - defensive
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)