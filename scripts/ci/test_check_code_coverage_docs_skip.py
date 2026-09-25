"""Regression tests for the Code Coverage Gate docs-only skip (Issue #3662).

History: PR #3660 — a docs-only PR touching only ``CONTRIBUTING.md``
and ``AGENTS.md`` — failed the ``Code Coverage Gate (Issue #1932)``
workflow with ``conduction_zone: branch coverage 65.51% < 65.62%
ratchet floor`` (a 0.11pp dip inside typical ``cargo llvm-cov``
measurement variance). The PR added zero code lines so coverage on the
critical path could not have actually regressed; the gate's noise floor
needed alignment.

Issue #3662 originally landed the Wave 8 / #3367 carve-out as a
``paths-ignore:`` block on the ``pull_request:`` trigger so docs-only
PRs would not invoke ``scripts/coverage_critical_paths.py --gate`` at
all.

Phase-gated CI (#4018 / #4019) moved the heavy workflows — including
Code Coverage — from ``pull_request`` triggers to ``workflow_run``
triggers behind CI Gates. ``workflow_run`` triggers do not support
``paths-ignore``, so the skip mechanism moved with it: the workflow
now carries a ``precheck`` job whose ``docs-only-gate`` step
(``.github/actions/docs-only-gate``, ADR-0016 lane-2 deny list)
computes ``docs_only`` from the upstream PR's file list, and the real
gate only runs when ``should_run == 'true'``. The old
``on.pull_request.paths-ignore`` block is gone; a ``paths-ignore``
block remains under ``on.workflow_run`` purely as documentation of
the carve-out (GitHub ignores it for ``workflow_run`` triggers).

These tests pin the new contract:

* ``test_code_coverage_workflow_precheck_uses_docs_only_gate`` — the
  ``precheck`` job exists and invokes the local docs-only-gate action
  (the LIVE skip mechanism).
* ``test_code_coverage_workflow_documents_docs_carve_out`` — the
  documentary ``on.workflow_run.paths-ignore`` block still lists the
  canonical documentation globs, so the carve-out stays visible and
  honest.
* ``test_docs_only_gate_action_covers_canonical_docs_patterns`` —
  the action's own match pattern covers ``docs/*``, ``*.md``,
  ``*.mdx`` (the shapes a docs-only PR takes).
* ``test_code_coverage_workflow_does_not_ignore_source_paths`` —
  defensive: the documented carve-out MUST NOT swallow ``src/**``,
  ``Cargo.toml``, ``scripts/**``, etc.
* Synthetic ``tmp_path`` fixtures exercise the inverse cases.

Mirrors the ``test_check_concurrency_keys.py`` / ``test_check_workflow_pin.py``
pattern (load the real workflow, assert its structure with a YAML
parser + targeted invariants).
"""

from __future__ import annotations

from pathlib import Path

import yaml  # type: ignore[import-untyped]

# Repo-relative paths resolved against ``Path(__file__)`` so the tests
# run unchanged from any worktree (CI + local-dev + agent worktrees).
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = (
    REPO_ROOT / ".github" / "workflows" / "code-coverage.yml"
)
DOCS_ONLY_GATE_ACTION_PATH = (
    REPO_ROOT / ".github" / "actions" / "docs-only-gate" / "action.yml"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_workflow(path: Path) -> dict:
    """Parse the YAML workflow at ``path`` and return the doc."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _precheck_steps(workflow: dict) -> list[dict]:
    """Return the ``precheck`` job's steps (possibly empty)."""
    jobs = workflow.get("jobs") or {}
    precheck = jobs.get("precheck") or {}
    return precheck.get("steps") or []


def _documented_carve_out(workflow: dict) -> list[str]:
    """Return the documentary ``on.workflow_run.paths-ignore`` list.

    Post-#4019 this block is documentation only — GitHub does not
    honor ``paths-ignore`` on ``workflow_run`` triggers — but it is
    kept in the file as the human-readable carve-out, so pin it.
    """
    on_block = workflow.get(True) or workflow.get("on") or {}
    workflow_run = on_block.get("workflow_run") or {}
    paths_ignore = workflow_run.get("paths-ignore") or []
    assert isinstance(paths_ignore, list), (
        "workflow_run.paths-ignore must be a YAML sequence (list of "
        f"strings), got {type(paths_ignore).__name__}"
    )
    return [str(p) for p in paths_ignore]


def _action_docs_alternatives() -> list[str]:
    """Extract the ``case`` pattern alternatives from the action.

    The docs-only-gate action matches PR file paths with a shell
    ``case`` statement like::

        docs/*|*.md|*.mdx|LICENSE|...)

    Return the ``|``-separated alternatives so tests can assert the
    canonical docs shapes are covered.
    """
    text = DOCS_ONLY_GATE_ACTION_PATH.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("docs/*|") and stripped.endswith(")"):
            return stripped[:-1].split("|")
    raise AssertionError(
        f"{DOCS_ONLY_GATE_ACTION_PATH} no longer declares its docs "
        "match pattern as a `docs/*|...` case statement — the skip "
        "contract moved; update these tests to the new shape."
    )


# Canonical documentation carve-out, mirrored verbatim from the
# documentary `.github/workflows/code-coverage.yml::on.workflow_run.
# paths-ignore` block. The explicit root `.md` entries are defensive
# (`**.md` already matches them) and document which root files count
# as "documentation".
EXPECTED_DOCS_GLOBS: tuple[str, ...] = (
    "docs/**",
    "**.md",
    "**.mdx",
    "README.md",
    "ARCHITECTURE.md",
    "AGENTS.md",
    "RULES.md",
    "CODEBASE_MAP.md",
    "SCORECARD.md",
    "KNOWN_ISSUES.md",
    "ASHRAE140_RESULTS.md",
    "CONTRIBUTING.md",
)


# Source-side patterns the carve-out MUST NOT swallow — a regression
# here would silently exempt code-changing PRs from coverage. The
# strings are tested as fnmatch-style glob fragments against each
# carve-out entry.
FORBIDDEN_DOCS_SKIP_GLOBS: tuple[str, ...] = (
    "src/**",
    "fluxion-core/src/**",
    "Cargo.toml",
    "Cargo.lock",
    "**/Cargo.toml",
    "**/Cargo.lock",
    "scripts/**",
    "**/*.rs",
)


def _write_workflow(tmp_path: Path, body: str, name: str = "code-coverage.yml") -> Path:
    """Write a synthetic workflow into ``tmp_path/.github/workflows``."""
    target = tmp_path / ".github" / "workflows" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Real-tree assertions
# ---------------------------------------------------------------------------


def test_code_coverage_workflow_file_exists():
    """Sanity: the workflow file is at the expected location."""
    assert WORKFLOW_PATH.is_file(), (
        f"Expected workflow at {WORKFLOW_PATH}, but the file does not "
        "exist. The Issue #3662 fix is anchored to "
        "`.github/workflows/code-coverage.yml`."
    )


def test_docs_only_gate_action_exists():
    """Sanity: the local action the precheck depends on exists."""
    assert DOCS_ONLY_GATE_ACTION_PATH.is_file(), (
        f"Expected action at {DOCS_ONLY_GATE_ACTION_PATH}, but the "
        "file does not exist. The phase-gated docs-only skip "
        "(#4018/#4019) depends on it."
    )


def test_code_coverage_workflow_precheck_uses_docs_only_gate():
    """The ``precheck`` job MUST invoke the local docs-only-gate action.

    Post-#4019 this step is the LIVE docs-only skip mechanism:
    ``workflow_run`` triggers do not support ``paths-ignore``, so the
    old ``on.pull_request.paths-ignore`` block is gone and the
    precheck's ``docs_only`` output decides whether the gate runs.
    Removing this step would re-open the PR #3660 noise-floor failure
    mode for every docs-only PR.
    """
    workflow = _load_workflow(WORKFLOW_PATH)
    steps = _precheck_steps(workflow)
    assert steps, (
        ".github/workflows/code-coverage.yml has no `precheck` job "
        "steps. The phase-gated docs-only skip (#4018/#4019) requires "
        "a precheck job driving `.github/actions/docs-only-gate`."
    )
    uses_values = [str(s.get("uses") or "") for s in steps]
    assert "./.github/actions/docs-only-gate" in uses_values, (
        ".github/workflows/code-coverage.yml::jobs.precheck does not "
        "invoke `./.github/actions/docs-only-gate`. Without it, "
        "docs-only PRs reach `scripts/coverage_critical_paths.py "
        "--gate` and can misfire the 1%-relative ratchet on "
        "measurement noise (Issue #3662, PR #3660)."
    )


def test_code_coverage_workflow_documents_docs_carve_out():
    """The documentary carve-out lists every canonical docs glob.

    ``on.workflow_run.paths-ignore`` is not honored by GitHub for
    ``workflow_run`` triggers — it is kept as the human-readable
    record of the carve-out. Pin it so the documentation cannot
    silently drift from the skip contract.
    """
    workflow = _load_workflow(WORKFLOW_PATH)
    carve_out = _documented_carve_out(workflow)
    missing = [g for g in EXPECTED_DOCS_GLOBS if g not in carve_out]
    assert not missing, (
        ".github/workflows/code-coverage.yml::on.workflow_run."
        f"paths-ignore is missing the canonical documentation globs: "
        f"{missing}. Keep the documented carve-out in sync with the "
        "docs-only skip contract."
    )


def test_docs_only_gate_action_covers_canonical_docs_patterns():
    """The action's match pattern covers the docs-only PR shapes.

    A docs-only PR touches ``docs/**``, ``*.md``, or ``*.mdx``. If
    the action's ``case`` pattern ever drops one of these shapes,
    such PRs would stop being detected as docs-only and would run
    the full coverage long pole (re-opening the #3660 failure mode).
    """
    alternatives = _action_docs_alternatives()
    for shape in ("docs/*", "*.md", "*.mdx"):
        assert shape in alternatives, (
            f".github/actions/docs-only-gate no longer matches "
            f"{shape!r} (pattern alternatives: {alternatives}). "
            "Docs-only PRs with that shape would no longer be "
            "detected as docs-only."
        )


def test_code_coverage_workflow_does_not_ignore_source_paths():
    """Defensive: the documented carve-out MUST NOT swallow source paths.

    A regression where ``src/**`` ends up in the carve-out would
    mis-document the skip contract and could mask a future change
    that re-introduces a live path filter. Use fnmatch-style glob
    containment to catch both literal entries and over-broad
    patterns.
    """
    import fnmatch

    workflow = _load_workflow(WORKFLOW_PATH)
    carve_out = _documented_carve_out(workflow)

    leaked: list[str] = []
    for forbidden in FORBIDDEN_DOCS_SKIP_GLOBS:
        for entry in carve_out:
            if fnmatch.fnmatchcase(forbidden, entry):
                leaked.append(f"{forbidden!r} matched by {entry!r}")

    assert not leaked, (
        ".github/workflows/code-coverage.yml::on.workflow_run."
        "paths-ignore swallowed a code-side pattern; the docs-only "
        f"carve-out must NEVER cover source-side changes. Findings: "
        f"{leaked}."
    )


# ---------------------------------------------------------------------------
# Synthetic-fixture invariants — pin the matcher semantics
# ---------------------------------------------------------------------------


def test_synthetic_precheck_without_docs_only_gate_is_rejected(tmp_path):
    """A precheck job without the docs-only-gate step is rejected.

    Pins the production assertion that the gate step is required, not
    optional, in the precheck job.
    """
    body = (
        "name: Code Coverage\n"
        "\n"
        "on:\n"
        "  workflow_run:\n"
        "    workflows: [\"CI Gates\"]\n"
        "    types: [completed]\n"
        "\n"
        "jobs:\n"
        "  precheck:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: upstream\n"
        "        run: echo hi\n"
    )
    workflow_path = _write_workflow(tmp_path, body)
    workflow = _load_workflow(workflow_path)
    steps = _precheck_steps(workflow)
    uses_values = [str(s.get("uses") or "") for s in steps]
    assert "./.github/actions/docs-only-gate" not in uses_values, (
        "Synthetic fixture: a precheck job without the docs-only-gate "
        "step must be detected as missing the skip mechanism."
    )


def test_synthetic_carve_out_missing_canonical_root_md_is_rejected(tmp_path):
    """A documentary carve-out missing ``KNOWN_ISSUES.md`` is rejected.

    Pins that the canonical root-markdown enumeration is the
    contract for the documented block.
    """
    partial = [g for g in EXPECTED_DOCS_GLOBS if g != "KNOWN_ISSUES.md"]
    body = (
        "name: Code Coverage\n"
        "\n"
        "on:\n"
        "  workflow_run:\n"
        "    workflows: [\"CI Gates\"]\n"
        "    types: [completed]\n"
        "    paths-ignore:\n"
        + "".join(f'      - "{p}"\n' for p in partial)
        + "\n"
        "jobs:\n"
        "  precheck:\n"
        "    runs-on: ubuntu-latest\n"
    )
    workflow_path = _write_workflow(tmp_path, body)
    workflow = _load_workflow(workflow_path)
    carve_out = _documented_carve_out(workflow)
    missing = [g for g in EXPECTED_DOCS_GLOBS if g not in carve_out]
    assert missing == ["KNOWN_ISSUES.md"], (
        "Synthetic fixture: the production assertion must surface "
        f"the missing entry; got missing={missing}"
    )


def test_synthetic_carve_out_with_src_is_rejected(tmp_path):
    """A carve-out containing ``src/**`` trips the source-leak guard.

    Pins that planting ``src/**`` in the documentary block is
    detected — the carve-out must never cover code-side paths.
    """
    import fnmatch

    body = (
        "name: Code Coverage\n"
        "\n"
        "on:\n"
        "  workflow_run:\n"
        "    workflows: [\"CI Gates\"]\n"
        "    types: [completed]\n"
        "    paths-ignore:\n"
        '      - "docs/**"\n'
        '      - "**.md"\n'
        '      - "src/**"\n'  # regression: code-side pattern
        "\n"
        "jobs:\n"
        "  precheck:\n"
        "    runs-on: ubuntu-latest\n"
    )
    workflow_path = _write_workflow(tmp_path, body)
    workflow = _load_workflow(workflow_path)
    carve_out = _documented_carve_out(workflow)

    leaked = []
    for forbidden in FORBIDDEN_DOCS_SKIP_GLOBS:
        for entry in carve_out:
            if fnmatch.fnmatchcase(forbidden, entry):
                leaked.append(f"{forbidden!r} matched by {entry!r}")
    assert leaked, (
        "Synthetic fixture: planting `src/**` in the documentary "
        "carve-out MUST be detected by the source-leak guard. The "
        "fixture did not trip the guard — the matcher regressed."
    )


def test_documented_carve_out_accepts_missing_block(tmp_path):
    """A workflow without the documentary block yields an empty list.

    Pins the parser branch for workflows that (legitimately) omit
    the documentary ``paths-ignore`` — the parser must not crash.
    """
    body = (
        "name: Code Coverage\n"
        "\n"
        "on:\n"
        "  workflow_run:\n"
        "    workflows: [\"CI Gates\"]\n"
        "    types: [completed]\n"
        "\n"
        "jobs:\n"
        "  precheck:\n"
        "    runs-on: ubuntu-latest\n"
    )
    workflow_path = _write_workflow(tmp_path, body)
    workflow = _load_workflow(workflow_path)
    assert _documented_carve_out(workflow) == []
