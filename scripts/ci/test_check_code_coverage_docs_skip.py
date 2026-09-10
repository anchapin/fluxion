"""Regression tests for the Code Coverage Gate docs-only skip (Issue #3662).

PR #3660 — a docs-only PR touching only ``CONTRIBUTING.md`` and
``AGENTS.md`` — failed the ``Code Coverage Gate (Issue #1932)`` workflow
with ``conduction_zone: branch coverage 65.51% < 65.62% ratchet floor``
(a 0.11pp dip inside typical ``cargo llvm-cov`` measurement variance).
The PR added zero code lines so coverage on the critical path could not
have actually regressed; the gate's noise floor needs alignment.

Issue #3662 lands the Wave 8 / #3367 carve-out already used by
``.github/workflows/rust-tests.yml`` (Issue #3367, PR #3369): declare a
``paths-ignore:`` block on the ``pull_request:`` trigger so docs-only
PRs do not invoke ``scripts/coverage_critical_paths.py --gate`` at all.
The script itself is unchanged — the fix lives at the workflow layer.

These tests pin the contract that the workflow file
(``.github/workflows/code-coverage.yml``) carries the
``paths-ignore:`` block with the canonical documentation patterns, so
a future contributor tightening (or accidentally removing) the carve-out
surfaces here instead of letting the next docs-only PR misfire the
ratchet gate. Mirrors the ``test_check_concurrency_keys.py`` /
``test_check_workflow_pin.py`` pattern (load the real workflow, assert
its structure with a YAML parser + targeted invariants).

Coverage:

* ``test_code_coverage_workflow_has_paths_ignore`` — the workflow's
  ``on.pull_request`` block declares ``paths-ignore``.
* ``test_code_coverage_workflow_paths_ignore_contains_docs_globs`` —
  the carve-out covers ``docs/**``, ``**.md``, ``**.mdx``, and the
  canonical root markdown files, matching the
  ``rust-tests.yml::paths-ignore`` set so a docs PR cannot slip
  through either workflow.
* ``test_code_coverage_workflow_paths_ignore_in_sync_with_rust_tests``
  — defensive invariant: the two path-filter sets stay in lock-step
  (the docs-only carve-out is a single repo-wide invariant, not a
  per-workflow one). Drift here would mean a docs PR that is filtered
  out of one heavy workflow still triggers another.
* ``test_code_coverage_workflow_does_not_ignore_source_paths`` —
  defensive: the carve-out MUST NOT ignore ``src/**``,
  ``fluxion-core/src/**``, ``Cargo.toml``, ``Cargo.lock``, or
  ``scripts/**`` so any code-touching PR still triggers the gate.
* Synthetic ``tmp_path`` workflow fixtures exercise the inverse
  cases (no ``paths-ignore`` -> assertion failure; root ``.md`` entry
  missing -> assertion failure; source path leaked into the
  carve-out -> assertion failure).
"""

from __future__ import annotations

from pathlib import Path

import yaml  # type: ignore[import-untyped]

# Repo-relative paths resolved against ``Path(__file__)`` so the tests
# run unchanged from any worktree (CI + local-dev + the agent worktree
# at ``../worktrees/issue-3662-coverage-flake-fix``).
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = (
    REPO_ROOT / ".github" / "workflows" / "code-coverage.yml"
)
RUST_TESTS_WORKFLOW_PATH = (
    REPO_ROOT / ".github" / "workflows" / "rust-tests.yml"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_workflow(path: Path) -> dict:
    """Parse the YAML workflow at ``path`` and return the doc.

    ``yaml.safe_load`` is consistent with the existing
    ``test_check_required_checks_sync.py`` / ``test_release_gate_checker.py``
    parsers — using the same library across the test suite keeps a
    parser regression surfaced uniformly.
    """
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _paths_ignore(workflow: dict) -> list[str]:
    """Return the ``on.pull_request.paths-ignore`` list (possibly empty)."""
    on_block = workflow.get(True) or workflow.get("on") or {}
    pull_request = on_block.get("pull_request") or {}
    if isinstance(pull_request, list):
        # `on.pull_request: [foo, bar]` form (no map) — no path filter
        # can be attached in this shape.
        return []
    paths_ignore = pull_request.get("paths-ignore") or []
    assert isinstance(paths_ignore, list), (
        "pull_request.paths-ignore must be a YAML sequence (list of "
        f"strings), got {type(paths_ignore).__name__}"
    )
    return [str(p) for p in paths_ignore]


# Canonical documentation carve-out, mirrored verbatim from
# `.github/workflows/rust-tests.yml::on.pull_request.paths-ignore`
# (Issue #3367, PR #3369, Wave 8). The carve-out is a single
# repo-wide invariant — every heavy workflow that path-filters docs PRs
# must declare the same list so a docs PR does not slip past one and
# trip another.
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
# ``paths-ignore`` entry.
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
    """Sanity: the workflow file is at the expected location.

    The test fails loudly if the workflow is renamed/moved so the
    follow-up invariants do not silently pass on a missing target.
    """
    assert WORKFLOW_PATH.is_file(), (
        f"Expected workflow at {WORKFLOW_PATH}, but the file does not "
        f"exist. The Issue #3662 fix is anchored to "
        f"`.github/workflows/code-coverage.yml`."
    )


def test_code_coverage_workflow_has_paths_ignore():
    """``on.pull_request.paths-ignore`` MUST be declared.

    Pre-#3662 the workflow declared only ``on.pull_request:`` (a bare
    scalar) so every PR — including docs-only ones like #3660 — ran
    the full cargo-llvm-cov pipeline and could misfire the 1%
    ratchet on measurement noise. The fix is to attach a
    ``paths-ignore`` block listing the canonical documentation globs.
    """
    workflow = _load_workflow(WORKFLOW_PATH)
    paths_ignore = _paths_ignore(workflow)
    assert paths_ignore, (
        ".github/workflows/code-coverage.yml is missing "
        "`on.pull_request.paths-ignore:`. The Code Coverage Gate "
        "(Issue #1932) cannot fire on docs-only PRs (Issue #3662) "
        "without this block — see "
        ".github/workflows/rust-tests.yml::on.pull_request.paths-ignore "
        "for the Wave 8 / #3367 pattern."
    )


def test_code_coverage_workflow_paths_ignore_contains_docs_globs():
    """Every canonical docs glob is present in the carve-out.

    Documents the invariant that a docs PR touching any of these
    paths is exempted from the gate. Removing any entry re-opens the
    noise-floor failure mode that PR #3660 triggered.
    """
    workflow = _load_workflow(WORKFLOW_PATH)
    paths_ignore = _paths_ignore(workflow)
    missing = [
        g for g in EXPECTED_DOCS_GLOBS if g not in paths_ignore
    ]
    assert not missing, (
        ".github/workflows/code-coverage.yml::on.pull_request."
        "paths-ignore is missing the canonical documentation globs: "
        f"{missing}. The Issue #3662 carve-out must match the "
        ".github/workflows/rust-tests.yml::on.pull_request."
        "paths-ignore set so docs-only PRs are skipped by both "
        "heavy workflows uniformly."
    )


def test_code_coverage_workflow_paths_ignore_in_sync_with_rust_tests():
    """The Code Coverage and Rust Tests path-filters stay in lock-step.

    The docs-only carve-out is a single repo-wide invariant, not a
    per-workflow one. Drift between the two path-filter sets would
    mean a docs PR is filtered out of one heavy workflow but still
    triggers another, defeating the Wave 8 / #3367 decoupling.
    """
    code_cov = _load_workflow(WORKFLOW_PATH)
    rust_tests = _load_workflow(RUST_TESTS_WORKFLOW_PATH)
    code_cov_set = set(_paths_ignore(code_cov))
    rust_tests_set = set(_paths_ignore(rust_tests))

    missing_from_code_cov = rust_tests_set - code_cov_set
    assert not missing_from_code_cov, (
        ".github/workflows/code-coverage.yml::on.pull_request."
        f"paths-ignore is missing entries that "
        f".github/workflows/rust-tests.yml::on.pull_request."
        f"paths-ignore declares: {sorted(missing_from_code_cov)}. "
        "The two heavy workflows must agree on the docs-only carve-out "
        "so a docs PR is either skipped by both or runs through both "
        "(never a mix)."
    )


def test_code_coverage_workflow_does_not_ignore_source_paths():
    """Defensive: the carve-out MUST NOT swallow source paths.

    A regression where ``src/**`` ends up under ``paths-ignore`` would
    silently exempt every code-changing PR from coverage. Use
    fnmatch-style glob containment to catch both literal entries
    (e.g. ``src/**``) and over-broad patterns (e.g. ``**/*.rs``).
    """
    import fnmatch

    workflow = _load_workflow(WORKFLOW_PATH)
    paths_ignore = _paths_ignore(workflow)

    leaked: list[str] = []
    for forbidden in FORBIDDEN_DOCS_SKIP_GLOBS:
        for entry in paths_ignore:
            if fnmatch.fnmatchcase(forbidden, entry):
                leaked.append(f"{forbidden!r} matched by {entry!r}")

    assert not leaked, (
        ".github/workflows/code-coverage.yml::on.pull_request."
        "paths-ignore swallowed a code-side pattern; the docs-only "
        "carve-out must NEVER exempt source-side changes. Findings: "
        f"{leaked}. The Code Coverage Gate must still run on every "
        "PR that touches src/, fluxion-core/src/, Cargo.toml, "
        "Cargo.lock, scripts/, or *.rs files."
    )


# ---------------------------------------------------------------------------
# Synthetic-fixture invariants — pin the matcher semantics
# ---------------------------------------------------------------------------


def test_synthetic_workflow_without_paths_ignore_is_rejected(tmp_path):
    """A workflow that omits ``paths-ignore`` triggers the assertion.

    Pins the production assertion that the carve-out is required, not
    optional, on the Code Coverage workflow.
    """
    body = (
        "name: Code Coverage\n"
        "\n"
        "on:\n"
        "  pull_request:\n"
        "  push:\n"
        "    branches: [develop]\n"
        "\n"
        "jobs:\n"
        "  coverage:\n"
        "    runs-on: ubuntu-latest\n"
    )
    workflow_path = _write_workflow(tmp_path, body)
    workflow = _load_workflow(workflow_path)
    assert _paths_ignore(workflow) == [], (
        "Synthetic fixture: a workflow with bare `on.pull_request:` "
        "must yield an empty paths-ignore list (the regression we "
        "are guarding against is exactly this empty case)."
    )
    # And the production assertion fires:
    assert not _paths_ignore(workflow), (
        "production gate: empty paths-ignore is rejected"
    )


def test_synthetic_workflow_missing_canonical_root_md_is_rejected(tmp_path):
    """A carve-out missing ``KNOWN_ISSUES.md`` is rejected.

    Pins that the canonical root-markdown enumeration is the
    contract — removing a single root ``.md`` entry from the carve-out
    re-opens the noise-floor failure mode for that file.
    """
    partial = [
        "docs/**",
        "**.md",
        "**.mdx",
        "README.md",
        "ARCHITECTURE.md",
        "AGENTS.md",
        "RULES.md",
        "CODEBASE_MAP.md",
        "SCORECARD.md",
        # KNOWN_ISSUES.md deliberately omitted
        "ASHRAE140_RESULTS.md",
        "CONTRIBUTING.md",
    ]
    body = (
        "name: Code Coverage\n"
        "\n"
        "on:\n"
        "  pull_request:\n"
        "    paths-ignore:\n"
        + "".join(f'      - "{p}"\n' for p in partial)
        + "\n"
        "  push:\n"
        "    branches: [develop]\n"
        "\n"
        "jobs:\n"
        "  coverage:\n"
        "    runs-on: ubuntu-latest\n"
    )
    workflow_path = _write_workflow(tmp_path, body)
    workflow = _load_workflow(workflow_path)
    paths_ignore = _paths_ignore(workflow)

    missing = [
        g for g in EXPECTED_DOCS_GLOBS if g not in paths_ignore
    ]
    assert missing == ["KNOWN_ISSUES.md"], (
        "Synthetic fixture: the production assertion must surface "
        f"the missing entry; got missing={missing}"
    )


def test_synthetic_workflow_with_src_in_paths_ignore_is_rejected(tmp_path):
    """A carve-out containing ``src/**`` triggers the source-leak check.

    Pins that adding ``src/**`` (or any source-side pattern) to
    ``paths-ignore`` re-opens a silent coverage blind-spot: a
    code-changing PR would skip the gate entirely.
    """
    import fnmatch

    body = (
        "name: Code Coverage\n"
        "\n"
        "on:\n"
        "  pull_request:\n"
        "    paths-ignore:\n"
        '      - "docs/**"\n'
        '      - "**.md"\n'
        '      - "src/**"\n'  # regression: code-side pattern
        "\n"
        "  push:\n"
        "    branches: [develop]\n"
        "\n"
        "jobs:\n"
        "  coverage:\n"
        "    runs-on: ubuntu-latest\n"
    )
    workflow_path = _write_workflow(tmp_path, body)
    workflow = _load_workflow(workflow_path)
    paths_ignore = _paths_ignore(workflow)

    leaked = []
    for forbidden in FORBIDDEN_DOCS_SKIP_GLOBS:
        for entry in paths_ignore:
            if fnmatch.fnmatchcase(forbidden, entry):
                leaked.append(f"{forbidden!r} matched by {entry!r}")
    assert leaked, (
        "Synthetic fixture: planting `src/**` in paths-ignore MUST "
        "be detected by the source-leak guard. The fixture did not "
        "trip the guard — the matcher regressed."
    )


def test_paths_ignore_accepts_bare_pull_request_form(tmp_path):
    """``on.pull_request:`` (bare scalar) yields an empty list.

    Pins the parser branch that returns an empty list when
    ``on.pull_request`` is declared as a bare scalar rather than a
    map. This is the pre-#3662 shape — the regression we are
    guarding against — and the parser must not crash on it.
    """
    body = (
        "name: Code Coverage\n"
        "\n"
        "on:\n"
        "  pull_request:\n"
        "  push:\n"
        "    branches: [develop]\n"
        "\n"
        "jobs:\n"
        "  coverage:\n"
        "    runs-on: ubuntu-latest\n"
    )
    workflow_path = _write_workflow(tmp_path, body)
    workflow = _load_workflow(workflow_path)
    assert _paths_ignore(workflow) == []
