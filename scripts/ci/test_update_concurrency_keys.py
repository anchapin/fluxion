"""Tests for ``scripts/update_concurrency_keys.py`` -- Issue #4206.

``update_concurrency_keys.py`` is the *writer* half of the ADR-0015
concurrency contract: it rewrites the ``concurrency:`` block in every
``.github/workflows/*.yml`` to the per-``head_sha`` template.
``scripts/ci/test_check_concurrency_keys.py`` covers only the *reader*
(``check_concurrency_keys.py``). These tests bind the two: every
synthetic fixture the mutator rewrites must (a) still parse as valid
YAML, (b) be byte-identical on a second run (the script's docstring
claims idempotence), (c) be reported clean by
``check_concurrency_keys.check_workflow`` -- the gate that is the only
CI protection against a misconfigured writer, (d) keep any
per-workflow ``group:`` prefix byte-exact through the rewrite, and
(e) honor ``main()``'s documented exit-code contract.

Mirrors the ``load_script`` + ``tmp_path`` mock-repo pattern from
``test_check_branch_protection_diff.py`` (#3426). All fixtures are
hand-built strings -- never generated from the script's own template
constants -- so a regression in the script's template or regexes
surfaces instead of being hidden by re-serialization (same rationale
as the #3426 / #3444 companions).

Safety: the mutator derives its workflows directory from module-level
constants, so every test redirects ``REPO_ROOT`` / ``WORKFLOWS_DIR``
at a ``tmp_path`` mock repo. No test ever points the mutator at the
real ``.github/workflows/`` tree.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

SCRIPT_NAME = "update_concurrency_keys"
CHECKER_NAME = "check_concurrency_keys"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def mutator(load_script):
    """Freshly-loaded copy of ``scripts/update_concurrency_keys.py``."""
    return load_script(SCRIPT_NAME)


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of ``scripts/check_concurrency_keys.py``."""
    return load_script(CHECKER_NAME)


def _point_at_tmp(module, monkeypatch, tmp_path: Path) -> Path:
    """Redirect a script module's path constants at a ``tmp_path`` mock
    repo and return its (possibly not-yet-created) workflows dir.

    Both scripts derive ``WORKFLOWS_DIR`` from ``__file__``; without this
    redirect any ``main()``/``update_workflow`` drive would touch the
    real repo tree.
    """
    workflows = tmp_path / ".github" / "workflows"
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "WORKFLOWS_DIR", workflows)
    return workflows


def _write_workflow(tmp_path: Path, name: str, text: str) -> Path:
    """Write a synthetic workflow into the mock repo's workflows dir."""
    target = tmp_path / ".github" / "workflows" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def _mutate_all(mutator, tmp_path: Path, *, dry_run: bool = False) -> None:
    """Drive ``update_workflow`` over every fixture in the mock repo."""
    workflows = tmp_path / ".github" / "workflows"
    for path in sorted(workflows.glob("*.yml")):
        if path.is_file():
            mutator.update_workflow(path, dry_run=dry_run)


def _mutate_and_load(mutator, text: str) -> str:
    """Run the mutator over an in-memory fixture (no disk I/O)."""
    new_text, replaced = mutator.replace_concurrency_block(text)
    if not replaced:
        new_text, _ = mutator.insert_concurrency_block(new_text)
    return new_text


# ---------------------------------------------------------------------------
# Synthetic fixtures (hand-built; see module docstring)
# ---------------------------------------------------------------------------

# Legacy shape: bare `${{ github.workflow }}-${{ github.ref }}` group and
# unconditional cancel -- the pre-ADR-0015 block every real workflow had.
_LEGACY_BARE_WORKFLOW = (
    "name: CI\n"
    "\n"
    "on:\n"
    "  push:\n"
    "    branches: [develop]\n"
    "\n"
    "concurrency:\n"
    "  group: ${{ github.workflow }}-${{ github.ref }}\n"
    "  cancel-in-progress: true\n"
    "\n"
    "jobs:\n"
    "  build:\n"
    "    runs-on: ubuntu-latest\n"
)

# Legacy shape with a per-workflow `group:` prefix (cf. the real
# `ashrae-140-strict-energy-gate.yml` style): the literal prefix must
# survive the rewrite.
_LEGACY_PREFIXED_WORKFLOW = (
    "name: ASHRAE 140 Strict Energy Gate\n"
    "\n"
    "on:\n"
    "  pull_request:\n"
    "\n"
    "concurrency:\n"
    "  group: ashrae-140-strict-energy-gate-${{ github.ref }}\n"
    "  cancel-in-progress: true\n"
    "\n"
    "jobs:\n"
    "  gate:\n"
    "    runs-on: ubuntu-latest\n"
)

# No `concurrency:` block at all: the canonical insert goes immediately
# after the `on:` block, before `jobs:`.
_NO_BLOCK_WORKFLOW = (
    "name: CI\n"
    "\n"
    "on:\n"
    "  push:\n"
    "    branches: [develop]\n"
    "\n"
    "jobs:\n"
    "  build:\n"
    "    runs-on: ubuntu-latest\n"
)

# No block, but a top-level comment sits between `on:` and `jobs:`: the
# mutator's comment-skipping loop must land `concurrency:` *after* the
# comments (a naive `on:`-anchored insert would wedge it between `on:`
# and the comment, breaking the file's documented layout).
_NO_BLOCK_WITH_COMMENTS_WORKFLOW = (
    "name: CI\n"
    "\n"
    "on:\n"
    "  pull_request:\n"
    "\n"
    "# Keep the concurrency block adjacent to `on:` per ADR-0015.\n"
    "# Future editors: do not move this comment.\n"
    "\n"
    "jobs:\n"
    "  build:\n"
    "    runs-on: ubuntu-latest\n"
)

# Degenerate workflow with no `on:` at all: falls back to inserting
# after the `name:` line.
_NAME_ONLY_WORKFLOW = (
    "name: Legacy name-only workflow\n\njobs:\n  build:\n    runs-on: ubuntu-latest\n"
)

# Already carries the ADR-0015 template: every drive must be a no-op.
_ALREADY_UPDATED_WORKFLOW = (
    "name: CI\n"
    "\n"
    "on:\n"
    "  push:\n"
    "\n"
    "concurrency:\n"
    "  group: >-\n"
    "    ${{ github.workflow }}-\n"
    "    ${{\n"
    "      github.event_name == 'pull_request' &&\n"
    "      github.event.pull_request.head.sha\n"
    "      || github.ref\n"
    "    }}\n"
    "  cancel-in-progress: >-\n"
    "    ${{\n"
    "      github.event_name == 'push'\n"
    "      && contains('refs/heads/main,refs/heads/develop', github.ref)\n"
    "    }}\n"
    "\n"
    "jobs:\n"
    "  build:\n"
    "    runs-on: ubuntu-latest\n"
)

# Every fixture the mutator is expected to touch (the already-updated
# one must survive untouched).
_FIXTURES = {
    "legacy-bare.yml": _LEGACY_BARE_WORKFLOW,
    "legacy-prefixed.yml": _LEGACY_PREFIXED_WORKFLOW,
    "no-block.yml": _NO_BLOCK_WORKFLOW,
    "no-block-comments.yml": _NO_BLOCK_WITH_COMMENTS_WORKFLOW,
    "name-only.yml": _NAME_ONLY_WORKFLOW,
    "already-updated.yml": _ALREADY_UPDATED_WORKFLOW,
}


# ---------------------------------------------------------------------------
# replace_concurrency_block / insert_concurrency_block unit behavior
# ---------------------------------------------------------------------------


def test_replace_rewrites_legacy_block_to_template(mutator):
    """The legacy `github.ref` + `true` block is replaced in full."""
    new_text, changed = mutator.replace_concurrency_block(_LEGACY_BARE_WORKFLOW)
    assert changed is True
    assert "github.event.pull_request.head.sha" in new_text
    assert "cancel-in-progress: true" not in new_text
    assert "group: ${{ github.workflow }}-${{ github.ref }}\n" not in (new_text)


def test_replace_is_noop_on_already_updated_block(mutator):
    """Idempotence at the block level: the ADR-0015 template (marker
    match) is returned byte-identical with ``changed=False``."""
    new_text, changed = mutator.replace_concurrency_block(_ALREADY_UPDATED_WORKFLOW)
    assert changed is False
    assert new_text == _ALREADY_UPDATED_WORKFLOW


def test_replace_preserves_per_workflow_group_prefix(mutator):
    """The literal text before `${{ github.ref }}` is preserved on the
    first line of the new folded `group:` scalar."""
    new_text, changed = mutator.replace_concurrency_block(_LEGACY_PREFIXED_WORKFLOW)
    assert changed is True
    assert re.search(r"^    ashrae-140-strict-energy-gate-$", new_text, re.MULTILINE), (
        "per-workflow prefix lost in rewrite"
    )


def test_insert_places_block_after_on_block(mutator):
    """A workflow with no `concurrency:` key gets the template between
    `on:` and `jobs:` -- the canonical location."""
    new_text, changed = mutator.insert_concurrency_block(_NO_BLOCK_WORKFLOW)
    assert changed is True
    assert (
        new_text.index("branches: [develop]")
        < new_text.index("concurrency:")
        < new_text.index("jobs:")
    )


def test_insert_skips_comment_block_after_on(mutator):
    """The comment-skipping loop lands the block *after* top-level
    comments following `on:` (a naive `on:`-anchored regex would wedge
    it between `on:` and the comments)."""
    new_text, changed = mutator.insert_concurrency_block(
        _NO_BLOCK_WITH_COMMENTS_WORKFLOW
    )
    assert changed is True
    assert (
        new_text.index("# Keep the concurrency block adjacent")
        < new_text.index("concurrency:")
        < new_text.index("jobs:")
    )


def test_insert_falls_back_to_after_name(mutator):
    """With no `on:` block at all, the template goes after `name:`."""
    new_text, changed = mutator.insert_concurrency_block(_NAME_ONLY_WORKFLOW)
    assert changed is True
    assert (
        new_text.index("name: Legacy name-only workflow")
        < new_text.index("concurrency:")
        < new_text.index("jobs:")
    )


def test_insert_is_noop_when_block_exists(mutator):
    """`insert_concurrency_block` never touches a workflow that already
    declares `concurrency:` (replace owns that path)."""
    new_text, changed = mutator.insert_concurrency_block(_LEGACY_BARE_WORKFLOW)
    assert changed is False
    assert new_text == _LEGACY_BARE_WORKFLOW


# ---------------------------------------------------------------------------
# (a) Every mutated fixture still parses as valid YAML
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,text", list(_FIXTURES.items()))
def test_mutated_fixture_parses_as_valid_yaml(mutator, name, text):
    """The property the regexes are most likely to break: the writer's
    output must round-trip through a YAML loader with the ADR-0015
    `concurrency:` mapping intact. The content assertions tie validity
    to the template, not just to parseability."""
    mutated = _mutate_and_load(mutator, text)
    doc = yaml.safe_load(mutated)
    assert isinstance(doc, dict), name
    assert "concurrency" in doc, name
    group = doc["concurrency"]["group"]
    cancel = doc["concurrency"]["cancel-in-progress"]
    assert "github.event.pull_request.head.sha" in group, name
    assert "|| github.ref" in group, name
    assert "github.event_name == 'push'" in cancel, name
    assert "contains('refs/heads/main,refs/heads/develop'" in cancel, name


def test_yaml_loader_rejects_malformed_output():
    """Sanity check on the assertion above: the loader genuinely raises
    on broken YAML, so a mutator regression that emits garbage cannot
    slip through a vacuous assertion."""
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load("concurrency:\n  group: [unclosed\n")


# ---------------------------------------------------------------------------
# (b) Idempotence: two runs produce byte-identical output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,text", list(_FIXTURES.items()))
def test_mutator_is_idempotent(mutator, name, text):
    """The script's docstring claims re-running is a no-op. The second
    drive must return `changed=False` and byte-identical text."""
    once = _mutate_and_load(mutator, text)
    twice = _mutate_and_load(mutator, once)
    assert twice == once, name


def test_update_workflow_second_drive_reports_up_to_date(
    mutator, monkeypatch, tmp_path
):
    """End-to-end through `update_workflow` on disk: the first drive
    converges the fixture, the second reports `already up-to-date` and
    leaves the bytes untouched."""
    _point_at_tmp(mutator, monkeypatch, tmp_path)
    target = _write_workflow(tmp_path, "ci.yml", _LEGACY_BARE_WORKFLOW)

    changed, msg = mutator.update_workflow(target, dry_run=False)
    assert (changed, msg) == (True, "replaced")
    converged = target.read_text(encoding="utf-8")

    changed, msg = mutator.update_workflow(target, dry_run=False)
    assert (changed, msg) == (False, "already up-to-date")
    assert target.read_text(encoding="utf-8") == converged


# ---------------------------------------------------------------------------
# (c) Writer/reader agreement: the gate accepts the writer's output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,text", list(_FIXTURES.items()))
def test_gate_accepts_mutated_fixture(
    mutator, checker, monkeypatch, tmp_path, name, text
):
    """For every mutated fixture, `check_concurrency_keys.check_workflow`
    -- the only CI protection against a misconfigured writer -- reports
    zero findings. The writer's output is provably accepted by the gate
    that enforces ADR-0015."""
    _point_at_tmp(checker, monkeypatch, tmp_path)
    target = _write_workflow(tmp_path, name, text)
    mutated = _mutate_and_load(mutator, text)
    target.write_text(mutated, encoding="utf-8")
    assert checker.check_workflow(target) == [], name


# ---------------------------------------------------------------------------
# (d) Per-workflow group: prefix preserved byte-exact
# ---------------------------------------------------------------------------


def test_per_workflow_prefix_survives_full_pipeline(
    mutator, checker, monkeypatch, tmp_path
):
    """The prefixed fixture keeps its exact `ashrae-140-...-` literal
    through rewrite, stays valid YAML, and still passes the gate (the
    gate's marker tests must not be confused by the literal prefix)."""
    _point_at_tmp(checker, monkeypatch, tmp_path)
    target = _write_workflow(tmp_path, "ashrae.yml", _LEGACY_PREFIXED_WORKFLOW)
    mutated = _mutate_and_load(mutator, _LEGACY_PREFIXED_WORKFLOW)
    target.write_text(mutated, encoding="utf-8")

    assert re.search(r"^    ashrae-140-strict-energy-gate-$", mutated, re.MULTILINE)
    doc = yaml.safe_load(mutated)
    assert doc["concurrency"]["group"].startswith("ashrae-140-strict-energy-gate-")
    assert checker.check_workflow(target) == []


def test_extract_group_prefix_falls_back_to_workflow_name(mutator):
    """A legacy block with no literal prefix (bare
    `${{ github.workflow }}-${{ github.ref }}`) yields the workflow-name
    prefix, not an empty string."""
    block = (
        "concurrency:\n"
        "  group: ${{ github.workflow }}-${{ github.ref }}\n"
        "  cancel-in-progress: true\n"
    )
    assert mutator.extract_group_prefix(block) == "${{ github.workflow }}-"


# ---------------------------------------------------------------------------
# (e) main() exit-code contract: 0 converged / 1 failure / 2 env error
# ---------------------------------------------------------------------------


def _run_main(mutator, monkeypatch, argv):
    monkeypatch.setattr(mutator.sys, "argv", argv)
    return mutator.main()


def test_main_exit_0_when_all_workflows_converge(
    mutator, monkeypatch, tmp_path, capsys
):
    """Mixed repo (legacy replace + missing-block insert + already
    current): exit 0, every file converged, and a second run is fully
    `already up-to-date`."""
    _point_at_tmp(mutator, monkeypatch, tmp_path)
    for name, text in _FIXTURES.items():
        _write_workflow(tmp_path, name, text)

    assert _run_main(mutator, monkeypatch, ["update_concurrency_keys.py"]) == 0
    out = capsys.readouterr().out
    assert "Failures:           0" in out

    # Second run: everything already up-to-date, still exit 0.
    assert _run_main(mutator, monkeypatch, ["update_concurrency_keys.py"]) == 0
    out = capsys.readouterr().out
    assert "Already up-to-date: 6" in out


def test_main_dry_run_lists_changes_without_writing(
    mutator, monkeypatch, tmp_path, capsys
):
    """`--dry-run` reports the planned changes (exit 0) but leaves the
    fixture bytes on disk untouched."""
    _point_at_tmp(mutator, monkeypatch, tmp_path)
    target = _write_workflow(tmp_path, "ci.yml", _LEGACY_BARE_WORKFLOW)
    before = target.read_text(encoding="utf-8")

    code = _run_main(mutator, monkeypatch, ["update_concurrency_keys.py", "--dry-run"])
    assert code == 0
    assert target.read_text(encoding="utf-8") == before
    assert "would replaced" in capsys.readouterr().out


def test_main_workflow_filter_restricts_to_named_file(mutator, monkeypatch, tmp_path):
    """`--workflow` scopes the mutation to one file; the other fixture
    is left byte-identical."""
    _point_at_tmp(mutator, monkeypatch, tmp_path)
    alpha = _write_workflow(tmp_path, "alpha.yml", _LEGACY_BARE_WORKFLOW)
    beta = _write_workflow(tmp_path, "beta.yml", _LEGACY_BARE_WORKFLOW)
    beta_before = beta.read_text(encoding="utf-8")

    code = _run_main(
        mutator,
        monkeypatch,
        ["update_concurrency_keys.py", "--workflow", "alpha.yml"],
    )
    assert code == 0
    assert "github.event.pull_request.head.sha" in alpha.read_text(encoding="utf-8")
    assert beta.read_text(encoding="utf-8") == beta_before


def test_main_exit_1_on_unreadable_workflow(mutator, monkeypatch, tmp_path, capsys):
    """A `*.yml` entry that cannot be read (here: a directory) is a
    per-file failure -> exit 1, while the healthy fixture still
    converges (partial failure, not a total abort)."""
    workflows = _point_at_tmp(mutator, monkeypatch, tmp_path)
    _write_workflow(tmp_path, "good.yml", _LEGACY_BARE_WORKFLOW)
    (workflows / "broken.yml").mkdir(parents=True)

    assert _run_main(mutator, monkeypatch, ["update_concurrency_keys.py"]) == 1
    captured = capsys.readouterr()
    assert "broken.yml" in captured.err
    assert "Failures:           1" in captured.out
    # The healthy fixture still converged despite the sibling failure.
    good = (workflows / "good.yml").read_text(encoding="utf-8")
    assert "github.event.pull_request.head.sha" in good


def test_main_exit_2_when_workflows_dir_missing(mutator, monkeypatch, tmp_path, capsys):
    """A missing `.github/workflows/` is a script error (exit 2), not a
    convergence failure."""
    _point_at_tmp(mutator, monkeypatch, tmp_path)  # dir never created
    assert _run_main(mutator, monkeypatch, ["update_concurrency_keys.py"]) == 2
    assert "not found" in capsys.readouterr().err


def test_main_exit_2_on_unknown_workflow_filter(mutator, monkeypatch, tmp_path, capsys):
    _point_at_tmp(mutator, monkeypatch, tmp_path)
    _write_workflow(tmp_path, "alpha.yml", _LEGACY_BARE_WORKFLOW)
    code = _run_main(
        mutator,
        monkeypatch,
        ["update_concurrency_keys.py", "--workflow", "nope.yml"],
    )
    assert code == 2
    assert "workflow nope.yml not found" in capsys.readouterr().err
