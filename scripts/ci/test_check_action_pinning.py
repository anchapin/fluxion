"""Tests for ``scripts/check_action_pinning.py`` -- Issue #4187.

The gate rejects mutable action refs (``stable`` / ``main`` / ``master`` /
``HEAD`` / ``v1``-``v4``) in workflow files. Issue #4187 extended its scan
to ``.github/actions/**/*.{yml,yaml}``: the repo's local composite actions
carry nested third-party ``uses:`` references that were pinned but
unenforced, and a mutable tag there resolves exactly like one in a
workflow.

The tests below drive the gate against hermetic ``tmp_path`` mock repos
(the ``load_script`` + ``monkeypatch`` pattern from
``test_check_workflow_pin.py``), proving that:

* a tag-pinned ``uses:`` inside a nested composite action fails the gate
  (exit 1, finding names the action file),
* SHA-pinned refs inside composite actions pass (exit 0),
* ``uses:./.github/actions/...`` local references remain allowed,
* nested workflow directories stay covered,
* a missing ``.github`` tree exits 2.

No test depends on the network or on the repo's real workflows; the
end-to-end against the real tree is the direct
``check_action_pinning.py`` invocation in ``scripts-tests.yml``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPT_NAME = "check_action_pinning"

_SHA = "3d3c42e5aac5ba805825da76410c181273ba90b1"

_COMPLIANT_WORKFLOW = (
    "name: ok\n"
    "jobs:\n"
    "  build:\n"
    "    steps:\n"
    f"      - uses: actions/checkout@{_SHA}  # v7.0.1\n"
    "      - uses: ./.github/actions/setup-rust-env\n"
)

_COMPLIANT_ACTION = (
    "name: my-action\n"
    "description: test composite action\n"
    "runs:\n"
    "  using: composite\n"
    "  steps:\n"
    f"      - uses: actions/setup-python@{_SHA}  # v4\n"
)

_TAG_PINNED_ACTION = (
    "name: my-action\n"
    "description: test composite action\n"
    "runs:\n"
    "  using: composite\n"
    "  steps:\n"
    "      - uses: actions/checkout@v4\n"
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of ``scripts/check_action_pinning.py``."""
    return load_script(SCRIPT_NAME)


def _redirect(checker, monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    """Point the gate's module constants at a ``tmp_path`` mock repo and
    return its ``(.github/workflows, .github/actions)`` directories.

    Both directories are created eagerly: the gate treats a missing scan
    root as a script error (exit 2), so every mock repo carries both.
    """
    workflows = tmp_path / ".github" / "workflows"
    actions = tmp_path / ".github" / "actions"
    workflows.mkdir(parents=True, exist_ok=True)
    actions.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "WORKFLOWS_DIR", workflows)
    monkeypatch.setattr(checker, "ACTIONS_DIR", actions)
    return workflows, actions


def _write(root: Path, *parts: str, text: str) -> Path:
    """Write a synthetic file into the mock repo."""
    target = root.joinpath(*parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def _compliant_tree(checker, monkeypatch, tmp_path: Path) -> None:
    """A mock repo whose workflow and composite action are both clean."""
    workflows, actions = _redirect(checker, monkeypatch, tmp_path)
    _write(workflows, "ci.yml", text=_COMPLIANT_WORKFLOW)
    _write(actions, "my-action", "action.yml", text=_COMPLIANT_ACTION)


# ---------------------------------------------------------------------------
# Issue #4187: composite-action scan
# ---------------------------------------------------------------------------


def test_tag_pinned_uses_in_composite_action_fails(checker, monkeypatch,
                                                   tmp_path, capsys):
    """A tag-pinned third-party ``uses:`` nested inside a composite action
    fails the gate -- the action directory does not dodge the scan."""
    workflows, actions = _redirect(checker, monkeypatch, tmp_path)
    _write(workflows, "ci.yml", text=_COMPLIANT_WORKFLOW)
    _write(actions, "my-action", "action.yml", text=_TAG_PINNED_ACTION)

    assert checker.main([]) == 1
    out = capsys.readouterr().out
    assert ".github/actions/my-action/action.yml:" in out
    assert "actions/checkout@v4" in out
    assert "ACTION PINNING FAILED" in out


def test_sha_pinned_uses_in_composite_action_passes(checker, monkeypatch,
                                                   tmp_path, capsys):
    """SHA-pinned ``uses:`` inside composite actions is the compliant
    steady state (matches the real tree's five actions)."""
    _compliant_tree(checker, monkeypatch, tmp_path)

    assert checker.main([]) == 0
    out = capsys.readouterr().out
    assert "ACTION PINNING PASS" in out
    assert ".github/actions/my-action/action.yml" in out


def test_branch_pinned_uses_in_composite_action_fails(checker, monkeypatch,
                                                      tmp_path, capsys):
    """A branch-pinned ref (``@stable``) inside a composite action fails."""
    workflows, actions = _redirect(checker, monkeypatch, tmp_path)
    _write(workflows, "ci.yml", text=_COMPLIANT_WORKFLOW)
    _write(actions, "my-action", "action.yml", text=(
        "name: my-action\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "      - uses: dtolnay/rust-toolchain@stable\n"
    ))

    assert checker.main([]) == 1
    out = capsys.readouterr().out
    assert "dtolnay/rust-toolchain@stable" in out


def test_local_action_reference_still_allowed(checker, monkeypatch,
                                              tmp_path, capsys):
    """``uses:./.github/actions/...`` stays allowed -- a local path carries
    no mutable ``@`` ref (Issue #4187 acceptance criterion)."""
    _compliant_tree(checker, monkeypatch, tmp_path)

    assert checker.main([]) == 0
    out = capsys.readouterr().out
    assert "0 violation(s)" in out


def test_nested_workflow_directory_still_scanned(checker, monkeypatch,
                                                tmp_path, capsys):
    """The recursive workflow scan is intact: a tag-pinned ``uses:`` in a
    nested workflow subdirectory fails the gate."""
    workflows, _ = _redirect(checker, monkeypatch, tmp_path)
    _write(workflows, "nested", "deep", "ci.yml", text=(
        "name: nested\n"
        "jobs:\n"
        "  build:\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
    ))

    assert checker.main([]) == 1
    out = capsys.readouterr().out
    assert ".github/workflows/nested/deep/ci.yml:" in out


def test_yaml_extension_scanned_in_actions(checker, monkeypatch,
                                           tmp_path, capsys):
    """``*.yaml`` (not just ``*.yml``) composite-action files are scanned."""
    workflows, actions = _redirect(checker, monkeypatch, tmp_path)
    _write(workflows, "ci.yml", text=_COMPLIANT_WORKFLOW)
    _write(actions, "my-action", "action.yaml", text=_TAG_PINNED_ACTION)

    assert checker.main([]) == 1
    out = capsys.readouterr().out
    assert ".github/actions/my-action/action.yaml:" in out


def test_missing_dirs_exit_2(checker, monkeypatch, tmp_path, capsys):
    """No ``.github`` tree at all is a script error (exit 2), not a pass."""
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "WORKFLOWS_DIR",
                        tmp_path / ".github" / "workflows")
    monkeypatch.setattr(checker, "ACTIONS_DIR",
                        tmp_path / ".github" / "actions")

    assert checker.main([]) == 2
    err = capsys.readouterr().err
    assert "ERROR" in err
