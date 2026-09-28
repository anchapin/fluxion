"""Tests for ``scripts/check_nextest_doc_drift.py`` -- Issue #4177.

The script requires every ``cargo nextest run`` command documented in
``AGENTS.md`` (command block) or ``docs/**/*.md`` (inline backtick
quotes) to appear in at least one ``.github/workflows/*.yml`` — the
per-PR gate runs a curated subset, so a documented full-workspace
command that no workflow runs is doc fiction (the Issue #4177 drift).

The tests drive the matcher against hermetic ``tmp_path`` mock repos
(the ``load_script`` + ``tmp_path`` pattern from
``test_check_workflow_pin.py``, Issue #3475); the end-to-end against
the real tree is the direct ``check_nextest_doc_drift.py`` invocation
in ``scripts-tests.yml``.

Coverage:

* ``_documented_commands()`` -- AGENTS.md bare-line extraction (with
  ``#`` comment stripping), docs backtick extraction, ``docs/archive``
  exclusion, de-duplication,
* ``check_commands()`` -- findings for absent commands; placeholder /
  allowlist skipping; substring matching against the workflow corpus,
* ``main()`` exit-code contract: 0 clean / 1 drift / 2 script error.

No test depends on the network or on the repo's real docs; ``docs/``
is rebuilt under ``tmp_path`` for every case.
"""

from __future__ import annotations

import pytest

SCRIPT_NAME = "check_nextest_doc_drift"

FULL_CMD = (
    "cargo nextest run --workspace --exclude fluxion-tauri "
    "--all-targets --test-threads=2 --no-fail-fast"
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of ``scripts/check_nextest_doc_drift.py``."""
    return load_script(SCRIPT_NAME)


@pytest.fixture
def mock_repo(checker, monkeypatch, tmp_path):
    """Point the checker's module constants at a ``tmp_path`` mock repo.

    Returns ``(workflows_dir, docs_dir, agents_md)``; the caller writes
    files into them.
    """
    workflows = tmp_path / ".github" / "workflows"
    docs = tmp_path / "docs"
    agents_md = tmp_path / "AGENTS.md"
    workflows.mkdir(parents=True)
    docs.mkdir(parents=True)
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "WORKFLOWS_DIR", workflows)
    monkeypatch.setattr(checker, "AGENTS_MD", agents_md)
    monkeypatch.setattr(checker, "DOCS_DIR", docs)
    return workflows, docs, agents_md


def _run(checker, workflows, text: str) -> list[str]:
    """Write one workflow and run the finding pass."""
    (workflows / "w.yml").write_text(text, encoding="utf-8")
    return checker.check_commands(
        checker._documented_commands(), checker._workflow_corpus()
    )


# ---------------------------------------------------------------------------
# _documented_commands() -- extraction
# ---------------------------------------------------------------------------


def test_agents_md_bare_line_extraction_strips_comment(checker, mock_repo):
    workflows, docs, agents_md = mock_repo
    agents_md.write_text(f"{FULL_CMD}  # canonical CI command\n",
                         encoding="utf-8")
    cmds = checker._documented_commands()
    assert cmds == [(FULL_CMD, "AGENTS.md:1")]


def test_docs_backtick_extraction(checker, mock_repo):
    workflows, docs, agents_md = mock_repo
    (docs / "g.md").write_text(
        "Run `cargo nextest run -p foo --lib` for the lib.\n",
        encoding="utf-8",
    )
    cmds = checker._documented_commands()
    assert cmds == [("cargo nextest run -p foo --lib", "docs/g.md:1")]


def test_docs_archive_excluded(checker, mock_repo):
    workflows, docs, agents_md = mock_repo
    archived = docs / "archive"
    archived.mkdir()
    (archived / "old.md").write_text(
        "Run `cargo nextest run --workspace --nope`.\n", encoding="utf-8"
    )
    assert checker._documented_commands() == []


def test_duplicate_commands_deduplicated(checker, mock_repo):
    workflows, docs, agents_md = mock_repo
    agents_md.write_text(f"{FULL_CMD}\n", encoding="utf-8")
    (docs / "g.md").write_text(f"Also `{FULL_CMD}`.\n", encoding="utf-8")
    cmds = checker._documented_commands()
    assert len(cmds) == 1
    assert cmds[0][0] == FULL_CMD


# ---------------------------------------------------------------------------
# check_commands() -- findings
# ---------------------------------------------------------------------------


def test_documented_command_in_workflow_passes(checker, mock_repo):
    workflows, docs, agents_md = mock_repo
    agents_md.write_text(f"{FULL_CMD}  # canonical CI command\n",
                         encoding="utf-8")
    findings = _run(checker, workflows, f"steps:\n  - run: {FULL_CMD}\n")
    assert findings == []


def test_documented_command_missing_from_workflows_fails(checker, mock_repo):
    workflows, docs, agents_md = mock_repo
    (docs / "g.md").write_text(
        "Run `cargo nextest run --workspace --no-such-flag`.\n",
        encoding="utf-8",
    )
    findings = _run(checker, workflows, "jobs:\n  t:\n    steps: []\n")
    assert len(findings) == 1
    assert "cargo nextest run --workspace --no-such-flag" in findings[0]
    assert "g.md:1" in findings[0]


def test_placeholder_commands_skipped(checker, mock_repo):
    workflows, docs, agents_md = mock_repo
    (docs / "g.md").write_text(
        "Run `cargo nextest run --workspace --features X`.\n"
        "Or `cargo nextest run` bare.\n",
        encoding="utf-8",
    )
    findings = _run(checker, workflows, "jobs: {}\n")
    assert findings == []


def test_historical_allowlist_skipped(checker, mock_repo):
    workflows, docs, agents_md = mock_repo
    (docs / "g.md").write_text(
        "Audit `cargo nextest run --lib --test-threads=2` x5.\n",
        encoding="utf-8",
    )
    findings = _run(checker, workflows, "jobs: {}\n")
    assert findings == []


def test_multiline_workflow_command_matches(checker, mock_repo):
    """``\\``-continuations in workflow ``run:`` blocks are joined
    before matching, so a wrapped documented command still counts."""
    workflows, docs, agents_md = mock_repo
    agents_md.write_text(f"{FULL_CMD}\n", encoding="utf-8")
    (workflows / "w.yml").write_text(
        "steps:\n"
        "  - run: |\n"
        "      cargo nextest run --workspace --exclude fluxion-tauri \\\n"
        "        --all-targets --test-threads=2 --no-fail-fast\n",
        encoding="utf-8",
    )
    findings = checker.check_commands(
        checker._documented_commands(), checker._workflow_corpus()
    )
    assert findings == []


# ---------------------------------------------------------------------------
# main() -- exit-code contract
# ---------------------------------------------------------------------------


def test_main_returns_0_when_no_drift(checker, mock_repo):
    workflows, docs, agents_md = mock_repo
    agents_md.write_text(f"{FULL_CMD}\n", encoding="utf-8")
    (workflows / "w.yml").write_text(f"run: {FULL_CMD}\n", encoding="utf-8")
    assert checker.main([]) == 0


def test_main_returns_1_on_drift(checker, mock_repo, capsys):
    workflows, docs, agents_md = mock_repo
    (docs / "g.md").write_text(
        "Run `cargo nextest run --workspace --no-such-flag`.\n",
        encoding="utf-8",
    )
    (workflows / "w.yml").write_text("jobs: {}\n", encoding="utf-8")
    assert checker.main([]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_main_returns_2_when_workflows_dir_missing(
    checker, monkeypatch, tmp_path
):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        checker, "WORKFLOWS_DIR", tmp_path / ".github" / "workflows"
    )
    assert checker.main([]) == 2
