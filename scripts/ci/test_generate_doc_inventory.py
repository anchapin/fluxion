"""
Tests for ``scripts/generate_doc_inventory.py`` -- Issue #2765.

The freshness gate ``check_doc_inventory_fresh.py`` shells out to this
generator and byte-compares the committed ``docs/doc-inventory.md`` against
its output, so the generator's table builder + marker-substitution are the
load-bearing pieces of that gate. These tests pin:
  * ``has_seven_line_summary`` (the ✅/❌ classifier),
  * ``build_inventory_table`` (row enumeration + status emoji),
  * ``regenerate`` (marker detection, drift rewrite, idempotence, error paths).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPT_NAME = "generate_doc_inventory"


@pytest.fixture
def gen(load_script):
    return load_script(SCRIPT_NAME)


GOOD_DOC = (
    "# Title\n"
    "\n"
    "Line 3 summary content here.\n"
    "Line 4 summary content here.\n"
    "Line 5 summary content here.\n"
    "Line 6 summary content here.\n"
    "Line 7 summary content here.\n"
    "Line 8 summary content here.\n"
    "\n"
    "Body.\n"
)


def _redirect(gen, tmp_path, monkeypatch) -> None:
    # ``INVENTORY_FILE`` lives OUTSIDE ``DOCS_ROOT`` so the generator does not
    # self-enumerate its own output (the real committed ``docs/doc-inventory.md``
    # sidesteps that feedback loop by already carrying a 7-line summary; keeping
    # the file out of the enumerated tree makes the test hermetic).
    monkeypatch.setattr(gen, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(gen, "DOCS_ROOT", tmp_path / "docs")
    monkeypatch.setattr(gen, "INVENTORY_FILE", tmp_path / "doc-inventory.md")


# ---------------------------------------------------------------------------
# has_seven_line_summary
# ---------------------------------------------------------------------------


def test_has_summary_true_for_good_doc(gen, tmp_path):
    f = tmp_path / "g.md"
    f.write_text(GOOD_DOC, encoding="utf-8")
    assert gen.has_seven_line_summary(f) is True


def test_has_summary_false_for_missing_or_short(gen, tmp_path):
    assert gen.has_seven_line_summary(tmp_path / "nope.md") is False
    short = tmp_path / "s.md"
    short.write_text("# T\nx\n", encoding="utf-8")
    assert gen.has_seven_line_summary(short) is False


# ---------------------------------------------------------------------------
# build_inventory_table
# ---------------------------------------------------------------------------


def test_build_table_header_only_when_no_docs(gen, tmp_path, monkeypatch):
    _redirect(gen, tmp_path, monkeypatch)
    (tmp_path / "docs").mkdir()
    table = gen.build_inventory_table()
    assert table.startswith("| Doc | Path | Status |")
    assert table.count("\n") == 1  # header + separator, no data rows


def test_build_table_marks_good_and_bad_docs(gen, tmp_path, monkeypatch):
    _redirect(gen, tmp_path, monkeypatch)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "good.md").write_text(GOOD_DOC, encoding="utf-8")
    (docs / "bad.md").write_text("# T\nx\n", encoding="utf-8")
    table = gen.build_inventory_table()
    assert "good.md" in table
    assert "bad.md" in table
    good_row = [ln for ln in table.splitlines() if "good.md" in ln][0]
    bad_row = [ln for ln in table.splitlines() if "bad.md" in ln][0]
    assert "✅" in good_row
    assert "❌" in bad_row


# ---------------------------------------------------------------------------
# regenerate
# ---------------------------------------------------------------------------

_INVENTORY_TEMPLATE = (
    "# Doc Inventory\n"
    "\n"
    "<!-- BEGIN AUTO-GENERATED INVENTORY -->\n"
    "OLD CONTENT\n"
    "<!-- END AUTO-GENERATED INVENTORY -->\n"
    "\n"
    "Trailing prose.\n"
)


def test_regenerate_returns_one_when_inventory_missing(gen, tmp_path, monkeypatch):
    _redirect(gen, tmp_path, monkeypatch)
    (tmp_path / "docs").mkdir()
    assert gen.regenerate() == 1


def test_regenerate_returns_one_when_markers_absent(gen, tmp_path, monkeypatch):
    _redirect(gen, tmp_path, monkeypatch)
    gen.INVENTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    gen.INVENTORY_FILE.write_text("# No markers here\n", encoding="utf-8")
    assert gen.regenerate() == 1


def test_regenerate_rewrites_block_when_drifted(gen, tmp_path, monkeypatch):
    _redirect(gen, tmp_path, monkeypatch)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "good.md").write_text(GOOD_DOC, encoding="utf-8")
    gen.INVENTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    gen.INVENTORY_FILE.write_text(_INVENTORY_TEMPLATE, encoding="utf-8")
    assert gen.regenerate() == 0
    result = gen.INVENTORY_FILE.read_text(encoding="utf-8")
    assert "OLD CONTENT" not in result
    assert "good.md" in result
    # Intro + trailing prose preserved outside the marker block.
    assert result.startswith("# Doc Inventory")
    assert "Trailing prose." in result
    assert gen.BEGIN_MARKER in result and gen.END_MARKER in result


def test_regenerate_is_idempotent(gen, tmp_path, monkeypatch, capsys):
    _redirect(gen, tmp_path, monkeypatch)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "good.md").write_text(GOOD_DOC, encoding="utf-8")
    gen.INVENTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    gen.INVENTORY_FILE.write_text(_INVENTORY_TEMPLATE, encoding="utf-8")
    # First run rewrites the drifted block.
    assert gen.regenerate() == 0
    snap = gen.INVENTORY_FILE.read_text(encoding="utf-8")
    # Second run detects no change -> exit 0, file untouched.
    assert gen.regenerate() == 0
    assert gen.INVENTORY_FILE.read_text(encoding="utf-8") == snap
    assert "already up to date" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Git-tracked enumeration (Issue #2961)
# ---------------------------------------------------------------------------
#
# The generator must read its `docs/**/*.md` list from the git index, not
# the filesystem walk.  This pins two regressions:
#   1. Walked files that match `.gitignore` (``**/*_PLAN.md``,
#      ``**/*_ANALYSIS.md``) must NOT appear in the inventory.  The
#      filesystem walk enumeration would pick them up on a local working
#      tree, but CI's fresh checkout never has them — exactly the bug
#      that broke PR #2957.
#   2. The generator must fall back to a filesystem walk when the
#      working tree is not a git repo (e.g. the ``_redirect`` harness
#      uses ``tmp_path`` instead of the real ``REPO_ROOT``).  A trivial
#      in-test git repo lets us assert that the tracked path is
#      preferred and the gitignored file is excluded.


def _init_git_repo(repo_root: Path) -> None:
    """Initialize a git repo with local user identity and the issue-2961 gitignore."""
    repo_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-q", "--initial-branch=main"],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo_root,
        check=True,
    )
    # Mirror the production `.gitignore` patterns that issue #2961 fixes.
    (repo_root / ".gitignore").write_text(
        "**/*_PLAN.md\n**/*_ANALYSIS.md\n", encoding="utf-8"
    )
    subprocess.run(
        ["git", "add", ".gitignore"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )


def test_build_table_excludes_gitignored_plan_file(gen, tmp_path, write_file, monkeypatch):
    """Issue #2961: ``**/*_PLAN.md`` is gitignored → must not appear in inventory."""
    _init_git_repo(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    # A tracked good summary file (will be committed).
    write_file(docs / "good.md", GOOD_DOC)
    # A gitignored file — present on disk, tracked NEVER (matches .gitignore).
    write_file(docs / "_TEST_PLAN.md", "# Test plan\n")
    subprocess.run(
        ["git", "add", "docs/good.md"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    monkeypatch.setattr(gen, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(gen, "DOCS_ROOT", docs)
    monkeypatch.setattr(gen, "INVENTORY_FILE", tmp_path / "doc-inventory.md")

    table = gen.build_inventory_table()

    assert "good.md" in table, "tracked good.md should appear in inventory"
    assert "_TEST_PLAN.md" not in table, (
        "gitignored _TEST_PLAN.md must NOT appear in inventory (issue #2961)"
    )


def test_build_table_excludes_gitignored_analysis_file(gen, tmp_path, write_file, monkeypatch):
    """Companion case: ``**/*_ANALYSIS.md`` is also gitignored."""
    _init_git_repo(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    write_file(docs / "good.md", GOOD_DOC)
    write_file(docs / "_TEST_ANALYSIS.md", "# Test analysis\n")
    subprocess.run(
        ["git", "add", "docs/good.md"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    monkeypatch.setattr(gen, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(gen, "DOCS_ROOT", docs)
    monkeypatch.setattr(gen, "INVENTORY_FILE", tmp_path / "doc-inventory.md")

    table = gen.build_inventory_table()

    assert "good.md" in table
    assert "_TEST_ANALYSIS.md" not in table, (
        "gitignored _TEST_ANALYSIS.md must NOT appear in inventory (issue #2961)"
    )


def test_list_tracked_docs_returns_none_when_not_git_repo(gen, tmp_path):
    """Non-git directories (e.g. synthetic test roots) trigger the fallback."""
    # tmp_path is a normal directory, not a git repo.
    assert gen._list_tracked_docs(tmp_path) is None


def test_list_tracked_docs_returns_paths_in_git_repo(gen, tmp_path, write_file):
    """Happy-path: returns the tracked docs/**/*.md paths."""
    _init_git_repo(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    write_file(docs / "a.md", "# A\n")
    write_file(docs / "subdir" / "b.md", "# B\n")
    write_file(docs / "PLAN.md", "# plan\n")  # gitignored → never tracked
    subprocess.run(
        ["git", "add", "docs/a.md", "docs/subdir/b.md"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    tracked = gen._list_tracked_docs(tmp_path)
    assert tracked is not None
    rels = sorted(p.relative_to(tmp_path).as_posix() for p in tracked)
    assert rels == ["docs/a.md", "docs/subdir/b.md"]


def test_list_tracked_docs_handles_top_level_and_nested_files(
    gen, tmp_path, write_file
):
    """Regression: a single ``docs/**/*.md`` glob in git ls-files misses
    top-level files (git's pathspec semantics treat ``**`` as ≥1
    intermediate component).  The joint pattern in the generator must
    catch both ``docs/a.md`` and ``docs/sub/b.md``.
    """
    _init_git_repo(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    write_file(docs / "top.md", "# Top\n")
    write_file(docs / "sub" / "nested.md", "# Nested\n")
    subprocess.run(
        ["git", "add", "docs/top.md", "docs/sub/nested.md"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    tracked = gen._list_tracked_docs(tmp_path)
    assert tracked is not None
    rels = sorted(p.relative_to(tmp_path).as_posix() for p in tracked)
    assert rels == ["docs/sub/nested.md", "docs/top.md"]
