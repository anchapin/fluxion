from __future__ import annotations

import pytest

SCRIPT_NAME = "check_doc_link_integrity"


@pytest.fixture
def checker(load_script):
    return load_script(SCRIPT_NAME)


def test_extract_and_filter_references(checker, tmp_path):
    text = "[ok](README.md) [remote](https://example.com) <docs/API.md>\n```\n[ignored](missing.md)\n```"
    refs = checker.extract_references(text, tmp_path / "README.md")
    assert refs == [("README.md", 1), ("https://example.com", 1), ("docs/API.md", 1)]
    assert checker.looks_like_path("README.md")
    assert not checker.looks_like_path("plain prose")
    assert checker.is_external_ref("https://example.com")


def test_collect_files_and_resolve_references(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    (tmp_path / "README.md").write_text("ok", encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    doc = docs / "guide.md"
    doc.write_text("[root](README.md) [local](other.md) [remote](https://x.test/a.md)", encoding="utf-8")
    assert checker.collect_markdown_files() == [tmp_path / "README.md", doc]
    assert checker.resolve_reference("README.md", doc)
    assert not checker.resolve_reference("other.md", doc)


def test_main_exit_codes(checker, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    (tmp_path / "README.md").write_text("[ok](README.md)", encoding="utf-8")
    monkeypatch.setattr(checker, "ROOT_ALLOW", ("README.md",))
    assert checker.main() == 0
    assert "PASS" in capsys.readouterr().out
    (tmp_path / "README.md").write_text("[bad](missing.md)", encoding="utf-8")
    assert checker.main() == 1


def test_worktree_subdir_is_skipped(checker, monkeypatch, tmp_path):
    """Issue #3808 — `.planning/worktrees/**` is the parallel-agent
    worktree store (gitignored runtime state). The checker must NOT
    pick up markdown files there, since their cross-references
    resolve against the worktree root and produce false "broken
    reference" failures during normal CI on develop (the worktree
    tree is never present in CI)."""
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    # Redirect the worktree-skip tuple at the per-test tmp_path so
    # WORKTREE_SKIP_PARTS evaluates against THIS repo's planning dir
    # (the script's constant resolves against the real repo at import
    # time, so without this patch the predicate would skip the
    # real .planning/worktrees tree, not the test's).
    monkeypatch.setattr(
        checker, "WORKTREE_SKIP_PARTS", (tmp_path / ".planning" / "worktrees",)
    )
    planning = tmp_path / ".planning"
    planning.mkdir()
    worktree_root = planning / "worktrees" / "t1-lane-split"
    worktree_root.mkdir(parents=True)
    (worktree_root / "README.md").write_text("[npm](../npm/README.md)", encoding="utf-8")
    # A real .planning file outside worktrees must still be picked up.
    planning_md = planning / "overview.md"
    planning_md.write_text("[root](README.md)", encoding="utf-8")
    monkeypatch.setattr(checker, "ROOT_ALLOW", ())
    collected = checker.collect_markdown_files()
    assert planning_md in collected
    assert worktree_root / "README.md" not in collected


def test_is_skipped_worktree_predicate(checker, tmp_path):
    """The is_skipped_worktree helper accepts repo-relative and
    absolute paths, returns False for paths outside the skip tree,
    and works for files at any nesting depth."""
    # absolute path inside worktree
    assert checker.is_skipped_worktree(
        checker.REPO_ROOT / ".planning" / "worktrees" / "t1" / "deep" / "x.md"
    )
    # repo-root-relative path string (Path objects always resolve from repo)
    assert not checker.is_skipped_worktree(checker.REPO_ROOT / "docs" / "x.md")
    assert not checker.is_skipped_worktree(checker.REPO_ROOT / ".planning" / "overview.md")
    assert not checker.is_skipped_worktree(checker.REPO_ROOT / "README.md")
