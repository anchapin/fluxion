from __future__ import annotations

import pytest

SCRIPT_NAME = "check_ignore_tracking"


@pytest.fixture
def checker(load_script):
    return load_script(SCRIPT_NAME)


def test_find_ignored_tests_extracts_inline_and_doc_issues(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    source = tmp_path / "tests" / "sample.rs"
    source.parent.mkdir()
    source.write_text("/// ignored pending #123\n#[ignore = \"slow case #456\"]\nfn one() {}\n#[ignore]\nfn two() {}", encoding="utf-8")
    found = checker.find_ignored_tests()
    assert len(found) == 2
    assert found[0]["issues"] == ["123", "456"]
    assert found[1]["issues"] == []


def test_find_ignored_tests_ignores_non_test_paths(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "x.rs").write_text("#[ignore]\n", encoding="utf-8")
    assert checker.find_ignored_tests() == []


def test_main_by_issue_output(checker, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    source = tmp_path / "tests" / "sample.rs"
    source.parent.mkdir()
    source.write_text('#[ignore = "blocked #42"]\nfn one() {}', encoding="utf-8")
    monkeypatch.setattr(checker.sys, "argv", ["check_ignore_tracking.py", "--by-issue"])
    assert checker.main() == 0
    assert "#42:" in capsys.readouterr().out
