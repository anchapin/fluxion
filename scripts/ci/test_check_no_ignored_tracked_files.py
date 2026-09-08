from __future__ import annotations

from types import SimpleNamespace

import pytest

SCRIPT_NAME = "check_no_ignored_tracked_files"


@pytest.fixture
def checker(load_script):
    return load_script(SCRIPT_NAME)


def test_find_tracked_but_ignored_uses_git_checks(checker, monkeypatch, tmp_path):
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if args[1] == "check-ignore":
            return SimpleNamespace(returncode=0, stdout=".gitignore:1:*.tmp\tfile.tmp\n", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker.subprocess, "run", run)
    assert checker.find_tracked_but_ignored(["file.tmp"]) == [(".gitignore:1:*.tmp\tfile.tmp", "file.tmp")]
    assert len(calls) == 2


def test_main_returns_zero_without_hits(checker, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    def run(args, **kwargs):
        if args[1] == "rev-parse":
            return SimpleNamespace(returncode=0, stdout=str(tmp_path) + "\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="file.rs\0", stderr="")
    monkeypatch.setattr(checker.subprocess, "run", run)
    monkeypatch.setattr(checker, "find_tracked_but_ignored", lambda tracked: [])
    assert checker.main() == 0
    assert "none ignored" in capsys.readouterr().out


def test_main_returns_one_for_new_regression(checker, monkeypatch, tmp_path):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    def run(args, **kwargs):
        if args[1] == "rev-parse":
            return SimpleNamespace(returncode=0, stdout=str(tmp_path) + "\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="new.tmp\0", stderr="")
    monkeypatch.setattr(checker.subprocess, "run", run)
    monkeypatch.setattr(checker, "find_tracked_but_ignored", lambda tracked: [("rule", "new.tmp")])
    assert checker.main() == 1
