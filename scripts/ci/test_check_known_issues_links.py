from __future__ import annotations

import pytest

SCRIPT_NAME = "check_known_issues_links"


@pytest.fixture
def checker(load_script):
    return load_script(SCRIPT_NAME)


def test_is_intra_repo_classifies_targets(checker):
    assert checker.is_intra_repo("guide.md#intro")
    assert not checker.is_intra_repo("#intro")
    assert not checker.is_intra_repo("https://example.com/guide.md")
    assert not checker.is_intra_repo("mailto:test@example.com")


def test_audit_file_reports_missing_targets(checker, tmp_path):
    doc = tmp_path / "FAQ.md"
    doc.write_text("[present](present.md) [missing](missing.md) [web](https://example.com)", encoding="utf-8")
    (tmp_path / "present.md").write_text("ok", encoding="utf-8")
    assert checker.audit_file(doc) == [("missing.md", f"-> {tmp_path / 'missing.md'} (does not exist)")]


def test_main_returns_two_for_missing_audited_file(checker, monkeypatch, tmp_path):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "AUDITED_FILES", (tmp_path / "FAQ.md",))
    assert checker.main() == 1
