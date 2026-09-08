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
