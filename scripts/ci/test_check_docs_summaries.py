"""
Tests for ``scripts/check_docs_summaries.py`` -- Issue #2466 acceptance
criterion #2 (every ``docs/**/*.md`` has a 7-line summary at lines 2-8).

Pins the summary-block classifier and the directory walk so a regression
in the content-counting heuristic cannot silently green a missing summary.
The classifier is duplicated (verbatim) across the docs hygiene scripts;
keeping these tests aligned covers the gate's core rule.
"""

from __future__ import annotations

import pytest

SCRIPT_NAME = "check_docs_summaries"


@pytest.fixture
def checker(load_script):
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
    "Body paragraph that is not part of the summary block.\n"
)


def _redirect(checker, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "DOCS_ROOT", tmp_path / "docs")


# ---------------------------------------------------------------------------
# has_seven_line_summary
# ---------------------------------------------------------------------------


def test_has_summary_true_for_well_formed_doc(checker, tmp_path):
    f = tmp_path / "good.md"
    f.write_text(GOOD_DOC, encoding="utf-8")
    ok, reason = checker.has_seven_line_summary(f)
    assert ok is True
    assert reason == "ok"


def test_has_summary_false_for_missing_file(checker, tmp_path):
    ok, reason = checker.has_seven_line_summary(tmp_path / "nope.md")
    assert ok is False
    assert "missing" in reason


def test_has_summary_false_when_too_few_lines(checker, tmp_path):
    f = tmp_path / "short.md"
    f.write_text("# T\nline\nline\n", encoding="utf-8")
    ok, reason = checker.has_seven_line_summary(f)
    assert ok is False
    assert "need >= 9" in reason


def test_has_summary_false_when_only_two_content_lines(checker, tmp_path):
    # 9+ lines but the lines 2-8 block has < 3 substantive (>=3 char) lines.
    f = tmp_path / "thin.md"
    f.write_text("# T\nfoo\nbar\n\n\n\n\n\nbody\n", encoding="utf-8")
    ok, reason = checker.has_seven_line_summary(f)
    assert ok is False
    assert "2 content line" in reason


def test_has_summary_counts_substantive_html_comment(checker, tmp_path):
    # HTML comments with >5 chars of inner text count as summary content.
    doc = (
        "# T\n"
        "<!-- Line 2: the first summary line is here -->\n"
        "<!-- Line 3: second summary line is here -->\n"
        "<!-- Line 4: third summary line is here -->\n"
        "<!-- Line 5: fourth summary line is here -->\n"
        "<!-- Line 6: fifth summary line is here -->\n"
        "<!-- Line 7: sixth summary line is here -->\n"
        "<!-- Line 8: seventh summary line is here -->\n"
        "body\n"
    )
    f = tmp_path / "html.md"
    f.write_text(doc, encoding="utf-8")
    assert checker.has_seven_line_summary(f)[0] is True


def test_has_summary_skips_last_updated_marker_but_real_lines_still_count(
    checker, tmp_path
):
    # A *Last Updated line is metadata (skipped), but 3 real >=3-char content
    # lines in the block still satisfy the minimum -> PASS.
    doc = "# T\nfoo\nbar\nbaz\n*Last Updated: 2026-01-01*\n\n\n\nbody\n"
    f = tmp_path / "marker.md"
    f.write_text(doc, encoding="utf-8")
    ok, _ = checker.has_seven_line_summary(f)
    assert ok is True


def test_last_updated_marker_alone_does_not_satisfy_minimum(checker, tmp_path):
    # Lines 2-8 are ALL *Last Updated / blank -> < 3 content lines -> fail.
    doc = (
        "# T\n"
        "*Last Updated: 2026-01-01*\n\n\n\n\n\n\n"
        "*Last Updated: 2026-01-02*\nbody\n"
    )
    f = tmp_path / "onlymarker.md"
    f.write_text(doc, encoding="utf-8")
    ok, reason = checker.has_seven_line_summary(f)
    assert ok is False
    assert "content line" in reason


# ---------------------------------------------------------------------------
# find_docs_md_files
# ---------------------------------------------------------------------------


def test_find_docs_lists_md_files_sorted(checker, tmp_path, monkeypatch):
    _redirect(checker, tmp_path, monkeypatch)
    docs = tmp_path / "docs"
    (docs / "a").mkdir(parents=True)
    (docs / "z.md").write_text(GOOD_DOC, encoding="utf-8")
    (docs / "a" / "b.md").write_text(GOOD_DOC, encoding="utf-8")
    found = checker.find_docs_md_files()
    names = [p.name for p in found]
    assert names == ["b.md", "z.md"]


def test_find_docs_empty_when_no_md(checker, tmp_path, monkeypatch):
    _redirect(checker, tmp_path, monkeypatch)
    (tmp_path / "docs").mkdir()
    assert checker.find_docs_md_files() == []


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_returns_zero_when_all_summaries_present(
    checker, tmp_path, monkeypatch, capsys
):
    _redirect(checker, tmp_path, monkeypatch)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "ok.md").write_text(GOOD_DOC, encoding="utf-8")
    assert checker.main() == 0
    assert "PASS" in capsys.readouterr().out


def test_main_returns_one_when_a_doc_lacks_summary(
    checker, tmp_path, monkeypatch, capsys
):
    _redirect(checker, tmp_path, monkeypatch)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "ok.md").write_text(GOOD_DOC, encoding="utf-8")
    (tmp_path / "docs" / "bad.md").write_text("# T\nx\n", encoding="utf-8")
    rc = checker.main()
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "bad.md" in out
