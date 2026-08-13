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
