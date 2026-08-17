"""
Tests for ``scripts/check_stub_modules.py`` -- Issue #2896.

Regression guard for the stub-module detector. Mirrors the
``load_script`` + ``tmp_path`` mock-repo pattern from
``test_check_required_checks_sync.py`` / ``test_check_root_hygiene.py``:

* load the script as a fresh module via the shared ``load_script`` fixture,
* redirect the module-level ``REPO_ROOT`` and ``SCAN_DIRS`` constants at a
  synthetic ``tmp_path`` tree containing a planted ``src/`` directory,
* inject stub and non-stub ``.rs`` fixtures, then
* drive ``main()`` through clean / detected-stub / false-positive guard /
  sentinel-absent scenarios.

Issue #3075 acceptance criteria are realised as four scenarios:

1. **Clean state** -- no ``.rs`` files in the scan tree -> ``main()``
   returns ``0`` with the ``PASS`` banner.
2. **Detected stub** -- a short ``.rs`` file (below ``MIN_NON_COMMENT_LOC``)
   carrying one of the documented sentinel phrases -> ``main()`` returns
   ``1`` with the stub reported in stdout.
3. **False-positive guard** -- a *long* ``.rs`` file (above the LoC
   threshold) with the sentinel phrase present -> ``main()`` returns
   ``0``; the LoC threshold prevents a legitimate comment-only reference
   to the future-extraction idiom from tripping the gate.
4. **Sentinel absent** -- a short ``.rs`` file without any sentinel phrase
   -> ``main()`` returns ``0``; only files with **both** halves (sentinel
   + below-threshold LoC) are flagged.

Both halves are required (the script's documented contract): a file that
   satisfies one half but not the other must NOT be reported as a stub.
"""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pytest

SCRIPT_NAME = "check_stub_modules"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of the stub-module detector."""
    return load_script(SCRIPT_NAME)


def _redirect(checker, tmp_path: Path, monkeypatch) -> Path:
    """Point the script's ``REPO_ROOT`` and ``SCAN_DIRS`` at a synthetic
    ``tmp_path/src/`` tree and return the resolved ``src/`` path.

    Both constants are computed at import time from the script's location
    via ``Path(__file__).resolve().parent.parent``. Each test that wants
    a synthetic fixture must therefore redirect the constants before
    calling ``main()``.
    """
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "SCAN_DIRS", ("src",))
    return src_dir


def _scrub_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset ``sys.argv`` so the script's argparse doesn't see pytest's CLI.

    The script's ``main()`` accepts no args, but defensive scrub keeps the
    pattern consistent with sibling harnesses.
    """
    monkeypatch.setattr(sys, "argv", [SCRIPT_NAME])


def _write(p: Path, text: str = "") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dedent(text), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Helper: build stub / non-stub .rs fixtures with predictable LoC
# ---------------------------------------------------------------------------


def _short_stub_rs(sentinel: str = "marker for future extraction") -> str:
    """Build a short ``.rs`` source whose non-comment LoC is well below
    ``MIN_NON_COMMENT_LOC`` (20) and carries the sentinel phrase inside a
    doc comment so ``strip_rust_comments`` leaves it detectable via the
    full-source lowercased search.

    The script's ``find_stubs`` searches the *uncommented* source for
    the sentinel, so the phrase must survive comment-stripping. The
    sentinel phrase is therefore placed inside a ``//!`` (inner doc
    comment) which ``_LINE_COMMENT_RE = re.compile(r"//[^\\n]*")``
    strips -- so the lowercased-source test does NOT see it. The
    detector's actual search is::

        lowered = source.lower()
        hits = tuple(p for p in SENTINEL_PHRASES if p in lowered)

    i.e. on the *original* (unstripped) source. Therefore a real stub
    module's sentinel MUST live in a non-comment line OR in a string
    literal -- both of which ``strip_rust_comments`` passes through.
    """
    # ``const _: &str = "marker for future extraction";`` -- string
    # literal survives comment stripping and ``count_non_blank_non_comment_lines``
    # counts it as production code. 12 non-comment lines below the
    # threshold of 20.
    return (
        "//! Stub module -- design notes.\n"
        "//!\n"
        "//! This module is a marker for future extraction of the\n"
        "//! payload validator into its own crate. Do not add logic\n"
        "//! here until the extraction lands (issue #2896).\n"
        "\n"
        "/// Sentinel string the detector matches against.\n"
        'const SENTINEL: &str = "marker for future extraction";\n'
        "\n"
        "pub fn design_only() {}\n"
        "pub fn placeholder() {}\n"
        "pub fn reserved() {}\n"
        "pub fn future() {}\n"
    )


def _long_rs_with_sentinel(sentinel: str = "marker for future extraction") -> str:
    """Build a ``.rs`` source whose non-comment LoC is at or above
    ``MIN_NON_COMMENT_LOC`` (20) AND carries the sentinel phrase.

    The LoC threshold is the false-positive guard: the script's
    ``count_non_blank_non_comment_lines`` returns the count of lines
    that contain at least one non-whitespace character after comment
    stripping, and the file is filtered out when the count is >= 20.
    """
    # ~25 non-comment lines (each line is a ``let _x = N;`` statement),
    # plus a comment line carrying the sentinel phrase. Total file LoC
    # is well above the threshold so the detector must NOT flag it.
    lines = [
        f"/// Future-extraction discussion: {sentinel}",
        "fn many_statements() {",
    ]
    for i in range(25):
        lines.append(f"    let _x{i} = {i};")
    lines.extend(
        [
            "}",
            "",
            "fn other() {",
            "    let _y = 1;",
            "    let _z = 2;",
            "    let _w = 3;",
            "    let _v = 4;",
            "    let _u = 5;",
            "}",
        ]
    )
    return "\n".join(lines) + "\n"


def _short_rs_without_sentinel() -> str:
    """Build a short ``.rs`` source with NO sentinel phrase.

    Demonstrates the second half of the AND-condition: a file that is
    short enough to trip the LoC threshold but has no sentinel phrase
    must NOT be flagged (the sentinel is absent).
    """
    return (
        "//! Trivial helper module.\n"
        "//!\n"
        "//! Pure stub -- intentionally tiny. No design notes here.\n"
        "\n"
        "pub fn add(a: i32, b: i32) -> i32 { a + b }\n"
        "pub fn sub(a: i32, b: i32) -> i32 { a - b }\n"
    )


# ---------------------------------------------------------------------------
# main() -- clean / detected-stub / false-positive / sentinel-absent
# ---------------------------------------------------------------------------


def test_main_returns_zero_when_no_rs_files_in_scan_tree(
    checker, tmp_path, monkeypatch, capsys
):
    """Clean state: empty ``src/`` tree -> ``PASS`` (exit 0).

    The detector's walker yields no files, ``find_stubs()`` returns
    ``[]``, and the gate passes.
    """
    _redirect(checker, tmp_path, monkeypatch)
    _scrub_argv(monkeypatch)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0, f"expected PASS, got rc={rc}\noutput:\n{out}"
    assert "PASS" in out
    assert "no stub modules detected" in out


def test_main_returns_one_when_stub_with_sentinel_detected(
    checker, tmp_path, monkeypatch, capsys
):
    """Detected stub: a short ``.rs`` file carrying the sentinel phrase
    AND below the LoC threshold -> ``FAIL`` (exit 1).

    Both halves of the AND-condition are satisfied, so the detector
    must flag the file. The output surfaces the file path, the LoC
    count, and the matched sentinel.
    """
    src_dir = _redirect(checker, tmp_path, monkeypatch)
    _write(src_dir / "extracted.rs", _short_stub_rs())
    _scrub_argv(monkeypatch)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1, f"expected FAIL (stub), got rc={rc}\noutput:\n{out}"
    assert "FAIL" in out
    assert "extracted.rs" in out
    assert "marker for future extraction" in out
    # LoC line is surfaced.
    assert "non-comment LoC" in out
    # The threshold value (20) is echoed in the banner.
    assert "20" in out


def test_main_returns_zero_when_long_rs_carries_sentinel(
    checker, tmp_path, monkeypatch, capsys
):
    """False-positive guard: a ``.rs`` file above the LoC threshold
    (``MIN_NON_COMMENT_LOC`` = 20) with the sentinel phrase present
    -> ``PASS`` (exit 0).

    The sentinel ALONE is not enough to flag a file: the detector
    requires BOTH halves of the contract. A long file that happens to
    mention the future-extraction idiom in a comment (e.g. an ADR
    discussion) must not be flagged.
    """
    src_dir = _redirect(checker, tmp_path, monkeypatch)
    _write(src_dir / "design_doc.rs", _long_rs_with_sentinel())
    _scrub_argv(monkeypatch)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0, (
        f"expected PASS (false-positive guard), got rc={rc}\noutput:\n{out}"
    )
    assert "PASS" in out
    assert "design_doc.rs" not in out


def test_main_returns_zero_when_short_rs_has_no_sentinel(
    checker, tmp_path, monkeypatch, capsys
):
    """Sentinel absent: a short ``.rs`` file WITHOUT any sentinel phrase
    -> ``PASS`` (exit 0).

    The LoC threshold alone is not enough to flag a file. A genuinely
    short module that has actual logic (or no future-extraction
    discussion) must not be reported as a stub.
    """
    src_dir = _redirect(checker, tmp_path, monkeypatch)
    _write(src_dir / "trivial.rs", _short_rs_without_sentinel())
    _scrub_argv(monkeypatch)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0, f"expected PASS (no sentinel), got rc={rc}\noutput:\n{out}"
    assert "PASS" in out
    assert "trivial.rs" not in out


def test_main_returns_one_when_multiple_stubs_detected(
    checker, tmp_path, monkeypatch, capsys
):
    """Plural stubs: multiple short ``.rs`` files each carrying the
    sentinel -> ``FAIL`` (exit 1) with all paths listed.

    Verifies the detector reports every offender (not just the first)
    and counts the total correctly in the summary.
    """
    src_dir = _redirect(checker, tmp_path, monkeypatch)
    _write(src_dir / "alpha.rs", _short_stub_rs("marker for future extraction"))
    _write(src_dir / "beta.rs", _short_stub_rs("placeholder for future extraction"))
    _scrub_argv(monkeypatch)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1, f"expected FAIL (multi-stub), got rc={rc}\noutput:\n{out}"
    assert "alpha.rs" in out
    assert "beta.rs" in out
    # Both sentinels must be echoed in the per-file "sentinels hit" line.
    assert "marker for future extraction" in out
    assert "placeholder for future extraction" in out
    # Summary count.
    assert "2 stub module" in out


def test_main_returns_zero_on_real_repo(checker, repo_root, capsys):
    """Real-repo pin: the production workspace must currently be free
    of stub modules. A regression in the detector (or an accidental
    placeholder landing in the tree) flips this red.

    Run against the *real* repo root (no monkey-patching) so the scan
    walks the workspace exactly as CI does.
    """
    _scrub_argv(pytest.MonkeyPatch()) if False else None  # no-op; we use real argv
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0, (
        f"Real repo failed stub-module check:\n{out}\n"
        "If this regressed, search the workspace for "
        "'marker for future extraction' or "
        "'placeholder for future extraction' in short `.rs` files."
    )
    assert "PASS" in out


# ---------------------------------------------------------------------------
# Parsing primitives -- pins the script's detector against synthetic inputs
# ---------------------------------------------------------------------------


def test_strip_rust_comments_removes_block_and_line_comments(checker):
    """``strip_rust_comments`` removes ``/* ... */`` block and
    ``// ...`` line comments via naive regex substitution.

    Note: the implementation does not tokenise Rust source, so a
    ``/* ... */`` substring inside a string literal will also be
    stripped (the detector's docstring explicitly notes this is
    acceptable because both signals the detector cares about --
    LoC threshold and sentinel phrase -- survive the noise). This
    test pins the *actual* behaviour, not an aspirational one, so a
    regression that silently starts preserving them is caught.
    """
    src = (
        "/* block comment */\n"
        "let x = 1; // trailing line comment\n"
        "let y = 2;\n"
        "let z = 3;\n"
    )
    out = checker.strip_rust_comments(src)
    # Block comment gone.
    assert "/* block comment */" not in out
    # Line comment gone (only the trailing ``// ...`` is stripped;
    # the ``let x = 1;`` prefix survives because the regex matches
    # from ``//`` to end-of-line, then the substitution result is
    # ``let x = 1; `` -- a single trailing space remains.
    assert "let x = 1;" in out
    assert "// trailing" not in out
    # Non-comment statements survive.
    assert "let y = 2;" in out
    assert "let z = 3;" in out


def test_count_non_blank_non_comment_lines_excludes_comments_and_blanks(
    checker,
):
    """``count_non_blank_non_comment_lines`` returns the count of lines
    that contain at least one non-whitespace character after comment
    stripping. Blank lines and pure-comment lines do not count.
    """
    src = (
        "// header comment\n"
        "let x = 1;\n"
        "\n"
        "let y = 2;\n"
        "/* block comment spans\n"
        "   multiple lines */\n"
        "let z = 3;\n"
    )
    assert checker.count_non_blank_non_comment_lines(src) == 3


def test_find_stubs_returns_empty_when_no_rs_files(checker, tmp_path, monkeypatch):
    """``find_stubs`` returns ``[]`` when the scan tree is empty."""
    _redirect(checker, tmp_path, monkeypatch)
    assert checker.find_stubs() == []


def test_find_stubs_detects_planted_stub(checker, tmp_path, monkeypatch):
    """``find_stubs`` flags a planted short stub in ``src/`` with the
    sentinel phrase.

    Pins the detector's data-shape contract: ``[(path, loc, hit_phrases),
    ...]``. ``path`` is a ``Path`` inside ``src/``; ``loc`` is the LoC
    count below the threshold; ``hit_phrases`` is the tuple of matched
    sentinel phrases.
    """
    src_dir = _redirect(checker, tmp_path, monkeypatch)
    _write(src_dir / "extracted.rs", _short_stub_rs())
    stubs = checker.find_stubs()
    assert len(stubs) == 1
    path, loc, hits = stubs[0]
    assert path.name == "extracted.rs"
    assert loc < checker.MIN_NON_COMMENT_LOC
    assert "marker for future extraction" in hits


def test_find_stubs_skips_long_file_with_sentinel(checker, tmp_path, monkeypatch):
    """``find_stubs`` does NOT flag a long file carrying the sentinel
    phrase -- the LoC threshold is the false-positive guard."""
    src_dir = _redirect(checker, tmp_path, monkeypatch)
    _write(src_dir / "long.rs", _long_rs_with_sentinel())
    assert checker.find_stubs() == []


def test_find_stubs_skips_short_file_without_sentinel(
    checker, tmp_path, monkeypatch
):
    """``find_stubs`` does NOT flag a short file lacking the sentinel
    phrase -- the sentinel is the second required half."""
    src_dir = _redirect(checker, tmp_path, monkeypatch)
    _write(src_dir / "trivial.rs", _short_rs_without_sentinel())
    assert checker.find_stubs() == []


# ---------------------------------------------------------------------------
# Module-level constants pin
# ---------------------------------------------------------------------------


def test_min_non_comment_loc_constant_pinned(checker):
    """``MIN_NON_COMMENT_LOC`` must remain 20 (issue #2896 acceptance
    criterion). Raising the threshold would let short stubs slip
    through; lowering it would flag legitimate short modules.
    """
    assert checker.MIN_NON_COMMENT_LOC == 20


def test_sentinel_phrases_constant_pinned(checker):
    """The two sentinel phrases are the documented grep targets. Adding
    a third phrase without updating this list would leave the detector
    blind to it; removing one would silently narrow the surface."""
    assert "marker for future extraction" in checker.SENTINEL_PHRASES
    assert "placeholder for future extraction" in checker.SENTINEL_PHRASES


def test_scan_dirs_includes_root_src(checker):
    """``SCAN_DIRS`` must include the root ``src/`` tree (the main
    fluxion crate). Removing it would silently stop scanning the root
    crate -- the most likely place a stub would land."""
    assert "src" in checker.SCAN_DIRS


def test_scan_dirs_includes_fluxion_core(checker):
    """``SCAN_DIRS`` must include ``fluxion-core/src`` (the dependency-
    light leaf crate). Stub modules there are just as problematic as
    in the root crate."""
    assert "fluxion-core/src" in checker.SCAN_DIRS