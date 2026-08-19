"""
Tests for ``scripts/check_examples_present.py`` -- Issue #3125.

Regression guard for the example-presence detector. Mirrors the
``load_script`` + ``tmp_path`` mock-repo pattern from
``test_check_stub_modules.py`` / ``test_check_root_hygiene.py``:

* load the script as a fresh module via the shared ``load_script`` fixture,
* redirect the module-level ``REPO_ROOT`` and ``EXAMPLES_DIR`` constants
  at a synthetic ``tmp_path/examples/`` tree,
* plant example fixtures (above-threshold, below-threshold, missing dir,
  zero-count) and assert on the structured result.

Issue #3125 acceptance criteria are realised as five scenarios:

1. **Above threshold** -- ``MIN_EXAMPLE_COUNT`` or more example files
   in ``examples/`` -> ``main()`` returns ``0`` with the ``PASS``
   banner.
2. **Below threshold** -- fewer than ``MIN_EXAMPLE_COUNT`` example files
   -> ``main()`` returns ``1`` with the deficit surfaced.
3. **At threshold** -- exactly ``MIN_EXAMPLE_COUNT`` example files
   -> ``main()`` returns ``0``; the boundary is inclusive.
4. **Missing directory** -- no ``examples/`` directory at all ->
   ``main()`` returns ``2`` (script error), distinct from the below-
   threshold failure code.
5. **Non-.rs files ignored** -- only files with the ``.rs`` suffix
   contribute to the count; a planted ``examples/foo.txt`` does not
   pad the count.

Plus a real-repo pin: the production workspace must currently carry at
least ``MIN_EXAMPLE_COUNT`` example files. A regression that empties
or removes ``examples/`` flips this red.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPT_NAME = "check_examples_present"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of the example-presence detector."""
    return load_script(SCRIPT_NAME)


def _redirect(checker, tmp_path: Path, monkeypatch, *, examples_dir: str = "examples") -> Path:
    """Point the script's ``REPO_ROOT`` and ``EXAMPLES_DIR`` at a
    synthetic ``tmp_path/<examples_dir>/`` tree and return the resolved
    examples directory path.

    Both constants are computed at import time from the script's
    location via ``Path(__file__).resolve().parent.parent``. Each test
    that wants a synthetic fixture must therefore redirect the
    constants before calling ``main()``.
    """
    target = tmp_path / examples_dir
    target.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "EXAMPLES_DIR", examples_dir)
    return target


def _scrub_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset ``sys.argv`` so the script's argparse doesn't see pytest's CLI.

    The script's ``main()`` accepts no args, but defensive scrub keeps
    the pattern consistent with sibling harnesses.
    """
    monkeypatch.setattr(sys, "argv", [SCRIPT_NAME])


def _write(p: Path, text: str = "fn main() {}\n") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# main() -- above-threshold / below-threshold / boundary / missing / filter
# ---------------------------------------------------------------------------


def test_main_returns_zero_when_above_threshold(
    checker, tmp_path, monkeypatch, capsys
):
    """Above threshold: more than ``MIN_EXAMPLE_COUNT`` example files
    -> ``PASS`` (exit 0).

    The detector must surface every example path in the diagnostic
    output so a future maintainer can see what survived a cleanup.
    """
    examples_dir = _redirect(checker, tmp_path, monkeypatch)
    # Plant 6 .rs files -- one above the threshold of 5.
    for i in range(6):
        _write(examples_dir / f"ex_{i}.rs")
    _scrub_argv(monkeypatch)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0, f"expected PASS, got rc={rc}\noutput:\n{out}"
    assert "PASS" in out
    assert ">= minimum 5" in out
    # Every planted file must be listed.
    for i in range(6):
        assert f"ex_{i}.rs" in out


def test_main_returns_one_when_below_threshold(
    checker, tmp_path, monkeypatch, capsys
):
    """Below threshold: fewer than ``MIN_EXAMPLE_COUNT`` example files
    -> ``FAIL`` (exit 1).

    The detector surfaces the deficit so a maintainer can decide
    whether to restore the deleted files or to formally lower the
    threshold in the issue thread.
    """
    examples_dir = _redirect(checker, tmp_path, monkeypatch)
    # Plant 3 .rs files -- well below the threshold of 5.
    for i in range(3):
        _write(examples_dir / f"survivor_{i}.rs")
    _scrub_argv(monkeypatch)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1, f"expected FAIL (below threshold), got rc={rc}\noutput:\n{out}"
    assert "FAIL" in out
    assert "below minimum 5" in out
    # All survivors are listed so the diagnostic is actionable.
    for i in range(3):
        assert f"survivor_{i}.rs" in out


def test_main_returns_zero_at_exact_threshold(
    checker, tmp_path, monkeypatch, capsys
):
    """Boundary: exactly ``MIN_EXAMPLE_COUNT`` example files -> ``PASS``
    (exit 0).

    The threshold is inclusive (``count >= MIN_EXAMPLE_COUNT``); the
    detector must NOT fail on the boundary value.
    """
    examples_dir = _redirect(checker, tmp_path, monkeypatch)
    for i in range(5):
        _write(examples_dir / f"boundary_{i}.rs")
    _scrub_argv(monkeypatch)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0, f"expected PASS (at threshold), got rc={rc}\noutput:\n{out}"
    assert "PASS" in out


def test_main_returns_two_when_examples_dir_missing(
    checker, tmp_path, monkeypatch, capsys
):
    """Missing directory: no ``examples/`` directory at all -> script
    error (exit 2).

    A wholesale removal of ``examples/`` is distinct from a below-
    threshold count: it implies the public surface was deleted
    intentionally, which the gate must escalate as a different
    failure mode.
    """
    # Don't mkdir the examples dir -- the redirect fixture skips mkdir
    # when EXAMPLES_DIR is overridden. We just leave tmp_path empty.
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "EXAMPLES_DIR", "examples")
    _scrub_argv(monkeypatch)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 2, f"expected ERROR (missing dir), got rc={rc}\noutput:\n{out}"
    assert "FAIL" in out or "ERROR" in out
    assert "examples directory missing" in out


def test_main_ignores_non_rs_files(checker, tmp_path, monkeypatch, capsys):
    """Filter: only ``.rs`` files count toward the threshold.

    A planted ``examples/foo.txt`` does not pad the count. The
    detector must filter on the suffix so a future contributor
    adding README / config / shell snippets to ``examples/`` does
    not accidentally satisfy the gate.
    """
    examples_dir = _redirect(checker, tmp_path, monkeypatch)
    # Plant 4 .rs files -- below the threshold of 5.
    for i in range(4):
        _write(examples_dir / f"real_{i}.rs")
    # Plus noise files in the same directory.
    _write(examples_dir / "README.md", "# notes\n")
    _write(examples_dir / "notes.txt", "scratch\n")
    _write(examples_dir / "Makefile", "all:\n\t@true\n")
    _scrub_argv(monkeypatch)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1, f"expected FAIL (only 4 .rs), got rc={rc}\noutput:\n{out}"
    assert "below minimum 5" in out
    # The .rs files are listed; the noise files are not.
    assert "real_0.rs" in out
    assert "README.md" not in out
    assert "notes.txt" not in out
    assert "Makefile" not in out


def test_main_returns_zero_on_real_repo(checker, repo_root, capsys):
    """Real-repo pin: the production workspace must currently carry at
    least ``MIN_EXAMPLE_COUNT`` example files. A regression that
    empties or removes ``examples/`` flips this red.

    Run against the *real* repo root (no monkey-patching) so the
    detector walks the same tree CI does.
    """
    _scrub_argv(pytest.MonkeyPatch()) if False else None  # no-op; we use real argv
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0, (
        f"Real repo failed example-presence check:\n{out}\n"
        "If this regressed, examples/*.rs were either deleted or the "
        "examples/ directory was removed. See issue #3125 for the "
        "rationale."
    )
    assert "PASS" in out


# ---------------------------------------------------------------------------
# list_example_files -- data-shape contract
# ---------------------------------------------------------------------------


def test_list_example_files_returns_sorted_paths(
    checker, tmp_path, monkeypatch
):
    """``list_example_files`` returns the sorted list of ``.rs`` paths
    under ``examples/``. Sorted order keeps the diagnostic output
    deterministic across filesystems.
    """
    examples_dir = _redirect(checker, tmp_path, monkeypatch)
    # Plant in non-sorted order on purpose.
    names = ["zeta.rs", "alpha.rs", "mu.rs", "beta.rs"]
    for name in names:
        _write(examples_dir / name)
    files = checker.list_example_files()
    actual_names = [p.name for p in files]
    assert actual_names == sorted(names)


def test_list_example_files_returns_empty_when_dir_missing(
    checker, tmp_path, monkeypatch
):
    """``list_example_files`` returns ``[]`` when the ``examples/``
    directory is absent, NOT raising. The directory-presence check
    is owned by ``main()`` which returns exit code 2; the helper
    itself stays a quiet enumerator.
    """
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "EXAMPLES_DIR", "does_not_exist")
    assert checker.list_example_files() == []


def test_list_example_files_filters_to_rs_suffix(
    checker, tmp_path, monkeypatch
):
    """``list_example_files`` filters to the ``.rs`` suffix; non-Rust
    files in the same directory are excluded.
    """
    examples_dir = _redirect(checker, tmp_path, monkeypatch)
    _write(examples_dir / "keep.rs")
    _write(examples_dir / "drop.md")
    _write(examples_dir / "drop.toml")
    files = checker.list_example_files()
    names = [p.name for p in files]
    assert names == ["keep.rs"]


# ---------------------------------------------------------------------------
# Module-level constants pin
# ---------------------------------------------------------------------------


def test_min_example_count_constant_pinned(checker):
    """``MIN_EXAMPLE_COUNT`` must remain 5 (issue #3125 acceptance
    criterion ``ls examples/*.rs | wc -l >= 5``).

    Raising the threshold without a corresponding issue-thread
    decision would let a future cleanup slip through silently;
    lowering it would weaken the public-surface contract.
    """
    assert checker.MIN_EXAMPLE_COUNT == 5


def test_examples_dir_constant_pinned(checker):
    """``EXAMPLES_DIR`` must remain ``"examples"`` (the repo-root user-
    facing example set). Renaming the directory requires a deliberate
    decision and a corresponding update to this constant + the
    workspace `[[example]]` entries in ``Cargo.toml``.
    """
    assert checker.EXAMPLES_DIR == "examples"
