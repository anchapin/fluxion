"""
Tests for ``scripts/check_doc_inventory_fresh.py`` -- Issue #2765.

Regression guard for the doc-inventory freshness gate. The script runs
``scripts/generate_doc_inventory.py`` as a subprocess and byte-compares
the committed ``docs/doc-inventory.md`` against its output. The gate
fails on drift (exit 1) and on generator errors (exit 2).

The script reads three module-level path constants at import time:

* ``REPO_ROOT``       -- the project root (used for ``cwd=``)
* ``INVENTORY_FILE``  -- the committed ``docs/doc-inventory.md``
* ``GENERATOR``       -- the path to ``generate_doc_inventory.py``

Each test redirects all three at a ``tmp_path`` synthetic tree. The
fake generator is a Python file that reads its target bytes from a
sidecar text file (``fake_body.txt``) so the same code path can drive
the fresh / drifted / missing / crash scenarios without invoking the
real ``generate_doc_inventory.py``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

SCRIPT_NAME = "check_doc_inventory_fresh"
SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "check_doc_inventory_fresh.py"
)


_FAKE_GENERATOR_TEMPLATE = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    \"\"\"Test fixture for check_doc_inventory_fresh: writes the contents of
    ``body.txt`` next to itself to the path in env ``FLUXION_TEST_INVENTORY_FILE``.
    Exits non-zero if ``body.txt`` contains the literal token ``__CRASH__``.

    The real script invokes this generator via ``python <path>`` with no
    extra argv but inherits the parent's env, so we resolve the target
    path from the ``FLUXION_TEST_INVENTORY_FILE`` env var set by the
    test harness.
    \"\"\"
    import os
    import sys
    from pathlib import Path

    target = Path(os.environ["FLUXION_TEST_INVENTORY_FILE"])
    body = Path(__file__).resolve().parent.joinpath("body.txt").read_text(
        encoding="utf-8"
    )
    if "__CRASH__" in body:
        sys.exit(3)
    target.write_text(body, encoding="utf-8")
    sys.exit(0)
    """
).strip()


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of the freshness gate."""
    return load_script(SCRIPT_NAME)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _redirect(checker, tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """Point the script's three path constants at a synthetic tree.

    Returns ``(inventory_path, scripts_dir)`` so the test can plant
    inventory content and the fake generator + body sidecar as needed.
    """
    inventory = tmp_path / "docs" / "doc-inventory.md"
    scripts_dir = tmp_path / "scripts"
    generator = scripts_dir / "fake_generator.py"
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "INVENTORY_FILE", inventory)
    monkeypatch.setattr(checker, "GENERATOR", generator)
    return inventory, scripts_dir


COMMITTED_BODY = (
    "# Doc Inventory\n"
    "\n"
    "<!-- BEGIN AUTO-GENERATED INVENTORY -->\n"
    "| Doc | Path | Status |\n"
    "| --- | ---- | ------ |\n"
    "| good.md | docs/good.md | OK |\n"
    "<!-- END AUTO-GENERATED INVENTORY -->\n"
    "\n"
    "Trailing prose.\n"
)


# ---------------------------------------------------------------------------
# Clean fixture (committed == generator output)
# ---------------------------------------------------------------------------


def test_main_returns_zero_when_committed_matches_generator(
    checker, tmp_path, monkeypatch, capsys
):
    """Fresh fixture: generator writes back the committed bytes → exit 0."""
    inventory, scripts_dir = _redirect(checker, tmp_path, monkeypatch)
    monkeypatch.setenv("FLUXION_TEST_INVENTORY_FILE", str(inventory))
    _write(inventory, COMMITTED_BODY)
    _write(scripts_dir / "fake_generator.py", _FAKE_GENERATOR_TEMPLATE)
    _write(scripts_dir / "body.txt", COMMITTED_BODY)

    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0, f"expected exit 0, got {rc}\noutput:\n{out}"
    assert "PASS" in out
    assert "fresh" in out.lower()


# ---------------------------------------------------------------------------
# Drift (committed ≠ generator output)
# ---------------------------------------------------------------------------


def test_main_returns_one_when_committed_drifts_from_generator(
    checker, tmp_path, monkeypatch, capsys
):
    """Drifted fixture: generator writes back different bytes → exit 1.

    The script must restore the committed bytes on the way out so a
    failed check does not silently rewrite the working tree.
    """
    inventory, scripts_dir = _redirect(checker, tmp_path, monkeypatch)
    monkeypatch.setenv("FLUXION_TEST_INVENTORY_FILE", str(inventory))
    _write(inventory, COMMITTED_BODY)
    drifted_body = COMMITTED_BODY.replace("| good.md |", "| stale.md |")
    _write(scripts_dir / "fake_generator.py", _FAKE_GENERATOR_TEMPLATE)
    _write(scripts_dir / "body.txt", drifted_body)

    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1, f"expected exit 1, got {rc}\noutput:\n{out}"
    assert "FAIL" in out
    assert "stale" in out.lower()
    # The committed bytes must be restored.
    assert inventory.read_text(encoding="utf-8") == COMMITTED_BODY


# ---------------------------------------------------------------------------
# Missing inputs (file or generator)
# ---------------------------------------------------------------------------


def test_main_returns_one_when_inventory_missing(checker, tmp_path, monkeypatch, capsys):
    """No committed inventory → exit 1.

    The script cannot byte-compare what does not exist; this is the
    "fresh clone, never generated" branch.
    """
    inventory, scripts_dir = _redirect(checker, tmp_path, monkeypatch)
    # Do NOT create inventory. The generator + body sidecar are created
    # so we don't trip the "GENERATOR missing" branch first.
    _write(scripts_dir / "fake_generator.py", _FAKE_GENERATOR_TEMPLATE)
    _write(scripts_dir / "body.txt", COMMITTED_BODY)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out
    assert "does not exist" in out


def test_main_returns_two_when_generator_missing(checker, tmp_path, monkeypatch, capsys):
    """No generator script → exit 2 (script-error, distinct from drift=1).

    Issue #2765 fail-loud contract: a missing generator must propagate
    as exit code 2, not silently green-light the freshness check.
    """
    inventory, scripts_dir = _redirect(checker, tmp_path, monkeypatch)
    _write(inventory, COMMITTED_BODY)
    # Do NOT create generator.
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 2
    assert "does not exist" in out


def test_main_returns_two_when_generator_crashes(
    checker, tmp_path, monkeypatch, capsys
):
    """Generator exits non-zero → exit 2 (script-error contract).

    A crashed generator must NEVER silently green-light the gate
    (fail-loud). The committed file must be restored so the working
    tree is not left in a half-written state.
    """
    inventory, scripts_dir = _redirect(checker, tmp_path, monkeypatch)
    _write(inventory, COMMITTED_BODY)
    _write(scripts_dir / "fake_generator.py", _FAKE_GENERATOR_TEMPLATE)
    _write(scripts_dir / "body.txt", "__CRASH__\n")

    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 2
    assert "generator exited" in out
    # Committed bytes restored.
    assert inventory.read_text(encoding="utf-8") == COMMITTED_BODY


# ---------------------------------------------------------------------------
# Exit-code wrapper contract
# ---------------------------------------------------------------------------


def test_main_wrapper_uses_try_except():
    """The ``__main__`` block wraps ``main()`` in ``try/except`` so an
    unhandled exception becomes ``sys.exit(2)`` rather than a traceback.

    Guards the fail-loud contract on the wrapper side: an unexpected
    ``KeyError`` or ``OSError`` inside ``main()`` must surface as a
    non-zero exit, never as Python's default traceback-then-1 contract.
    """
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in text
    wrapper = text.split('if __name__ == "__main__":', 1)[1]
    assert "try:" in wrapper
    assert "except Exception" in wrapper
    assert "sys.exit(2)" in wrapper
