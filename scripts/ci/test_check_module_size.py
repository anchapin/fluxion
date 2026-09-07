"""Tests for ``scripts/check_module_size.py`` -- Issue #2878.

Regression guard for the god-struct decomposition ceiling enforced by
``scripts/check_module_size.py``. Mirrors the ``load_script`` + ``tmp_path``
mock-repo pattern from ``test_check_architecture_drift.py``:

* load the script as a fresh module via the shared ``load_script`` fixture,
* redirect the module-level ``REPO_ROOT`` constant at a synthetic
  ``tmp_path`` fixture that contains a minimal ``src/sim/`` tree,
* drive ``Limit.effective_max`` and ``check()`` through
  in-budget / over-budget / ratchet-tighten scenarios.

The script's ``LIMITS`` table is computed at import time using
``REPO_ROOT / "src" / "sim" / "thermal_model_data.rs"`` (and the
``mod.rs`` directory form). The fixture must therefore redirect
``REPO_ROOT`` *before* invoking any function that walks ``LIMITS``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPT_NAME = "check_module_size"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of ``scripts/check_module_size.py``."""
    return load_script(SCRIPT_NAME)


def _redirect(checker, tmp_path: Path, monkeypatch) -> None:
    """Point the script's ``REPO_ROOT`` at a synthetic ``tmp_path`` tree.

    The LIMITS table is built at import time with
    ``REPO_ROOT / "src" / "sim" / "thermal_model_data.rs"`` (and the
    ``mod.rs`` directory form). Without this redirect the tests would
    silently exercise the real ``src/sim/thermal_model_data.rs`` instead
    of the synthetic fixture, and a regression in the parser would
    cross-pollute the real repo's CI status.
    """
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    # Rebuild the LIMITS list pointing at tmp_path so the freshly-loaded
    # module's imports read from the synthetic tree.
    monkeypatch.setattr(
        checker,
        "LIMITS",
        [
            checker.Limit(
                path=tmp_path / "src" / "sim" / "thermal_model_data.rs",
                max_lines=200,
                ratchet_path=tmp_path
                / "tests"
                / "reference_data"
                / "module_size"
                / "thermal_model_data_ratchet.json",
                reason="Issue #2878 acceptance: ratchet for the god-struct.",
            ),
            checker.Limit(
                path=tmp_path / "src" / "sim" / "thermal_model_data" / "mod.rs",
                max_lines=200,
                ratchet_path=tmp_path
                / "tests"
                / "reference_data"
                / "module_size"
                / "thermal_model_data_ratchet.json",
                reason="Issue #2878 acceptance (directory form).",
            ),
        ],
    )


def _write_source(path: Path, lines: int) -> None:
    """Create a synthetic ``.rs`` file with the given number of lines.

    Each line is a placeholder Rust statement so the file parses cleanly
    if anything ever runs `rustc --emit=metadata` against the fixture.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"// synthetic line {i:04d}" for i in range(lines))
    path.write_text(body + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# count_lines
# ---------------------------------------------------------------------------


def test_count_lines_returns_zero_for_missing_path(checker, tmp_path):
    """A path that doesn't exist must report 0 lines (no exception).

    The script's ``check()`` treats a missing path as "skip this entry"
    so the gate is a no-op when the god-struct has been fully decomposed
    into sub-modules — this test pins the contract that the gate
    silently does NOT raise on missing files.
    """
    assert checker.count_lines(tmp_path / "does_not_exist.rs") == 0


def test_count_lines_matches_line_count(checker, tmp_path):
    """An N-line file must report N lines (trailing newline included)."""
    p = tmp_path / "sample.rs"
    _write_source(p, 42)
    assert checker.count_lines(p) == 42


def test_count_lines_handles_empty_file(checker, tmp_path):
    """An empty file must report 0 lines (not raise)."""
    p = tmp_path / "empty.rs"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")
    assert checker.count_lines(p) == 0


# ---------------------------------------------------------------------------
# Limit.effective_max — ratchet
# ---------------------------------------------------------------------------


def test_effective_max_uses_max_when_no_ratchet(checker, tmp_path, monkeypatch):
    """No ratchet file → effective ceiling equals ``max_lines``."""
    _redirect(checker, tmp_path, monkeypatch)
    limit = checker.LIMITS[0]
    assert limit.effective_max() == limit.max_lines


def test_effective_max_uses_max_when_ratchet_is_strictly_lower(
    checker, tmp_path, monkeypatch
):
    """``max(max_lines, ratchet_max)`` means the ratchet tracks the
    historical maximum — a tighter ratchet than the YAML ceiling is
    ignored, so the YAML is the floor."""
    _redirect(checker, tmp_path, monkeypatch)
    ratchet = checker.LIMITS[0].ratchet_path
    ratchet.parent.mkdir(parents=True, exist_ok=True)
    ratchet.write_text(
        '{"max_lines": 150, "history": [{"actual": 150}]}\n', encoding="utf-8"
    )
    # max_lines = 200, ratchet says 150 — effective_max is 200 (the higher).
    assert checker.LIMITS[0].effective_max() == 200


def test_effective_max_uses_ratchet_when_strictly_higher(
    checker, tmp_path, monkeypatch
):
    """A ratchet above the YAML ceiling RAISES the effective ceiling —
    this is the documented "ratchet DOWN, never up" semantics inverted
    in code: the ratchet JSON stores the historical max, so it can
    only loosen the ceiling as the historical max grows. To tighten
    the bound, an operator must lower ``max_lines`` in the script."""
    _redirect(checker, tmp_path, monkeypatch)
    ratchet = checker.LIMITS[0].ratchet_path
    ratchet.parent.mkdir(parents=True, exist_ok=True)
    ratchet.write_text(
        '{"max_lines": 999, "history": [{"actual": 999}]}\n', encoding="utf-8"
    )
    assert checker.LIMITS[0].effective_max() == 999


def test_effective_max_raises_on_malformed_ratchet(checker, tmp_path, monkeypatch):
    """A non-JSON ratchet must SystemExit — silent acceptance would let
    a corrupt ratchet disable the gate. The exit code carries the
    error message in ``code`` (the script does ``raise SystemExit(
    f"ERROR: ..." from exc``), so we assert the message shape rather
    than a numeric code."""
    _redirect(checker, tmp_path, monkeypatch)
    ratchet = checker.LIMITS[0].ratchet_path
    ratchet.parent.mkdir(parents=True, exist_ok=True)
    ratchet.write_text("this is not json", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        checker.LIMITS[0].effective_max()
    assert "ERROR: could not read ratchet JSON" in str(exc.value)


# ---------------------------------------------------------------------------
# check — in-budget / over-budget
# ---------------------------------------------------------------------------


def test_check_passes_when_under_ceiling(checker, tmp_path, monkeypatch):
    """A file below the ceiling must return ``Result.passed=True``."""
    _redirect(checker, tmp_path, monkeypatch)
    src = tmp_path / "src" / "sim" / "thermal_model_data.rs"
    _write_source(src, 161)  # current observed size on develop
    result = checker.check(checker.LIMITS[0])
    assert result is not None
    assert result.passed is True
    assert result.actual == 161


def test_check_fails_when_over_ceiling(checker, tmp_path, monkeypatch):
    """A file over the ceiling must return ``Result.passed=False``."""
    _redirect(checker, tmp_path, monkeypatch)
    src = tmp_path / "src" / "sim" / "thermal_model_data.rs"
    _write_source(src, 250)
    result = checker.check(checker.LIMITS[0])
    assert result is not None
    assert result.passed is False
    assert result.actual == 250


def test_check_returns_none_for_missing_file(checker, tmp_path, monkeypatch):
    """The mod.rs form is a no-op when the file form exists and vice versa."""
    _redirect(checker, tmp_path, monkeypatch)
    # Only the file form exists; mod.rs entry should return None.
    src = tmp_path / "src" / "sim" / "thermal_model_data.rs"
    _write_source(src, 100)
    result = checker.check(checker.LIMITS[1])
    assert result is None


def test_check_returns_none_when_neither_form_exists(checker, tmp_path, monkeypatch):
    """Both forms missing → both ``check()`` calls return None (gate is a
    no-op after the god-struct has been decomposed into sub-modules)."""
    _redirect(checker, tmp_path, monkeypatch)
    assert checker.check(checker.LIMITS[0]) is None
    assert checker.check(checker.LIMITS[1]) is None
