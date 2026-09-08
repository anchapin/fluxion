"""Tests for ``scripts/check_module_size.py`` -- Issues #2878, #3457.

Regression guard for the god-struct decomposition ceiling enforced by
``scripts/check_module_size.py``. Mirrors the ``load_script`` + ``tmp_path``
mock-repo pattern from ``test_check_architecture_drift.py``:

* load the script as a fresh module via the shared ``load_script`` fixture,
* redirect the module-level ``REPO_ROOT`` constant at a synthetic
  ``tmp_path`` fixture that contains a minimal ``src/sim/`` tree,
* drive ``Limit.effective_max`` and ``check()`` through
  in-budget / over-budget / ratchet-tighten scenarios, and
* drive ``check_baseline_drift()`` through the freeze-snapshot contract
  introduced by Issue #3457 (mirrors ``BASELINE_KNOWN_ORPHANS`` /
  ``_BASELINE_KNOWN_ORPHANS_SET`` from
  ``scripts/check_orphan_modules.py``).

The script's ``LIMITS`` table is computed at import time using
``REPO_ROOT / "src" / "..."``. The fixture must therefore redirect
``REPO_ROOT`` *before* invoking any function that walks ``LIMITS``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPT_NAME = "check_module_size"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of ``scripts/check_module_size.py``."""
    return load_script(SCRIPT_NAME)


def _redirect(checker, tmp_path: Path, monkeypatch) -> None:
    """Point the script's ``REPO_ROOT`` at a synthetic ``tmp_path`` tree.

    The LIMITS table is built at import time with
    ``REPO_ROOT / "src" / "..."``. Without this redirect the tests
    would silently exercise the real ``src/`` tree instead of the
    synthetic fixture, and a regression in the parser would
    cross-pollute the real repo's CI status.

    The synthetic tree below is a representative subset of the real
    LIMITS list: the two ``thermal_model_data`` forms (Issue #2878)
    plus one new Issue #3457 entry (``src/ai/surrogate.rs``).
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
            checker.Limit(
                path=tmp_path / "src" / "ai" / "surrogate.rs",
                max_lines=5726,
                ratchet_path=tmp_path
                / "tests"
                / "reference_data"
                / "module_size"
                / "surrogate_ratchet.json",
                reason="Issue #3457: surrogate module ratcheted at current size.",
            ),
        ],
    )
    # Shrink the baseline count + freeze snapshot to match the synthetic
    # 3-entry LIMITS list so the drift check stays neutral in tests that
    # only exercise the per-file ceiling semantics.
    monkeypatch.setattr(checker, "BASELINE_MODULE_SIZE_LIMITS", 3)
    monkeypatch.setattr(
        checker,
        "_BASELINE_MODULE_SIZE_LIMITS_SET",
        frozenset(
            {
                "src/sim/thermal_model_data.rs",
                "src/sim/thermal_model_data/mod.rs",
                "src/ai/surrogate.rs",
            }
        ),
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


# ---------------------------------------------------------------------------
# Issue #3457 — ratcheted ceiling at the file's CURRENT size
# ---------------------------------------------------------------------------


def test_check_passes_at_snapshotted_size(checker, tmp_path, monkeypatch):
    """An Issue #3457 entry must pass when the file is exactly at the
    snapshotted size (current line count), and fail the moment it
    grows by one line.

    This is the "the bound can only tighten" contract: the YAML
    ceiling IS the snapshotted current size, so the gate is
    tight-but-not-over-budget out of the box. Companion cleanup
    PRs that decompose the file are expected to lower the YAML
    ceiling alongside the decomposition.
    """
    _redirect(checker, tmp_path, monkeypatch)
    src = tmp_path / "src" / "ai" / "surrogate.rs"
    _write_source(src, 5726)  # current size on develop
    surrogate_limit = checker.LIMITS[2]
    assert surrogate_limit.max_lines == 5726
    result = checker.check(surrogate_limit)
    assert result is not None
    assert result.passed is True
    assert result.actual == 5726
    assert result.max == 5726


def test_check_fails_when_issue_3457_file_grows(checker, tmp_path, monkeypatch):
    """An Issue #3457 entry must FAIL when the file grows by even one
    line past the snapshotted ceiling — the gate is the regression
    detector, not a soft target."""
    _redirect(checker, tmp_path, monkeypatch)
    src = tmp_path / "src" / "ai" / "surrogate.rs"
    _write_source(src, 5727)  # +1 line
    result = checker.check(checker.LIMITS[2])
    assert result is not None
    assert result.passed is False
    assert result.actual == 5727


def test_check_baseline_drift_is_clean_for_frozen_limits(
    checker, tmp_path, monkeypatch
):
    """When ``LIMITS`` exactly matches the freeze snapshot, drift is empty."""
    _redirect(checker, tmp_path, monkeypatch)
    drift = checker.check_baseline_drift()
    assert drift == [], f"expected no drift, got: {drift}"


def test_check_baseline_drift_detects_new_entry(checker, tmp_path, monkeypatch):
    """Adding a new entry to ``LIMITS`` that is NOT in the freeze
    snapshot must surface as drift, naming the new path so the diff
    is visible in CI output.

    Mirrors ``_BASELINE_KNOWN_ORPHANS_SET`` from
    ``check_orphan_modules.py``: editing ``LIMITS`` alone, without
    mirroring the change into the freeze set, must be loud.
    """
    _redirect(checker, tmp_path, monkeypatch)
    # Append a new entry without updating the freeze snapshot.
    extra = checker.Limit(
        path=tmp_path / "src" / "validation" / "ashrae_140_cases.rs",
        max_lines=4764,
        ratchet_path=tmp_path
        / "tests"
        / "reference_data"
        / "module_size"
        / "ashrae_140_cases_ratchet.json",
        reason="Issue #3457 (synthetic): would-be new entry.",
    )
    monkeypatch.setattr(checker, "LIMITS", list(checker.LIMITS) + [extra])
    drift = checker.check_baseline_drift()
    assert any("src/validation/ashrae_140_cases.rs" in msg for msg in drift), (
        f"new path not reported in drift: {drift}"
    )
    assert any("BASELINE_MODULE_SIZE_LIMITS" in msg for msg in drift), (
        f"baseline count drift not reported: {drift}"
    )


def test_check_baseline_drift_passes_when_baseline_raised(
    checker, tmp_path, monkeypatch
):
    """When the freeze snapshot AND the count baseline are both raised
    to match the new LIMITS list, drift must be empty — that's the
    "raise the baseline" lever and is the only sanctioned way to add
    entries."""
    _redirect(checker, tmp_path, monkeypatch)
    extra = checker.Limit(
        path=tmp_path / "src" / "validation" / "ashrae_140_cases.rs",
        max_lines=4764,
        ratchet_path=tmp_path
        / "tests"
        / "reference_data"
        / "module_size"
        / "ashrae_140_cases_ratchet.json",
        reason="Issue #3457 (synthetic): raised baseline to match.",
    )
    monkeypatch.setattr(checker, "LIMITS", list(checker.LIMITS) + [extra])
    monkeypatch.setattr(checker, "BASELINE_MODULE_SIZE_LIMITS", 4)
    monkeypatch.setattr(
        checker,
        "_BASELINE_MODULE_SIZE_LIMITS_SET",
        frozenset(
            {
                "src/sim/thermal_model_data.rs",
                "src/sim/thermal_model_data/mod.rs",
                "src/ai/surrogate.rs",
                "src/validation/ashrae_140_cases.rs",
            }
        ),
    )
    drift = checker.check_baseline_drift()
    assert drift == [], f"expected no drift after raising baseline, got: {drift}"


def test_check_baseline_drift_detects_count_drift_without_new_paths(
    checker, tmp_path, monkeypatch
):
    """If the count baseline is wrong (lower than ``len(LIMITS)``) but
    the freeze snapshot still matches, the count drift must still
    fire. The count check and the path check are independent."""
    _redirect(checker, tmp_path, monkeypatch)
    monkeypatch.setattr(checker, "BASELINE_MODULE_SIZE_LIMITS", 2)
    drift = checker.check_baseline_drift()
    assert any("BASELINE_MODULE_SIZE_LIMITS drift" in msg for msg in drift), (
        f"count drift not reported: {drift}"
    )


# ---------------------------------------------------------------------------
# Real-repo smoke test (regression-locking).
# ---------------------------------------------------------------------------


def test_script_exits_zero_on_real_repo(repo_root):
    """Clean-tree smoke test against the real repo.

    Runs ``scripts/check_module_size.py`` against the production
    workspace (no monkey-patching) and asserts it exits 0. A
    regression in the per-file ratchet, the freeze snapshot, or the
    ``update_ratchet`` logic that mis-seeds a baseline flips this
    red.

    Per the issue brief, the test is driven via ``subprocess.run``
    so it exercises the script exactly as
    ``.github/workflows/architecture_drift.yml`` does — not the
    in-process ``main()``.
    """
    result = subprocess.run(
        ["python3", str(repo_root / "scripts" / "check_module_size.py")],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"expected exit 0 (all limits satisfied), got rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    # Banner phrases pin the public output shape against accidental
    # renames in the workflow wiring.
    assert "module-size gate" in result.stdout
    assert "All module-size limits satisfied." in result.stdout
    # The drift check is quiet when the LIMITS list matches the freeze
    # snapshot. If this assertion fails, somebody either added a new
    # LIMITS entry without updating the freeze (regression) or
    # changed the drift message wording (cosmetic).
    assert "BASELINE DRIFT" not in result.stdout
