"""
Tests for ``scripts/check_known_issues_stale.py`` -- Issue #1723.

Regression guard for the ``KNOWN_ISSUES.md`` freshness gate. Mirrors the
``load_script`` + ``tmp_path`` mock-repo pattern from
``test_check_root_hygiene.py``:

* load the script as a fresh module via the shared ``load_script`` fixture,
* redirect the module-level ``KNOWN_ISSUES_PATH`` constant to a synthetic
  ``tmp_path`` file, then
* drive ``main()`` through clean (fresh date) and violation (stale date)
  scenarios.

The script resolves ``KNOWN_ISSUES_PATH`` at import time from a
module-level string constant, so the freshly-loaded module carries the
*real* repo path. Each test that wants a synthetic fixture must therefore
redirect the constant before calling ``main()``.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

SCRIPT_NAME = "check_known_issues_stale"


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of the stale-check script."""
    return load_script(SCRIPT_NAME)


def _redirect(checker, tmp_path: Path, monkeypatch) -> Path:
    """Point the script's ``KNOWN_ISSUES_PATH`` at a synthetic file in
    ``tmp_path`` and return the resolved file path.

    The constant is a plain string (not a ``Path``), so the script's
    ``open()`` call resolves to the test fixture location.
    """
    target = tmp_path / "KNOWN_ISSUES.md"
    monkeypatch.setattr(checker, "KNOWN_ISSUES_PATH", str(target))
    return target


def _write_with_last_updated(path: Path, dt: date) -> None:
    """Write a minimal file whose ``*Last Updated: YYYY-MM-DD*`` line
    carries ``dt``."""
    path.write_text(
        f"# Known Issues\n\n*Last Updated: {dt.isoformat()}*\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# main() — clean / violation scenarios
# ---------------------------------------------------------------------------


def test_main_returns_zero_when_last_updated_is_today(checker, tmp_path, monkeypatch, capsys):
    """Clean fixture: today → exit 0 with OK banner."""
    target = _redirect(checker, tmp_path, monkeypatch)
    _write_with_last_updated(target, date.today())
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK" in out
    assert "within" in out


def test_main_returns_zero_when_last_updated_within_threshold(
    checker, tmp_path, monkeypatch, capsys
):
    """Date 30 days old → exit 0 (within 60-day threshold)."""
    target = _redirect(checker, tmp_path, monkeypatch)
    _write_with_last_updated(target, date.today() - timedelta(days=30))
    assert checker.main() == 0


def test_main_returns_zero_at_threshold_boundary(
    checker, tmp_path, monkeypatch, capsys
):
    """Date exactly at the 60-day boundary → exit 0 (cutoff uses ``>=``)."""
    target = _redirect(checker, tmp_path, monkeypatch)
    # The script uses `cutoff = date.today() - 60 days`. `last_updated >= cutoff`
    # is true when `last_updated == cutoff`, so dates 60 days old (or newer)
    # are considered fresh.
    boundary = date.today() - timedelta(days=60)
    _write_with_last_updated(target, boundary)
    assert checker.main() == 0


def test_main_returns_one_when_last_updated_is_stale(
    checker, tmp_path, monkeypatch, capsys
):
    """Date 90 days old → exit 1 with FAIL banner.

    This is the planted-violation case from the issue acceptance criteria:
    a fresh clone whose ``*Last Updated:*`` line is stale fails the gate.
    """
    target = _redirect(checker, tmp_path, monkeypatch)
    stale = date.today() - timedelta(days=90)
    _write_with_last_updated(target, stale)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out
    assert str(stale) in out
    assert "Update the '*Last Updated:*'" in out


def test_main_returns_one_when_one_day_past_threshold(
    checker, tmp_path, monkeypatch, capsys
):
    """Date exactly 61 days old → exit 1 (one day past the 60-day cutoff)."""
    target = _redirect(checker, tmp_path, monkeypatch)
    _write_with_last_updated(target, date.today() - timedelta(days=61))
    assert checker.main() == 1


# ---------------------------------------------------------------------------
# Missing / malformed file paths (gate must not false-positive on missing files)
# ---------------------------------------------------------------------------


def test_main_returns_zero_when_file_missing(checker, tmp_path, monkeypatch, capsys):
    """Missing KNOWN_ISSUES.md → exit 0 with skip notice.

    Issue #1723 explicitly says: "If the file doesn't exist, skip the
    check (not a failure)". The gate must NEVER false-positive on a fresh
    checkout where the file has not been generated yet.
    """
    _redirect(checker, tmp_path, monkeypatch)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "not found" in out.lower()


def test_main_returns_one_when_marker_absent(checker, tmp_path, monkeypatch, capsys):
    """File exists but has no ``*Last Updated:*`` marker → exit 1 with ERROR.

    The gate must BLOCK when the marker is missing -- per issue #3105, the
    pre-2026-08-17 silent-skip behavior was the bug that allowed format
    drift to silently disable the gate (the ``WARN + exit 0`` branch made
    this CI check inoperative even after a contributor renamed the
    ``*Last Updated:*`` heading). The fix: a recognizable marker absent is
    a structural failure of the gate, NOT a tolerable format drift. PR
    #3112 lands the regex hardening that makes this test green.

    If a contributor legitimately renames the heading they MUST also bump
    the ``*Last Updated:*`` line OR update the regex -- the gate's job is
    to force that attention, not to silently forgive it.
    """
    target = _redirect(checker, tmp_path, monkeypatch)
    target.write_text("# Known Issues\n\nNo timestamp here.\n", encoding="utf-8")
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "ERROR" in out
    assert "Last Updated" in out
    assert "regex" in out.lower()


def test_main_returns_zero_when_marker_has_parenthetical_summary(
    checker, tmp_path, monkeypatch, capsys
):
    """Marker with a parenthetical summary between date and closing ``*`` is recognised.

    The established ``*Last Updated:*`` line format (used by every wave-
    orchestration PR since #3071) places the closing ``*`` after an
    optional parenthetical summary of LIMIT-N additions, e.g.::

        *Last Updated: 2026-08-17 (LIMIT-08 + LIMIT-09 + ...)*

    Pre-#3105 the script's regex required the closing ``*`` to IMMEDIATELY
    follow the date, which caused every LIMIT-N addition to trigger a
    silent WARN + exit 0. PR #3112 hardens the regex to accept the
    parenthetical; this test pins the post-fix behavior so future
    refactors don't silently regress it.
    """
    target = _redirect(checker, tmp_path, monkeypatch)
    today = date.today()
    target.write_text(
        f"# Known Issues\n\n*Last Updated: {today.isoformat()} "
        f"(LIMIT-08 + LIMIT-09 + LIMIT-10 retained)*\n",
        encoding="utf-8",
    )
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK" in out
    assert str(today) in out
    assert "within" in out


# ---------------------------------------------------------------------------
# Marker format variants
# ---------------------------------------------------------------------------


def test_main_handles_marker_with_extra_whitespace(checker, tmp_path, monkeypatch):
    """Marker with extra whitespace after the colon is recognised.

    The script's regex ``\\*Last Updated:\\s*(\\d{4}-\\d{2}-\\d{2})\\*``
    uses ``\\s*`` after the colon so a contributor who adds a stray
    space does not silently regress the gate.
    """
    target = _redirect(checker, tmp_path, monkeypatch)
    fresh = date.today() - timedelta(days=5)
    target.write_text(
        f"# Known Issues\n\n*Last Updated:  {fresh.isoformat()}*\n",
        encoding="utf-8",
    )
    assert checker.main() == 0


# ---------------------------------------------------------------------------
# Module-level constants pin
# ---------------------------------------------------------------------------


def test_stale_threshold_is_sixty_days(checker):
    """``STALE_THRESHOLD_DAYS`` must remain 60.

    A regression that bumps this to 90 (or 365) would silently extend
    the freshness window. The constant lives at module scope so it is
    directly assertable.
    """
    assert checker.STALE_THRESHOLD_DAYS == 60
