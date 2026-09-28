"""Tests for ``scripts/refresh_known_issues.sh`` -- Issue #4209.

The refresh script is the *writer* behind the staleness gate
(``scripts/check_known_issues_stale.py``): running it satisfies the
<=60-day signal without necessarily doing the review work the signal
stands for. These tests drive the shell script via ``subprocess``
against ``tmp_path`` fixture copies of a ``KNOWN_ISSUES.md``-shaped file
(the script's ``--file`` flag redirects it off the default
``docs/KNOWN_ISSUES.md`` path) and assert:

(a) after a run, ``check_known_issues_stale.py`` passes against the
    updated file (the script is executed as a subprocess with
    ``cwd=tmp_path`` so its ``docs/KNOWN_ISSUES.md`` path constant
    resolves to the fixture),
(b) idempotence — a second run on the same day is a no-op exiting 0
    (file byte-identical),
(c) a file with NO ``*Last Updated:*`` marker is reported as a failure
    (non-zero exit) and left unmodified,
(d) ``--dry-run`` leaves the file byte-identical,
(e) the ``## Summary`` table still matches the section headers per
    ``check_known_issues_summary.py --check`` after the run.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "refresh_known_issues.sh"
STALE_SCRIPT = REPO_ROOT / "scripts" / "check_known_issues_stale.py"
TODAY = date.today()


def _run_script(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run the refresh script as ``bash scripts/refresh_known_issues.sh``."""
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def _write_fixture(
    tmp_path: Path,
    load_script,
    *,
    last_updated: date | None,
    parenthetical: str = "",
    marker: bool = True,
) -> Path:
    """Write a ``KNOWN_ISSUES.md``-shaped fixture under
    ``tmp_path/docs/KNOWN_ISSUES.md`` whose ``## Summary`` table is
    derived from the real ``check_known_issues_summary.py`` renderer, so
    the (e) gate passes on the pristine fixture.

    ``last_updated=None`` (or ``marker=False``) produces a file with no
    ``*Last Updated:*`` marker for the (c) case.
    """
    summary_mod = load_script("check_known_issues_summary")
    if marker and last_updated is not None:
        preamble = f"# Known Issues\n\n*Last Updated: {last_updated.isoformat()}{parenthetical}*\n\n"
    else:
        preamble = "# Known Issues\n\nNo marker in this file.\n\n"
    sections = (
        "\n## Limits\n\n"
        "### LIMIT-01: Example limit\n\n"
        "**Status:** Open — under investigation\n"
    )
    # The derived counts see the same headers/sections with or without the
    # rendered table (the table itself contributes no `### ` headers), so
    # deriving from the table-less body is exact.
    counts = summary_mod.extract_counts(preamble + sections)
    table = summary_mod.render_table(counts) + summary_mod.render_legend()
    target = tmp_path / "docs" / "KNOWN_ISSUES.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(preamble + table + sections, encoding="utf-8")
    return target


def _stale_gate(tmp_path: Path) -> int:
    """Run the real staleness gate against ``tmp_path/docs/KNOWN_ISSUES.md``."""
    return subprocess.run(
        [sys.executable, str(STALE_SCRIPT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    ).returncode


def _summary_check(target: Path, load_script, monkeypatch) -> int:
    """Run ``check_known_issues_summary.py --check`` against ``target``."""
    summary_mod = load_script("check_known_issues_summary")
    # The module's PASS path prints KNOWN_ISSUES.relative_to(REPO_ROOT),
    # so the fake root must be the fixture's parent-of-docs.
    monkeypatch.setattr(summary_mod, "REPO_ROOT", target.parent.parent)
    monkeypatch.setattr(summary_mod, "KNOWN_ISSUES", target)
    monkeypatch.setattr(
        sys, "argv", ["check_known_issues_summary.py", "--check"]
    )
    return summary_mod.main()


# ---------------------------------------------------------------------------
# (a) refresh makes the staleness gate pass
# ---------------------------------------------------------------------------


def test_refresh_makes_stale_gate_pass(tmp_path, load_script):
    target = _write_fixture(
        tmp_path, load_script, last_updated=TODAY - timedelta(days=61)
    )
    # The fixture is genuinely stale first — otherwise the assertion below
    # would prove nothing.
    assert _stale_gate(tmp_path) == 1

    result = _run_script("--file", str(target), cwd=tmp_path)
    assert result.returncode == 0, result.stderr

    text = target.read_text(encoding="utf-8")
    assert f"*Last Updated: {TODAY.isoformat()}*" in text
    assert _stale_gate(tmp_path) == 0


def test_refresh_preserves_parenthetical_review_summary(tmp_path, load_script):
    """The parenthetical form (what the real file uses, and what the
    staleness gate's regex accepts) gets its date refreshed while the
    review summary is preserved. The pre-#4209 script silently no-op'd
    here while reporting success."""
    target = _write_fixture(
        tmp_path,
        load_script,
        last_updated=TODAY - timedelta(days=61),
        parenthetical=" (LIMIT-30 UPDATE — review note)",
    )
    assert _stale_gate(tmp_path) == 1

    result = _run_script("--file", str(target), cwd=tmp_path)
    assert result.returncode == 0, result.stderr

    text = target.read_text(encoding="utf-8")
    assert (
        f"*Last Updated: {TODAY.isoformat()} (LIMIT-30 UPDATE — review note)*"
        in text
    )
    assert _stale_gate(tmp_path) == 0


# ---------------------------------------------------------------------------
# (b) idempotence
# ---------------------------------------------------------------------------


def test_second_run_same_day_is_noop(tmp_path, load_script):
    target = _write_fixture(
        tmp_path, load_script, last_updated=TODAY - timedelta(days=61)
    )
    first = _run_script("--file", str(target), cwd=tmp_path)
    assert first.returncode == 0, first.stderr

    before = target.read_bytes()
    second = _run_script("--file", str(target), cwd=tmp_path)
    assert second.returncode == 0, second.stderr
    assert "already current" in second.stdout
    assert target.read_bytes() == before


# ---------------------------------------------------------------------------
# (c) missing marker fails loudly and leaves the file untouched
# ---------------------------------------------------------------------------


def test_missing_marker_fails_and_leaves_file_untouched(tmp_path, load_script):
    target = _write_fixture(tmp_path, load_script, last_updated=None)
    before = target.read_bytes()

    result = _run_script("--file", str(target), cwd=tmp_path)
    assert result.returncode != 0
    assert "FAIL" in result.stderr
    assert "Last Updated" in result.stderr
    assert target.read_bytes() == before


# ---------------------------------------------------------------------------
# (d) dry-run
# ---------------------------------------------------------------------------


def test_dry_run_leaves_file_byte_identical(tmp_path, load_script):
    target = _write_fixture(
        tmp_path, load_script, last_updated=TODAY - timedelta(days=61)
    )
    before = target.read_bytes()

    result = _run_script("--dry-run", "--file", str(target), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "DRY-RUN" in result.stdout
    assert target.read_bytes() == before
    # The pending change is still pending: the gate still fails.
    assert _stale_gate(tmp_path) == 1


# ---------------------------------------------------------------------------
# (e) summary table consistency after the run
# ---------------------------------------------------------------------------


def test_summary_table_still_matches_after_run(tmp_path, load_script, monkeypatch):
    target = _write_fixture(
        tmp_path, load_script, last_updated=TODAY - timedelta(days=61)
    )
    assert _summary_check(target, load_script, monkeypatch) == 0

    result = _run_script("--file", str(target), cwd=tmp_path)
    assert result.returncode == 0, result.stderr

    assert _summary_check(target, load_script, monkeypatch) == 0


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


def test_help_exits_zero_and_names_flags(tmp_path):
    result = _run_script("--help", cwd=tmp_path)
    assert result.returncode == 0
    assert "--dry-run" in result.stdout
    assert "--file" in result.stdout


def test_unknown_argument_is_usage_error(tmp_path, load_script):
    target = _write_fixture(
        tmp_path, load_script, last_updated=TODAY - timedelta(days=61)
    )
    before = target.read_bytes()
    result = _run_script("--bogus", "--file", str(target), cwd=tmp_path)
    assert result.returncode == 2
    assert target.read_bytes() == before


def test_missing_file_is_failure(tmp_path):
    result = _run_script(
        "--file", str(tmp_path / "docs" / "MISSING.md"), cwd=tmp_path
    )
    assert result.returncode != 0
    assert "not found" in result.stderr
