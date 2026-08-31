"""
Tests for ``scripts/check_audit_ignores_fresh.py`` -- Issue #2912.

Regression guard for the audit-ignore freshness gate. Mirrors the
``load_script`` + ``tmp_path`` mock-repo pattern from
``test_check_required_checks_sync.py`` / ``test_check_rumqttc_upstream.py``:

* load the script as a fresh module via the shared ``load_script`` fixture,
* redirect the module-level ``AUDIT_TOML`` / ``CARGO_LOCK`` constants at a
  synthetic ``tmp_path`` tree containing a planted ``.cargo/audit.toml`` +
  ``Cargo.lock`` pair, then
* drive ``main()`` through clean / stale / missing-removal-condition
  scenarios.

Issue #3075 acceptance criteria are realised as four scenarios:

1. **Clean state** -- the ignore list documents an unsatisfied REMOVE
   condition (``Cargo.lock resolves `<crate>` to >=<v>`` is NOT yet true)
   -> ``main()`` returns ``0`` with the ``PASS`` banner.
2. **Stale entry** -- an ignore entry whose REMOVE block references a crate
   that is no longer present in ``Cargo.lock`` -> the gate flags the
   absence and returns ``1`` (the entry should be removed).
3. **Missing entry** -- an ignore entry whose REMOVE block constraint IS
   satisfied by the current ``Cargo.lock`` -> the gate flags the
   met-removal-condition and returns ``1``.
4. **Output structure** -- the human-readable stdout carries the expected
   document sections (header, per-entry block, ``PASS`` / ``FAIL``
   banner, advisory IDs) so a downstream tool can reliably parse it.

The script does not ship a ``--json`` flag (issue #3075 originally
considered one but it was deferred); the "structured output" test
therefore pins the textual schema the script does emit. Adding ``--json``
later would extend this test rather than replace it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from textwrap import dedent

import pytest

SCRIPT_NAME = "check_audit_ignores_fresh"


def _scrub_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset ``sys.argv`` so the script's argparse doesn't see pytest's CLI.

    The script's ``main()`` calls ``argparse.ArgumentParser().parse_args()``
    with no explicit args, which defaults to ``sys.argv[1:]``. Under pytest
    that's ``["-v", "--no-cov", ...]`` and argparse rejects the unknown
    flags with exit 2, masking whatever the test was trying to verify.
    """
    monkeypatch.setattr(sys, "argv", [SCRIPT_NAME])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of the audit-ignore freshness script."""
    return load_script(SCRIPT_NAME)


def _redirect(checker, tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """Point the script's ``AUDIT_TOML``, ``CARGO_LOCK``, and ``DENY_TOML``
    at synthetic files in ``tmp_path`` and return both resolved paths.

    All three constants are computed at import time from the script's location
    via ``Path(__file__).resolve().parent.parent``. Each test that wants
    a synthetic fixture must therefore redirect the constants before
    calling ``main()``.
    """
    audit = tmp_path / ".cargo" / "audit.toml"
    lock = tmp_path / "Cargo.lock"
    deny = tmp_path / "deny.toml"
    monkeypatch.setattr(checker, "AUDIT_TOML", audit)
    monkeypatch.setattr(checker, "CARGO_LOCK", lock)
    monkeypatch.setattr(checker, "DENY_TOML", deny)
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    return audit, lock


def _write(p: Path, text: str = "") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dedent(text), encoding="utf-8")
    return p


def _cargo_lock_with(crates: dict[str, str]) -> str:
    """Build a minimal ``Cargo.lock`` resolving each ``crate -> version`` pair.

    Mirrors the package-block shape the parser expects:

        ``[[package]]\\n name = \"<n>\"\\n version = \"<v>\"\\n ...``
    """
    blocks = []
    for name, version in crates.items():
        blocks.append(
            f"[[package]]\n"
            f'name = "{name}"\n'
            f'version = "{version}"\n'
            f'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        )
    return "# synthetic Cargo.lock for check_audit_ignores_fresh tests\n" + "".join(blocks)


# ---------------------------------------------------------------------------
# main() -- clean / stale / missing-removal-condition scenarios
# ---------------------------------------------------------------------------


def test_main_returns_zero_when_ignore_entries_are_still_blocked(
    checker, tmp_path, monkeypatch, capsys
):
    """Clean state: the ignore list documents an unsatisfied REMOVE
    condition -> ``PASS``.

    The REMOVE block says ``Cargo.lock resolves `mycrate` to >=99.0.0`` but
    Cargo.lock resolves mycrate to ``0.1.0`` -- the condition is NOT met, so
    the ignore is still justified and the gate must PASS.
    """
    audit, lock = _redirect(checker, tmp_path, monkeypatch)
    _write(
        audit,
        """\
        [advisories]
        ignore = [
            # >>> REMOVE this entry once Cargo.lock resolves `mycrate` to >=99.0.0
            "RUSTSEC-2026-9000",
        ]
        """,
    )
    _write(lock, _cargo_lock_with({"mycrate": "0.1.0"}))
    _scrub_argv(monkeypatch)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0, f"expected PASS, got rc={rc}\noutput:\n{out}"
    assert "PASS" in out
    assert "RUSTSEC-2026-9000" in out
    # (b) condition NOT met -- script surfaces the lock-version line.
    assert "mycrate" in out
    assert "0.1.0" in out


def test_main_returns_one_when_ignore_entry_crate_absent_from_lock(
    checker, tmp_path, monkeypatch, capsys
):
    """Stale entry: an ignore entry whose REMOVE block references a crate
    that no longer appears in ``Cargo.lock`` -> the gate FLAGS the entry.

    The absence itself is the trigger: the REMOVE block says ``Cargo.lock
    resolves `obsoleted` to >=1.0.0`` but Cargo.lock has no ``obsoleted``
    package at all -- the advisory is cleared from the workspace, so the
    ignore entry should be removed.
    """
    audit, lock = _redirect(checker, tmp_path, monkeypatch)
    _write(
        audit,
        """\
        [advisories]
        ignore = [
            # >>> REMOVE this entry once Cargo.lock resolves `obsoleted` to >=1.0.0
            "RUSTSEC-2026-9001",
        ]
        """,
    )
    # Cargo.lock does NOT contain `obsoleted` -> condition (b) "crate absent" fires.
    _write(lock, _cargo_lock_with({"some-other-crate": "2.0.0"}))
    _scrub_argv(monkeypatch)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1, f"expected FAIL (stale), got rc={rc}\noutput:\n{out}"
    assert "FAIL" in out
    assert "RUSTSEC-2026-9001" in out
    assert "obsoleted" in out
    # The script's "absent from lock" reason text must appear.
    assert "does not contain" in out or "no longer contains" in out


def test_main_returns_one_when_ignore_entry_constraint_is_satisfied(
    checker, tmp_path, monkeypatch, capsys
):
    """Missing entry / constraint met: the ignore list documents a REMOVE
    condition that IS satisfied by the current ``Cargo.lock`` -> the gate
    FLAGS the entry.

    The REMOVE block says ``Cargo.lock resolves `fixed` to >=1.2.0`` and
    Cargo.lock resolves ``fixed`` to ``1.2.0`` -- the condition is met, so
    the ignore entry should be removed.
    """
    audit, lock = _redirect(checker, tmp_path, monkeypatch)
    _write(
        audit,
        """\
        [advisories]
        ignore = [
            # >>> REMOVE this entry once Cargo.lock resolves `fixed` to >=1.2.0
            "RUSTSEC-2026-9002",
        ]
        """,
    )
    _write(lock, _cargo_lock_with({"fixed": "1.2.0"}))
    _scrub_argv(monkeypatch)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1, f"expected FAIL (constraint met), got rc={rc}\noutput:\n{out}"
    assert "FAIL" in out
    assert "RUSTSEC-2026-9002" in out
    # The (b) constraint-met reason text must appear with the version.
    assert "fixed" in out
    assert "1.2.0" in out
    # The summary line lists the surfacable entry.
    assert "removal conditions" in out


def test_main_returns_one_when_ignore_entry_removal_date_is_past(
    checker, tmp_path, monkeypatch, capsys
):
    """Bonus coverage of the (c) date path: an ignore entry whose REMOVE
    block records a removal-date in the past -> the gate FLAGS the entry.

    This complements the two (b)-path tests above and pins the third
    REMOVE-condition branch the script implements.
    """
    audit, lock = _redirect(checker, tmp_path, monkeypatch)
    _write(
        audit,
        """\
        [advisories]
        ignore = [
            # >>> REMOVE this entry on 2000-01-01 (cleanup window elapsed)
            "RUSTSEC-2026-9003",
        ]
        """,
    )
    _write(lock, _cargo_lock_with({}))
    _scrub_argv(monkeypatch)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1, f"expected FAIL (date), got rc={rc}\noutput:\n{out}"
    assert "FAIL" in out
    assert "RUSTSEC-2026-9003" in out
    assert "2000-01-01" in out
    assert "removal-date" in out or "removal date" in out


def test_main_returns_two_when_cargo_lock_missing(
    checker, tmp_path, monkeypatch, capsys
):
    """Missing lock file: ``Cargo.lock`` not present -> ``main()``
    returns ``2`` (script error).

    The script's exit-code contract uses ``2`` for unrecoverable
    pre-conditions (missing / unparseable lock file); the gate must
    surface this rather than silently falling back to PASS.

    Issue #3237 note: the pre-#3237 script returned 2 when
    ``.cargo/audit.toml`` was missing; the post-#3237 script scans both
    ``.cargo/audit.toml`` and ``deny.toml`` and skips missing files, so
    the only remaining ``2`` path is a missing ``Cargo.lock``.
    """
    audit, lock = _redirect(checker, tmp_path, monkeypatch)
    # Only create audit.toml -- Cargo.lock is intentionally absent.
    _write(
        audit,
        """[advisories]
ignore = []
""",
    )
    _scrub_argv(monkeypatch)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 2
    assert "Cargo.lock" in out or "cargo lock" in out.lower()


def test_main_returns_zero_when_audit_toml_missing(
    checker, tmp_path, monkeypatch, capsys
):
    """Missing ``.cargo/audit.toml`` (but present ``Cargo.lock``) ->
    ``main()`` returns ``0``.

    Issue #3237: the script scans both config files and skips missing
    ones, so a missing audit.toml alone is not an error.
    """
    audit, lock = _redirect(checker, tmp_path, monkeypatch)
    # Only create Cargo.lock -- audit.toml is intentionally absent.
    _write(lock, _cargo_lock_with({}))
    _scrub_argv(monkeypatch)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "audit.toml" in out


# ---------------------------------------------------------------------------
# Output structure (issue #3075 acceptance criterion #4)
# ---------------------------------------------------------------------------


def test_main_stdout_contains_expected_structural_sections(
    checker, tmp_path, monkeypatch, capsys
):
    """Issue #3075 acceptance criterion #4: the script's stdout carries
    the expected structural sections (header, per-entry block,
    ``PASS`` / ``FAIL`` banner, advisory IDs) so downstream tooling can
    rely on it.

    The script does not ship a ``--json`` flag (issue #3075 deferred
    that), so this test pins the *textual* schema: the header line, the
    per-entry ``-- RUSTSEC-... --`` separator, the REMOVE-block echo, the
    summary line, and the trailing ``PASS`` / ``FAIL`` banner.
    """
    audit, lock = _redirect(checker, tmp_path, monkeypatch)
    _write(
        audit,
        """\
        [advisories]
        ignore = [
            # >>> REMOVE this entry once Cargo.lock resolves `mycrate` to >=99.0.0
            "RUSTSEC-2026-9100",
        ]
        """,
    )
    _write(lock, _cargo_lock_with({"mycrate": "0.1.0"}))
    _scrub_argv(monkeypatch)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0, out
    # Header banner
    assert "Fluxion Audit-Ignore Freshness Check" in out
    assert "Issue #2912" in out
    # Repo + paths. Issue #3237 changed the per-path header from a single
    # "Audit config:" line to per-file "Scanning <path>..." lines (the
    # script now scans both .cargo/audit.toml and deny.toml).
    assert "Repo:" in out
    assert "Scanning" in out
    assert "audit.toml" in out
    assert "Lock file:" in out
    # Per-entry section
    assert "-- RUSTSEC-2026-9100 (line" in out
    # REMOVE block echo (script prints the first comment line)
    assert "REMOVE" in out
    # Status banner
    assert "STATUS:" in out
    assert "PASS" in out


def test_main_summary_section_is_well_formed(checker, tmp_path, monkeypatch, capsys):
    """The trailing summary section prints a single ``PASS: ...`` or
    ``FAIL: ...`` line followed by the count of advisories, the count of
    packages loaded from Cargo.lock, and a final ``Action: ...`` block on
    FAIL. Pins the summary schema so a regression that drops the
    ``Found N advisory ignore entries`` line is caught.
    """
    audit, lock = _redirect(checker, tmp_path, monkeypatch)
    _write(
        audit,
        """\
        [advisories]
        ignore = [
            # >>> REMOVE this entry once Cargo.lock resolves `obsoleted` to >=1.0.0
            "RUSTSEC-2026-9200",
        ]
        """,
    )
    _write(lock, _cargo_lock_with({"obsoleted": "0.5.0"}))
    _scrub_argv(monkeypatch)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0, out
    # Counts
    assert re.search(r"Found \d+ advisory ignore entries", out), out
    assert re.search(r"Loaded \d+ packages from", out), out
    # PASS summary text (verbatim from the script)
    assert "PASS:" in out
    assert "blocking condition" in out


# ---------------------------------------------------------------------------
# Parsing primitives -- pins the script's parser against synthetic inputs
# ---------------------------------------------------------------------------


def test_parse_ignore_entries_extracts_remove_block(checker):
    """``parse_ignore_entries`` returns one record per advisory with the
    matching ``>>> REMOVE`` comment block attached. The parser is
    exercised directly against a synthetic ``audit.toml`` text so a
    regression in the line-splitting / block-binding logic surfaces
    independent of the workspace ``.cargo/audit.toml``.
    """
    text = (
        "[advisories]\n"
        "ignore = [\n"
        '    # >>> REMOVE this entry once Cargo.lock resolves `a` to >=1.0.0\n'
        '    "RUSTSEC-2026-0001",\n'
        '    # plain trailing comment\n'
        '    "RUSTSEC-2026-0002",\n'
        "]\n"
    )
    entries = checker.parse_ignore_entries(text)
    assert [e["id"] for e in entries] == ["RUSTSEC-2026-0001", "RUSTSEC-2026-0002"]
    # First entry has the REMOVE block attached.
    assert "REMOVE" in entries[0]["remove_block"][0]
    assert "Cargo.lock resolves `a`" in " ".join(entries[0]["remove_block"])
    # Second entry inherits the preceding REMOVE block (parser does NOT
    # reset on every entry -- this is the documented behaviour: multiple
    # consecutive entries share the same block until a new
    # ``>>> REMOVE`` directive appears).
    assert "REMOVE" in entries[1]["remove_block"][0]


def test_parse_ignore_entries_returns_empty_when_no_ignore_block(checker):
    """Audit config without an ``ignore = [...]`` block -> empty list."""
    text = "[advisories]\ninformational_warnings = [\"unmaintained\"]\n"
    assert checker.parse_ignore_entries(text) == []


def test_parse_cargo_lock_versions_handles_missing_file(checker, tmp_path):
    """``parse_cargo_lock_versions`` raises ``FileNotFoundError`` when
    the lock file does not exist (the caller maps this to exit ``2``)."""
    with pytest.raises(FileNotFoundError):
        checker.parse_cargo_lock_versions(tmp_path / "absent.lock")


def test_extract_cargo_lock_constraint_parses_canonical_phrase(checker):
    """The constraint extractor recognises the documented
    ``Cargo.lock resolves `<crate>` to >=<version>`` phrase used by
    every REMOVE block in ``.cargo/audit.toml``.
    """
    block = [
        "# >>> REMOVE this entry once Cargo.lock resolves `rustls-webpki` to >=0.103.13"
    ]
    assert checker.extract_cargo_lock_constraint(block) == ("rustls-webpki", "0.103.13")


def test_extract_absence_constraint_detects_no_patched_versions(checker):
    """The absence extractor recognises the documented
    ``no patched versions`` phrase paired with a crate name.
    """
    block = [
        "# `paste` has no patched versions; drop this ignore once the",
        "# upstream chain migrates off paste.",
    ]
    assert checker.extract_absence_constraint(block) == "paste"


def test_extract_removal_date_parses_iso_8601(checker):
    """The removal-date extractor parses ISO-8601 dates from the REMOVE
    block."""
    from datetime import date

    block = ["# >>> REMOVE this entry on 2026-12-31 (cleanup window)"]
    assert checker.extract_removal_date(block) == date(2026, 12, 31)


def test_version_gte_handles_dotted_triple_semver(checker):
    """The semver-gte helper compares dotted triples correctly."""
    assert checker.version_gte("1.2.3", "1.2.0") is True
    assert checker.version_gte("1.2.0", "1.2.3") is False
    assert checker.version_gte("1.2.3", "1.2.3") is True
    assert checker.version_gte("0.103.13", "0.103.12") is True
    assert checker.version_gte("0.102.8", "0.103.10") is False