"""
Tests for ``scripts/check_audit_deny_agree.py`` -- Issue #3654.

Regression guard for the audit↔deny advisory-agreement gate
(``scripts/check_audit_deny_agree.py``). Mirrors the ``load_script`` +
``tmp_path`` mock-repo pattern from ``test_check_audit_config_unique.py``:

* load the script as a fresh module via the shared ``load_script`` fixture,
* monkey-patch its module-level ``AUDIT_TOML`` / ``DENY_TOML`` constants to
  point at a synthetic ``tmp_path`` tree, then
* drive the parser helpers and ``main()`` through clean and drift
  scenarios.

Acceptance criteria from the issue body are realised as follows:

* (a) *agreement* fixtures (identical sets; or audit-only entries carrying
    the ``deny-scope-exempt`` marker under ``unmaintained = "workspace"``)
    -> ``main()`` returns ``0`` and prints ``PASS``.
* (b) *deny-only* fixture (an advisory ignored in ``deny.toml`` but missing
    from ``.cargo/audit.toml``) -> ``main()`` returns ``1`` and names the
    advisory as missing.
* (c) *unmarked audit-only* fixture -> ``main()`` returns ``1`` and prints
    the marker/mirror remediation guidance.
* (d) *stale exemption* fixtures (marker present but deny.toml's
    ``unmaintained`` scope is not ``"workspace"``) -> ``main()`` returns
    ``1``.
* (e) *structural* fixtures (missing file, missing ``ignore`` block,
    inline-empty list, commented-out-only entries, duplicate ids,
    marker-scoping) pin the parser's behaviour.

A final test pins the script against the real ``fluxion`` checkout so a
parser regression surfaces locally before it does in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPT_NAME = "check_audit_deny_agree"
SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "check_audit_deny_agree.py"
)


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of the audit↔deny agreement script."""
    return load_script(SCRIPT_NAME)


def _write(p: Path, text: str = "") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


AUDIT_EMPTY = '[advisories]\nignore = []\n'
DENY_EMPTY = (
    "[advisories]\n"
    'unmaintained = "workspace"\n'
    "ignore = []\n"
)


def _mock_repo(
    checker,
    tmp_path,
    monkeypatch,
    audit_text: str,
    deny_text: str,
) -> None:
    """Write both config files and redirect the script's path constants.

    The script resolves ``AUDIT_TOML`` / ``DENY_TOML`` once at import time
    from its own location, so the freshly-loaded module carries the *real*
    repo paths. Each test that wants a synthetic fixture must redirect the
    constants before calling the parser or ``main()``.
    """
    _write(tmp_path / ".cargo" / "audit.toml", audit_text)
    _write(tmp_path / "deny.toml", deny_text)
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "AUDIT_TOML", tmp_path / ".cargo" / "audit.toml")
    monkeypatch.setattr(checker, "DENY_TOML", tmp_path / "deny.toml")



def _scrub_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset ``sys.argv`` so the script's argparse doesn't see pytest's CLI.

    Mirrors the same-named helper in ``test_check_audit_ignores_fresh.py``:
    the script's ``main()`` calls ``argparse.parse_args()`` with no explicit
    args, which defaults to ``sys.argv[1:]`` — under pytest that would be
    the test-file path and fail with "unrecognized arguments".
    """
    monkeypatch.setattr(sys, "argv", [SCRIPT_NAME])


# ---------------------------------------------------------------------------
# parse_ignore_entries() — the core parser
# ---------------------------------------------------------------------------


def test_parse_finds_active_entries_with_lines(checker):
    """Active quoted entries are captured with 1-indexed line numbers."""
    text = (
        "[advisories]\n"
        "ignore = [\n"
        '    # reason A\n'
        '    "RUSTSEC-2024-0436",\n'
        '    "RUSTSEC-2026-0192",\n'
        "]\n"
    )
    entries = checker.parse_ignore_entries(text)
    assert [(e["id"], e["line"]) for e in entries] == [
        ("RUSTSEC-2024-0436", 4),
        ("RUSTSEC-2026-0192", 5),
    ]
    assert all(not e["exempt"] for e in entries)


def test_parse_ignores_commented_out_entries(checker):
    """Commented-out entries are documentation, not enforcement (#3654)."""
    text = (
        "[advisories]\n"
        "ignore = [\n"
        '    # "RUSTSEC-2024-0436",\n'
        '    #   - RUSTSEC-2026-0192 was removed in #2749\n'
        "]\n"
    )
    assert checker.parse_ignore_entries(text) == []


def test_parse_inline_empty_list(checker):
    """``ignore = []`` on a single line parses to no entries, not a error."""
    assert checker.parse_ignore_entries("[advisories]\nignore = []\n") == []


def test_parse_missing_block_raises(checker):
    """A config without an ``ignore`` block is a parse error (exit 2 path)."""
    with pytest.raises(ValueError, match="ignore"):
        checker.parse_ignore_entries("[advisories]\nunmaintained = \"all\"\n")


def test_parse_exempt_marker_on_preceding_comment(checker):
    """The ``deny-scope-exempt`` token in the preceding comment marks the entry."""
    text = (
        "[advisories]\n"
        "ignore = [\n"
        "    # paste is unmaintained, transitive via gemm.\n"
        "    # deny-scope-exempt (transitive-unmaintained)\n"
        '    "RUSTSEC-2024-0436",\n'
        "]\n"
    )
    (entry,) = checker.parse_ignore_entries(text)
    assert entry["exempt"] is True


def test_parse_exempt_marker_on_same_line(checker):
    """The marker is also recognised in the entry's trailing comment."""
    text = (
        "[advisories]\n"
        "ignore = [\n"
        '    "RUSTSEC-2024-0411",  # gdkwayland-sys (deny-scope-exempt)\n'
        "]\n"
    )
    (entry,) = checker.parse_ignore_entries(text)
    assert entry["exempt"] is True


def test_parse_marker_applies_only_to_next_entry(checker):
    """The comment buffer resets per entry: a marker must not leak forward.

    Without the reset, one marker line would silently exempt every later
    entry in the list and hide real drift.
    """
    text = (
        "[advisories]\n"
        "ignore = [\n"
        "    # deny-scope-exempt (transitive-unmaintained)\n"
        '    "RUSTSEC-2024-0436",\n'
        "    # plain reason, no marker\n"
        '    "RUSTSEC-2026-0192",\n'
        "]\n"
    )
    first, second = checker.parse_ignore_entries(text)
    assert first["exempt"] is True
    assert second["exempt"] is False


def test_parse_unmaintained_scope(checker):
    """The deny.toml ``unmaintained`` scope value is extracted."""
    assert checker.parse_unmaintained_scope('unmaintained = "workspace"\n') == "workspace"
    assert checker.parse_unmaintained_scope('  unmaintained = "all"\n') == "all"
    assert checker.parse_unmaintained_scope("[advisories]\nignore = []\n") is None


# ---------------------------------------------------------------------------
# main() — exit codes + diagnostic output
# ---------------------------------------------------------------------------


def test_main_passes_on_identical_sets(checker, tmp_path, monkeypatch, capsys):
    """Identical ignore sets -> exit 0 + PASS."""
    _scrub_argv(monkeypatch)
    _mock_repo(
        checker,
        tmp_path,
        monkeypatch,
        '[advisories]\nignore = [\n    "RUSTSEC-2024-0436",\n]\n',
        '[advisories]\nunmaintained = "workspace"\nignore = [\n    "RUSTSEC-2024-0436",\n]\n',
    )
    assert checker.main() == 0
    assert "PASS" in capsys.readouterr().out


def test_main_passes_on_marker_exempt_audit_only(checker, tmp_path, monkeypatch, capsys):
    """Audit-only entry WITH the marker under workspace scope -> exit 0.

    This is the repository's steady state: all transitive-unmaintained
    ignores live in ``.cargo/audit.toml`` only, each marked exempt.
    """
    _scrub_argv(monkeypatch)
    _mock_repo(
        checker,
        tmp_path,
        monkeypatch,
        '[advisories]\nignore = [\n    # deny-scope-exempt (transitive-unmaintained)\n    "RUSTSEC-2024-0436",\n]\n',
        DENY_EMPTY,
    )
    assert checker.main() == 0
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "1 deny-scope-exempt" in out


def test_main_fails_on_deny_only_entry(checker, tmp_path, monkeypatch, capsys):
    """deny.toml-only ignore -> exit 1 and names the advisory as missing."""
    _scrub_argv(monkeypatch)
    _mock_repo(
        checker,
        tmp_path,
        monkeypatch,
        AUDIT_EMPTY,
        '[advisories]\nunmaintained = "workspace"\nignore = [\n    "RUSTSEC-2025-0134",\n]\n',
    )
    assert checker.main() == 1
    out = capsys.readouterr().out
    assert "RUSTSEC-2025-0134" in out
    assert "MISSING from .cargo/audit.toml" in out


def test_main_fails_on_unmarked_audit_only_entry(checker, tmp_path, monkeypatch, capsys):
    """Audit-only entry WITHOUT the marker -> exit 1 with remediation."""
    _scrub_argv(monkeypatch)
    _mock_repo(
        checker,
        tmp_path,
        monkeypatch,
        '[advisories]\nignore = [\n    "RUSTSEC-2024-0436",\n]\n',
        DENY_EMPTY,
    )
    assert checker.main() == 1
    out = capsys.readouterr().out
    assert "RUSTSEC-2024-0436" in out
    assert "deny-scope-exempt" in out
    assert "FAIL" in out


def test_main_fails_when_marker_present_but_scope_is_all(checker, tmp_path, monkeypatch, capsys):
    """Exempt entries are invalid once deny.toml enforces all unmaintained."""
    _scrub_argv(monkeypatch)
    _mock_repo(
        checker,
        tmp_path,
        monkeypatch,
        '[advisories]\nignore = [\n    # deny-scope-exempt (transitive-unmaintained)\n    "RUSTSEC-2024-0436",\n]\n',
        '[advisories]\nunmaintained = "all"\nignore = []\n',
    )
    assert checker.main() == 1
    out = capsys.readouterr().out
    assert 'unmaintained = "all"' in out


def test_main_fails_when_marker_present_but_scope_missing(checker, tmp_path, monkeypatch, capsys):
    """A missing ``unmaintained`` key invalidates the exemption contract."""
    _scrub_argv(monkeypatch)
    _mock_repo(
        checker,
        tmp_path,
        monkeypatch,
        '[advisories]\nignore = [\n    # deny-scope-exempt (transitive-unmaintained)\n    "RUSTSEC-2024-0436",\n]\n',
        "[advisories]\nignore = []\n",
    )
    assert checker.main() == 1
    assert "<unset>" in capsys.readouterr().out


def test_main_fails_on_duplicate_id_in_audit(checker, tmp_path, monkeypatch, capsys):
    """Duplicate ids within one file hide copy-paste drift -> exit 1."""
    _scrub_argv(monkeypatch)
    _mock_repo(
        checker,
        tmp_path,
        monkeypatch,
        '[advisories]\nignore = [\n    "RUSTSEC-2024-0436",\n    # deny-scope-exempt\n    "RUSTSEC-2024-0436",\n]\n',
        DENY_EMPTY,
    )
    assert checker.main() == 1
    out = capsys.readouterr().out
    assert "more than once in .cargo/audit.toml" in out


def test_main_fails_on_missing_audit_file(checker, tmp_path, monkeypatch, capsys):
    """Missing ``.cargo/audit.toml`` -> exit 2 (fail-closed)."""
    _scrub_argv(monkeypatch)
    _write(tmp_path / "deny.toml", DENY_EMPTY)
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "AUDIT_TOML", tmp_path / ".cargo" / "audit.toml")
    monkeypatch.setattr(checker, "DENY_TOML", tmp_path / "deny.toml")
    with pytest.raises(SystemExit) as exc:
        checker.main()
    assert exc.value.code == 2


def test_main_fails_on_unparseable_deny_block(checker, tmp_path, monkeypatch, capsys):
    """deny.toml without an ``ignore`` block -> exit 2 (fail-closed)."""
    _scrub_argv(monkeypatch)
    _write(tmp_path / ".cargo" / "audit.toml", AUDIT_EMPTY)
    _write(tmp_path / "deny.toml", "[licenses]\nallow = []\n")
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "AUDIT_TOML", tmp_path / ".cargo" / "audit.toml")
    monkeypatch.setattr(checker, "DENY_TOML", tmp_path / "deny.toml")
    with pytest.raises(SystemExit) as exc:
        checker.main()
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# Pin against the real checkout
# ---------------------------------------------------------------------------


def test_real_repo_passes(checker, capsys):
    """The real fluxion checkout must satisfy the gate at head.

    Guards against a scanner regression that would flip the repo from
    clean to failing (or silently exempt drift) without any test
    noticing — mirrors the real-repo pin in
    ``test_check_audit_config_unique.py``.
    """
    assert SCRIPT_PATH.exists()
    assert checker.AUDIT_TOML.exists()
    assert checker.DENY_TOML.exists()
    rc = checker.main(argv=[])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PASS" in out
