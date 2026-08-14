"""
Tests for ``scripts/check_audit_config_unique.py`` -- Issue #2773.

Regression guard for the ``audit-config-uniqueness`` gate
(``scripts/check_audit_config_unique.py``, added in #2773). Mirrors the
``load_script`` + ``tmp_path`` mock-repo pattern from
``test_check_physics_sim_cycle.py`` / ``test_check_ashrae_cases_cycle.py``:

* load the script as a fresh module via the shared ``load_script`` fixture,
* monkey-patch its module-level ``REPO_ROOT`` to point at a synthetic
  ``tmp_path`` tree, then
* drive ``find_audit_tomls()`` + ``main()`` through clean and offender
  scenarios.

Acceptance criteria from the issue body are realised as follows:

* (a) *clean* fixture (only the canonical ``.cargo/audit.toml``) ->
    ``main()`` returns ``0`` and prints ``PASS``.
* (b) *duplicate* fixture (both root ``audit.toml`` and
    ``.cargo/audit.toml``) -> ``main()`` returns ``1`` and prints the
    duplicate count.
* (c) *ignore* / root-stray fixture (a root-level ``audit.toml`` without
    the canonical file) -> ``main()`` returns ``1``, tags the file as
    ``STRAY``, and prints the remediation block.
* (d) *missing* fixture (no ``audit.toml`` anywhere) -> ``main()``
    returns ``1`` and prints ``Found 0`` followed by remediation.

A final pair of tests pin the script's behavior against the real
``fluxion`` checkout so a scanner regression surfaces locally before it
does in CI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_NAME = "check_audit_config_unique"
SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "check_audit_config_unique.py"
)


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of the audit-config uniqueness script."""
    return load_script(SCRIPT_NAME)


def _write(p: Path, text: str = "") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _redirect(checker, tmp_path, monkeypatch) -> None:
    """Point the script's module-level ``REPO_ROOT`` at a synthetic tree.

    The script resolves ``REPO_ROOT`` once at import time from the script's
    own location, so the freshly-loaded module carries the *real* repo
    root. Each test that wants a synthetic fixture must therefore redirect
    the constant before calling ``find_audit_tomls()`` / ``main()``.
    """
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)


# ---------------------------------------------------------------------------
# find_audit_tomls() — the core scanner
# ---------------------------------------------------------------------------


def test_find_returns_only_canonical_when_clean(checker, tmp_path, monkeypatch):
    """Clean fixture: only ``.cargo/audit.toml`` -> exactly one match."""
    _redirect(checker, tmp_path, monkeypatch)
    _write(tmp_path / ".cargo" / "audit.toml", "[advisories]\nignore = []\n")
    found = checker.find_audit_tomls()
    assert len(found) == 1
    assert found[0] == tmp_path / ".cargo" / "audit.toml"


def test_find_returns_two_when_canonical_and_stray(checker, tmp_path, monkeypatch):
    """Both root ``audit.toml`` and ``.cargo/audit.toml`` -> two matches.

    This is the "duplicate" case from the issue acceptance criteria: two
    ``audit.toml`` files in the canonical *and* root-level locations.
    """
    _redirect(checker, tmp_path, monkeypatch)
    _write(tmp_path / ".cargo" / "audit.toml", "[advisories]\nignore = []\n")
    _write(tmp_path / "audit.toml", "[advisories]\nignore = []\n")
    found = checker.find_audit_tomls()
    assert len(found) == 2
    assert tmp_path / ".cargo" / "audit.toml" in found
    assert tmp_path / "audit.toml" in found


def test_find_returns_only_root_when_no_cargo_dir(checker, tmp_path, monkeypatch):
    """No ``.cargo/`` directory -> only root-level matches count."""
    _redirect(checker, tmp_path, monkeypatch)
    _write(tmp_path / "audit.toml", "[advisories]\nignore = []\n")
    found = checker.find_audit_tomls()
    assert len(found) == 1
    assert found[0] == tmp_path / "audit.toml"


def test_find_returns_empty_when_no_files(checker, tmp_path, monkeypatch):
    """Empty repo + empty ``.cargo/`` -> no ``audit.toml`` files at all."""
    _redirect(checker, tmp_path, monkeypatch)
    _write(tmp_path / ".cargo" / "config.toml", "noop\n")  # wrong name
    _write(tmp_path / "src" / "lib.rs", "noop\n")
    assert checker.find_audit_tomls() == []


def test_find_is_non_recursive_in_cargo(checker, tmp_path, monkeypatch):
    """A nested ``.cargo/sub/audit.toml`` must NOT be detected (non-recursive).

    Without this guard, a future contributor could accidentally regress the
    scanner into a recursive walk that double-counts nested copies.
    """
    _redirect(checker, tmp_path, monkeypatch)
    _write(tmp_path / ".cargo" / "audit.toml", "[advisories]\nignore = []\n")
    _write(
        tmp_path / ".cargo" / "nested" / "audit.toml",
        "[advisories]\nignore = []\n",
    )
    found = checker.find_audit_tomls()
    assert len(found) == 1
    assert found[0] == tmp_path / ".cargo" / "audit.toml"


def test_find_is_non_recursive_at_root(checker, tmp_path, monkeypatch):
    """A nested ``subdir/audit.toml`` at the repo root must NOT be detected."""
    _redirect(checker, tmp_path, monkeypatch)
    _write(tmp_path / ".cargo" / "audit.toml", "[advisories]\nignore = []\n")
    _write(tmp_path / "subdir" / "audit.toml", "[advisories]\nignore = []\n")
    found = checker.find_audit_tomls()
    assert len(found) == 1
    assert found[0] == tmp_path / ".cargo" / "audit.toml"


def test_find_ignores_almost_matches(checker, tmp_path, monkeypatch):
    """Files like ``audit.toml.bak`` / ``my-audit.toml`` must NOT trip the scanner."""
    _redirect(checker, tmp_path, monkeypatch)
    _write(tmp_path / ".cargo" / "audit.toml", "[advisories]\nignore = []\n")
    _write(tmp_path / ".cargo" / "audit.toml.bak", "stale backup\n")
    _write(tmp_path / "my-audit.toml", "wrong name\n")
    _write(tmp_path / "audit.toml.example", "template\n")
    found = checker.find_audit_tomls()
    assert len(found) == 1
    assert found[0] == tmp_path / ".cargo" / "audit.toml"


def test_find_returns_sorted(checker, tmp_path, monkeypatch):
    """``find_audit_tomls`` must return its result sorted for stable output."""
    _redirect(checker, tmp_path, monkeypatch)
    _write(tmp_path / ".cargo" / "audit.toml", "[advisories]\nignore = []\n")
    _write(tmp_path / "audit.toml", "[advisories]\nignore = []\n")
    found = checker.find_audit_tomls()
    # Sorted lexicographically: ".cargo/audit.toml" < "audit.toml"
    assert found == sorted(found)
    assert found[0].name == ".cargo" or str(found[0]).endswith("/.cargo/audit.toml")


def test_find_ignores_directory_named_audit_toml(checker, tmp_path, monkeypatch):
    """A directory named ``audit.toml`` is not a file -> skipped.

    Without the ``p.is_file()`` guard, the script would misclassify an
    empty directory as a stray config and emit a phantom FAIL.
    """
    _redirect(checker, tmp_path, monkeypatch)
    _write(tmp_path / ".cargo" / "audit.toml", "[advisories]\nignore = []\n")
    # Also create a directory literally named `audit.toml` at the repo
    # root, plus a non-empty sibling file so ``iterdir`` doesn't yield
    # nothing on some platforms.
    (tmp_path / "audit.toml").mkdir()
    _write(tmp_path / "audit.toml" / "notes.txt", "empty config dir\n")
    found = checker.find_audit_tomls()
    assert found == [tmp_path / ".cargo" / "audit.toml"]


# ---------------------------------------------------------------------------
# main() — exit codes + diagnostic output
# ---------------------------------------------------------------------------


def test_main_returns_zero_on_clean_canonical(checker, tmp_path, monkeypatch, capsys):
    """Clean fixture -> exit 0 with the PASS banner.

    Realisation of acceptance criterion (a): exactly one canonical
    ``.cargo/audit.toml``, no root-level stray -> PASS.
    """
    _redirect(checker, tmp_path, monkeypatch)
    _write(tmp_path / ".cargo" / "audit.toml", "[advisories]\nignore = []\n")
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "PASS" in out
    assert "CANONICAL" in out
    assert "Found 1" in out
    assert "Exactly one" in out


def test_main_returns_one_on_duplicate(checker, tmp_path, monkeypatch, capsys):
    """Duplicate (canonical + stray root) -> exit 1 with the FAIL banner.

    Realisation of acceptance criterion (b): a root-level ``audit.toml``
    shadows or contradicts the canonical file, so the gate must FAIL when
    both are present.
    """
    _redirect(checker, tmp_path, monkeypatch)
    _write(tmp_path / ".cargo" / "audit.toml", "[advisories]\nignore = []\n")
    _write(tmp_path / "audit.toml", "[advisories]\nignore = []\n")  # stray
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out
    assert "Found 2" in out
    assert "STRAY" in out
    assert "audit.toml" in out


def test_main_returns_one_on_root_stray_only(checker, tmp_path, monkeypatch, capsys):
    """Only a root-level stray (canonical missing) -> exit 1.

    Realisation of acceptance criterion (c): a stray root file with no
    canonical counterpart is a "stray" because cargo-audit reads
    ``.cargo/audit.toml`` only. The gate must surface this with the
    STRAY tag and remediation guidance.
    """
    _redirect(checker, tmp_path, monkeypatch)
    _write(tmp_path / "audit.toml", "[advisories]\nignore = []\n")
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "STRAY" in out
    assert "Found 1" in out
    # Remediation block must include "git rm" guidance (issue #2773).
    assert "Remediation" in out
    assert "git rm" in out
    # Canonical must be flagged as missing.
    assert "Expected exactly one" in out


def test_main_returns_one_when_no_files(checker, tmp_path, monkeypatch, capsys):
    """No ``audit.toml`` files anywhere -> exit 1.

    Realisation of acceptance criterion (d): the canonical config is
    missing entirely. The gate must FAIL and point the operator at the
    canonical-path requirement.
    """
    _redirect(checker, tmp_path, monkeypatch)
    _write(tmp_path / ".cargo" / "config.toml", "noop\n")
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "Found 0" in out
    assert "Remediation" in out


def test_main_failure_prints_remediation(checker, tmp_path, monkeypatch, capsys):
    """Every FAIL path must emit the issue-#2773 remediation block."""
    _redirect(checker, tmp_path, monkeypatch)
    _write(tmp_path / "audit.toml", "[advisories]\nignore = []\n")
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "Remediation" in out
    # Issue-#2773 / AGENTS.md §Toolchain Quirks wording must appear.
    assert "cargo-audit" in out
    assert ".cargo/audit.toml" in out


def test_main_tags_canonical_vs_stray(checker, tmp_path, monkeypatch, capsys):
    """Both files present -> the canonical one is tagged CANONICAL, root stray STRAY."""
    _redirect(checker, tmp_path, monkeypatch)
    _write(tmp_path / ".cargo" / "audit.toml", "[advisories]\nignore = []\n")
    _write(tmp_path / "audit.toml", "[advisories]\nignore = []\n")
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1
    # CANONICAL must precede STRAY (or at least both appear with the
    # correct paths).
    assert "[CANONICAL] .cargo/audit.toml" in out
    assert "[STRAY] audit.toml" in out


# ---------------------------------------------------------------------------
# Real-repo pinning (gate's self-test)
# ---------------------------------------------------------------------------


def test_main_returns_zero_on_real_repo(checker, repo_root, capsys):
    """The real checkout must pass -- a scanner regression flips this red.

    This is the gate's own self-test: if a regression in the scanner
    flips the real repo from clean to failing, this test fails locally
    before CI does. Run with the un-monkey-patched module so the script
    scans ``repo_root`` (the real repo root).
    """
    # Sanity: the freshly-loaded module's REPO_ROOT is the real root.
    assert checker.REPO_ROOT == repo_root
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0, (
        f"Real repo failed uniqueness check:\n{out}\n"
        "If this regressed, look for a stray `audit.toml` at the repo "
        "root or a missing `.cargo/audit.toml`."
    )
    assert "PASS" in out
    assert "Found 1" in out
    assert "[CANONICAL]" in out


def test_subprocess_passes_against_real_repo():
    """Subprocess smoke test against the committed ``.cargo/audit.toml``.

    Guards the ``__main__`` wrapper contract: ``python3 script.py`` must
    exit 0 against the real checkout, and an in-process call would not
    catch a regression in the ``try/except`` translation logic.
    """
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, (
        f"Script exited {result.returncode} against real repo.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "PASS" in result.stdout


# ---------------------------------------------------------------------------
# __main__ wrapper contract
# ---------------------------------------------------------------------------


def test_main_wrapper_delegates_to_sys_exit():
    """The ``if __name__ == '__main__'`` block must route ``main()`` through ``sys.exit()``.

    Reads the script tail directly and asserts the wrapper contract:
    ``sys.exit(main())`` is wrapped in ``try/except`` so unhandled
    exceptions become exit ``2`` rather than a traceback. A regression
    that drops the wrapper would silently change the exit contract for
    ``python3 script.py`` (e.g. a Python exception would no longer
    surface as a non-zero exit code).
    """
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in src
    assert "sys.exit(main())" in src
    wrapper = src.split('if __name__ == "__main__":', 1)[1]
    assert "try:" in wrapper
    assert "except Exception" in wrapper
    assert "sys.exit(2)" in wrapper
