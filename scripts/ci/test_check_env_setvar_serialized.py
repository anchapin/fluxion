"""Tests for ``scripts/check_env_setvar_serialized.py`` -- Issue #3453.

The guard enforces the ``ENV_LOCK`` (or any ``*_LOCK: ... Mutex ...``)
serialisation convention on every ``tests/**/*.rs`` file that mutates a
process-wide env var (``std::env::set_var`` / ``std::env::remove_var``).
Until #3453 the convention was carried by only three integration
binaries (``onnx_signature_integration``, ``email_notifier_header_safety``,
``ai/surrogate.rs::tests``); the ``ashrae_140_diagnostic_integration_test``
binary shipped with bare ``env::set_var`` / ``env::remove_var`` calls and
fell through the ADR-0014 nextest-migration safety net.

These tests drive the matcher's documented invariants against hermetic
``tmp_path`` mock-repo fixtures so a regex regression (e.g. the env-mutation
detector losing the ``\\b(?:std::)?env::`` prefix, or the lock-declaration
matcher missing ``_LOCK`` suffixes, or ``main()`` exit-code contract
flipping) fails here instead of on a real PR.

Mirrors the ``load_script`` + ``tmp_path`` mock-repo pattern from
``test_check_concurrency_keys.py`` (#3444):

* ``has_env_mutation`` / ``has_lock_declaration`` -- the regex primitives,
* ``check_test_file`` -- per-file findings for each planted violation,
* ``iter_test_files`` -- directory walk + skip-list semantics,
* ``main()`` exit-code contract: 0 clean / 1 drift / 2 script error.

All fixture files are synthetic ``.rs`` files written under ``tmp_path`` --
no test depends on the network or on the repo's real test tree (the
end-to-end against the real tree is the direct
``check_env_setvar_serialized.py`` invocation in ``scripts-tests.yml``).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

SCRIPT_NAME = "check_env_setvar_serialized"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of ``scripts/check_env_setvar_serialized.py``."""
    return load_script(SCRIPT_NAME)


def _write_test_file(tmp_path: Path, name: str, body: str) -> Path:
    """Write a synthetic ``tests/<name>.rs`` file into ``tmp_path/tests``.

    The fixture is laid out so ``iter_test_files`` discovers the file
    under a ``tests/`` root (the script's ``TESTS_DIR`` constant) --
    redirect the constant to ``tmp_path`` before invoking the matcher.
    """
    target = tmp_path / "tests" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(body).lstrip("\n"), encoding="utf-8")
    return target


def _redirect(checker, monkeypatch, tmp_path: Path) -> Path:
    """Point the checker's module constants at a ``tmp_path`` mock repo.

    Returns the redirected ``tests/`` directory so callers can add
    synthetic files via ``_write_test_file``.
    """
    tests_dir = tmp_path / "tests"
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "TESTS_DIR", tests_dir)
    return tests_dir


def _findings_for(checker, monkeypatch, tmp_path, name: str, body: str) -> list:
    """Return ``check_test_file`` findings for a single synthetic test file."""
    _redirect(checker, monkeypatch, tmp_path)
    return checker.check_test_file(_write_test_file(tmp_path, name, body))


# The compliant shape mirrors the convention carried by
# ``tests/onnx_signature_integration.rs:25``: a `static ENV_LOCK` decl
# plus an env-mutation call site guarded by ``.lock().unwrap_or_else``.
_COMPLIANT_FILE = """\
    use std::sync::Mutex;
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn example() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("FOO", "1");
        std::env::remove_var("FOO");
    }
"""

# Variant: an alternate lock name (e.g. `SHUTDOWN_ENV_LOCK` from
# `src/api/server.rs:4730`). Must satisfy the matcher.
_COMPLIANT_FILE_ALT_NAME = """\
    static SHUTDOWN_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn example() {
        let _guard = SHUTDOWN_ENV_LOCK.lock().unwrap();
        std::env::set_var("FOO", "1");
    }
"""

# Exempt shape: a tests/*.rs file that does NOT call env::set_var /
# env::remove_var at all -- the matcher must skip it.
_EXEMPT_FILE = """\
    #[test]
    fn pure_rust_logic() {
        assert_eq!(2 + 2, 4);
    }
"""

# Invariant 1 violation: env mutation present, NO lock declaration.
_UNGUARDED_FILE = """\
    #[test]
    fn bad() {
        std::env::set_var("FOO", "1");
        std::env::remove_var("FOO");
    }
"""

# Invariant 2 violation: a `static ENV_LOCK` declaration exists but its
# type is NOT a Mutex (e.g. a `RwLock` placeholder, or a typo'd name).
# The matcher requires the type to contain `Mutex`.
_WRONG_TYPE_FILE = """\
    static ENV_LOCK: RwLock<()> = RwLock::new(());

    #[test]
    fn bad() {
        let _g = ENV_LOCK.read().unwrap();
        std::env::set_var("FOO", "1");
    }
"""

# Invariant 3 violation: env mutation but no `_LOCK`-suffix static.
# Demonstrates that a generic `Mutex` import does NOT satisfy the gate.
_WRONG_NAME_FILE = """\
    use std::sync::Mutex;

    #[test]
    fn bad() {
        let m = Mutex::new(());
        let _g = m.lock().unwrap();
        std::env::set_var("FOO", "1");
    }
"""


# ---------------------------------------------------------------------------
# has_env_mutation
# ---------------------------------------------------------------------------


def test_has_env_mutation_detects_set_var(checker):
    assert checker.has_env_mutation(
        "std::env::set_var(\"FOO\", \"1\");"
    ) is True


def test_has_env_mutation_detects_remove_var(checker):
    assert checker.has_env_mutation(
        "std::env::remove_var(\"FOO\");"
    ) is True


def test_has_env_mutation_detects_unqualified(checker):
    """Without `std::` prefix -- e.g. inside `use std::env;` scope."""
    assert checker.has_env_mutation(
        "env::set_var(\"FOO\", \"1\");"
    ) is True


def test_has_env_mutation_ignores_module_declaration(checker):
    """`use std::env;` and `use env;` are NOT call sites."""
    assert checker.has_env_mutation("use std::env;") is False
    assert checker.has_env_mutation("use env;") is False


def test_has_env_mutation_ignores_identifier_substring(checker):
    """A variable named `env_set_var` must not satisfy the matcher."""
    assert checker.has_env_mutation("let env_set_var = 1;") is False


def test_has_env_mutation_ignores_strings(checker):
    """A docstring containing `env::set_var` prose must not satisfy the
    matcher because the regex anchors on the closing `(`."""
    assert checker.has_env_mutation(
        "// env::set_var is unsafe to call concurrently."
    ) is False


# ---------------------------------------------------------------------------
# has_lock_declaration
# ---------------------------------------------------------------------------


def test_has_lock_declaration_accepts_ENV_LOCK(checker):
    """The canonical convention: `static ENV_LOCK: Mutex<()> = ...`."""
    assert checker.has_lock_declaration(
        "static ENV_LOCK: Mutex<()> = Mutex::new(());"
    ) is True


def test_has_lock_declaration_accepts_std_sync_mutex(checker):
    """`static SHUTDOWN_ENV_LOCK: std::sync::Mutex<()> = ...`."""
    assert checker.has_lock_declaration(
        "static SHUTDOWN_ENV_LOCK: std::sync::Mutex<()> = "
        "std::sync::Mutex::new(());"
    ) is True


def test_has_lock_declaration_rejects_non_lock_static(checker):
    """A `static FOO: Mutex<()>` (no `_LOCK` suffix) must NOT satisfy
    the matcher -- the suffix is the load-bearing convention marker."""
    assert checker.has_lock_declaration(
        "static FOO: Mutex<()> = Mutex::new(());"
    ) is False


def test_has_lock_declaration_rejects_wrong_type(checker):
    """A `static ENV_LOCK: RwLock<()>` (no `Mutex` in type) fails."""
    assert checker.has_lock_declaration(
        "static ENV_LOCK: RwLock<()> = RwLock::new(());"
    ) is False


def test_has_lock_declaration_rejects_lowercase(checker):
    """Lowercase `env_lock` -- the convention is SCREAMING_SNAKE_CASE."""
    assert checker.has_lock_declaration(
        "static env_lock: Mutex<()> = Mutex::new(());"
    ) is False


# ---------------------------------------------------------------------------
# check_test_file -- per-file findings
# ---------------------------------------------------------------------------


def test_check_test_file_exempt_when_no_env_mutation(
    checker, monkeypatch, tmp_path
):
    """A test file with no env::set_var calls is exempt from the gate."""
    assert _findings_for(
        checker, monkeypatch, tmp_path,
        "pure.rs", _EXEMPT_FILE,
    ) == []


def test_check_test_file_compliant_has_no_findings(
    checker, monkeypatch, tmp_path
):
    assert _findings_for(
        checker, monkeypatch, tmp_path,
        "ok.rs", _COMPLIANT_FILE,
    ) == []


def test_check_test_file_alt_lock_name_accepted(
    checker, monkeypatch, tmp_path
):
    """An alternate `<NAME>_LOCK` name must satisfy the gate."""
    assert _findings_for(
        checker, monkeypatch, tmp_path,
        "alt.rs", _COMPLIANT_FILE_ALT_NAME,
    ) == []


def test_check_test_file_reports_unguarded_mutation(
    checker, monkeypatch, tmp_path
):
    """The Issue #3453 violation: env mutation, no lock declaration."""
    findings = _findings_for(
        checker, monkeypatch, tmp_path,
        "bad.rs", _UNGUARDED_FILE,
    )
    assert len(findings) == 1
    assert "tests/bad.rs" in findings[0]
    assert "std::env::set_var" in findings[0]
    assert "_LOCK" in findings[0]
    assert "Mutex" in findings[0]
    assert "Issue #3453" in findings[0]


def test_check_test_file_rejects_lock_with_wrong_type(
    checker, monkeypatch, tmp_path
):
    """A `static ENV_LOCK: RwLock<()>` (no `Mutex` in type) must trip
    the gate -- the convention specifically requires `Mutex`."""
    findings = _findings_for(
        checker, monkeypatch, tmp_path,
        "wrong_type.rs", _WRONG_TYPE_FILE,
    )
    assert len(findings) == 1
    assert "tests/wrong_type.rs" in findings[0]


def test_check_test_file_rejects_lock_without_suffix(
    checker, monkeypatch, tmp_path
):
    """A bare `use std::sync::Mutex;` with no `*_LOCK` static does NOT
    satisfy the gate -- the `_LOCK` suffix is the load-bearing marker."""
    findings = _findings_for(
        checker, monkeypatch, tmp_path,
        "wrong_name.rs", _WRONG_NAME_FILE,
    )
    assert len(findings) == 1
    assert "tests/wrong_name.rs" in findings[0]


# ---------------------------------------------------------------------------
# iter_test_files -- directory walk + skip-list semantics
# ---------------------------------------------------------------------------


def test_iter_test_files_skips_reference_data(
    checker, monkeypatch, tmp_path
):
    """``tests/reference_data/`` contains golden fixtures, not Rust
    source; the walk must skip it even if a ``.rs`` lands there."""
    tests_dir = _redirect(checker, monkeypatch, tmp_path)
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "real.rs").write_text(_COMPLIANT_FILE, encoding="utf-8")
    (tests_dir / "reference_data").mkdir(parents=True)
    (tests_dir / "reference_data" / "fixture.rs").write_text(
        _UNGUARDED_FILE, encoding="utf-8"
    )
    names = {p.name for p in checker.iter_test_files(tests_dir)}
    assert "real.rs" in names
    assert "fixture.rs" not in names


def test_iter_test_files_skips_fixtures(
    checker, monkeypatch, tmp_path
):
    """``tests/fixtures/`` must also be skipped (same rationale)."""
    tests_dir = _redirect(checker, monkeypatch, tmp_path)
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "real.rs").write_text(_COMPLIANT_FILE, encoding="utf-8")
    (tests_dir / "fixtures").mkdir(parents=True)
    (tests_dir / "fixtures" / "noise.rs").write_text(
        _UNGUARDED_FILE, encoding="utf-8"
    )
    names = {p.name for p in checker.iter_test_files(tests_dir)}
    assert "real.rs" in names
    assert "noise.rs" not in names


def test_iter_test_files_walks_nested_dirs(
    checker, monkeypatch, tmp_path
):
    """Nested ``tests/<sub>/<name>.rs`` files must be discovered."""
    tests_dir = _redirect(checker, monkeypatch, tmp_path)
    tests_dir.mkdir(parents=True, exist_ok=True)
    nested = tests_dir / "subdir" / "nested.rs"
    nested.parent.mkdir(parents=True)
    nested.write_text(_COMPLIANT_FILE, encoding="utf-8")
    names = {p.name for p in checker.iter_test_files(tests_dir)}
    assert "nested.rs" in names


# ---------------------------------------------------------------------------
# main() exit-code contract: 0 clean / 1 drift / 2 script error
# ---------------------------------------------------------------------------


def test_main_exit_0_when_all_compliant(
    checker, monkeypatch, tmp_path, capsys
):
    _redirect(checker, monkeypatch, tmp_path)
    _write_test_file(tmp_path, "a.rs", _COMPLIANT_FILE)
    _write_test_file(tmp_path, "b.rs", _EXEMPT_FILE)
    monkeypatch.setattr(
        checker.sys, "argv", ["check_env_setvar_serialized.py"]
    )
    assert checker.main() == 0
    out = capsys.readouterr().out
    assert "OK" in out
    assert "2 test file(s)" in out
    assert "1 env-mutating file(s)" in out


def test_main_exit_1_on_planted_violation(
    checker, monkeypatch, tmp_path, capsys
):
    _redirect(checker, monkeypatch, tmp_path)
    _write_test_file(tmp_path, "ok.rs", _COMPLIANT_FILE)
    _write_test_file(tmp_path, "bad.rs", _UNGUARDED_FILE)
    monkeypatch.setattr(
        checker.sys, "argv", ["check_env_setvar_serialized.py"]
    )
    assert checker.main() == 1
    err = capsys.readouterr().err
    assert "ENV_LOCK drift detected" in err
    assert "tests/bad.rs" in err
    assert "Issue #3453" in err


def test_main_exit_2_when_tests_dir_missing(
    checker, monkeypatch, tmp_path, capsys
):
    _redirect(checker, monkeypatch, tmp_path)
    monkeypatch.setattr(
        checker.sys, "argv", ["check_env_setvar_serialized.py"]
    )
    assert checker.main() == 2
    assert "not found" in capsys.readouterr().err


def test_main_exit_2_on_unknown_test_filter(
    checker, monkeypatch, tmp_path, capsys
):
    _redirect(checker, monkeypatch, tmp_path)
    _write_test_file(tmp_path, "a.rs", _COMPLIANT_FILE)
    monkeypatch.setattr(
        checker.sys,
        "argv",
        ["check_env_setvar_serialized.py", "--test", "nope.rs"],
    )
    assert checker.main() == 2
    assert "nope.rs not found" in capsys.readouterr().err


def test_main_test_filter_restricts_scan_to_named_file(
    checker, monkeypatch, tmp_path, capsys
):
    """`--test` scopes the scan: a repo with one compliant and one
    violating file passes when only the compliant one is named."""
    _redirect(checker, monkeypatch, tmp_path)
    _write_test_file(tmp_path, "ok.rs", _COMPLIANT_FILE)
    _write_test_file(tmp_path, "bad.rs", _UNGUARDED_FILE)
    monkeypatch.setattr(
        checker.sys,
        "argv",
        ["check_env_setvar_serialized.py", "--test", "ok.rs"],
    )
    assert checker.main() == 0
