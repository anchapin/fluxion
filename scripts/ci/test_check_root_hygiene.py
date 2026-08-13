"""
Tests for ``scripts/check_root_hygiene.py`` -- Issue #2466 (root ``.md`` policy,
widened to scratch extensions / no-ext blobs / scratch directories).

PR #2814 exposed ``scan_root(repo_root) -> RootScan`` precisely so a pytest
harness could drive it; this file replaces the ad-hoc ``--self-test`` with
parametric clean-tree / planted-violation cases. ``scan_root`` takes the
repo root as an argument, so the scanner is exercised directly against a
``tmp_path`` mock root (no monkey-patching of the scanner itself needed).
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPT_NAME = "check_root_hygiene"


@pytest.fixture
def checker(load_script):
    return load_script(SCRIPT_NAME)


def _touch(root: Path, name: str, *, is_dir: bool = False) -> Path:
    p = root / name
    if is_dir:
        p.mkdir()
    else:
        p.write_text("x", encoding="utf-8")
    return p


ALLOWLIST_MD = [
    "README.md",
    "ARCHITECTURE.md",
    "CODEBASE_MAP.md",
    "CONTRIBUTING.md",
    "RULES.md",
    "CHANGELOG.md",
    "AGENTS.md",
    "SCORECARD.md",
]


# ---------------------------------------------------------------------------
# scan_root classification buckets
# ---------------------------------------------------------------------------


def test_scan_clean_root_has_no_violations(checker, tmp_path):
    for name in ALLOWLIST_MD:
        _touch(tmp_path, name)
    _touch(tmp_path, "src", is_dir=True)
    _touch(tmp_path, "Cargo.toml")  # never-blocked extension
    _touch(tmp_path, ".gitignore")  # dotfile skipped
    scan = checker.scan_root(tmp_path)
    assert scan.violations == []
    assert len(scan.md_allow) == len(ALLOWLIST_MD)


@pytest.mark.parametrize("name", ["CASE_600.md", "BATCH_REPORT.md", "scratch.md"])
def test_scan_flags_transient_md(checker, tmp_path, name):
    _touch(tmp_path, name)
    scan = checker.scan_root(tmp_path)
    assert [p.name for p in scan.md_transient] == [name]
    assert scan.violations == [tmp_path / name]


@pytest.mark.parametrize("ext", [".txt", ".csv", ".rs", ".py", ".sh", ".json", ".zip"])
def test_scan_flags_blocked_extension(checker, tmp_path, ext):
    _touch(tmp_path, f"dump{ext}")
    scan = checker.scan_root(tmp_path)
    assert [p.name for p in scan.blocked_ext] == [f"dump{ext}"]
    assert scan.violations == [tmp_path / f"dump{ext}"]


@pytest.mark.parametrize("name", ["requirements-dev.txt", "requirements.txt"])
def test_scan_allows_blocked_ext_exceptions(checker, tmp_path, name):
    _touch(tmp_path, name)
    assert checker.scan_root(tmp_path).violations == []


@pytest.mark.parametrize("ext", [".toml", ".yaml", ".yml", ".lock", ".pyi"])
def test_scan_allows_never_blocked_extensions(checker, tmp_path, ext):
    _touch(tmp_path, f"config{ext}")
    assert checker.scan_root(tmp_path).violations == []


def test_scan_flags_no_extension_blob(checker, tmp_path):
    _touch(tmp_path, "mystery-binary")
    scan = checker.scan_root(tmp_path)
    assert [p.name for p in scan.no_ext_blocked] == ["mystery-binary"]
    assert scan.violations == [tmp_path / "mystery-binary"]


@pytest.mark.parametrize("name", ["LICENSE", "Dockerfile", "Makefile"])
def test_scan_allows_no_ext_exceptions(checker, tmp_path, name):
    _touch(tmp_path, name)
    assert checker.scan_root(tmp_path).violations == []


@pytest.mark.parametrize(
    "name", ["fixes", "results", "reports", "scratch", "output", "artifacts"]
)
def test_scan_flags_blocked_directory(checker, tmp_path, name):
    _touch(tmp_path, name, is_dir=True)
    scan = checker.scan_root(tmp_path)
    assert [p.name for p in scan.blocked_dirs] == [name]
    assert scan.violations == [tmp_path / name]


@pytest.mark.parametrize("name", ["src", "docs", "tests", "scripts", "crates"])
def test_scan_allows_legit_directory(checker, tmp_path, name):
    _touch(tmp_path, name, is_dir=True)
    assert checker.scan_root(tmp_path).violations == []


def test_scan_classifies_multiple_findings_into_separate_buckets(checker, tmp_path):
    _touch(tmp_path, "CASE_X.md")
    _touch(tmp_path, "dump.csv")
    _touch(tmp_path, "blob")
    _touch(tmp_path, "fixes", is_dir=True)
    scan = checker.scan_root(tmp_path)
    assert [p.name for p in scan.md_transient] == ["CASE_X.md"]
    assert [p.name for p in scan.blocked_ext] == ["dump.csv"]
    assert [p.name for p in scan.no_ext_blocked] == ["blob"]
    assert [p.name for p in scan.blocked_dirs] == ["fixes"]
    assert len(scan.violations) == 4


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_has_extension_helper(checker):
    assert checker._has_extension("Cargo.toml") is True
    assert checker._has_extension("LICENSE") is False


def test_is_dotfile_helper(checker):
    assert checker._is_dotfile(".gitignore") is True
    assert checker._is_dotfile("README.md") is False


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_self_test_passes_deterministically(checker, capsys):
    """The bundled ``--self-test`` builds a mock repo and must exit 0."""
    assert checker.main(["--self-test"]) == 0


def test_main_returns_zero_on_clean_real_repo(checker, repo_root, capsys):
    """The real checkout is a required-check that stays clean; a scanner
    regression would surface a phantom violation here."""
    assert checker.main() == 0, capsys.readouterr().out
