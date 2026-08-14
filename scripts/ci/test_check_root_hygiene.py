"""
Tests for ``scripts/check_root_hygiene.py`` -- Issue #2466 (root ``.md`` policy,
widened to scratch extensions / no-ext blobs / scratch directories) and
issue #2954 (root dotfile/dotdir cross-check against ``.gitignore`` /
git tracking).

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


@pytest.mark.parametrize(
    "ext",
    [
        ".onnx",  # ONNX (PyTorch export / sklearn-onnx)
        ".bin",  # HuggingFace / generic binary checkpoint
        ".pt",  # PyTorch state_dict
        ".pkl",  # sklearn / pickle
        ".h5",  # Keras / TensorFlow HDF5
    ],
)
def test_scan_flags_blocked_ml_model_extension(checker, tmp_path, ext):
    """Issue #2949: binary ML model extensions at the repo root are transient
    artifacts. Legit model files live inside ``models/``, ``assets/``,
    ``examples/`` directories and are never seen by the root-only scan."""
    _touch(tmp_path, f"tests_tmp_dummy{ext}")
    scan = checker.scan_root(tmp_path)
    assert [p.name for p in scan.blocked_ext] == [f"tests_tmp_dummy{ext}"]
    assert scan.violations == [tmp_path / f"tests_tmp_dummy{ext}"]


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


# ---------------------------------------------------------------------------
# Dotfile / dotdir cross-check (issue #2954)
# ---------------------------------------------------------------------------


# Legit root dotfile config — allow-listed, no gitignore needed.
ROOT_DOTCONFIG_ALLOWLIST = [
    ".agents",
    ".cargo",
    ".cargoignore",
    ".dockerignore",
    ".editorconfig",
    ".env.example",
    ".git",
    ".github",
    ".gitattributes",
    ".githooks",
    ".gitignore",
    ".npmignore",
    ".planning",
    ".pre-commit-config.yaml",
    ".rustfmt.toml",
]


@pytest.mark.parametrize("name", ROOT_DOTCONFIG_ALLOWLIST)
def test_scan_allows_root_dotconfig(checker, tmp_path, name):
    """Issue #2954: a dotfile/dotdir in ROOT_DOTFILE_ALLOWLIST is permitted
    without any gitignore/tracked cross-check (legit root config)."""
    p = tmp_path / name
    if name in {".agents", ".cargo", ".githooks", ".github", ".git", ".planning"}:
        p.mkdir()
    else:
        p.write_text("x", encoding="utf-8")
    assert checker.scan_root(tmp_path).violations == []
    assert checker.scan_root(tmp_path).dotfile_unmanaged == []


def test_scan_flags_unmanaged_dotdir(checker, tmp_path, monkeypatch):
    """A dotdir at root that is neither allow-listed, nor gitignored, nor
    tracked must be reported as ``dotfile_unmanaged`` (issue #2954)."""
    _touch(tmp_path, ".mytool", is_dir=True)
    # tmp_path has no git, so neutralize every signal.
    monkeypatch.setattr(checker, "is_gitignored", lambda p: False)
    monkeypatch.setattr(checker, "is_tracked", lambda p: False)
    scan = checker.scan_root(tmp_path)
    assert [p.name for p in scan.dotfile_unmanaged] == [".mytool"]
    assert scan.violations == [tmp_path / ".mytool"]


def test_scan_passes_dotdir_when_gitignored(checker, tmp_path, monkeypatch):
    """A dotdir matched by `.gitignore` is compliant."""
    _touch(tmp_path, ".mytool", is_dir=True)
    monkeypatch.setattr(checker, "is_gitignored", lambda p: p.name == ".mytool")
    monkeypatch.setattr(checker, "is_tracked", lambda p: False)
    scan = checker.scan_root(tmp_path)
    assert scan.dotfile_unmanaged == []
    assert scan.violations == []


def test_scan_passes_dotdir_when_tracked(checker, tmp_path, monkeypatch):
    """A dotdir already tracked in git (legacy) is compliant even when it
    has no `.gitignore` line of its own — the tracked fallback accepts it."""
    _touch(tmp_path, ".mytool", is_dir=True)
    monkeypatch.setattr(checker, "is_gitignored", lambda p: False)
    monkeypatch.setattr(checker, "is_tracked", lambda p: p.name == ".mytool")
    scan = checker.scan_root(tmp_path)
    assert scan.dotfile_unmanaged == []
    assert scan.violations == []


def test_main_fails_on_unmanaged_dotdir_then_passes_after_gitignore(
    checker, tmp_path, monkeypatch, capsys
):
    """Issue #2954 end-to-end: a fresh-clone dotdir at root fails the gate;
    adding it to ``.gitignore`` flips it back to PASS. Uses a real ``git
    init`` so the existing ``is_gitignored()`` (which shells out to
    ``git check-ignore``) sees the gitignore line."""
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)

    # `.gitignore` does NOT mention `.mytool`, and `.mytool/` exists.
    (tmp_path / ".gitignore").write_text("*.tmp\n", encoding="utf-8")
    _touch(tmp_path, ".mytool", is_dir=True)
    (tmp_path / ".mytool" / "x.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)

    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)

    # Phase 1: gate FAILS — `.mytool` is not allow-listed / gitignored / tracked.
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1, f"expected FAIL (rc=1), got rc={rc}\noutput:\n{out}"
    assert ".mytool" in out, f"expected '.mytool' in output, got:\n{out}"

    # Phase 2: add `.mytool/` to `.gitignore` → gate PASSES.
    (tmp_path / ".gitignore").write_text("*.tmp\n.mytool/\n", encoding="utf-8")
    assert checker.main() == 0


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


def test_main_fails_on_onnx_escapee_then_passes_after_gitignore(
    checker, tmp_path, monkeypatch, capsys
):
    """Issue #2949 regression test, end-to-end.

    A 122-byte ``tests_tmp_dummy.onnx`` escaped a temp dir and landed at the
    repo root. The hygiene gate must FAIL on a freshly-planted ``.onnx`` file
    at the root (binary ML model extensions are blocked per #2949) and must
    pass once the file is gone. We run ``git init`` so ``is_gitignored``
    (which shells out to ``git check-ignore``) exercises the real code
    path — the planted ``.gitignore`` deliberately does NOT mention
    ``*.onnx`` so the file is treated as un-ignored and un-tracked.
    """
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)

    # `.gitignore` does NOT mention `.onnx` and the file is not tracked.
    (tmp_path / ".gitignore").write_text("*.tmp\n", encoding="utf-8")
    (tmp_path / "tests_tmp_dummy.onnx").write_bytes(b"\x00" * 122)
    subprocess.run(["git", "add", ".gitignore"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)

    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)

    # Phase 1: gate FAILS — `tests_tmp_dummy.onnx` is a blocked extension at root.
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1, f"expected FAIL (rc=1), got rc={rc}\noutput:\n{out}"
    assert "tests_tmp_dummy.onnx" in out, (
        f"expected 'tests_tmp_dummy.onnx' in output, got:\n{out}"
    )
    assert ".onnx" in out, f"expected '.onnx' in output, got:\n{out}"

    # Phase 2: delete the escapee → gate PASSES.
    (tmp_path / "tests_tmp_dummy.onnx").unlink()
    assert checker.main() == 0


# ---------------------------------------------------------------------------
# AGENTS.md runtime dirs (issue #2984)
# ---------------------------------------------------------------------------

# The six runtime dirs declared in AGENTS.md §Repository Hygiene as
# "gitignored — never commit, never create at repo root".  The untrack
# step in #2984 removed them from git's index, but the gate would
# silently regress if any of them got re-added later.  These tests
# pin ``git ls-files <dir>/`` to zero so a fresh re-commit flips the
# test red.
AGENTS_MD_RUNTIME_DIRS = [
    ".automaker",
    ".serena",
    ".sisyphus",
    ".jules",
    ".superset",
    ".gitnexus",
]


@pytest.mark.parametrize("dir_name", AGENTS_MD_RUNTIME_DIRS)
def test_agents_md_runtime_dir_has_no_tracked_files(repo_root, dir_name):
    """Issue #2984 regression: every dir declared in AGENTS.md §Repository
    Hygiene as a local-only runtime dir must have **zero** entries in
    ``git ls-files``. A tracked file inside one of these dirs means the
    agent-runtime state is being shared across contributors — exactly the
    bug the gitignore + untrack pattern prevents.

    The dirs themselves may legitimately exist on the developer's machine
    (their own runtime state); the test only checks the index, not the
    working tree.
    """
    import subprocess

    result = subprocess.run(
        ["git", "ls-files", "--", f"{dir_name}/"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    tracked = [line for line in result.stdout.splitlines() if line.strip()]
    assert tracked == [], (
        f"runtime dir {dir_name}/ has {len(tracked)} tracked file(s) at HEAD; "
        f"AGENTS.md §Repository Hygiene forbids tracking them. "
        f"Run `git rm -r --cached {dir_name}/` and verify the matching "
        f"`.gitignore` line exists. Tracked entries: {tracked}"
    )


def test_agents_md_runtime_dirs_have_gitignore_entries(repo_root):
    """Issue #2984 — belt-and-braces: each of the six runtime dirs declared
    in AGENTS.md §Repository Hygiene must have a matching `.gitignore`
    entry. The gitignore line is the real defense; the untrack step just
    removes the legacy tracked copies. A fresh `git clone` would re-allow
    commits of agent-runtime state if the gitignore line went missing.
    """
    gitignore_path = repo_root / ".gitignore"
    gitignore_text = gitignore_path.read_text(encoding="utf-8")
    for dir_name in AGENTS_MD_RUNTIME_DIRS:
        assert dir_name in gitignore_text, (
            f".gitignore is missing a line for {dir_name}/. The directory is "
            f"declared as a local-only runtime dir in AGENTS.md §Repository "
            f"Hygiene; without a gitignore entry, fresh checkouts would not "
            f"prevent future commits."
        )
