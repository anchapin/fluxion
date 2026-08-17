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
# Note: `.agents/` is intentionally absent — it is a local agent runtime
# dir that is matched by `.gitignore` on every fresh checkout (issue
# #2981) and is therefore covered by the dotfile cross-check's
# ``is_gitignored`` branch rather than this allow-list. See
# ``test_agents_md_runtime_dirs_have_gitignore_entries`` for the regression
# guard that pins the gitignore line in place.
ROOT_DOTCONFIG_ALLOWLIST = [
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
    if name in {".cargo", ".githooks", ".github", ".git", ".planning"}:
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


# ---------------------------------------------------------------------------
# `.agents/` runtime dir (issue #2981)
# ---------------------------------------------------------------------------


def test_agents_dir_has_gitignore_entry(repo_root):
    """Issue #2981 regression: `.agents/` is a local agent-runtime dir
    (issues/, results/, skills/, coverage/) that the
    github-wave-orchestrator and parallel-issue-generator skills write to.
    During the 2026-08-14 wave run, every sub-agent wrote a
    `result-backend-*.md` to `.agents/results/`; the files were untracked
    but present on disk, so `git pull --ff-only` failed with
    "Please commit or stash them" on every wave boundary.

    The `.gitignore` line is the real defense — once `.agents/` is
    gitignored, Git treats new sub-agent result files as ignored (not
    dirty) and `git pull --ff-only` proceeds cleanly. Without this line,
    a fresh `git clone` would re-allow untracked agent telemetry to
    block fast-forward pulls.
    """
    gitignore_text = (repo_root / ".gitignore").read_text(encoding="utf-8")
    assert ".agents/" in gitignore_text, (
        "`.gitignore` is missing a line for `.agents/`. The directory is "
        "a local agent runtime dir (issues/, results/, skills/, "
        "coverage/); without a gitignore entry, fresh checkouts would "
        "not prevent untracked sub-agent result files from blocking "
        "`git pull --ff-only` (issue #2981)."
    )


def test_agents_dir_not_in_root_dotfile_allowlist(checker):
    """Issue #2981 regression: `.agents/` MUST NOT be in the gate's
    ``ROOT_DOTFILE_ALLOWLIST``. The allow-list is reserved for legit
    root config that Git itself needs to see (``.git/``, ``.cargo/``,
    ``.gitignore``, ...); agent-runtime state is per-developer and
    belongs under the gitignore + dotfile-cross-check path (mirroring
    ``.sdd/``, ``.automaker/``, ``.serena/``, ``.sisyphus/``, ``.jules/``,
    ``.superset/``, ``.gitnexus/``, ``.issues/``, ``.opencode/``,
    ``.claude/`` per #2954). A regression that re-adds `.agents` here
    would silently bypass the gate on fresh checkouts.
    """
    assert ".agents" not in checker.ROOT_DOTFILE_ALLOWLIST, (
        "`.agents` must not be in ROOT_DOTFILE_ALLOWLIST (issue #2981). "
        "The gate accepts the directory via `is_gitignored()` because "
        "`.gitignore` matches `.agents/`; an allow-list entry would "
        "bypass that check and let the directory sneak back into the "
        "tree on fresh checkouts."
    )


def test_main_passes_after_agents_gitignore_added(checker, tmp_path, monkeypatch, capsys):
    """Issue #2981 end-to-end: a fresh-clone dotdir at root that is NOT in
    ``ROOT_DOTFILE_ALLOWLIST`` fails the gate; adding ``.agents/`` to
    ``.gitignore`` flips it back to PASS. Mirrors the ``.mytool``
    regression in #2954 — except the dotdir under test is the real
    `.agents/` from the production ``.gitignore``.

    Uses a real ``git init`` so the existing ``is_gitignored()`` (which
    shells out to ``git check-ignore``) sees the gitignore line.
    """
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)

    # `.gitignore` does NOT mention `.agents/`, and `.agents/` exists.
    (tmp_path / ".gitignore").write_text("*.tmp\n", encoding="utf-8")
    _touch(tmp_path, ".agents", is_dir=True)
    (tmp_path / ".agents" / "result-backend-1281.md").write_text(
        "x", encoding="utf-8"
    )
    subprocess.run(["git", "add", ".gitignore"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)

    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)

    # Phase 1: gate FAILS — `.agents/` is not allow-listed / gitignored /
    # tracked. (`.agents/` happens to also be the dotdir under test in
    # the real repo, but the test only exercises the gate's logic.)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1, f"expected FAIL (rc=1), got rc={rc}\noutput:\n{out}"
    assert ".agents" in out, f"expected '.agents' in output, got:\n{out}"

    # Phase 2: add `.agents/` to `.gitignore` → gate PASSES.
    (tmp_path / ".gitignore").write_text("*.tmp\n.agents/\n", encoding="utf-8")
    assert checker.main() == 0


# ---------------------------------------------------------------------------
# Tracked files inside gitignored runtime dirs (issue #3076)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dir_name",
    [
        ".agents",
        ".automaker",
        ".claude",
        ".gitnexus",
        ".jules",
        ".opencode",
        ".serena",
        ".sisyphus",
        ".superset",
    ],
)
def test_find_tracked_in_gitignored_dirs_flags_tracked_files(
    checker, tmp_path, dir_name
):
    """Issue #3076: ``find_tracked_in_gitignored_dirs`` must return the
    tracked entries for any runtime dir whose files were committed before
    its `.gitignore` rule was added.

    The dotfile/dotdir cross-check (#2954) only inspects the directory
    itself, so a runtime dir with N tracked FILES still passes that
    check as long as the dir entry was untracked — leaving the contents
    silently tracked. This probe catches the blind spot at the file
    granularity.
    """
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)

    # Plant the runtime dir + tracked file BEFORE adding the gitignore rule
    # (mirrors the historical sequence that produced the #3076 regression:
    # file committed first, gitignore rule added later, untrack step
    # forgotten).
    target = tmp_path / dir_name
    target.mkdir()
    (target / "stale-result.md").write_text("x", encoding="utf-8")
    # First commit WITHOUT the gitignore rule so the file is tracked.
    subprocess.run(["git", "add", f"{dir_name}/stale-result.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    # Now add the gitignore rule (the real-world sequence: rule added
    # in #2981/#2984 but ``git rm --cached`` forgotten).
    (tmp_path / ".gitignore").write_text(
        f"*.tmp\n{dir_name}/\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", ".gitignore"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "ignore"], cwd=tmp_path, check=True)

    tracked = checker.find_tracked_in_gitignored_dirs(tmp_path)
    assert dir_name in tracked, (
        f"expected tracked entries for {dir_name}/, got {sorted(tracked)}"
    )
    assert f"{dir_name}/stale-result.md" in tracked[dir_name]


def test_find_tracked_in_gitignored_dirs_empty_when_clean(checker, tmp_path):
    """Issue #3076: with no tracked entries, the probe returns ``{}`` and
    the gate does NOT fail on this bucket. Pin against a freshly-initialised
    empty repo so a regression that always-returns-non-empty would flip
    the test red."""
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)

    (tmp_path / ".gitignore").write_text("*.tmp\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)

    assert checker.find_tracked_in_gitignored_dirs(tmp_path) == {}


def test_find_tracked_in_gitignored_dirs_outside_git_returns_empty(checker, tmp_path):
    """Issue #3076: the probe is a no-op outside a Git working tree
    (so the gate stays green in CI sandboxes that copy the repo without
    ``.git/``)."""
    # tmp_path has no .git/.gitignore at all — git ls-files will fail.
    assert checker.find_tracked_in_gitignored_dirs(tmp_path) == {}


def test_main_fails_on_tracked_but_ignored_file_then_passes_after_untrack(
    checker, tmp_path, monkeypatch, capsys
):
    """Issue #3076 end-to-end regression test.

    The exact failure mode that produced issue #3076: 104 files under
    ``.agents/`` were committed before the ``.agents/`` gitignore rule
    was added in #2981, then ``git rm --cached`` was forgotten in the
    #2984 untrack step. The gate must FAIL on that state, then PASS once
    every tracked entry is untracked.

    Mirrors the existing ``test_main_fails_on_onnx_escapee_*`` /
    ``test_main_fails_on_unmanaged_dotdir_*`` patterns: real ``git init``
    so the probe (``find_tracked_in_gitignored_dirs`` → ``git ls-files``)
    exercises the real code path.
    """
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)

    # Plant the tracked file FIRST (without the gitignore rule), then
    # add the gitignore rule on a second commit. This mirrors the
    # historical sequence that produced the #3076 regression.
    (tmp_path / ".agents").mkdir()
    (tmp_path / ".agents" / "stale-result.md").write_text("x", encoding="utf-8")
    subprocess.run(
        ["git", "add", ".agents/stale-result.md"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("*.tmp\n.agents/\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "ignore"], cwd=tmp_path, check=True)

    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)

    # Phase 1: gate FAILS — `.agents/stale-result.md` is tracked in a
    # gitignored runtime dir.
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1, f"expected FAIL (rc=1), got rc={rc}\noutput:\n{out}"
    assert ".agents/stale-result.md" in out, (
        f"expected '.agents/stale-result.md' in output, got:\n{out}"
    )
    assert "tracked file in gitignored runtime dir" in out, (
        f"expected tracked-but-ignored message in output, got:\n{out}"
    )

    # Phase 2: ``git rm --cached`` the legacy tracked file → gate PASSES.
    # On-disk copy survives because the `.agents/` gitignore rule is in
    # place (added in #2981, still pinned by
    # ``test_agents_dir_has_gitignore_entry``).
    subprocess.run(
        ["git", "rm", "--cached", "--quiet", ".agents/stale-result.md"],
        cwd=tmp_path,
        check=True,
    )
    assert checker.main() == 0, (
        f"expected PASS after untrack, got FAIL.\noutput:\n{capsys.readouterr().out}"
    )


def test_scan_root_records_tracked_but_ignored_bucket(checker, tmp_path, monkeypatch):
    """Issue #3076: ``scan_root`` populates ``tracked_in_gitignored_dirs``
    so the structured-result API (used by the pytest harness) reflects the
    new bucket independently of ``violations``.

    The bucket uses ``{dir_name: [path, ...]}`` (not ``list[Path]``)
    because a single tracked dir typically contains many files, and the
    structured key preserves the per-dir grouping for the FAIL summary.
    """
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)

    (tmp_path / ".agents").mkdir()
    (tmp_path / ".agents" / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / ".agents" / "b.md").write_text("x", encoding="utf-8")
    # Commit the files first, then add the gitignore rule.
    subprocess.run(
        ["git", "add", ".agents/a.md", ".agents/b.md"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("*.tmp\n.agents/\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "ignore"], cwd=tmp_path, check=True)

    scan = checker.scan_root(tmp_path)
    assert ".agents" in scan.tracked_in_gitignored_dirs
    assert sorted(scan.tracked_in_gitignored_dirs[".agents"]) == [
        ".agents/a.md",
        ".agents/b.md",
    ]
    assert scan.tracked_violation_count == 2


def test_tracked_but_ignored_runtime_dirs_tuple_matches_gitignore(checker, repo_root):
    """Issue #3076: ``TRACKED_BUT_IGNORED_RUNTIME_DIRS`` is the source of
    truth the probe iterates over. Every entry MUST also appear in
    ``.gitignore`` — otherwise the probe would flag legitimate root
    config (e.g. ``.cargo/``) that is intentionally tracked.

    Mirrors the existing ``test_agents_md_runtime_dirs_have_gitignore_entries``
    belt-and-braces check (#2984), extended to cover all nine dirs
    declared in #3076.
    """
    gitignore_text = (repo_root / ".gitignore").read_text(encoding="utf-8")
    for dir_name in checker.TRACKED_BUT_IGNORED_RUNTIME_DIRS:
        assert f"{dir_name}/" in gitignore_text, (
            f".gitignore is missing a line for {dir_name}/. The probe "
            f"in `find_tracked_in_gitignored_dirs` only reports dirs "
            f"declared in TRACKED_BUT_IGNORED_RUNTIME_DIRS; a missing "
            f"gitignore line would let the dir sneak into the index on "
            f"fresh checkouts without being detected (issue #3076)."
        )
