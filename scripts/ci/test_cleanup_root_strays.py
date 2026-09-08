"""
Tests for ``scripts/cleanup_root_strays.sh`` — issue #3438 operator-action
helper for deleting known stale root transient artifacts (the two
``.csv`` files from the pre-#3303 CLI sensitivity command that trip the
``scripts/check_root_hygiene.py`` gate on the operator's machine).

The bash script is exercised via ``subprocess.run`` against a hermetic
``tmp_path`` repo; we mirror the same ``git init`` + ``.gitignore``
planting pattern used in ``test_check_root_hygiene.py`` so the safety
predicates (``git check-ignore`` and ``git ls-files --error-unmatch``)
exercise the real Git code path.

The script's safety invariants are encoded in ``KNOWN_STRAY_PATTERNS``
(narrow, hand-curated names) plus the two git predicates, so the tests
focus on the happy path, the no-op cases, and the exit-code contract.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "cleanup_root_strays.sh"


pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("git") is None,
    reason="bash + git required to exercise the cleanup helper",
)


@pytest.fixture
def worktree_repo(tmp_path):
    """Initialize a hermetic git repo with the two known stray patterns
    planted in ``.gitignore`` (mirrors ``.gitignore`` lines 170 and 220
    from the production repo). Returns the tmp_path root."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text(
        "hourly_output*.csv\n*_profile_hourly.csv\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", ".gitignore"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True
    )
    return tmp_path


def _run_script(repo_root: Path, *extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT_PATH), *extra_args],
        capture_output=True,
        text=True,
        cwd=repo_root,
        timeout=30,
    )


def test_script_exists_and_is_executable():
    """The bash script must exist and be executable so the operator can
    invoke it directly from a worktree root."""
    assert SCRIPT_PATH.exists(), f"missing helper at {SCRIPT_PATH}"
    import stat

    mode = SCRIPT_PATH.stat().st_mode
    assert mode & stat.S_IXUSR, "cleanup_root_strays.sh must be user-executable"


def test_script_help_flag_prints_usage():
    """``--help`` prints the header docstring and exits 0 without
    touching any state."""
    proc = _run_script(REPO_ROOT, "--help")
    assert proc.returncode == 0, proc.stderr
    assert "scripts/cleanup_root_strays.sh" in proc.stdout
    assert "Usage" in proc.stdout
    assert "--apply" in proc.stdout


def test_script_dry_run_is_noop(worktree_repo):
    """Dry-run mode never deletes anything. Plant the two known strays
    and confirm both survive the dry-run."""
    (worktree_repo / "hourly_output.csv").write_text("x", encoding="utf-8")
    (worktree_repo / "case_900ff_profile_hourly.csv").write_text(
        "x" * 100, encoding="utf-8"
    )
    proc = _run_script(worktree_repo)  # default = dry-run
    assert proc.returncode == 0, proc.stderr
    assert (worktree_repo / "hourly_output.csv").exists()
    assert (worktree_repo / "case_900ff_profile_hourly.csv").exists()
    assert "hourly_output.csv" in proc.stdout
    assert "case_900ff_profile_hourly.csv" in proc.stdout
    assert "DRY-RUN" in proc.stdout


def test_script_apply_deletes_eligible_strays(worktree_repo):
    """``--apply --yes`` deletes both known strays. Exit 0."""
    (worktree_repo / "hourly_output.csv").write_text("x", encoding="utf-8")
    (worktree_repo / "case_900ff_profile_hourly.csv").write_text(
        "x" * 100, encoding="utf-8"
    )
    proc = _run_script(worktree_repo, "--apply", "--yes")
    assert proc.returncode == 0, proc.stderr
    assert not (worktree_repo / "hourly_output.csv").exists()
    assert not (worktree_repo / "case_900ff_profile_hourly.csv").exists()
    assert "deleted: hourly_output.csv" in proc.stdout
    assert "deleted: case_900ff_profile_hourly.csv" in proc.stdout


def test_script_refuses_to_delete_tracked_files(worktree_repo):
    """A gitignored file that is ALSO committed to git must be skipped:
    the script only deletes untracked files. The operator should run
    ``git rm --cached`` instead (issue #3076). ``-f`` is required to
    track a gitignored file in the first place."""
    tracked = worktree_repo / "hourly_output.csv"
    tracked.write_text("committed", encoding="utf-8")
    subprocess.run(
        ["git", "add", "-f", "hourly_output.csv"], cwd=worktree_repo, check=True
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "track"], cwd=worktree_repo, check=True
    )
    proc = _run_script(worktree_repo, "--apply", "--yes")
    assert proc.returncode == 0, proc.stderr
    assert tracked.exists(), "tracked file must NOT be deleted"
    assert "tracked" in proc.stdout
    assert "Skipped" in proc.stdout


def test_script_exits_zero_when_nothing_to_do(worktree_repo):
    """Empty repo + no known strays = PASS, exit 0."""
    proc = _run_script(worktree_repo)
    assert proc.returncode == 0, proc.stderr
    assert "PASS" in proc.stdout


def test_script_list_flag_prints_only_paths(worktree_repo):
    """``--list`` prints one eligible path per line and exits 0."""
    (worktree_repo / "hourly_output.csv").write_text("x", encoding="utf-8")
    (worktree_repo / "case_900ff_profile_hourly.csv").write_text("x", encoding="utf-8")
    proc = _run_script(worktree_repo, "--list")
    assert proc.returncode == 0, proc.stderr
    lines = [
        line.strip() for line in proc.stdout.splitlines() if line.strip()
    ]
    assert "hourly_output.csv" in lines
    assert "case_900ff_profile_hourly.csv" in lines


def test_script_unknown_flag_errors_out(worktree_repo):
    """An unknown CLI flag exits 2 with a usage hint."""
    proc = _run_script(worktree_repo, "--bogus-flag")
    assert proc.returncode == 2, proc.stderr
    assert "unknown flag" in proc.stderr or "unknown flag" in proc.stdout


def test_script_outside_git_errors_out(tmp_path):
    """Outside a git repo, the script exits 2 and never touches the
    filesystem (the safety predicates depend on ``git rev-parse``)."""
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    proc = _run_script(not_a_repo)
    assert proc.returncode == 2, proc.stderr
    assert "not inside a Git working tree" in proc.stderr