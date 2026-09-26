"""Scan-determinism regression tests — Issue #4069.

The AST inventory scan (``scripts/generate_test_inventory.py``) must be
a pure function of the *tracked* tree: untracked scratch ``.rs`` files
(e.g. agent ``.planning/worktrees/`` copies inside the repo) must not
shift the enumeration, and a clean checkout of HEAD must reproduce the
committed ``tests/test_inventory.json`` snapshot.

Layout mirrors the ``load_script`` + ``tmp_path`` hermetic pattern from
``test_check_test_inventory_drift.py``: the generator computes
``REPO_ROOT`` at import time, so each test monkeypatches it at a
synthetic git repo before invoking the scan.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git is required for these tests"
)


@pytest.fixture
def generator(load_script):
    """Freshly-loaded copy of the inventory generator."""
    return load_script("generate_test_inventory")


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"git {' '.join(args)} failed: {proc.stderr}"
    return proc.stdout


def _make_repo(root: Path) -> Path:
    """Minimal git-tracked workspace: root crate + one member crate."""
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "inventory-test")
    (root / "Cargo.toml").write_text(
        '[workspace]\nmembers = ["member-a"]\n', encoding="utf-8"
    )
    (root / "src").mkdir()
    (root / "src" / "lib.rs").write_text(
        "#[test]\nfn tracked_root_one() {}\n#[test]\nfn tracked_root_two() {}\n",
        encoding="utf-8",
    )
    (root / "tests").mkdir()
    (root / "tests" / "keeper.rs").write_text(
        "#[test]\nfn integration_keeper() {}\n", encoding="utf-8"
    )
    member = root / "member-a"
    (member / "src").mkdir(parents=True)
    (member / "Cargo.toml").write_text(
        '[package]\nname = "member-a"\n', encoding="utf-8"
    )
    (member / "src" / "lib.rs").write_text(
        "#[test]\nfn member_test() {}\n", encoding="utf-8"
    )
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "initial tracked tree")
    return root


def _scan(generator, repo: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(generator, "REPO_ROOT", repo)
    return generator.scan_source_inventory([])


def test_untracked_rs_files_do_not_shift_counts(
    generator, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Issue #4069: untracked ``.rs`` files must not enter the scan.

    Fails against the pre-#4069 on-disk ``rglob`` walk (which counted
    the phantom files); passes with the ``git ls-files`` enumeration.
    """
    repo = _make_repo(tmp_path / "repo")
    base = _scan(generator, repo, monkeypatch)
    assert base["totals"]["lib_tests_root"] == 2
    assert base["totals"]["workspace_tests"] == 4  # 2 root lib + 1 member lib + 1 integration

    # The #4069 failure mode: untracked scratch files carrying tests.
    (repo / "src" / "scratch.rs").write_text(
        "#[test]\nfn phantom_one() {}\n#[test]\nfn phantom_two() {}\n"
        "#[test]\nfn phantom_three() {}\n#[test]\nfn phantom_four() {}\n",
        encoding="utf-8",
    )
    (repo / "member-a" / "src" / "wip.rs").write_text(
        "#[test]\nfn phantom_member() {}\n", encoding="utf-8"
    )
    (repo / ".planning").mkdir()
    (repo / ".planning" / "notes.rs").write_text(
        "#[test]\nfn phantom_notes() {}\n", encoding="utf-8"
    )

    after = _scan(generator, repo, monkeypatch)
    assert after["totals"] == base["totals"]
    assert after["by_crate"] == base["by_crate"]


def test_enumeration_matches_git_ls_files(
    generator, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The scanned file set is exactly the tracked ``src/``/``tests/`` set."""
    repo = _make_repo(tmp_path / "repo")
    # Untracked noise that must not appear in the enumeration.
    (repo / "src" / "scratch.rs").write_text("#[test]\nfn s() {}\n", encoding="utf-8")
    monkeypatch.setattr(generator, "REPO_ROOT", repo)

    members = [repo] + generator._walk_workspace_members()
    scanned = {
        p.relative_to(repo).as_posix()
        for _, _, _, p in generator._scan_file_list(members, [])
    }

    ls_out = _git(repo, "ls-files", "-z", "--", "*.rs")
    expected = set()
    for posix_path in ls_out.split("\0"):
        if not posix_path:
            continue
        parts = posix_path.split("/")
        # The generator assigns each tracked path to a member by
        # longest-prefix match, then requires the next segment to be
        # src/ or tests/. Reproduce that rule independently here.
        if parts[0] in ("src", "tests"):
            in_scope = len(parts) > 1
        else:
            in_scope = len(parts) > 2 and parts[1] in ("src", "tests")
        if in_scope and (repo / posix_path).is_file():
            expected.add(posix_path)

    assert scanned == expected


def test_scan_is_stable_across_runs(
    generator, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Two consecutive scans of the same tree enumerate identically."""
    repo = _make_repo(tmp_path / "repo")
    monkeypatch.setattr(generator, "REPO_ROOT", repo)
    members = [repo] + generator._walk_workspace_members()
    first = generator._scan_file_list(members, [])
    second = generator._scan_file_list(members, [])
    assert [str(p) for _, _, _, p in first] == [str(p) for _, _, _, p in second]
    # Deterministic (sorted) order, not filesystem order.
    assert [str(p) for _, _, _, p in first] == sorted(str(p) for _, _, _, p in first)


def test_falls_back_to_ondisk_walk_without_git(
    generator, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Without a git checkout the scanner degrades to the on-disk walk."""
    repo = tmp_path / "nogit"
    (repo / "src").mkdir(parents=True)
    (repo / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
    (repo / "src" / "lib.rs").write_text(
        "#[test]\nfn a() {}\n#[test]\nfn b() {}\n", encoding="utf-8"
    )
    inventory = _scan(generator, repo, monkeypatch)
    assert inventory["totals"]["lib_tests_root"] == 2


def test_clean_checkout_scan_matches_committed_snapshot(
    generator, monkeypatch: pytest.MonkeyPatch
):
    """A clean checkout of HEAD reproduces the committed snapshot.

    Compares the fresh AST scan against ``HEAD:tests/test_inventory.json``
    (read from git, so the drift gate's rewrite side-effect cannot make
    this vacuous). Only ``totals``/``by_crate`` are compared —
    ``generated_at``/``repo_root`` are intentionally volatile.
    """
    proc = subprocess.run(
        ["git", "show", "HEAD:tests/test_inventory.json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, "could not read HEAD:tests/test_inventory.json"
    committed = json.loads(proc.stdout)
    if (committed.get("verify") or {}).get("matched"):
        pytest.skip("committed snapshot is cargo-verified; AST self-check N/A")

    monkeypatch.setattr(generator, "REPO_ROOT", REPO_ROOT)
    live = generator.scan_source_inventory(["fluxion-tauri"])
    assert live["totals"] == committed["totals"], (
        "fresh AST scan diverged from the committed tests/test_inventory.json "
        f"(live={live['totals']}, committed={committed['totals']})"
    )
    assert live["by_crate"] == committed["by_crate"]
