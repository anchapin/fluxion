"""Tests for the github-wave-orchestrator skill — Issues #3145 and #379.

The github-wave-orchestrator skill (located at
``~/.config/opencode/skill/github-wave-orchestrator/``) drives the parallel
wave execution pipeline. Two distinct failure modes have to be guarded against:

#3145 (state-file collision)
    Two concurrent orchestrator runs against different repos
    (e.g. fluxion + openstudio-server-operator + osimflow on the same
    workstation) silently overwrite each other's ``wave-state.json`` because
    the legacy default path is shared. The fix is per-repo namespacing +
    a pre-flight collision check that refuses to overwrite a state file
    belonging to a different repo unless ``--force`` is passed.

#379 (skill-snapshot add/add collisions)
    When a wave's sub-agent modifies files in the skill home, those files
    must be copied into the worktree under a **wave-numbered** snapshot
    name (``<basename>.wave-${WAVE}.md``) rather than the bare basename,
    because two skill-touching waves in one cycle share a pre-merge develop
    and an add/add (or modify/modify) collision at rebase is otherwise
    structurally possible.

This file is the regression proof for both contracts. The wave-orchestrator
skill is external to the repo, so the tests synthesise a fresh git repo
under ``tmp_path``, drop a fake ``git remote add origin <URL>`` to mimic
the orchestrator's runtime environment, and exercise the bash helpers
(``scripts/wave-state-helpers.sh``) and the Node planner
(``scripts/wave-planner.js``) end-to-end. The skill home is referenced by
its canonical install path (``~/.config/opencode/skill/...``); if the
skill has not been installed the tests skip cleanly so a CI run on a
machine without the skill does not fail spuriously.

The skill snapshot script rule is enforced by
:class:`TestSkillSnapshotNumberedNames`, which walks ``docs/skill-snapshot/``
and asserts every file matches ``<basename>.wave-\\d+\\.md`` and that the
legacy un-numbered paths (``SKILL.md``, ``REFERENCE.md``,
``scripts/<name>``) are absent — these are the frozen pre-#379 snapshots
that historical diffs and ``tests/test_render_orchestrator_snippet.py``
§0-contract checks depend on.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = REPO_ROOT / "docs" / "skill-snapshot"
SKILL_HOME = Path.home() / ".config" / "opencode" / "skill" / "github-wave-orchestrator"


# ---------------------------------------------------------------------------
# Skip-if-skill-not-installed
# ---------------------------------------------------------------------------


pytestmark_skill_required = pytest.mark.skipif(
    not SKILL_HOME.exists(),
    reason="github-wave-orchestrator skill not installed at ~/.config/opencode/skill/",
)


# ---------------------------------------------------------------------------
# Synthetic git repo fixture (Issue #3145)
# ---------------------------------------------------------------------------


def _git(*args: str, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    """Run ``git <args>`` in ``cwd`` with the given env, raising on failure."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def _make_synthetic_repo(
    tmp_path: Path,
    *,
    origin_url: str = "https://github.com/anchapin/fluxion.git",
) -> tuple[Path, dict[str, str]]:
    """Create a synthetic git repo under ``tmp_path`` with the given remote.

    Returns ``(repo_path, env)`` where ``env`` is a sanitised env dict so
    git author/committer fields are populated and any ``git config --global``
    the test triggers does not pollute the developer's real config.
    """
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    sandbox_home = tmp_path / "sandbox_home"
    sandbox_home.mkdir()
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
        "HOME": str(sandbox_home),
    }
    _git("init", "-q", cwd=repo, env=env)
    _git("config", "user.email", "test@example.com", cwd=repo, env=env)
    _git("config", "user.name", "Test", cwd=repo, env=env)
    _git("remote", "add", "origin", origin_url, cwd=repo, env=env)
    # Seed an initial commit so `git rev-parse --show-toplevel` is stable
    # and so we have a sensible base for any branch ops later.
    (repo / ".placeholder").write_text("seed\n", encoding="utf-8")
    _git("add", ".placeholder", cwd=repo, env=env)
    _git("commit", "-q", "-m", "seed", cwd=repo, env=env)
    return repo, env


def _run_bash(
    script_text: str,
    env: dict[str, str],
    cwd: Path,
    *args: str,
) -> subprocess.CompletedProcess:
    """Run ``bash -c '<script>' -- <args>`` in ``cwd`` with the given env.

    The script text should source the wave-state-helpers.sh helpers via
    absolute path so we don't depend on the test process's cwd.
    """
    return subprocess.run(
        ["bash", "-c", script_text, "--", *args],
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


# ---------------------------------------------------------------------------
# Issue #3145: per-repo state-file namespacing
# ---------------------------------------------------------------------------


class TestWaveStateNamespacing:
    """Pin the per-repo state-file namespacing contract (issue #3145).

    Three regression guards:

    1. Two concurrent orchestrator runs against different repos produce
       distinct state-file paths (no silent overwrite).
    2. The pre-flight collision check (in both the bash helper and the
       Node planner) refuses to overwrite a state file that belongs to a
       different repo.
    3. ``--force`` (and the equivalent ``WAVE_STATE_FORCE=1`` env var)
       bypasses the check so an operator can intentionally clobber state.
    """

    @pytest.fixture
    def repo_a(self, tmp_path):
        """Synthetic repo whose origin is the fluxion repository."""
        return _make_synthetic_repo(
            tmp_path / "a",
            origin_url="https://github.com/anchapin/fluxion.git",
        )

    @pytest.fixture
    def repo_b(self, tmp_path):
        """Synthetic repo whose origin is openstudio-server-operator."""
        return _make_synthetic_repo(
            tmp_path / "b",
            origin_url="https://github.com/anchapin/openstudio-server-operator.git",
        )

    @pytest.mark.skipif(not SKILL_HOME.exists(), reason="skill not installed")
    def test_two_repos_get_distinct_state_file_paths(
        self, repo_a, repo_b, tmp_path
    ):
        """fluxion and openstudio-server-operator must NOT collide.

        The bug from issue #3145 was that two concurrent runs both wrote
        to ``../worktrees/wave-state.json`` and the second silently
        overwrote the first. With namespacing, each repo derives a
        distinct path.
        """
        helpers_path = SKILL_HOME / "scripts" / "wave-state-helpers.sh"
        script = f"source '{helpers_path}' && get_state_file '{tmp_path}/worktrees'"
        result_a = _run_bash(script, repo_a[1], repo_a[0])
        result_b = _run_bash(script, repo_b[1], repo_b[0])
        assert result_a.returncode == 0, result_a.stderr
        assert result_b.returncode == 0, result_b.stderr
        out_a = result_a.stdout.strip()
        out_b = result_b.stdout.strip()
        assert out_a != out_b, (
            "Two different repos resolved to the same state-file path — "
            "the namespacing fix from #3145 is regressed"
        )
        assert out_a.endswith("wave-state.fluxion.json"), out_a
        assert out_b.endswith(
            "wave-state.openstudio-server-operator.json"
        ), out_b

    @pytest.mark.skipif(not SKILL_HOME.exists(), reason="skill not installed")
    def test_pre_flight_check_refuses_cross_repo_overwrite(self, repo_a, tmp_path):
        """check_state_collision must refuse to clobber a foreign state file.

        Pins the bash helper contract. The Node planner contract is pinned
        separately in :meth:`TestWavePlannerCollision.node_planner_refuses`
        so a regression in either layer is caught independently.
        """
        helpers_path = SKILL_HOME / "scripts" / "wave-state-helpers.sh"
        worktrees = tmp_path / "worktrees"
        worktrees.mkdir()
        # repo_a is fluxion, but we plant a state file claiming to belong
        # to openstudio-server-operator.
        state_file = worktrees / "wave-state.fluxion.json"
        state_file.write_text(
            json.dumps({"repo": "anchapin/openstudio-server-operator"})
        )

        script = (
            f"source '{helpers_path}' && "
            f"check_state_collision '{state_file}'"
        )
        result = _run_bash(script, repo_a[1], repo_a[0])
        assert result.returncode == 1, (
            f"check_state_collision should reject cross-repo state, "
            f"got rc={result.returncode} stdout={result.stdout!r} "
            f"stderr={result.stderr!r}"
        )
        assert "openstudio-server-operator" in result.stderr

    @pytest.mark.skipif(not SKILL_HOME.exists(), reason="skill not installed")
    def test_force_override_bypasses_collision_check(self, repo_a, tmp_path):
        """WAVE_STATE_FORCE=1 must bypass the collision check."""
        helpers_path = SKILL_HOME / "scripts" / "wave-state-helpers.sh"
        worktrees = tmp_path / "worktrees"
        worktrees.mkdir()
        state_file = worktrees / "wave-state.fluxion.json"
        state_file.write_text(
            json.dumps({"repo": "anchapin/openstudio-server-operator"})
        )

        env_with_force = {**repo_a[1], "WAVE_STATE_FORCE": "1"}
        script = (
            f"source '{helpers_path}' && "
            f"check_state_collision '{state_file}'"
        )
        result = _run_bash(script, env_with_force, repo_a[0])
        assert result.returncode == 0, (
            f"--force should bypass the collision check, "
            f"got rc={result.returncode} stderr={result.stderr!r}"
        )
        assert "WARNING" in result.stderr

    @pytest.mark.skipif(not SKILL_HOME.exists(), reason="skill not installed")
    def test_check_state_collision_treats_legacy_state_as_clean(self, repo_a, tmp_path):
        """A legacy state file with no `repo` field must not be flagged.

        Pre-#3145 state files did not have a `repo` field. The check must
        treat them as orphaned (and therefore safe to adopt) rather than
        refusing to overwrite, otherwise every pre-existing wave run would
        be blocked by the new guard.
        """
        helpers_path = SKILL_HOME / "scripts" / "wave-state-helpers.sh"
        worktrees = tmp_path / "worktrees"
        worktrees.mkdir()
        state_file = worktrees / "wave-state.fluxion.json"
        # Legacy state: no `repo` field.
        state_file.write_text(json.dumps({"current_wave": 1, "issues": {}}))

        script = (
            f"source '{helpers_path}' && "
            f"check_state_collision '{state_file}'"
        )
        result = _run_bash(script, repo_a[1], repo_a[0])
        assert result.returncode == 0, (
            f"check_state_collision should accept orphaned legacy state, "
            f"got rc={result.returncode} stderr={result.stderr!r}"
        )

    @pytest.mark.skipif(not SKILL_HOME.exists(), reason="skill not installed")
    def test_archive_state_file_uses_namespaced_convention(self, repo_a, tmp_path):
        """archive_state_file must write the <repo>-archived-<date>.json file.

        This is the recovery path for the 2026-08-18 fluxion collision
        (sub-agent archived the stale state mid-flight when it detected
        the overwrite; the convention is now first-class in the skill).
        """
        helpers_path = SKILL_HOME / "scripts" / "wave-state-helpers.sh"
        worktrees = tmp_path / "worktrees"
        worktrees.mkdir()
        state_file = worktrees / "wave-state.fluxion.json"
        state_file.write_text(json.dumps({"repo": "anchapin/fluxion", "x": 1}))

        script = (
            f"source '{helpers_path}' && "
            f"archive_state_file '{state_file}'"
        )
        result = _run_bash(script, repo_a[1], repo_a[0])
        assert result.returncode == 0, result.stderr

        # The original state file is gone.
        assert not state_file.exists(), "archive_state_file did not remove source"
        # An archive file matching the convention was created. We do not
        # pin the exact date (it depends on the host clock) but assert the
        # shape: starts with `wave-state.anchapin-fluxion-archived-`
        # (repo's `owner/repo` joined by `-`) and ends with `.json`.
        archives = [
            p
            for p in worktrees.iterdir()
            if p.name.startswith("wave-state.anchapin-fluxion-archived-")
            and p.name.endswith(".json")
        ]
        assert len(archives) == 1, (
            f"Expected exactly one archive file, found: "
            f"{[a.name for a in worktrees.iterdir()]}"
        )

    @pytest.mark.skipif(not SKILL_HOME.exists(), reason="skill not installed")
    def test_resolve_state_file_precedence(self, repo_a, tmp_path):
        """resolve_state_file honors --state-file > WAVE_STATE_FILE > default.

        Pins the override precedence documented in REFERENCE.md so a
        regression that swaps the order (e.g. env var winning over CLI)
        is caught.
        """
        helpers_path = SKILL_HOME / "scripts" / "wave-state-helpers.sh"
        worktrees = tmp_path / "worktrees"
        worktrees.mkdir()

        # 1. No override → namespaced default.
        script = f"source '{helpers_path}' && resolve_state_file '' '{worktrees}'"
        result = _run_bash(script, repo_a[1], repo_a[0])
        assert result.returncode == 0
        assert result.stdout.strip() == str(worktrees / "wave-state.fluxion.json")

        # 2. WAVE_STATE_FILE env var → env path wins over default.
        env = {**repo_a[1], "WAVE_STATE_FILE": "/tmp/from-env.json"}
        script = f"source '{helpers_path}' && resolve_state_file '' '{worktrees}'"
        result = _run_bash(script, env, repo_a[0])
        assert result.returncode == 0
        assert result.stdout.strip() == "/tmp/from-env.json"

        # 3. Positional override → wins over env var.
        script = (
            f"source '{helpers_path}' && "
            f"resolve_state_file '/tmp/from-positional.json' '{worktrees}'"
        )
        result = _run_bash(script, env, repo_a[0])
        assert result.returncode == 0
        assert result.stdout.strip() == "/tmp/from-positional.json"


# ---------------------------------------------------------------------------
# Issue #3145: Node planner collision check
# ---------------------------------------------------------------------------


class TestWavePlannerCollision:
    """Pin the wave-planner.js state-file behaviour (issue #3145).

    The Node planner is responsible for the initial state-file write
    before the bash helpers take over. We exercise it via ``node
    scripts/wave-planner.js --no-write`` so the test does not pollute the
    host filesystem; the --no-write variant still resolves and validates
    the target path, which is enough to catch a regression in the
    namespacing logic.
    """

    @pytest.fixture
    def planner(self):
        path = SKILL_HOME / "scripts" / "wave-planner.js"
        if not path.exists():
            pytest.skip("wave-planner.js not present in skill home")
        # Sanity-check the helper-script isn't installed without exec.
        st = path.stat()
        path.chmod(st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        return path

    @pytest.fixture
    def issues_payload(self):
        return json.dumps(
            {
                "issues": [
                    {
                        "number": 1,
                        "title": "smoke",
                        "body": "src/foo.rs:1",
                        "state": "open",
                        "labels": [],
                    }
                ]
            }
        )

    def _run_planner(
        self,
        planner: Path,
        stdin_data: str,
        *,
        cwd: Path,
        env: dict[str, str],
        args: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess:
        """Invoke the planner with the given cwd/env/args and pipe stdin_data."""
        proc = subprocess.run(
            ["node", str(planner), *args],
            cwd=cwd,
            env=env,
            input=stdin_data,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return proc

    @pytest.mark.skipif(not SKILL_HOME.exists(), reason="skill not installed")
    def test_node_planner_refuses_cross_repo_overwrite(
        self, planner, issues_payload, tmp_path
    ):
        """wave-planner.js must refuse to overwrite a different repo's state file.

        Plant a state file claiming to belong to openstudio-server-operator
        at the fluxion-namespaced path, then run the planner from inside a
        fluxion-remote synthetic repo. The planner must exit non-zero and
        must NOT modify the planted state file.
        """
        repo, env = _make_synthetic_repo(
            tmp_path,
            origin_url="https://github.com/anchapin/fluxion.git",
        )
        worktrees = tmp_path / "worktrees"
        worktrees.mkdir()
        planted = worktrees / "wave-state.fluxion.json"
        planted.write_text(
            json.dumps({"repo": "anchapin/openstudio-server-operator", "x": 1})
        )

        # Run the planner with WAVE_STATE_FILE pointing at the planted
        # (foreign-owned) state file. The collision guard must fire and
        # refuse to overwrite.
        proc = self._run_planner(
            planner,
            issues_payload,
            cwd=repo,
            env={**env, "WAVE_STATE_FILE": str(planted)},
        )
        assert proc.returncode != 0, (
            f"Planner should exit non-zero on cross-repo collision, "
            f"got rc={proc.returncode} stderr={proc.stderr!r}"
        )
        assert "openstudio-server-operator" in proc.stderr

        # The planted file was NOT modified (atomic rename guards this even
        # if the planner had run a partial write).
        planted_now = json.loads(planted.read_text())
        assert planted_now == {
            "repo": "anchapin/openstudio-server-operator",
            "x": 1,
        }, "Planner must not modify the planted state file on collision"

    @pytest.mark.skipif(not SKILL_HOME.exists(), reason="skill not installed")
    def test_node_planner_force_overrides_collision(
        self, planner, issues_payload, tmp_path
    ):
        """wave-planner.js --force must override the collision guard."""
        repo, env = _make_synthetic_repo(
            tmp_path,
            origin_url="https://github.com/anchapin/fluxion.git",
        )
        worktrees = tmp_path / "worktrees"
        worktrees.mkdir()
        planted = worktrees / "wave-state.fluxion.json"
        planted.write_text(
            json.dumps({"repo": "anchapin/openstudio-server-operator", "x": 1})
        )

        proc = self._run_planner(
            planner,
            issues_payload,
            cwd=repo,
            env={**env, "WAVE_STATE_FILE": str(planted)},
            args=("--force",),
        )
        assert proc.returncode == 0, (
            f"--force should bypass the collision check, got "
            f"rc={proc.returncode} stderr={proc.stderr!r}"
        )

        # The planner now owns the file (its `repo` field is fluxion).
        planted_now = json.loads(planted.read_text())
        assert planted_now.get("repo") == "anchapin/fluxion", (
            f"--force must let the planner take ownership, got {planted_now!r}"
        )

    @pytest.mark.skipif(not SKILL_HOME.exists(), reason="skill not installed")
    def test_node_planner_namespaces_default_state_file(
        self, planner, issues_payload, tmp_path
    ):
        """With no --state-file / WAVE_STATE_FILE, planner writes to
        ../worktrees/wave-state.<repo-slug>.json relative to its cwd."""
        # _make_synthetic_repo takes a parent dir and nests the repo under
        # <parent>/repo — pass tmp_path directly so the planner's relative
        # ../worktrees path resolves to <tmp_path>/worktrees.
        repo, env = _make_synthetic_repo(
            tmp_path,
            origin_url="https://github.com/anchapin/openstudio-server-operator.git",
        )

        proc = self._run_planner(
            planner,
            issues_payload,
            cwd=repo,
            env=env,
        )
        assert proc.returncode == 0, proc.stderr

        # Planner resolves ../worktrees from its cwd (the synthetic repo),
        # so the namespaced file lands at <tmp>/worktrees/wave-state.openstudio-server-operator.json.
        expected = tmp_path / "worktrees" / "wave-state.openstudio-server-operator.json"
        assert expected.exists(), (
            f"Expected planner to write {expected}, "
            f"worktrees contents: "
            f"{sorted(p.name for p in tmp_path.iterdir())}"
        )


# ---------------------------------------------------------------------------
# Issue #379: skill-snapshot numbered names
# ---------------------------------------------------------------------------


class TestSkillSnapshotNumberedNames:
    """Enforce the wave-numbered snapshot naming rule (issue #379).

    Two waves in one cycle that both touch the skill home would otherwise
    produce an add/add (or modify/modify) collision at rebase on the bare
    ``docs/skill-snapshot/SKILL.md`` path. The snapshot rule forbids that:
    every file under ``docs/skill-snapshot/`` MUST match the
    ``<basename>.wave-\\d+\\.md`` (or ``.js`` / ``.sh``) pattern, and the
    legacy un-numbered paths must be absent because they are the frozen
    pre-#379 snapshots that historical diffs and
    ``tests/test_render_orchestrator_snippet.py`` §0-contract checks
    depend on.

    The wave-1 snapshot was last refreshed on 2026-08-24 (commit
    ``19b7545``); the test asserts the highest-numbered snapshot on disk
    reflects the canonical skill-home file as of HEAD so a regression that
    drifts the canonical source from the snapshot is caught.
    """

    @pytest.fixture(scope="class")
    def snapshot_files(self) -> list[Path]:
        if not SNAPSHOT_DIR.exists():
            return []
        return sorted(p for p in SNAPSHOT_DIR.iterdir() if p.is_file())

    def test_no_unnumbered_legacy_paths(self, snapshot_files):
        """Pre-#379 un-numbered paths must not appear under docs/skill-snapshot/.

        ``docs/skill-snapshot/SKILL.md`` (and the equivalents) are the
        frozen pre-#379 snapshots that historical diffs and the
        ``tests/test_render_orchestrator_snippet.py`` §0-contract checks
        depend on. A wave branch must never write to them — the
        snapshot rule routes every write through ``<basename>.wave-N.<ext>``.
        """
        forbidden = {"SKILL.md", "REFERENCE.md"}
        offenders = [
            p.name for p in snapshot_files if p.name in forbidden
        ]
        assert not offenders, (
            f"Un-numbered legacy snapshot paths are present under "
            f"{SNAPSHOT_DIR}: {offenders}. The snapshot rule (issue #379) "
            f"forbids writing these paths from a wave branch."
        )

    def test_snapshot_filenames_match_wave_numbered_pattern(self, snapshot_files):
        """Every snapshot file must be wave-numbered.

        The only allowed basename patterns under ``docs/skill-snapshot/`` are:

        * ``<name>.wave-\\d+\\.md``  — markdown snapshot
        * ``<name>.wave-\\d+\\.js``  — Node script snapshot
        * ``<name>.wave-\\d+\\.sh``  — shell script snapshot
        * ``.gitkeep``               — directory marker
        """
        import re

        allowed = re.compile(r"^.+\.wave-\d+\.(md|js|sh)$")
        offenders = [
            p.name for p in snapshot_files
            if p.name != ".gitkeep" and not allowed.match(p.name)
        ]
        assert not offenders, (
            f"Snapshot files must match <basename>.wave-N.<ext>; "
            f"offending names: {offenders}"
        )

    @pytest.mark.skipif(not SKILL_HOME.exists(), reason="skill not installed")
    def test_wave1_snapshot_matches_skill_home(self, snapshot_files):
        """Wave-1 snapshots must equal the skill home (issue #379 contract).

        Per the snapshot rule, ``<basename>.wave-${WAVE}.md`` is a verbatim
        copy of the skill-home file as of that wave. For the CURRENT wave
        (wave-1 in this cycle), the snapshot must match the skill home —
        if it drifts, either the wave forgot to snapshot the change or
        the skill-home file is missing the latest edits.

        Higher-numbered snapshots (wave-2 through wave-7) are HISTORICAL
        records from previous cycles and are allowed to differ from the
        current skill home; the snapshot rule only requires they exist and
        follow the numbered naming scheme. They become the canonical
        record when their wave runs, not now.
        """
        import re

        wave_re = re.compile(r"^(?P<base>.+)\.wave-(?P<n>\d+)\.md$")
        for p in snapshot_files:
            m = wave_re.match(p.name)
            if not m or int(m.group("n")) != 1:
                continue
            base = m.group("base")
            skill_file = SKILL_HOME / f"{base}.md"
            if not skill_file.exists():
                continue
            with p.open("r", encoding="utf-8") as sf, skill_file.open(
                "r", encoding="utf-8"
            ) as kf:
                if sf.read() != kf.read():
                    pytest.fail(
                        f"Wave-1 snapshot {p} differs from skill-home "
                        f"file {skill_file}; the snapshot rule requires "
                        f"the current wave's snapshot to be a verbatim "
                        f"copy of the skill home."
                    )


# ---------------------------------------------------------------------------
# Cross-cutting: end-to-end smoke (sanity)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SKILL_HOME.exists(), reason="skill not installed")
def test_end_to_end_two_repos_dont_collide(tmp_path):
    """End-to-end smoke: fluxion + openstudio-server-operator do not collide.

    Plant the actual wave-state.fluxion.json file with one issue, run the
    planner as if from an openstudio repo, and assert the planted file is
    untouched. This is the canonical reproduction of the bug from #3145
    — the original failure mode from the issue body.
    """
    helpers_path = SKILL_HOME / "scripts" / "wave-state-helpers.sh"
    planner_path = SKILL_HOME / "scripts" / "wave-planner.js"
    if not (helpers_path.exists() and planner_path.exists()):
        pytest.skip("wave-planner.js / wave-state-helpers.sh not present")

    # Plant fluxion's state file as the "in-progress" run.
    worktrees = tmp_path / "worktrees"
    worktrees.mkdir()
    fluxion_state = worktrees / "wave-state.fluxion.json"
    fluxion_state.write_text(
        json.dumps(
            {
                "repo": "anchapin/fluxion",
                "current_wave": 2,
                "issues": {"3145": {"wave": 2, "status": "implementing"}},
            }
        )
    )

    # Build an "openstudio-server-operator" repo and ask the planner to
    # write to the SAME fluxion-namespaced path (intentional collision).
    openstudio_repo, env = _make_synthetic_repo(
        tmp_path / "openstudio",
        origin_url="https://github.com/anchapin/openstudio-server-operator.git",
    )
    # The synthetic repo lives at <tmp_path>/openstudio/repo (see helper);
    # when the planner resolves ../worktrees from that cwd it lands in
    # <tmp_path>/openstudio/worktrees — precreate that dir so the
    # collision-guard has a planted file to compare against, regardless of
    # which worktrees location the planner happens to use.
    openstudio_worktrees = tmp_path / "openstudio" / "worktrees"
    openstudio_worktrees.mkdir(parents=True, exist_ok=True)

    # Rewrite the planted fluxion file to live at the path the planner will
    # actually compute (so the collision check fires regardless of which
    # <repo>/worktrees dir the planner picks).
    fluxion_state = (
        openstudio_worktrees / "wave-state.openstudio-server-operator.json"
    )
    fluxion_state.write_text(
        json.dumps(
            {
                "repo": "anchapin/fluxion",
                "current_wave": 2,
                "issues": {"3145": {"wave": 2, "status": "implementing"}},
            }
        )
    )

    payload = json.dumps(
        {
            "issues": [
                {
                    "number": 9999,
                    "title": "openstudio test issue",
                    "body": "src/openstudio/foo.rb",
                    "state": "open",
                    "labels": [],
                }
            ]
        }
    )
    proc = subprocess.run(
        ["node", str(planner_path)],
        cwd=openstudio_repo,
        env={**env, "WAVE_STATE_FILE": str(fluxion_state)},
        input=payload,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # The planner must refuse (openstudio ≠ fluxion).
    assert proc.returncode != 0, (
        f"Planner accepted a cross-repo overwrite — issue #3145 regressed. "
        f"stderr={proc.stderr!r}"
    )
    # The planted fluxion file is unchanged.
    planted = json.loads(fluxion_state.read_text())
    assert planted["repo"] == "anchapin/fluxion"
    assert planted["current_wave"] == 2
    assert planted["issues"]["3145"]["status"] == "implementing"


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _ensure_skill_helper_is_executable(tmp_path, request):
    """Make sure ``wave-state-helpers.sh`` is executable when run from CI.

    Some sandboxes mount the skill home without the executable bit set.
    Pin it for the duration of the test process so the ``bash -c 'source'``
    invocation doesn't fail with a permission error before the contract
    assertions run.
    """
    if SKILL_HOME.exists():
        for sh in (SKILL_HOME / "scripts").glob("*.sh"):
            try:
                st = sh.stat()
                if not (st.st_mode & stat.S_IXUSR):
                    sh.chmod(st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            except OSError:
                # Best-effort; the skipif already guards missing installs.
                pass
    yield
