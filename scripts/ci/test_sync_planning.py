"""
Tests for ``scripts/sync_planning.py`` — Issue #4195.

The sync must patch only its owned (marker-delimited) sections, preserve
unknown text, derive milestone metadata from ``.planning/milestones/``
instead of hardcoding it, be idempotent, and leave the tree untouched
in ``--dry-run`` mode.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SCRIPT_NAME = "sync_planning"


@pytest.fixture
def planner(load_script):
    return load_script(SCRIPT_NAME)


def _make_planning_tree(root: Path) -> Path:
    """Build a minimal ``.planning/`` source tree."""
    planning = root / ".planning"
    phases = planning / "phases"
    milestones = planning / "milestones"

    phase_dir = phases / "44-test-phase"
    phase_dir.mkdir(parents=True)
    (phase_dir / "PLAN.md").write_text(
        "# Phase 44\n\n## Plans\n\n- [x] Plan A\n- [ ] Plan B\n",
        encoding="utf-8",
    )

    milestones.mkdir(parents=True)
    (milestones / "v2-0-ROADMAP.md").write_text(
        "# v2.0 Test Milestone\n\n🚧 EXECUTING\n\nPhase 44\n\nProgress: 50%\n",
        encoding="utf-8",
    )
    return planning


def _sync(planner_mod, root: Path, dry_run: bool = False):
    sync = planner_mod.PlanningSync(project_root=root, dry_run=dry_run)
    assert sync.sync_all()
    return sync


# ---------------------------------------------------------------------------


def test_sync_creates_marker_sections(planner, tmp_path):
    _make_planning_tree(tmp_path)
    _sync(planner, tmp_path)

    state = (tmp_path / ".planning" / "STATE.md").read_text()
    assert "<!-- sync-planning:state-frontmatter:START -->" in state
    assert "<!-- sync-planning:state-summary:START -->" in state

    roadmap = (tmp_path / ".planning" / "ROADMAP.md").read_text()
    assert "<!-- sync-planning:roadmap-header:START -->" in roadmap
    assert "<!-- sync-planning:roadmap-milestones:START -->" in roadmap
    assert "<!-- sync-planning:roadmap-phases:START -->" in roadmap


def test_unknown_text_is_preserved(planner, tmp_path):
    """Issue #4195: hand-written text outside markers must survive."""
    planning = _make_planning_tree(tmp_path)
    state_path = planning / "STATE.md"
    state_path.write_text(
        "# My hand-written header\n\n"
        "Some operator notes that must survive.\n\n"
        "<!-- sync-planning:state-frontmatter:START -->\n"
        "old frontmatter\n"
        "<!-- sync-planning:state-frontmatter:END -->\n",
        encoding="utf-8",
    )

    _sync(planner, tmp_path)

    state = state_path.read_text()
    assert "# My hand-written header" in state
    assert "Some operator notes that must survive." in state
    assert "old frontmatter" not in state


def test_milestone_comes_from_planning_source(planner, tmp_path):
    """Issue #4195: no hardcoded v1.2 — milestone derived from source."""
    _make_planning_tree(tmp_path)
    _sync(planner, tmp_path)

    state = (tmp_path / ".planning" / "STATE.md").read_text()
    roadmap = (tmp_path / ".planning" / "ROADMAP.md").read_text()
    # The fixture milestone is v2.0; v1.2 must not appear.
    assert "v2.0" in state
    assert "v2.0" in roadmap
    assert "v1.2" not in state
    assert "v1.2" not in roadmap


def test_idempotent_second_run_is_byte_identical(planner, tmp_path):
    """Issue #4195: idempotence."""
    _make_planning_tree(tmp_path)
    _sync(planner, tmp_path)
    state_path = tmp_path / ".planning" / "STATE.md"
    roadmap_path = tmp_path / ".planning" / "ROADMAP.md"
    first_state, first_roadmap = state_path.read_bytes(), roadmap_path.read_bytes()

    _sync(planner, tmp_path)

    assert state_path.read_bytes() == first_state
    assert roadmap_path.read_bytes() == first_roadmap


def test_dry_run_writes_nothing(planner, tmp_path):
    """Issue #4195: dry-run immutability."""
    planning = _make_planning_tree(tmp_path)
    _sync(planner, tmp_path, dry_run=True)

    assert not (planning / "STATE.md").exists()
    assert not (planning / "ROADMAP.md").exists()


def test_dry_run_does_not_modify_existing(planner, tmp_path):
    planning = _make_planning_tree(tmp_path)
    state_path = planning / "STATE.md"
    state_path.write_text("original content\n", encoding="utf-8")

    _sync(planner, tmp_path, dry_run=True)

    assert state_path.read_text() == "original content\n"
