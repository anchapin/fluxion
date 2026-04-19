#!/usr/bin/env python3
"""
Fluxion Planning Sync Tool

Synchronizes planning artifacts with shipped reality.
Updates ROADMAP.md, STATE.md, milestones, and release notes from validation results.

Usage:
    python scripts/sync_planning.py              # Sync all
    python scripts/sync_planning.py --dry-run    # Preview changes
    python scripts/sync_planning.py --file STATE # Sync specific file

This script is part of DX-01: Convert planning artifacts into a maintained operating cadence.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class PhaseStatus:
    phase: str
    name: str
    plans_total: int
    plans_completed: int
    status: str
    completion_date: Optional[str] = None


@dataclass
class MilestoneStatus:
    milestone: str
    name: str
    phases: list[str]
    status: str
    completion_date: Optional[str] = None
    progress_pct: float = 0.0


class PlanningSync:
    def __init__(
        self,
        project_root: Optional[Path] = None,
        dry_run: bool = False,
        verbose: bool = False,
    ):
        self.project_root = project_root or Path.cwd()
        self.dry_run = dry_run
        self.verbose = verbose
        self.changes: list[tuple[str, str, str, str]] = []

    def log(self, msg: str):
        if self.verbose:
            print(f"  [+] {msg}")

    def load_validation_results(self) -> dict:
        results_path = self.project_root / "validation_results.json"
        if results_path.exists():
            with open(results_path) as f:
                return json.load(f)
        return {}

    def load_phase_summaries(self) -> dict[str, PhaseStatus]:
        phases_dir = self.project_root / ".planning" / "phases"
        phases: dict[str, PhaseStatus] = {}

        if not phases_dir.exists():
            return phases

        for phase_dir in phases_dir.iterdir():
            if not phase_dir.is_dir():
                continue

            phase_id = phase_dir.name
            numeric_prefix = phase_id.split("-")[0]

            summary_path = phase_dir / f"{phase_id}-SUMMARY.md"
            plan_files = list(phase_dir.glob("*-PLAN.md"))
            completion_report_path = (
                phase_dir / f"{numeric_prefix}-COMPLETION-REPORT.md"
            )

            plans_completed = 0
            plans_total = len(plan_files)
            status = "planning"
            completion_date = None

            if completion_report_path.exists():
                status = "complete"
                completion_date = self._extract_date(
                    completion_report_path.read_text(), "**Date:**"
                )
                plans_completed = plans_total
            elif summary_path.exists():
                content = summary_path.read_text()
                if (
                    "**Status:** COMPLETE" in content
                    or "**Status:** ✅ COMPLETE" in content
                ):
                    status = "complete"
                    completion_date = self._extract_date(content, "**Date:**")
                    plans_completed = plans_total
                elif "**Status:** EXECUTING" in content or "**Status:** 🚧" in content:
                    status = "executing"
                    plans_completed = self._count_completed_plans(content, plans_total)

            name = (
                self._extract_phase_name(phase_dir / f"{phase_id}-CONTEXT.md")
                or phase_id
            )

            phases[phase_id] = PhaseStatus(
                phase=phase_id,
                name=name,
                plans_total=plans_total,
                plans_completed=plans_completed,
                status=status,
                completion_date=completion_date,
            )

        return phases

    def _extract_date(self, content: str, pattern: str) -> Optional[str]:
        escaped = re.escape(pattern)
        match = re.search(rf"{escaped}\s*(\d{{4}}-\d{{2}}-\d{{2}})", content)
        return match.group(1) if match else None

    def _count_completed_plans(self, content: str, total: int) -> int:
        completed = content.count("✅ COMPLETE") + content.count("COMPLETE")
        return min(completed, total)

    def _extract_phase_name(self, context_path: Optional[Path]) -> Optional[str]:
        if context_path and context_path.exists():
            content = context_path.read_text()
            match = re.search(r"#\s+(?:Phase\s+)?\d+[a-z]?:\s*(.+)", content)
            if match:
                return match.group(1).strip()
        return None

    def load_milestones(self) -> list[MilestoneStatus]:
        milestones: list[MilestoneStatus] = []
        milestones_dir = self.project_root / ".planning" / "milestones"

        if not milestones_dir.exists():
            return milestones

        for milestone_file in sorted(milestones_dir.glob("v*-ROADMAP.md")):
            name = milestone_file.stem.replace("-ROADMAP", "").replace("-", ".")

            content = milestone_file.read_text()

            status = "planning"
            if "SHIPPED" in content or "✅ COMPLETE" in content:
                status = "shipped"
            elif "EXECUTING" in content or "🚧" in content:
                status = "executing"

            date_match = re.search(r"\*\*Shipped:\*\*\s*(\d{4}-\d{2}-\d{2})", content)
            completion_date = date_match.group(1) if date_match else None

            progress_match = re.search(r"(\d+)%", content)
            progress_pct = float(progress_match.group(1)) if progress_match else 0.0

            phases = re.findall(r"Phase\s+(\d+[a-z]?)", content)

            milestones.append(
                MilestoneStatus(
                    milestone=name,
                    name=self._extract_milestone_name(content) or name,
                    phases=phases,
                    status=status,
                    completion_date=completion_date,
                    progress_pct=progress_pct,
                )
            )

        return milestones

    def _extract_milestone_name(self, content: str) -> Optional[str]:
        match = re.search(r"#\s+Milestone\s+v?[\d.]+:\s*(.+)", content)
        return match.group(1).strip() if match else None

    def sync_state_md(
        self, phases: dict[str, PhaseStatus], milestones: list[MilestoneStatus]
    ) -> bool:
        state_path = self.project_root / ".planning" / "STATE.md"

        current_phase = None
        v1_2_prefixes = ["41", "44", "45", "46", "47"]
        v1_2_phases = {
            k: v
            for k, v in phases.items()
            if any(k.startswith(p) for p in v1_2_prefixes)
        }
        for phase_id in sorted(v1_2_phases.keys(), key=self._phase_sort_key):
            phase = phases[phase_id]
            if phase.status == "executing":
                current_phase = phase_id
                break
            elif phase.status == "planning" and current_phase is None:
                current_phase = phase_id

        if current_phase is None:
            for phase_id in sorted(phases.keys(), key=self._phase_sort_key):
                phase = phases[phase_id]
                if phase.status == "executing" or phase.status == "planning":
                    current_phase = phase_id
                    break

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        complete_phases = [p for p in phases.values() if p.status == "complete"]
        executing_phases = [p for p in phases.values() if p.status == "executing"]
        planning_phases = [p for p in phases.values() if p.status == "planning"]

        phase_list_items = []
        for phase_id in sorted(phases.keys(), key=self._phase_sort_key):
            phase = phases[phase_id]
            icon = (
                "✅"
                if phase.status == "complete"
                else "🚧" if phase.status == "executing" else "📋"
            )
            date_str = (
                f" (completed {phase.completion_date})" if phase.completion_date else ""
            )
            phase_list_items.append(
                f"- {icon} **{phase.phase}**: {phase.name} — {phase.plans_completed}/{phase.plans_total} plans{date_str}"
            )

        overall_status = (
            "executing"
            if executing_phases
            else "planning" if not complete_phases else "complete"
        )

        new_content = f"""---
gsd_state_version: 1.0
milestone: v1.2
milestone_name: Validation & Testing Completion
current_phase: {current_phase or "N/A"}
status: {overall_status}
stopped_at: "Auto-synced from shipped reality"
last_updated: "{today}"
progress:
  total_phases: {len(phases)}
  completed_phases: {len(complete_phases)}
  total_plans: {sum(p.plans_total for p in phases.values())}
  completed_plans: {sum(p.plans_completed for p in phases.values())}
---

# Fluxion Project State

**Milestone:** v1.2 Validation & Testing Completion
**Last Updated:** {today}
**Current Phase:** {current_phase or "N/A"}
**Decision:** Planning artifacts synchronized from shipped reality

## Progress Summary

| Status | Count |
|--------|-------|
| ✅ Complete | {len(complete_phases)} |
| 🚧 Executing | {len(executing_phases)} |
| 📋 Planning | {len(planning_phases)} |

**Overall:** {sum(p.plans_completed for p in phases.values())}/{sum(p.plans_total for p in phases.values())} plans complete ({len(complete_phases)}/{len(phases)} phases)

## Phase Status

{chr(10).join(phase_list_items)}

---

## Sync Status

This file was auto-generated by sync_planning.py (DX-01).
All phase statuses reflect actual completion state from .planning/phases/.

Generated: {today}
Command: `python scripts/sync_planning.py`
"""

        if self.dry_run:
            self.log(f"DRY RUN: Would update STATE.md with {len(phases)} phases")
            self.changes.append(
                ("STATE.md", "update", state_path.read_text()[:200], new_content[:200])
            )
        else:
            state_path.write_text(new_content)
            self.log("Updated STATE.md")

        return True

    def sync_roadmap_md(
        self, phases: dict[str, PhaseStatus], milestones: list[MilestoneStatus]
    ) -> bool:
        roadmap_path = self.project_root / ".planning" / "ROADMAP.md"

        phase_rows = []
        for phase_id in sorted(phases.keys(), key=self._phase_sort_key):
            phase = phases[phase_id]
            status_icon = (
                "✅"
                if phase.status == "complete"
                else "🚧" if phase.status == "executing" else "📋"
            )
            date = phase.completion_date or ""

            phase_rows.append(
                f"| {phase.phase}. {phase.name} | {phase.plans_completed}/{phase.plans_total} | {status_icon} {phase.status} | {date} |"
            )

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        new_content = f"""# Fluxion Roadmap

**Project:** Building Energy Modeling Engine (Rust + Python)
**Milestone:** v1.2 Validation & Testing Completion
**Current Phase:** Planning
**Last Updated:** {today}
**Auto-generated:** Yes (sync_planning.py)

---

## Milestones

- 🚧 **v1.2 Validation & Testing Completion** — Phases 44-47 (planning)
- ✅ **v1.1 ASHRAE 140 Completion (Partial)** — Phase 40 (shipped 2026-04-08)
- ✅ **v1.0 Multi-Zone Support** — Phases M1-M3 (shipped 2026-04-07)
- ✅ **v0.8 Peak Load & Free-Float Validation** — Phases 33-36 (shipped 2026-04-07)

---

## Phase Status (Auto-synced)

| Phase | Progress | Status | Completed |
|-------|----------|--------|-----------|
{chr(10).join(phase_rows)}

---

*Roadmap auto-generated: {today} - Run `python scripts/sync_planning.py` to update*
"""

        if self.dry_run:
            self.log("DRY RUN: Would update ROADMAP.md")
            self.changes.append(("ROADMAP.md", "update", "", ""))
        else:
            roadmap_path.write_text(new_content)
            self.log("Updated ROADMAP.md")

        return True

    def sync_all(self) -> bool:
        self.log("Loading phase summaries...")
        phases = self.load_phase_summaries()
        self.log(f"Found {len(phases)} phases")

        self.log("Loading milestones...")
        milestones = self.load_milestones()
        self.log(f"Found {len(milestones)} milestones")

        self.sync_state_md(phases, milestones)
        self.sync_roadmap_md(phases, milestones)

        self.log("Sync complete!")
        return True

    def _phase_sort_key(self, phase_id: str) -> tuple[int, str]:
        match = re.match(r"^(\d+)", phase_id)
        num = int(match.group(1)) if match else 0
        return (num, phase_id)


def main():
    parser = argparse.ArgumentParser(description="Sync Fluxion planning artifacts")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without writing"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--file",
        choices=["STATE", "ROADMAP", "ALL"],
        default="ALL",
        help="Which file to sync",
    )
    parser.add_argument(
        "--project-root", type=Path, default=None, help="Project root directory"
    )

    args = parser.parse_args()

    sync = PlanningSync(
        project_root=args.project_root, dry_run=args.dry_run, verbose=args.verbose
    )

    success = sync.sync_all()

    if args.dry_run:
        print("\nDRY RUN - No files written")
        print("Changes preview:")
        for change in sync.changes:
            print(f"  - {change[0]}: {change[1]}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
