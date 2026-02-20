# Parallel Agent Launch Guide

This guide shows how to launch background agents to work on parallel tracks using the Fluxion parallel issues execution plan.

## Quick Start

1. **Review the execution plan**: See `docs/PARALLEL_EXECUTION_PLAN.md`

2. **Setup worktrees**: Run the setup script
   ```bash
   ./scripts/setup_parallel_worktrees.sh
   ```

3. **Launch agents**: Use the Task tool to launch background agents for each track

---

## Agent Launch Commands

### Track 1: Solar & HVAC (Issues #303, #299, #281, #278, #276)

```
Task(
  subagent_type="general-purpose",
  prompt="Work on Track 1 issues in this order:
   1. #303: Detailed Internal Radiation Network (HIGH priority, 120 pts)
   2. #299: Refine Window Angular Dependence Model (HIGH priority, 120 pts)
   3. #281: Investigation: Construction U-values (MEDIUM priority, 80 pts)
   4. #278: Investigation: Solar gain calculation (MEDIUM priority, 80 pts)
   5. #276: Enhancement: HVAC control logic (MEDIUM priority, 80 pts)

  Worktree path: ../feature-issue-303-detailed-internal-radiation-network

  Instructions:
   1. Navigate to the worktree
   2. Read the issue details on GitHub
   3. Implement the fix/enhancement
   4. Run tests: cargo test
   5. Format code: cargo fmt
   6. Run linter: cargo clippy
   7. When an issue is complete, create a PR:
      - git push origin feature/issue-XXX
      - gh pr create --title 'Fix #XXX: Title' --body 'Closes #XXX'
      - Move to next issue in the track

  Important:
   - Solar issues modify: src/sim/solar.rs, src/sim/shading.rs, src/sim/sky_radiation.rs
   - HVAC issues modify: src/sim/hvac.rs, src/sim/demand_response.rs
   - Follow CLAUDE.md guidelines for physics changes
   - Document any parameter changes",
  run_in_background=true
)
```

### Track 2: Physics Core & Construction (Issues #273, #294, #280, #272)

```
Task(
  subagent_type="general-purpose",
  prompt="Work on Track 2 issues in this order:
   1. #273: Case 960 multi-zone investigation (CRITICAL priority, 150 pts)
   2. #294: ISO 13790 Annex C Mapping (HIGH priority, 120 pts)
   3. #280: Internal heat gains investigation (MEDIUM priority, 80 pts)
   4. #272: Peak load values investigation (MEDIUM priority, 80 pts)

  Worktree path: ../feature-issue-273-investigation-case-960-multi-zone

  Instructions:
   1. Navigate to the worktree
   2. Read the issue details on GitHub
   3. For #273: Diagnose why Case 960 has 20x higher cooling energy
      - Check inter-zone heat transfer
      - Verify solar gain distribution
      - Compare with reference programs
   4. For #294: Implement ISO 13790 Annex C for construction parameters
   5. For investigations: Create diagnostic outputs and analysis
   6. Run tests: cargo test
   7. Format code: cargo fmt
   8. Run linter: cargo clippy
   9. Create PRs when complete

  Important:
   - Core files: src/sim/engine.rs, src/sim/thermal_model.rs, src/sim/construction.rs
   - Validation files: src/validation/, src/validation/ashrae_140/
   - Run ASHRAE 140 validation after changes: fluxion validate --all
   - Document findings in issue comments",
  run_in_background=true
)
```

### Track 3: Zone Physics & Radiation (Issues #302, #295, #279, #274)

```
Task(
  subagent_type="general-purpose",
  prompt="Work on Track 3 issues in this order:
   1. #302: Inter-Zone Longwave Radiation (CRITICAL priority, 150 pts)
   2. #295: Multiple Surface Conductances (HIGH priority, 120 pts)
   3. #279: Infiltration modeling investigation (MEDIUM priority, 80 pts)
   4. #274: Thermal mass modeling investigation (MEDIUM priority, 80 pts)

  Worktree path: ../feature-issue-302-refine-inter-zone-longwave-radiation-case-960-

  Instructions:
   1. Navigate to the worktree
   2. Read the issue details on GitHub
   3. For #302: Fix inter-zone longwave radiation coupling
      - Implement radiative exchange between zones
      - Improve surface-to-surface coupling
      - This is CRITICAL for Case 960 fix
   4. For #295: Implement per-zone surface conductances (h_is)
   5. For investigations: Analyze and document findings
   6. Run tests: cargo test
   7. Format code: cargo fmt
   8. Run linter: cargo clippy
   9. Create PRs when complete

  Important:
   - Core files: src/sim/engine.rs, src/sim/thermal_model.rs, src/sim/construction.rs
   - Ventilation files: src/sim/ventilation.rs, src/sim/occupancy.rs
   - Validation files: src/validation/, src/validation/ashrae_140/
   - This track contains one of the CRITICAL path issues (#302)",
  run_in_background=true
)
```

### Track 4: Ventilation & Tools (Issues #301, #297, #304, #275, #277)

```
Task(
  subagent_type="general-purpose",
  prompt="Work on Track 4 issues in this order:
   1. #301: Dynamic Sensitivity Tensors (HIGH priority, 120 pts)
   2. #297: Geometric Solar Distribution (HIGH priority, 120 pts)
   3. #304: Automated Hourly Delta Analysis (MEDIUM priority, 80 pts)
   4. #275: Free-floating temperature validation (MEDIUM priority, 80 pts)
   5. #277: Roadmap tracking (LOW priority, 20 pts)

  Worktree path: ../feature-issue-301-dynamic-sensitivity-tensors-for-variable-infiltrat

  Instructions:
   1. Navigate to the worktree
   2. Read the issue details on GitHub
   3. For #301: Implement dynamic sensitivity tensors for variable infiltration
      - Update sensitivity calculations when ventilation changes
      - Handle night ventilation (Case 650)
   4. For #297: Implement beam-to-floor solar distribution
   5. For #304: Create automated hourly delta analysis tool
      - Script to compare Fluxion vs EnergyPlus hour-by-hour
      - Add to tools/ directory
   6. For investigations: Analyze and document findings
   7. Run tests: cargo test
   8. Format code: cargo fmt
   9. Run linter: cargo clippy
   10. Create PRs when complete

  Important:
   - Ventilation files: src/sim/ventilation.rs, src/sim/occupancy.rs
   - Solar files: src/sim/solar.rs, src/sim/shading.rs
   - Tool files: tools/, src/validation/diagnostic.rs
   - Update roadmap #277 as milestones are achieved",
  run_in_background=true
)
```

---

## Monitoring Agent Progress

After launching agents, monitor their progress using TaskOutput:

```bash
# Check Track 1 agent progress
TaskOutput(task_id="track-1-agent")

# Check Track 2 agent progress
TaskOutput(task_id="track-2-agent")

# Check Track 3 agent progress
TaskOutput(task_id="track-3-agent")

# Check Track 4 agent progress
TaskOutput(task_id="track-4-agent")
```

---

## Completion Workflow

When an agent completes an issue:

1. **Navigate to worktree**:
   ```bash
   cd ../feature-issue-XXX
   ```

2. **Run final validation**:
   ```bash
   cargo fmt
   cargo clippy
   cargo test
   fluxion validate --all  # For validation-related issues
   ```

3. **Push and create PR**:
   ```bash
   git push origin feature/issue-XXX
   gh pr create --title "Fix #XXX: Issue Title" --body "Closes #XXX"
   ```

4. **Remove worktree**:
   ```bash
   cd ../fluxion
   git worktree remove ../feature-issue-XXX
   ```

5. **Continue to next issue** in the track

---

## Stopping Agents

If you need to stop an agent:

```bash
TaskStop(task_id="track-1-agent")
```

---

## Tips for Success

1. **Start with CRITICAL issues first**: Tracks 2 and 3 contain CRITICAL issues (#273, #302)

2. **Frequent testing**: Run `cargo test` after each significant change

3. **Document physics changes**: Update CLAUDE.md if you change parameter mappings

4. **Validate against analytical baseline**: For physics changes, compare with ground truth

5. **Use proper commit messages**: Follow `<type>(<scope>): <subject>` format

6. **Check for file conflicts**: If tracks modify similar files, coordinate or work sequentially

---

## Emergency Workflow

If an agent gets stuck or conflicts arise:

1. **Stop the agent**: `TaskStop(task_id="track-X-agent")`

2. **Resolve conflicts manually**:
   ```bash
   cd ../feature-issue-XXX
   git status
   # Resolve conflicts
   git add .
   git commit -m "fix: resolve conflicts"
   ```

3. **Restart agent with updated instructions**

---

## Success Metrics

Track progress against these goals:

- **Week 1-2**: CRITICAL issues complete (#273, #302)
- **Week 2-4**: HIGH priority issues complete (6 issues)
- **Week 4-6**: MEDIUM issues complete (9 issues)

Final goals:
- ASHRAE 140 pass rate: ≥ 70% (25/36 metrics)
- Mean Absolute Error: < 25%
- All 18 issues resolved

---

## References

- **docs/PARALLEL_EXECUTION_PLAN.md**: Full execution plan
- **docs/ARCHITECTURE.md**: Architecture documentation
- **CLAUDE.md**: Developer guidelines
- **Issue #277**: ASHRAE 140 roadmap
