# Fluxion Parallel Issues Execution Plan

**Date**: 2026-02-19
**Repository**: anchapin/fluxion
**Total Issues**: 18
**Max Parallel Tracks**: 4

## Overview

This plan analyzes all open GitHub issues and creates an execution strategy for parallel development using git worktrees. Issues are prioritized based on ASHRAE 140 validation goals and grouped into 4 parallel tracks to maximize throughput while minimizing conflicts.

## Priority Matrix

| Priority | Score Range | Description |
|----------|-------------|-------------|
| **CRITICAL** | ≥150 pts | Blocks validation, core physics issues |
| **HIGH** | 120-149 pts | Major accuracy improvements |
| **MEDIUM** | 80-119 pts | Enhancements, investigations |
| **LOW** | <80 pts | Tools, future features |

## Issue Breakdown by Priority

- **CRITICAL (≥150 pts)**: 2 issues (300 pts)
- **HIGH (120-149 pts)**: 6 issues (720 pts)
- **MEDIUM (80-119 pts)**: 9 issues (720 pts)
- **LOW (<80 pts)**: 1 issue (20 pts)

## Issue Breakdown by Component

| Component | Issues | Priority Score |
|-----------|--------|----------------|
| VALIDATION | 10 | 1060 pts |
| SOLAR | 3 | 320 pts |
| VENTILATION | 2 | 200 pts |
| TOOLS | 1 | 80 pts |
| HVAC | 1 | 80 pts |
| PHYSICS_CORE | 1 | 20 pts |

---

## ⚠️ CRITICAL PATH

These issues **must be completed first** as they block validation progress:

1. **#302: Refine Inter-Zone Longwave Radiation (Case 960)**
   - Priority: CRITICAL (150 pts)
   - Files: `src/sim/engine.rs`, `src/sim/thermal_model.rs`, `src/sim/construction.rs`
   - Impact: Fixes multi-zone radiation coupling

2. **#273: Investigation: Case 960 multi-zone sunspace cooling energy 20x higher than reference**
   - Priority: CRITICAL (150 pts)
   - Files: `src/sim/engine.rs`, `src/sim/thermal_model.rs`, `src/sim/construction.rs`
   - Impact: Diagnoses 20x cooling energy discrepancy

---

## PARALLEL TRACKS

### Track 1: Solar & HVAC (480 pts)

**Issues**:
1. **#303: Detailed Internal Radiation Network** (HIGH, 120 pts)
   - Files: `src/sim/engine.rs`, `src/sim/thermal_model.rs`, `src/sim/construction.rs`

2. **#299: Refine Window Angular Dependence Model** (HIGH, 120 pts)
   - Files: `src/sim/solar.rs`, `src/sim/shading.rs`, `src/sim/sky_radiation.rs`

3. **#281: Investigation: Construction U-values** (MEDIUM, 80 pts)
   - Files: `src/sim/engine.rs`, `src/sim/thermal_model.rs`, `src/sim/construction.rs`

4. **#278: Investigation: Solar gain calculation** (MEDIUM, 80 pts)
   - Files: `src/sim/solar.rs`, `src/sim/shading.rs`, `src/sim/sky_radiation.rs`

5. **#276: Enhancement: HVAC control logic** (MEDIUM, 80 pts)
   - Files: `src/sim/hvac.rs`, `src/sim/demand_response.rs`

### Track 2: Physics Core & Construction (430 pts)

**Issues**:
1. **#273: Case 960 multi-zone investigation** (CRITICAL, 150 pts)
   - Files: `src/sim/engine.rs`, `src/sim/thermal_model.rs`, `src/sim/construction.rs`

2. **#294: ISO 13790 Annex C Mapping** (HIGH, 120 pts)
   - Files: `src/sim/engine.rs`, `src/sim/thermal_model.rs`, `src/sim/construction.rs`

3. **#280: Investigation: Internal heat gains** (MEDIUM, 80 pts)
   - Files: `src/validation/`, `src/validation/ashrae_140/`

4. **#272: Investigation: Peak load values** (MEDIUM, 80 pts)
   - Files: `src/validation/`, `src/validation/ashrae_140/`

### Track 3: Zone Physics & Radiation (430 pts)

**Issues**:
1. **#302: Inter-Zone Longwave Radiation** (CRITICAL, 150 pts)
   - Files: `src/sim/engine.rs`, `src/sim/thermal_model.rs`, `src/sim/construction.rs`

2. **#295: Multiple Surface Conductances** (HIGH, 120 pts)
   - Files: `src/sim/engine.rs`, `src/sim/thermal_model.rs`, `src/sim/construction.rs`

3. **#279: Investigation: Infiltration modeling** (MEDIUM, 80 pts)
   - Files: `src/sim/ventilation.rs`, `src/sim/occupancy.rs`, `src/validation/`

4. **#274: Investigation: Thermal mass modeling** (MEDIUM, 80 pts)
   - Files: `src/sim/engine.rs`, `src/sim/thermal_model.rs`, `src/sim/construction.rs`

### Track 4: Ventilation & Tools (420 pts)

**Issues**:
1. **#301: Dynamic Sensitivity Tensors** (HIGH, 120 pts)
   - Files: `src/sim/ventilation.rs`, `src/sim/occupancy.rs`, `src/validation/`

2. **#297: Geometric Solar Distribution** (HIGH, 120 pts)
   - Files: `src/sim/solar.rs`, `src/sim/shading.rs`, `src/sim/sky_radiation.rs`

3. **#304: Automated Hourly Delta Analysis** (MEDIUM, 80 pts)
   - Files: `src/validation/`, `src/validation/ashrae_140/`, `tools/`

4. **#275: Investigation: Free-floating temps** (MEDIUM, 80 pts)
   - Files: `src/validation/`, `src/validation/ashrae_140/`

5. **#277: Roadmap tracking** (LOW, 20 pts)
   - Files: `src/sim/engine.rs`, `src/sim/thermal_model.rs`, `src/sim/construction.rs`

---

## GIT WORKTREE SETUP COMMANDS

```bash
# Create all worktrees
git worktree add ../feature-issue-302-refine-inter-zone-longwave-radiation-case-960- -b feature/issue-302
git worktree add ../feature-issue-273-investigation-case-960-multi-zone-sunspace-cooling -b feature/issue-273
git worktree add ../feature-issue-303-detailed-internal-radiation-network -b feature/issue-303
git worktree add ../feature-issue-301-dynamic-sensitivity-tensors-for-variable-infiltrat -b feature/issue-301
git worktree add ../feature-issue-299-refine-window-angular-dependence-model -b feature/issue-299
git worktree add ../feature-issue-297-geometric-solar-distribution-beam-to-floor-logic- -b feature/issue-297
git worktree add ../feature-issue-295-implement-multiple-surface-conductances-h_is-per-z -b feature/issue-295
git worktree add ../feature-issue-294-implement-rigorous-iso-13790-annex-c-mapping-for-c -b feature/issue-294
git worktree add ../feature-issue-304-automated-hourly-delta-analysis-against-energyplus -b feature/issue-304
git worktree add ../feature-issue-281-investigation-construction-u-values-and-thermal-re -b feature/issue-281
git worktree add ../feature-issue-280-investigation-internal-heat-gains-scheduling-and-m -b feature/issue-280
git worktree add ../feature-issue-279-investigation-infiltration-modeling-and-air-change -b feature/issue-279
git worktree add ../feature-issue-278-investigation-solar-gain-calculation-accuracy-for- -b feature/issue-278
git worktree add ../feature-issue-276-enhancement-implement-ideal-hvac-control-logic-wit -b feature/issue-276
git worktree add ../feature-issue-275-investigation-free-floating-temperature-validation -b feature/issue-275
git worktree add ../feature-issue-274-investigation-thermal-mass-modeling-differences-be -b feature/issue-274
git worktree add ../feature-issue-272-investigation-peak-load-values-significantly-lower -b feature/issue-272
git worktree add ../feature-issue-277-roadmap-ashrae-140-ci-pass-rate-improvement-milest -b feature/issue-277
```

---

## RECOMMENDED EXECUTION ORDER

### Phase 1: Critical Investigation (Week 1-2)

Start with CRITICAL priority issues in parallel:

- **#302**: Refine Inter-Zone Longwave Radiation (Case 960)
- **#273**: Investigation: Case 960 multi-zone sunspace

**Goal**: Diagnose and fix the 20x cooling energy discrepancy in Case 960.

### Phase 2: High-Priority Fixes (Week 2-4)

Address HIGH priority issues based on investigation results. These can work in 3-4 parallel tracks:

- **#303**: Detailed Internal Radiation Network
- **#301**: Dynamic Sensitivity Tensors for Variable Infiltration/Ventilation
- **#299**: Refine Window Angular Dependence Model
- **#297**: Geometric Solar Distribution (Beam-to-Floor Logic)
- **#295**: Implement Multiple Surface Conductances (h_is) per Zone
- **#294**: Implement Rigorous ISO 13790 Annex C Mapping

**Goal**: Achieve 50%+ pass rate in ASHRAE 140 validation.

### Phase 3: Validation & Tools (Week 4-6)

Complete remaining MEDIUM priority issues:

- **#304**: Build automated hourly delta analysis tool
- **#281-#280**: Construction U-values and internal heat gains investigations
- **#279**: Infiltration modeling investigation
- **#278**: Solar gain calculation investigation
- **#276**: HVAC control logic enhancement
- **#275**: Free-floating temperature validation
- **#274**: Thermal mass modeling investigation
- **#272**: Peak load values investigation

**Goal**: Achieve 70%+ pass rate and build diagnostic tooling.

---

## PARALLEL EXECUTION WORKFLOW

### Step 1: Setup Worktrees

Run the git worktree commands above to create parallel work environments.

### Step 2: Assign Agents

For each track, launch a background agent using the Task tool:

```bash
# Example: Launch agent for Track 1 (Solar & HVAC)
Task(
  subagent_type="general-purpose",
  prompt="Work on Track 1 issues: #303, #299, #281, #278, #276.
  Worktree: ../feature-issue-303-detailed-internal-radiation-network
  Work through issues in priority order.
  Run tests frequently: cargo test
  When done, create PRs for each issue.",
  run_in_background=true
)
```

Each agent receives:
- Track number and component area
- List of issues to work on
- Worktree path
- Dependencies to check

### Step 3: Monitor Progress

Use TaskOutput to check agent progress:

```bash
TaskOutput(task_id="agent-track-1")
```

Each agent should:
1. Navigate to worktree
2. Work through issues in priority order
3. Run tests frequently: `cargo test`
4. Create PRs when done

### Step 4: Create Pull Requests

When an agent completes work for an issue:

```bash
cd ../feature-issue-XXX
git push origin feature/issue-XXX
gh pr create --title "Fix #XXX: Issue Title" --body "Closes #XXX"
cd ../fluxion
git worktree remove ../feature-issue-XXX
```

### Step 5: Track Completion

Update roadmap #277 as milestones are achieved.

---

## SUCCESS CRITERIA

### Phase 1 (Week 1-2)
- [ ] Case 960 cooling energy within 2x of reference (down from 20x)
- [ ] Inter-zone radiation coupling validated

### Phase 2 (Week 2-4)
- [ ] ASHRAE 140 pass rate ≥ 50% (18/36 metrics)
- [ ] Mean Absolute Error < 50%
- [ ] At least 9 cases fully passing

### Phase 3 (Week 4-6)
- [ ] ASHRAE 140 pass rate ≥ 70% (25/36 metrics)
- [ ] Mean Absolute Error < 25%
- [ ] Hourly delta analysis tool operational

---

## NOTES

1. **File Conflicts**: Issues within the same track may have file dependencies. Work through them sequentially within each track.

2. **Testing**: Always run `cargo test` before committing. For physics changes, validate against analytical baseline.

3. **Code Quality**: Run `cargo fmt && cargo clippy` before creating PRs.

4. **CI Integration**: Update ASHRAE 140 validation thresholds as pass rate improves (see #277).

5. **Documentation**: Document any physics parameter changes in CLAUDE.md and relevant docs.

---

## References

- **Issue #277**: ASHRAE 140 CI pass rate improvement milestones (roadmap)
- **Issue #271**: Annual energy variance investigation
- **docs/ARCHITECTURE.md**: Core architecture documentation
- **CLAUDE.md**: Developer guidelines and patterns
