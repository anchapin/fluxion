# Fluxion GitHub Issue Generation Prompt

## Context

You are analyzing the Fluxion building energy modeling engine to identify gaps and generate actionable GitHub issues. Fluxion is a Rust-based BEM engine with:
- Physics-based thermal networks (5R1C/6R2C/9R4C)
- AI surrogates (ONNX Runtime)
- Multi-language bindings (Python/PyO3, Node.js/NAPI, FMI 2.0)
- ASHRAE 140 validation framework

**Critical Reference Documents** (read these first):
- `ARCHITECTURE.md` - Module boundaries, trait contracts, data flow (~1154 lines)
- `KNOWN_ISSUES.md` - Open physics limitations with severity and status
- `ASHRAE140_RESULTS.md` - Current 18.8% pass rate (12/64 passed)
- `release_gates.yaml` - Pass/fail thresholds
- `docs/ASHRAE140_RESULTS.md` - Detailed per-case results

## Task

Generate 5-8 GitHub issues for the Fluxion repository that address genuine gaps blocking progress toward the project's goals:
1. Improving ASHRAE 140 validation pass rate (currently 18.8%)
2. Completing the Gauge-Solver research program (#1461, #1462, #1464, #1465)
3. Closing remaining physics gaps (roof-solar, night ventilation, shading)
4. Completing interop bridges (IDF import, IFC export)
5. Improving performance to meet gates (≥150 configs/sec)
6. Completing workspace crates (fluxion-city, fluxion-behavior, fluxion-twin)

## Issue Generation Strategy

For each issue, apply this template:

```markdown
## [Issue Title]

**Labels**: `bug` | `enhancement` | `physics` | `validation` | `interop` | `performance` | `gaps-foraging`

### Problem Statement
[Specific description of what's broken or missing. Include error messages, test failures, or mathematical deviations. Be precise - e.g., "Case 950 annual heating: Fluxion reports 34208.50 kWh vs ASHRAE ref 0.00 kWh (+342% OVER)"]

### Evidence
- Source file(s) with the issue
- Test output or calculation showing the deviation
- Reference to related issues in KNOWN_ISSUES.md if applicable

### Impact
[How does this block progress? Which release gate does it affect?]

### Root Cause Hypothesis
[For physics issues - what's the mathematical or modeling root cause per KNOWN_ISSUES.md?]

### Suggested Approach
[Concrete next step: which module to investigate, which file to modify, what test to write, what reference data to add]

### Acceptance Criteria
[How would we know this issue is closed? Be specific with test names or tolerance bands]
```

## Gap Categories to Investigate

### 1. ASHRAE 140 Physics Gaps (HIGH PRIORITY)
From `KNOWN_ISSUES.md` and `ASHRAE140_RESULTS.md`:

| Gap | Cases Affected | Status | Root Cause |
|-----|---------------|--------|------------|
| peak_heating UNDER | 610, 630, 640 (Case 600 series) | Open | Discrete-node solar-injection pathology |
| peak_cooling OVER | 610, 620, 630, 640, 650 (Case 600 series) | Open | Same root cause as above |
| annual_cooling UNDER | 620, 640, 650 | Open | Thermal mass integration |
| free-float min temp too warm | 600FF, 650FF | Open | Damped diurnal swing |
| FREE-01/03 thermal lag | 900FF, 950FF | Open | 5R1C lumped mass limitation |
| SOLAR-03 shading sensitivity | 610, 630, 910, 930 | Open | Shading coefficient not propagating correctly |
| SOLAR-04 night ventilation | 650, 950 | Open | Ventilation rate not multiplied during night hours |
| LIMIT-05 cooling direction | 900, 910, 920, 930, 940, 950 | Open | Roof-solar under-counting (~3x), routed to GaugeSolver |

**Issue candidates**:
- Investigate and fix Case 600 series peak_heating under-prediction (610, 630, 640)
- Investigate Case 650 night ventilation implementation (SOLAR-04)
- Verify shading device configuration propagation (SOLAR-03)
- Complete GaugeSolver Phase 3 - ASHRAE 140 Case 900 validation

### 2. Module Isolation Gaps

From ARCHITECTURE.md:
- HVAC BESTEST validation scaffold is empty (`tests/validation/hvac_bestest/mod.rs`) - needs case definitions
- IDF/epJSON import: `TryFrom<IdfFile> for SimulationSchema` pending (design §4.3)
- IFC export still design-only (#1121)
- `fluxion-city` (urban radiation) - status unclear
- `fluxion-behavior` (thermal comfort) - status unclear
- `fluxion-twin` (UKF digital twin) - status unclear

**Issue candidates**:
- Populate HVAC BESTEST case definitions for RP-865
- Complete IDF → SimulationSchema conversion
- Assess and complete fluxion-city urban radiation module
- Assess fluxion-behavior thermal comfort implementation

### 3. Remaining Cycle Breaks

From ARCHITECTURE.md §"Remaining cycles":
- `fluxion::sim::construction` depends on `fluxion::physics::continuous` - next cycle-break target
- `fluxion::physics::{wall_spec, method_selector, wall_properties}` reference physics internals

**Issue candidate**: Break the `physics ↔ sim` cycle in construction module

### 4. Performance Gaps

From release_gates.yaml:
- Throughput: min 150 configs/sec, current ~157 on CI, target 200
- Latency: max 10ms/config

**Issue candidates**:
- Optimize BatchOracle throughput beyond 200 configs/sec
- Implement multi-zone scaling performance test

### 5. Testing Infrastructure Gaps

- GaugeSolver Phase 3 harness uses synthetic CSV (needs real EnergyPlus Case 900 hourly data)
- `test_case_900_blind_energy_infrastructure` passes but annual ±15% tests are `#[ignore]`
- Coverage baseline has 0.0 entries (unenforced)

**Issue candidates**:
- Obtain or generate EnergyPlus hourly Case 900 reference data for GaugeSolver validation
- Enable the `#[ignore]` annual-energy Case 900 tests once roof-solar gap closes

### 6. API Completeness Gaps

From ARCHITECTURE.md §"Supporting Traits":
- `part_load_curves.rs` - coefficients not yet exposed (accessors exist but need testing)
- `CavTerminal` trait (#1903) - mentioned but VAV is implemented, CAV not

**Issue candidates**:
- Implement and test CAV terminal unit
- Add part-load curve coefficient accessors to Python bindings

## Output Format

Create a JSON file at `.agents/results/issue-proposals-{sessionId}.json` with this structure:

```json
{
  "session_id": "<uuid>",
  "generated_at": "<ISO timestamp>",
  "repository": "anchapin/fluxion",
  "issues": [
    {
      "title": "fix(physics): Investigate Case 600 series peak_heating under-prediction",
      "body": "<markdown issue body per template above>",
      "labels": ["bug", "physics", "validation"],
      "priority": "high",
      "gap_category": "ashrae_physics"
    }
  ],
  "summary": {
    "total_issues_proposed": N,
    "by_priority": {"high": N, "medium": N, "low": N},
    "by_category": {"ashrae_physics": N, "interop": N, "performance": N, ...}
  }
}
```

## Instructions

1. **Read the reference documents** - Use `read` tool to examine KNOWN_ISSUES.md, ASHRAE140_RESULTS.md, and relevant sections of ARCHITECTURE.md

2. **Identify specific gaps** - Not vague ideas, but concrete issues with:
   - File paths and line numbers where possible
   - Numerical evidence (e.g., "3.26 kW vs 4.30-5.70 kW reference")
   - Which release gate or CI check is failing

3. **Prioritize by impact** - Issues blocking the ASHRAE 140 pass rate or GaugeSolver research program are highest priority

4. **Write focused issues** - Each issue should be actionable by a single developer in 1-3 days

5. **No parameter tuning** - Issues must seek to "fix the underlying math" per RULES.md, not adjust constants to match reference values

6. **Check for duplicates** - Before proposing, grep existing issues to avoid duplicates

## Validation Criteria

Your output will be validated against:
- Each issue has a specific, measurable acceptance criterion
- Evidence citations reference actual files or test outputs
- No issue duplicates an existing open issue
- Issues are within scope (physics/validation/interop/performance, not refactoring for its own sake)
- Priority matches impact (high priority = blocks pass rate improvement or critical gate)
