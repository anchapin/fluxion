# Phase 18 Wave Analysis

## Phase Goal
Implement comprehensive diagnostic case coverage for ASHRAE 140 validation, including Cases 195-470 (in-depth diagnostics), Cases 800-810 (HVAC equipment), non-residential cases, solid conduction variants, and solar gain variants.

## Requirements
- DIAG-01: Implement ASHRAE 140 Cases 195-470 (in-depth diagnostics)
- DIAG-02: Implement ASHRAE 140 Cases 800-810 (HVAC equipment)
- DIAG-03: Implement non-residential cases from ASHRAE 140
- DIAG-04: Implement solid conduction test variants
- DIAG-05: Implement solar gain diagnostic variants

## Dependency Analysis

### Independent Work (Wave 1 - Parallel)
1. **Plan 18-01: Wave 0 Infrastructure** - Test stubs and validation module scaffolding
   - No dependencies
   - Creates test stubs and consolidated validation logic module

2. **Plan 18-02: Cases 195-470 Implementation** - In-depth diagnostics
   - No dependencies (extends ASHRAE140Case enum, independent)
   - Creates case specifications for Cases 195-470

3. **Plan 18-03: Cases 800-810 Implementation** - HVAC equipment
   - No dependencies (Phase 15 equipment complete, stubs exist)
   - Replaces TODO stubs with full implementations

### Dependent Work (Wave 2 - Parallel after Wave 1)
4. **Plan 18-04: Non-Residential Cases** - Office, Retail, School building types
   - Depends on Plan 18-01 (needs consolidated validation module)
   - Depends on Phase 17 (building profiles already exist)
   - Extends ASHRAE140Case with non-residential variants

5. **Plan 18-05: Solid Conduction Variants** - Different construction, thermal bridges
   - Depends on Plan 18-02 (uses Case 195 as baseline)
   - Adds variants: high-mass walls, no loads, no solar

### Dependent Work (Wave 3 - Sequential after Wave 2)
6. **Plan 18-06: Solar Gain Variants** - Zero solar, diffuse solar, albedo variations
   - Depends on Plan 18-02 (uses Case 195 as zero-solar baseline)
   - Adds variants: SHGC=0.3/0.6/0.9, albedo=0.1/0.5/0.9

7. **Plan 18-07: Multi-Reference DB Population** - Reference ranges for all diagnostic cases
   - Depends on Plans 18-02, 18-03, 18-04, 18-05, 18-06 (all cases implemented)
   - Populates docs/ashrae_140_references.json with diagnostic case reference ranges

### Final Work (Wave 4 - Sequential)
8. **Plan 18-08: CLI Integration and Smart Validation** - fluxion validate enhancements
   - Depends on Plan 18-07 (multi-reference DB populated)
   - Adds CLI subcommands for diagnostic case validation
   - Implements smart re-run logic

## Wave Structure

| Wave | Plans | Parallelization | Notes |
|------|-------|----------------|-------|
| 1 | 18-01, 18-02, 18-03 | 3 parallel plans | Infrastructure + core diagnostic cases |
| 2 | 18-04, 18-05 | 2 parallel plans | Non-residential + solid conduction variants |
| 3 | 18-06 | 1 plan | Solar gain variants |
| 4 | 18-07, 18-08 | 2 sequential plans | Reference data + CLI integration |

## File Ownership Analysis

### Wave 1
- 18-01: `tests/ashrae_140/diagnostics.rs` (new), `tests/ashrae_140_case_195_470.rs` (new), `tests/ashrae_140_case_800_810.rs` (modify)
- 18-02: `src/validation/ashrae_140_cases.rs` (modify - add Case196-Case470 enum variants)
- 18-03: `tests/ashrae_140_case_800_810.rs` (modify - replace TODO stubs)

No file conflicts → can run parallel

### Wave 2
- 18-04: `src/validation/ashrae_140_cases.rs` (modify - add non-residential variants)
- 18-05: `src/validation/ashrae_140_cases.rs` (modify - add solid conduction variants)
- `src/validation/ashrae_140_cases.rs` conflict → cannot run parallel

**Adjustment needed:** Merge 18-04 and 18-05 into sequential or split by different files

### Wave 3
- 18-06: `src/validation/ashrae_140_cases.rs` (modify - add solar gain variants)
- Conflicts with 18-04 and 18-05 → must be sequential

### Wave 4
- 18-07: `docs/ashrae_140_references.json` (modify)
- 18-08: `src/bin/fluxion.rs` or CLI module (modify)

No conflict → can be sequential (18-07 → 18-08)

## Revised Wave Structure

| Wave | Plans | Parallelization | Notes |
|------|-------|----------------|-------|
| 1 | 18-01, 18-02, 18-03 | 3 parallel | Infrastructure + core diagnostic cases |
| 2 | 18-04 | 1 plan | Non-residential cases (ashrae_140_cases.rs) |
| 3 | 18-05 | 1 plan | Solid conduction variants (ashrae_140_cases.rs) |
| 4 | 18-06 | 1 plan | Solar gain variants (ashrae_140_cases.rs) |
| 5 | 18-07, 18-08 | 2 sequential | Reference data + CLI integration |

## Plan Count
Total: 8 plans
- Wave 1: 3 plans (parallel)
- Wave 2: 1 plan
- Wave 3: 1 plan
- Wave 4: 1 plan
- Wave 5: 2 plans (sequential)

## Context Budget Estimate
Each plan: 2-3 tasks, ~50% context target
Total plans: 8
Estimated phase context: ~8 × 50% = 400% (over 4 phases of work)

**Adjustment:** Consider merging some plans to reduce total plan count while maintaining wave parallelization.

### Merged Plan Option A
- 18-04: Non-residential + solid conduction (both add to ashrae_140_cases.rs, sequential in same plan)
- Total: 7 plans

### Merged Plan Option B
- 18-02, 18-03, 18-04 merged: Core diagnostic cases (195-470, 800-810, non-residential)
- Total: 5 plans

**Decision:** Use Option A (7 plans) - maintains clear scope boundaries while reducing ashrae_140_cases.rs conflicts to 3 sequential plans instead of 4.

## Final Wave Structure

| Wave | Plans | Parallelization | Scope |
|------|-------|----------------|-------|
| 1 | 18-01, 18-02 | 2 parallel | Infrastructure + Cases 195-470 |
| 2 | 18-03 | 1 plan | Cases 800-810 (HVAC equipment) |
| 3 | 18-04 | 1 plan | Non-residential + solid conduction variants |
| 4 | 18-05 | 1 plan | Solar gain variants |
| 5 | 18-06, 18-07 | 2 sequential | Multi-reference DB + CLI integration |

Total: 7 plans
Estimated phase context: ~7 × 50% = 350% (3.5 phases of work - more manageable)
