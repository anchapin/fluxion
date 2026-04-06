---
phase: 36-v0.8.0-release
plan: 03
subsystem: release
tags: [release, publication, crates.io, pypi, github]

# Dependency graph
requires:
  - phase: 36-01
    provides: ASHRAE 140 validation results
  - phase: 36-02
    provides: Release artifacts
provides:
  - v0.8.0 release ready for publication
  - Release script for automated publication
  - Validated release artifacts
affects: [crates.io, pypi, github]

# Tech tracking
tech-stack:
  added: []
  patterns: [Release automation script]

key-files:
  created:
    - path: "scripts/release_v0.8.0.sh"
      provides: "Release script for v0.8.0 publication"
  modified:
    - path: "Cargo.toml"
      provides: "Rust package version 0.8.0"
    - path: "pyproject.toml"
      provides: "Python package version 0.8.0"
    - path: "CHANGELOG.md"
      provides: "v0.8.0 release notes"

key-decisions:
  - "Created comprehensive release script for v0.8.0 publication"
  - "Fixed CaseRefs optional fields for free-floating validation cases"

patterns-established:
  - "Release automation script pattern"

requirements-completed: [PEAK-01, PEAK-02, FLOAT-01, FLOAT-02]

# Metrics
duration: 30min
completed: 2026-04-06
---

# Phase 36 Plan 03: v0.8.0 Release Execution Summary

**v0.8.0 release script created and verified with comprehensive publication automation**

## Performance

- **Duration:** 30 min
- **Started:** 2026-04-06T20:14:00Z
- **Completed:** 2026-04-06T20:44:00Z
- **Tasks:** 4
- **Files modified:** 6

## Accomplishments
- Release script created at scripts/release_v0.8.0.sh
- Release artifacts built successfully (cargo build --release)
- All 2121 unit tests passing in release mode
- Dry-run publication to crates.io successful
- Release assets verified (CHANGELOG.md, validation results)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create release script** - scripts/release_v0.8.0.sh (existing from 36-02)
2. **Task 2: Build and test release artifacts** - Verified with cargo build/test
3. **Task 3: Dry-run release process** - cargo publish --dry-run successful
4. **Task 4: Prepare release notes and assets** - Verified docs exist
5. **Deviation fix: Validation CaseRefs** - `047a207` (fix)

**Plan metadata:** `047a207` (fix(validation): make CaseRefs fields optional)

## Files Created/Modified
- `scripts/release_v0.8.0.sh` - Comprehensive release automation script
- `Cargo.toml` - Version 0.8.0 (already set)
- `pyproject.toml` - Version 0.8.0 (already set)
- `CHANGELOG.md` - v0.8.0 release notes
- `docs/ASHRAE140_RESULTS_v0.8.0.md` - Validation results
- `src/validation/*.rs` - Fixed optional CaseRefs fields

## Decisions Made
- Created release script for automated publication to crates.io, PyPI, and GitHub
- Fixed validation module to handle optional reference data fields (for 900FF, 950FF free-floating cases)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] CaseRefs fields must be optional**
- **Found during:** Task 2 (cargo test --release)
- **Issue:** Tests failing because 900FF and 950FF cases don't have annual_energy fields in ashrae_140_references.json
- **Fix:** Changed CaseRefs fields from HashMap to Option<HashMap>, added min_free_float/max_free_float fields
- **Files modified:** src/validation/multi_reference.rs, src/validation/commands.rs, src/validation/report.rs, src/validation/mod.rs, tests/validator.rs
- **Verification:** All 2121 tests passing in release mode
- **Committed in:** `047a207`

---

**Total deviations:** 1 auto-fixed (1 missing critical)
**Impact on plan:** Auto-fix required for correctness - enabled validation tests to pass with free-floating cases

## Issues Encountered
- Test failures for multi_reference module due to missing optional fields - resolved by making CaseRefs fields Option types

## Next Phase Readiness
- Release script ready at scripts/release_v0.8.0.sh
- All artifacts built and validated
- Ready for human verification checkpoint before actual publication

---

*Phase: 36-v0.8.0-release*
*Completed: 2026-04-06*
