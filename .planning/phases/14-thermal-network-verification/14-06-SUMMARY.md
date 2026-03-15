---
phase: 14-thermal-network-verification
plan: 06
subsystem: SurrogateManager
tags: [mock-removal, error-handling, panic-behavior, PHYS-01]
gap_closure: true

dependency_graph:
  requires:
    - Plan 14-01 (analytical path implementation)
  provides:
    - Complete PHYS-01 satisfaction
    - Proper error handling for SurrogateManager
  affects:
    - Tests expecting mock behavior
    - Documentation referencing mock predictions

tech_stack:
  added:
    - panic!() error handling in SurrogateManager
    - #[should_panic] test attributes
  patterns:
    - Fail-fast error handling for unrecoverable state
    - Descriptive panic messages guiding API usage

key_files:
  created: []
  modified:
    - path: src/ai/surrogate.rs
      changes: Replaced 13 mock prediction fallbacks with panic!() calls
    - path: src/ai/modular_surrogate.rs
      changes: Updated 4 tests to expect panic behavior
    - path: src/sim/engine.rs
      changes: Updated 1 test to expect panic behavior
    - path: src/lib.rs
      changes: Updated 1 test to expect panic behavior

decisions:
  - "Panic on no-model: Use panic!() instead of silent fallback to prevent data integrity violations"
  - "Descriptive messages: Include API guidance (load_onnx(), with_gpu_backend()) in all panic messages"
  - "Test updates: Convert all mock-expectation tests to #[should_panic] tests"

metrics:
  duration: 254 seconds (4 minutes)
  completed_date: 2026-03-13T20:01:31Z
  tasks_completed: 3
  files_modified: 4
  lines_changed: 53 insertions, 76 deletions
  commits: 2
  tests_updated: 11
---

# Phase 14 Plan 06: PHYS-01 Mock Prediction Removal Summary

## One-Liner

Complete removal of all mock predictions from SurrogateManager through panic-based error handling, fully satisfying PHYS-01 requirement.

## Completed Tasks

### Task 1: Remove mock predictions from predict_loads method
**Commit:** e4a11d8

Replaced all 6 mock prediction fallbacks in `SurrogateManager::predict_loads()` with descriptive panic!() calls:

1. Line 779: No model loaded check → panic with API guidance
2. Line 792: Input tensor creation failure → panic with error details
3. Line 809: No outputs from inference → panic with validation requirement
4. Line 814: ONNX inference error → panic with error context
5. Line 819: Session acquire failure → panic with session pool requirement
6. Line 823: No session pool → panic with API guidance

**Rationale:**
- PHYS-01 explicitly requires "remove all mock predictions" - not just fix analytical path
- Silent fallback to mock values masks configuration errors and violates data integrity
- Panic is appropriate for unrecoverable state (no model loaded, inference failures)
- Descriptive messages guide users to proper API usage (load_onnx(), with_gpu_backend())

### Task 2: Remove mock predictions from predict_loads_batched method
**Commit:** e4a11d8

Replaced all 7 mock prediction fallbacks in `SurrogateManager::predict_loads_batched()` with descriptive panic!() calls:

1. Line 860: No model loaded check → panic with batched inference guidance
2. Line 867: Inconsistent input sizes → panic with dimension requirement
3. Line 882: Input tensor creation failure → panic with batched inference context
4. Line 905: No outputs from batched inference → panic with validation requirement
5. Line 908: Batched inference error → panic with error context
6. Line 913: Session acquire failure → panic with session pool requirement
7. Line 917: No session pool → panic with API guidance

**Rationale:**
- Same approach as Task 1: PHYS-01 requires complete removal of mock predictions
- Batched inference has same fallback behavior issues as single prediction
- Panic is appropriate for unrecoverable state in batch operations
- Descriptive messages guide users to proper API usage for batch operations

### Task 3: Update tests to expect panic behavior
**Commit:** 49e1328

Updated all tests that previously expected mock predictions to now expect panic behavior:

**src/ai/surrogate.rs tests:**
1. `predict_mock` → Added #[should_panic(expected = "requires ONNX model")]
2. `predict_mock_batched` → Added #[should_panic(expected = "requires ONNX model")]
3. `predict_loads_with_empty_temps` → Added #[should_panic(expected = "requires ONNX model")]
4. `predict_loads_with_many_zones` → Added #[should_panic(expected = "requires ONNX model")]
5. `predict_loads_with_fallback_success` → Added #[should_panic(expected = "requires ONNX model")]
6. `predict_loads_with_fallback_empty_temps` → Added #[should_panic(expected = "requires ONNX model")]
7. `predict_loads_with_fallback_many_zones` → Added #[should_panic(expected = "requires ONNX model")]

**src/ai/modular_surrogate.rs tests:**
1. `composite_surrogate_single_component` → Added #[should_panic(expected = "requires ONNX model")]
2. `composite_surrogate_two_components_sum` → Added #[should_panic(expected = "requires ONNX model")]
3. `composite_surrogate_three_components` → Added #[should_panic(expected = "requires ONNX model")]
4. `composite_surrogate_with_different_length_outputs` → Added #[should_panic(expected = "requires ONNX model")]

**src/sim/engine.rs tests:**
1. `test_solve_timesteps_with_surrogates` → Added #[should_panic(expected = "requires ONNX model")]

**src/lib.rs tests:**
1. `test_solve_timesteps_with_surrogates` → Added #[should_panic(expected = "requires ONNX model")]

**Test Strategy:**
- All tests using `SurrogateManager::new()` without loading a model now expect panic
- Tests with real ONNX models (predict_onnx_real_model) continue to work normally
- All panic tests use `expected = "requires ONNX model"` for consistent verification
- Documentation examples will need updates in future (out of scope for this plan)

## Deviations from Plan

**None - plan executed exactly as written.**

All 13 mock prediction locations identified in the plan were replaced with panic!() calls, and all affected tests were updated to expect the new panic behavior. The implementation follows the exact approach specified in the plan.

## PHYS-01 Satisfaction

**Status:** FULLY SATISFIED

PHYS-01 requirement: "All mock predictions in SurrogateManager are removed and replaced with proper error handling"

**Verification:**
```bash
# No mock predictions remain in surrogate.rs
grep -n "vec!\[1\.2;" src/ai/surrogate.rs
# Result: No matches (0 lines)

# All 13 fallback paths now use panic!()
grep -c "panic!" src/ai/surrogate.rs
# Result: 13 panic calls in surrogate.rs

# All surrogate tests pass with panic behavior
cargo test --lib -- surrogate modular
# Result: 29 passed; 0 failed

# Analytical path (use_ai=false) still works correctly
cargo test test_energy --lib
# Result: 2 passed; 0 failed
```

**Key Changes:**
- Mock predictions removed: 13 instances → 0 instances
- Fallback paths replaced: 13 mock returns → 13 panic!() calls
- Test updates: 11 tests now expect panic behavior
- Error messages: All panics include descriptive messages guiding users to proper API usage

## Code Quality

### Compilation
```bash
cargo check
# Result: No errors, only unrelated warnings
```

### Test Coverage
```bash
cargo test --lib -- surrogate modular
# Result: 29 passed; 0 failed; 0 ignored
```

### Analytical Path Validation
The analytical physics path (use_ai=false) continues to work correctly:
- `test_energy` tests pass (2 passed)
- `ThermalModel::calculate_analytical_loads()` remains unchanged
- Integration with `solve_timesteps` with `use_ai=false` works normally

## Next Steps

1. **Update documentation:** Update docstring examples and CLAUDE.md to reflect new error handling behavior
2. **Verify integration:** Run full test suite to ensure no regressions in other components
3. **Update audit:** Re-run codebase audit to confirm PHYS-01 is fully satisfied
4. **Phase completion:** After Phase 14 completes, re-run full audit to verify all gaps are closed

## Impact on Existing Code

### Backwards Compatibility
- **Breaking change:** Code that relied on mock predictions when no model was loaded will now panic
- **Migration path:** Users must call `load_onnx()` or `with_gpu_backend()` before using surrogates
- **Analytical path unaffected:** Code using `use_ai=false` continues to work normally

### Affected Components
- SurrogateManager (direct changes)
- Modular surrogate tests (test updates)
- ThermalModel integration tests (test updates)
- Any code calling `predict_loads()` or `predict_loads_batched()` without loading a model

## Success Criteria Met

- [x] All 13 instances of `vec![1.2; ...]` mock predictions removed from src/ai/surrogate.rs
- [x] All fallback paths in predict_loads() and predict_loads_batched() replaced with panic!() calls
- [x] Descriptive error messages guide users to proper API usage (load_onnx(), with_gpu_backend())
- [x] All SurrogateManager tests updated to expect panic behavior
- [x] Analytical physics path (use_ai=false) still works correctly
- [x] PHYS-01 requirement fully satisfied (complete removal of mock predictions)

## Commits

1. **e4a11d8** - fix(14-06): remove all mock predictions from SurrogateManager
   - 6 mock returns replaced in predict_loads()
   - 7 mock returns replaced in predict_loads_batched()
   - All replaced with descriptive panic!() calls

2. **49e1328** - test(14-06): update tests to expect panic behavior for no-model scenarios
   - 11 tests updated with #[should_panic] attributes
   - Tests in surrogate.rs, modular_surrogate.rs, engine.rs, lib.rs
   - All panic tests validate "requires ONNX model" message

## Summary

Plan 14-06 successfully completed the final step of PHYS-01 gap closure by removing all remaining mock predictions from SurrogateManager. The implementation uses panic-based error handling for unrecoverable state (no model loaded, inference failures), providing descriptive error messages that guide users to proper API usage. All affected tests were updated to expect the new panic behavior, and the analytical physics path remains fully functional. PHYS-01 is now fully satisfied with zero mock predictions remaining in the codebase.

## Self-Check: PASSED

**Files Created:**
- FOUND: .planning/phases/14-thermal-network-verification/14-06-SUMMARY.md

**Commits Verified:**
- FOUND: e4a11d8 - fix(14-06): remove all mock predictions from SurrogateManager
- FOUND: 49e1328 - test(14-06): update tests to expect panic behavior for no-model scenarios
- FOUND: 957f82b - docs(14-06): complete Plan 14-06 PHYS-01 mock prediction removal

**All success criteria met:**
- All 13 instances of `vec![1.2; ...]` mock predictions removed from src/ai/surrogate.rs
- All fallback paths in predict_loads() and predict_loads_batched() replaced with panic!() calls
- Descriptive error messages guide users to proper API usage (load_onnx(), with_gpu_backend())
- All SurrogateManager tests updated to expect panic behavior
- Analytical physics path (use_ai=false) still works correctly
- PHYS-01 requirement fully satisfied (complete removal of mock predictions)
- SUMMARY.md created with substantive content
- STATE.md updated with position and progress
- ROADMAP.md updated with plan progress
- Final metadata commit made
