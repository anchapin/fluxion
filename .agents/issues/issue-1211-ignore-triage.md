## Issue Description

There are **33 `#[ignore]` tests** across the test suite. Many have "TODO: Fix" comments but **no associated GitHub issue numbers**.

### Conduction (7 tests)
- Lines 459, 540, 598: `#[ignore = "Requires FiveR1CSolver transient dynamics"]`
- Lines 674, 744, 809: Time constant tests

### Zone Balance (2 tests)
- Lines 845, 908: `#[ignore = "blocked by cooling-load physics gap"]`

### Conductance (2 tests)
- Lines 133, 396: `#[ignore // TODO: Fix - ventilation conductance calculation]`

### Solar (3 tests)
- Lines 16, 59, 99: `#[ignore // TODO: Fix - solar distribution calculation issue]`

### Step Physics (1 test)
- Line 267: `#[ignore // TODO: Fix - energy accumulation calculation issue]`

### Others
- `test_constants_integration.rs`
- `test_session_pool.rs`
- `test_modular_surrogates.rs`
- `energyplus_comparison_tests.rs`
- `case_900ff_multinode_validation.rs`
- `issue_1168_free_float_diagnostic.rs`

## Required Action

For each `#[ignore]` test:
1. **If it's a known bug** → Create GitHub issue, link in `#[ignore]` comment
2. **If it's obsolete** → Remove the test entirely
3. **If it's waiting on external dependency** → Add issue number and target milestone

## Acceptance Criteria

- [ ] Zero `#[ignore]` tests without associated GitHub issue
- [ ] Every `#[ignore]` comment includes `fixes #XXXX` or `ref: #XXXX`
- [ ] Orphaned "TODO: Fix" comments converted to issues