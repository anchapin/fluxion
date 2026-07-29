# Debug Report: PR 2198 CI Failures

## Executive Summary

**Root Cause**: The CI failures are NOT caused by PR 2198's code changes. PR 2198 (commit 418670e) only modifies `src/sim/hvac/equipment.rs` to normalize polynomial efficiency curves. The actual CI failure is a **pre-existing issue** in the branch caused by the `integration-fluid-energy-conservation` test missing `required-features = ["fluid"]` in its Cargo.toml definition.

## What Tests Are Failing

The `integration-fluid-energy-conservation` test fails to compile because:
```
error[E0433]: cannot find module or crate `fluxion_fluid` in this scope
  --> tests/integration/test_fluid_energy_conservation.rs:12:5
```

This test was added in commit 599c55a (NOT part of PR 2198) and is missing the required feature gate.

## Code Changes in PR 2198

PR 2198 consists of a single commit (418670e) that modifies only `src/sim/hvac/equipment.rs`:

**File changed**: `src/sim/hvac/equipment.rs` (+44 lines, -15 lines)

### Chiller Efficiency Normalization (lines 251-266)
Before:
```rust
self.efficiency_curve_cooling.cop_at(plr, outdoor_temp)
```

After:
```rust
let poly_cop = self.efficiency_curve_cooling.cop_at(plr, outdoor_temp);
let poly_cop_at_rated = self.efficiency_curve_cooling.cop_at(1.0, self.design_temp);
if poly_cop_at_rated > 0.0 && self.cooling_cop > 0.0 {
    (poly_cop / poly_cop_at_rated) * self.cooling_cop
} else {
    poly_cop
}
```

### HeatPump Efficiency Normalization (lines 579-610)
Similar normalization applied for both heating and cooling modes.

### Test Updates
- Updated `cop_design` assertion from `4.15` (raw polynomial) to `4.5` (rated COP)
- Changed `assert!(cop_hot < 4.5)` to `assert!(cop_hot < cop_design)` for better semantics

## Analysis of Why Tests Are Failing

1. **Compilation Failure (immediate)**: The `integration-fluid-energy-conservation` test uses `use fluxion_fluid::energy::{...}` at the top level, but `fluxion_fluid` is only available when the `fluid` feature is enabled. The `[[test]]` definition in Cargo.toml lacks `required-features = ["fluid"]`.

2. **Affected CI Configurations**:
   - `Test (ubuntu-latest, no-default)` - runs `cargo test --lib` without default features
   - `Test (ubuntu-latest, wiring-tracing)` - runs with `--features wiring-tracing`
   - `Test (ubuntu-latest, multi-zone)` - runs with `--features wiring-tracing,multi-zone`
   
   None of these enable the `fluid` feature, causing the test to fail compilation.

## Suggested Fixes

### Option 1: Add `required-features` (Recommended)

In `Cargo.toml`, change:
```toml
[[test]]
name = "integration-fluid-energy-conservation"
path = "tests/integration/test_fluid_energy_conservation.rs"
harness = true
```

To:
```toml
[[test]]
name = "integration-fluid-energy-conservation"
path = "tests/integration/test_fluid_energy_conservation.rs"
harness = true
required-features = ["fluid"]
```

### Option 2: Add Feature Gating in Test File

Wrap the entire test content with:
```rust
#[cfg(feature = "fluid")]
mod tests {
    // ... existing test code ...
}
```

And add `#[cfg(feature = "fluid")]` to the `use` statement.

## Note on PR 2198 Itself

The actual PR 2198 changes (efficiency normalization) are logically sound and the library unit tests pass. The HVAC BESTEST tests for HA004 should work correctly with the normalization change. The CI failures are caused by the unrelated `fluxion-fluid` test configuration issue in the branch, not by the PR's actual code changes.

## Branch Context

This branch (`fix/issue-2197-hvac-bestest-ha004`) contains many commits beyond PR 2198:
- 418670e - PR 2198: fix(hvac): normalize polynomial efficiency curves to rated COP
- 599c55a - feat(architecture): resolve #1980 — Create fluxion-fluid crate (NOT part of PR 2198)

The `fluxion-fluid` test was added in commit 599c55a which is not part of PR 2198.
