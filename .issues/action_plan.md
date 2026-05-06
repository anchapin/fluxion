# Issue #666: Free-Floating Cases Excessive Temperatures - Root Cause Analysis

## Summary of Findings

### Root Causes Identified (Priority Order)

#### 1. **CTF Solver Warmup Missing** (PRIMARY - 90% likely)
- **Location**: `src/physics/ctf_solver_wrapper.rs:148`
- **Issue**: `CTFSolver::new()` initializes all history to 20°C with zero flux
- **Problem**: `with_warmup()` exists but is NOT called during initialization
- **Impact**: Unphysical heat fluxes at simulation start cause temperature runaway

#### 2. **Window SHGC Mismatch for Low-Mass Cases** (SECONDARY)
- **Location**: `src/validation/ashrae_140_cases.rs:2263`
- **Issue**: Case 600FF uses `double_clear_glass()` (SHGC=0.789)
- **Should be**: Single pane clear glass (SHGC≈0.86) per ASHRAE 140 Table 3
- **Impact**: 600FF gets ~3% less solar gain than specified

#### 3. **Weather Data Source Mismatch** (UNCLEAR IMPACT)
- **Location**: `src/validation/ashrae_140_validator.rs:404-406`
- **Issue**: Uses Denver TMY instead of ASHRAE 140 synthetic clear-day data
- **Impact**: Variable cloud conditions vs prescribed clear-day values

## Files to Modify

1. `src/physics/ctf_solver_wrapper.rs:148` - Use `with_warmup()` instead of `new()`
2. `src/validation/ashrae_140_cases.rs` - Add `single_clear_glass()` method, fix case_600ff()

## Verification

Before any fix, current baseline temperatures:
- 600FF: Min=-28.38°C, Max=105.85°C (Ref: -18.8 to -15.6 / 64.9 to 75.1)
- 900FF: Min=-10.78°C, Max=137.71°C (Ref: -6.4 to -1.6 / 41.8 to 46.4)

After CTF warmup fix, expect:
- 900FF max temp should drop significantly (perhaps to 60-80°C range initially)
- Still likely above reference due to other issues

After SHGC fix:
- 600FF max temp should increase slightly (more solar gain)
- 900FF unchanged (correct SHGC already)

## Implementation Plan

### Step 1: Fix CTF Warmup
```rust
// In ctf_solver_wrapper.rs:148, change:
self.solver = Some(CTFSolver::new(coeffs.clone(), config));
// To:
self.solver = Some(CTFSolver::with_warmup(
    coeffs.clone(),
    config,
    20.0,  // t_interior_initial
    20.0,  // t_exterior_initial
    7,     // warmup_days
));
```

### Step 2: Fix Window SHGC
Add to WindowSpec:
```rust
pub fn single_clear_glass() -> Self {
    WindowSpec::new(5.8, 0.86, 0.90, GlassType::SingleClear)
}
```

Update case_600ff() at line 2263:
```rust
.with_window_properties(WindowSpec::single_clear_glass())
```

### Step 3: Validate
Run blind validation and compare temperatures before/after each fix.