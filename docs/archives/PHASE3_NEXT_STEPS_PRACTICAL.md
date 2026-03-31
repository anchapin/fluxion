# Phase 3 Practical Next Steps

**Date:** 2026-03-29
**Status:** Analysis Complete, Implementation Recommended

---

## Issue: Simplified Testing Approach

Both Phase 3 parametric τ studies attempted (for 600-series and 900-series) failed to produce meaningful results due to:

1. **Oversimplified Solar Model** - Using simplified `dni * floor_area * factor` produces massive solar gains
   - 600-series cooling: 2000+ MWh vs expected 7 MWh
   - 900-series cooling: 200-400 MWh vs expected 3 MWh
   - Root cause: DNI can be 800+ W/m², multiplied by floor area (64 m²) gives unrealistic gains

2. **Complex Solar Calculation Required** - Actual validation uses sophisticated solar model:
   - `calculate_hourly_solar()` with latitude, longitude, timezone
   - Window properties (area, orientation, shading, overhangs)
   - Diffuse vs direct radiation components
   - Multiple surfaces (walls, windows, roof, floor)
   - Replicating this in diagnostic tools is impractical

## Solution: Use Actual Validation Infrastructure

Instead of creating new diagnostic tools with simplified physics, modify the existing validation code to run parametric studies.

### Approach 1: Modify Validation to Test Different τ Values

**File:** `src/sim/engine.rs`

**Current Code (lines 708-715):**
```rust
let target_tau_hours = match mass_class {
    crate::sim::construction::MassClass::VeryLight => 6.0,
    crate::sim::construction::MassClass::Light => 7.0,
    crate::sim::construction::MassClass::Medium => 8.0,
    crate::sim::construction::MassClass::Heavy => 10.0,
    crate::sim::construction::MassClass::VeryHeavy => 12.0,
};
```

**Implementation Plan:**

1. Add environment variable or constant for τ override:
   ```rust
   // Allow runtime τ override for testing
   let target_tau_hours = if let Ok(tau_str) = std::env::var("FLUXION_TARGET_TAU") {
       tau_str.parse().expect("Invalid FLUXION_TARGET_TAU")
   } else {
       match mass_class {
           // ... existing values ...
       }
   };
   ```

2. Run validation multiple times with different τ values:
   ```bash
   # Test for 600-series (VeryLight)
   FLUXION_TARGET_TAU=4.0 cargo run --bin fluxion validate --case 600
   FLUXION_TARGET_TAU=5.0 cargo run --bin fluxion validate --case 600
   FLUXION_TARGET_TAU=6.0 cargo run --bin fluxion validate --case 600
   FLUXION_TARGET_TAU=7.0 cargo run --bin fluxion validate --case 600
   FLUXION_TARGET_TAU=8.0 cargo run --bin fluxion validate --case 600

   # Test for 900-series (VeryHeavy)
   FLUXION_TARGET_TAU=10.0 cargo run --bin fluxion validate --case 900
   FLUXION_TARGET_TAU=15.0 cargo run --bin fluxion validate --case 900
   FLUXION_TARGET_TAU=20.0 cargo run --bin fluxion validate --case 900
   FLUXION_TARGET_TAU=25.0 cargo run --bin fluxion validate --case 900
   FLUXION_TARGET_TAU=30.0 cargo run --bin fluxion validate --case 900
   ```

3. Collect results and identify optimal τ values

### Approach 2: Create Parametric Sweep Tool

**File:** Create new binary `src/bin/parametric_tau_sweep.rs`

**Functionality:**
1. Parse command-line arguments for case ID and τ values
2. Create modified ThermalModel with τ override
3. Run full validation simulation for each τ value
4. Output results in CSV format for analysis

**Example Usage:**
```bash
cargo run --release --bin parametric_tau_sweep --case 600 --tau 4,5,6,7,8
cargo run --release --bin parametric_tau_sweep --case 900 --tau 10,15,20,25,30
```

---

## Specific Recommendations for Next Work

### Recommendation 1: Implement τ Override for Testing

**File:** `src/sim/engine.rs`

**Changes:**

1. Add τ override environment variable support around line 708:

```rust
// Allow τ override for testing via environment variable
// e.g., FLUXION_TARGET_TAU_VERYLIGHT=8.0
let target_tau_hours = match mass_class {
    crate::sim::construction::MassClass::VeryLight => {
        std::env::var("FLUXION_TARGET_TAU_VERYLIGHT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(6.0)
    }
    crate::sim::construction::MassClass::Light => {
        std::env::var("FLUXION_TARGET_TAU_LIGHT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(7.0)
    }
    crate::sim::construction::MassClass::Medium => {
        std::env::var("FLUXION_TARGET_TAU_MEDIUM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8.0)
    }
    crate::sim::construction::MassClass::Heavy => {
        std::env::var("FLUXION_TARGET_TAU_HEAVY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10.0)
    }
    crate::sim::construction::MassClass::VeryHeavy => {
        std::env::var("FLUXION_TARGET_TAU_VERYHEAVY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(12.0)  // Current value
    }
};
```

2. Document the environment variables in README.md

### Recommendation 2: Run Comprehensive τ Sweep

After implementing τ override, run comprehensive sweeps:

**For 600-Series (VeryLight):**
```bash
# Create test script
for tau in 4.0 5.0 6.0 7.0 8.0; do
    echo "Testing tau=$tau"
    FLUXION_TARGET_TAU_VERYLIGHT=$tau cargo run --release --bin fluxion validate --case 600
done
```

**For 900-Series (VeryHeavy):**
```bash
for tau in 10.0 15.0 20.0 25.0 30.0; do
    echo "Testing tau=$tau"
    FLUXION_TARGET_TAU_VERYHEAVY=$tau cargo run --release --bin fluxion validate --case 900
done
```

### Recommendation 3: Verify Internal Gains

**Priority 1 from Summary - Investigate internal gains for 600-series**

**Action:**
1. Check ASHRAE 140 specification for Case 600 internal gains
2. Compare with values in `src/validation/ashrae_140_cases.rs`
3. If mismatched, update case specification
4. Re-run validation to check impact

**File to Check:** `src/validation/ashrae_140_cases.rs` around Case 600 definition

### Recommendation 4: Consider τ Scaling Formula

**Priority 4 from Summary - Consider τ scaling by mass class**

**Current Issue:**
- VeryLight C_m = 2,396 kJ/K, τ = 6.0h
- VeryHeavy C_m = 19,946 kJ/K, τ = 12.0h
- τ ratio: 12.0/6.0 = 2.0x
- C_m ratio: 19,946/2,396 = 8.3x

**Observation:** τ scales less than C_m, suggesting the formula doesn't account for actual thermal physics across mass classes.

**Alternative Formulas to Test:**

1. **Linear Scaling:** τ = k × C_m
   ```rust
   let tau_hours = (k * thermal_capacitance) / 3600.0;
   // With k tuned to give ~6h for VeryLight
   let k = 6.0 * 3600.0 / 2396.0;  // k ≈ 9.0
   // Then τ for VeryHeavy = 9.0 * 19946 / 3600 ≈ 50h
   ```

2. **Square Root Scaling:** τ = k × √C_m
   ```rust
   let tau_hours = k * (thermal_capacitance.sqrt() / 60.0);
   // With k tuned to give ~6h for VeryLight
   let k = 6.0 * 60.0 / (2396.0_f64.sqrt());
   // Then τ for VeryHeavy = k * (19946_f64.sqrt() / 60.0) ≈ 17h
   ```

3. **Power Law Scaling:** τ = k × C_m^p with 0 < p < 1
   ```rust
   let tau_hours = k * thermal_capacitance.powf(p) / 3600.0.powf(p);
   ```

**Testing Approach:**
Implement these as alternatives and test against validation to see which produces best results.

---

## Summary of Practical Next Steps

### Immediate Actions (Do First)

1. **Implement τ override environment variables** in `src/sim/engine.rs`
2. **Document the environment variables** in README.md
3. **Test τ=6.0h vs τ=8.0h for Case 600** to confirm approach works
4. **Run τ sweep for Case 900** to find optimal VeryHeavy τ

### Secondary Actions (Do After Immediate)

5. **Verify internal gain values** against ASHRAE 140 specification
6. **Consider τ scaling formulas** if fixed τ values don't work well
7. **Systematic parameter sensitivity analysis** for remaining parameters

### Documentation Actions

8. **Update CLAUDE.md** with new environment variable usage
9. **Create parametric testing guide** in docs/
10. **Document optimal τ values** once found

---

## Validation of Approach

The key insight from failed parametric studies is that:

✅ **Correct physics modeling is critical** - Simplified solar models produce meaningless results
✅ **Use existing infrastructure** - The validation code already has correct solar calculations
✅ **Environment variable approach** is practical - Allows testing without code rebuilds for each value
✅ **Systematic sweeping** will yield optimal values - Test ranges systematically, not just single values

---

## Expected Timeline

| Task | Effort | Expected Outcome |
|------|----------|------------------|
| Implement τ override | Low | Easy testing of different τ values |
| Run 600-series τ sweep | Medium | Find optimal VeryLight τ |
| Run 900-series τ sweep | Medium | Find optimal VeryHeavy τ |
| Verify internal gains | Low | Rule in/out or find correction needed |
| Test τ scaling formulas | Medium | Find formula that works across mass classes |
| Systematic parameter sweep | High | Complete parameter optimization |

---

**Phase 3 Practical Next Steps Documented.** Ready for implementation.
