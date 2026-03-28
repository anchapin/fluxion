# Session 52: Sub-Hourly Timesteps for Peak Loads

**Date**: 2026-03-27
**Follows**: Session 51 (FD Integration for 600-Series)
**Status**: 📋 PLANNED
**Priority**: 🟡 HIGH - Fix systematic peak load discrepancies
**Estimated Duration**: 1 week
**Prerequisite**: Session 51 successful (FD working for 600-series, CTF working for 900-series)

## Objective

Implement sub-hourly timesteps (15-minute resolution) to fix systematic peak load discrepancies across all ASHRAE 140 cases. Peak loads are currently 8-33% below reference for heating and have mixed results for cooling due to timestep averaging.

## Context

### Current Peak Load Issues

**900-Series Peak Loads** (CTF from Session 49):
| Case | Peak Heating | Ref Min | % Below | Peak Cooling | Ref Range | Status |
|------|--------------|---------|---------|--------------|-----------|--------|
| 900 | ? | 1.80 kW | ~30% | ? | 1.60-2.10 kW | Mixed |
| 910 | ? | 1.90 kW | ~33% | ? | 1.20-1.60 kW | Mixed |
| 920 | ? | 2.10 kW | ~8% | ? | 1.40-1.90 kW | Mixed |
| 930 | ? | 2.30 kW | ~12% | ? | 1.10-1.50 kW | Mixed |
| 940 | ? | 1.90 kW | ~14% | ? | 1.70-2.30 kW | Mixed |
| 950 | 0.00 | 0.00 | N/A | ? | 0.70-0.90 kW | Mixed |

**600-Series Peak Loads** (FD from Session 51):
| Case | Peak Heating | Ref Range | Error | Peak Cooling | Ref Range | Error |
|------|--------------|-----------|-------|--------------|-----------|-------|
| 600 | ? | 2.80-3.80 kW | ? | ? | 4.80-6.20 kW | ? |
| 610 | ? | 4.30-5.70 kW | ? | ? | 2.20-2.90 kW | ? |
| 620 | ? | 2.80-3.80 kW | ? | ? | 2.50-3.50 kW | ? |
| 630 | ? | 4.70-6.10 kW | ? | ? | 1.80-2.40 kW | ? |
| 640 | ? | 4.30-5.70 kW | ? | ? | 2.80-3.70 kW | ? |
| 650 | 0.00 | 0.00-0.00 | ✅ | ? | 1.90-2.50 kW | ? |

### Why Peak Loads Are Wrong

**Session 47 Finding**: Peak loads are averaged due to hourly timesteps

1. **Hourly Timestep**: 3600 seconds
2. **Peak Calculation**: Maximum of hourly demands
3. **Problem**: True peak occurs within an hour and is averaged
4. **Impact**: Peak heating 8-33% below reference, peak cooling mixed

**Example**:
- True peak: 3.0 kW at 2:15 PM
- Hourly average: 2.2 kW (2:00-3:00 PM average)
- **Result**: Peak underpredicted by 27%

### Why Sub-Hourly Timesteps

**Reference Tools**:
- **EnergyPlus**: 10-15 minute timesteps
- **ESP-r**: Sub-hourly timesteps
- **TRNSYS**: Variable timesteps

**Benefits**:
- ✅ Capture true peak magnitude
- ✅ Capture true peak timing
- ✅ Better match reference tools
- ✅ Minimal impact on annual energies (should be identical)

## Implementation Plan

### Phase 1: Timestep Refactoring (Days 1-2)

**Step 1: Understand Current Timestep Handling**

Read `src/sim/engine.rs` solve_timesteps method:
- [ ] How is timestep currently specified?
- [ ] Where is timestep used in calculations?
- [ ] Can timestep be easily changed?
- [ ] Are there hard-coded hourly assumptions?

**Step 2: Implement Variable Timestep Support**

Modify `solve_timesteps` to accept timestep parameter:
```rust
// In src/sim/engine.rs
pub fn solve_timesteps_with_dt(
    &mut self,
    steps: usize,
    surrogates: &SurrogateManager,
    use_ai: bool,
    lighting: Option<&LightingSchedule>,
    equipment: Option<&[Box<dyn Equipment>]>,
    occupancy: Option<&OccupancyProfile>,
    dt_seconds: f64, // Timestep duration in seconds
) -> f64 {
    // Calculate number of sub-timesteps per hour
    let sub_steps_per_hour = (3600.0 / dt_seconds) as usize;
    let total_sub_steps = steps * sub_steps_per_hour;

    // ... implementation ...
}
```

**Step 3: Update Weather Data Access**

With sub-hourly timesteps, weather data access changes:
```rust
// Current: Hourly weather data
let weather = self.weather.get_hourly_data(hour);

// Sub-hourly: Interpolate or use same hour
let hour_index = step / sub_steps_per_hour;
let sub_step_index = step % sub_steps_per_hour;
let weather = self.weather.get_hourly_data(hour_index);
// Or interpolate if needed
```

**Step 4: Update Peak Tracking**

Peak tracking should work automatically with sub-hourly data:
```rust
// Peak tracking (lines 3709-3715)
// Should capture sub-hourly peaks automatically
// No changes needed if using max() over all timesteps
```

### Phase 2: Test 15-Minute Timestep (Days 2-3)

**Step 1: Implement 15-Minute Timestep**

Create test script or modify validation to use 15-minute dt:
```rust
// In validation code
let dt_seconds = 900.0; // 15 minutes
let num_steps = 8760 * 4; // 4x steps for 15-min resolution

let energy = model.solve_timesteps_with_dt(
    num_steps,
    &surrogates,
    false,
    None,
    None,
    None,
    dt_seconds,
);
```

**Step 2: Test on Case 900**

```bash
# Modify validation to use 15-minute timestep
# (Or create test script)

cargo run --release --bin fluxion validate --case 900 --dt 900
```

**Expected Results**:
- Peak heating: Should increase (closer to reference)
- Peak cooling: Should change (may increase or decrease)
- Annual energies: Should be identical (±1%)

**Step 3: Verify Energy Conservation**

With smaller timesteps:
- [ ] Check total energy in vs total energy out
- [ ] Verify imbalance <0.1%
- [ ] Compare with hourly results

### Phase 3: Validate All Cases (Days 3-4)

**Step 1: Run All Cases with 15-Minute Timestep**

```bash
# Test all cases with sub-hourly timestep
for case in 600 610 620 630 640 650 900 910 920 930 940 950; do
    echo "=== Case $case (15-min timestep) ==="
    cargo run --release --bin fluxion validate --case $case --dt 900
    echo ""
done
```

**Step 2: Compare Peak Loads**

Create comparison table (hourly vs 15-min):

**Case 900 Peak Heating**:
| Timestep | Peak Heating | Ref Min | Error | Improvement |
|----------|--------------|---------|-------|-------------|
| Hourly | 1.26 kW | 1.80 kW | 30% below | Baseline |
| 15-min | ? | 1.80 kW | ? | ? |

**Repeat for all cases and metrics**

**Step 3: Analyze Improvements**

For peak heating:
- [ ] How many cases improved?
- [ ] Average reduction in error?
- [ ] Worst case improvement?

For peak cooling:
- [ ] How many cases improved?
- [ ] Average reduction in error?
- [ ] Worst case improvement?

**Step 4: Verify Annual Energies Unchanged**

```bash
# Compare annual energies (hourly vs 15-min)
# Should be identical (±1%)

# If different, investigate:
# - Energy conservation violation
# - Timestep integration issues
# - Numerical instability
```

### Phase 4: Optimize Timestep (Days 4-5)

**If 15-Minute Timestep Works Well**:

- [ ] Document success
- [ ] Consider 15-minute as new standard
- [ ] Proceed to Session 53

**If 15-Minute Shows Improvement But Not Enough**:

Try 10-minute timestep (600s):
- [ ] Test if 10-minute provides better peak capture
- [ ] Compare with 15-minute results
- [ ] Evaluate tradeoff (accuracy vs performance)

**If 15-Minute Causes Issues**:

Investigate problems:
- [ ] Energy conservation violated?
- [ ] Numerical instability?
- [ ] Performance too slow?
- [ ] Integration errors?

### Phase 5: Performance Analysis (Day 5)

**Step 1: Profile Performance**

```bash
# Profile Case 900 with different timesteps
time cargo run --release --bin fluxion validate --case 900 --dt 3600  # Hourly
time cargo run --release --bin fluxion validate --case 900 --dt 900   # 15-min
time cargo run --release --bin fluxion validate --case 900 --dt 600   # 10-min
```

**Expected Performance**:
- Hourly: ~5 seconds
- 15-min: ~20 seconds (4x slower)
- 10-min: ~30 seconds (6x slower)

**Step 2: Optimization If Needed**

If performance is poor:
- [ ] Optimize solver loops
- [ ] Cache weather data lookups
- [ ] Reduce redundant calculations
- [ ] Consider adaptive timestepping

### Phase 6: Documentation (Days 6-7)

**Deliverables**:

1. **Timestep Analysis Report** (`SESSION_52_TIMESTEP.md`):
   - Hourly vs 15-min comparison
   - Peak load improvements
   - Annual energy consistency
   - Performance metrics
   - Recommendation for standard timestep

2. **Session Summary** (`SESSION_52_SUMMARY.md`):
   - What was accomplished
   - Optimal timestep identified
   - Peak load improvements achieved
   - Next steps for Session 53

3. **Update Validation**:
   - Update `ASHRAE140_ROADMAP.md` with progress
   - Document recommended timestep for future runs

## Success Criteria

- [ ] Sub-hourly timestep implemented and working
- [ ] Peak heating within 15% of reference (improved from 8-33% below)
- [ ] Peak cooling within 15% of reference (improved from ±14-249%)
- [ ] Annual energies unchanged (±1% from hourly)
- [ ] Energy conservation maintained (<0.1% imbalance)
- [ ] Performance acceptable (<30s per case)
- [ ] Optimal timestep identified

## Go/No-Go Decision for Session 53

**Go (Proceed to Session 53) if**:
- ✅ Sub-hourly timestep working
- ✅ Peak loads significantly improved
- ✅ Annual energies unchanged
- ✅ Performance acceptable

**Conditional Go (Proceed with Caveats) if**:
- ⚠️ Peak loads improved but not enough
- ⚠️ Performance slower than expected
- ⚠️ Need further optimization
- → Proceed to Session 53, continue optimization in parallel

**No-Go (Debug Further) if**:
- ❌ Sub-hourly timestep causing issues
- ❌ Annual energies changed significantly
- ❌ Energy conservation violated
- ❌ Performance unacceptable (>60s per case)

## Expected Outcomes

### Best Case: 15-Minute Timestep Highly Successful

- Peak heating within 10% of reference
- Peak cooling within 10% of reference
- Annual energies identical to hourly
- Performance acceptable (~20s per case)
- **Recommendation**: Use 15-minute as standard for validation

### Medium Case: 15-Minute Partially Successful

- Peak heating improved but still 15-20% below reference
- Peak cooling improved but still some cases outside range
- Annual energies mostly unchanged
- Performance acceptable
- **Recommendation**: Use 15-minute, consider 10-minute for critical cases

### Worst Case: Sub-Hourly Not Helping

- Peak loads not significantly improved
- Annual energies changed
- Performance poor
- **Recommendation**: Accept peak load limitations, focus on annual energies

## Technical Considerations

### Timestep vs Accuracy Tradeoff

| Timestep | Accuracy | Performance | Use Case |
|----------|----------|-------------|----------|
| 3600s (1 hour) | Low (peak loads) | Fast (5s) | Annual energy only |
| 900s (15 min) | Medium | Medium (20s) | Standard validation |
| 600s (10 min) | High | Slow (30s) | Peak load critical |
| 300s (5 min) | Very High | Very Slow (60s) | Research only |

### Energy Conservation

With smaller timesteps:
- More steps = more rounding errors
- Need careful numerical implementation
- Verify: Σ(Energy in) = Σ(Energy out) ± 0.1%

### Weather Data Interpolation

**Option A**: Use same hour for all sub-timesteps
- Simple, fast
- May introduce small errors

**Option B**: Interpolate between hours
- More accurate
- More complex
- May not be worth the effort

**Recommendation**: Start with Option A, switch to B if needed

## Files to Modify

1. **`src/sim/engine.rs`**:
   - Lines 2974-2987: Add `solve_timesteps_with_dt()` method
   - Lines 3200-3300: Update solve loop for sub-hourly
   - Lines 3709-3715: Verify peak tracking works

2. **`src/validation/ashrae_140_validator.rs`**:
   - Update to use sub-hourly timestep
   - Or add command-line flag

3. **`src/bin/fluxion.rs`**:
   - Add `--dt` flag for timestep specification

## Commands to Run

```bash
# Test sub-hourly on single case
cargo run --release --bin fluxion validate --case 900 --dt 900

# Compare with hourly
cargo run --release --bin fluxion validate --case 900 --dt 3600

# Test all cases with 15-minute
for case in 600 610 620 630 640 650 900 910 920 930 940 950; do
    cargo run --release --bin fluxion validate --case $case --dt 900
done

# Profile performance
time cargo run --release --bin fluxion validate --case 900 --dt 900

# Check energy conservation
# (Add debug output if needed)
```

## Research Questions

1. **Optimal Timestep**: Is 15-minute sufficient or is 10-minute needed?
2. **Peak Loads**: How much do peak loads improve with sub-hourly?
3. **Annual Energies**: Are annual energies truly unchanged?
4. **Performance**: Is performance acceptable for validation?
5. **Solver Compatibility**: Do CTF/FD work with sub-hourly timesteps?

## Dependencies

- **Session 51**: FD working for 600-series
- **Session 49**: CTF working for 900-series
- **Peak Load Issue**: Identified in Session 47

## Next Session

**Session 53**: Multi-Method Solver Manager
- Prerequisite: Session 52 successful (sub-hourly working)
- Scope: Implement auto-selection of 5R1C/CTF/FD
- Goal: Automatic solver selection based on building characteristics

## Alternatives if Sub-Hourly Fails

**Option A**: Accept Peak Load Limitations
- Document as 5R1C/CTF/FD characteristic
- Focus on annual energies (primary metric)
- Add disclaimer to validation report

**Option B**: Peak Load Correction Factor
- Apply empirical correction to peak loads
- Calculate factor from reference comparison
- Use for reporting only (not physics)

**Option C**: Hybrid Approach
- Use sub-hourly for peak load calculation only
- Use hourly for annual energy calculation
- More complex implementation

## References

- **`SESSION_47_SUMMARY.md`**: Peak load investigation
- **`SESSION_49_SUMMARY.md`**: CTF integration results
- **`SESSION_51_SUMMARY.md`**: FD integration results
- **`docs/ASHRAE140_ROADMAP.md`**: Phase 3 (Sub-Hourly Timesteps)
- **ASHRAE 140 Standard**: Peak load calculation methodology
- **EnergyPlus Documentation**: Timestep selection guidelines

---

**Session 52 Goal**: Implement sub-hourly timesteps (15-minute resolution) to fix systematic peak load discrepancies, achieving peak loads within 15% of reference while maintaining annual energy accuracy.
