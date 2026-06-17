# Physics-Based Refactoring - Session 4 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 3 Recap
- Created `src/sim/hvac/ideal_loads.rs` with:
  - `ZoneIdealLoads` struct (calculates what zone NEEDS at 100% efficiency)
  - `SimpleHVACEquipment` struct (converts thermal to electrical via COP/efficiency)
  - `IdealLoadsSystem` struct (combines both)
  - `HVACEnergyResult` struct (thermal_load_watts + electrical_kw)
- Added `IdealLoadsSystem` as a field in `ThermalModel`
- Integrated electrical energy calculation in `step_physics_5r1c()`
- Default ASHRAE 140 values: cooling COP=3.0, heating efficiency=0.9

---

## Session 4 Task: Remove Empirical Corrections from Validator

### Objective
Remove hardcoded COP/efficiency divisors from validation output processing now that IdealLoads properly calculates electrical consumption.

### Background
The validation layer currently has empirical correction factors that divide energy values to convert from thermal to electrical. With the new IdealLoadsSystem integration (Session 3), the ThermalModel now tracks:
- `annual_heating_energy` - thermal energy (what zone needs)
- `annual_cooling_energy` - thermal energy (what zone needs)
- `annual_electrical_energy` - electrical energy (what equipment uses with COP/efficiency)

We need to verify if the validation layer is still applying these empirical corrections and remove them if no longer needed.

### Steps

#### Part A: Audit Current Validation Layer Corrections

1. Search for empirical COP/efficiency divisors in the validator:
```bash
grep -n "3.0\|0.9\|cop\|efficiency" src/validation/ashrae_140_validator.rs | head -30
```

2. Document current locations with case numbers and line numbers

3. Understand what each correction is doing:
   - Is it dividing thermal energy to get electrical? (No longer needed - IdealLoads does this)
   - Is it a different type of correction? (May still be needed)

#### Part B: Remove Unnecessary Corrections

1. **For cases using IdealLoadsSystem (default cases without hvac_equipment)**:
   - The `annual_electrical_energy` field now contains proper electrical consumption
   - Remove any validation-layer division that converts thermal → electrical
   - Use `annual_electrical_energy` directly for comparison

2. **For cases still requiring validation-layer corrections**:
   - Document why the correction is still needed
   - Keep the correction but add TODO comment for future investigation

#### Part C: Update Validation to Use Electrical Energy

1. Modify validation to use `annual_electrical_energy` where appropriate:
```rust
// Instead of dividing annual_cooling_energy by COP
let electrical_cooling = model.annual_electrical_energy;

// Use directly if model tracks electrical separately
```

2. Run basic validation test to confirm changes work

### Expected Architecture After Fix

```
Validation Output (for comparison with ASHRAE reference)
├── Use model.annual_electrical_energy (from IdealLoads)
└── No longer need to divide by COP/efficiency in validator
```

### Deliverable
- Updated validator with removed empirical corrections
- Clear documentation of any corrections that remain (with rationale)
- Test verification

### Success Criteria
- [ ] Empirical COP/efficiency divisors removed where appropriate
- [ ] Code compiles without errors
- [ ] Unit tests still pass
- [ ] Any remaining corrections documented with rationale
- [ ] No regression in validation behavior

### Important Notes
- Some cases (like Case 960) may have different corrections that serve a different purpose - keep those
- The goal is to remove ONLY the "thermal to electrical" conversions that are now handled by IdealLoads
- Keep the thermal tracking (annual_heating_energy/cooling_energy) for diagnostic purposes
