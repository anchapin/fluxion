# Physics-Based Refactoring - Session 6 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 5 Recap
- Researched CTF (Conduction Transfer Functions) implementation requirements
- Found existing CTF modules: ctf_coefficients.rs, ctf_solver.rs, multi_node_ctf.rs, per_surface_ctf.rs
- Identified automatic CTF selection for HighMass (900-series) cases via `enable_advanced_solver()`
- Created `docs/ctf_requirements.md` with complete technical specification

---

## Session 6 Task: Verify CTF Integration and Validate Against ASHRAE 140

### Objective
Verify that CTF is actually being used for high-mass (900-series) cases and validate results against ASHRAE 140 reference values.

### Background
Session 5 found that CTF infrastructure exists and automatic selection is in place. Now we need to verify it's working correctly and producing accurate results.

### Steps

#### Part A: Verify CTF Activation for 900-Series

1. **Run a 900-series test case** with verbose output:
```bash
cargo test --test ashrae_140_validation case_900 -- --nocapture 2>&1 | head -50
```

2. **Check for CTF solver messages** in the output - should see:
   - "Enabled CTF solver" or similar message
   - CTF coefficient calculation logs

3. **Examine the code path** in `enable_advanced_solver()`:
   - Confirm it's being called for high-mass cases
   - Verify `enable_ctf_with_fd_fallback()` is being invoked

#### Part B: Run ASHRAE 140 Validation

1. **Run the full test suite**:
```bash
cargo test --test ashrae_140_validation 2>&1 | tail -100
```

2. **Analyze results by case type**:
   - 600-series (low-mass): Should use 5R1C fallback
   - 900-series (high-mass): Should use CTF
   - Note which cases are passing/failing

3. **Identify specific issues**:
   - For failing 900-series cases, check if CTF is producing correct thermal mass behavior
   - Compare against ASHRAE 140 reference values in `src/validation/ashrae_140_cases.rs`

#### Part C: EnergyPlus Comparison (if time permits)

1. **Compare with EnergyPlus outputs** (available in `energyplus/` directory if present)
2. **Check for known discrepancies** between CTF and 5R1C behavior

#### Part D: Fix Any Integration Issues Found

If CTF is not being activated or is producing incorrect results:
1. Debug the `enable_advanced_solver()` function
2. Check `enable_ctf_with_fd_fallback()` implementation
3. Verify coefficient calculation is working

### Expected Architecture

```
Thermal Model
├── CTF Solver (for 900-series)
│   ├── Coefficients from material properties
│   └── Runtime heat flux calculation
└── 5R1C Fallback (for 600-series)
    └── Lumped capacitance model
```

### Deliverable
- Summary of CTF activation status for 900-series cases
- ASHRAE 140 validation results with pass/fail breakdown
- Any bugs found and fixed in CTF integration

### Success Criteria
- [ ] CTF confirmed active for 900-series (high-mass) cases
- [ ] ASHRAE 140 validation run completed
- [ ] Results analyzed by case type
- [ ] Any CTF integration bugs identified and fixed

### Important Notes
- Focus on verifying the existing CTF infrastructure works correctly
- Don't implement new CTF features - just validate current implementation
- If CTF is working, document any remaining discrepancies vs ASHRAE 140