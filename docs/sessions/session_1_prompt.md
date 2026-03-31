# Physics-Based Refactoring - First Session Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Current Problem
The codebase currently has a ~3.1% pass rate on ASHRAE 140 tests. This is because the physics model relies on empirical correction factors (COP divisors, efficiency adjustments) in the validation layer rather than first-principles thermodynamics.

## Goal
Eradicate all empirical correction factors and replace them with robust physics-based solutions to achieve ≥90% pass rate on ASHRAE 140.

---

## Session 1 Task: Audit and Document Current Empirical Hacks

### Objective
Locate and document all empirical COP and efficiency corrections in the validation layer (`ashrae_140_validator.rs`). This is the first step in the phased refactoring plan.

### Background
In standard BEM terminology (EnergyPlus), an "Ideal Loads Air System" calculates the sensible and latent thermal energy required to meet a zone setpoint—it assumes 100% efficiency and infinite capacity. Converting that thermal load to electrical power via COP is the job of an Equipment/Plant Model.

The current code incorrectly applies empirical corrections directly in the validation output, masking underlying physics issues rather than fixing them.

### Steps
1. Search `src/validation/ashrae_140_validator.rs` for:
   - Hardcoded COP values (e.g., 3.0, 2.0, 2.2)
   - Efficiency divisors (e.g., 0.9, 0.95)
   - Session-specific corrections (grep for "SESSION")
   - Case-specific adjustment factors

2. Document each location with:
   - File path and line number
   - Current correction value
   - Which ASHRAE 140 cases it affects
   - Apparent purpose (based on code context)

3. Create a tracking table summarizing all empirical factors to be removed

### Expected Findings (based on initial analysis)
- Line ~981-997: Session 91/95 corrections for Case 900
- Line ~1057-1066: Case 960 cooling COP adjustment (2.0 → 2.2 in Session 70)
- Line ~2087-2094: General COP/efficiency divisors (3.0 for cooling, 0.9 for heating)
- Line ~2132: Phase 8 case-specific COP corrections

### Deliverable
Create `docs/empirical_hacks_audit.md` containing:
1. Summary table of all empirical corrections found
2. Each entry with: location, value, affected cases, apparent rationale
3. Total count of correction factors to be removed

### Success Criteria
- [ ] All COP/efficiency corrections in validator identified
- [ ] Each correction documented with file path and line number
- [ ] Clear mapping to which ASHRAE 140 cases are affected
- [ ] Document created at `docs/empirical_hacks_audit.md`

### Important Notes
- Focus ONLY on the validation layer corrections (not physics engine corrections)
- Do NOT modify any code yet—just document
- Note any "SESSION" markers as these indicate previous fix attempts
- If there are corrections in other files, note them but focus on validator first
