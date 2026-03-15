# Phase 14: Thermal Network Verification - Context

**Gathered:** 2026-03-13
**Status:** Ready for planning

---

<domain>
## Phase Boundary

Fix high-mass annual energy error by addressing thermal network structure, mock data cleanup, and establishing unit conventions.

**What this delivers:**
- Remove all mock predictions from SurrogateManager and replace with analytical physics calculations (PHYS-01)
- Implement thermal mass corrections achieving coupling ratios > 0.1 for high-mass buildings (PHYS-04)
- Implement mode-specific thermal mass coupling (heating vs cooling: h_tr_em_heating, h_tr_em_cooling) (PHYS-05)
- Codebase audit documenting all placeholder/mock/hardcoded values with remediation plan (DATA-01)

This phase improves physics engine accuracy—no API changes to BatchOracle or Model.

</domain>

---

<decisions>
## Implementation Decisions

### Mock Data Replacement Strategy

**Approach:** Analytical physics calculations
- Replace SurrogateManager mock predictions (vec![1.2; ...]) with deterministic analytical physics
- Not training ONNX models for this phase (deferred to v2.0 if needed)

**Integration:** Delegate to physics engine
- ThermalModel::solve_timesteps() calculates loads directly when `use_ai=false`
- Remove SurrogateManager parameter from solve_timesteps() call when using analytical path
- Simpler call chain: ThermalModel handles its own physics without surrogate abstraction layer

**Validation:** Both approaches
- ASHRAE 140 comparison: Run all cases with analytical loads, compare against baseline mock predictions
- Energy balance test: Add unit test verifying total energy in = total energy out over 8760 timesteps
- Comprehensive validation ensures physics replacements don't introduce conservation errors

### Thermal Mass Coupling Approach

**Implementation:** Adjust h_tr_em directly
- Increase exterior-to-mass conductance (h_tr_em) to strengthen coupling
- Target coupling ratio > 0.1 (current high-mass buildings have ~0.05)
- Direct physics change affects all cases, needs careful validation

**Derivation:** ASHRAE 140 reference values
- Use standard construction properties from ASHRAE 140 reference documents
- Documented, traceable, consistent with standards
- Avoid case-specific calibration; use standard thermal mass values

**Targeting:** All cases > threshold
- Apply adjustment to any building case with thermal capacitance exceeding threshold
- More generalizable than targeting specific cases (900, 960)
- Automatically handles future high-mass cases

**Validation:** Both: full + subset
- Quick subset validation: Test only affected cases (high-mass + threshold boundary) for fast iteration
- Full suite validation: Run all ASHRAE 140 cases before committing changes
- Balanced approach for development efficiency + comprehensive coverage

### Mode-Specific Coupling Implementation

**Mode detection:** Ti_free to HVAC setpoint comparison
- If free-floating temperature (Ti_free) < HVAC setpoint → heating mode
- If Ti_free > HVAC setpoint → cooling mode
- Matches HVAC control logic, simple to implement, no state tracking needed

**Data structure:** Dynamic adjustment factor
- Single h_tr_em field with heating_factor and cooling_factor applied at runtime
- More memory-efficient than separate h_tr_em_heating/h_tr_em_cooling VectorFields
- Runtime calculation overhead is negligible for single multiplication per timestep

**Factor derivation:** ASHRAE 140 empirical values
- Use documented empirical values from ASHRAE 140 reference
- Traceable to standard thermal mass and construction properties
- Avoids case-specific calibration, maintains consistency

**Validation:** Compare before/after on Case 900
- Run Case 900 with and without mode-specific coupling
- Measure annual energy reduction for heating and cooling separately
- Focused validation on the primary high-mass case

### Codebase Audit Methodology

**Scope:** Full codebase grep
- Search entire src/ directory for patterns: TODO, FIXME, mock, placeholder, hardcoded
- Most thorough approach ensures nothing is missed
- May find noise items but comprehensive coverage is priority

**Report format:** JSON
- Generate audit_report.json with structured, machine-readable data
- Supports automated checks and CI integration
- Easy to parse for remediation tracking

**Categorization:** By priority/impact
- Critical: Blocks PHYS-01 (mock removal), high-mass coupling fixes
- Warning: Affects accuracy but not blocking validation
- Info: Cosmetic issues, documentation gaps, non-critical TODOs
- Clear remediation priority based on impact

**Remediation tracking:** GitHub issues
- Create GitHub issue for each critical finding
- Track in issue tracker, audit JSON references issue URLs
- Best for long-term tracking and cross-session continuity

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets

**SurrogateManager (src/ai/surrogate.rs):**
- Mock predictions in `new()` constructor: returns vec![1.2; ...] for thermal loads
- Multi-device configuration support (MultiDeviceConfig) for GPU inference
- SessionPool for concurrent ONNX inference (when models are loaded)
- QuantizationConfig for FP32/FP16/INT8 inference

**ThermalModel (src/sim/engine.rs):**
- 5R1C thermal network structure with h_tr_em, h_tr_ms, h_tr_is, h_tr_w, h_ve conductances
- solve_timesteps() method currently takes &SurrogateManager parameter
- apply_parameters() maps gene vector to model state (window U-value, HVAC setpoint)
- Already has 6R2C experimental support (configure_6r2c_model())

**Thermal mass validation (src/validation/thermal_mass.rs):**
- calculate_thermal_mass_correction() function: implements sqrt-based correction factor
- validate_thermal_mass(): compares Case 600 (low-mass) vs Case 900 (high-mass)
- validate_6r2c_thermal_mass(): validates 6R2C envelope/internal mass configuration
- generate_thermal_mass_report(): produces validation output

**Known limitations documentation:**
- docs/KNOWN_LIMITATIONS.md: Documents 5R1C fundamental limitation with high-mass annual energy error
- 6R2C_DECISION.md (if exists): Documents Phase 12 evaluation showing no accuracy improvement
- Phase 8 investigation results: Analyzed Case 960 issues, documented in validation

### Established Patterns

**Physics-first approach (Phase 1-4):**
- Address accuracy before optimization to avoid optimizing incorrect physics
- Core thermal network validated before performance tuning
- Apply same principle: fix coupling/mocks before optimizing

**5R1C as default (Phase 12):**
- 6R2C showed no accuracy improvement with 1.5-2x performance penalty
- Keep 5R1C as default thermal network structure
- Corrections should work within 5R1C constraints, not require new structure

**Validation-driven development:**
- ASHRAE 140 suite is primary validation target
- Compare against reference ranges (±15% annual, ±10% monthly)
- Use before/after measurements to quantify improvement

**BatchOracle pattern constraint:**
- Pre-commit hook enforces single-level parallelism (par_iter at population level only)
- Physics changes should not introduce nested par_iter() calls
- Maintain >1,000 configs/sec throughput for population evaluation

### Integration Points

**Where mock data removal happens:**
- src/ai/surrogate.rs — SurrogateManager::new() constructor
- src/sim/engine.rs — solve_timesteps() method calls SurrogateManager
- May need to adjust method signatures to remove SurrogateManager dependency when use_ai=false

**Where thermal mass corrections integrate:**
- src/sim/engine.rs — ThermalModel struct, h_tr_em VectorField
- apply_parameters() method: may need to apply adjustment based on thermal capacitance threshold
- solve_timesteps() loop: applies mode-specific factors at each timestep

**Where mode detection happens:**
- src/sim/engine.rs — solve_timesteps() inner loop, where Ti_free is calculated
- After Ti_free calculation, compare to hvac_setpoint to determine heating/cooling mode
- Apply corresponding factor to h_tr_em before calculating HVAC demand

**Where audit tool lives:**
- New tool: src/bin/audit_codebase.rs or add to existing fluxion CLI
- Output: audit_report.json in root directory (gitignored for local work, committed if final)
- May integrate with CI for automated checks in future

**Where audit report lives:**
- docs/AUDIT_REPORT.md (human-readable summary derived from JSON)
- Links to GitHub issues for critical findings
- References KNOWN_LIMITATIONS.md for related physics issues

</code_context>

---

<specifics>
## Specific Ideas

**Mock data removal:**
- Current surrogate.rs:line ~100+ shows `vec![1.2; num_zones]` mock prediction
- Replace with analytical solar gain calculation (irradiance * SHGC * area) if no ONNX model
- Or call ThermalModel internal methods for load calculation if already exists
- Ensure backward compatibility: users calling SurrogateManager::new() should still work

**Thermal mass threshold:**
- Case 600 (low-mass) has ~2.4e6 J/K thermal capacitance
- Case 900 (high-mass) has ~1.2e7 J/K thermal capacitance (5x difference)
- Threshold could be ~5e6 J/K (between low and high mass)
- ASHRAE 140 standard: High-mass buildings have >3x low-mass capacitance

**Coupling ratio formula:**
- Current: ratio = h_tr_em / h_tr_ms ≈ 0.05 for high-mass
- Target: ratio > 0.1
- Implementation: h_tr_em_new = max(h_tr_em_current, 0.1 * h_tr_ms) for high-mass cases

**Mode-specific factors:**
- Heating factor: Apply when Ti_free < hvac_setpoint (e.g., 1.2x stronger coupling for heating)
- Cooling factor: Apply when Ti_free > hvac_setpoint (e.g., 0.8x weaker coupling for cooling)
- Derive from ASHRAE 140: Different thermal mass behavior in heating vs cooling seasons

**Audit grep patterns:**
- `TODO|FIXME` — track deferred work
- `mock|placeholder` — track non-production values
- `hardcoded` — track magic numbers that should be configurable
- `vec!\[.*\]` (in load predictions) — track mock data initialization

**Audit JSON structure:**
```json
{
  "generated": "2026-03-13T18:00:00Z",
  "findings": [
    {
      "file": "src/ai/surrogate.rs",
      "line": 100,
      "pattern": "mock",
      "content": "vec![1.2; num_zones]",
      "priority": "critical",
      "requirement": "PHYS-01",
      "issue_url": "https://github.com/owner/repo/issues/XXX"
    }
  ]
}
```

**Validation test additions:**
- test_energy_conservation: Verify Σenergy_in = Σenergy_out over 8760 timesteps
- test_thermal_mass_coupling: Verify h_tr_em / h_tr_ms ratio > 0.1 for high-mass cases
- test_mode_specific_coupling: Verify different factors applied for heating vs cooling modes
- test_audit_completeness: Verify JSON report structure and critical findings count

</specifics>

---

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. All decisions relate to mock data removal, thermal mass coupling, mode-specific coupling, and codebase auditing as defined in Phase 14 requirements.

</deferred>

---

*Phase: 14-thermal-network-verification*
*Context gathered: 2026-03-13*
