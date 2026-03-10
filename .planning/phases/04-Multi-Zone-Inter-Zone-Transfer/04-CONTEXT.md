# Phase 4: Multi-Zone Inter-Zone Transfer - Context

**Gathered:** 2026-03-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Verify and correct inter-zone heat transfer calculations for multi-zone Case 960 (sunspace + conditioned zone) using 5R1C thermal network structure.

**Scope:**
- Inter-zone conductance (h_tr_iz) calculations based on ASHRAE 140 Case 960 specs
- Radiative heat transfer between zones with full nonlinear Stefan-Boltzmann model
- Temperature-dependent air exchange through door opening (stack effect)
- Validation against ASHRAE 140 reference values (not calibrated 5R1C ranges)

**Out of scope:**
- New building types beyond Case 960
- Changes to single-zone physics (h_tr_em, h_tr_ms, etc.)
- Solar radiation corrections (Phase 3 complete)
- Thermal mass dynamics (Phase 2 complete)

</domain>

<decisions>
## Implementation Decisions

### Inter-Zone Conductance Formula

**Source for h_tr_iz value:**
- Use ASHRAE 140 Case 960 common wall construction specifications (R-values)
- Calculate from first principles: h_tr_iz = A_common / R_common_wall
- Example: 200mm concrete wall → R = 0.14 m²K/W → h_tr_iz = 154 W/K
- Matches ISO 13790 5R1C methodology used for other conductances

**Directionality:**
- Implement directional conductance: separate values for back-zone→sunspace and sunspace→back-zone
- Accounts for asymmetric wall construction (insulation facing one side)
- Differentiates heat flow in each direction

**Storage format:**
- Store h_tr_iz as VectorField array: [h_iz_0_to_1, h_iz_1_to_0]
- Matches existing ThermalModel pattern (temperatures, loads, etc. are VectorField)
- Enables CTA operations on inter-zone coupling

**Validation:**
- Unit tests: validate conductance calculation against manual calculation from Case 960 specs
- Integration tests: full year simulation, compare zone temperature profiles to ASHRAE 140 reference
- Both approaches ensure correctness at different validation levels

---

### Radiative Heat Transfer Approach

**Radiative exchange model:**
- Use full nonlinear Stefan-Boltzmann radiation: Q_12 = σ·ε₁·ε₂·F₁₂·A₁·(T₁⁴ - T₂⁴)
- More accurate for large ΔT (sunspace can be 20°C+ different from back-zone)
- Improves accuracy compared to linearized approximation

**View factor calculation:**
- Implement Hottel's method for rectangular zones sharing a common wall
- More accurate than simplified analytical solution (area ratio only)
- Handles complex sunspace geometry appropriately

**Integration into 5R1C energy balance:**
- Keep radiative exchange separate with full nonlinear calculation
- Maintains distinction between conduction and radiation physics
- Q_rad term added to energy balance as explicit component

**Implementation approach:**
- Create general surface exchange function: calculate_surface_radiative_exchange()
- Reusable for any multi-zone case with interior surfaces
- Not Case 960-specific, improves code reusability

---

### Air Exchange Between Zones

**ACH determination model:**
- Temperature-dependent air exchange rate (more realistic for sunspace thermal behavior)
- Accounts for thermal buoyancy and stack effect
- Varies with zone temperature difference (ΔT), not constant

**Temperature-dependent ACH formula:**
- Implement stack effect formula: Q_vent = 0.025·A_door·√(ΔT/door_height)
- Accounts for thermal buoyancy-driven ventilation
- More realistic than constant ACH for sunspace dynamics

**Integration into 5R1C model:**
- Use air enthalpy method: Q_vent = ρ·Cp·ACH·V·(T₁ - T₂)
- Explicitly calculates ventilation heat transfer with full ACH formula
- ρ = air density (~1.2 kg/m³), Cp = specific heat (~1000 J/kgK)
- More thermodynamically rigorous than lumping into conductance

**Door geometry:**
- Add door_geometry field to ThermalModel with height, area parameters
- Configure during from_spec() initialization
- Separates geometry from thermal parameters

---

### Validation Reference Approach

**Reference values source:**
- Target ASHRAE 140 reference values (not calibrated 5R1C ranges)
- More rigorous validation with standard tolerances
- Standard tolerances: ±15% annual energy, ±10% monthly energy, ±10% peak loads

**Data source:**
- Search online ASHRAE 140 resources for Case 960 benchmark data
- EnergyPlus results, ESP-r results, TRNSYS results with reference ranges
- Research task to collect authoritative reference values

**Reference programs:**
- Planner/research to determine appropriate reference program(s)
- Consider EnergyPlus, ESP-r, TRNSYS comparison
- Multi-reference comparison if multiple programs available

**Integration with test framework:**
- Add Case 960 to ASHRAE140Validator benchmark data
- Use standard ASHRAE 140 tolerances (consistent with other cases)
- Validates with same framework as baseline cases (600-650, 900)

### Claude's Discretion

- Specific ASHRAE 140 reference program selection (EnergyPlus vs ESP-r vs TRNSYS vs multi-reference)
- Exact Hottel's method implementation details (numerical integration or lookup tables)
- Stack effect coefficient tuning (0.025 factor may need calibration)
- View factor calculation optimization for performance

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets

**Thermal model infrastructure:**
- `ThermalModel` with `num_zones` parameter (currently supports 2 zones for Case 960)
- VectorField operations for all state variables (temperatures, loads, conductances)
- `from_spec()` method initializes from CaseSpec configuration

**Validation framework:**
- `ASHRAE140Validator` — core validation logic with tolerance-based comparison
- `BenchmarkData` — reference values from multiple programs (EnergyPlus, ESP-r, TRNSYS)
- Comparison engine — Pass/Warning/Fail status with ±5% tolerance band

**Case specification:**
- `CaseSpec` with `common_walls` field for inter-zone connections
- `geometry` array supporting multiple zones
- `hvac` array with free-floating support (sunspace is free-floating)

**Diagnostic infrastructure:**
- `DiagnosticCollector` — event-driven data collection
- `HourlyData` — hourly temperature tracking for zone-by-zone analysis
- `EnergyBreakdown` — component-level energy (useful for inter-zone heat flow analysis)

### Established Patterns

**Thermal network structure:**
- 5R1C model with conductances stored as VectorField
- CTA operations (element-wise +, *, /) for thermal network solving
- Inter-zone conductance concept exists but not fully validated

**Builder pattern:**
- `CaseBuilder` for flexible multi-zone construction
- Supports common walls between zones
- Zone-specific HVAC configuration (conditioned vs free-floating)

**Temperature tracking:**
- Zone temperatures stored in `temperatures` VectorField (element 0 = zone 0, etc.)
- Enables inter-zone temperature difference calculation: ΔT = T₁ - T₂
- Used for ACH and radiative exchange calculations

### Integration Points

**Where new code connects:**
- `src/sim/interzone.rs` — Add directional h_tr_iz calculation and Hottel's view factor method
- `src/sim/engine.rs` — Update `step_physics()` to include inter-zone heat transfer (Q_iz)
- `src/validation/ashrae_140_cases.rs` — Add Case 960 benchmark data to BenchmarkData
- `src/validation/ashrae_140_validator.rs` — Integrate Case 960 validation with standard tolerances
- `tests/ashrae_140_validation.rs` — Add Case 960 to comprehensive validation suite

**ThermalModel extensions:**
- Add `h_tr_iz: VectorField` field (directional conductance)
- Add `door_geometry: DoorGeometry` field (height, area)
- Add `surface_emissivity: VectorField` field (for radiative exchange)

</code_context>

<specifics>
## Specific Ideas

### ASHRAE 140 Case 960 Reference Tolerances

After obtaining online ASHRAE 140 data, use these validation tolerances:

**Annual Energy:** ±15% tolerance
- Heating: target reference range from ASHRAE 140
- Cooling: target reference range from ASHRAE 140

**Monthly Energy:** ±10% tolerance
- Track monthly heating/cooling for both zones
- Validates seasonal accuracy, not just annual

**Peak Loads:** ±10% tolerance
- Peak heating: target reference range from ASHRAE 140
- Peak cooling: target reference range from ASHRAE 140

**Zone Temperature Gradients:**
- Sunspace should be between outdoor and back-zone temperatures
- Typical ΔT zones ≈ 2-5°C for sunspace buildings
- Extreme ΔT should be reasonable (< 50°C max, > -30°C min)

### Inter-Zone Heat Transfer Calculation Pattern

**In `src/sim/engine.rs::step_physics()`:**
```rust
// Calculate inter-zone conductance (directional)
let h_tr_iz_0_to_1 = calculate_conductive_conductance(&spec.common_walls[0]);
let h_tr_iz_1_to_0 = calculate_conductive_conductance(&spec.common_walls[0]);

// Calculate air exchange rate (temperature-dependent)
let delta_t = self.temperatures[1] - self.temperatures[0];  // T_sunspace - T_back
let ach_iz = calculate_stack_effect_ach(delta_t, self.door_geometry.height, self.door_geometry.area);

// Air enthalpy heat transfer
let q_vent_air = RHO_AIR * CP_AIR * ach_iz * zone_volume * delta_t;

// Radiative heat transfer (full nonlinear)
let delta_t4 = self.temperatures[1].powi(4) - self.temperatures[0].powi(4);
let view_factor = hottels_view_factor(&spec.common_walls[0], &spec.geometry[0], &spec.geometry[1]);
let q_rad = STEFAN_BOLTZMANN * self.surface_emissivity[0] * self.surface_emissivity[1]
    * view_factor * spec.common_walls[0].area * delta_t4;

// Conductive heat transfer
let q_cond = h_tr_iz_0_to_1 * delta_t;

// Total inter-zone heat transfer
let q_iz_total = q_vent_air + q_rad + q_cond;

// Apply to energy balance
self.loads[0] -= q_iz_total;  // Zone 0 loses heat
self.loads[1] += q_iz_total;  // Zone 1 gains heat
```

### Stack Effect ACH Formula

**Stack effect buoyancy-driven ventilation:**
```
Q_vent = 0.025 * A_door * sqrt(ΔT / door_height)
ACH_iz = Q_vent / V_zone
```

Where:
- A_door = door opening area (m²)
- ΔT = |T₁ - T₂| temperature difference (K)
- door_height = door opening height (m)
- V_zone = zone volume (m³)
- 0.025 = empirical coefficient (may need calibration)

### Hottel's Method Implementation

**View factor for rectangular zones sharing common wall:**
- Requires numerical integration or analytical solution
- More accurate than simplified area ratio: F_12 = (A_common/A₁)×(A_common/A₂)
- Accounts for actual geometry and angular dependencies

**Implementation notes:**
- May require lookup tables or precomputed values for performance
- Consider caching view factors for common geometries
- Validate against simplified analytical solution for sanity check

### Expected Validation Outcomes

After Phase 4 implementation:

**MULTI-01 (Inter-zone heat transfer validation):**
- Case 960 passes with ±15% annual energy tolerance
- Zone temperature gradients match ASHRAE 140 reference within ±5°C
- Inter-zone conductance calculated from Case 960 construction specs
- Radiative heat transfer modeled with full Stefan-Boltzmann

**Zone-specific energy:**
- Back-zone (conditioned): heating/cooling energy within ASHRAE 140 reference
- Sunspace (free-floating): no HVAC energy, temperature profile matches reference
- Inter-zone heat flow: realistic exchange rates between zones

**Thermal behavior:**
- Sunspace temperature swing larger than back-zone (free-floating effect)
- Sunspace temperature between outdoor and back-zone temperatures
- Winter: sunspace colder than back-zone
- Summer: sunspace may be warmer due to solar gains

</specifics>

<deferred>
## Deferred Ideas

None identified — Phase 4 scope focused on inter-zone heat transfer for Case 960 as discussed. No new capabilities suggested.
</deferred>

---

*Phase: 04-Multi-Zone-Inter-Zone-Transfer*
*Context gathered: 2026-03-09*
