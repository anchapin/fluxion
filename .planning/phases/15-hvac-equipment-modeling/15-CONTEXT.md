# Phase 15: HVAC Equipment Modeling - Context

**Gathered:** 2026-03-13
**Status:** Ready for planning

---

<domain>
## Phase Boundary

Implement realistic HVAC equipment models with efficiency curves and control strategies.

**What this delivers:**
- VAV and CAV system models responding correctly to load variations and setpoint changes
- Heat pump, chiller, and boiler equipment models with realistic efficiency curves and part-load degradation
- Economizer mode enabling free cooling when outdoor conditions are favorable
- Equipment cycling losses accurately modeled based on equipment runtime and load ratios

This phase enhances HVAC modeling accuracy — leverages existing hvac.rs infrastructure and Phase 14 thermal mass work.

</domain>

---

<decisions>
## Implementation Decisions

### Efficiency Curve Approach

**Curve type:** Polynomial curves
- Use polynomial functions for equipment efficiency curves (instead of lookup tables or simple linear degradation)
- Flexible and maintainable; can fit any curve shape
- More computation than simple linear but future-proof for new equipment types

**Curve inputs:** PLR + temperature (2D polynomial/surface)
- Part-load ratio (0-1) is the primary efficiency driver
- Outdoor temperature affects COP significantly (especially for heat pumps)
- Need separate curves for heating vs cooling modes
- 2D polynomial or surface interpolation approach

**Polynomial degree:** Cubic (degree 3)
- Cubic polynomials can capture typical S-shaped efficiency degradation patterns
- Standard for AHRI reference conditions
- Sufficient accuracy without excessive coefficients

**Coefficient source:** AHRI reference data
- Use AHRI Standard 550/590 for chiller and heat pump efficiency curves
- AHRI provides comprehensive reference data (more extensive than ASHRAE 140)
- Validate against ASHRAE Cases 800-810 as secondary check

### Equipment Depth

**Model detail:** Variable capacity (continuous modulation)
- Equipment models support continuous modulation from 0-100% capacity
- Represents variable speed drives, inverter compressors, modulating valves
- Most realistic approach, highest complexity
- Aligns with control strategy decision below

**Integration approach:** Unified trait
- Create `VariableCapacityEquipment` trait for all variable-capacity equipment types
- Common trait enables code reuse and consistent testing
- Implementations: VAV, CAV, HeatPump, Chiller, Boiler
- Enhance existing hvac.rs structures rather than complete rewrite

**Trait methods:** Capacity, efficiency, power, and PLR tracking
- Core methods: calculate_capacity(), calculate_efficiency(), calculate_power()
- Add part-load ratio (PLR) tracking to the trait
- Track runtime hours and startup/shutdown state
- Standardized interface for all equipment types

**Variable capacity limits:** AHRI + ASHRAE validation
- Use both AHRI reference data and ASHRAE 140 Cases 800-810
- AHRI provides manufacturer reference for min/max PLR limits
- ASHRAE Cases 800-810 validate specific equipment configurations
- Comprehensive testing approach ensures realistic bounds

### Control Strategies

**Control type:** Variable capacity modulation
- HVAC control continuously modulates capacity based on conditions
- Seamless integration with variable-capacity equipment models
- More refined than simple dual setpoint or staged control

**Control logic:** Predictive with thermal inertia
- Control signal considers thermal inertia, not just current temperature
- Smooths response, prevents overshoot/oscillation
- More realistic than deadband hysteresis for high-thermal-mass buildings

**Predictive factors:** Temperature, rate of change, and thermal mass state
- Current zone temperature
- Rate of temperature change (dT/dt) derived from previous timesteps
- Thermal mass temperature from 5R1C network (leveraging Phase 14 work)
- Physically meaningful decision that uses existing thermal model structure

**Inertia tuning:** ASHRAE + Guideline 14 stability criteria
- Tune thermal inertia gain to match ASHRAE Cases 800-810
- Validate against ASHRAE Guideline 14 stability criteria
- Ensure control is stable without excessive oscillation
- Balance response speed and stability

### Cycling & Losses

**Cycling model:** Combined approach (startup penalty + minimum runtime)
- Both fixed energy penalty for equipment startup AND minimum runtime constraints
- Most realistic modeling of compressor and equipment behavior
- Captures both startup surge and short-cycling prevention

**Startup penalty calculation:** Combined penalty model
- Part-load ratio degradation: Efficiency penalty increases at low PLR (e.g., +20% at 30% PLR)
- Startup penalty: Separate energy added for each startup cycle
- Both penalties combine to give total cycling loss
- Standard approach in ASHRAE 140 and energy modeling

**Minimum runtime enforcement:** Combined tracking
- Per-timestep state tracking to detect startup events and enforce minimum runtime
- Cumulative hours tracking for annual energy validation
- Most detailed validation capability

**Penalty values and limits:** AHRI reference data
- Use AHRI Standard data for startup energy penalties and minimum runtime limits
- AHRI provides manufacturer-specific data for different equipment types
- More comprehensive than ASHRAE 140 alone
- Validate against ASHRAE Cases 800-810 for consistency

### Claude's Discretion

- Exact polynomial coefficients (researcher will determine from AHRI data)
- Thermal inertia gain factor value (tune against ASHRAE + Guideline 14)
- Minimum runtime duration (5-15 minutes range, AHRI provides guidance)
- PLR degradation curve shape (researcher fits to AHRI data)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets

**HVAC module (src/sim/hvac.rs):**
- `HVACSystemType` enum: Simple, VAV, CAV, HeatPump, Ideal
- `VAVTerminal` struct with reheat coil calculation (max_airflow, min_airflow, reheat_capacity)
- `CAVSystem` struct with fan power and coil capacities
- `HeatPump` struct with COP degradation (simple linear: 2%/°C heating, 3%/°C cooling)
- Test suite validates basic VAV, CAV, and HeatPump behavior

**ThermalModel (src/sim/engine.rs):**
- 5R1C thermal network with thermal mass state (temperatures VectorField)
- `solve_timesteps()` loop calculates HVAC demand based on Ti_free and setpoints
- Already has basic setpoint-based control logic
- Thermal mass temperatures available from Phase 14 corrections

**Phase 14 thermal mass corrections:**
- Mode-specific coupling (h_tr_em_heating vs h_tr_em_cooling) implemented
- Thermal mass state is tracked in `mass_temperatures` VectorField
- Heating vs cooling mode detection already exists in solve_timesteps()

### Established Patterns

**Physics-first approach (Phase 1-4, Phase 14):**
- Address accuracy before optimization
- Validate against ASHRAE 140 reference ranges before feature completeness
- Apply same principle: validate equipment models against reference data

**Trait-based abstractions (ContinuousTensor, ContinuousField):**
- Codebase uses traits for common behavior across implementations
- Apply same pattern to `VariableCapacityEquipment` trait
- Supports code reuse and consistent testing

**Validation-driven development:**
- ASHRAE 140 suite is primary validation target
- Compare against reference ranges (±15% annual, ±10% monthly)
- Use before/after measurements to quantify improvement

**BatchOracle pattern constraint:**
- Pre-commit hook enforces single-level parallelism
- Equipment model calculations should not introduce nested par_iter() calls
- Maintain >1,000 configs/sec throughput for population evaluation

### Integration Points

**Where new trait lives:**
- `src/sim/hvac.rs` — Add `VariableCapacityEquipment` trait
- Existing VAVTerminal, CAVSystem, HeatPump will implement it
- Chiller and Boiler will be new structs implementing the trait

**Where chiller/boiler models live:**
- New structs in `src/sim/hvac.rs` following existing patterns
- Use `serde` for serialization (like VAVTerminal, CAVSystem)
- Follow same naming conventions and doc comment style

**Where efficiency curves live:**
- Could be inline in equipment structs or separate module
- Consider `src/sim/hvac/efficiency_curves.rs` if curve logic is complex
- Polynomial evaluation methods: `efficiency_at_plr_temp(plr, temp)`

**Where predictive control logic lives:**
- `src/sim/engine.rs` — ThermalModel::solve_timesteps() inner loop
- After Ti_free calculation, compute control signal with thermal inertia
- Modify existing HVAC demand calculation to use variable capacity instead of simple setpoint check

**Where cycling loss tracking lives:**
- Add state tracking to ThermalModel or equipment structs
- Track equipment state (on/off), runtime hours, startup count per timestep
- Apply penalties in energy calculation within solve_timesteps()

**Where economizer mode lives:**
- `src/sim/hvac.rs` — New method in HVACSystem or control logic
- Check outdoor temperature and enthalpy vs zone conditions
- Reduce or disable mechanical cooling when free cooling is available
- Requires psychrometrics (Phase 16) for enthalpy calculations

</code_context>

---

<specifics>
## Specific Ideas

**Polynomial curve coefficients:**
- AHRI Standard 550/590 provides reference data for chillers
- AHRI Standard 210/240 provides reference data for heat pumps
- Curve fitting will determine cubic coefficients (a, b, c, d) for: `COP(PLR, T) = a·PLR³ + b·PLR² + c·PLR + d`
- Separate curves for heating and cooling modes

**Variable capacity modulation:**
- Control signal range: 0-100% of rated capacity
- Modulation factor determined by temperature error + thermal inertia factor
- Equipment actual capacity = rated_capacity × modulation_factor

**Thermal inertia prediction:**
- Use Ti_free (free-floating temperature) and Tm (thermal mass temperature) from 5R1C
- Inertia factor = α·(Ti_free - Tm) + β·(dT_free/dt)
- Tune α and β against ASHRAE 800-810 for stability
- Control signal = f(zone_temp, setpoint, inertia_factor)

**Startup penalty calculation:**
- Startup energy = equipment_type.penalty_kwh × startup_count
- Apply at each detected startup transition (off → on)
- Add to cumulative energy consumption for the timestep

**Minimum runtime enforcement:**
- After startup, equipment must run for minimum_timesteps (e.g., 5 minutes = 5 timesteps)
- Maintain "must_run" flag until runtime threshold satisfied
- Prevents control signal from cycling equipment on/off rapidly

**PLR degradation:**
- Efficiency multiplier = 1.0 + degradation_factor × (1.0 - PLR)
- Example: At PLR=0.3, degradation=0.2 → multiplier = 1.0 + 0.2 × 0.7 = 1.14
- Actual COP = rated_COP × efficiency_multiplier

**Economizer mode:**
- Enable when outdoor dry bulb < zone setpoint AND outdoor enthalpy < zone enthalpy
- Requires Phase 16 psychrometrics for enthalpy calculations
- Reduce mechanical cooling capacity (set modulation_factor lower) when economizer is active
- Free cooling provided by increased ventilation air flow

**ASHRAE 140 Cases 800-810:**
- These cases specifically test HVAC equipment performance
- Provide reference values for energy consumption at different operating conditions
- Should be primary validation target for Phase 15

**Test additions:**
- test_variable_capacity_modulation: Verify continuous 0-100% capacity response
- test_polynomial_efficiency_curves: Validate COP calculations at various PLR and temperature
- test_thermal_inertia_control: Check smooth control signal without oscillation
- test_cycling_losses: Verify startup penalties and minimum runtime constraints
- test_ahri_coefficient_fitting: Confirm curves match AHRI reference data
- test_ashrae_800_810_validation: Run Cases 800-810, compare to reference

</specifics>

---

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. All decisions relate to efficiency curves, equipment depth, control strategies, and cycling losses as defined in Phase 15 requirements.

</deferred>

---

*Phase: 15-hvac-equipment-modeling*
*Context gathered: 2026-03-13*
