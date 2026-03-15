# Phase 16: Psychrometrics Module - Context

**Gathered:** 2026-03-13
**Status:** Ready for planning

---

<domain>
## Phase Boundary

Implement psychrometric calculations for accurate HVAC equipment verification.

**What this delivers:**
- Dew point, humidity ratio, enthalpy, and wet-bulb temperature calculations
- Integration with existing weather data and HVAC equipment models
- Validation against ASHRAE Fundamentals reference values
- Enable economizer enthalpy mode (deferred from Phase 15)

This phase provides the psychrometric foundation needed for accurate HVAC control—no API changes to BatchOracle or Model, but enables Enthalpy economizer mode.

---

<decisions>
## Implementation Decisions

### Calculation Methodology

**Psychrometric approach:** ASHRAE empirical formulas
- Use ASHRAE Handbook of Fundamentals empirical formulas
- Comprehensive and reference-based, computationally heavier
- Best for accuracy and ASHRAE 140 compliance
- Prioritizes correctness over optimization (performance addressed in Phase 20)

**Dew point calculation:** ASHRAE exact formulation
- Exact ASHRAE formulation using saturation vapor pressure curve
- Most accurate: ±0.01°C tolerance
- Slower: 10-15 iterations per calculation
- Use saturation vapor pressure: p_sat = f(T) from ASHRAE Fundamentals
- Dew point: Td = solve for T where p_sat(Td) = p_water(T, RH)

**Wet-bulb temperature:** Iterative psychrometric equation solving
- Solve psychrometric equations iteratively for wet-bulb temperature
- Matches ASHRAE Fundamentals methodology
- Accurate within ±0.1°C tolerance
- 8-10 iterations per calculation
- Balances accuracy with acceptable computational cost

**Enthalpy calculation:** ASHRAE exact formulation
- Formula: h = 1.006·t + ω·(2501 + 1.86·t) kJ/kg
- Matches ASHRAE reference tables exactly
- Considers both dry air and water vapor specific heats
- No approximations, full physical accuracy

### Unit Conventions

**Enthalpy:** kJ/kg
- Standard ASHRAE unit
- Matches ASHRAE 140 and Fundamentals reference tables
- Results: h in kJ/kg (divide by 1000 for J/kg if needed)

**Humidity ratio (ω):** kg/kg (kg_water_vapor / kg_dry_air)
- Standard ASHRAE unit
- Same magnitude as g/kg (divide by 1000 for conversion)
- Matches EPW file format for humidity ratio field

**Saturation vapor pressure (p_sat):** Pa
- Most common in ASHRAE 140
- 1 Pa = 0.01 kPa
- Standard for dew point calculations
- Atmospheric pressure range: ~101325 Pa at sea level

**Temperature inputs:** °C
- Standard ASHRAE unit
- All weather data uses °C
- HVAC calculations use °C
- Keep consistent across module (no Kelvin conversions)

### Validation Approach

**Reference values:** Both ASHRAE Fundamentals and ASHRAE 140 equipment cases
- Use ASHRAE Fundamentals Chapter 1 psychrometric tables for comprehensive validation
- Use ASHRAE 140 Cases 800-810 validation data for HVAC equipment context
- Most comprehensive validation approach
- Ensures both general accuracy and HVAC-specific correctness

**Test coverage density:** Fine grid (130 test points)
- Temperature grid: 2°C intervals from -10°C to 40°C (26 temperatures)
- Relative humidity grid: 10%, 30%, 50%, 70%, 90% (5 levels)
- Total: 26 × 5 = 130 test conditions
- Very thorough coverage of operating range
- Tests edge cases (cold, hot, dry, humid, moderate)

**Tolerance levels:** Strict
- Dew point and wet-bulb temperature: ±0.5°C
- Humidity ratio: ±1%
- Enthalpy: ±0.5 kJ/kg
- High confidence in accuracy
- Matches ASHRAE 140 validation standards

**Test structure:** Unit + property + integration tests
- Unit tests: Reference value validation at grid points
- Property tests: Invariant checks (dew_point ≤ dry_bulb, enthalpy increases with T/RH)
- Integration tests: Verify economizer control uses psychrometrics correctly
- Full coverage from low-level calculations to system integration

### Module Placement

**Module path:** src/weather/psychrometrics.rs
- New module with clean separation of concerns
- Reusable across codebase
- Easy to test independently
- Follows established mod.rs pattern (denver.rs, epw.rs as sub-modules)

**API pattern:** Trait methods
- Define PsychrometricCalculations trait
- Implement for relevant types (e.g., HourlyWeatherData)
- Consistent with ContinuousTensor/ContinuousField trait pattern in codebase
- Extensible for future psychrometric needs

**Module export:** weather:: re-export
- pub use self::psychrometrics::*; in src/weather/mod.rs
- Users call weather::calculate_dew_point()
- Clean namespace, follows existing weather module pattern
- Convenient for Python/Rust bindings

**Integration with weather data:** Helper functions
- Add helper functions like from_weather_data(weather: &HourlyWeatherData)
- Extracts inputs (temp, humidity) and calls calculation functions
- More convenient than passing individual fields
- Pure module: psychrometrics.rs contains only calculation functions
- Integration code in helpers bridges weather data to psychrometrics

### Claude's Discretion

- Exact saturation vapor pressure formula coefficients (researcher determines from ASHRAE)
- Iteration convergence tolerance for dew/wet-bulb calculations (1e-6 typical)
- Maximum iteration limits for iterative calculations (20 iterations prevents infinite loops)
- Property test generation strategy (quickcheck vs manual invariants)
- Integration test design (economizer mode activation conditions)

---

<code_context>
## Existing Code Insights

### Reusable Assets

**HourlyWeatherData (src/weather/mod.rs):**
- Contains dry_bulb_temp (°C) and humidity (% RH) fields
- Already validated for range and completeness
- WeatherSource trait provides iter_hours() for batch processing
- Has sky_temperature() and sky_emissivity() as reference for similar methods

**EconomizerMode enum (src/sim/hvac/economizer.rs):**
- Defines Disabled, DryBulb, Enthalpy variants
- is_economizer_active() has placeholder for enthalpy calculations
- calculate_free_cooling_capacity() uses hardcoded air density and specific heat
- Test: test_enthalpy_mode_deferred() expects enthalpy parameters

**WeatherSource trait (src/weather/mod.rs):**
- Unified interface for weather data access
- get_hourly_data() returns HourlyWeatherData
- iter_hours() provides convenient iteration over full year
- Validation methods: validate_all(), is_complete()

### Established Patterns

**Trait-based abstractions (ContinuousTensor, ContinuousField):**
- Codebase uses traits for common behavior across implementations
- Apply same pattern to PsychrometricCalculations trait
- Supports code reuse and consistent testing

**Physics-first approach (Phase 14, Phase 15):**
- Address accuracy before optimization
- Validate against ASHRAE 140 reference ranges before feature completeness
- Apply same principle: validate psychrometrics before economizer integration

**Validation-driven development:**
- ASHRAE 140 suite is primary validation target
- Compare against reference ranges with strict tolerances
- Use property tests for invariant verification

**BatchOracle pattern constraint:**
- Pre-commit hook enforces single-level parallelism (par_iter at population level only)
- Psychrometric calculations should not introduce nested par_iter() calls
- Maintain >1,000 configs/sec throughput for population evaluation

**Module organization (src/weather/):**
- Sub-modules for different weather sources (denver.rs, epw.rs)
- Each sub-module is self-contained with tests
- mod.rs re-exports sub-module contents
- Apply same pattern to psychrometrics.rs

### Integration Points

**Where psychrometrics module lives:**
- src/weather/psychrometrics.rs — New module following existing pattern
- Contains pure calculation functions and trait definitions
- Helper functions for HourlyWeatherData integration

**Where to expose psychrometric functions:**
- src/weather/mod.rs — Add `pub mod psychrometrics;`
- Re-export: `pub use self::psychrometrics::*;`
- Users access via weather::calculate_dew_point(), weather::enthalpy(), etc.

**Where economizer uses psychrometrics:**
- src/sim/hvac/economizer.rs — Update is_economizer_active() for Enthalpy mode
- Remove placeholder enthalpy calculation (lines 61-63)
- Call weather::enthalpy_from_weather() to get outdoor and zone enthalpy
- Enable full Enthalpy economizer mode functionality

**Where air properties live:**
- src/weather/psychrometrics.rs — Define constants for air properties
- Air density: ρ = 1.2 kg/m³ (used in free_cooling_capacity)
- Specific heat of dry air: cp_air = 1006 J/(kg·K)
- Specific heat of water vapor: cp_water = 1860 J/(kg·K)
- Use in enthalpy and wet-bulb calculations

**Where validation tests live:**
- src/weather/psychrometrics.rs — Module-level #[cfg(test)] tests
- Unit tests: test_dew_point_reference_values(), test_enthalpy_formula(), etc.
- Property tests: test_dew_point_le_dry_bulb(), test_enthalpy_monotonic(), etc.
- Integration tests: src/sim/hvac/economizer.rs::tests::test_enthalpy_mode_integration()

**Where documentation lives:**
- src/weather/psychrometrics.rs — Doc comments on each function
- ASHRAE Fundamentals chapter reference in module documentation
- Formula explanations with variable definitions
- Example usage in function documentation

---

<specifics>
## Specific Ideas

**Dew point calculation (ASHRAE exact):**
- Magnus-Tetens for saturation vapor pressure: p_sat(T) = 610.78 × exp((17.27 × T)/(T + 237.3))
- Convert to dew point: Solve p_sat(Td) = p_sat(T) × (RH/100)
- Use Newton-Raphson iteration for convergence
- Reference: ASHRAE Fundamentals Chapter 1, Psychrometrics

**Wet-bulb calculation (iterative):**
- Psychrometric equation: Tw = function(T, RH, P_atm)
- Iterate: Tw_{n+1} = f(Tw_n) until |Tw_{n+1} - Tw_n| < tolerance
- Enthalpy balance at wet-bulb: h(Tw, RH=100%) = h(T, RH)
- Use ASHRAE Fundamentals iterative algorithm
- Reference: ASHRAE Fundamentals Chapter 1

**Enthalpy calculation (ASHRAE exact):**
```rust
// Humidity ratio from relative humidity
ω = (0.622 × p_sat(T) × RH/100) / (P_atm - p_sat(T) × RH/100)

// Enthalpy of moist air (ASHRAE exact formula)
h = 1.006 × T + ω × (2501 + 1.86 × T)  // kJ/kg
```
- Where: 1.006 = specific heat of dry air (kJ/kg·K)
- 2501 = latent heat of vaporization at 0°C (kJ/kg)
- 1.86 = specific heat of water vapor (kJ/kg·K)

**Test reference values (ASHRAE Fundamentals):**
- At 25°C, 50% RH: Dew point ≈ 13.9°C, Enthalpy ≈ 50.4 kJ/kg
- At 20°C, 80% RH: Dew point ≈ 16.4°C, Enthalpy ≈ 49.0 kJ/kg
- At 30°C, 20% RH: Dew point ≈ 5.0°C, Enthalpy ≈ 36.3 kJ/kg
- Use these as anchor points in unit tests

**Property tests:**
- Invariant: dew_point(T, RH) ≤ T for all valid inputs
- Invariant: enthalpy(T, RH) increases monotonically with T at fixed RH
- Invariant: enthalpy(T, RH) increases monotonically with RH at fixed T
- Invariant: humidity_ratio(T, RH) increases monotonically with RH at fixed T

**Economizer integration:**
- Replace placeholder in is_economizer_active() (lines 61-63)
- Call: let outdoor_h = weather::enthalpy_from_weather(outdoor_weather)
- Call: let zone_h = weather::enthalpy_from_weather(zone_weather)
- Use outdoor_h < zone_h condition for Enthalpy mode
- Remove enthalpy Optional parameters (now always Some when psychrometrics available)

**Helper functions:**
```rust
// Convert HourlyWeatherData to psychrometric inputs
pub fn from_weather_data(weather: &HourlyWeatherData) -> PsychrometricInputs {
    PsychrometricInputs {
        temperature: weather.dry_bulb_temp,
        relative_humidity: weather.humidity,
        pressure: STANDARD_ATMOSPHERIC_PRESSURE_Pa, // 101325 Pa
    }
}

// Calculate enthalpy directly from weather data
pub fn enthalpy_from_weather(weather: &HourlyWeatherData) -> f64 {
    let inputs = from_weather_data(weather);
    calculate_enthalpy(inputs.temperature, inputs.relative_humidity, inputs.pressure)
}
```

**Saturation vapor pressure coefficients:**
- Magnus coefficients: A = 6.108, B = 17.27, C = 237.3 (for T in °C)
- Use ASHRAE-specific coefficients if available (researcher to verify)
- Validate against ASHRAE Fundamentals reference tables

---

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. All decisions relate to psychrometric calculations, unit conventions, validation approach, and module placement as defined in Phase 16 requirements.

---

*Phase: 16-psychrometrics-module*
*Context gathered: 2026-03-13*
