# Phase 20: Data Quality & Finalization - Context

**Gathered:** 2026-03-15
**Status:** Ready for planning

---

<domain>
## Phase Boundary

Replace all mock data, placeholders, and hardcodes with configurable, validated parameters for production readiness.

**What this delivers:**
- Configurable building assembly system (replaces hardcoded material properties) — PHYS-02, PHYS-07
- Standard constants module with source references (replaces hardcoded physical constants) — PHYS-03
- Complete TMY3/EPW parsing (replaces placeholder weather values) — WEATHER-01, WEATHER-04, WEATHER-05
- 8R3C thermal network evaluation (if warranted by findings) — PHYS-06
- Configuration validation and documentation (replaces unchecked inputs) — DATA-02, DATA-03, DATA-04, DATA-05

This phase finalizes v0.4 by ensuring all data is real, configurable, and validated—no more mock predictions, placeholders, or hardcoded values.

</domain>

---

<decisions>
## Implementation Decisions

### Building Assembly System Design

**Structure:** Hybrid approach
- Predefined assembly types in JSON (like building_profiles.json from Phase 17)
- Layer material properties as trait-based (MaterialLayer trait with conductivity, thickness, density, specific_heat)
- Combines user-friendly defaults with programmatic extensibility
- Follows established patterns: Equipment trait (Phase 17) + JSON profile loading (Phase 17)

**Layer composition:** To be determined based on Claude's analysis
- Options: Sequential builder API, Array-based definition, Template-based (LightMassWall::standard_exterior())
- User feedback: "Think about this and decide what's best solution or solutions would be"
- Researcher to evaluate trade-offs: flexibility vs simplicity vs ASHRAE 140 validation needs

**Material properties:** All categories selected
- Core properties: Thermal conductivity (W/mK), density (kg/m³), specific heat (J/kgK) — required for R-value and thermal mass
- Radiation properties: Thermal absorptance/emittance, emissivity — required for detailed solar/thermal coupling
- Advanced properties: Moisture (vapor permeability, absorption), structural (strength, stiffness) — enables future humidity modeling
- Source metadata: Material source database, property uncertainty ranges, applicability conditions — rich documentation for research use

**Mass classification:** Auto-calculated
- Calculated automatically from thermal capacitance = Σ(density × specific_heat × thickness × area)
- ISO 13790 Annex C provides thresholds (VeryLight < 50 kJ/m²K, Light 50-150, Medium 150-260, Heavy 260-370, VeryHeavy > 370)
- No user configuration required, always consistent with ASHRAE 140 methodology

### Constants Module Organization

**Organization:** Domain-based
- Group constants by physics domain: thermal.rs, solar.rs, atmospheric.rs
- Subfolders for reference sources: thermal/ashrae.rs, thermal/iso.rs, solar/ashrae.rs
- Clear separation, easy to find constants by physics domain
- Structure: `src/physics/constants/thermal/ashrae.rs`

**Documentation level:** Complete
- Value + units + source reference (ASHRAE 140 Table X, ISO 13790 Annex C)
- Validity conditions (temperature range, pressure range)
- Uncertainty range (±5%, ±0.01 typical for physical constants)
- Assumptions and references to primary literature
- Academic-grade documentation for research/engineering use

**Version handling:** Version modules
- Module subfolders for standards versions: `ashrae_140/v2021.rs`, `ashrae_140/v2023.rs`
- Cleaner separation than naming constants with version suffixes
- Can select version via feature flag or import
- Most flexible approach for tracking standards evolution

**Derived constants:** Hybrid
- Pre-calculate values for standard ASHRAE 140 constructions (e.g., EFFECTIVE_MASS_LIGHT = 50 kJ/m²K)
- Provide computation functions for custom constructions (e.g., calculate_effective_thermal_mass(layers))
- Best of both: fast lookup for common cases, flexibility for research
- Covers ISO 13790 Annex C effective thermal mass formulas

### Weather Data Completeness

**EPW version support:** All formats selected
- EPW v2 (hourly): 8760 hourly records, EnergyPlus default, ASHRAE 140 minimum
- EPW v3 (sub-hourly): 35040 15-minute records, high-resolution modeling, more complex parsing
- AMY format: Actual Meteorological Year, real historical data, different record structure
- IWEC format: International Weather for Energy Calculations, global weather sources, broader geographic support

**Missing fields:** All categories selected
- Ground temperature: For foundation heat loss calculations, important for low-rise buildings with slab floors
- Illuminance: Horizontal illuminance (lux) or diffuse/direct illuminance for daylighting calculations and electric lighting savings
- Snow depth/cover: For roof albedo variations, affects solar reflection and thermal mass, seasonal effects
- Present weather observations: Rain, snow, fog codes and visibility for advanced HVAC control (freeze protection, economizer)

**TMY3 embedding:** Download on-demand
- Embed location metadata (URL, lat/lon, elevation) in binary
- Download TMY3 data on-demand from weather repository at runtime
- Unlimited locations, up-to-date data
- Network dependency with caching required
- Recommended for modern cloud-native approach

**Sub-hourly interpolation:** Piecewise methods
- Combine linear segments with curve fitting at boundaries
- Best balance: reasonable complexity and accuracy
- More accurate than pure linear, less oscillation than cubic spline
- Different interpolation per field if needed (linear for temperature, step for discrete observations, cubic for radiation)

### Configuration Validation

**Validation level:** Full physics constraints
- Min/max bounds checking per parameter (e.g., thermal_conductivity > 0.01)
- Type validation (f64 for floats, proper ranges)
- Cross-field consistency (layer thickness sums to wall thickness, HVAC capacity matches thermal load)
- Physical constraints (energy balance, entropy non-decrease)
- Research-grade validation, ensures correct physics

**Error recovery:** Fail fast
- Return detailed error immediately with location and context
- Example: `Error::InvalidMaterialLayer { file: "config.json:42", thickness: -0.05, suggestion: "Use positive thickness" }`
- User fixes config, reloads simulation
- Clear error location, no hidden issues
- Recommended for ASHRAE 140 validation (requires exactness)

**Validation timing:** Both load and runtime
- Validate all at config load, reject invalid configs before simulation starts
- Critical checks at runtime (temperature bounds, energy conservation)
- Combines fast feedback with runtime safety
- Most comprehensive approach

**Error display:** Structured JSON
- Output to stdout as parseable JSON for tooling/CI integration
- Structure: `{"validation": "failed", "errors": [{"path": "src/config.json:42", "field": "layer.thickness", "value": -0.05, "message": "Thickness must be positive"}]}`
- Enables automation, post-processing, and CI integration
- Silent console output except JSON result

### Claude's Discretion

**Building assembly composition:**
- Determine best approach for layer composition (sequential builder, array-based, template-based)
- Evaluate trade-offs for ASHRAE 140 validation needs vs flexibility for research

**Weather caching strategy:**
- Cache downloaded TMY3 data locally to avoid repeated network calls
- Cache invalidation strategy (time-based, version-based, or checksum-based)
- Cache location and format (filesystem path, compressed, serialized)

**Sky model variations:**
- Implementation approach for clearness index and cloud cover effects (WEATHER-05)
- Integration with existing solar radiation calculations
- Validation against ASHRAE 140 reference cases

</decisions>

<specifics>
## Specific Ideas

**Building assembly system (Hybrid structure):**
```rust
// Material layer trait
pub trait MaterialLayer {
    fn conductivity(&self) -> f64;  // W/mK
    fn thickness(&self) -> f64;     // m
    fn density(&self) -> f64;        // kg/m³
    fn specific_heat(&self) -> f64;  // J/kgK
    fn absorptance(&self) -> f64;   // Solar absorptance
    fn emissivity(&self) -> f64;     // Thermal emissivity
}

// Builder for composing assemblies
pub struct ConstructionBuilder {
    layers: Vec<Box<dyn MaterialLayer>>,
}

impl ConstructionBuilder {
    pub fn add_layer(&mut self, layer: Box<dyn MaterialLayer>) -> &mut Self {
        self.layers.push(layer);
        self
    }

    pub fn build(&self) -> Result<BuildingAssembly, ConstructionError> {
        // Validate assembly
        // Calculate effective thermal mass
        // Determine mass classification
        BuildingAssembly::from_layers(self.layers)?
    }
}

// Predefined assemblies from JSON
// data/building_assemblies.json
{
  "light_mass_wall": {
    "layers": [
      {"type": "concrete", "thickness": 0.1},
      {"type": "insulation", "thickness": 0.05},
      {"type": "gypsum", "thickness": 0.012}
    ]
  },
  "heavy_mass_wall": {
    "layers": [...]
  }
}
```

**Constants module (Domain-based with versions):**
```
src/physics/constants/
├── thermal/
│   ├── mod.rs
│   ├── ashrae_140/
│   │   ├── v2021.rs
│   │   ├── v2023.rs
│   │   └── mod.rs
│   └── iso_13790/
│       └── annex_c.rs
├── solar/
│   ├── mod.rs
│   └── ashrae_140.rs
└── atmospheric.rs
```

**Example constant documentation (Complete level):**
```rust
/// Solar constant (total solar irradiance at Earth's mean distance).
///
/// **Value:** 1361 W/m²
/// **Units:** W/m² (watts per square meter)
/// **Source:** ASHRAE Handbook of Fundamentals, Chapter 14, Solar Radiation
/// **Reference:** IPCC AR6 (2021) - 1361.0 ±0.5 W/m²
/// **Uncertainty:** ±0.5 W/m² (0.04%) due to orbital variations
/// **Validity:** Valid for Earth's mean distance from Sun (1 AU). Varies ±3.4% annually at perihelion/aphelion.
/// **Assumptions:** Assumes solar spectrum outside atmosphere, clear sky conditions, Earth as sphere.
/// **Notes:** This is the extraterrestrial solar irradiance; ground-level irradiance is attenuated by atmosphere (~1000 W/m² peak).
pub const SOLAR_CONSTANT: f64 = 1361.0;
```

**EPW parsing structure (all versions):**
```rust
pub enum EpwVersion {
    V2 { records: Vec<HourlyRecord> },     // 8760 records
    V3 { records: Vec<SubHourlyRecord> }, // 35040 records (15-min)
    AMY { records: Vec<HourlyRecord> },     // Actual Meteorological Year
    IWEC { records: Vec<HourlyRecord> },    // International Weather for Energy Calculations
}

pub fn parse_epw<R: Read>(reader: R) -> Result<EpwVersion, EpwError> {
    // Detect version from file header
    // Parse based on version
}
```

**Sub-hourly interpolation (Piecewise methods):**
```rust
pub enum InterpolationMethod {
    Linear,           // T_30min = (T_hour + T_next_hour) / 2
    CubicSpline,     // Smooth transitions for radiation
    Step,             // Discrete observations (rain, snow)
    PiecewiseHermite, // Continuity at boundaries
}

pub fn interpolate_weather<T>(
    field: &str,
    t1: T,
    t2: T,
    fraction: f64,
    method: InterpolationMethod,
) -> T {
    // Select method based on field type
    match method {
        InterpolationMethod::Linear => linear_interpolate(t1, t2, fraction),
        InterpolationMethod::PiecewiseHermite => hermite_interpolate(t1, t2, fraction),
        // ...
    }
}
```

**Validation error structure (Structured JSON):**
```rust
#[derive(Serialize, Deserialize)]
pub struct ValidationError {
    pub path: String,           // e.g., "config.json:42"
    pub field: String,          // e.g., "layer.thickness"
    pub value: serde_json::Value,
    pub message: String,
    pub suggestion: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct ValidationResult {
    pub validation: String,       // "passed" or "failed"
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationError>,
}
```

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets

**construction.rs (src/sim/construction.rs):**
- Has interior/exterior film coefficients: `INTERIOR_FILM_COEFF`, `EXTERIOR_FILM_COEFF_DEFAULT`
- `SurfaceType` enum for surface-specific coefficients (Wall, Ceiling, Floor)
- `exterior_film_coeff()` function for wind-speed-based coefficient
- Foundation for building assembly R-value calculations

**weather/mod.rs:**
- `HourlyWeatherData` struct with weather fields (dry_bulb_temp, dni, dhi, ghi, wind_speed, humidity, horizontal_infrared)
- `WeatherSource` trait for abstracting weather data sources
- Foundation for EPW and TMY integration

**weather/denver.rs:**
- `DenverTmyWeather` struct with embedded synthetic weather
- Currently hardcoded synthetic data, not file-based
- Location metadata available (latitude, longitude, elevation)

**weather/epw.rs:**
- EPW parsing infrastructure exists but incomplete
- Can be extended for v2, v3, AMY, IWEC support

**building_profiles.json (from Phase 17):**
- Pattern for JSON-based configuration files
- `ProfileBundle` struct with lighting, equipment, occupancy profiles
- Can extend to building assemblies

**Phase 14 audit results (DATA-01):**
- Codebase audit documented all placeholders, mocks, and hardcoded values
- Provides starting point for data quality cleanup

### Established Patterns

**Trait-based abstractions (from Phases 15-17):**
- Equipment trait (Phase 17) for load modeling
- VariableCapacityEquipment trait (Phase 15) for HVAC equipment
- PsychrometricCalculations trait (Phase 16) for psychrometric functions
- Apply same pattern to MaterialLayer trait for building assemblies

**JSON configuration files (from Phase 17):**
- Building profiles loaded from JSON with serde deserialization
- Easy to modify without recompiling
- Supports user customization and external tooling

**Validation-driven development (from Phases 14-19):**
- All features validated against ASHRAE 140 reference ranges
- Apply same principle to building assemblies and constants
- Compare against reference values with strict tolerances

**Physics-first approach:**
- Address data quality before optimization
- Validate all parameters against ASHRAE/ISO standards
- Ensure physical correctness before performance tuning

### Integration Points

**Where building assembly system lives:**
- New module: `src/sim/assembly.rs` — Building assembly and material layer management
- Define `MaterialLayer` trait with thermal properties
- Define `BuildingAssembly` struct with composed layers
- Add to `src/sim/mod.rs`: `pub mod assembly;`

**Where constants module lives:**
- New directory: `src/physics/constants/` — Central constants module
- Subdirectories by domain: `thermal/`, `solar/`, `atmospheric.rs`
- Subdirectories by source: `ashrae_140/`, `iso_13790/`
- Add to `src/physics/mod.rs`: `pub mod constants;`

**Where extended weather parsing lives:**
- Extend `src/weather/epw.rs` for all EPW versions
- Add missing fields to `HourlyWeatherData` struct
- Implement sub-hourly interpolation functions
- Add TMY3 download/caching infrastructure

**Where configuration validation lives:**
- New module: `src/validation/config.rs` — Configuration validation
- Define validation error types and structures
- Implement validation functions for different config types (assembly, weather, HVAC)
- Add to `src/validation/mod.rs`: `pub mod config;`

**Where physical parameter documentation lives:**
- Doc comments in constants module files
- Complete documentation per constant (value, units, source, uncertainty, validity, assumptions)
- Markdown documentation in `docs/PHYSICAL_CONSTANTS.md` as reference

**Where integration with existing code happens:**
- `src/sim/engine.rs` — ThermalModel to use BuildingAssembly instead of hardcoded material properties
- `src/weather/mod.rs` — HourlyWeatherData to use extended fields
- CLI validation — `fluxion validate` to use config validation module

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. All decisions relate to building assembly system, constants module, weather data completeness, and configuration validation as defined in Phase 20 requirements (PHYS-02, PHYS-03, PHYS-06, PHYS-07, WEATHER-01, WEATHER-03, WEATHER-04, WEATHER-05, DATA-02, DATA-03, DATA-04, DATA-05).

</deferred>

---

*Phase: 20-data-quality-finalization*
*Context gathered: 2026-03-15*
