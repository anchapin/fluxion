# Phase 20: Data Quality & Finalization - Research

**Researched:** 2026-03-15
**Domain:** Building Energy Simulation Data Quality & Configuration
**Confidence:** HIGH

## Summary

Phase 20 finalizes the v0.4 ASHRAE 140 Compliance milestone by replacing all mock data, placeholders, and hardcoded values with configurable, validated parameters. The phase spans four major domains: (1) Building assembly system with configurable material properties, (2) Standard constants module with source references and uncertainty ranges, (3) Complete TMY3/EPW weather data parsing with multiple geographic locations, and (4) Comprehensive configuration validation with error reporting.

The codebase has strong foundations to build upon: existing `construction.rs` with ISO 13790 Annex C implementation, `assembly_library.rs` with YAML loading patterns from Phase 17, and robust EPW parsing in `epw.rs`. The Phase 14 audit (`audit_report.json`) identified specific mock data locations in AI modules (`batch_inference.rs`, `distributed.rs`, `ensemble.rs`) that need to be replaced with real implementations or proper test infrastructure.

**Primary recommendation:** Leverage existing patterns (trait-based abstractions, JSON configuration, validation-driven development) to build modular, testable systems. Use the building assembly library approach from Phase 17's `ProfileBundle` pattern as the template for material properties configuration.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Building Assembly System Design:**
- Hybrid approach: Predefined assembly types in JSON + trait-based layer properties
- MaterialLayer trait with core properties: conductivity, thickness, density, specific_heat, absorptance, emissivity
- ConstructionBuilder pattern for composing assemblies with validation
- JSON-based configuration following Phase 17 building_profiles.json pattern
- Auto-calculated mass classification using ISO 13790 Annex C thresholds

**Constants Module Organization:**
- Domain-based structure: `src/physics/constants/thermal/`, `src/physics/constants/solar/`, `src/physics/constants/atmospheric.rs`
- Subfolders by source: `ashrae_140/`, `iso_13790/`
- Version handling: Module subfolders (e.g., `ashrae_140/v2021.rs`, `v2023.rs`)
- Complete documentation level: Value + units + source reference + validity conditions + uncertainty range + assumptions
- Derived constants: Hybrid (pre-calculated for standard constructions + computation functions for custom)

**Weather Data Completeness:**
- EPW version support: V2 (8760 hourly), V3 (35040 sub-hourly), AMY, IWEC
- Missing fields: Ground temperature, illuminance, snow depth/cover, present weather observations
- TMY3 embedding: Download on-demand with metadata (URL, lat/lon, elevation) embedded in binary
- Sub-hourly interpolation: Piecewise methods (linear segments + curve fitting at boundaries)

**Configuration Validation:**
- Full physics constraints: Min/max bounds, type validation, cross-field consistency, physical constraints
- Error recovery: Fail fast with detailed error messages including path, field, value, suggestion
- Validation timing: Both load-time (all configs) and runtime (critical checks)
- Error display: Structured JSON output for tooling/CI integration

**Claude's Discretion Areas:**
- Building assembly composition: Determine best approach for layer composition (sequential builder, array-based, template-based)
- Weather caching strategy: Cache location, format, invalidation strategy
- Sky model variations: Implementation approach for clearness index and cloud cover effects

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope. All decisions relate to building assembly system, constants module, weather data completeness, and configuration validation as defined in Phase 20 requirements.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-------------------|
| PHYS-02 | Replace hardcoded material properties with configurable building assembly system | Existing `ConstructionLayer` and `AssemblyLibrary` patterns; JSON configuration from Phase 17; MaterialLayer trait design aligns with Equipment trait pattern |
| PHYS-03 | Replace hardcoded physical constants with standard constants module | Domain-based organization aligns with `src/weather/psychrometrics.rs` pattern; ISO 13790 Annex C has defined thresholds; ASHRAE 140 specifies film coefficients and material properties |
| PHYS-06 | Evaluate 8R3C thermal network (6R2C showed no improvement) | Phase 12 evaluation showed 6R2C had no accuracy improvement; 8R3C evaluation follows same methodology; requires additional thermal resistance nodes |
| PHYS-07 | Support multiple building types (lightweight to heavyweight construction) | ISO 13790 Annex C provides thermal mass classification thresholds; existing `ConstructionType::HighMass` enum; mass classification auto-calculation from thermal capacitance |
| WEATHER-01 | Remove placeholder weather values; implement complete TMY3/EPW parsing | Existing `EpwWeatherSource` parses EPW v2; needs extension for v3, AMY, IWEC; Phase 14 audit identified Denver synthetic weather as placeholder |
| WEATHER-03 | Implement advanced solar radiation interpolation for sub-hourly timesteps | Piecewise Hermite interpolation provides C1 continuity; existing solar calculations in `src/sim/solar.rs` and `src/sim/sky_radiation.rs` |
| WEATHER-04 | Support multiple geographic locations (not just Denver TMY) | TMY3 download on-demand requires HTTP client and caching; existing `WeatherSource` trait abstraction supports multiple sources |
| WEATHER-05 | Implement sky model variations (clearness index, cloud cover effects) | Existing sky emissivity calculation in `DenverTmyWeather::generate_hourly_data()`; clearness index from DHI/DNI ratio |
| DATA-02 | Replace all placeholder data with real implementations | Phase 14 audit identified 24+ mock locations in AI modules; need real ONNX models or test infrastructure |
| DATA-03 | Replace all hardcoded values with configuration | Constants module centralizes physical constants; building assembly system centralizes material properties |
| DATA-04 | Add validation for all configuration inputs | Structured JSON error format; validation functions for assemblies, weather, HVAC parameters |
| DATA-05 | Document all physical parameters with source references | Complete documentation level with uncertainty ranges; ASHRAE 140 and ISO 13790 source references |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `serde` | 1.0 | JSON/YAML configuration serialization | Used throughout codebase; battle-tested; type-safe |
| `serde_json` | 1.0 | JSON configuration files | Same as building_profiles.json from Phase 17 |
| `serde_yaml` | 0.9 | YAML configuration files | Used in `assembly_library.rs`; human-readable configs |
| `thiserror` | 1.0 | Error handling for validation | Used in `WeatherError`; provides structured error types |
| `reqwest` | 0.11 | HTTP client for TMY3 downloads | Async HTTP client; blocking mode available; TLS support |
| `chrono` | 0.4 | Date/time handling for weather data | Industry standard; time zone support; leap year handling |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `directories` | 5.0 | Cache directory location | Cross-platform cache paths (~/.cache/fluxion/) |
| `sha2` | 0.10 | TMY3 file checksum validation | Verify downloaded data integrity |
| `lazy_static` | 1.4 | Thread-safe cache initialization | For TMY3 download cache map |
| `rust-ini` | 0.18 | Configuration file parsing | If INI format preferred over JSON for user config |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| JSON config | TOML | TOML more readable but JSON ecosystem stronger in Rust |
| reqwest | surf | surf is async-first, but reqwest has better blocking API |
| chrono | time | time is newer but chrono has broader ecosystem adoption |
| serde_yaml | json5 | json5 allows comments but YAML is more powerful |

**Installation:**
```bash
# Update Cargo.toml dependencies
cargo add serde serde_json serde_yaml thiserror reqwest chrono directories sha2 lazy_static
```

## Architecture Patterns

### Recommended Project Structure

```
src/
├── physics/
│   └── constants/
│       ├── mod.rs                    # Constants module entry point
│       ├── thermal/
│       │   ├── mod.rs
│       │   ├── ashrae_140/
│       │   │   ├── mod.rs           # Version selection
│       │   │   ├── v2021.rs        # ASHRAE 140-2021 constants
│       │   │   └── v2023.rs        # ASHRAE 140-2023 constants
│       │   └── iso_13790/
│       │       └── annex_c.rs        # Thermal mass classification thresholds
│       ├── solar/
│       │   ├── mod.rs
│       │   └── ashrae_140.rs       # Solar constant, declination angles
│       └── atmospheric.rs            # Atmospheric pressure, air density formulas
├── sim/
│   ├── assembly.rs                  # Existing: extend with MaterialLayer trait
│   ├── construction.rs              # Existing: ISO 13790 Annex C implementation
│   └── profiles.rs                # Existing: ProfileBundle from Phase 17
├── weather/
│   ├── mod.rs                      # Existing: WeatherSource trait
│   ├── epw.rs                     # Existing: extend for v3, AMY, IWEC
│   ├── tmy3.rs                    # NEW: TMY3 download and caching
│   └── interpolation.rs            # NEW: Sub-hourly interpolation functions
└── validation/
    ├── mod.rs                      # Existing: validation module entry
    └── config.rs                  # NEW: Configuration validation

data/
├── assemblies.yaml                  # NEW: Building assembly definitions
├── materials.yaml                 # NEW: Material property database
└── weather_locations.json          # NEW: TMY3 location metadata

tests/
└── integration/
    ├── test_assembly_validation.rs   # NEW: Assembly system tests
    ├── test_constants_module.rs       # NEW: Constants validation tests
    └── test_weather_parsing.rs       # NEW: EPW/TMY3 parsing tests
```

### Pattern 1: Trait-Based Material Properties

**What:** `MaterialLayer` trait with thermal properties, implemented by concrete material structs.

**When to use:** When building needs extensible material types with consistent API.

**Example:**
```rust
// Source: src/sim/assembly.rs (new file)
pub trait MaterialLayer: Send + Sync {
    fn name(&self) -> &str;
    fn conductivity(&self) -> f64;      // W/mK
    fn thickness(&self) -> f64;         // m
    fn density(&self) -> f64;          // kg/m³
    fn specific_heat(&self) -> f64;    // J/kgK
    fn absorptance(&self) -> f64;     // Solar absorptance
    fn emissivity(&self) -> f64;       // Thermal emissivity
    fn r_value(&self) -> f64 {
        self.thickness() / self.conductivity()
    }
}

// Implementation for specific materials
pub struct ConcreteMaterial {
    name: String,
    conductivity: f64,
    density: f64,
    specific_heat: f64,
    thickness: f64,
    absorptance: f64,
    emissivity: f64,
}

impl MaterialLayer for ConcreteMaterial {
    fn name(&self) -> &str { &self.name }
    fn conductivity(&self) -> f64 { self.conductivity }
    fn thickness(&self) -> f64 { self.thickness }
    fn density(&self) -> f64 { self.density }
    fn specific_heat(&self) -> f64 { self.specific_heat }
    fn absorptance(&self) -> f64 { self.absorptance }
    fn emissivity(&self) -> f64 { self.emissivity }
}
```

### Pattern 2: Builder Pattern for Assembly Composition

**What:** Fluent API for constructing multi-layer assemblies with validation.

**When to use:** When construction requires multiple parameters and validation rules.

**Example:**
```rust
// Source: src/sim/assembly.rs (new file)
pub struct AssemblyBuilder {
    layers: Vec<Box<dyn MaterialLayer>>,
}

impl AssemblyBuilder {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add_layer(mut self, layer: Box<dyn MaterialLayer>) -> Self {
        self.layers.push(layer);
        self
    }

    pub fn build(self) -> Result<BuildingAssembly, AssemblyError> {
        if self.layers.is_empty() {
            return Err(AssemblyError::NoLayers);
        }

        // Validate layer properties
        for layer in &self.layers {
            if layer.thickness() <= 0.0 {
                return Err(AssemblyError::InvalidThickness {
                    material: layer.name().to_string(),
                    thickness: layer.thickness(),
                });
            }
            if layer.conductivity() <= 0.0 {
                return Err(AssemblyError::InvalidConductivity {
                    material: layer.name().to_string(),
                    conductivity: layer.conductivity(),
                });
            }
        }

        Ok(BuildingAssembly::from_layers(self.layers)?)
    }
}
```

### Pattern 3: Constants Module with Version Handling

**What:** Domain-based constants with version submodules for standard evolution.

**When to use:** When constants come from standards that evolve over time.

**Example:**
```rust
// Source: src/physics/constants/thermal/ashrae_140/v2021.rs
/// Interior film coefficient per ASHRAE 140-2021 specification.
///
/// **Value:** 8.29 W/m²K
/// **Units:** W/m²K (watts per square meter Kelvin)
/// **Source:** ASHRAE Standard 140-2021, Table X, Surface Heat Transfer Coefficients
/// **Uncertainty:** ±0.05 W/m²K (measurement variation)
/// **Validity:** Valid for indoor air temperatures 15-35°C, vertical surfaces
/// **Assumptions:** Natural convection, still air, surface emissivity 0.9
pub const INTERIOR_FILM_COEFF: f64 = 8.29;

// Source: src/physics/constants/thermal/ashrae_140/mod.rs
#[cfg(feature = "ashrae_140_v2021")]
pub use v2021::*;

#[cfg(feature = "ashrae_140_v2023")]
pub use v2023::*;

// Default to latest
#[cfg(not(any(feature = "ashrae_140_v2021", feature = "ashrae_140_v2023")))]
pub use v2023::*;
```

### Anti-Patterns to Avoid

- **Hardcoding constants inline:** Always centralize in constants module with source references
- **Ignoring version drift:** Track standard versions with feature flags, not code comments
- **Missing uncertainty ranges:** All constants should have documented uncertainty for research use
- **Skipping validation:** Validate all user inputs with clear error messages
- **Mocking in production code:** Mocks should only be in test modules or test fixtures

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| YAML/JSON parsing | Custom parsers | `serde_yaml`, `serde_json` | Type-safe, battle-tested, handles edge cases |
| HTTP downloads | `curl` or `reqwest::blocking` | `reqwest` with async+blocking modes | Proper TLS, connection pooling, retry logic |
| Date/time handling | Manual month/day calculations | `chrono` | Leap years, time zones, DST handling |
| File path resolution | String concatenation | `directories` crate | Cross-platform paths (~/.cache on Unix, %LOCALAPPDATA% on Windows) |
| Interpolation algorithms | Cubic spline from scratch | Existing `nalgebra` or custom implementation | Piecewise Hermite provides C1 continuity; avoid oscillation |
| Error reporting | `println!` macros | `thiserror` | Structured error types, `#[from]` derive, Display implementation |
| Thread-safe caching | `Arc<Mutex<HashMap>>` | `lazy_static` or `dashmap` | Safe initialization, lock-free reads |

**Key insight:** Building energy simulation has complex physical models (solar geometry, psychrometrics, thermal networks). Hand-rolling these leads to subtle bugs. Use established libraries and patterns; focus innovation on the 5R1C thermal network and AI surrogate integration.

## Common Pitfalls

### Pitfall 1: Inconsistent Units Across Modules

**What goes wrong:** Mixing W/m²K and BTU/h·ft²·°F, or meters vs feet, causing calculation errors.

**Why it happens:** Building simulation literature uses multiple unit systems; easy to copy values without conversion.

**How to avoid:** Document units in docstrings for all constants and parameters. Use type-safe unit wrappers if needed (e.g., `Kelvin` vs `Celsius`). Validate units at module boundaries.

**Warning signs:** Energy results off by factor of 3.5 (W to BTU conversion), 0.0929 (m² to ft²), or 0.3048 (m to ft).

### Pitfall 2: Missing Thermal Mass Classification

**What goes wrong:** Buildings misclassified as light mass when they're heavy mass, causing 229-322% annual energy error (observed in high-mass ASHRAE 140 cases).

**Why it happens:** Hardcoded mass classification doesn't account for actual thermal capacitance.

**How to avoid:** Auto-calculate thermal mass from Σ(density × specific_heat × thickness × area). Use ISO 13790 Annex C thresholds: VeryLight < 50 kJ/m²K, Light 50-150, Medium 150-260, Heavy 260-370, VeryHeavy > 370.

**Warning signs:** Annual energy errors > 15% for high-mass cases (Case 920, Case 960).

### Pitfall 3: Incomplete EPW Field Parsing

**What goes wrong:** Only parsing temperature and solar radiation, missing ground temperature or illuminance needed for advanced features.

**Why it happens:** EPW format has 35+ fields; easy to focus on basic weather variables.

**How to avoid:** Parse all EPW fields defined in EnergyPlus documentation. Use optional fields with sensible defaults. Validate field counts in tests.

**Warning signs:** NaN errors in advanced calculations, missing data in validation reports.

### Pitfall 4: No Configuration Validation

**What goes wrong:** Invalid material properties (negative conductivity, zero thickness) cause panic at runtime.

**Why it happens:** Assumption that config files are valid; no validation at load time.

**How to avoid:** Validate all configuration at load with clear error messages. Use structured JSON errors with path, field, value, suggestion. Fail fast before simulation starts.

**Warning signs:** Panics in thermal calculations, NaN values in results.

### Pitfall 5: Weather Data Not Cached

**What goes wrong:** Downloading TMY3 data on every simulation run causes slow startup and network dependency.

**Why it happens:** No caching strategy for downloaded weather files.

**How to avoid:** Implement file-based cache in `~/.cache/fluxion/tmy3/` with SHA-256 checksums. Check cache before download. Support cache invalidation (time-based, version-based, or manual).

**Warning signs:** Slow simulation startup, network errors on offline runs.

### Pitfall 6: Mock Data in Production Paths

**What goes wrong:** Mock predictions in `SurrogateManager` or `DistributedSurrogateManager` produce unrealistic results.

**Why it happens:** Tests use mock implementations that accidentally execute in production code paths.

**How to avoid:** Use feature flags (`#[cfg(test)]`) for mock implementations. Ensure production code uses real ONNX models. Validate that all mock paths are test-only.

**Warning signs:** Unreasonably perfect predictions, lack of variation, test-specific values in results.

## Code Examples

### Example 1: Building Assembly from JSON Configuration

```rust
// Source: data/assemblies.yaml
light_mass_wall:
  layers:
    - material: "Concrete"
      thickness: 0.1  # m
      conductivity: 1.4  # W/mK
      density: 2300.0  # kg/m³
      specific_heat: 840.0  # J/kgK
      absorptance: 0.7
      emissivity: 0.9
    - material: "Insulation"
      thickness: 0.05  # m
      conductivity: 0.04  # W/mK
      density: 50.0  # kg/m³
      specific_heat: 840.0  # J/kgK
      absorptance: 0.5
      emissivity: 0.9

// Source: src/sim/assembly.rs
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub struct AssemblyYAML {
    pub layers: Vec<LayerYAML>,
}

#[derive(Debug, Deserialize)]
pub struct LayerYAML {
    pub material: String,
    pub thickness: f64,
    pub conductivity: f64,
    pub density: f64,
    pub specific_heat: f64,
    pub absorptance: f64,
    pub emissivity: f64,
}

pub fn load_assemblies(path: &str) -> Result<HashMap<String, BuildingAssembly>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path, e))?;
    let map: HashMap<String, AssemblyYAML> = serde_yaml::from_str(&content)
        .map_err(|e| format!("Failed to parse {}: {}", path, e))?;

    let mut assemblies = HashMap::new();
    for (name, assembly_yaml) in map {
        let assembly = build_assembly(&assembly_yaml)?;
        assemblies.insert(name, assembly);
    }
    Ok(assemblies)
}
```

### Example 2: Constants with Complete Documentation

```rust
// Source: src/physics/constants/solar/ashrae_140.rs
/// Solar constant (total solar irradiance at Earth's mean distance).
///
/// **Value:** 1361.0 W/m²
/// **Units:** W/m² (watts per square meter)
/// **Source:** ASHRAE Handbook of Fundamentals, Chapter 14, Solar Radiation
/// **Reference:** IPCC AR6 (2021) - 1361.0 ±0.5 W/m²
/// **Uncertainty:** ±0.5 W/m² (0.04%) due to orbital variations
/// **Validity:** Valid for Earth's mean distance from Sun (1 AU). Varies ±3.4% annually at perihelion/aphelion.
/// **Assumptions:** Assumes solar spectrum outside atmosphere, clear sky conditions, Earth as sphere.
/// **Notes:** This is the extraterrestrial solar irradiance; ground-level irradiance is attenuated by atmosphere (~1000 W/m² peak).
///
/// # Examples
///
/// ```
/// use fluxion::physics::constants::solar::SOLAR_CONSTANT;
///
/// let total_irradiance = SOLAR_CONSTANT;
/// assert_eq!(total_irradiance, 1361.0);
/// ```
pub const SOLAR_CONSTANT: f64 = 1361.0;
```

### Example 3: TMY3 Download with Caching

```rust
// Source: src/weather/tmy3.rs (new file)
use reqwest::blocking::Client;
use sha2::{Sha256, Digest};
use std::path::{PathBuf, Path};
use std::fs;
use std::io::Write;

pub struct Tmy3Cache {
    cache_dir: PathBuf,
    client: Client,
}

impl Tmy3Cache {
    pub fn new() -> Result<Self, String> {
        let cache_dir = dirs::cache_dir()
            .ok_or("Failed to determine cache directory")?
            .join("fluxion/tmy3");

        fs::create_dir_all(&cache_dir)
            .map_err(|e| format!("Failed to create cache directory: {}", e))?;

        Ok(Tmy3Cache {
            cache_dir,
            client: Client::new(),
        })
    }

    pub fn get_or_download(&self, url: &str, location: &str) -> Result<PathBuf, String> {
        let filename = format!("{}.tmy3", location.replace(' ', '_'));
        let filepath = self.cache_dir.join(&filename);

        // Check cache
        if filepath.exists() {
            return Ok(filepath);
        }

        // Download file
        let response = self.client.get(url)
            .send()
            .map_err(|e| format!("Failed to download TMY3: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("HTTP error: {}", response.status()));
        }

        let content = response.bytes()
            .map_err(|e| format!("Failed to read response: {}", e))?;

        // Verify checksum if available
        let checksum = format!("{:x}", Sha256::digest(&content));

        // Write to cache
        let mut file = fs::File::create(&filepath)
            .map_err(|e| format!("Failed to create cache file: {}", e))?;
        file.write_all(&content)
            .map_err(|e| format!("Failed to write cache file: {}", e))?;

        // Write checksum file
        let checksum_path = filepath.with_extension("sha256");
        let mut checksum_file = fs::File::create(&checksum_path)
            .map_err(|e| format!("Failed to create checksum file: {}", e))?;
        checksum_file.write_all(checksum.as_bytes())
            .map_err(|e| format!("Failed to write checksum: {}", e))?;

        Ok(filepath)
    }
}
```

### Example 4: Configuration Validation with Structured Errors

```rust
// Source: src/validation/config.rs (new file)
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigValidationError {
    #[error("Invalid value at {path}: {field} = {value}")]
    InvalidValue {
        path: String,
        field: String,
        value: serde_json::Value,
    },

    #[error("Missing required field at {path}: {field}")]
    MissingField {
        path: String,
        field: String,
    },

    #[error("Validation failed at {path}: {message}")]
    ValidationError {
        path: String,
        message: String,
    },
}

#[derive(Debug, Serialize)]
pub struct ValidationError {
    pub path: String,
    pub field: String,
    pub value: serde_json::Value,
    pub message: String,
    pub suggestion: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ValidationResult {
    pub validation: String,  // "passed" or "failed"
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationError>,
}

pub fn validate_assembly(assembly: &AssemblyYAML, path: &str) -> ValidationResult {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    for (idx, layer) in assembly.layers.iter().enumerate() {
        let field_path = format!("{}.layers[{}]", path, idx);

        // Validate thickness
        if layer.thickness <= 0.0 {
            errors.push(ValidationError {
                path: field_path.clone(),
                field: "thickness".to_string(),
                value: serde_json::json!(layer.thickness),
                message: "Thickness must be positive".to_string(),
                suggestion: Some("Use thickness > 0.0".to_string()),
            });
        }

        // Validate conductivity
        if layer.conductivity <= 0.0 {
            errors.push(ValidationError {
                path: field_path.clone(),
                field: "conductivity".to_string(),
                value: serde_json::json!(layer.conductivity),
                message: "Conductivity must be positive".to_string(),
                suggestion: Some("Use conductivity > 0.0".to_string()),
            });
        }

        // Validate density
        if layer.density <= 0.0 {
            errors.push(ValidationError {
                path: field_path.clone(),
                field: "density".to_string(),
                value: serde_json::json!(layer.density),
                message: "Density must be positive".to_string(),
                suggestion: Some("Use density > 0.0".to_string()),
            });
        }

        // Validate specific heat
        if layer.specific_heat <= 0.0 {
            errors.push(ValidationError {
                path: field_path.clone(),
                field: "specific_heat".to_string(),
                value: serde_json::json!(layer.specific_heat),
                message: "Specific heat must be positive".to_string(),
                suggestion: Some("Use specific_heat > 0.0".to_string()),
            });
        }

        // Validate emissivity range
        if layer.emissivity < 0.0 || layer.emissivity > 1.0 {
            errors.push(ValidationError {
                path: field_path.clone(),
                field: "emissivity".to_string(),
                value: serde_json::json!(layer.emissivity),
                message: "Emissivity must be in range [0, 1]".to_string(),
                suggestion: Some("Use emissivity between 0.0 and 1.0".to_string()),
            });
        }

        // Validate absorptance range
        if layer.absorptance < 0.0 || layer.absorptance > 1.0 {
            errors.push(ValidationError {
                path: field_path,
                field: "absorptance".to_string(),
                value: serde_json::json!(layer.absorptance),
                message: "Absorptance must be in range [0, 1]".to_string(),
                suggestion: Some("Use absorptance between 0.0 and 1.0".to_string()),
            });
        }
    }

    ValidationResult {
        validation: if errors.is_empty() { "passed".to_string() } else { "failed".to_string() },
        errors,
        warnings,
    }
}
```

### Example 5: Sub-Hourly Weather Interpolation

```rust
// Source: src/weather/interpolation.rs (new file)
/// Interpolation method for weather data.
#[derive(Debug, Clone, Copy)]
pub enum InterpolationMethod {
    Linear,
    CubicSpline,
    PiecewiseHermite,
    Step,  // For discrete observations like rain codes
}

/// Interpolate weather value between two timesteps.
///
/// # Arguments
///
/// * `field` - Field name for method selection (e.g., "dry_bulb_temp")
/// * `t1` - Value at timestep 1
/// * `t2` - Value at timestep 2
/// * `fraction` - Fraction between timesteps (0.0 = t1, 1.0 = t2)
/// * `method` - Interpolation method
///
/// # Returns
///
/// Interpolated value
pub fn interpolate_weather(
    field: &str,
    t1: f64,
    t2: f64,
    fraction: f64,
    method: InterpolationMethod,
) -> f64 {
    match method {
        InterpolationMethod::Linear => {
            t1 + (t2 - t1) * fraction
        }
        InterpolationMethod::CubicSpline => {
            // Cubic Hermite spline with C1 continuity
            let h = t2 - t1;
            let t2_minus_t1 = t2 - t1;
            let h3 = h.powi(3);
            let h2 = h.powi(2);

            // Assume zero derivatives at boundaries (simplified)
            let p0 = t1;
            let p1 = t2;
            let m0 = 0.0;  // Zero derivative at t1
            let m1 = 0.0;  // Zero derivative at t2

            let t = fraction;
            let t2_ = t * t;
            let t3_ = t2_ * t;

            (2.0 * t3_ - 3.0 * t2_ + 1.0) * p0
                + (t3_ - 2.0 * t2_ + t) * m0 * h
                + (-2.0 * t3_ + 3.0 * t2_) * p1
                + (t3_ - t2_) * m1 * h
        }
        InterpolationMethod::PiecewiseHermite => {
            // Piecewise cubic Hermite with continuity at boundaries
            // More accurate than pure linear, less oscillation than cubic spline
            let t = fraction;
            let t2 = t * t;
            let t3 = t2 * t;

            // Hermite basis functions
            let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
            let h10 = t3 - 2.0 * t2 + t;
            let h01 = -2.0 * t3 + 3.0 * t2;
            let h11 = t3 - t2;

            // Assume zero derivatives (can be extended with slope estimation)
            let m0 = 0.0;
            let m1 = 0.0;

            h00 * t1 + h10 * m0 + h01 * t2 + h11 * m1
        }
        InterpolationMethod::Step => {
            // Step function for discrete observations
            if fraction < 0.5 { t1 } else { t2 }
        }
    }
}

/// Select interpolation method based on weather field.
///
/// - Temperature: Linear (smooth transitions)
/// - Solar radiation: Piecewise Hermite (continuous with reasonable smoothness)
/// - Discrete observations: Step (rain codes, cloud cover)
pub fn select_method_for_field(field: &str) -> InterpolationMethod {
    match field {
        "dry_bulb_temp" | "humidity" => InterpolationMethod::Linear,
        "dni" | "dhi" | "ghi" => InterpolationMethod::PiecewiseHermite,
        "present_weather" => InterpolationMethod::Step,
        _ => InterpolationMethod::Linear,
    }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hardcoded material properties in `ThermalModel` | Configurable building assemblies from YAML | Phase 17 (profile loading) | User-customizable buildings without recompilation |
| Embedded Denver synthetic weather only | Downloadable TMY3 data for any location | Phase 20 (planned) | Global weather support, real climate data |
| Mock ONNX predictions in tests | Real ONNX models or test fixtures | Phase 14 (DATA-01) | Production-accurate validation, test isolation |
| No configuration validation | Structured JSON validation at load time | Phase 20 (planned) | Fail-fast errors, better user experience |
| Constants scattered in codebase | Centralized constants module with source refs | Phase 20 (planned) | Research-grade documentation, version tracking |

**Deprecated/outdated:**
- **Embedded weather data only:** Denver synthetic weather will remain for ASHRAE 140 validation, but production simulations should use real TMY3 data
- **Mock data in production:** All mock predictions must be behind `#[cfg(test)]` or removed entirely
- **Hardcoded ASHRAE 140 values:** Move to constants module with versioning and uncertainty ranges

## Open Questions

1. **8R3C Thermal Network Evaluation**
   - **What we know:** Phase 12 evaluation showed 6R2C had no accuracy improvement with 1.5-2x performance penalty
   - **What's unclear:** Whether 8R3C (8 resistance nodes, 3 capacitance nodes) shows different results than 6R2C
   - **Recommendation:** Implement 8R3C evaluation following same methodology as Phase 12. Compare against ASHRAE 140 high-mass cases (Case 920, Case 960). If no accuracy improvement, document finding and keep 5R1C as default.

2. **TMY3 Data Source Selection**
   - **What we know:** Multiple TMY3 repositories exist (NREL, NOAA, EnergyPlus)
   - **What's unclear:** Which repository has most complete global coverage and best license
   - **Recommendation:** Use NREL's TMY3 repository (https://www.nrel.gov/grid/solar-resource/data.html) - free, comprehensive, well-documented. Fallback to NOAA GSOD if location not available.

3. **Weather Cache Invalidation Strategy**
   - **What we know:** TMY3 data changes rarely (typically every 10 years)
   - **What's unclear:** Whether to use time-based (e.g., 1 year), version-based (TMY3 version string), or checksum-based validation
   - **Recommendation:** Use checksum-based validation with optional time-based expiry (e.g., 365 days). Provides flexibility for force-refresh without hardcoding expiration dates.

4. **Building Assembly Composition Pattern**
   - **What we know:** CONTEXT.md defers this decision to Claude
   - **What's unclear:** Whether sequential builder, array-based definition, or template-based (e.g., `LightMassWall::standard_exterior()`) is best for ASHRAE 140 validation needs vs research flexibility
   - **Recommendation:** Use hybrid approach: JSON configuration for predefined assemblies (ASHRAE 140 cases, building types), builder API for programmatic composition. Provides both ease-of-use and extensibility.

5. **Sky Model Implementation for WEATHER-05**
   - **What we know:** Existing `DenverTmyWeather::generate_hourly_data()` calculates sky emissivity from DHI/DNI ratio
   - **What's unclear:** How to integrate clearness index and cloud cover effects with existing solar radiation calculations in `src/sim/solar.rs` and `src/sim/sky_radiation.rs`
   - **Recommendation:** Implement clearness index (kt = GHI / GHI_clear) and modify sky emissivity calculation to account for cloud cover. Validate against ASHRAE 140 reference cases with varying cloud conditions.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in (`cargo test`) |
| Config file | None (uses default Cargo.toml configuration) |
| Quick run command | `cargo test --lib validation::config` |
| Full suite command | `cargo test --lib` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PHYS-02 | Building assembly configuration and validation | unit | `cargo test --lib assembly` | ❌ Wave 0 |
| PHYS-03 | Constants module with source references | unit | `cargo test --lib constants` | ❌ Wave 0 |
| PHYS-06 | 8R3C thermal network evaluation | integration | `cargo test --lib test_8r3c_evaluation` | ❌ Wave 0 |
| PHYS-07 | Thermal mass classification | unit | `cargo test --lib thermal_mass_classification` | ❌ Wave 0 |
| WEATHER-01 | TMY3/EPW parsing for all versions | unit | `cargo test --lib weather::epw` | ⚠️ Partial (epw.rs has tests, needs v3/AMY/IWEC) |
| WEATHER-03 | Sub-hourly interpolation | unit | `cargo test --lib interpolation` | ❌ Wave 0 |
| WEATHER-04 | Multiple geographic locations | integration | `cargo test --lib tmy3_download` | ❌ Wave 0 |
| WEATHER-05 | Sky model variations | unit | `cargo test --lib sky_model` | ❌ Wave 0 |
| DATA-02 | Mock data replacement | integration | `cargo test --lib test_mock_removal` | ⚠️ Partial (audit_report.json exists) |
| DATA-03 | Hardcoded values configuration | unit | `cargo test --lib constants_config` | ❌ Wave 0 |
| DATA-04 | Configuration validation | unit | `cargo test --lib validation::config` | ❌ Wave 0 |
| DATA-05 | Physical parameter documentation | doc-test | `cargo test --doc` | ⚠️ Partial (some constants documented, needs completeness) |

### Sampling Rate

- **Per task commit:** `cargo test --lib <module_name> -- --nocapture` (targeted module tests)
- **Per wave merge:** `cargo test --lib` (all unit tests)
- **Phase gate:** Full suite + `cargo test --doc` + ASHRAE 140 validation run

### Wave 0 Gaps

- [ ] `tests/test_assembly_validation.rs` — covers PHYS-02, DATA-04
- [ ] `tests/test_constants_module.rs` — covers PHYS-03, DATA-05
- [ ] `tests/test_8r3c_evaluation.rs` — covers PHYS-06
- [ ] `tests/test_thermal_mass_classification.rs` — covers PHYS-07
- [ ] `tests/test_interpolation.rs` — covers WEATHER-03
- [ ] `tests/test_tmy3_download.rs` — covers WEATHER-04
- [ ] `tests/test_sky_model.rs` — covers WEATHER-05
- [ ] `tests/test_mock_removal.rs` — covers DATA-02
- [ ] `tests/test_constants_config.rs` — covers DATA-03
- [ ] `src/validation/config.rs` — validation module for DATA-04
- [ ] `src/physics/constants/` — constants module structure for PHYS-03
- [ ] `src/sim/assembly.rs` — assembly system for PHYS-02
- [ ] `src/weather/interpolation.rs` — interpolation functions for WEATHER-03
- [ ] `src/weather/tmy3.rs` — TMY3 download/caching for WEATHER-04

## Sources

### Primary (HIGH confidence)

- **Fluxion codebase analysis** — Reviewed existing `construction.rs`, `assembly_library.rs`, `epw.rs`, `denver.rs` modules
- **Phase 20 CONTEXT.md** — User decisions and constraints for building assembly, constants, weather, validation
- **Phase 14 audit_report.json** — Identified 24+ mock data locations in AI modules
- **ISO 13790 Annex C** — Thermal mass classification thresholds (documented in construction.rs)
- **ASHRAE 140 Standard** — Building material properties and test case specifications

### Secondary (MEDIUM confidence)

- **Phase 17 building_profiles.json** — JSON configuration pattern for load profiles
- **Rust ecosystem documentation** — serde, reqwest, chrono crate documentation
- **Building simulation best practices** — Configuration validation, error handling patterns

### Tertiary (LOW confidence)

- **TMY3 data source selection** — NREL, NOAA, EnergyPlus repositories (needs verification of global coverage and licensing)
- **8R3C vs 6R2C vs 5R1C** — Literature suggests diminishing returns beyond 5R1C for most applications (needs empirical validation)
- **Weather caching strategies** — Time-based, version-based, checksum-based approaches (needs performance benchmarking)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - serde ecosystem well-established; reqwest, chrono industry standards
- Architecture: HIGH - Existing codebase patterns (traits, builders, JSON config) proven in Phases 15-17
- Pitfalls: HIGH - Unit inconsistency, missing thermal mass classification, incomplete EPW parsing observed in Phase 14-19
- Weather data: MEDIUM - EPW parsing exists; TMY3 download/caching needs implementation
- 8R3C evaluation: LOW - Similar to Phase 12 6R2C evaluation, but untested

**Research date:** 2026-03-15
**Valid until:** 2026-04-15 (30 days for stable domains, 7 days for fast-moving dependencies)
