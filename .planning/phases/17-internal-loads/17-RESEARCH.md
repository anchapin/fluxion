# Phase 17: Internal Loads - Research

**Researched:** 2026-03-13
**Domain:** Building Energy Modeling - Internal Heat Gains
**Confidence:** HIGH

## Summary

Phase 17 implements realistic internal heat gain modeling with configurable schedules for lighting, equipment, and occupancy. This phase enhances Fluxion's 5R1C thermal network by adding time-varying internal loads that accurately represent real building operation patterns, essential for ASHRAE 140 Cases 600-960 compliance.

**Primary recommendation:** Follow existing patterns in `LightingSchedule`, `OccupancyProfile`, and `DailySchedule` modules. Use trait-based `Equipment` abstraction with `Vec<Box<dyn Equipment>>` for mixed equipment types. Extend `DailySchedule` with `Weekly` variant (168 values = 7 days × 24 hours) and `DayType` enum for weekday/weekend/holiday selection.

## User Constraints (from CONTEXT.md)

### Locked Decisions

**Weekly Schedule Structure:**
- Phase 17: Extend `DailySchedule` with `Weekly` type (168 values = 7 days × 24 hours)
- Future: Design `Schedule` trait for seasonal breaks, shift patterns, complex day types
- Schedule storage: Nested `[[f64; 24]; 7]` for clear semantics and cache-friendly access
- Pre-populated patterns: Office defaults included via factory method `DailySchedule::weekly("Office-Default").office_hours()`
- Day type identification: Flexible `DayType` enum `{Weekday, Weekend, Holiday, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday}` with individual day overrides

**Equipment Load Model:**
- Model depth: 3 core types in Phase 17 — `ComputerEquipment`, `ServerRack`, `GenericEquipment`
- Future: Add more types for retail, restaurant, hospital, etc.
- Shared behavior: `Equipment` trait with methods `id()`, `power_at_hour()`, `convective_gains()`, `radiative_gains()`
- Thermal characteristics: Full thermal profile with `radiative_fraction`, `convective_fraction`, `mass_coupling_factor` (0-1)
- Radiative vs convective split: Matches `LightingSchedule` and `OccupancyProfile` patterns
- Mass coupling factor: Equipment radiative heat absorption by thermal mass (high coupling = more heat to mass)

**ThermalModel Integration:**
- Integration approach: Direct module calls with optional arguments
- Solve method signature: `solve_timesteps(&mut self, steps: usize, lighting: Option<&LightingSchedule>, equipment: Option<&[Box<dyn Equipment>]>, occupancy: Option<&OccupancyProfile>) -> f64`
- Loads passed as optional arguments, not stored in `ThermalModel` (keeps `ThermalModel` minimal)
- Heat gain calculation: Mass-coupled radiative approach for equipment
- Schedule indexing: Day of year + hour with day_type lookup using `get_day_type(day_of_year, holiday_calendar)`

**Building Type Defaults:**
- Scope: Three building types — Office (primary for ASHRAE 140), Retail, School (tests extensibility)
- Deferred: Hospital, Hotel, Restaurant, Warehouse to future phases
- Default storage: JSON/YAML files at `data/building_profiles.json` (or YAML equivalent)
- Loading mechanism: Auto-load by type with `ThermalModel.building_type` field and `profiles::load_building_profile()`
- Override capability: Users can manually pass specific loads via optional arguments if needed
- Profile lookup: Load from JSON at first use, cache for performance

### Claude's Discretion

- Exact JSON/YAML schema and file format details
- Holiday calendar implementation (date range vs specific dates, built-in vs external file)
- Error handling for missing/invalid profile files
- Mass coupling factor default values per equipment type (determine from ASHRAE data)
- Radiative/convective fractions default values per equipment type (e.g., computers: 0.3/0.7, servers: 0.5/0.5)
- Schedule trait design and implementation for future seasonal/shift patterns

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope. All decisions relate to weekly schedules, equipment models, ThermalModel integration, and building type defaults as defined in Phase 17 requirements (LOADS-01 through LOADS-04).

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| LOADS-01 | Implement internal lighting loads with schedules | Existing `LightingSchedule` module with convective/radiative split; extend with weekly schedule pattern |
| LOADS-02 | Implement internal equipment loads with schedules | New `Equipment` trait with 3 core types; follow existing `LightingSchedule` pattern for thermal characteristics |
| LOADS-03 | Implement occupancy/people loads with schedules | Existing `OccupancyProfile` module with 168-value weekly schedule; already has sensible/latent heat split |
| LOADS-04 | Support customizable load profiles (weekday/weekend/holiday) | Extend `DailySchedule` with `Weekly` variant and `DayType` enum; add holiday calendar logic |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `serde` | 1.0.203 | Serialization/deserialization for JSON/YAML profiles | Existing codebase uses serde for all config/data files; standard Rust ecosystem choice |
| `serde_json` | 1.0.119 | JSON parsing for building profiles | Proven pattern in codebase (e.g., `validation/report.rs`, `validation/commands.rs`) |
| `serde_yaml` | 0.9.34+ | YAML alternative for building profiles | Used in existing codebase (e.g., `fluxion.rs` line 490, `assembly_library.rs` line 49) |
| `std::fs` | Rust std | File I/O for profile loading | Standard Rust library, no external dependency needed |
| `std::collections::HashMap` | Rust std | Profile caching and lookup | Existing pattern in codebase for key-value storage |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `once_cell` or `std::sync::OnceLock` | Rust std | Profile caching to avoid repeated file I/O | Used in existing codebase (e.g., `engine.rs` line 31 `DAILY_CYCLE: OnceLock<[f64; 24]>`) |
| `thiserror` | 1.0.61 (existing) | Error handling for profile loading | Existing codebase uses thiserror for error types |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| JSON | TOML | TOML more human-readable but JSON is standard for web APIs; both supported by serde |
| `DailySchedule::Weekly` | Separate `WeeklySchedule` struct | Reusing `DailySchedule` with `ScheduleType::Weekly` variant maintains consistent API |
| `Vec<Box<dyn Equipment>>` | Enum `EquipmentType` with variants | Trait approach more extensible; allows users to implement custom equipment types |

**Installation:**
```bash
# No new dependencies needed - all already in Cargo.toml
# Existing dependencies:
# serde = { version = "1.0", features = ["derive"] }
# serde_json = "1.0"
# serde_yaml = "0.9"
# thiserror = "1.0"
```

## Architecture Patterns

### Recommended Project Structure

```
src/sim/
├── schedule.rs         # Extend with Weekly type and DayType enum
├── lighting.rs        # Existing - extend with weekly schedule support
├── occupancy.rs       # Existing - already has weekly schedule (168 values)
├── equipment.rs       # NEW - Equipment trait + ComputerEquipment, ServerRack, GenericEquipment
├── profiles.rs        # NEW - Profile loading from JSON/YAML, caching
├── holiday.rs        # NEW - Holiday calendar and day type lookup
└── engine.rs         # Modify solve_timesteps() to accept optional load arguments

data/
└── building_profiles.json  # NEW - Default profiles for Office, Retail, School

tests/
├── test_equipment.rs       # NEW - Equipment trait and type tests
├── test_weekly_schedule.rs # NEW - Weekly schedule and DayType tests
├── test_profiles.rs        # NEW - Profile loading tests
└── test_internal_loads.rs  # NEW - Integration tests with ThermalModel
```

### Pattern 1: Weekly Schedule Extension

**What:** Extend existing `DailySchedule` with `ScheduleType::Weekly` variant and 168-value storage

**When to use:** Need hourly load patterns that vary by day of week (weekday/weekend/holiday)

**Example:**
```rust
// Source: src/sim/schedule.rs (existing pattern extended)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DayType {
    Weekday,    // Mon-Fri non-holiday
    Weekend,    // Sat-Sun non-holiday
    Holiday,    // Designated holiday
    Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday,  // Individual days override aggregates
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DailySchedule {
    pub name: String,
    pub schedule_type: ScheduleType,
    // REPLACE: pub values: [f64; 24]
    // WITH: Conditional values based on ScheduleType
    pub values: [[f64; 24]; 7],  // 7 days × 24 hours = 168 values
}

impl DailySchedule {
    pub fn weekly(name: String) -> Self {
        Self {
            name,
            schedule_type: ScheduleType::Weekly,
            values: [[0.0; 24]; 7],
        }
    }

    pub fn value_for_day(&self, day_type: DayType, hour: usize) -> f64 {
        let day_idx = match day_type {
            DayType::Weekday => 0,      // Default to Monday
            DayType::Weekend => 5,      // Default to Saturday
            DayType::Holiday => 0,       // Same as weekday
            DayType::Monday => 0,
            DayType::Tuesday => 1,
            DayType::Wednesday => 2,
            DayType::Thursday => 3,
            DayType::Friday => 4,
            DayType::Saturday => 5,
            DayType::Sunday => 6,
        };
        self.values[day_idx][hour % 24]
    }

    pub fn office_hours(mut self) -> Self {
        for day in 0..5 {  // Monday-Friday
            for hour in 8..=17 {  // 8am-6pm
                self.values[day][hour] = 1.0;
            }
        }
        self
    }
}
```

### Pattern 2: Equipment Trait Abstraction

**What:** Trait-based design for equipment load modeling following codebase patterns (`ContinuousTensor`, `PsychrometricCalculations`)

**When to use:** Need consistent API across different equipment types and support mixed equipment lists

**Example:**
```rust
// Source: src/sim/equipment.rs (NEW module)
pub trait Equipment {
    fn id(&self) -> &str;
    fn power_at_hour(&self, hour_of_year: usize) -> f64;
    fn convective_gains(&self, hour_of_year: usize) -> f64;
    fn radiative_gains(&self, hour_of_year: usize) -> f64;
    fn mass_coupling_factor(&self) -> f64;  // Equipment-specific radiative heat split to mass
}

pub struct ComputerEquipment {
    pub id: String,
    pub rated_power_w: f64,
    pub count: usize,
    pub schedule: DailySchedule,
    pub radiative_fraction: f64,   // e.g., 0.3
    pub convective_fraction: f64,   // e.g., 0.7
    pub mass_coupling_factor: f64,  // e.g., 0.2
}

impl Equipment for ComputerEquipment {
    fn id(&self) -> &str { &self.id }

    fn power_at_hour(&self, hour_of_year: usize) -> f64 {
        let day_of_week = (hour_of_year / 24) % 7;
        let hour = hour_of_year % 24;
        let day_type = get_day_type(hour_of_year / 24);
        let schedule_value = self.schedule.value_for_day(day_type, hour);
        self.rated_power_w * self.count as f64 * schedule_value
    }

    fn convective_gains(&self, hour_of_year: usize) -> f64 {
        self.power_at_hour(hour_of_year) * self.convective_fraction
    }

    fn radiative_gains(&self, hour_of_year: usize) -> f64 {
        self.power_at_hour(hour_of_year) * self.radiative_fraction
    }

    fn mass_coupling_factor(&self) -> f64 {
        self.mass_coupling_factor
    }
}
```

### Pattern 3: Mass-Coupled Radiative Heat Distribution

**What:** Equipment radiative heat splits between air and thermal mass based on `mass_coupling_factor`

**When to use:** More accurate 5R1C physics accounting for equipment placement and thermal characteristics

**Example:**
```rust
// Source: src/sim/engine.rs (solve_timesteps inner loop)
for step in 0..steps {
    let day_of_year = step / 24;
    let hour = step % 24;
    let day_type = holiday::get_day_type(day_of_year);

    // Calculate internal heat gains
    let mut internal_convective = 0.0;
    let mut internal_radiative_to_air = 0.0;
    let mut internal_radiative_to_mass = 0.0;

    // Lighting: fixed convective/radiative split
    if let Some(lighting) = lighting_ref {
        internal_convective += lighting.convective_heat_gains(hour);
        internal_radiative_to_mass += lighting.radiative_heat_gains(hour);
    }

    // Equipment: mass-coupled radiative heat split
    if let Some(equipment_list) = equipment_ref {
        for equipment in equipment_list {
            let equipment_rad = equipment.radiative_gains(step);
            internal_convective += equipment.convective_gains(step);
            internal_radiative_to_air += equipment_rad * (1.0 - equipment.mass_coupling_factor());
            internal_radiative_to_mass += equipment_rad * equipment.mass_coupling_factor();
        }
    }

    // Occupancy: fixed convective/radiative split (existing pattern)
    if let Some(occ) = occupancy_ref {
        internal_convective += occ.convective_heat_gains(step);
        internal_radiative_to_mass += occ.radiative_heat_gains(step);
    }

    // Add to energy balance (5R1C integration)
    // Ti: internal air temperature, Tm: thermal mass temperature
    // Air mass: zone air thermal capacity
    // Mass thermal cap: h_tr_ms coupling conductance
    dt = 3600.0; // 1 hour in seconds
    self.temperatures[zone] += internal_convective / air_mass * dt;
    self.temperatures[zone] += internal_radiative_to_air / air_mass * dt;
    self.mass_temperatures[zone] += internal_radiative_to_mass / mass_thermal_cap * dt;

    // ... rest of physics (solar, HVAC, etc.)
}
```

### Pattern 4: Profile Loading and Caching

**What:** Load building profiles from JSON/YAML with caching to avoid repeated file I/O

**When to use:** Need default load profiles per building type without recompiling

**Example:**
```rust
// Source: src/sim/profiles.rs (NEW module)
use std::collections::HashMap;
use std::sync::OnceLock;
use std::fs;
use serde_json;

static PROFILE_CACHE: OnceLock<HashMap<BuildingType, ProfileBundle>> = OnceLock::new();

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileBundle {
    pub lighting: LightingSchedule,
    pub equipment: Vec<Box<dyn Equipment>>,
    pub occupancy: OccupancyProfile,
}

pub fn load_building_profile(building_type: BuildingType) -> Result<ProfileBundle, String> {
    // Check cache first
    if let Some(cache) = PROFILE_CACHE.get() {
        if let Some(profile) = cache.get(&building_type) {
            return Ok(profile.clone());
        }
    }

    // Load from file
    let profile_path = format!("data/building_profiles.json");
    let content = fs::read_to_string(&profile_path)
        .map_err(|e| format!("Failed to read profile file: {}", e))?;

    let profiles: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse profile JSON: {}", e))?;

    let building_key = match building_type {
        BuildingType::Office => "office",
        BuildingType::Retail => "retail",
        BuildingType::School => "school",
        _ => return Err("Unsupported building type".to_string()),
    };

    // Extract and parse profile for building type
    let profile_data = &profiles["profiles"][building_key];
    let bundle = ProfileBundle::from_json(profile_data)?;

    // Cache for future use
    PROFILE_CACHE.get_or_init(|| {
        let mut cache = HashMap::new();
        cache.insert(building_type, bundle.clone());
        cache
    });

    Ok(bundle)
}
```

### Pattern 5: Holiday Calendar Integration

**What:** Determine day type (Weekday/Weekend/Holiday) for schedule lookup

**When to use:** Need to apply holiday schedules with correct day type resolution

**Example:**
```rust
// Source: src/sim/holiday.rs (NEW module)
use std::collections::HashSet;
use std::sync::OnceLock;

static US_FEDERAL_HOLIDAYS: OnceLock<HashSet<usize>> = OnceLock::new();

pub fn get_day_type(day_of_year: usize) -> DayType {
    let day_of_week = day_of_year % 7;  // 0=Monday, ..., 6=Sunday

    // Check if holiday
    let holidays = US_FEDERAL_HOLIDAYS.get_or_init(|| calculate_holidays_for_year());
    if holidays.contains(&day_of_year) {
        return DayType::Holiday;
    }

    // Weekday or Weekend
    match day_of_week {
        0..=4 => DayType::Weekday,  // Monday-Friday
        5..=6 => DayType::Weekend,  // Saturday-Sunday
        _ => DayType::Weekday,
    }
}

fn calculate_holidays_for_year() -> HashSet<usize> {
    // US federal holidays (year-agnostic formulas)
    // New Year's Day, MLK Day, Presidents' Day, Memorial Day, Juneteenth,
    // Independence Day, Labor Day, Columbus Day, Veterans Day, Thanksgiving, Christmas
    // Returns HashSet of day_of_year (1-365)
    todo!("Implement holiday calculations")
}
```

### Anti-Patterns to Avoid

- **Hardcoding schedule values in Rust code**: Use JSON/YAML profiles for easy modification without recompiling
- **Nested `par_iter()` in internal load calculations**: Violates BatchOracle pattern - must keep single-level parallelism at population level only
- **Storing loads in `ThermalModel` struct**: Pass as optional arguments to keep `ThermalModel` minimal and maintain Clone capability
- **Ignoring mass coupling factor**: Equipment radiative heat must split between air and mass for accurate 5R1C physics
- **Using flat 168-value array**: Use nested `[[f64; 24]; 7]` for clear semantics matching day_type lookup

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| JSON/YAML parsing | Custom string parsing | `serde_json`, `serde_yaml` | Handles edge cases, error handling, type safety |
| Profile caching | Repeated file I/O or global mutable state | `std::sync::OnceLock` or `once_cell` | Thread-safe lazy initialization, used in existing codebase |
| Schedule lookup logic | Manual day_of_week calculations | `DayType` enum with `value_for_day()` method | Clear semantics, matches existing patterns, supports future extensions |
| Error types | Custom error structs | `thiserror` macro | Existing codebase uses thiserror, consistent error handling |

**Key insight:** Internal load modeling requires careful thermal physics integration. Don't hand-roll schedule resolution, profile loading, or error handling - use proven patterns from the codebase and standard Rust ecosystem.

## Common Pitfalls

### Pitfall 1: Incorrect Radiative Heat Distribution

**What goes wrong:** Equipment radiative heat goes entirely to air or entirely to mass, causing inaccurate HVAC demand

**Why it happens:** Missing `mass_coupling_factor` or using fixed fractions instead of equipment-specific values

**How to avoid:**
- Always implement `mass_coupling_factor()` in `Equipment` trait
- Apply mass-coupled split: `radiative_to_mass = radiative * coupling_factor`, `radiative_to_air = radiative * (1.0 - coupling_factor)`
- Test with different coupling factors to verify HVAC demand changes appropriately

**Warning signs:**
- Cooling demand consistently too high/low across all equipment types
- Equipment type doesn't affect HVAC energy (all equipment treated identically)

### Pitfall 2: Violating BatchOracle Pattern

**What goes wrong:** Nested parallelism in internal load calculations causes performance degradation and thread safety issues

**Why it happens:** Adding `par_iter()` inside `solve_timesteps()` when processing equipment list or schedule lookups

**How to avoid:**
- Keep `solve_timesteps()` single-threaded (called per config)
- Only use `rayon::par_iter()` at population level in `BatchOracle::evaluate_population()`
- Pre-commit hook `batch-oracle-pattern` will catch violations

**Warning signs:**
- Performance regression in `BatchOracle::evaluate_population()`
- Pre-commit hook fails with nested par_iter() warning
- Race conditions or data corruption in tests

### Pitfall 3: Schedule Indexing Off-by-One Errors

**What goes wrong:** Wrong schedule values applied to wrong hours, causing unrealistic load patterns

**Why it happens:** Confusing `hour_of_year` (0-8759) vs `hour_of_day` (0-23) vs `day_of_week` (0-6)

**How to avoid:**
- Consistent indexing: `day_of_year = step / 24`, `hour_of_day = step % 24`, `day_of_week = day_of_year % 7`
- Use `DayType` enum lookup instead of manual day_of_week calculations
- Test schedule values at known times (e.g., verify 8am on Monday uses correct schedule value)

**Warning signs:**
- Loads active at wrong times (e.g., lights on at 3am)
- Weekly pattern shifted by one day
- Panic from array index out of bounds

### Pitfall 4: Missing Profile Error Handling

**What goes wrong:** Application crashes or returns confusing errors when profile file is missing or malformed

**Why it happens:** Using `unwrap()` or `expect()` instead of proper error propagation

**How to avoid:**
- Use `Result<T, String>` for profile loading functions
- Provide clear error messages: `"Failed to read profile file: {path}: {io_error}"`
- Validate fractions sum to 1.0: `radiative + convective == 1.0`
- Handle missing profile gracefully: return default or error

**Warning signs:**
- Test failures when `data/building_profiles.json` doesn't exist
- Panic with "index out of bounds" or "failed to parse JSON"
- Confusing error messages in production

### Pitfall 5: Breaking Clone on ThermalModel

**What goes wrong:** Can't clone `ThermalModel` for BatchOracle pattern

**Why it happens:** Adding non-Clone fields to `ThermalModel` (e.g., `Vec<Box<dyn Equipment>>` stored directly)

**How to avoid:**
- Pass loads as optional arguments to `solve_timesteps()`, not stored fields
- If storing is necessary, use `Arc<[Box<dyn Equipment>]>` for shared ownership
- Verify `ThermalModel: Clone` constraint is maintained

**Warning signs:**
- Compiler error: `the trait Clone is not implemented for ThermalModel`
- BatchOracle tests fail with clone-related errors

## Code Examples

Verified patterns from existing codebase:

### Lighting Schedule Factory Method

```rust
// Source: src/sim/lighting.rs (line 188-194)
impl LightingSchedule {
    pub fn office_schedule(power_density: f64, zone_area: f64) -> Self {
        let mut schedule = Self::new(power_density, zone_area);
        for hour in 8..=17 {
            schedule.hourly_schedule[hour] = 1.0;
        }
        schedule
    }
}
```

### Occupancy Weekly Schedule Pattern

```rust
// Source: src/sim/occupancy.rs (line 94-128)
impl OccupancyProfile {
    pub fn office_schedule(mut self) -> Self {
        self.hourly_schedule = vec![0.0; 168];  // 7 days × 24 hours

        for day in 0..5 {  // Weekdays
            for hour in 0..24 {
                let idx = day * 24 + hour;
                let fraction = match hour {
                    0..=6 => 0.05,
                    7 => 0.20,
                    8 => 0.50,
                    9..=11 => 0.90,
                    12 => 0.80,
                    13..=16 => 0.90,
                    17 => 0.70,
                    18 => 0.40,
                    19 => 0.20,
                    20..=23 => 0.10,
                    _ => 0.05,
                };
                self.hourly_schedule[idx] = fraction;
            }
        }

        // Weekend: minimal occupancy
        for day in 5..7 {
            for hour in 0..24 {
                let idx = day * 24 + hour;
                self.hourly_schedule[idx] = 0.05;
            }
        }

        self
    }
}
```

### Internal Loads Integration Pattern

```rust
// Source: src/sim/engine.rs (line 1214-1238)
// Current internal loads setup in ThermalModel::from_spec()
let mut loads_vec = Vec::with_capacity(num_zones);
for zone_idx in 0..num_zones {
    let zone_floor_area = if zone_idx < spec.geometry.len() {
        spec.geometry[zone_idx].floor_area()
    } else {
        spec.geometry[0].floor_area()
    };

    if zone_idx < spec.internal_loads.len() {
        if let Some(ref loads) = spec.internal_loads[zone_idx] {
            let load_per_m2 = loads.total_load / zone_floor_area;
            loads_vec.push(load_per_m2);
            if zone_idx == 0 {
                model.convective_fraction = loads.convective_fraction;
            }
        } else {
            loads_vec.push(0.0);
        }
    } else {
        loads_vec.push(0.0);
    }
}
model.loads = VectorField::new(loads_vec);
```

### JSON Parsing Pattern

```rust
// Source: src/validation/commands.rs (line 196)
use serde_json;
use std::fs;

let content = fs::read_to_string(&reference_path)?;
let parsed: MultiReferenceDB = serde_json::from_str(&content)?;
```

### OnceLock Caching Pattern

```rust
// Source: src/sim/engine.rs (line 31-52)
use std::sync::OnceLock;

static DAILY_CYCLE: OnceLock<[f64; 24]> = OnceLock::new();

fn get_daily_cycle() -> &'static [f64; 24] {
    DAILY_CYCLE.get_or_init(|| {
        let mut arr = [0.0; 24];
        for (h, val) in arr.iter_mut().enumerate() {
            *val = ((h as f64 / 24.0 * 2.0 * std::f64::consts::PI)
                - std::f64::consts::PI / 2.0).sin();
        }
        arr
    })
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Daily schedules only (24 values) | Weekly schedules (168 values) with DayType lookup | Phase 17 | Enables weekday/weekend/holiday patterns for realistic building operation |
| Single internal load value (W/m²) | Separate lighting, equipment, occupancy with time-varying schedules | Phase 17 | Accurate heat gain modeling for ASHRAE 140 compliance |
| Fixed radiative/convective split | Equipment-specific mass coupling factor for radiative heat | Phase 17 | More accurate 5R1C physics accounting for equipment placement |
| Hardcoded schedules in Rust code | JSON/YAML profile files with auto-loading | Phase 17 | Easy modification without recompiling, supports user customization |

**Deprecated/outdated:**
- Single 24-value daily schedule: Replaced by 168-value weekly schedule with DayType enum
- Fixed internal load per zone: Replaced by separate lighting/equipment/occupancy with schedules
- Hardcoded schedule values: Replaced by JSON/YAML profile files

## Open Questions

1. **Holiday calendar implementation details**
   - What we know: Need `get_day_type(day_of_year: usize) -> DayType` function
   - What's unclear: Should holidays be calculated year-agnostically (formulas) or loaded from external file? Should holiday dates be configurable per year?
   - Recommendation: Implement year-agnostic formulas for US federal holidays in Phase 17, add external file support in future phase if needed

2. **Mass coupling factor default values**
   - What we know: Equipment needs `mass_coupling_factor` (0-1) for radiative heat split
   - What's unclear: What are appropriate defaults for ComputerEquipment, ServerRack, GenericEquipment?
   - Recommendation: Use conservative defaults (0.2-0.4) based on typical equipment placement, allow user override via profile file

3. **Radiative/convective fraction defaults per equipment type**
   - What we know: Existing `LightingSchedule` uses 0.2/0.8 (convective/radiative), `OccupancyProfile` uses 0.6/0.4
   - What's unclear: What are typical values for computers, servers, generic equipment?
   - Recommendation: Research ASHRAE Fundamentals or EnergyPlus default values, use 0.3/0.7 for computers, 0.5/0.5 for servers as starting point

4. **Profile file location and error handling**
   - What we know: Need `data/building_profiles.json` with Office, Retail, School profiles
   - What's unclear: Should missing profile file be fatal error or use hardcoded defaults? Should profile file be included in git or generated at build time?
   - Recommendation: Include default profiles in git (committed), allow user override with local file, graceful error handling with clear messages

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Cargo test (Rust built-in) |
| Config file | Cargo.toml (no separate test config) |
| Quick run command | `cargo test --package fluxion --lib equipment::` |
| Full suite command | `cargo test --package fluxion --lib` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| LOADS-01 | Internal lighting loads with weekly schedules | unit | `cargo test test_weekly_schedule_factory` | ❌ Wave 0 |
| LOADS-01 | Lighting heat gains (convective/radiative) | unit | `cargo test test_lighting_heat_gains` | ✅ Existing (lighting.rs:330) |
| LOADS-02 | Equipment trait implementation | unit | `cargo test test_equipment_trait` | ❌ Wave 0 |
| LOADS-02 | Equipment power calculation with schedules | unit | `cargo test test_equipment_power_at_hour` | ❌ Wave 0 |
| LOADS-02 | Equipment mass-coupled radiative split | unit | `cargo test test_mass_coupled_radiative` | ❌ Wave 0 |
| LOADS-03 | Occupancy weekly schedules | unit | `cargo test test_occupancy_weekly_schedule` | ✅ Existing (occupancy.rs:361) |
| LOADS-03 | Occupancy heat gains (sensible/latent) | unit | `cargo test test_internal_gains` | ✅ Existing (occupancy.rs:384) |
| LOADS-04 | Day type lookup (weekday/weekend/holiday) | unit | `cargo test test_day_type_lookup` | ❌ Wave 0 |
| LOADS-04 | Profile loading from JSON | unit | `cargo test test_building_profile_loading` | ❌ Wave 0 |
| LOADS-01/02/03/04 | Internal loads integration with ThermalModel | integration | `cargo test test_internal_loads_integration` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `cargo test --package fluxion --lib equipment:: schedule:: profiles::`
- **Per wave merge:** `cargo test --package fluxion --lib` (full test suite)
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `src/sim/equipment.rs` - Equipment trait and implementations
- [ ] `src/sim/profiles.rs` - Profile loading and caching
- [ ] `src/sim/holiday.rs` - Holiday calendar and day type lookup
- [ ] `src/sim/schedule.rs` - Extend with Weekly type and DayType enum
- [ ] `tests/test_equipment.rs` - Equipment trait and type tests
- [ ] `tests/test_weekly_schedule.rs` - Weekly schedule and DayType tests
- [ ] `tests/test_profiles.rs` - Profile loading tests
- [ ] `tests/test_internal_loads.rs` - Integration tests with ThermalModel
- [ ] `data/building_profiles.json` - Default profiles for Office, Retail, School

*(If no gaps: "None — existing test infrastructure covers all phase requirements")*

## Sources

### Primary (HIGH confidence)

- **Existing codebase patterns** - Verified by reading source files:
  - `src/sim/schedule.rs` - DailySchedule structure and factory methods
  - `src/sim/lighting.rs` - LightingSchedule with convective/radiative split
  - `src/sim/occupancy.rs` - OccupancyProfile with 168-value weekly schedule
  - `src/sim/engine.rs` - ThermalModel structure and solve_timesteps loop
  - `src/validation/ashrae_140_cases.rs` - InternalLoads struct (200W typical value)
- ** serde documentation** - https://serde.rs/ - Serialization framework used throughout codebase
- ** Rust std library** - https://doc.rust.org/std/ - OnceLock, HashMap, fs module for caching and I/O

### Secondary (MEDIUM confidence)

- **ASHRAE 140 Standard** - Internal load values from test case specifications (200W typical, 0.6/0.4 convective/radiative split)
- **Existing test patterns** - `src/sim/lighting.rs` (line 330-343), `src/sim/occupancy.rs` (line 361-450) - Proven test structure for load modeling

### Tertiary (LOW confidence)

- **Building energy modeling best practices** - General knowledge of internal load modeling (marked for validation):
  - Typical lighting power density: 8-12 W/m² for office buildings
  - Typical equipment power density: 5-10 W/m² for office buildings
  - Typical occupancy density: 0.4-0.8 persons/m² for office buildings
  - Radiative/convective fractions vary by equipment type (need ASHRAE Fundamentals verification)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All dependencies already in Cargo.toml, proven patterns in codebase
- Architecture: HIGH - Existing modules provide clear patterns to follow, CONTEXT.md decisions are specific
- Pitfalls: HIGH - Well-understood BatchOracle pattern constraints, thermal physics integration clear from existing code

**Research date:** 2026-03-13
**Valid until:** 30 days (stable domain - internal load modeling fundamentals don't change rapidly)
