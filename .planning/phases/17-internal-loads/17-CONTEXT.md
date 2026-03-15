# Phase 17: Internal Loads - Context

**Gathered:** 2026-03-13
**Status:** Ready for planning

---

<domain>
## Phase Boundary

Implement realistic internal heat gain modeling with configurable schedules (lighting, equipment, occupancy).

**What this delivers:**
- Internal lighting loads with configurable schedules (weekday/weekend/holiday profiles)
- Internal equipment loads accurately representing office equipment heat gains with time-varying schedules
- Occupancy/people loads modeling both sensible and latent heat gains based on activity level and schedule
- Load profiles customizable per building type (Office, Retail, School) and validated against ASHRAE reference cases

This phase enhances internal load modeling — leverages existing lighting/occupancy modules, integrates new equipment module, and enables weekly schedule patterns.

</domain>

---

<decisions>
## Implementation Decisions

### Weekly Schedule Structure

**Approach:** Hybrid design for current + future needs
- Phase 17: Extend `DailySchedule` with `Weekly` type (168 values = 7 days × 24 hours)
- Future: Design `Schedule` trait for seasonal breaks, shift patterns, complex day types
- Rationale: Phase 17 delivers weekday/weekend/holiday for offices/commercial buildings; trait enables schools, seasonal facilities later without breaking changes

**Schedule storage:** Nested `[[f64; 24]; 7]`
- Clearer semantics: values[day_of_week][hour] vs flat indexing
- Matches codebase pattern for explicit structure
- Cache-friendly access with day-level grouping

**Pre-populated patterns:** Office defaults included
- Factory method: `DailySchedule::weekly("Office-Default").office_hours()`
- Pre-filled: 8am-6pm weekdays, minimal on weekends
- Reduces user configuration burden for common case
- Matches existing `LightingSchedule::office_schedule()` pattern

**Day type identification:** Flexible DayType enum
- Enum: `{Weekday, Weekend, Holiday, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday}`
- Individual days override aggregate types if provided
- Example: If Monday..Friday set, Weekday is ignored. If Saturday/Sunday set, Weekend is ignored.
- Supports both simple (3 types) and granular (10 types) schedules

### Equipment Load Model

**Model depth:** Detailed breakdown with core types
- Phase 17: 3 core types — `ComputerEquipment`, `ServerRack`, `GenericEquipment`
- Future: Add more types for retail, restaurant, hospital, etc.
- Balances complexity with ASHRAE 140 validation needs (office cases need total heat gain, not itemized breakdown)

**Shared behavior:** Equipment trait
- Trait methods: `id()`, `power_at_hour()`, `convective_gains()`, `radiative_gains()`
- All equipment types implement `Equipment` trait
- Consistent API, easier testing, follows codebase trait pattern (ContinuousTensor, PsychrometricCalculations)
- Supports `Vec<Box<dyn Equipment>>` for mixed equipment in zone

**Thermal characteristics:** Full thermal profile
- Fields: `id`, `rated_power_w`, `count`, `schedule: DailySchedule`, `radiative_fraction`, `convective_fraction`, `mass_coupling_factor`
- Radiative vs convective split: Matches `LightingSchedule` and `OccupancyProfile` patterns
- Mass coupling factor (0-1): Equipment radiative heat absorption by thermal mass
- High coupling (e.g., servers in data center) → more heat to mass
- Low coupling (e.g., office PCs) → more heat to air

### ThermalModel Integration

**Integration approach:** Direct module calls
- Solve method signature: `solve_timesteps(&mut self, steps: usize, lighting: Option<&LightingSchedule>, equipment: Option<&[Box<dyn Equipment>]>, occupancy: Option<&OccupancyProfile>) -> f64`
- Loads passed as optional arguments, not stored in `ThermalModel`
- Keeps `ThermalModel` minimal, no new fields added
- Matches user's choice for direct optional args

**Heat gain calculation:** Mass-coupled radiative approach
- Equipment radiative heat splits between air and mass based on `mass_coupling_factor`
```rust
let equipment_radiative = equipment.radiative(hour);
let coupled_to_mass = equipment_radiative * equipment.mass_coupling_factor;
let coupled_to_air = equipment_radiative * (1.0 - equipment.mass_coupling_factor);

Tm_next += coupled_to_mass / mass_thermal_cap * dt;
Ti_act += coupled_to_air / air_mass * dt;
```
- More accurate 5R1C physics: accounts for equipment placement and thermal characteristics
- Lighting and occupancy radiative heat uses fixed fractions (convective to air, radiative to mass)

**Schedule indexing:** Day of year + hour with day_type lookup
- Indexing: `day_of_year = timestep / 24`, `hour = timestep % 24`
- Day type lookup: `get_day_type(day_of_year, holiday_calendar)` → `Weekday/Weekend/Holiday`
- Schedule access: `schedule.value_for_day(day_type, hour)`
- Most flexible approach for future needs (seasonal breaks, shift patterns, holidays)
- Supports weekly patterns (Phase 17) and complex schedules (future Schedule trait)

### Building Type Defaults

**Scope:** Three building types
- Primary: Office (for ASHRAE 140 Cases 600-960 compliance)
- Additional: Retail, School (tests different schedule patterns, validates extensibility)
- Deferred: Hospital, Hotel, Restaurant, Warehouse to future phases
- Reasonable scope for Phase 17: covers common commercial building types without excessive scope creep

**Default storage:** JSON/YAML files
- Location: `data/building_profiles.json` (or YAML equivalent)
- Structure: Nested by building type → load category → parameters
- Example:
```json
{
  "office": {
    "lighting": { "power_density_w_m2": 10.0 },
    "occupancy": { "max_occupancy": 100, "schedule": "8-18 weekdays" },
    "equipment": { "total_power_w": 5000 }
  },
  "retail": { /* ... */ },
  "school": { /* ... */ }
}
```
- Easy to modify without recompiling
- Supports user customization and external tooling

**Loading mechanism:** Auto-load by type
- `ThermalModel` gains `building_type: BuildingType` field
- `solve_timesteps()` auto-loads appropriate profile based on `building_type`
- Convenient: single `model.building_type = BuildingType::Office;` call
- Override capability: Users can manually pass specific loads via optional arguments if needed
- Profile lookup: Load from JSON at first use, cache for performance

### Claude's Discretion

- Exact JSON/YAML schema and file format details
- Holiday calendar implementation (date range vs specific dates, built-in vs external file)
- Error handling for missing/invalid profile files
- Mass coupling factor default values per equipment type (researcher to determine from ASHRAE data)
- Radiative/convective fractions default values per equipment type (e.g., computers: 0.3/0.7, servers: 0.5/0.5)
- Schedule trait design and implementation for future seasonal/shift patterns

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets

**DailySchedule (src/sim/schedule.rs):**
- Has `ScheduleType` enum with `Constant`, `DailyCycle` placeholders for `Weekly` and `Custom`
- 24-value array: `pub values: [f64; 24]`
- Factory methods: `constant()`, `with_operating_hours()`
- Pattern: Extend to support `Weekly` type with 168 values

**LightingSchedule (src/sim/lighting.rs):**
- Already has convective/radiative heat split: `convective_fraction`, `radiative_fraction`
- Factory method: `office_schedule(power_density, zone_area)` exists
- Methods: `lighting_power()`, `convective_heat_gains()`, `radiative_heat_gains()`
- Pattern: Reuse convective/radiative split approach for equipment

**OccupancyProfile (src/sim/occupancy.rs):**
- Already has sensible/latent heat split: `sensible_heat_per_person`, `latent_heat_per_person`
- Already has 7 building types: `Office`, `Retail`, `School`, `Hospital`, `Hotel`, `Restaurant`, `Warehouse`
- Already has 168-value schedule: `hourly_schedule: Vec<f64>` (7 days × 24 hours)
- Heat gains: `internal_gains()`, `convective_heat_gains()`, `radiative_heat_gains()`
- Proven pattern: 168-value weekly schedule works

**ThermalModel (src/sim/engine.rs):**
- 5R1C thermal network with `Ti` (air temps), `Tm` (mass temps) VectorFields
- `solve_timesteps()` calculates HVAC demand from `Ti_free` and setpoints
- Already has `internal_loads` reference (line 1223): spec.internal_loads[zone_idx]
- Integration point: Add internal heat gains to energy balance after solar gain calculation

### Established Patterns

**Trait-based abstractions (ContinuousTensor, ContinuousField, PsychrometricCalculations):**
- Codebase uses traits for common behavior across implementations
- Apply same pattern to `Equipment` trait for load modeling
- Supports code reuse and consistent testing

**Physics-first approach (Phase 14, Phase 15, Phase 16):**
- Address accuracy before optimization
- Validate against ASHRAE 140 reference ranges before feature completeness
- Apply same principle: validate internal load models against reference cases

**Validation-driven development:**
- ASHRAE 140 suite is primary validation target
- Compare against reference ranges with strict tolerances
- Use property tests for invariant verification

**BatchOracle pattern constraint:**
- Pre-commit hook enforces single-level parallelism (par_iter at population level only)
- Internal load calculations should not introduce nested par_iter() calls
- Maintain >1,000 configs/sec throughput for population evaluation

### Integration Points

**Where weekly schedule lives:**
- `src/sim/schedule.rs` — Extend `DailySchedule` struct
- Add `ScheduleType::Weekly` variant
- Add `pub values: [[f64; 24]; 7]` field (replace `[f64; 24]`)
- Add factory methods: `weekly()`, `office_hours()`, `fill_weekday()`, `fill_weekend()`, `fill_holiday()`
- Add `DayType` enum: `{Weekday, Weekend, Holiday, Monday, Tuesday, ..., Sunday}`

**Where equipment module lives:**
- `src/sim/equipment.rs` — New module for equipment load modeling
- Define `Equipment` trait with core methods
- Define `ComputerEquipment`, `ServerRack`, `GenericEquipment` structs implementing trait
- Add to `src/sim/mod.rs`: `pub mod equipment;`

**Where building profiles live:**
- `data/building_profiles.json` — New JSON file for default profiles
- Or YAML equivalent: `data/building_profiles.yaml`
- Add to `.gitignore` if user customization allowed, or commit if defaults are fixed

**Where profile loading logic lives:**
- `src/sim/profiles.rs` — New module for profile management
- Functions: `load_building_profile(type: BuildingType) -> Result<ProfileBundle, Error>`
- Cache loaded profiles to avoid repeated file I/O
- Add to `src/sim/mod.rs`: `pub mod profiles;`

**Where ThermalModel integration happens:**
- `src/sim/engine.rs` — `ThermalModel` struct
- Add `building_type: BuildingType` field
- Modify `solve_timesteps()` signature to accept optional load arguments
- Add internal heat gain calculation in solve loop (after solar, before HVAC)
- Profile lookup: Call `profiles::load_building_profile(model.building_type)` if loads not provided

**Where schedule indexing happens:**
- `src/sim/engine.rs` — `solve_timesteps()` inner loop
- Calculate: `day_of_year = timestep / 24`, `hour = timestep % 24`
- Lookup: `day_type = get_day_type(day_of_year, holiday_calendar)`
- Access: `schedule.value_for_day(day_type, hour)`

**Where holiday calendar lives:**
- New module: `src/sim/holiday.rs` or integrate into profiles.rs
- Simple approach: List of holiday dates for year
- Or calculation-based: US federal holidays formula
- Function: `get_day_type(day_of_year: usize) -> DayType`

**Where tests live:**
- `src/sim/schedule.rs` — Module-level tests for weekly schedule
- `src/sim/equipment.rs` — Module-level tests for Equipment trait and types
- `src/sim/profiles.rs` — Module-level tests for profile loading
- Integration tests: `src/sim/engine.rs` — Test with internal loads enabled vs disabled

</code_context>

---

<specifics>
## Specific Ideas

**Weekly schedule factory methods:**
```rust
impl DailySchedule {
    pub fn weekly(name: String) -> Self {
        Self {
            name,
            schedule_type: ScheduleType::Weekly,
            values: [[0.0; 24]; 7],  // 7 days × 24 hours
        }
    }

    pub fn office_hours(mut self) -> Self {
        for day in 0..5 {  // Monday-Friday
            for hour in 8..=17 {  // 8am-6pm
                self.values[day][hour] = 1.0;
            }
        }
        self
    }

    pub fn fill_weekday(&mut self, start_hour: usize, end_hour: usize, value: f64) {
        for day in 0..5 {
            for hour in start_hour..end_hour {
                self.values[day][hour] = value;
            }
        }
    }

    pub fn fill_weekend(&mut self, start_hour: usize, end_hour: usize, value: f64) {
        for day in 5..7 {
            for hour in start_hour..end_hour {
                self.values[day][hour] = value;
            }
        }
    }

    pub fn fill_holiday(&mut self, start_hour: usize, end_hour: usize, value: f64) {
        for day in 0..7 {
            for hour in start_hour..end_hour {
                self.values[day][hour] = value;
            }
        }
    }

    pub fn value_for_day(&self, day_type: DayType, hour: usize) -> f64 {
        let day_idx = match day_type {
            DayType::Weekday => 0,      // Default to Monday if no specific day
            DayType::Weekend => 5,      // Default to Saturday if no specific day
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
}
```

**Equipment trait:**
```rust
pub trait Equipment {
    fn id(&self) -> &str;
    fn power_at_hour(&self, hour: usize) -> f64;
    fn convective_gains(&self, hour: usize) -> f64;
    fn radiative_gains(&self, hour: usize) -> f64;
}
```

**ComputerEquipment implementation:**
```rust
pub struct ComputerEquipment {
    pub id: String,
    pub rated_power_w: f64,
    pub count: usize,
    pub schedule: DailySchedule,
    pub radiative_fraction: f64,  // e.g., 0.3
    pub convective_fraction: f64,  // e.g., 0.7
    pub mass_coupling_factor: f64,  // e.g., 0.2
}

impl Equipment for ComputerEquipment {
    fn id(&self) -> &str { &self.id }

    fn power_at_hour(&self, hour_of_year: usize) -> f64 {
        let hour = hour_of_year % 24;
        self.rated_power_w * self.count as f64 * self.schedule.value(hour)
    }

    fn convective_gains(&self, hour_of_year: usize) -> f64 {
        self.power_at_hour(hour_of_year) * self.convective_fraction
    }

    fn radiative_gains(&self, hour_of_year: usize) -> f64 {
        self.power_at_hour(hour_of_year) * self.radiative_fraction
    }
}
```

**ThermalModel integration (solve_timesteps):**
```rust
impl ThermalModel {
    pub fn building_type: BuildingType,  // NEW

    pub fn solve_timesteps(
        &mut self,
        steps: usize,
        use_surrogates: bool,
        lighting: Option<&LightingSchedule>,
        equipment: Option<&[Box<dyn Equipment>]>,
        occupancy: Option<&OccupancyProfile>,
    ) -> f64 {
        // Load building profile if not manually provided
        let (lighting_ref, equipment_ref, occupancy_ref) = match (lighting, equipment, occupancy) {
            (None, None, None) => {
                let profile = profiles::load_building_profile(self.building_type)?;
                (Some(&profile.lighting), Some(&profile.equipment), Some(&profile.occupancy))
            },
            _ => (lighting, equipment, occupancy),  // Use provided overrides
        };

        for step in 0..steps {
            let day_of_year = step / 24;
            let hour = step % 24;
            let day_type = holiday::get_day_type(day_of_year);

            // Calculate internal heat gains
            let internal_convective = 0.0;
            let internal_radiative_to_air = 0.0;
            let internal_radiative_to_mass = 0.0;

            if let Some(lighting) = lighting_ref {
                let hour_idx = day_type.as_hour_index() * 24 + hour;
                internal_convective += lighting.convective_gains(hour_idx);
                internal_radiative_to_mass += lighting.radiative_gains(hour_idx);
            }

            if let Some(equipment_list) = equipment_ref {
                for equipment in equipment_list {
                    let equipment_rad = equipment.radiative_gains(step);
                    internal_convective += equipment.convective_gains(step);
                    internal_radiative_to_air += equipment_rad * (1.0 - equipment.mass_coupling_factor());
                    internal_radiative_to_mass += equipment_rad * equipment.mass_coupling_factor();
                }
            }

            if let Some(occ) = occupancy_ref {
                internal_convective += occ.convective_heat_gains(step);
                internal_radiative_to_mass += occ.radiative_heat_gains(step);
            }

            // Add to energy balance (simplified 5R1C integration)
            // Ti: internal air temperature, Tm: thermal mass temperature
            // Air mass: zone air thermal capacity
            // Mass thermal cap: h_tr_ms coupling conductance
            self.temperatures[zone] += internal_convective / air_mass * dt;
            self.temperatures[zone] += internal_radiative_to_air / air_mass * dt;
            self.mass_temperatures[zone] += internal_radiative_to_mass / mass_thermal_cap * dt;

            // ... rest of physics (solar, HVAC, etc.)
        }

        total_energy
    }
}
```

**Building profile JSON schema:**
```json
{
  "version": "1.0",
  "profiles": {
    "office": {
      "lighting": {
        "power_density_w_m2": 10.0,
        "convective_fraction": 0.2,
        "radiative_fraction": 0.8,
        "schedule": {
          "type": "weekly",
          "pattern": "8-18 weekdays"
        }
      },
      "occupancy": {
        "max_occupancy": 100,
        "sensible_heat_per_person_w": 75.0,
        "latent_heat_per_person_w": 55.0,
        "schedule": {
          "type": "weekly",
          "pattern": "8-18 weekdays"
        }
      },
      "equipment": {
        "computers": {
          "count": 100,
          "rated_power_w": 150,
          "radiative_fraction": 0.3,
          "convective_fraction": 0.7,
          "mass_coupling_factor": 0.2,
          "schedule": { "type": "daily", "pattern": "8-18" }
        },
        "servers": {
          "count": 5,
          "rated_power_w": 500,
          "radiative_fraction": 0.5,
          "convective_fraction": 0.5,
          "mass_coupling_factor": 0.8,
          "schedule": { "type": "constant", "value": 1.0 }
        }
      }
    },
    "retail": { /* similar structure */ },
    "school": { /* similar structure */ }
  }
}
```

**Holiday calendar (simple approach):**
```rust
pub fn get_day_type(day_of_year: usize) -> DayType {
    // Simple US federal holidays (year-agnostic formula)
    // Returns DayType::Holiday for holiday dates
    // Returns DayType::Weekday for Mon-Fri non-holidays
    // Returns DayType::Weekend for Sat-Sun
}
```

**Test additions:**
- `test_weekly_schedule_factory()`: Verify office_hours() fills correct hours
- `test_day_type_override()`: Verify individual days override Weekday/Weekend
- `test_equipment_trait()`: Verify Equipment trait methods work for all types
- `test_mass_coupled_radiative()`: Verify split between air and mass
- `test_building_profile_loading()`: Verify JSON parses correctly
- `test_internal_loads_integration()`: Run solve_timesteps with loads vs without, compare energy
- `test_ashrae_validation()`: Run Cases 600-960 with internal loads, compare to reference

</specifics>

---

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. All decisions relate to weekly schedules, equipment models, ThermalModel integration, and building type defaults as defined in Phase 17 requirements (LOADS-01 through LOADS-04).

</deferred>

---

*Phase: 17-internal-loads*
*Context gathered: 2026-03-13*
