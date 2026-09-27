// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Unified Simulation Schema for Fluxion Building Energy Modeling.
//!
//! This module defines the canonical versioned schema for building energy simulations,
//! unifying geometry, constructions, schedules, weather, controls, and outputs
//! into a single contract for both CLI and Python pathways.
//!
//! # Schema Version
//!
//! The current schema version is `1.0`. All schema types are versioned to ensure
//! backward compatibility and clear migration paths when the schema evolves.
//!
//! # Core Types
//!
//! - [`SimulationSchema`]: Top-level container for all simulation data
//! - [`Geometry`]: Building geometry (zones, dimensions)
//! - [`ConstructionSet`]: Building envelope constructions
//! - [`ScheduleSet`]: Time-based schedules for occupancy, lighting, HVAC
//! - [`WeatherData`]: Weather data reference or inline data
//! - [`ControlSet`]: HVAC control configurations
//! - [`SimulationOutput`]: Simulation results
//!
//! # Example
//!
//! ```rust
//! use fluxion::api::schema::{
//!     SimulationSchema, SimulationSchemaV1, Geometry, ConstructionSet, ScheduleSet,
//!     WeatherData, ControlSet, SchemaVersion, SimulationOutput,
//! };
//!
//! // Create a minimal schema
//! let schema = SimulationSchema::V1(SimulationSchemaV1 {
//!     version: SchemaVersion::V1,
//!     metadata: Default::default(),
//!     geometry: Geometry::default(),
//!     constructions: ConstructionSet::default(),
//!     schedules: ScheduleSet::default(),
//!     weather: WeatherData::default(),
//!     controls: ControlSet::default(),
//!     output: SimulationOutput::default(),
//! });
//! ```

use serde::{Deserialize, Deserializer, Serialize};
use std::path::PathBuf;

use crate::sim::construction::ConstructionLayer;
use crate::sim::schedule::{DailySchedule, HVACSchedule};
use crate::weather::HourlyWeatherData;

/// Custom deserializer for the `path` field of `WeatherData::EpwFile`.
///
/// Gates inbound paths on `validate_epw_path` (Issue #2915) so that an
/// authenticated REST client cannot reach `EpwWeatherSource::from_file`
/// (which `std::fs::File::open`s the path with no canonicalization) by
/// pointing at an arbitrary server-readable file. CWE-22 closure.
fn deserialize_validated_epw_path<'de, D>(deserializer: D) -> Result<PathBuf, D::Error>
where
    D: Deserializer<'de>,
{
    let path = PathBuf::deserialize(deserializer)?;
    let as_str = path.to_string_lossy().into_owned();
    crate::weather::epw::validate_epw_path(&as_str).map_err(serde::de::Error::custom)?;
    Ok(path)
}

/// Schema version for forward compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchemaVersion {
    V1,
}

impl Default for SchemaVersion {
    fn default() -> Self {
        SchemaVersion::V1
    }
}

/// Metadata about the simulation schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchemaMetadata {
    pub name: String,
    pub description: String,
    pub author: Option<String>,
    pub created_at: Option<String>,
    pub schema_version: SchemaVersion,
}

impl Default for SchemaMetadata {
    fn default() -> Self {
        SchemaMetadata {
            name: "Untitled Simulation".to_string(),
            description: String::new(),
            author: None,
            created_at: None,
            schema_version: SchemaVersion::V1,
        }
    }
}

/// Zone geometry specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZoneGeometry {
    pub name: String,
    pub floor_area: f64,
    pub volume: f64,
    pub height: f64,
}

impl Default for ZoneGeometry {
    fn default() -> Self {
        ZoneGeometry {
            name: "Zone 1".to_string(),
            floor_area: 48.0,
            volume: 129.6,
            height: 2.7,
        }
    }
}

/// Building geometry specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Geometry {
    pub zones: Vec<ZoneGeometry>,
    pub total_floor_area: f64,
    pub total_volume: f64,
    pub number_of_floors: usize,
    pub floor_height: f64,
}

impl Default for Geometry {
    fn default() -> Self {
        Geometry {
            zones: vec![ZoneGeometry::default()],
            total_floor_area: 48.0,
            total_volume: 129.6,
            number_of_floors: 1,
            floor_height: 2.7,
        }
    }
}

/// Window specification within a construction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WindowSpec {
    pub window_area: f64,
    pub window_u_value: f64,
    pub window_shgc: f64,
}

impl Default for WindowSpec {
    fn default() -> Self {
        WindowSpec {
            window_area: 12.0,
            window_u_value: 1.5,
            window_shgc: 0.3,
        }
    }
}

/// Surface construction specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceConstruction {
    pub name: String,
    pub layers: Vec<ConstructionLayer>,
    pub window: Option<WindowSpec>,
}

impl Default for SurfaceConstruction {
    fn default() -> Self {
        SurfaceConstruction {
            name: "Default Wall".to_string(),
            layers: vec![
                ConstructionLayer::new("Plasterboard", 0.16, 950.0, 840.0, 0.012),
                ConstructionLayer::new("Fiberglass", 0.04, 12.0, 840.0, 0.066),
                ConstructionLayer::new("Wood siding", 0.14, 500.0, 1300.0, 0.009),
            ],
            window: Some(WindowSpec::default()),
        }
    }
}

/// Set of construction assemblies for a building.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstructionSet {
    pub wall: SurfaceConstruction,
    pub roof: SurfaceConstruction,
    pub floor: SurfaceConstruction,
    pub interzone: Option<SurfaceConstruction>,
}

impl Default for ConstructionSet {
    fn default() -> Self {
        ConstructionSet {
            wall: SurfaceConstruction::default(),
            roof: SurfaceConstruction::default(),
            floor: SurfaceConstruction::default(),
            interzone: None,
        }
    }
}

/// Schedule set for building operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScheduleSet {
    pub occupancy: DailySchedule,
    pub lighting: DailySchedule,
    pub hvac: HVACSchedule,
    pub infiltration: Option<DailySchedule>,
    /// Issue #4101 — lighting power density in W/m². When `None` (or
    /// non-positive) the run keeps the historical behaviour: the REST path
    /// applies a zero lighting schedule and no auto-loaded profile. Set to
    /// a positive value to apply `schedules.lighting` (hourly fractions
    /// 0-1) at this density and meter it as an end-use series.
    #[serde(default)]
    pub lighting_power_density_w_m2: Option<f64>,
    /// Issue #4101 — plug/process equipment applied to the run and metered
    /// as an end-use series. Empty keeps the historical behaviour (no
    /// equipment gains). Each entry becomes one
    /// [`crate::sim::equipment::Equipment`] via [`EquipmentSpec::build`].
    #[serde(default)]
    pub equipment: Vec<EquipmentSpec>,
}

impl Default for ScheduleSet {
    fn default() -> Self {
        ScheduleSet {
            occupancy: DailySchedule::weekly("Occupancy".to_string()),
            lighting: DailySchedule::weekly("Lighting".to_string()),
            hvac: HVACSchedule::constant_schedule(20.0, 24.0)
                .expect("constant_schedule on fresh daily schedules cannot fail"),
            infiltration: None,
            lighting_power_density_w_m2: None,
            equipment: Vec::new(),
        }
    }
}

/// Issue #4101 — schema-supplied plug/process equipment.
///
/// A minimal, hand-computable description of one equipment item. The REST
/// `/v1/simulate` path (and the CLI schema path) converts each spec into a
/// real [`crate::sim::equipment::Equipment`] with
/// [`EquipmentSpec::build`]; the item's thermal gains are applied through
/// `StepParameters` and its electric draw is metered per timestep via
/// `Equipment::power_at_hour`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EquipmentSpec {
    /// Equipment class: `"computer"`, `"server"`, or `"generic"`
    /// (case-insensitive).
    pub equipment_type: String,
    /// Rated electric power per unit, Watts.
    pub rated_power_w: f64,
    /// Number of units. Defaults to 1.
    #[serde(default = "default_equipment_count")]
    pub count: usize,
    /// Hourly utilization fractions (0-1), indexed by hour of day.
    /// Defaults to always-on.
    #[serde(default = "always_on_hourly_fractions")]
    pub hourly_fractions: [f64; 24],
    /// Fraction of the heat release that is radiative (0-1). Defaults to 0.3.
    #[serde(default = "default_radiative_fraction")]
    pub radiative_fraction: f64,
    /// Fraction of the heat release that is convective (0-1). Defaults to 0.7.
    #[serde(default = "default_convective_fraction")]
    pub convective_fraction: f64,
    /// Fraction of radiative heat absorbed by thermal mass (0-1).
    /// Defaults to 0.2.
    #[serde(default = "default_mass_coupling_factor")]
    pub mass_coupling_factor: f64,
}

fn default_equipment_count() -> usize {
    1
}

fn always_on_hourly_fractions() -> [f64; 24] {
    [1.0; 24]
}

fn default_radiative_fraction() -> f64 {
    0.3
}

fn default_convective_fraction() -> f64 {
    0.7
}

fn default_mass_coupling_factor() -> f64 {
    0.2
}

impl EquipmentSpec {
    /// Validate the spec, returning human-readable errors (empty = valid).
    /// Called from [`SimulationSchemaV1::validate`].
    pub fn validate(&self, index: usize) -> Vec<ValidationError> {
        let base = format!("schedules.equipment[{index}]");
        let mut errors = Vec::new();
        match self.equipment_type.to_ascii_lowercase().as_str() {
            "computer" | "server" | "generic" => {}
            other => errors.push(ValidationError::new(
                format!("{base}.equipment_type"),
                format!("unknown equipment type {other:?}"),
                "use \"computer\", \"server\", or \"generic\"".to_string(),
            )),
        }
        if self.rated_power_w.is_nan() || self.rated_power_w < 0.0 {
            errors.push(ValidationError::new(
                format!("{base}.rated_power_w"),
                format!("must be >= 0, got {}", self.rated_power_w),
                "set rated_power_w to the per-unit electric draw in Watts".to_string(),
            ));
        }
        if self.count == 0 {
            errors.push(ValidationError::new(
                format!("{base}.count"),
                "must be >= 1, got 0".to_string(),
                "set count to the number of installed units".to_string(),
            ));
        }
        for (h, &f) in self.hourly_fractions.iter().enumerate() {
            if !(0.0..=1.0).contains(&f) {
                errors.push(ValidationError::new(
                    format!("{base}.hourly_fractions[{h}]"),
                    format!("must be in [0, 1], got {f}"),
                    "set the hour's utilization fraction between 0 and 1".to_string(),
                ));
            }
        }
        let total = self.radiative_fraction + self.convective_fraction;
        if (total - 1.0).abs() > 1e-9 {
            errors.push(ValidationError::new(
                format!("{base}.radiative_fraction"),
                format!("radiative + convective fractions must sum to 1.0, got {total}"),
                "adjust radiative_fraction / convective_fraction to sum to 1.0".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.mass_coupling_factor) {
            errors.push(ValidationError::new(
                format!("{base}.mass_coupling_factor"),
                format!("must be in [0, 1], got {}", self.mass_coupling_factor),
                "set the mass coupling factor between 0 and 1".to_string(),
            ));
        }
        errors
    }

    /// Build the runtime equipment object. `id` names the item (the REST
    /// path generates `schema-equipment-{index}`).
    ///
    /// # Errors
    /// Returns `Err` for an unknown `equipment_type`.
    pub fn build(&self, id: String) -> Result<Box<dyn crate::sim::equipment::Equipment>, String> {
        use crate::sim::equipment::{ComputerEquipment, Equipment, GenericEquipment, ServerRack};

        let mut schedule = DailySchedule::new();
        for (hour, &fraction) in self.hourly_fractions.iter().enumerate() {
            schedule
                .set_hour(hour, fraction)
                .map_err(|e| format!("equipment {id}: invalid schedule: {e}"))?;
        }

        // Concrete fields are public; apply the spec's heat-split fractions
        // after the builder (builders only set id / power / count / schedule).
        match self.equipment_type.to_ascii_lowercase().as_str() {
            "computer" => {
                let mut item = ComputerEquipment::new(id, self.rated_power_w, self.count)
                    .with_schedule(schedule);
                item.radiative_fraction = self.radiative_fraction;
                item.convective_fraction = self.convective_fraction;
                item.mass_coupling_factor = self.mass_coupling_factor;
                Ok(Box::new(item) as Box<dyn Equipment>)
            }
            "server" => {
                let mut item =
                    ServerRack::new(id, self.rated_power_w, self.count).with_schedule(schedule);
                item.radiative_fraction = self.radiative_fraction;
                item.convective_fraction = self.convective_fraction;
                item.mass_coupling_factor = self.mass_coupling_factor;
                Ok(Box::new(item) as Box<dyn Equipment>)
            }
            "generic" => {
                let mut item = GenericEquipment::new(id, self.rated_power_w, self.count)
                    .with_schedule(schedule);
                item.radiative_fraction = self.radiative_fraction;
                item.convective_fraction = self.convective_fraction;
                item.mass_coupling_factor = self.mass_coupling_factor;
                Ok(Box::new(item) as Box<dyn Equipment>)
            }
            other => Err(format!(
                "equipment {id}: unknown equipment_type {other:?} \
                 (expected \"computer\", \"server\", or \"generic\")"
            )),
        }
    }
}

/// Weather data specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WeatherData {
    /// Reference to an external EPW file.
    ///
    /// Deserialization is gated by `validate_epw_path` (Issue #2915) so an
    /// authenticated REST client cannot point at arbitrary server-readable
    /// files via `WeatherData::EpwFile { path }` on `/v1/simulate` or
    /// `/v1/campaign/*` requests.
    #[serde(rename = "epw")]
    EpwFile {
        #[serde(deserialize_with = "deserialize_validated_epw_path")]
        path: PathBuf,
    },

    /// Reference to an embedded TMY location.
    #[serde(rename = "tmy")]
    TmyLocation { location: String },

    /// Inline hourly weather data.
    #[serde(rename = "inline")]
    Inline { hourly_data: Vec<HourlyWeatherData> },
}

impl Default for WeatherData {
    fn default() -> Self {
        WeatherData::TmyLocation {
            location: "Denver, CO".to_string(),
        }
    }
}

/// HVAC control configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControlConfig {
    pub heating_setpoint: f64,
    pub cooling_setpoint: f64,
    /// Issue #4103 — setpoint tolerance (°C) used ONLY for unmet-hours
    /// reporting. Defaults to [`UNMET_HOURS_TOLERANCE_DEFAULT_C`] (0.2 °C,
    /// matching EnergyPlus "Time Setpoint Not Met", EnergyPlus Input
    /// Output Reference, System Summary table) when the schema leaves the
    /// field unset; explicit values are honored unchanged. This field is
    /// not a physics control deadband — the HVAC controllers keep their
    /// own 0.5 °C default.
    #[serde(default = "default_unmet_hours_tolerance")]
    pub deadband_tolerance: f64,
    pub heating_capacity: f64,
    pub cooling_capacity: f64,
}

/// Issue #4103 — default setpoint tolerance for unmet-hours reporting:
/// 0.2 °C, matching EnergyPlus "Time Setpoint Not Met" (EnergyPlus Input
/// Output Reference, System Summary table, which applies a 0.2 °C
/// tolerance band when counting occupied hours the setpoint was not met).
pub const UNMET_HOURS_TOLERANCE_DEFAULT_C: f64 = 0.2;

fn default_unmet_hours_tolerance() -> f64 {
    UNMET_HOURS_TOLERANCE_DEFAULT_C
}

impl Default for ControlConfig {
    fn default() -> Self {
        ControlConfig {
            heating_setpoint: 20.0,
            cooling_setpoint: 24.0,
            // Issue #4103: unset schemas default to the EnergyPlus
            // "Time Setpoint Not Met" 0.2 °C tolerance, not the 0.5 °C
            // HVAC control deadband.
            deadband_tolerance: UNMET_HOURS_TOLERANCE_DEFAULT_C,
            heating_capacity: 100_000.0,
            cooling_capacity: 100_000.0,
        }
    }
}

/// Set of control configurations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControlSet {
    pub zone_control: ControlConfig,
    pub global_control: Option<ControlConfig>,
}

impl Default for ControlSet {
    fn default() -> Self {
        ControlSet {
            zone_control: ControlConfig::default(),
            global_control: None,
        }
    }
}

/// Simulation output results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationOutput {
    pub eui: f64,
    pub total_energy: f64,
    /// Peak heating demand observed during the run, in Watts.
    pub peak_heating_load: f64,
    /// Peak cooling demand observed during the run, in Watts.
    pub peak_cooling_load: f64,
    pub heating_energy: f64,
    pub cooling_energy: f64,
    pub zone_temperatures: Option<Vec<f64>>,
    /// Issue #763 — full hourly zone temperature profiles.
    /// Format: [num_zones][8760] hourly temperatures in °C.
    pub hourly_zone_temperatures: Option<Vec<Vec<f64>>>,
    /// Issue #3305 — the zone solver that ACTUALLY executed, derived from
    /// the step dispatcher's real per-step outcome (gauge success vs.
    /// 5R1C/9R4C fall-through), not from the requested selector. Populated
    /// by the REST `/v1/simulate` run on success; schema-embedded output
    /// templates leave it `None`, and it is omitted from serialisation when
    /// `None` so existing wire shapes are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_solver: Option<String>,
    /// Issue #3988 — unmet heating hours (occupied-only): occupied hours
    /// where a zone's temperature fell below
    /// `heating_setpoint - deadband_tolerance`, summed across zones. The
    /// tolerance defaults to 0.2 °C when the schema leaves
    /// `deadband_tolerance` unset (EnergyPlus "Time Setpoint Not Met"
    /// parity; EnergyPlus Input Output Reference, System Summary table).
    /// See [`SimulationOutput::unmet_hours`].
    #[serde(default)]
    pub unmet_heating_hours: f64,
    /// Issue #3988 — unmet cooling hours (occupied-only): occupied hours
    /// where a zone's temperature rose above
    /// `cooling_setpoint + deadband_tolerance`, summed across zones.
    /// Tolerance default as for `unmet_heating_hours`. See
    /// [`SimulationOutput::unmet_hours`].
    #[serde(default)]
    pub unmet_cooling_hours: f64,
    /// Issue #4103 — unmet heating hours over ALL hours (per ASHRAE 90.1
    /// Appendix G §G3.1.2.2): same setpoint-deviation test as
    /// `unmet_heating_hours` but evaluated for every timestep, not just
    /// occupied hours. The 90.1 Performance Rating Method caps these at
    /// 300 hours and requires proposed-design unmet hours ≤ baseline + 50.
    /// See [`SimulationOutput::unmet_hours_all_hours`].
    #[serde(default)]
    pub unmet_heating_hours_all_hours: f64,
    /// Issue #4103 — unmet cooling hours over ALL hours (ASHRAE 90.1
    /// Appendix G §G3.1.2.2). See `unmet_heating_hours_all_hours`.
    #[serde(default)]
    pub unmet_cooling_hours_all_hours: f64,
    /// Issue #4101 — timestep-indexed heating energy, kWh per timestep.
    /// One entry per solver timestep; omitted when the run did not record
    /// end-use metering so existing wire shapes are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hourly_heating_kwh: Option<Vec<f64>>,
    /// Issue #4101 — timestep-indexed cooling energy, kWh per timestep.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hourly_cooling_kwh: Option<Vec<f64>>,
    /// Issue #4101 — timestep-indexed lighting electric energy, kWh per
    /// timestep.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hourly_lighting_kwh: Option<Vec<f64>>,
    /// Issue #4101 — timestep-indexed equipment electric energy, kWh per
    /// timestep.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hourly_equipment_kwh: Option<Vec<f64>>,
    /// Issue #4102 — monthly heating energy, kWh per month. Outer vec is
    /// the 0-based year index, inner vec is 12 calendar months (Jan = 0).
    /// Pure post-processing of [`SimulationOutput::hourly_heating_kwh`];
    /// omitted when the run did not record end-use metering.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monthly_heating_kwh: Option<Vec<Vec<f64>>>,
    /// Issue #4102 — monthly heating peak demand, max timestep-average kW
    /// per month. Same `[year][month]` shape as `monthly_heating_kwh`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monthly_heating_peak_kw: Option<Vec<Vec<f64>>>,
    /// Issue #4102 — monthly cooling energy, kWh per month (`[year][month]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monthly_cooling_kwh: Option<Vec<Vec<f64>>>,
    /// Issue #4102 — monthly cooling peak demand, kW (`[year][month]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monthly_cooling_peak_kw: Option<Vec<Vec<f64>>>,
    /// Issue #4102 — monthly lighting energy, kWh per month (`[year][month]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monthly_lighting_kwh: Option<Vec<Vec<f64>>>,
    /// Issue #4102 — monthly lighting peak demand, kW (`[year][month]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monthly_lighting_peak_kw: Option<Vec<Vec<f64>>>,
    /// Issue #4102 — monthly equipment energy, kWh per month (`[year][month]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monthly_equipment_kwh: Option<Vec<Vec<f64>>>,
    /// Issue #4102 — monthly equipment peak demand, kW (`[year][month]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monthly_equipment_peak_kw: Option<Vec<Vec<f64>>>,
}

impl Default for SimulationOutput {
    fn default() -> Self {
        SimulationOutput {
            eui: 0.0,
            total_energy: 0.0,
            peak_heating_load: 0.0,
            peak_cooling_load: 0.0,
            heating_energy: 0.0,
            cooling_energy: 0.0,
            zone_temperatures: None,
            hourly_zone_temperatures: None,
            effective_solver: None,
            unmet_heating_hours: 0.0,
            unmet_cooling_hours: 0.0,
            unmet_heating_hours_all_hours: 0.0,
            unmet_cooling_hours_all_hours: 0.0,
            hourly_heating_kwh: None,
            hourly_cooling_kwh: None,
            hourly_lighting_kwh: None,
            hourly_equipment_kwh: None,
            monthly_heating_kwh: None,
            monthly_heating_peak_kw: None,
            monthly_cooling_kwh: None,
            monthly_cooling_peak_kw: None,
            monthly_lighting_kwh: None,
            monthly_lighting_peak_kw: None,
            monthly_equipment_kwh: None,
            monthly_equipment_peak_kw: None,
        }
    }
}

/// Issue #4102 — monthly end-use summary produced by
/// [`SimulationOutput::monthly_end_use_summary`].
///
/// Both arrays are `[end_use][year][month]` with end uses in
/// heating/cooling/lighting/equipment order and months Jan = 0. `kwh`
/// holds the month's energy sum; `peak_kw` holds the month's maximum
/// timestep-average demand.
#[derive(Debug, Clone, Default)]
pub struct MonthlyEndUseSummary {
    pub kwh: [Vec<Vec<f64>>; 4],
    pub peak_kw: [Vec<Vec<f64>>; 4],
}

impl SimulationOutput {
    /// Issue #4102 — monthly end-use summary: 12 bins per simulated year
    /// per end use, derived from the timestep-indexed hourly series
    /// (#4101). Pure post-processing — no sim-loop involvement.
    ///
    /// Each input series is kWh per timestep; `dt_seconds` is the run's
    /// constant timestep duration. Returns `None` when any series is
    /// missing or empty (all-or-nothing, mirroring the #4101 exposure).
    /// Otherwise the arrays are `[end_use][year][month]` in
    /// heating/cooling/lighting/equipment order, with timestep 0 starting
    /// Jan 1 00:00 of the weather year (non-leap, per ASHRAE 140
    /// convention).
    pub fn monthly_end_use_summary(
        heating_kwh: Option<Vec<f64>>,
        cooling_kwh: Option<Vec<f64>>,
        lighting_kwh: Option<Vec<f64>>,
        equipment_kwh: Option<Vec<f64>>,
        dt_seconds: f64,
    ) -> Option<MonthlyEndUseSummary> {
        use crate::validation::report::BenchmarkReport;
        let series = [heating_kwh?, cooling_kwh?, lighting_kwh?, equipment_kwh?];
        let mut kwh: [Vec<Vec<f64>>; 4] = Default::default();
        let mut peak_kw: [Vec<Vec<f64>>; 4] = Default::default();
        for (u, s) in series.iter().enumerate() {
            let bins = BenchmarkReport::calculate_monthly_end_use(s, dt_seconds, false);
            if bins.is_empty() {
                return None;
            }
            let years = bins.len().div_ceil(12);
            let mut kwh_years = vec![vec![0.0; 12]; years];
            let mut peak_years = vec![vec![0.0; 12]; years];
            for (flat, bin) in bins.iter().enumerate() {
                kwh_years[flat / 12][flat % 12] = bin.kwh;
                peak_years[flat / 12][flat % 12] = bin.peak_kw;
            }
            kwh[u] = kwh_years;
            peak_kw[u] = peak_years;
        }
        Some(MonthlyEndUseSummary { kwh, peak_kw })
    }

    /// Compute unmet heating/cooling hours from hourly zone temperatures.
    ///
    /// Issue #3988. `hourly_temps` is zone-major (`[zone][timestep]`, one
    /// step per hour) as returned by the thermal model's
    /// `get_hourly_temperatures`. An hour counts as occupied when the
    /// occupancy schedule's value for that hour-of-day exceeds 0.05 (the
    /// weekday profile is used for weekly schedules). A zone-hour is
    /// unmet-heating when its temperature falls below
    /// `heating_setpoint - tolerance`, unmet-cooling when it rises above
    /// `cooling_setpoint + tolerance`.
    ///
    /// Returns `(unmet_heating_hours, unmet_cooling_hours)` summed across
    /// zones. Shared by the REST path and the CLI direct path.
    pub fn unmet_hours(
        hourly_temps: &[Vec<f64>],
        occupancy: &DailySchedule,
        heating_setpoint: f64,
        cooling_setpoint: f64,
        tolerance: f64,
    ) -> (f64, f64) {
        Self::unmet_hours_impl(
            hourly_temps,
            Some(occupancy),
            heating_setpoint,
            cooling_setpoint,
            tolerance,
        )
    }

    /// Compute unmet heating/cooling hours over ALL timesteps.
    ///
    /// Issue #4103. Same setpoint-deviation test as [`Self::unmet_hours`]
    /// but evaluated for every timestep instead of only occupied ones, per
    /// ASHRAE 90.1 Appendix G §G3.1.2.2 (unmet load hours). The 90.1
    /// Performance Rating Method caps these at 300 hours and requires the
    /// proposed design to stay within baseline + 50 unmet hours.
    ///
    /// Returns `(unmet_heating_hours_all_hours,
    /// unmet_cooling_hours_all_hours)` summed across zones.
    pub fn unmet_hours_all_hours(
        hourly_temps: &[Vec<f64>],
        heating_setpoint: f64,
        cooling_setpoint: f64,
        tolerance: f64,
    ) -> (f64, f64) {
        Self::unmet_hours_impl(
            hourly_temps,
            None,
            heating_setpoint,
            cooling_setpoint,
            tolerance,
        )
    }

    /// Shared implementation for [`Self::unmet_hours`] (occupied-only,
    /// `occupancy = Some`) and [`Self::unmet_hours_all_hours`] (all hours,
    /// `occupancy = None`).
    fn unmet_hours_impl(
        hourly_temps: &[Vec<f64>],
        occupancy: Option<&DailySchedule>,
        heating_setpoint: f64,
        cooling_setpoint: f64,
        tolerance: f64,
    ) -> (f64, f64) {
        let mut unmet_heating = 0.0;
        let mut unmet_cooling = 0.0;
        for zone_temps in hourly_temps {
            for (t, &temp) in zone_temps.iter().enumerate() {
                if let Some(occ) = occupancy {
                    if occ.value(t % 24) <= 0.05 {
                        continue;
                    }
                }
                if temp < heating_setpoint - tolerance {
                    unmet_heating += 1.0;
                } else if temp > cooling_setpoint + tolerance {
                    unmet_cooling += 1.0;
                }
            }
        }
        (unmet_heating, unmet_cooling)
    }
}

/// Version 1 of the simulation schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationSchemaV1 {
    pub version: SchemaVersion,
    pub metadata: SchemaMetadata,
    pub geometry: Geometry,
    pub constructions: ConstructionSet,
    pub schedules: ScheduleSet,
    pub weather: WeatherData,
    pub controls: ControlSet,
    pub output: SimulationOutput,
}

/// Unified simulation schema container with version support.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SimulationSchema {
    V1(SimulationSchemaV1),
}

impl SimulationSchema {
    pub fn v1(schema: SimulationSchemaV1) -> Self {
        SimulationSchema::V1(schema)
    }

    pub fn version(&self) -> SchemaVersion {
        match self {
            SimulationSchema::V1(s) => s.version,
        }
    }
}

/// A single actionable input-validation error.
///
/// Issue #3993: the schema used to fail with panics or silently clamp bad
/// values (`.max(1.0)`). Each error names the offending field (JSON-path
/// style), states the problem, and tells the modeler how to fix it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationError {
    /// Field path, e.g. `geometry.zones[0].floor_area`.
    pub field: String,
    /// What is wrong, e.g. `must be > 0, got -5.0`.
    pub problem: String,
    /// How to fix it, e.g. `set floor_area to the zone's floor area in m²`.
    pub fix: String,
}

impl ValidationError {
    fn new(field: impl Into<String>, problem: impl Into<String>, fix: impl Into<String>) -> Self {
        ValidationError {
            field: field.into(),
            problem: problem.into(),
            fix: fix.into(),
        }
    }
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}. Fix: {}", self.field, self.problem, self.fix)
    }
}

impl SimulationSchemaV1 {
    /// Validate the schema, returning every actionable error found.
    ///
    /// An empty vec means the schema is valid. Callers (REST
    /// `/v1/simulate`, the `fluxion` CLI direct path) must reject the input
    /// when this is non-empty rather than simulating with clamped values.
    pub fn validate(&self) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        // --- Geometry ---
        if self.geometry.zones.is_empty() {
            errors.push(ValidationError::new(
                "geometry.zones",
                "must contain at least one zone, got 0",
                "add a zone object with name, floor_area (m²), volume (m³), height (m)",
            ));
        }
        for (i, zone) in self.geometry.zones.iter().enumerate() {
            let base = format!("geometry.zones[{i}]");
            if zone.floor_area <= 0.0 {
                errors.push(ValidationError::new(
                    format!("{base}.floor_area"),
                    format!("must be > 0, got {}", zone.floor_area),
                    "set floor_area to the zone's floor area in m²",
                ));
            }
            if zone.height <= 0.0 {
                errors.push(ValidationError::new(
                    format!("{base}.height"),
                    format!("must be > 0, got {}", zone.height),
                    "set height to the zone's floor-to-ceiling height in m",
                ));
            }
            if zone.volume <= 0.0 {
                errors.push(ValidationError::new(
                    format!("{base}.volume"),
                    format!("must be > 0, got {}", zone.volume),
                    "set volume to the zone's air volume in m³ (usually floor_area × height)",
                ));
            }
        }
        if self.geometry.total_floor_area <= 0.0 {
            errors.push(ValidationError::new(
                "geometry.total_floor_area",
                format!("must be > 0, got {}", self.geometry.total_floor_area),
                "set total_floor_area to the sum of zone floor areas in m²",
            ));
        }

        // --- Constructions ---
        for (label, sc) in [
            ("wall", &self.constructions.wall),
            ("roof", &self.constructions.roof),
            ("floor", &self.constructions.floor),
        ] {
            let base = format!("constructions.{label}");
            if sc.layers.is_empty() {
                errors.push(ValidationError::new(
                    format!("{base}.layers"),
                    "must contain at least one material layer, got 0",
                    "add at least one layer with conductivity (W/m·K), density (kg/m³), specific_heat (J/kg·K), thickness (m)",
                ));
            }
            for (j, layer) in sc.layers.iter().enumerate() {
                let lbase = format!("{base}.layers[{j}]");
                for (prop, v) in [
                    ("thickness", layer.thickness),
                    ("conductivity", layer.conductivity),
                    ("density", layer.density),
                    ("specific_heat", layer.specific_heat),
                ] {
                    if v <= 0.0 {
                        errors.push(ValidationError::new(
                            format!("{lbase}.{prop}"),
                            format!("must be > 0, got {v}"),
                            format!(
                                "set {prop} to a positive physical value for '{}'",
                                layer.name
                            ),
                        ));
                    }
                }
            }
            if let Some(w) = &sc.window {
                let wbase = format!("{base}.window");
                if w.window_area <= 0.0 {
                    errors.push(ValidationError::new(
                        format!("{wbase}.window_area"),
                        format!("must be > 0, got {}", w.window_area),
                        "set window_area to the total window area in m², or remove the window object for no glazing",
                    ));
                }
                if w.window_u_value <= 0.0 {
                    errors.push(ValidationError::new(
                        format!("{wbase}.window_u_value"),
                        format!("must be > 0, got {}", w.window_u_value),
                        "set window_u_value to the glazing U-value in W/m²·K (typical: 1.0–3.0)",
                    ));
                }
                if !(0.0 < w.window_shgc && w.window_shgc <= 1.0) {
                    errors.push(ValidationError::new(
                        format!("{wbase}.window_shgc"),
                        format!("must be in (0, 1], got {}", w.window_shgc),
                        "set window_shgc to the solar heat gain coefficient as a fraction (typical: 0.2–0.8)",
                    ));
                }
            }
        }

        // --- Controls ---
        let cbase = "controls.zone_control";
        let heating = self.controls.zone_control.heating_setpoint;
        let cooling = self.controls.zone_control.cooling_setpoint;
        if heating >= cooling {
            errors.push(ValidationError::new(
                format!("{cbase}.heating_setpoint / {cbase}.cooling_setpoint"),
                format!("heating_setpoint ({heating}) must be < cooling_setpoint ({cooling})"),
                "set heating_setpoint below cooling_setpoint (typical: 20.0 / 24.0 °C)",
            ));
        }
        for (prop, v) in [
            (
                "heating_capacity",
                self.controls.zone_control.heating_capacity,
            ),
            (
                "cooling_capacity",
                self.controls.zone_control.cooling_capacity,
            ),
        ] {
            if v <= 0.0 {
                errors.push(ValidationError::new(
                    format!("{cbase}.{prop}"),
                    format!("must be > 0, got {v}"),
                    format!("set {prop} to the system capacity in W"),
                ));
            }
        }

        // --- Internal loads (Issue #4101) ---
        if let Some(density) = self.schedules.lighting_power_density_w_m2 {
            if density < 0.0 {
                errors.push(ValidationError::new(
                    "schedules.lighting_power_density_w_m2",
                    format!("must be >= 0, got {density}"),
                    "set lighting_power_density_w_m2 to the lighting power density in W/m² (omit for no lighting)",
                ));
            }
            if density.is_nan() {
                errors.push(ValidationError::new(
                    "schedules.lighting_power_density_w_m2",
                    "must not be NaN".to_string(),
                    "set lighting_power_density_w_m2 to the lighting power density in W/m² (omit for no lighting)",
                ));
            }
        }
        for (i, spec) in self.schedules.equipment.iter().enumerate() {
            errors.extend(spec.validate(i));
        }

        // --- Weather ---
        if let WeatherData::EpwFile { path } = &self.weather {
            if path.as_os_str().is_empty() {
                errors.push(ValidationError::new(
                    "weather.path",
                    "epw file path must not be empty",
                    "set weather to {\"type\": \"epw\", \"path\": \"<path to .epw>\"}",
                ));
            }
        }

        errors
    }
}

impl Default for SimulationSchemaV1 {
    fn default() -> Self {
        SimulationSchemaV1 {
            version: SchemaVersion::V1,
            metadata: SchemaMetadata::default(),
            geometry: Geometry::default(),
            constructions: ConstructionSet::default(),
            schedules: ScheduleSet::default(),
            weather: WeatherData::default(),
            controls: ControlSet::default(),
            output: SimulationOutput::default(),
        }
    }
}

impl Default for SimulationSchema {
    fn default() -> Self {
        SimulationSchema::V1(SimulationSchemaV1::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unmet_hours_all_comfortable() {
        let occupancy = DailySchedule::weekly("occ".to_string()).office_hours();
        // 48 hours, 1 zone, always 21°C within the 20/24 band.
        let hourly = vec![vec![21.0; 48]];
        let (h, c) = SimulationOutput::unmet_hours(&hourly, &occupancy, 20.0, 24.0, 0.5);
        assert_eq!((h, c), (0.0, 0.0));
    }

    #[test]
    fn test_unmet_hours_counts_occupied_only() {
        let occupancy = DailySchedule::weekly("occ".to_string()).office_hours();
        // Zone is cold (15°C) for all 48 hours, but only office hours count.
        let hourly = vec![vec![15.0; 48]];
        let (h, c) = SimulationOutput::unmet_hours(&hourly, &occupancy, 20.0, 24.0, 0.5);
        // office_hours sets 8..=17 (10 hours/day) on weekdays; value() falls
        // back to Monday's profile, so 10 occupied hours/day × 2 days.
        assert_eq!(h, 20.0);
        assert_eq!(c, 0.0);
    }

    #[test]
    fn test_unmet_hours_cooling_and_tolerance() {
        let occupancy = DailySchedule::weekly("occ".to_string()).office_hours();
        // 30°C during occupied hours -> unmet cooling; 24.2°C is inside the
        // cooling setpoint + 0.5 tolerance band and must not count.
        let mut temps = vec![21.0; 48];
        for t in [9, 10, 33] {
            temps[t] = 30.0;
        }
        temps[11] = 24.2;
        let hourly = vec![temps];
        let (h, c) = SimulationOutput::unmet_hours(&hourly, &occupancy, 20.0, 24.0, 0.5);
        assert_eq!(h, 0.0);
        assert_eq!(c, 3.0);
    }

    #[test]
    fn test_unmet_hours_sums_across_zones() {
        let occupancy = DailySchedule::weekly("occ".to_string()).office_hours();
        let hourly = vec![vec![15.0; 48], vec![15.0; 48]];
        let (h, _) = SimulationOutput::unmet_hours(&hourly, &occupancy, 20.0, 24.0, 0.5);
        assert_eq!(h, 40.0);
    }

    #[test]
    fn test_unmet_hours_tolerance_defaults_to_0_2_c() {
        // Issue #4103: an unset schema deadband defaults to the EnergyPlus
        // "Time Setpoint Not Met" 0.2 °C tolerance, not 0.5 °C.
        assert_eq!(
            ControlConfig::default().deadband_tolerance,
            UNMET_HOURS_TOLERANCE_DEFAULT_C
        );
        assert_eq!(UNMET_HOURS_TOLERANCE_DEFAULT_C, 0.2);
        // A schema JSON that omits deadband_tolerance must still
        // deserialize, picking up the 0.2 °C default.
        let cfg: ControlConfig = serde_json::from_str(
            r#"{"heating_setpoint":20.0,"cooling_setpoint":24.0,
                "heating_capacity":100000.0,"cooling_capacity":100000.0}"#,
        )
        .expect("unset deadband_tolerance must deserialize");
        assert_eq!(cfg.deadband_tolerance, 0.2);
        // An explicit value is honored unchanged.
        let cfg: ControlConfig = serde_json::from_str(
            r#"{"heating_setpoint":20.0,"cooling_setpoint":24.0,
                "deadband_tolerance":0.5,
                "heating_capacity":100000.0,"cooling_capacity":100000.0}"#,
        )
        .expect("explicit deadband_tolerance must deserialize");
        assert_eq!(cfg.deadband_tolerance, 0.5);
    }

    #[test]
    fn test_unmet_hours_all_hours_diverges_on_unoccupied_night() {
        // Issue #4103: a deviation during unoccupied night hours counts in
        // the all-hours variant (ASHRAE 90.1 G3.1.2.2) but not in the
        // occupied-only variant.
        let occupancy = DailySchedule::weekly("occ".to_string()).office_hours();
        // 24 h trace: comfortable 21 °C all day except midnight (t=0),
        // which is unoccupied and drops to 15 °C.
        let mut temps = vec![21.0; 24];
        temps[0] = 15.0;
        let hourly = vec![temps];
        let tol = UNMET_HOURS_TOLERANCE_DEFAULT_C;
        let (occ_h, occ_c) = SimulationOutput::unmet_hours(&hourly, &occupancy, 20.0, 24.0, tol);
        let (all_h, all_c) = SimulationOutput::unmet_hours_all_hours(&hourly, 20.0, 24.0, tol);
        assert_eq!((occ_h, occ_c), (0.0, 0.0));
        assert_eq!((all_h, all_c), (1.0, 0.0));
    }

    #[test]
    fn test_unmet_hours_all_hours_sums_across_zones() {
        // Issue #4103: the all-hours variant also sums across zones, with
        // the same heating/cooling deviation test.
        let hourly = vec![vec![15.0; 24], vec![30.0; 24]];
        let tol = UNMET_HOURS_TOLERANCE_DEFAULT_C;
        let (h, c) = SimulationOutput::unmet_hours_all_hours(&hourly, 20.0, 24.0, tol);
        assert_eq!(h, 24.0);
        assert_eq!(c, 24.0);
        // A fully comfortable trace reports zero in both conventions.
        let comfortable = vec![vec![21.0; 24]];
        assert_eq!(
            SimulationOutput::unmet_hours_all_hours(&comfortable, 20.0, 24.0, tol),
            (0.0, 0.0)
        );
    }

    #[test]
    fn test_validate_default_schema_is_valid() {
        // The default schema must pass validation — it is the baseline all
        // other tests and examples build from.
        let schema = SimulationSchemaV1::default();
        assert!(schema.validate().is_empty());
    }

    #[test]
    fn test_validate_catches_bad_zone_geometry() {
        let mut schema = SimulationSchemaV1::default();
        schema.geometry.zones[0].floor_area = -5.0;
        schema.geometry.zones[0].height = 0.0;
        let errors = schema.validate();
        let fields: Vec<&str> = errors.iter().map(|e| e.field.as_str()).collect();
        assert!(fields.contains(&"geometry.zones[0].floor_area"));
        assert!(fields.contains(&"geometry.zones[0].height"));
        // Every error must name the problem and a fix.
        for e in &errors {
            assert!(!e.problem.is_empty(), "missing problem for {}", e.field);
            assert!(!e.fix.is_empty(), "missing fix for {}", e.field);
        }
    }

    #[test]
    fn test_validate_catches_empty_zones_and_bad_setpoints() {
        let mut schema = SimulationSchemaV1::default();
        schema.geometry.zones.clear();
        schema.controls.zone_control.heating_setpoint = 26.0;
        schema.controls.zone_control.cooling_setpoint = 24.0;
        let errors = schema.validate();
        let fields: Vec<&str> = errors.iter().map(|e| e.field.as_str()).collect();
        assert!(fields.contains(&"geometry.zones"));
        assert!(fields.iter().any(|f| f.contains("heating_setpoint")));
    }

    #[test]
    fn test_validate_catches_bad_construction_layer() {
        let mut schema = SimulationSchemaV1::default();
        schema.constructions.wall.layers = vec![crate::sim::construction::ConstructionLayer {
            name: "bad".to_string(),
            conductivity: 0.04,
            density: 12.0,
            specific_heat: 840.0,
            thickness: 0.0,
            absorptance: 0.7,
            emissivity: 0.9,
        }];
        let errors = schema.validate();
        assert!(errors
            .iter()
            .any(|e| e.field == "constructions.wall.layers[0].thickness"));
    }

    #[test]
    fn test_validate_catches_bad_window_shgc() {
        let mut schema = SimulationSchemaV1::default();
        if let Some(w) = &mut schema.constructions.wall.window {
            w.window_shgc = 1.5;
        }
        let errors = schema.validate();
        assert!(errors
            .iter()
            .any(|e| e.field == "constructions.wall.window.window_shgc"));
    }

    #[test]
    fn test_schema_version_default() {
        let version = SchemaVersion::default();
        assert_eq!(version, SchemaVersion::V1);
    }

    #[test]
    fn test_geometry_default() {
        let geometry = Geometry::default();
        assert_eq!(geometry.zones.len(), 1);
        assert_eq!(geometry.total_floor_area, 48.0);
    }

    #[test]
    fn test_construction_set_default() {
        let construction = SurfaceConstruction::default();
        assert_eq!(construction.layers.len(), 3);
        assert!(construction.window.is_some());
    }

    #[test]
    fn test_weather_data_default() {
        let weather = WeatherData::default();
        match weather {
            WeatherData::TmyLocation { location } => {
                assert_eq!(location, "Denver, CO");
            }
            _ => panic!("Expected TmyLocation variant"),
        }
    }

    #[test]
    fn test_control_config_default() {
        let control = ControlConfig::default();
        assert_eq!(control.heating_setpoint, 20.0);
        assert_eq!(control.cooling_setpoint, 24.0);
    }

    #[test]
    fn test_simulation_schema_v1_default() {
        let schema = SimulationSchemaV1::default();
        assert_eq!(schema.version, SchemaVersion::V1);
        assert_eq!(schema.geometry.zones.len(), 1);
    }

    #[test]
    fn test_simulation_schema_default() {
        let schema = SimulationSchema::default();
        assert_eq!(schema.version(), SchemaVersion::V1);
    }

    #[test]
    fn test_schema_serialization() {
        let schema = SimulationSchema::V1(SimulationSchemaV1::default());
        let json = serde_json::to_string(&schema).unwrap();
        let deserialized: SimulationSchema = serde_json::from_str(&json).unwrap();
        assert_eq!(schema, deserialized);
    }

    #[test]
    fn test_zone_geometry_serialization() {
        let zone = ZoneGeometry::default();
        let json = serde_json::to_string(&zone).unwrap();
        let deserialized: ZoneGeometry = serde_json::from_str(&json).unwrap();
        assert_eq!(zone, deserialized);
    }

    #[test]
    fn test_construction_layer_serialization() {
        let layer = ConstructionLayer::new("Test", 0.04, 12.0, 840.0, 0.066);
        let json = serde_json::to_string(&layer).unwrap();
        let deserialized: ConstructionLayer = serde_json::from_str(&json).unwrap();
        assert_eq!(layer.name, deserialized.name);
        assert_eq!(layer.conductivity, deserialized.conductivity);
    }

    #[test]
    fn test_hvac_schedule_serialization() {
        let schedule = HVACSchedule::constant_schedule(20.0, 24.0).unwrap();
        let json = serde_json::to_string(&schedule).unwrap();
        let deserialized: HVACSchedule = serde_json::from_str(&json).unwrap();
        assert_eq!(
            schedule.heating_setpoint(0),
            deserialized.heating_setpoint(0)
        );
    }

    #[test]
    fn test_simulation_output_serialization() {
        let output = SimulationOutput::default();
        let json = serde_json::to_string(&output).unwrap();
        let deserialized: SimulationOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(output.eui, deserialized.eui);
    }

    #[test]
    fn test_schema_metadata_with_author() {
        let metadata = SchemaMetadata {
            name: "Test Schema".to_string(),
            description: "A test schema".to_string(),
            author: Some("Test Author".to_string()),
            created_at: Some("2026-04-17".to_string()),
            schema_version: SchemaVersion::V1,
        };
        let json = serde_json::to_string(&metadata).unwrap();
        let deserialized: SchemaMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(metadata.author, deserialized.author);
    }

    // ===== Issue #2915 — EpwFile path validation on deserialize =====
    //
    // `WeatherData::EpwFile { path }` is the inbound payload of every
    // `/v1/simulate` and `/v1/campaign/*` request; a missing validation
    // gate would let an authenticated REST client reach
    // `EpwWeatherSource::from_file` with `std::fs::File::open` — i.e.
    // arbitrary server-readable file read (CWE-22). These tests assert
    // that `serde_json::from_str` refuses `/etc/passwd` and round-trips a
    // real `.epw` file inside the `FLUXION_EPW_DIR` allow-list.
    //
    // `FLUXION_EPW_DIR` is the process-wide env var read by
    // `validate_epw_path`; a `Mutex` serialises every test that mutates
    // it so parallel `cargo test` threads cannot stomp on each other.

    /// Shared mutex serialising every test in this module that mutates
    /// `FLUXION_EPW_DIR`. Without it, parallel `cargo test` threads would
    /// race on the env var and produce flaky failures.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Helper: set `FLUXION_EPW_DIR` to the given path for the duration
    /// of the closure. Returns whatever the closure returns. Any prior
    /// value of `FLUXION_EPW_DIR` is restored on scope exit (success or
    /// panic) so tests cannot leak env state into siblings.
    fn with_epw_dir<F: FnOnce() -> R, R>(dir: &std::path::Path, f: F) -> R {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let previous = std::env::var("FLUXION_EPW_DIR").ok();
        std::env::set_var("FLUXION_EPW_DIR", dir);
        let result = f();
        match previous {
            Some(prev) => std::env::set_var("FLUXION_EPW_DIR", prev),
            None => std::env::remove_var("FLUXION_EPW_DIR"),
        }
        result
    }

    /// Inbound `/etc/passwd` (no `.epw` extension) is rejected by the
    /// deserializer before it ever reaches `EpwWeatherSource::from_file`.
    /// This is the canonical path-traversal probe from Issue #2915.
    #[test]
    fn deserialize_epw_file_rejects_etc_passwd() {
        if !std::path::Path::new("/etc/passwd").is_file() {
            eprintln!("skipping: /etc/passwd not present on this platform");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let json = r#"{"type":"epw","path":"/etc/passwd"}"#;
        let result = with_epw_dir(dir.path(), || serde_json::from_str::<WeatherData>(json));
        assert!(
            result.is_err(),
            "/etc/passwd must be rejected at deserialize time: {result:?}"
        );
        // Generic message — the raw user-supplied path must not be
        // reflected back through the deserializer error chain.
        let err = result.unwrap_err().to_string();
        assert!(!err.contains("passwd"), "error must not echo path: {err}");
    }

    /// A `.epw` file inside the `FLUXION_EPW_DIR` allow-list serializes
    /// and round-trips through the deserializer without error.
    #[test]
    fn deserialize_epw_file_round_trips_inside_allowlist() {
        let dir = tempfile::tempdir().unwrap();
        let epw = dir.path().join("USA_CO_Denver.epw");
        std::fs::write(&epw, b"LOCATION,Denver,CO\n").unwrap();

        let original = WeatherData::EpwFile { path: epw.clone() };
        let json = with_epw_dir(dir.path(), || serde_json::to_string(&original).unwrap());
        let deserialized: WeatherData =
            with_epw_dir(dir.path(), || serde_json::from_str(&json).unwrap());
        assert_eq!(original, deserialized);
    }

    /// A `.epw` file that lives outside the allow-list is rejected on
    /// deserialize, even when the path itself is well-formed.
    #[test]
    fn deserialize_epw_file_rejects_traversal_outside_allowlist() {
        let allowed = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let evil = outside.path().join("evil.epw");
        std::fs::write(&evil, b"pwned").unwrap();
        let json = serde_json::json!({
            "type": "epw",
            "path": evil.to_string_lossy().into_owned(),
        })
        .to_string();
        let result = with_epw_dir(allowed.path(), || {
            serde_json::from_str::<WeatherData>(&json)
        });
        assert!(
            result.is_err(),
            "out-of-allowlist epw must be rejected: {result:?}"
        );
    }

    /// The deserializer refuses an `.epw` path that contains `..`
    /// traversal reaching outside the allow-list (defence in depth on
    /// top of `validate_epw_path`'s canonicalize + `starts_with` check).
    #[test]
    fn deserialize_epw_file_rejects_dotdot_traversal() {
        let allowed = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let real = outside.path().join("secret.epw");
        std::fs::write(&real, b"x").unwrap();
        // Build a traversal path: from inside the allowed dir, climb out
        // via `..` and into the outside dir's basename.
        let traversal = allowed
            .path()
            .join("..")
            .join(outside.path().file_name().unwrap())
            .join("secret.epw")
            .to_string_lossy()
            .into_owned();
        let json = serde_json::json!({
            "type": "epw",
            "path": traversal,
        })
        .to_string();
        let result = with_epw_dir(allowed.path(), || {
            serde_json::from_str::<WeatherData>(&json)
        });
        assert!(result.is_err(), "dotdot traversal must be rejected");
    }

    /// `TmyLocation` and `Inline` variants are unaffected by the new
    /// `EpwFile` deserializer gate (regression guard for the tag-based
    /// enum dispatch).
    #[test]
    fn deserialize_non_epw_variants_unaffected() {
        let dir = tempfile::tempdir().unwrap();
        let json = r#"{"type":"tmy","location":"Denver, CO"}"#;
        let result: WeatherData = with_epw_dir(dir.path(), || serde_json::from_str(json).unwrap());
        assert_eq!(
            result,
            WeatherData::TmyLocation {
                location: "Denver, CO".to_string(),
            }
        );
    }
}
