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
//!     SimulationSchema, Geometry, ConstructionSet, ScheduleSet,
//!     WeatherData, ControlSet, SchemaVersion,
//! };
//!
//! // Create a minimal schema
//! let schema = SimulationSchema::v1(SimulationSchemaV1 {
//!     version: SchemaVersion::V1,
//!     metadata: Default::default(),
//!     geometry: Geometry::default(),
//!     constructions: ConstructionSet::default(),
//!     schedules: ScheduleSet::default(),
//!     weather: WeatherData::default(),
//!     controls: ControlSet::default(),
//!     output: Default::default(),
//! });
//! ```

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::sim::construction::ConstructionLayer;
use crate::sim::schedule::{DailySchedule, HVACSchedule};
use crate::weather::HourlyWeatherData;

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
}

impl Default for ScheduleSet {
    fn default() -> Self {
        ScheduleSet {
            occupancy: DailySchedule::weekly("Occupancy".to_string()),
            lighting: DailySchedule::weekly("Lighting".to_string()),
            hvac: HVACSchedule::constant_schedule(20.0, 24.0),
            infiltration: None,
        }
    }
}

/// Weather data specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WeatherData {
    /// Reference to an external EPW file.
    #[serde(rename = "epw")]
    EpwFile { path: PathBuf },

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
    pub deadband_tolerance: f64,
    pub heating_capacity: f64,
    pub cooling_capacity: f64,
}

impl Default for ControlConfig {
    fn default() -> Self {
        ControlConfig {
            heating_setpoint: 20.0,
            cooling_setpoint: 24.0,
            deadband_tolerance: 0.5,
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
        }
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
        let schedule = HVACSchedule::constant_schedule(20.0, 24.0);
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
}
