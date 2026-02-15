//! ASHRAE Standard 140 test case definitions and specifications.
//!
//! This module provides comprehensive data structures for all ASHRAE 140 test cases,
//! including case variants, specifications, and a builder pattern for easy configuration.
//!
//! # Overview
//!
//! ASHRAE Standard 140 specifies test cases for validating building energy simulation software.
//! The test cases are organized into series:
//!
//! - **Low Mass (600 series)**: Lightweight construction buildings
//! - **High Mass (900 series)**: Heavy construction buildings (concrete)
//! - **Free-Float (FF series)**: Buildings without HVAC control
//! - **Special cases**: Multi-zone (960 sunspace), solid conduction (195)
//!
//! # Example
//!
//! ```rust
//! use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, CaseBuilder};
//!
//! // Get a predefined case specification
//! let case_spec = ASHRAE140Case::Case600.spec();
//!
//! // Or build a custom case
//! let custom_spec = CaseBuilder::new()
//!     .low_mass_construction()
//!     .with_dimensions(8.0, 6.0, 2.7)
//!     .with_south_window(12.0)
//!     .with_hvac_setpoints(20.0, 27.0)
//!     .build()
//!     .unwrap();
//! ```

use crate::sim::construction::{Assemblies, Construction};
use serde::{Deserialize, Serialize};

/// Window specification with U-value, SHGC, and optical properties.
///
/// This struct defines the thermal and solar properties of window glazing systems.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WindowSpec {
    /// Window U-value (thermal transmittance) in W/m²K
    pub u_value: f64,
    /// Solar Heat Gain Coefficient (0-1)
    pub shgc: f64,
    /// Normal beam transmittance (0-1)
    pub normal_transmittance: f64,
    /// Glass type
    pub glass_type: GlassType,
}

impl WindowSpec {
    /// Creates a new window specification.
    pub fn new(u_value: f64, shgc: f64, normal_transmittance: f64, glass_type: GlassType) -> Self {
        WindowSpec {
            u_value,
            shgc,
            normal_transmittance,
            glass_type,
        }
    }

    /// Creates a double clear glass window specification (ASHRAE 140 typical).
    ///
    /// - U-value: 3.0 W/m²K
    /// - SHGC: 0.789
    /// - Normal transmittance: 0.86156
    pub fn double_clear_glass() -> Self {
        WindowSpec::new(3.0, 0.789, 0.86156, GlassType::DoubleClear)
    }

    /// Creates a double low-e glass window specification.
    pub fn double_low_e() -> Self {
        WindowSpec::new(2.0, 0.65, 0.70, GlassType::DoubleLowE)
    }
}

/// Glass type enumeration for window specifications.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GlassType {
    /// Single pane clear glass
    SingleClear,
    /// Double pane clear glass
    DoubleClear,
    /// Double pane with low-emissivity coating
    DoubleLowE,
    /// Triple pane clear glass
    TripleClear,
    /// Triple pane with low-emissivity coating
    TripleLowE,
}

impl GlassType {
    /// Returns the number of panes in the glazing system.
    pub fn num_panes(&self) -> u8 {
        match self {
            GlassType::SingleClear => 1,
            GlassType::DoubleClear | GlassType::DoubleLowE => 2,
            GlassType::TripleClear | GlassType::TripleLowE => 3,
        }
    }
}

/// ASHRAE 140 test case enumeration.
///
/// Each variant represents a specific test case defined in ASHRAE Standard 140
/// for validating building energy simulation software.
///
/// The cases are organized by construction type (low/high mass) and variant
/// (baseline, shading, orientation, scheduling, ventilation, free-floating).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ASHRAE140Case {
    // Low mass cases (600 series)
    /// Case 600 - Low mass baseline
    ///
    /// Reference low-mass building with standard construction and south-facing windows.
    Case600,
    /// Case 610 - Low mass with south shading
    ///
    /// Same as Case 600 with 1m overhang on south wall.
    Case610,
    /// Case 620 - Low mass with east/west windows
    ///
    /// Windows split between east and west walls (6m² each) instead of south.
    Case620,
    /// Case 630 - Low mass with east/west shading
    ///
    /// Case 620 with 1m overhang and 1m shade fins on E/W walls.
    Case630,
    /// Case 640 - Low mass with thermostat setback
    ///
    /// Case 600 with heating setback to 10°C overnight (23:00-07:00).
    Case640,
    /// Case 650 - Low mass with night ventilation
    ///
    /// Case 600 with heating disabled and night ventilation fan (18:00-07:00).
    Case650,
    /// Case 600FF - Low mass free-floating
    ///
    /// Same as Case 600 but with no HVAC control (free-floating temperatures).
    Case600FF,
    /// Case 650FF - Low mass free-floating with night ventilation
    ///
    /// Same as Case 650 but with no HVAC control.
    Case650FF,

    // High mass cases (900 series)
    /// Case 900 - High mass baseline
    ///
    /// Reference high-mass building (concrete construction) with south-facing windows.
    Case900,
    /// Case 910 - High mass with south shading
    ///
    /// Same as Case 900 with 1m overhang on south wall.
    Case910,
    /// Case 920 - High mass with east/west windows
    ///
    /// Windows split between east and west walls (6m² each) instead of south.
    Case920,
    /// Case 930 - High mass with east/west shading
    ///
    /// Case 920 with 1m overhang and 1m shade fins on E/W walls.
    Case930,
    /// Case 940 - High mass with thermostat setback
    ///
    /// Case 900 with heating setback to 10°C overnight (23:00-07:00).
    Case940,
    /// Case 950 - High mass with night ventilation
    ///
    /// Case 900 with heating disabled and night ventilation fan (18:00-07:00).
    Case950,
    /// Case 900FF - High mass free-floating
    ///
    /// Same as Case 900 but with no HVAC control (free-floating temperatures).
    Case900FF,
    /// Case 950FF - High mass free-floating with night ventilation
    ///
    /// Same as Case 950 but with no HVAC control.
    Case950FF,

    // Special cases
    /// Case 960 - Sunspace (2-zone building)
    ///
    /// Multi-zone building with back-zone and attached sunspace.
    /// Tests inter-zone heat transfer through common wall.
    Case960,
    /// Case 195 - Solid conduction
    ///
    /// Conduction-only problem with no windows, infiltration, or internal loads.
    /// Tests radiative/convective heat transfer in opaque surfaces.
    Case195,
}

impl ASHRAE140Case {
    /// Returns the case number as a string.
    ///
    /// # Example
    /// ```
    /// use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
    ///
    /// assert_eq!(ASHRAE140Case::Case600.number(), "600");
    /// assert_eq!(ASHRAE140Case::Case650FF.number(), "650FF");
    /// ```
    pub fn number(&self) -> String {
        match self {
            ASHRAE140Case::Case600 => "600".to_string(),
            ASHRAE140Case::Case610 => "610".to_string(),
            ASHRAE140Case::Case620 => "620".to_string(),
            ASHRAE140Case::Case630 => "630".to_string(),
            ASHRAE140Case::Case640 => "640".to_string(),
            ASHRAE140Case::Case650 => "650".to_string(),
            ASHRAE140Case::Case600FF => "600FF".to_string(),
            ASHRAE140Case::Case650FF => "650FF".to_string(),
            ASHRAE140Case::Case900 => "900".to_string(),
            ASHRAE140Case::Case910 => "910".to_string(),
            ASHRAE140Case::Case920 => "920".to_string(),
            ASHRAE140Case::Case930 => "930".to_string(),
            ASHRAE140Case::Case940 => "940".to_string(),
            ASHRAE140Case::Case950 => "950".to_string(),
            ASHRAE140Case::Case900FF => "900FF".to_string(),
            ASHRAE140Case::Case950FF => "950FF".to_string(),
            ASHRAE140Case::Case960 => "960".to_string(),
            ASHRAE140Case::Case195 => "195".to_string(),
        }
    }

    /// Returns a human-readable description of the test case.
    pub fn description(&self) -> String {
        match self {
            ASHRAE140Case::Case600 => {
                "Low mass baseline - standard construction with south windows".to_string()
            }
            ASHRAE140Case::Case610 => "Low mass with south shading (1m overhang)".to_string(),
            ASHRAE140Case::Case620 => "Low mass with east/west windows (6m² each)".to_string(),
            ASHRAE140Case::Case630 => {
                "Low mass with east/west shading (overhang + fins)".to_string()
            }
            ASHRAE140Case::Case640 => "Low mass with thermostat setback (overnight)".to_string(),
            ASHRAE140Case::Case650 => "Low mass with night ventilation (no heating)".to_string(),
            ASHRAE140Case::Case600FF => "Low mass free-floating (no HVAC)".to_string(),
            ASHRAE140Case::Case650FF => "Low mass free-floating with night ventilation".to_string(),
            ASHRAE140Case::Case900 => {
                "High mass baseline - concrete construction with south windows".to_string()
            }
            ASHRAE140Case::Case910 => "High mass with south shading (1m overhang)".to_string(),
            ASHRAE140Case::Case920 => "High mass with east/west windows (6m² each)".to_string(),
            ASHRAE140Case::Case930 => {
                "High mass with east/west shading (overhang + fins)".to_string()
            }
            ASHRAE140Case::Case940 => "High mass with thermostat setback (overnight)".to_string(),
            ASHRAE140Case::Case950 => "High mass with night ventilation (no heating)".to_string(),
            ASHRAE140Case::Case900FF => "High mass free-floating (no HVAC)".to_string(),
            ASHRAE140Case::Case950FF => {
                "High mass free-floating with night ventilation".to_string()
            }
            ASHRAE140Case::Case960 => {
                "Sunspace - 2-zone building (back-zone + sunspace)".to_string()
            }
            ASHRAE140Case::Case195 => {
                "Solid conduction - no windows, no infiltration, no loads".to_string()
            }
        }
    }

    /// Returns the construction type (low mass vs high mass).
    pub fn construction_type(&self) -> ConstructionType {
        match self {
            ASHRAE140Case::Case600
            | ASHRAE140Case::Case610
            | ASHRAE140Case::Case620
            | ASHRAE140Case::Case630
            | ASHRAE140Case::Case640
            | ASHRAE140Case::Case650
            | ASHRAE140Case::Case600FF
            | ASHRAE140Case::Case650FF => ConstructionType::LowMass,
            ASHRAE140Case::Case900
            | ASHRAE140Case::Case910
            | ASHRAE140Case::Case920
            | ASHRAE140Case::Case930
            | ASHRAE140Case::Case940
            | ASHRAE140Case::Case950
            | ASHRAE140Case::Case900FF
            | ASHRAE140Case::Case950FF => ConstructionType::HighMass,
            ASHRAE140Case::Case960 => ConstructionType::Special,
            ASHRAE140Case::Case195 => ConstructionType::Special,
        }
    }

    /// Returns true if this is a free-floating case (no HVAC control).
    pub fn is_free_floating(&self) -> bool {
        matches!(
            self,
            ASHRAE140Case::Case600FF
                | ASHRAE140Case::Case650FF
                | ASHRAE140Case::Case900FF
                | ASHRAE140Case::Case950FF
        )
    }

    /// Returns the case specification for this test case.
    pub fn spec(&self) -> CaseSpec {
        // Get the appropriate preset from CaseBuilder
        match self {
            ASHRAE140Case::Case600 => CaseBuilder::case_600_baseline(),
            ASHRAE140Case::Case610 => CaseBuilder::case_610_south_shading(),
            ASHRAE140Case::Case620 => CaseBuilder::case_620_ew_windows(),
            ASHRAE140Case::Case630 => CaseBuilder::case_630_ew_shading(),
            ASHRAE140Case::Case640 => CaseBuilder::case_640_setback(),
            ASHRAE140Case::Case650 => CaseBuilder::case_650_night_vent(),
            ASHRAE140Case::Case600FF => CaseBuilder::case_600ff(),
            ASHRAE140Case::Case650FF => CaseBuilder::case_650ff(),
            ASHRAE140Case::Case900 => CaseBuilder::case_900_baseline(),
            ASHRAE140Case::Case910 => CaseBuilder::case_910_south_shading(),
            ASHRAE140Case::Case920 => CaseBuilder::case_920_ew_windows(),
            ASHRAE140Case::Case930 => CaseBuilder::case_930_ew_shading(),
            ASHRAE140Case::Case940 => CaseBuilder::case_940_setback(),
            ASHRAE140Case::Case950 => CaseBuilder::case_950_night_vent(),
            ASHRAE140Case::Case900FF => CaseBuilder::case_900ff(),
            ASHRAE140Case::Case950FF => CaseBuilder::case_950ff(),
            ASHRAE140Case::Case960 => CaseBuilder::case_960_sunspace(),
            ASHRAE140Case::Case195 => CaseBuilder::case_195_solid_conduction(),
        }
    }
}

/// Construction type for ASHRAE 140 test cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstructionType {
    /// Low mass construction (lightweight materials like plasterboard, fiberglass, wood)
    LowMass,
    /// High mass construction (heavy materials like concrete)
    HighMass,
    /// Special construction (multi-zone, solid conduction)
    Special,
}

/// Orientation of a surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Orientation {
    North,
    East,
    South,
    West,
    Up,   // Roof
    Down, // Floor
    Horizontal,
}

impl Orientation {
    /// Returns the azimuth angle in degrees (0° = North, clockwise).
    pub fn azimuth_deg(&self) -> f64 {
        match self {
            Orientation::North => 0.0,
            Orientation::East => 90.0,
            Orientation::South => 180.0,
            Orientation::West => 270.0,
            Orientation::Up | Orientation::Down | Orientation::Horizontal => -1.0,
        }
    }

    /// Returns the azimuth angle in degrees according to ASHRAE 140 (0° = South, clockwise).
    pub fn azimuth(&self) -> f64 {
        match self {
            Orientation::South => 0.0,
            Orientation::West => 90.0,
            Orientation::North => 180.0,
            Orientation::East => 270.0,
            Orientation::Up | Orientation::Down | Orientation::Horizontal => -1.0,
        }
    }
}


/// Window specification with area and orientation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WindowArea {
    /// Window area in square meters (m²)
    pub area: f64,
    /// Wall orientation
    pub orientation: Orientation,
    /// Window height in meters (m)
    pub height: f64,
    /// Window width in meters (m)
    pub width: f64,
    /// Offset from floor in meters (m)
    pub sill_height: f64,
    /// Offset from left edge in meters (m)
    pub left_offset: f64,
}

impl WindowArea {
    /// Creates a new window area specification.
    pub fn new(area: f64, orientation: Orientation) -> Self {
        WindowArea {
            area,
            orientation,
            height: 2.0,       // Default height (Case 600 windows are 2m tall)
            width: area / 2.0, // Default width derived from area
            sill_height: 0.2,  // Default offset from floor (Case 600 has 0.2m sill)
            left_offset: 0.5,  // Default offset from left edge
        }
    }

    /// Creates a window with full dimensions (height, width, sill, offset).
    pub fn with_dimensions(
        area: f64,
        orientation: Orientation,
        height: f64,
        width: f64,
        sill_height: f64,
        left_offset: f64,
    ) -> Self {
        WindowArea {
            area,
            orientation,
            height,
            width,
            sill_height,
            left_offset,
        }
    }
}

/// Shading device specification.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ShadingDevice {
    /// Type of shading device
    pub shading_type: ShadingType,
    /// Depth of overhang in meters (m)
    pub overhang_depth: f64,
    /// Width of shade fins in meters (m)
    pub fin_width: f64,
    /// Height from ground in meters (m)
    pub mounting_height: f64,
}

/// Type of shading device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShadingType {
    /// No shading
    None,
    /// Overhang (horizontal projection)
    Overhang,
    /// Shade fins (vertical projections)
    Fins,
    /// Both overhang and fins
    OverhangAndFins,
}

impl ShadingDevice {
    /// Creates a no-shading specification.
    pub fn none() -> Self {
        ShadingDevice {
            shading_type: ShadingType::None,
            overhang_depth: 0.0,
            fin_width: 0.0,
            mounting_height: 0.0,
        }
    }

    /// Creates an overhang shading device.
    pub fn overhang(depth: f64, height: f64) -> Self {
        ShadingDevice {
            shading_type: ShadingType::Overhang,
            overhang_depth: depth,
            fin_width: 0.0,
            mounting_height: height,
        }
    }

    /// Creates shade fins.
    pub fn fins(width: f64) -> Self {
        ShadingDevice {
            shading_type: ShadingType::Fins,
            overhang_depth: 0.0,
            fin_width: width,
            mounting_height: 0.0, // Fins extend from roof to ground
        }
    }

    /// Creates both overhang and fins.
    pub fn overhang_and_fins(overhang_depth: f64, fin_width: f64, height: f64) -> Self {
        ShadingDevice {
            shading_type: ShadingType::OverhangAndFins,
            overhang_depth,
            fin_width,
            mounting_height: height,
        }
    }
}

/// Internal loads specification.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InternalLoads {
    /// Total continuous load in Watts (W)
    pub total_load: f64,
    /// Fraction of load that is radiative (0.0 to 1.0)
    pub radiative_fraction: f64,
    /// Fraction of load that is convective (0.0 to 1.0)
    pub convective_fraction: f64,
}

impl InternalLoads {
    /// Creates new internal loads specification.
    ///
    /// # Panics
    /// Panics if radiative_fraction + convective_fraction is not approximately 1.0.
    pub fn new(total_load: f64, radiative_fraction: f64, convective_fraction: f64) -> Self {
        assert!(
            (radiative_fraction + convective_fraction - 1.0).abs() < 0.01,
            "Radiative + convective fractions must sum to 1.0"
        );
        InternalLoads {
            total_load,
            radiative_fraction,
            convective_fraction,
        }
    }

    /// Returns the radiative portion of the load (W).
    pub fn radiative_load(&self) -> f64 {
        self.total_load * self.radiative_fraction
    }

    /// Returns the convective portion of the load (W).
    pub fn convective_load(&self) -> f64 {
        self.total_load * self.convective_fraction
    }
}

/// HVAC schedule specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HvacSchedule {
    /// Heating setpoint (°C) when HVAC is enabled
    pub heating_setpoint: f64,
    /// Cooling setpoint (°C) when HVAC is enabled
    pub cooling_setpoint: f64,
    /// Operating hours (start_hour, end_hour) when HVAC is active
    pub operating_hours: (u8, u8),
    /// Night setback setpoint (°C), if applicable
    pub setback_setpoint: Option<f64>,
    /// Setback hours (start_hour, end_hour), if applicable
    pub setback_hours: Option<(u8, u8)>,
    /// HVAC efficiency (0.0 to 1.0, where 1.0 = 100% efficient)
    pub efficiency: f64,
}

impl HvacSchedule {
    /// Creates a constant HVAC schedule (no setback).
    ///
    /// # Arguments
    /// * `heating_setpoint` - Heating temperature setpoint in °C
    /// * `cooling_setpoint` - Cooling temperature setpoint in °C
    pub fn constant(heating_setpoint: f64, cooling_setpoint: f64) -> Self {
        HvacSchedule {
            heating_setpoint,
            cooling_setpoint,
            operating_hours: (0, 24),
            setback_setpoint: None,
            setback_hours: None,
            efficiency: 1.0,
        }
    }

    /// Creates an HVAC schedule with setback.
    ///
    /// # Arguments
    /// * `heating_setpoint` - Normal heating setpoint in °C
    /// * `cooling_setpoint` - Cooling setpoint in °C
    /// * `setback_setpoint` - Reduced heating setpoint during setback period in °C
    /// * `setback_start` - Hour when setback starts (0-23)
    /// * `setback_end` - Hour when setback ends (0-23)
    pub fn with_setback(
        heating_setpoint: f64,
        cooling_setpoint: f64,
        setback_setpoint: f64,
        setback_start: u8,
        setback_end: u8,
    ) -> Self {
        HvacSchedule {
            heating_setpoint,
            cooling_setpoint,
            operating_hours: (0, 24),
            setback_setpoint: Some(setback_setpoint),
            setback_hours: Some((setback_start, setback_end)),
            efficiency: 1.0,
        }
    }

    /// Creates an HVAC schedule with operating hours restriction.
    ///
    /// # Arguments
    /// * `heating_setpoint` - Heating setpoint in °C
    /// * `cooling_setpoint` - Cooling setpoint in °C
    /// * `operating_start` - Hour when HVAC turns on (0-23)
    /// * `operating_end` - Hour when HVAC turns off (0-23)
    pub fn with_operating_hours(
        heating_setpoint: f64,
        cooling_setpoint: f64,
        operating_start: u8,
        operating_end: u8,
    ) -> Self {
        HvacSchedule {
            heating_setpoint,
            cooling_setpoint,
            operating_hours: (operating_start, operating_end),
            setback_setpoint: None,
            setback_hours: None,
            efficiency: 1.0,
        }
    }

    /// Creates a free-floating schedule (no HVAC control).
    pub fn free_floating() -> Self {
        HvacSchedule {
            heating_setpoint: 0.0,
            cooling_setpoint: 0.0,
            operating_hours: (0, 0),
            setback_setpoint: None,
            setback_hours: None,
            efficiency: 0.0,
        }
    }

    /// Returns true if HVAC is enabled.
    pub fn is_enabled(&self) -> bool {
        self.efficiency > 0.0 && self.operating_hours != (0, 0)
    }

    /// Returns true if this is a free-floating schedule.
    pub fn is_free_floating(&self) -> bool {
        !self.is_enabled()
    }

    /// Gets the heating setpoint for a given hour.
    pub fn heating_setpoint_at_hour(&self, hour: u8) -> Option<f64> {
        if !self.is_enabled() {
            return None;
        }

        let current_setpoint = if let Some((setback_start, setback_end)) = self.setback_hours {
            if setback_start <= hour || hour < setback_end {
                // During setback period
                self.setback_setpoint.unwrap_or(self.heating_setpoint)
            } else {
                // Normal period
                self.heating_setpoint
            }
        } else {
            self.heating_setpoint
        };

        // Check if HVAC is operating at this hour
        let (start, end) = self.operating_hours;
        if start <= hour || hour < end {
            return Some(current_setpoint);
        }

        None
    }

    /// Gets the cooling setpoint for a given hour.
    pub fn cooling_setpoint_at_hour(&self, hour: u8) -> Option<f64> {
        if !self.is_enabled() {
            return None;
        }

        // Check if HVAC is operating at this hour
        let (start, end) = self.operating_hours;
        if start <= hour || hour < end {
            return Some(self.cooling_setpoint);
        }

        None
    }
}

/// Night ventilation specification.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NightVentilation {
    /// Fan capacity in standard m³/h
    pub fan_capacity: f64,
    /// Operating hours (start_hour, end_hour) when fan is active
    pub operating_hours: (u8, u8),
    /// Whether fan adds waste heat to zone (always false for ASHRAE 140)
    pub adds_heat: bool,
}

impl NightVentilation {
    /// Creates a night ventilation specification.
    ///
    /// # Arguments
    /// * `fan_capacity` - Fan capacity in standard m³/h
    /// * `start_hour` - Hour when fan turns on (0-23)
    /// * `end_hour` - Hour when fan turns off (0-23)
    pub fn new(fan_capacity: f64, start_hour: u8, end_hour: u8) -> Self {
        NightVentilation {
            fan_capacity,
            operating_hours: (start_hour, end_hour),
            adds_heat: false,
        }
    }

    /// Creates the ASHRAE 140 Case 650 night ventilation specification.
    pub fn case_650() -> Self {
        NightVentilation {
            fan_capacity: 1703.16,    // standard m³/h (from ASHRAE 140 spec)
            operating_hours: (18, 7), // 18:00 to 07:00 (wraps midnight)
            adds_heat: false,
        }
    }

    /// Returns true if ventilation is active at the given hour.
    pub fn is_active_at_hour(&self, hour: u8) -> bool {
        let (start, end) = self.operating_hours;
        start <= hour || hour < end
    }
}

/// Complete case specification for an ASHRAE 140 test case.
///
/// This struct contains all the information needed to configure a ThermalModel
/// for a specific ASHRAE 140 test case, including geometry, construction,
/// windows, shading, HVAC, internal loads, and infiltration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseSpec {
    /// Case identifier (e.g., "600", "650FF")
    pub case_id: String,

    /// Human-readable description
    pub description: String,

    /// Geometry specifications
    pub geometry: GeometrySpec,

    /// Construction assemblies for each surface type
    pub construction: ConstructionSpec,

    /// Window specifications
    pub windows: Vec<WindowArea>,

    /// Window properties (U-value, SHGC, etc.)
    pub window_properties: WindowSpec,

    /// Shading devices
    pub shading: Option<ShadingDevice>,

    /// Internal heat gains
    pub internal_loads: Option<InternalLoads>,

    /// HVAC control schedule
    pub hvac: HvacSchedule,

    /// Night ventilation (if applicable)
    pub night_ventilation: Option<NightVentilation>,

    /// Infiltration rate in air changes per hour (ACH)
    pub infiltration_ach: f64,

    /// Number of zones (1 for most cases, 2 for Case 960 sunspace)
    pub num_zones: usize,
}

/// Geometry specification for a building zone.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GeometrySpec {
    /// Zone width in meters (m)
    pub width: f64,
    /// Zone depth in meters (m)
    pub depth: f64,
    /// Zone height in meters (m)
    pub height: f64,
}

impl GeometrySpec {
    /// Creates a new geometry specification.
    pub fn new(width: f64, depth: f64, height: f64) -> Self {
        GeometrySpec {
            width,
            depth,
            height,
        }
    }

    /// Returns the floor area in square meters (m²).
    pub fn floor_area(&self) -> f64 {
        self.width * self.depth
    }

    /// Returns the zone volume in cubic meters (m³).
    pub fn volume(&self) -> f64 {
        self.width * self.depth * self.height
    }

    /// Returns the total wall area in square meters (m²).
    pub fn wall_area(&self) -> f64 {
        let perimeter = 2.0 * (self.width + self.depth);
        perimeter * self.height
    }

    /// Returns the roof area in square meters (m²).
    pub fn roof_area(&self) -> f64 {
        self.floor_area()
    }

    /// Returns the total opaque surface area (walls + roof + floor).
    pub fn total_opaque_area(&self) -> f64 {
        self.wall_area() + self.roof_area() + self.floor_area()
    }
}

/// Construction specification for building envelope assemblies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstructionSpec {
    /// Wall construction assembly
    pub wall: Construction,
    /// Roof construction assembly
    pub roof: Construction,
    /// Floor construction assembly
    pub floor: Construction,
}

impl ConstructionSpec {
    /// Creates a construction specification with given assemblies.
    pub fn new(wall: Construction, roof: Construction, floor: Construction) -> Self {
        ConstructionSpec { wall, roof, floor }
    }

    /// Returns the total wall U-value (with ASHRAE film coefficients).
    pub fn wall_u_value(&self) -> f64 {
        self.wall.u_value(None)
    }

    /// Returns the total roof U-value (with ASHRAE film coefficients).
    pub fn roof_u_value(&self) -> f64 {
        self.roof.u_value(None)
    }

    /// Returns the total floor U-value (with ASHRAE film coefficients).
    pub fn floor_u_value(&self) -> f64 {
        self.floor.u_value(None)
    }
}

impl CaseSpec {
    /// Validates the case specification.
    ///
    /// # Returns
    /// Ok(()) if valid, Err(String) with description if invalid.
    pub fn validate(&self) -> Result<(), String> {
        // Check geometry
        if self.geometry.width <= 0.0 || self.geometry.depth <= 0.0 || self.geometry.height <= 0.0 {
            return Err("Geometry dimensions must be positive".to_string());
        }

        // Check windows
        if self.windows.is_empty() && !self.case_id.contains("195") {
            return Err("At least one window required (except Case 195)".to_string());
        }

        for window in &self.windows {
            if window.area <= 0.0 {
                return Err("Window area must be positive".to_string());
            }
            if window.height <= 0.0 || window.width <= 0.0 {
                return Err("Window dimensions must be positive".to_string());
            }
        }

        // Check infiltration
        if self.infiltration_ach < 0.0 {
            return Err("Infiltration rate cannot be negative".to_string());
        }

        // Check HVAC schedule
        if !self.hvac.is_free_floating() {
            // Allow heating == cooling for bang-bang control (Case 195)
            if self.hvac.heating_setpoint > self.hvac.cooling_setpoint {
                return Err(
                    "Heating setpoint must be less than or equal to cooling setpoint".to_string(),
                );
            }
            if self.hvac.efficiency <= 0.0 || self.hvac.efficiency > 1.0 {
                return Err("HVAC efficiency must be in (0, 1]".to_string());
            }
        }

        // Check internal loads
        if let Some(loads) = self.internal_loads {
            if loads.total_load < 0.0 {
                return Err("Internal loads cannot be negative".to_string());
            }
        }

        Ok(())
    }

    /// Returns the total window area across all orientations.
    pub fn total_window_area(&self) -> f64 {
        self.windows.iter().map(|w| w.area).sum()
    }

    /// Returns window area for a specific orientation.
    pub fn window_area_by_orientation(&self, orientation: Orientation) -> f64 {
        self.windows
            .iter()
            .filter(|w| w.orientation == orientation)
            .map(|w| w.area)
            .sum()
    }

    /// Returns true if this is a free-floating case.
    pub fn is_free_floating(&self) -> bool {
        self.hvac.is_free_floating()
    }

    /// Returns true if this case has night ventilation.
    pub fn has_night_ventilation(&self) -> bool {
        self.night_ventilation.is_some()
    }

    /// Returns true if this case has shading devices.
    pub fn has_shading(&self) -> bool {
        self.shading.is_some() && self.shading.as_ref().unwrap().shading_type != ShadingType::None
    }
}

/// Builder for creating ASHRAE 140 case specifications.
///
/// The builder provides a fluent API for configuring test cases with sensible defaults
/// and validation.
///
/// # Example
///
/// ```rust
/// use fluxion::validation::ashrae_140_cases::CaseBuilder;
///
/// let spec = CaseBuilder::new()
///     .low_mass_construction()
///     .with_dimensions(8.0, 6.0, 2.7)
///     .with_south_window(12.0)
///     .with_hvac_setpoints(20.0, 27.0)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct CaseBuilder {
    case_id: Option<String>,
    description: String,
    geometry: Option<GeometrySpec>,
    construction_type: ConstructionType,
    construction: Option<ConstructionSpec>,
    windows: Vec<WindowArea>,
    window_properties: WindowSpec,
    shading: Option<ShadingDevice>,
    internal_loads: Option<InternalLoads>,
    hvac: HvacSchedule,
    night_ventilation: Option<NightVentilation>,
    infiltration_ach: f64,
    num_zones: usize,
}

impl Default for CaseBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CaseBuilder {
    /// Creates a new CaseBuilder with default values.
    pub fn new() -> Self {
        CaseBuilder {
            case_id: None,
            description: String::new(),
            geometry: None,
            construction_type: ConstructionType::LowMass,
            construction: None,
            windows: Vec::new(),
            window_properties: WindowSpec::double_clear_glass(),
            shading: None,
            internal_loads: None,
            hvac: HvacSchedule::constant(20.0, 27.0),
            night_ventilation: None,
            infiltration_ach: 0.5,
            num_zones: 1,
        }
    }

    /// Sets the case identifier.
    pub fn with_case_id(mut self, case_id: String) -> Self {
        self.case_id = Some(case_id);
        self
    }

    /// Sets the case description.
    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }

    /// Sets the zone dimensions (width, depth, height in meters).
    pub fn with_dimensions(mut self, width: f64, depth: f64, height: f64) -> Self {
        self.geometry = Some(GeometrySpec::new(width, depth, height));
        self
    }

    /// Sets the construction type to low mass.
    pub fn low_mass_construction(mut self) -> Self {
        self.construction_type = ConstructionType::LowMass;
        self
    }

    /// Sets the construction type to high mass.
    pub fn high_mass_construction(mut self) -> Self {
        self.construction_type = ConstructionType::HighMass;
        self
    }

    /// Sets custom construction assemblies.
    pub fn with_construction(
        mut self,
        wall: Construction,
        roof: Construction,
        floor: Construction,
    ) -> Self {
        self.construction = Some(ConstructionSpec::new(wall, roof, floor));
        self
    }

    /// Adds a window with specified area and orientation.
    pub fn with_window(mut self, area: f64, orientation: Orientation) -> Self {
        self.windows.push(WindowArea::new(area, orientation));
        self
    }

    /// Adds a south-facing window with default dimensions (Case 600 style).
    pub fn with_south_window(mut self, area: f64) -> Self {
        self.windows.push(WindowArea::new(area, Orientation::South));
        self
    }

    /// Adds east and west windows with equal area.
    pub fn with_ew_windows(mut self, each_area: f64) -> Self {
        self.windows
            .push(WindowArea::new(each_area, Orientation::East));
        self.windows
            .push(WindowArea::new(each_area, Orientation::West));
        self
    }

    /// Sets window properties.
    pub fn with_window_properties(mut self, window_properties: WindowSpec) -> Self {
        self.window_properties = window_properties;
        self
    }

    /// Sets shading device.
    pub fn with_shading(mut self, shading: ShadingDevice) -> Self {
        self.shading = Some(shading);
        self
    }

    /// Sets internal loads.
    pub fn with_internal_loads(mut self, loads: InternalLoads) -> Self {
        self.internal_loads = Some(loads);
        self
    }

    /// Sets HVAC schedule.
    pub fn with_hvac(mut self, hvac: HvacSchedule) -> Self {
        self.hvac = hvac;
        self
    }

    /// Sets HVAC setpoints (heating, cooling).
    pub fn with_hvac_setpoints(mut self, heating: f64, cooling: f64) -> Self {
        self.hvac = HvacSchedule::constant(heating, cooling);
        self
    }

    /// Sets HVAC with setback.
    pub fn with_hvac_setback(mut self, heating: f64, cooling: f64, setback: f64) -> Self {
        self.hvac = HvacSchedule::with_setback(heating, cooling, setback, 23, 7);
        self
    }

    /// Sets night ventilation.
    pub fn with_night_ventilation(mut self, ventilation: NightVentilation) -> Self {
        self.night_ventilation = Some(ventilation);
        self
    }

    /// Sets infiltration rate (ACH).
    pub fn with_infiltration(mut self, ach: f64) -> Self {
        self.infiltration_ach = ach;
        self
    }

    /// Sets number of zones.
    pub fn with_num_zones(mut self, num_zones: usize) -> Self {
        self.num_zones = num_zones;
        self
    }

    /// Builds and validates the case specification.
    ///
    /// # Returns
    /// Ok(CaseSpec) if validation passes, Err(String) if validation fails.
    pub fn build(self) -> Result<CaseSpec, String> {
        // Use default construction if not specified
        let construction = self
            .construction
            .unwrap_or_else(|| match self.construction_type {
                ConstructionType::LowMass => ConstructionSpec::new(
                    Assemblies::low_mass_wall(),
                    Assemblies::low_mass_roof(),
                    Assemblies::insulated_floor(),
                ),
                ConstructionType::HighMass => ConstructionSpec::new(
                    Assemblies::high_mass_wall(),
                    Assemblies::high_mass_roof(),
                    Assemblies::insulated_floor(),
                ),
                ConstructionType::Special => ConstructionSpec::new(
                    Assemblies::low_mass_wall(),
                    Assemblies::low_mass_roof(),
                    Assemblies::insulated_floor(),
                ),
            });

        let spec = CaseSpec {
            case_id: self.case_id.unwrap_or_else(|| "custom".to_string()),
            description: self.description,
            geometry: self.geometry.ok_or("Geometry must be specified")?,
            construction,
            windows: self.windows,
            window_properties: self.window_properties,
            shading: self.shading,
            internal_loads: self.internal_loads,
            hvac: self.hvac,
            night_ventilation: self.night_ventilation,
            infiltration_ach: self.infiltration_ach,
            num_zones: self.num_zones,
        };

        // Validate the spec
        spec.validate()?;

        Ok(spec)
    }

    // ===== Predefined ASHRAE 140 Case Specifications =====

    /// Case 600 - Low mass baseline.
    pub fn case_600_baseline() -> CaseSpec {
        Self::new()
            .with_case_id("600".to_string())
            .with_description(
                "Low mass baseline - standard construction with south windows".to_string(),
            )
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
            .with_hvac_setpoints(20.0, 27.0)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 600 should validate")
    }

    /// Case 610 - Low mass with south shading.
    pub fn case_610_south_shading() -> CaseSpec {
        Self::new()
            .with_case_id("610".to_string())
            .with_description("Low mass with south shading (1m overhang)".to_string())
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_shading(ShadingDevice::overhang(1.0, 2.7))
            .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
            .with_hvac_setpoints(20.0, 27.0)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 610 should validate")
    }

    /// Case 620 - Low mass with east/west windows.
    pub fn case_620_ew_windows() -> CaseSpec {
        Self::new()
            .with_case_id("620".to_string())
            .with_description("Low mass with east/west windows (6m² each)".to_string())
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_ew_windows(6.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
            .with_hvac_setpoints(20.0, 27.0)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 620 should validate")
    }

    /// Case 630 - Low mass with east/west shading.
    pub fn case_630_ew_shading() -> CaseSpec {
        Self::new()
            .with_case_id("630".to_string())
            .with_description("Low mass with east/west shading (overhang + fins)".to_string())
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_ew_windows(6.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_shading(ShadingDevice::overhang_and_fins(1.0, 1.0, 2.7))
            .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
            .with_hvac_setpoints(20.0, 27.0)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 630 should validate")
    }

    /// Case 640 - Low mass with thermostat setback.
    pub fn case_640_setback() -> CaseSpec {
        Self::new()
            .with_case_id("640".to_string())
            .with_description("Low mass with thermostat setback (overnight)".to_string())
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
            .with_hvac_setback(20.0, 27.0, 10.0)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 640 should validate")
    }

    /// Case 650 - Low mass with night ventilation.
    pub fn case_650_night_vent() -> CaseSpec {
        Self::new()
            .with_case_id("650".to_string())
            .with_description("Low mass with night ventilation (no heating)".to_string())
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
            .with_hvac(HvacSchedule::with_operating_hours(20.0, 27.0, 7, 18))
            .with_night_ventilation(NightVentilation::case_650())
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 650 should validate")
    }

    /// Case 600FF - Low mass free-floating.
    pub fn case_600ff() -> CaseSpec {
        Self::new()
            .with_case_id("600FF".to_string())
            .with_description("Low mass free-floating (no HVAC)".to_string())
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
            .with_hvac(HvacSchedule::free_floating())
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 600FF should validate")
    }

    /// Case 650FF - Low mass free-floating with night ventilation.
    pub fn case_650ff() -> CaseSpec {
        Self::new()
            .with_case_id("650FF".to_string())
            .with_description("Low mass free-floating with night ventilation".to_string())
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
            .with_hvac(HvacSchedule::free_floating())
            .with_night_ventilation(NightVentilation::case_650())
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 650FF should validate")
    }

    /// Case 900 - High mass baseline.
    pub fn case_900_baseline() -> CaseSpec {
        Self::new()
            .with_case_id("900".to_string())
            .with_description(
                "High mass baseline - concrete construction with south windows".to_string(),
            )
            .with_dimensions(8.0, 6.0, 2.7)
            .high_mass_construction()
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
            .with_hvac_setpoints(20.0, 27.0)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 900 should validate")
    }

    /// Case 910 - High mass with south shading.
    pub fn case_910_south_shading() -> CaseSpec {
        Self::new()
            .with_case_id("910".to_string())
            .with_description("High mass with south shading (1m overhang)".to_string())
            .with_dimensions(8.0, 6.0, 2.7)
            .high_mass_construction()
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_shading(ShadingDevice::overhang(1.0, 2.7))
            .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
            .with_hvac_setpoints(20.0, 27.0)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 910 should validate")
    }

    /// Case 920 - High mass with east/west windows.
    pub fn case_920_ew_windows() -> CaseSpec {
        Self::new()
            .with_case_id("920".to_string())
            .with_description("High mass with east/west windows (6m² each)".to_string())
            .with_dimensions(8.0, 6.0, 2.7)
            .high_mass_construction()
            .with_ew_windows(6.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
            .with_hvac_setpoints(20.0, 27.0)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 920 should validate")
    }

    /// Case 930 - High mass with east/west shading.
    pub fn case_930_ew_shading() -> CaseSpec {
        Self::new()
            .with_case_id("930".to_string())
            .with_description("High mass with east/west shading (overhang + fins)".to_string())
            .with_dimensions(8.0, 6.0, 2.7)
            .high_mass_construction()
            .with_ew_windows(6.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_shading(ShadingDevice::overhang_and_fins(1.0, 1.0, 2.7))
            .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
            .with_hvac_setpoints(20.0, 27.0)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 930 should validate")
    }

    /// Case 940 - High mass with thermostat setback.
    pub fn case_940_setback() -> CaseSpec {
        Self::new()
            .with_case_id("940".to_string())
            .with_description("High mass with thermostat setback (overnight)".to_string())
            .with_dimensions(8.0, 6.0, 2.7)
            .high_mass_construction()
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
            .with_hvac_setback(20.0, 27.0, 10.0)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 940 should validate")
    }

    /// Case 950 - High mass with night ventilation.
    pub fn case_950_night_vent() -> CaseSpec {
        Self::new()
            .with_case_id("950".to_string())
            .with_description("High mass with night ventilation (no heating)".to_string())
            .with_dimensions(8.0, 6.0, 2.7)
            .high_mass_construction()
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
            .with_hvac(HvacSchedule::with_operating_hours(20.0, 27.0, 7, 18))
            .with_night_ventilation(NightVentilation::case_650())
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 950 should validate")
    }

    /// Case 900FF - High mass free-floating.
    pub fn case_900ff() -> CaseSpec {
        Self::new()
            .with_case_id("900FF".to_string())
            .with_description("High mass free-floating (no HVAC)".to_string())
            .with_dimensions(8.0, 6.0, 2.7)
            .high_mass_construction()
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
            .with_hvac(HvacSchedule::free_floating())
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 900FF should validate")
    }

    /// Case 950FF - High mass free-floating with night ventilation.
    pub fn case_950ff() -> CaseSpec {
        Self::new()
            .with_case_id("950FF".to_string())
            .with_description("High mass free-floating with night ventilation".to_string())
            .with_dimensions(8.0, 6.0, 2.7)
            .high_mass_construction()
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
            .with_hvac(HvacSchedule::free_floating())
            .with_night_ventilation(NightVentilation::case_650())
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 950FF should validate")
    }

    /// Case 960 - Sunspace (2-zone building).
    pub fn case_960_sunspace() -> CaseSpec {
        Self::new()
            .with_case_id("960".to_string())
            .with_description("Sunspace - 2-zone building (back-zone + sunspace)".to_string())
            .with_dimensions(8.0, 8.0, 2.7) // Extended for sunspace
            .low_mass_construction()
            .with_south_window(12.0) // Back-zone windows
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
            .with_hvac_setpoints(20.0, 27.0)
            .with_infiltration(0.5)
            .with_num_zones(2) // 2 zones: back-zone + sunspace
            .build()
            .expect("Case 960 should validate")
    }

    /// Case 195 - Solid conduction (no windows, no infiltration, no loads).
    pub fn case_195_solid_conduction() -> CaseSpec {
        Self::new()
            .with_case_id("195".to_string())
            .with_description(
                "Solid conduction - no windows, no infiltration, no loads".to_string(),
            )
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_window_properties(WindowSpec::double_clear_glass())
            // No windows - this is a solid conduction problem
            .with_internal_loads(InternalLoads::new(0.0, 0.6, 0.4)) // No internal loads
            .with_hvac_setpoints(20.0, 20.0) // Bang-bang control
            .with_infiltration(0.0) // No infiltration
            .with_num_zones(1)
            .build()
            .expect("Case 195 should validate")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ashrae_case_enum() {
        // Test case numbers
        assert_eq!(ASHRAE140Case::Case600.number(), "600");
        assert_eq!(ASHRAE140Case::Case650FF.number(), "650FF");
        assert_eq!(ASHRAE140Case::Case960.number(), "960");

        // Test descriptions
        assert!(ASHRAE140Case::Case600.description().contains("baseline"));
        assert!(ASHRAE140Case::Case610.description().contains("shading"));
        assert!(ASHRAE140Case::Case960.description().contains("sunspace"));

        // Test construction types
        assert_eq!(
            ASHRAE140Case::Case600.construction_type(),
            ConstructionType::LowMass
        );
        assert_eq!(
            ASHRAE140Case::Case900.construction_type(),
            ConstructionType::HighMass
        );
        assert_eq!(
            ASHRAE140Case::Case960.construction_type(),
            ConstructionType::Special
        );

        // Test free-floating detection
        assert!(ASHRAE140Case::Case600FF.is_free_floating());
        assert!(ASHRAE140Case::Case950FF.is_free_floating());
        assert!(!ASHRAE140Case::Case600.is_free_floating());
        assert!(!ASHRAE140Case::Case900.is_free_floating());
    }

    #[test]
    fn test_orientation() {
        assert_eq!(Orientation::North.azimuth_deg(), 0.0);
        assert_eq!(Orientation::East.azimuth_deg(), 90.0);
        assert_eq!(Orientation::South.azimuth_deg(), 180.0);
        assert_eq!(Orientation::West.azimuth_deg(), 270.0);
        assert_eq!(Orientation::Horizontal.azimuth_deg(), -1.0);
    }

    #[test]
    fn test_window_area() {
        let window = WindowArea::new(12.0, Orientation::South);
        assert_eq!(window.area, 12.0);
        assert_eq!(window.orientation, Orientation::South);
        assert_eq!(window.height, 2.0);
        assert_eq!(window.sill_height, 0.2);

        let window2 = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        assert_eq!(window2.width, 6.0);
        assert_eq!(window2.left_offset, 0.5);
    }

    #[test]
    fn test_shading_device() {
        let none = ShadingDevice::none();
        assert_eq!(none.shading_type, ShadingType::None);

        let overhang = ShadingDevice::overhang(1.0, 2.7);
        assert_eq!(overhang.shading_type, ShadingType::Overhang);
        assert_eq!(overhang.overhang_depth, 1.0);

        let fins = ShadingDevice::fins(1.0);
        assert_eq!(fins.shading_type, ShadingType::Fins);
        assert_eq!(fins.fin_width, 1.0);

        let both = ShadingDevice::overhang_and_fins(1.0, 1.0, 2.7);
        assert_eq!(both.shading_type, ShadingType::OverhangAndFins);
    }

    #[test]
    fn test_internal_loads() {
        let loads = InternalLoads::new(200.0, 0.6, 0.4);
        assert_eq!(loads.total_load, 200.0);
        assert_eq!(loads.radiative_fraction, 0.6);
        assert_eq!(loads.convective_fraction, 0.4);
        assert_eq!(loads.radiative_load(), 120.0);
        assert_eq!(loads.convective_load(), 80.0);
    }

    #[test]
    #[should_panic(expected = "Radiative + convective fractions must sum to 1.0")]
    fn test_internal_loads_invalid_fractions() {
        InternalLoads::new(200.0, 0.5, 0.3); // Sum is 0.8, not 1.0
    }

    #[test]
    fn test_hvac_schedule() {
        let constant = HvacSchedule::constant(20.0, 27.0);
        assert!(constant.is_enabled());
        assert!(!constant.is_free_floating());
        assert_eq!(constant.heating_setpoint_at_hour(12), Some(20.0));
        assert_eq!(constant.cooling_setpoint_at_hour(12), Some(27.0));

        let setback = HvacSchedule::with_setback(20.0, 27.0, 10.0, 23, 7);
        assert_eq!(setback.heating_setpoint_at_hour(0), Some(10.0)); // During setback
        assert_eq!(setback.heating_setpoint_at_hour(12), Some(20.0)); // Normal period

        let free_floating = HvacSchedule::free_floating();
        assert!(!free_floating.is_enabled());
        assert!(free_floating.is_free_floating());
        assert_eq!(free_floating.heating_setpoint_at_hour(12), None);
    }

    #[test]
    fn test_night_ventilation() {
        let vent = NightVentilation::case_650();
        assert_eq!(vent.fan_capacity, 1703.16);
        assert_eq!(vent.operating_hours, (18, 7));
        assert!(!vent.adds_heat);
        assert!(vent.is_active_at_hour(20)); // 20:00 is active
        assert!(!vent.is_active_at_hour(12)); // 12:00 is not active
    }

    #[test]
    fn test_geometry_spec() {
        let geo = GeometrySpec::new(8.0, 6.0, 2.7);
        assert_eq!(geo.width, 8.0);
        assert_eq!(geo.depth, 6.0);
        assert_eq!(geo.height, 2.7);
        assert_eq!(geo.floor_area(), 48.0);
        assert!((geo.volume() - 129.6).abs() < 1e-10); // Account for floating point
        assert!((geo.wall_area() - 75.6).abs() < 1e-10); // Account for floating point
        assert_eq!(geo.roof_area(), 48.0);
    }

    #[test]
    fn test_case_spec_validation() {
        let spec = CaseBuilder::case_600_baseline();
        assert!(spec.validate().is_ok());

        // Test invalid geometry
        let invalid_geo = GeometrySpec::new(0.0, 6.0, 2.7);
        let invalid_spec = CaseSpec {
            geometry: invalid_geo,
            ..spec.clone()
        };
        assert!(invalid_spec.validate().is_err());

        // Test invalid HVAC setpoints
        let invalid_hvac = HvacSchedule::constant(25.0, 20.0); // Heating > cooling
        let invalid_spec2 = CaseSpec {
            hvac: invalid_hvac,
            ..spec.clone()
        };
        assert!(invalid_spec2.validate().is_err());
    }

    #[test]
    fn test_case_spec_methods() {
        let spec = CaseBuilder::case_600_baseline();
        assert_eq!(spec.case_id, "600");
        assert_eq!(spec.total_window_area(), 12.0);
        assert_eq!(spec.window_area_by_orientation(Orientation::South), 12.0);
        assert_eq!(spec.window_area_by_orientation(Orientation::North), 0.0);
        assert!(!spec.is_free_floating());
        assert!(!spec.has_night_ventilation());
        assert!(!spec.has_shading());

        let ff_spec = CaseBuilder::case_600ff();
        assert!(ff_spec.is_free_floating());

        let vent_spec = CaseBuilder::case_650_night_vent();
        assert!(vent_spec.has_night_ventilation());

        let shade_spec = CaseBuilder::case_610_south_shading();
        assert!(shade_spec.has_shading());
    }

    #[test]
    fn test_case_builder() {
        let spec = CaseBuilder::new()
            .with_case_id("custom".to_string())
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_south_window(12.0)
            .with_hvac_setpoints(20.0, 27.0)
            .with_infiltration(0.5)
            .build()
            .unwrap();

        assert_eq!(spec.case_id, "custom");
        assert_eq!(spec.geometry.floor_area(), 48.0);
        assert_eq!(spec.total_window_area(), 12.0);
    }

    #[test]
    fn test_case_builder_missing_geometry() {
        let result = CaseBuilder::new()
            .with_case_id("invalid".to_string())
            .low_mass_construction()
            .with_south_window(12.0)
            .build();

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Geometry"));
    }

    #[test]
    fn test_all_case_presets() {
        // Test that all case presets can be built successfully
        let cases = vec![
            CaseBuilder::case_600_baseline(),
            CaseBuilder::case_610_south_shading(),
            CaseBuilder::case_620_ew_windows(),
            CaseBuilder::case_630_ew_shading(),
            CaseBuilder::case_640_setback(),
            CaseBuilder::case_650_night_vent(),
            CaseBuilder::case_600ff(),
            CaseBuilder::case_650ff(),
            CaseBuilder::case_900_baseline(),
            CaseBuilder::case_910_south_shading(),
            CaseBuilder::case_920_ew_windows(),
            CaseBuilder::case_930_ew_shading(),
            CaseBuilder::case_940_setback(),
            CaseBuilder::case_950_night_vent(),
            CaseBuilder::case_900ff(),
            CaseBuilder::case_950ff(),
            CaseBuilder::case_960_sunspace(),
            CaseBuilder::case_195_solid_conduction(),
        ];

        assert_eq!(cases.len(), 18);

        // Verify all validate
        for case in cases {
            assert!(
                case.validate().is_ok(),
                "Case {} should validate",
                case.case_id
            );
        }
    }

    #[test]
    fn test_ashrae_case_spec() {
        // Test that ASHRAE140Case enum can generate specs
        let spec = ASHRAE140Case::Case600.spec();
        assert_eq!(spec.case_id, "600");

        let spec = ASHRAE140Case::Case960.spec();
        assert_eq!(spec.case_id, "960");
        assert_eq!(spec.num_zones, 2);

        let spec = ASHRAE140Case::Case195.spec();
        assert_eq!(spec.case_id, "195");
        assert_eq!(spec.infiltration_ach, 0.0);
        assert_eq!(spec.internal_loads.unwrap().total_load, 0.0);
    }

    #[test]
    fn test_low_mass_vs_high_mass() {
        let low_mass = ASHRAE140Case::Case600.spec();
        let high_mass = ASHRAE140Case::Case900.spec();

        // Both should have the same geometry
        assert_eq!(
            low_mass.geometry.floor_area(),
            high_mass.geometry.floor_area()
        );

        // But different construction U-values
        assert_ne!(
            low_mass.construction.wall_u_value(),
            high_mass.construction.wall_u_value()
        );
    }
}
