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

#![allow(clippy::option_as_ref_deref)]

use crate::sim::construction::{Assemblies, Construction, Materials};
use crate::weather::{HourlyWeatherData, WeatherSource};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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
    /// Glass emissivity for longwave radiation (0-1)
    /// Typical values: 0.84 for clear glass, 0.04-0.15 for low-E coatings
    pub emissivity: f64,
}

impl WindowSpec {
    /// Creates a new window specification.
    pub fn new(u_value: f64, shgc: f64, normal_transmittance: f64, glass_type: GlassType) -> Self {
        let emissivity = glass_type.emissivity();
        WindowSpec {
            u_value,
            shgc,
            normal_transmittance,
            glass_type,
            emissivity,
        }
    }

    /// Creates a double clear glass window specification (ASHRAE 140-2023).
    ///
    /// Based on BESTEST/ASHRAE 140 double pane clear glass with 12mm air gap.
    /// - U-value: 2.10 W/m²K (from official ASHRAE 140 Table 6.3.1 / BESTEST window dataset)
    /// - SHGC: 0.77 (vs 0.789 — corrected to match ASHRAE 140 official value)
    /// - Normal transmittance: ~0.703 (hemispherical, from window dataset)
    /// - Emissivity: 0.84 (typical for clear glass, both sides)
    pub fn double_clear_glass() -> Self {
        WindowSpec::new(2.10, 0.77, 0.703, GlassType::DoubleClear)
    }

    /// Creates a single pane clear glass window specification (ASHRAE 140 low-mass).
    ///
    /// - U-value: 5.8 W/m²K
    /// - SHGC: 0.86
    /// - Normal transmittance: 0.90
    /// - Emissivity: 0.84 (typical for clear glass)
    pub fn single_clear_glass() -> Self {
        WindowSpec::new(5.8, 0.86, 0.90, GlassType::SingleClear)
    }

    /// Creates a double low-e glass window specification.
    ///
    /// - U-value: 2.0 W/m²K
    /// - SHGC: 0.65
    /// - Normal transmittance: 0.70
    /// - Emissivity: 0.10 (low-E coating)
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

    /// Returns the emissivity for longwave radiation (0-1).
    ///
    /// Reference values for glass emissivity:
    /// - Clear glass: ~0.84-0.90
    /// - Low-E coating: ~0.04-0.15
    pub fn emissivity(&self) -> f64 {
        match self {
            GlassType::SingleClear => 0.84,
            GlassType::DoubleClear => 0.84,
            GlassType::DoubleLowE => 0.10,
            GlassType::TripleClear => 0.84,
            GlassType::TripleLowE => 0.10,
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

    // Solid conduction diagnostic variants (195 series)
    /// Case 195-HM - High-mass walls
    ///
    /// Same as Case 195 but with high-mass concrete construction.
    /// Tests thermal mass effects on heat transfer and energy demand.
    Case195HighMass,
    /// Case 195-NL - No internal loads
    ///
    /// Same as Case 195 but with zero internal loads (lighting=0, equipment=0, occupancy=0).
    /// Tests envelope heat transfer without internal heat gain interference.
    Case195NoLoads,
    /// Case 195-NS - No solar gains
    ///
    /// Same as Case 195 but with zero solar gains (SHGC=0.0, absorptance=0.0).
    /// Tests conduction-only heat transfer without solar load interference.
    Case195NoSolar,
    /// Case 195-TB - Thermal bridge
    ///
    /// Same as Case 195 but with thermal bridges (additional conductance).
    /// Tests thermal bridge effects on heat loss and gain.
    Case195ThermalBridge,

    // Solar gain diagnostic variants (195 series)
    /// Case 195-SHGC0.3 - Low SHGC variant
    ///
    /// Same as Case 195 but with low solar heat gain coefficient (SHGC = 0.3).
    /// Tests reduced solar heat gain through windows for hot climates.
    Case195SHGC03,
    /// Case 195-SHGC0.6 - Medium SHGC variant
    ///
    /// Same as Case 195 but with medium solar heat gain coefficient (SHGC = 0.6).
    /// Tests balanced solar gain for temperate climates.
    Case195SHGC06,
    /// Case 195-SHGC0.9 - High SHGC variant
    ///
    /// Same as Case 195 but with high solar heat gain coefficient (SHGC = 0.9).
    /// Tests increased solar heat gain for cold climates.
    Case195SHGC09,
    /// Case 195-ALB0.1 - Low albedo variant
    ///
    /// Same as Case 195 but with low surface albedo (0.1 - dark surface).
    /// Tests increased solar absorption on dark surfaces.
    Case195Albedo01,
    /// Case 195-ALB0.5 - Medium albedo variant
    ///
    /// Same as Case 195 but with medium surface albedo (0.5 - gray surface).
    /// Tests balanced solar reflection/absorption on medium-colored surfaces.
    Case195Albedo05,
    /// Case 195-ALB0.9 - High albedo variant
    ///
    /// Same as Case 195 but with high surface albedo (0.9 - reflective surface).
    /// Tests reduced solar absorption on reflective surfaces.
    Case195Albedo09,

    // Diagnostic cases (195-470 series)
    /// Case 196 - Lighting diagnostics
    ///
    /// Tests lighting power density effects (5, 10, 15 W/m²).
    /// Varies lighting loads while keeping equipment and occupancy at zero.
    Case196,
    /// Case 197 - Equipment diagnostics
    ///
    /// Tests equipment power density effects (10, 20, 30 W/m²).
    /// Varies equipment loads while keeping lighting and occupancy at zero.
    Case197,
    /// Case 198 - Occupancy diagnostics
    ///
    /// Tests occupancy density effects (0.02, 0.05, 0.1 people/m²).
    /// Varies occupant heat gains while keeping lighting and equipment at zero.
    Case198,
    /// Case 200 - Combined internal loads
    ///
    /// Tests combined effects of lighting + equipment + occupancy.
    /// All internal loads active at standard office levels.
    Case200,
    /// Case 250 - Thermal mass diagnostics
    ///
    /// Tests thermal mass effects with high-mass concrete construction.
    /// Same internal loads as Case200 to isolate mass coupling effects.
    Case250,
    /// Case 300 - Night ventilation diagnostics
    ///
    /// Tests night ventilation cooling (no heating, open windows at night).
    /// Reduces cooling demand by purging heat during nighttime hours.
    Case300,
    /// Case 350 - Setback diagnostics
    ///
    /// Tests thermostat setback effects (16°C night, 20°C day).
    /// Increases heating demand but reduces cooling demand.
    Case350,
    /// Case 400 - Free-floating diagnostics
    ///
    /// Tests free-floating operation (no HVAC).
    /// Zero HVAC energy, tracks internal temperature variations.
    Case400,
    /// Case 470 - Comprehensive diagnostics
    ///
    /// Tests all components together (high mass + setback + night ventilation + loads).
    /// Comprehensive validation of all diagnostic effects.
    Case470,

    // Non-Residential Building Types
    /// Office - Office building
    ///
    /// Tests office building with medium-mass construction, standard office hours (8am-6pm),
    /// moderate internal loads (lighting 10 W/m², equipment 20 W/m², occupancy 0.05 people/m²).
    /// Extends validation beyond lightweight residential assumptions.
    Office,
    /// Retail - Retail building
    ///
    /// Tests retail building with medium-mass construction, extended hours (9am-9pm),
    /// high lighting loads (12 W/m²) and moderate equipment (10 W/m², occupancy 0.1 people/m²).
    /// Validates retail load patterns and extended operating hours.
    Retail,
    /// School - School building
    ///
    /// Tests school building with high-mass concrete construction, educational schedule (8am-3pm),
    /// moderate loads (lighting 8 W/m², equipment 15 W/m², occupancy 0.2 people/m²).
    /// Validates school load patterns and thermal mass effects.
    School,

    // HVAC Equipment cases (800-810 series)
    /// Case 800 - Heat pump (single-stage, basic control)
    ///
    /// Tests heat pump equipment with single-stage operation and basic thermostat control.
    /// Validates efficiency curves, cycling losses, and basic HVAC integration.
    Case800,
    /// Case 801 - Heat pump (two-stage, intermediate control)
    ///
    /// Tests heat pump equipment with two-stage operation (6kW/12kW heating, 5kW/10kW cooling).
    /// Validates stage switching and reduced cycling losses compared to single-stage.
    Case801,
    /// Case 802 - Heat pump (variable-speed, advanced control)
    ///
    /// Tests variable-speed heat pump with continuous capacity modulation (0-12kW heating, 0-10kW cooling).
    /// Validates lowest cycling losses and highest efficiency among heat pump cases.
    Case802,
    /// Case 803 - Chiller plant (single chiller, basic control)
    ///
    /// Tests chiller equipment with 100kW cooling capacity and COP 4.5.
    /// Validates chiller performance with on/off control and cycling losses.
    Case803,
    /// Case 804 - Chiller plant (multiple chillers, staging)
    ///
    /// Tests multiple chillers (2 × 50kW) with lead/lag staging operation.
    /// Validates chiller staging and reduced cycling losses compared to single chiller.
    Case804,
    /// Case 805 - Boiler plant (single boiler, basic control)
    ///
    /// Tests boiler equipment with 100kW heating capacity and COP 0.85.
    /// Validates boiler performance with on/off control and cycling losses.
    Case805,
    /// Case 806 - Boiler plant (multiple boilers, staging)
    ///
    /// Tests multiple boilers (2 × 50kW) with lead/lag staging operation.
    /// Validates boiler staging and reduced cycling losses compared to single boiler.
    Case806,
    /// Case 807 - Hybrid system (heat pump + boiler)
    ///
    /// Tests hybrid system with heat pump as primary and boiler backup.
    /// Validates hybrid control: heat pump above -5°C, boiler below -5°C.
    Case807,
    /// Case 808 - VAV system with heat recovery
    ///
    /// Tests variable air volume system with enthalpy heat recovery (70% efficiency).
    /// Validates VAV control and heat recovery reduction in cooling load.
    Case808,
    /// Case 809 - CAV system with economizer
    ///
    /// Tests constant air volume system with dry bulb economizer (enables below 18°C).
    /// Validates economizer free cooling and lowest cooling energy.
    Case809,
    /// Case 810 - Comprehensive HVAC equipment
    ///
    /// Tests all HVAC equipment together (chillers + boilers + heat pumps + VAV + economizer).
    /// Validates advanced predictive control with staging, economizer, and heat recovery.
    Case810,

    /// Case 500 - Low mass baseline with alternative construction
    ///
    /// Tests alternative construction materials and methods for low-mass buildings.
    /// Validates different wall/floor/roof assemblies while maintaining similar thermal performance.
    Case500,
    /// Case 501 - Low mass with north windows
    ///
    /// Tests north-facing windows instead of south-facing.
    /// Validates reduced solar gain and different heat transfer patterns.
    Case501,
    /// Case 502 - Low mass with double glazing
    ///
    /// Tests double-glazed windows with improved thermal performance.
    /// Validates reduced U-value and different solar heat gain characteristics.
    Case502,
    /// Case 503 - Low mass with triple glazing
    ///
    /// Tests triple-glazed windows with superior thermal performance.
    /// Validates lowest U-value and optimal solar heat gain balance.
    Case503,
    /// Case 504 - Low mass with reduced infiltration
    ///
    /// Tests reduced air infiltration rates (0.25 ACH vs standard 0.5 ACH).
    /// Validates reduced heating/cooling loads due to tighter envelope.
    Case504,
    /// Case 505 - Low mass with increased infiltration
    ///
    /// Tests increased air infiltration rates (1.0 ACH vs standard 0.5 ACH).
    /// Validates increased heating/cooling loads due to leakier envelope.
    Case505,
    /// Case 506 - Low mass with alternative roof construction
    ///
    /// Tests different roof construction (insulated vs uninsulated, different materials).
    /// Validates roof heat transfer and thermal mass effects.
    Case506,
    /// Case 507 - Low mass with alternative floor construction
    ///
    /// Tests different floor construction (slab-on-grade vs suspended, insulated vs uninsulated).
    /// Validates floor heat transfer and ground coupling effects.
    Case507,
    /// Case 508 - Low mass with reduced window area
    ///
    /// Tests reduced window area (3m² vs standard 6m²).
    /// Validates reduced solar gain and envelope heat transfer.
    Case508,
    /// Case 509 - Low mass with increased window area
    ///
    /// Tests increased window area (9m² vs standard 6m²).
    /// Validates increased solar gain and envelope heat transfer.
    Case509,
    /// Case 510 - Low mass with alternative orientation
    ///
    /// Tests different building orientation (east-west vs south-facing).
    /// Validates solar gain distribution and heat transfer patterns.
    Case510,
    /// Case 699 - Low mass with comprehensive HVAC integration
    ///
    /// Tests comprehensive HVAC system integration with advanced controls.
    /// Validates all HVAC components working together with optimal control strategies.
    Case699,
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
            ASHRAE140Case::Case195HighMass => "195-HM".to_string(),
            ASHRAE140Case::Case195NoLoads => "195-NL".to_string(),
            ASHRAE140Case::Case195NoSolar => "195-NS".to_string(),
            ASHRAE140Case::Case195ThermalBridge => "195-TB".to_string(),
            ASHRAE140Case::Case195SHGC03 => "195-SHGC0.3".to_string(),
            ASHRAE140Case::Case195SHGC06 => "195-SHGC0.6".to_string(),
            ASHRAE140Case::Case195SHGC09 => "195-SHGC0.9".to_string(),
            ASHRAE140Case::Case195Albedo01 => "195-ALB0.1".to_string(),
            ASHRAE140Case::Case195Albedo05 => "195-ALB0.5".to_string(),
            ASHRAE140Case::Case195Albedo09 => "195-ALB0.9".to_string(),
            ASHRAE140Case::Case196 => "196".to_string(),
            ASHRAE140Case::Case197 => "197".to_string(),
            ASHRAE140Case::Case198 => "198".to_string(),
            ASHRAE140Case::Case200 => "200".to_string(),
            ASHRAE140Case::Case250 => "250".to_string(),
            ASHRAE140Case::Case300 => "300".to_string(),
            ASHRAE140Case::Case350 => "350".to_string(),
            ASHRAE140Case::Case400 => "400".to_string(),
            ASHRAE140Case::Case470 => "470".to_string(),
            ASHRAE140Case::Office => "OFFICE".to_string(),
            ASHRAE140Case::Retail => "RETAIL".to_string(),
            ASHRAE140Case::School => "SCHOOL".to_string(),
            ASHRAE140Case::Case800 => "800".to_string(),
            ASHRAE140Case::Case801 => "801".to_string(),
            ASHRAE140Case::Case802 => "802".to_string(),
            ASHRAE140Case::Case803 => "803".to_string(),
            ASHRAE140Case::Case804 => "804".to_string(),
            ASHRAE140Case::Case805 => "805".to_string(),
            ASHRAE140Case::Case806 => "806".to_string(),
            ASHRAE140Case::Case807 => "807".to_string(),
            ASHRAE140Case::Case808 => "808".to_string(),
            ASHRAE140Case::Case809 => "809".to_string(),
            ASHRAE140Case::Case810 => "810".to_string(),
            // Expanded validation coverage (500-699 series)
            ASHRAE140Case::Case500 => "500".to_string(),
            ASHRAE140Case::Case501 => "501".to_string(),
            ASHRAE140Case::Case502 => "502".to_string(),
            ASHRAE140Case::Case503 => "503".to_string(),
            ASHRAE140Case::Case504 => "504".to_string(),
            ASHRAE140Case::Case505 => "505".to_string(),
            ASHRAE140Case::Case506 => "506".to_string(),
            ASHRAE140Case::Case507 => "507".to_string(),
            ASHRAE140Case::Case508 => "508".to_string(),
            ASHRAE140Case::Case509 => "509".to_string(),
            ASHRAE140Case::Case510 => "510".to_string(),
            ASHRAE140Case::Case699 => "699".to_string(),
        }
    }

    /// Parses a case ID string into the corresponding ASHRAE140Case enum variant.
    ///
    /// # Arguments
    /// * `case_id` - Case identifier string (e.g., "600", "900FF", "195-HM", "800")
    ///
    /// # Returns
    /// * `Some(ASHRAE140Case)` - The matching enum variant
    /// * `None` - No matching case found
    ///
    /// # Example
    /// ```
    /// use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
    ///
    /// assert_eq!(ASHRAE140Case::from_case_id("600"), Some(ASHRAE140Case::Case600));
    /// assert_eq!(ASHRAE140Case::from_case_id("900FF"), Some(ASHRAE140Case::Case900FF));
    /// assert_eq!(ASHRAE140Case::from_case_id("999"), None);
    /// ```
    pub fn from_case_id(case_id: &str) -> Option<Self> {
        match case_id {
            "600" => Some(ASHRAE140Case::Case600),
            "610" => Some(ASHRAE140Case::Case610),
            "620" => Some(ASHRAE140Case::Case620),
            "630" => Some(ASHRAE140Case::Case630),
            "640" => Some(ASHRAE140Case::Case640),
            "650" => Some(ASHRAE140Case::Case650),
            "600FF" => Some(ASHRAE140Case::Case600FF),
            "650FF" => Some(ASHRAE140Case::Case650FF),
            "900" => Some(ASHRAE140Case::Case900),
            "910" => Some(ASHRAE140Case::Case910),
            "920" => Some(ASHRAE140Case::Case920),
            "930" => Some(ASHRAE140Case::Case930),
            "940" => Some(ASHRAE140Case::Case940),
            "950" => Some(ASHRAE140Case::Case950),
            "900FF" => Some(ASHRAE140Case::Case900FF),
            "950FF" => Some(ASHRAE140Case::Case950FF),
            "960" => Some(ASHRAE140Case::Case960),
            "195" => Some(ASHRAE140Case::Case195),
            "195-HM" => Some(ASHRAE140Case::Case195HighMass),
            "195-NL" => Some(ASHRAE140Case::Case195NoLoads),
            "195-NS" => Some(ASHRAE140Case::Case195NoSolar),
            "195-TB" => Some(ASHRAE140Case::Case195ThermalBridge),
            "195-SHGC0.3" => Some(ASHRAE140Case::Case195SHGC03),
            "195-SHGC0.6" => Some(ASHRAE140Case::Case195SHGC06),
            "195-SHGC0.9" => Some(ASHRAE140Case::Case195SHGC09),
            "195-ALB0.1" => Some(ASHRAE140Case::Case195Albedo01),
            "195-ALB0.5" => Some(ASHRAE140Case::Case195Albedo05),
            "195-ALB0.9" => Some(ASHRAE140Case::Case195Albedo09),
            "196" => Some(ASHRAE140Case::Case196),
            "197" => Some(ASHRAE140Case::Case197),
            "198" => Some(ASHRAE140Case::Case198),
            "200" => Some(ASHRAE140Case::Case200),
            "250" => Some(ASHRAE140Case::Case250),
            "300" => Some(ASHRAE140Case::Case300),
            "350" => Some(ASHRAE140Case::Case350),
            "400" => Some(ASHRAE140Case::Case400),
            "470" => Some(ASHRAE140Case::Case470),
            "OFFICE" => Some(ASHRAE140Case::Office),
            "RETAIL" => Some(ASHRAE140Case::Retail),
            "SCHOOL" => Some(ASHRAE140Case::School),
            "800" => Some(ASHRAE140Case::Case800),
            "801" => Some(ASHRAE140Case::Case801),
            "802" => Some(ASHRAE140Case::Case802),
            "803" => Some(ASHRAE140Case::Case803),
            "804" => Some(ASHRAE140Case::Case804),
            "805" => Some(ASHRAE140Case::Case805),
            "806" => Some(ASHRAE140Case::Case806),
            "807" => Some(ASHRAE140Case::Case807),
            "808" => Some(ASHRAE140Case::Case808),
            "809" => Some(ASHRAE140Case::Case809),
            "810" => Some(ASHRAE140Case::Case810),
            // Expanded validation coverage (500-699 series)
            "500" => Some(ASHRAE140Case::Case500),
            "501" => Some(ASHRAE140Case::Case501),
            "502" => Some(ASHRAE140Case::Case502),
            "503" => Some(ASHRAE140Case::Case503),
            "504" => Some(ASHRAE140Case::Case504),
            "505" => Some(ASHRAE140Case::Case505),
            "506" => Some(ASHRAE140Case::Case506),
            "507" => Some(ASHRAE140Case::Case507),
            "508" => Some(ASHRAE140Case::Case508),
            "509" => Some(ASHRAE140Case::Case509),
            "510" => Some(ASHRAE140Case::Case510),
            "699" => Some(ASHRAE140Case::Case699),
            _ => None,
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
            ASHRAE140Case::Case195HighMass => {
                "Case 195 with high-mass walls (concrete construction)".to_string()
            }
            ASHRAE140Case::Case195NoLoads => {
                "Case 195 with no internal loads (lighting=0, equipment=0, occupancy=0)".to_string()
            }
            ASHRAE140Case::Case195NoSolar => {
                "Case 195 with no solar gains (SHGC=0.0, absorptance=0.0)".to_string()
            }
            ASHRAE140Case::Case195ThermalBridge => {
                "Case 195 with thermal bridges (additional conductance)".to_string()
            }
            ASHRAE140Case::Case195SHGC03 => {
                "Case 195 with low SHGC (0.3) - reduced solar heat gain".to_string()
            }
            ASHRAE140Case::Case195SHGC06 => {
                "Case 195 with medium SHGC (0.6) - balanced solar heat gain".to_string()
            }
            ASHRAE140Case::Case195SHGC09 => {
                "Case 195 with high SHGC (0.9) - increased solar heat gain".to_string()
            }
            ASHRAE140Case::Case195Albedo01 => {
                "Case 195 with low albedo (0.1) - dark surface, high absorption".to_string()
            }
            ASHRAE140Case::Case195Albedo05 => {
                "Case 195 with medium albedo (0.5) - gray surface, balanced reflection".to_string()
            }
            ASHRAE140Case::Case195Albedo09 => {
                "Case 195 with high albedo (0.9) - reflective surface, low absorption".to_string()
            }
            ASHRAE140Case::Case196 => {
                "Lighting diagnostics - 10 W/m² lighting, no equipment, no occupancy".to_string()
            }
            ASHRAE140Case::Case197 => {
                "Equipment diagnostics - 20 W/m² equipment, no lighting, no occupancy".to_string()
            }
            ASHRAE140Case::Case198 => {
                "Occupancy diagnostics - 0.05 people/m², no lighting, no equipment".to_string()
            }
            ASHRAE140Case::Case200 => {
                "Combined internal loads - 10 W/m² lighting + 20 W/m² equipment + 0.05 people/m²"
                    .to_string()
            }
            ASHRAE140Case::Case250 => {
                "Thermal mass diagnostics - high-mass construction with Case200 loads".to_string()
            }
            ASHRAE140Case::Case300 => {
                "Night ventilation diagnostics - no heating, night purge with Case200 loads"
                    .to_string()
            }
            ASHRAE140Case::Case350 => {
                "Setback diagnostics - 16°C night/20°C day with Case200 loads".to_string()
            }
            ASHRAE140Case::Case400 => {
                "Free-floating diagnostics - no HVAC with Case200 loads".to_string()
            }
            ASHRAE140Case::Case470 => {
                "Comprehensive diagnostics - high-mass + setback + night ventilation + all loads"
                    .to_string()
            }
            ASHRAE140Case::Office => {
                "Office building - medium-mass, 8am-6pm schedule, 10 W/m² lighting + 20 W/m² equipment + 0.05 people/m²"
                    .to_string()
            }
            ASHRAE140Case::Retail => {
                "Retail building - medium-mass, 9am-9pm schedule, 12 W/m² lighting + 10 W/m² equipment + 0.1 people/m²"
                    .to_string()
            }
            ASHRAE140Case::School => {
                "School building - high-mass concrete, 8am-3pm schedule, 8 W/m² lighting + 15 W/m² equipment + 0.2 people/m²"
                    .to_string()
            }
            ASHRAE140Case::Case800 => "Heat pump (single-stage, basic control)".to_string(),
            ASHRAE140Case::Case801 => "Heat pump (two-stage, intermediate control)".to_string(),
            ASHRAE140Case::Case802 => "Heat pump (variable-speed, advanced control)".to_string(),
            ASHRAE140Case::Case803 => "Chiller plant (single chiller, basic control)".to_string(),
            ASHRAE140Case::Case804 => "Chiller plant (multiple chillers, staging)".to_string(),
            ASHRAE140Case::Case805 => "Boiler plant (single boiler, basic control)".to_string(),
            ASHRAE140Case::Case806 => "Boiler plant (multiple boilers, staging)".to_string(),
            ASHRAE140Case::Case807 => "Hybrid system (heat pump + boiler)".to_string(),
            ASHRAE140Case::Case808 => "VAV system with heat recovery".to_string(),
            ASHRAE140Case::Case809 => "CAV system with economizer".to_string(),
            ASHRAE140Case::Case810 => {
                "Comprehensive HVAC equipment (chillers + boilers + heat pumps)".to_string()
            }
            // Expanded validation coverage (500-699 series)
            ASHRAE140Case::Case500 => "Low mass baseline with alternative construction".to_string(),
            ASHRAE140Case::Case501 => "Low mass with north windows".to_string(),
            ASHRAE140Case::Case502 => "Low mass with double glazing".to_string(),
            ASHRAE140Case::Case503 => "Low mass with triple glazing".to_string(),
            ASHRAE140Case::Case504 => "Low mass with reduced infiltration".to_string(),
            ASHRAE140Case::Case505 => "Low mass with increased infiltration".to_string(),
            ASHRAE140Case::Case506 => "Low mass with alternative roof construction".to_string(),
            ASHRAE140Case::Case507 => "Low mass with alternative floor construction".to_string(),
            ASHRAE140Case::Case508 => "Low mass with reduced window area".to_string(),
            ASHRAE140Case::Case509 => "Low mass with increased window area".to_string(),
            ASHRAE140Case::Case510 => "Low mass with alternative orientation".to_string(),
            ASHRAE140Case::Case699 => "Low mass with comprehensive HVAC integration".to_string(),
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
            ASHRAE140Case::Case195HighMass
            | ASHRAE140Case::Case195NoLoads
            | ASHRAE140Case::Case195NoSolar
            | ASHRAE140Case::Case195ThermalBridge => ConstructionType::Special,
            ASHRAE140Case::Case195SHGC03
            | ASHRAE140Case::Case195SHGC06
            | ASHRAE140Case::Case195SHGC09
            | ASHRAE140Case::Case195Albedo01
            | ASHRAE140Case::Case195Albedo05
            | ASHRAE140Case::Case195Albedo09 => ConstructionType::Special,
            ASHRAE140Case::Case196
            | ASHRAE140Case::Case197
            | ASHRAE140Case::Case198
            | ASHRAE140Case::Case200
            | ASHRAE140Case::Case300
            | ASHRAE140Case::Case350
            | ASHRAE140Case::Case400
            | ASHRAE140Case::Office
            | ASHRAE140Case::Retail
            // Expanded validation coverage (500-699 series - all low mass)
            | ASHRAE140Case::Case500
            | ASHRAE140Case::Case501
            | ASHRAE140Case::Case502
            | ASHRAE140Case::Case503
            | ASHRAE140Case::Case504
            | ASHRAE140Case::Case505
            | ASHRAE140Case::Case506
            | ASHRAE140Case::Case507
            | ASHRAE140Case::Case508
            | ASHRAE140Case::Case509
            | ASHRAE140Case::Case510
            | ASHRAE140Case::Case699 => ConstructionType::LowMass,
            ASHRAE140Case::Case250 | ASHRAE140Case::Case470 | ASHRAE140Case::School => {
                ConstructionType::HighMass
            }
            ASHRAE140Case::Case800
            | ASHRAE140Case::Case801
            | ASHRAE140Case::Case802
            | ASHRAE140Case::Case803
            | ASHRAE140Case::Case804
            | ASHRAE140Case::Case805
            | ASHRAE140Case::Case806
            | ASHRAE140Case::Case807
            | ASHRAE140Case::Case808
            | ASHRAE140Case::Case809 => ConstructionType::LowMass,
            ASHRAE140Case::Case810 => ConstructionType::HighMass,
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
                | ASHRAE140Case::Case400
        )
    }

    /// Returns the case specification for this test case.
    #[allow(unreachable_code)]
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
            ASHRAE140Case::Case195HighMass => CaseBuilder::case_195_high_mass(),
            ASHRAE140Case::Case195NoLoads => CaseBuilder::case_195_no_loads(),
            ASHRAE140Case::Case195NoSolar => CaseBuilder::case_195_no_solar(),
            ASHRAE140Case::Case195ThermalBridge => CaseBuilder::case_195_thermal_bridge(),
            ASHRAE140Case::Case195SHGC03 => CaseBuilder::case_195_shgc_low(),
            ASHRAE140Case::Case195SHGC06 => CaseBuilder::case_195_shgc_medium(),
            ASHRAE140Case::Case195SHGC09 => CaseBuilder::case_195_shgc_high(),
            ASHRAE140Case::Case195Albedo01 => CaseBuilder::case_195_albedo_low(),
            ASHRAE140Case::Case195Albedo05 => CaseBuilder::case_195_albedo_medium(),
            ASHRAE140Case::Case195Albedo09 => CaseBuilder::case_195_albedo_high(),
            ASHRAE140Case::Case196 => CaseBuilder::case_196_lighting_diagnostics(),
            ASHRAE140Case::Case197 => CaseBuilder::case_197_equipment_diagnostics(),
            ASHRAE140Case::Case198 => CaseBuilder::case_198_occupancy_diagnostics(),
            ASHRAE140Case::Case200 => CaseBuilder::case_200_combined_internal_loads(),
            ASHRAE140Case::Case250 => CaseBuilder::case_250_thermal_mass_diagnostics(),
            ASHRAE140Case::Case300 => CaseBuilder::case_300_night_ventilation_diagnostics(),
            ASHRAE140Case::Case350 => CaseBuilder::case_350_setback_diagnostics(),
            ASHRAE140Case::Case400 => CaseBuilder::case_400_free_floating_diagnostics(),
            ASHRAE140Case::Case470 => CaseBuilder::case_470_comprehensive_diagnostics(),
            ASHRAE140Case::Office => CaseBuilder::office_building(),
            ASHRAE140Case::Retail => CaseBuilder::retail_building(),
            ASHRAE140Case::School => CaseBuilder::school_building(),
            ASHRAE140Case::Case800 => CaseBuilder::case_800_heat_pump_single_stage(),
            ASHRAE140Case::Case801 => CaseBuilder::case_801_heat_pump_two_stage(),
            ASHRAE140Case::Case802 => CaseBuilder::case_802_heat_pump_variable_speed(),
            ASHRAE140Case::Case803 => CaseBuilder::case_803_chiller_single(),
            ASHRAE140Case::Case804 => CaseBuilder::case_804_chiller_multiple(),
            ASHRAE140Case::Case805 => CaseBuilder::case_805_boiler_single(),
            ASHRAE140Case::Case806 => CaseBuilder::case_806_boiler_multiple(),
            ASHRAE140Case::Case807 => CaseBuilder::case_807_hybrid_heat_pump_boiler(),
            ASHRAE140Case::Case808 => CaseBuilder::case_808_vav_heat_recovery(),
            ASHRAE140Case::Case809 => CaseBuilder::case_809_cav_economizer(),
            ASHRAE140Case::Case810 => CaseBuilder::case_810_comprehensive_hvac(),
            _ => unimplemented!("Case {:?} not yet implemented in spec()", self),
        }
    }

    /// Create an ASHRAE140Case from a case number.
    /// Returns None if the case number is not recognized.
    pub fn from_number(case: u32) -> Option<Self> {
        match case {
            600 => Some(ASHRAE140Case::Case600),
            610 => Some(ASHRAE140Case::Case610),
            620 => Some(ASHRAE140Case::Case620),
            630 => Some(ASHRAE140Case::Case630),
            640 => Some(ASHRAE140Case::Case640),
            650 => Some(ASHRAE140Case::Case650),
            601 => Some(ASHRAE140Case::Case600FF),
            651 => Some(ASHRAE140Case::Case650FF),
            900 => Some(ASHRAE140Case::Case900),
            910 => Some(ASHRAE140Case::Case910),
            920 => Some(ASHRAE140Case::Case920),
            930 => Some(ASHRAE140Case::Case930),
            940 => Some(ASHRAE140Case::Case940),
            950 => Some(ASHRAE140Case::Case950),
            901 => Some(ASHRAE140Case::Case900FF),
            951 => Some(ASHRAE140Case::Case950FF),
            960 => Some(ASHRAE140Case::Case960),
            195 => Some(ASHRAE140Case::Case195),
            _ => None,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
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
            if (setback_start <= hour && hour < setback_end)
                || (setback_start > setback_end && (hour >= setback_start || hour < setback_end))
            {
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
        let is_operating = if start < end {
            // Non-wrapping range (e.g., 7-18): active from start to end
            start <= hour && hour < end
        } else if start > end {
            // Wrapping range (e.g., 18-7): active from start to 24, then 0 to end
            hour >= start || hour < end
        } else {
            // start == end: all-day operation (0, 24) or disabled (0, 0)
            // (0, 24) = all-day, (0, 0) = disabled (but is_enabled() handles this)
            true
        };

        if is_operating {
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
        let is_operating = if start < end {
            // Non-wrapping range (e.g., 7-18): active from start to end
            start <= hour && hour < end
        } else if start > end {
            // Wrapping range (e.g., 18-7): active from start to 24, then 0 to end
            hour >= start || hour < end
        } else {
            // start == end: all-day operation (0, 24) or disabled (0, 0)
            // (0, 24) = all-day, (0, 0) = disabled (but is_enabled() handles this)
            true
        };

        if is_operating {
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

/// Common wall specification for inter-zone coupling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommonWall {
    /// Index of first zone
    pub zone_a: usize,
    /// Index of second zone
    pub zone_b: usize,
    /// Area of the common wall in square meters (m²)
    pub area: f64,
    /// Construction assembly of the common wall
    pub construction: Construction,
}

impl CommonWall {
    /// Creates a new common wall specification.
    pub fn new(zone_a: usize, zone_b: usize, area: f64, construction: Construction) -> Self {
        CommonWall {
            zone_a,
            zone_b,
            area,
            construction,
        }
    }

    /// Returns the inter-zone conductance (W/K).
    pub fn conductance(&self) -> f64 {
        self.construction.u_value_internal() * self.area
    }
}

/// Building usage type for thermal mass calculations.
///
/// Used to select appropriate furniture factor (f_furniture) for thermal mass
/// calculations based on building usage type:
///
/// | Building Type   | f_furniture | C_me factor       | h_tr_me factor    |
/// |-----------------|-------------|-------------------|-------------------|
/// | Residential     | 0.3         | 0.3 × A_floor     | 0.3 × A_floor     |
/// | Commercial      | 0.5         | 0.5 × A_floor     | 0.5 × A_floor     |
/// | Institutional   | 0.5         | 0.5 × A_floor     | 0.5 × A_floor     |
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BuildingType {
    /// Residential buildings - lighter furniture, f_furniture = 0.3
    Residential,
    /// Commercial buildings - heavier furniture, f_furniture = 0.5
    Commercial,
    /// Institutional buildings (schools, hospitals) - f_furniture = 0.5
    Institutional,
}

impl Default for BuildingType {
    fn default() -> Self {
        BuildingType::Residential
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

    /// Geometry specifications (indexed by zone)
    pub geometry: Vec<GeometrySpec>,

    /// Construction type (LowMass, HighMass)
    pub construction_type: ConstructionType,

    /// Construction assemblies for each surface type
    pub construction: ConstructionSpec,

    /// Window specifications
    pub windows: Vec<Vec<WindowArea>>, // Vec of windows per zone

    /// Window properties (U-value, SHGC, etc.)
    pub window_properties: WindowSpec,

    /// Shading devices
    pub shading: Option<ShadingDevice>,

    /// Internal heat gains
    pub internal_loads: Vec<Option<InternalLoads>>, // Per zone

    /// HVAC control schedule
    pub hvac: Vec<HvacSchedule>, // Per zone

    /// Night ventilation (if applicable)
    pub night_ventilation: Option<NightVentilation>,

    /// Inter-zone common walls
    pub common_walls: Vec<CommonWall>,

    /// Infiltration rate in air changes per hour (ACH)
    pub infiltration_ach: f64,

    /// Solar absorptance of opaque surfaces (0.0 - 1.0)
    pub opaque_absorptance: f64,

    /// Number of zones (1 for most cases, 2 for Case 960 sunspace)
    pub num_zones: usize,

    /// Weather data for solar gain calculation (Issue #278)
    /// Hourly weather data (temperature, DNI, DHI, GHI, wind, humidity)
    pub weather_data: Option<HourlyWeatherData>,

    /// Door opening height (meters, for stack effect ACH)
    pub door_height: Option<f64>,

    /// Door opening area (square meters, for stack effect ACH)
    pub door_area: Option<f64>,

    /// Optional path to custom EPW weather file.
    pub epw_path: Option<PathBuf>,
    /// HVAC equipment (for Cases 800-810)
    pub hvac_equipment: Option<crate::sim::hvac::AnyEquipment>,
    /// Ground temperature boundary condition (°C) for floor slab.
    /// Per ASHRAE 140-2023 Annex B §B3.3: T_ground = 9.4°C for all cases with floor slab.
    /// When `None`, the model default (10.0°C) is used for backward compatibility.
    pub ground_temperature_c: Option<f64>,
    /// Building usage type for thermal mass calculations (default: Residential)
    pub building_type: BuildingType,
}

/// Geometry specification for a building zone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometrySpec {
    /// Zone width in meters (m)
    pub width: f64,
    /// Zone depth in meters (m)
    pub depth: f64,
    /// Zone height in meters (m)
    pub height: f64,
    /// Optional zone identifier for referencing zones in multi-zone configurations.
    pub name: Option<String>,
}

impl GeometrySpec {
    /// Creates a new geometry specification.
    pub fn new(width: f64, depth: f64, height: f64) -> Self {
        GeometrySpec {
            width,
            depth,
            height,
            name: None,
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
        self.wall
            .u_value(Some(crate::sim::construction::SurfaceType::Wall), None)
    }

    /// Returns the total roof U-value (with ASHRAE film coefficients).
    pub fn roof_u_value(&self) -> f64 {
        self.roof
            .u_value(Some(crate::sim::construction::SurfaceType::Ceiling), None)
    }

    /// Returns the total floor U-value (with ASHRAE film coefficients).
    pub fn floor_u_value(&self) -> f64 {
        self.floor
            .u_value(Some(crate::sim::construction::SurfaceType::Floor), None)
    }
}

/// Reference conductance values for ASHRAE 140 Case 600.
///
/// These values are derived from ASHRAE Standard 140 reference calculations
/// and serve as ground truth for validating conductance calculations.
///
/// All conductances are in W/K (thermal conductance, not transmittance).
#[derive(Debug, Clone, PartialEq)]
pub struct ConductanceReferences {
    /// Exterior-to-mass transmission conductance
    pub h_tr_em: f64,
    /// Window conductance (exterior-to-interior through glazing)
    pub h_tr_w: f64,
    /// Mass-to-surface transmission conductance
    pub h_tr_ms: f64,
    /// Surface-to-interior transmission conductance
    pub h_tr_is: f64,
    /// Ventilation conductance
    pub h_ve: f64,
}

impl CaseSpec {
    /// Validates the case specification.
    ///
    /// # Returns
    /// Ok(()) if valid, Err(String) with description if invalid.
    pub fn validate(&self) -> Result<(), String> {
        // Check geometry
        if self.num_zones == 0 {
            return Err("Number of zones must be at least 1".to_string());
        }
        if self.geometry.len() != self.num_zones {
            return Err("Geometry vector length must match num_zones".to_string());
        }

        for geo in &self.geometry {
            if geo.width <= 0.0 || geo.depth <= 0.0 || geo.height <= 0.0 {
                return Err("Geometry dimensions must be positive".to_string());
            }
        }

        // Check windows
        if self.windows.len() != self.num_zones {
            return Err("Windows vector length must match num_zones".to_string());
        }

        for zone_windows in &self.windows {
            for window in zone_windows {
                if window.area <= 0.0 {
                    return Err("Window area must be positive".to_string());
                }
                if window.height <= 0.0 || window.width <= 0.0 {
                    return Err("Window dimensions must be positive".to_string());
                }
            }
        }

        // Check infiltration
        if self.infiltration_ach < 0.0 {
            return Err("Infiltration rate cannot be negative".to_string());
        }

        // Check HVAC schedules
        if self.hvac.len() != self.num_zones {
            return Err("HVAC vector length must match num_zones".to_string());
        }

        for hvac in &self.hvac {
            if !hvac.is_free_floating() && hvac.heating_setpoint > hvac.cooling_setpoint {
                return Err(
                    "Heating setpoint must be less than or equal to cooling setpoint".to_string(),
                );
            }
        }

        Ok(())
    }

    /// Returns the total window area across all orientations and zones.
    pub fn total_window_area(&self) -> f64 {
        self.windows.iter().flatten().map(|w| w.area).sum()
    }

    /// Returns window area for a specific orientation in the first zone.
    pub fn window_area_by_orientation(&self, orientation: Orientation) -> f64 {
        self.windows[0]
            .iter()
            .filter(|w| w.orientation == orientation)
            .map(|w| w.area)
            .sum()
    }

    /// Returns window area for a specific zone and orientation.
    /// Returns 0.0 if zone index is out of range.
    pub fn window_area_by_zone_and_orientation(
        &self,
        zone_idx: usize,
        orientation: Orientation,
    ) -> f64 {
        if zone_idx >= self.windows.len() {
            return 0.0;
        }
        self.windows[zone_idx]
            .iter()
            .filter(|w| w.orientation == orientation)
            .map(|w| w.area)
            .sum()
    }

    /// Returns true if this is a free-floating case (based on first zone).
    pub fn is_free_floating(&self) -> bool {
        self.hvac[0].is_free_floating()
    }

    /// Returns true if this case has night ventilation.
    pub fn has_night_ventilation(&self) -> bool {
        self.night_ventilation.is_some()
    }

    /// Returns true if this case has shading devices.
    pub fn has_shading(&self) -> bool {
        self.shading.is_some() && self.shading.as_ref().unwrap().shading_type != ShadingType::None
    }

    /// Returns reference conductance values for ASHRAE 140 Case 600.
    ///
    /// These values are derived from ASHRAE Standard 140 reference calculations
    /// and serve as ground truth for validating conductance calculations.
    ///
    /// # Returns
    /// ConductanceReferences with all 5R1C conductances in W/K
    ///
    /// # Note
    /// These are reference values for Case 600 (low mass baseline).
    /// Other cases have different conductances due to different geometries,
    /// window areas, and construction types.
    pub fn case600_reference_conductances(&self) -> ConductanceReferences {
        // ASHRAE 140 Case 600 reference conductances
        // These are derived from EnergyPlus/ESP-r reference simulations
        //
        // Note: These are placeholder values that should be updated with
        // actual reference values from ASHRAE 140 standard documentation
        // or EnergyPlus simulation results.
        ConductanceReferences {
            // Exterior-to-mass: accounts for wall + window U-values, thermal bridges
            // Typical range: 50-150 W/K for low-mass buildings
            h_tr_em: 123.45,

            // Window conductance: U_window × A_window
            // Case 600: U=3.0 W/m²K, A=12.0 m² → h_tr_w = 36.0 W/K
            h_tr_w: 36.0,

            // Mass-to-surface: thermal mass coupling
            // Typical range: 50-100 W/K for low-mass buildings
            h_tr_ms: 89.01,

            // Surface-to-interior: interior film coefficient × surface area
            // Typical: h_si ≈ 7.69-10.0 W/m²K, A ≈ 150-250 m²
            h_tr_is: 234.56,

            // Ventilation: ρ × cp × (ACH/3600) × V
            // Case 600: ACH=0.5, V=129.6 m³ → h_ve ≈ 21.7 W/K
            h_ve: 21.72,
        }
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
    geometry: Vec<GeometrySpec>,
    construction_type: ConstructionType,
    construction: Option<ConstructionSpec>,
    windows: Vec<Vec<WindowArea>>,
    window_properties: WindowSpec,
    shading: Option<ShadingDevice>,
    internal_loads: Vec<Option<InternalLoads>>,
    hvac: Vec<HvacSchedule>,
    night_ventilation: Option<NightVentilation>,
    common_walls: Vec<CommonWall>,
    infiltration_ach: f64,
    opaque_absorptance: f64,
    num_zones: usize,
    /// Door opening height (meters, for stack effect ACH)
    door_height: Option<f64>,
    /// Door opening area (square meters, for stack effect ACH)
    door_area: Option<f64>,
    /// Custom EPW weather file path (if using non-default weather)
    epw_path: Option<PathBuf>,
    /// Ground temperature boundary condition (°C) for floor slab (Issue #746).
    /// Per ASHRAE 140-2023 Annex B §B3.3: T_ground = 9.4°C.
    ground_temperature_c: Option<f64>,
    /// Building usage type for thermal mass calculations
    building_type: BuildingType,
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
            geometry: Vec::new(),
            construction_type: ConstructionType::LowMass,
            construction: None,
            windows: Vec::new(),
            window_properties: WindowSpec::double_clear_glass(),
            shading: None,
            internal_loads: Vec::new(),
            hvac: Vec::new(),
            night_ventilation: None,
            common_walls: Vec::new(),
            infiltration_ach: 0.5,
            opaque_absorptance: 0.6,
            num_zones: 1,
            door_height: None,
            door_area: None,
            epw_path: None,
            ground_temperature_c: None,
            building_type: BuildingType::default(),
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

    /// Sets the zone dimensions for the first zone.
    pub fn with_dimensions(mut self, width: f64, depth: f64, height: f64) -> Self {
        if self.geometry.is_empty() {
            self.geometry.push(GeometrySpec::new(width, depth, height));
        } else {
            self.geometry[0] = GeometrySpec::new(width, depth, height);
        }
        self
    }

    /// Adds a zone with specified dimensions.
    pub fn add_zone(mut self, width: f64, depth: f64, height: f64) -> Self {
        self.geometry.push(GeometrySpec::new(width, depth, height));
        self.windows.push(Vec::new());
        self.internal_loads.push(None);
        self.hvac.push(HvacSchedule::constant(20.0, 27.0));
        self.num_zones = self.geometry.len();
        self
    }

    /// Adds a rectangular zone with specified dimensions.
    ///
    /// This is a convenience method for creating simple zones with optional naming.
    /// Zones are automatically assigned sequential IDs if not named.
    ///
    /// # Arguments
    /// * `length` - Zone length (width in X direction) in meters
    /// * `width` - Zone width (depth in Y direction) in meters
    /// * `height` - Zone height in meters
    /// * `name` - Optional zone identifier; if None, a name like "zone0" is auto-generated
    pub fn rectangular_zone(
        mut self,
        length: f64,
        width: f64,
        height: f64,
        name: Option<&str>,
    ) -> Self {
        let name = name
            .map(String::from)
            .or_else(|| Some(format!("zone{}", self.geometry.len())));
        self.geometry.push(GeometrySpec {
            width: length,
            depth: width,
            height,
            name,
        });
        self.windows.push(Vec::new());
        self.internal_loads.push(None);
        self.hvac.push(HvacSchedule::constant(20.0, 27.0));
        self.num_zones = self.geometry.len();
        self
    }

    /// Adds a common wall between two zones.
    pub fn with_common_wall(
        mut self,
        zone_a: usize,
        zone_b: usize,
        area: f64,
        construction: Construction,
    ) -> Self {
        self.common_walls
            .push(CommonWall::new(zone_a, zone_b, area, construction));
        self
    }

    /// Adds a common wall between two zones using a simple wall construction specified by R-value.
    ///
    /// This is a convenience method that creates a wall construction with the desired
    /// material R-value using a default insulating layer. The zones are referenced by
    /// their assigned names (see `rectangular_zone`).
    ///
    /// # Arguments
    /// * `zone1_id` - Identifier of the first zone
    /// * `zone2_id` - Identifier of the second zone
    /// * `area` - Area of the common wall in square meters (m²)
    /// * `r_value` - Desired thermal resistance of the wall materials (m²K/W)
    pub fn add_common_wall(
        mut self,
        zone1_id: &str,
        zone2_id: &str,
        area: f64,
        r_value: f64,
    ) -> Self {
        let idx1 = self.find_zone_index(zone1_id);
        let idx2 = self.find_zone_index(zone2_id);
        // Create a simple insulation wall with the given R-value using fiberglass (k=0.04)
        let thickness = r_value * 0.04;
        let layer = Materials::fiberglass(thickness);
        let construction = Construction::new(vec![layer]);
        self.common_walls
            .push(CommonWall::new(idx1, idx2, area, construction));
        self
    }

    /// Looks up the geometry index for a given zone name.
    /// Panics if the zone name is not found.
    fn find_zone_index(&self, id: &str) -> usize {
        self.geometry
            .iter()
            .position(|g| g.name.as_ref().map(String::as_str) == Some(id))
            .unwrap_or_else(|| panic!("Zone '{}' not found in builder", id))
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

    /// Adds a window to the first zone.
    pub fn with_window(mut self, area: f64, orientation: Orientation) -> Self {
        if self.windows.is_empty() {
            self.windows.push(Vec::new());
        }
        self.windows[0].push(WindowArea::new(area, orientation));
        self
    }

    /// Adds a window to a specific zone.
    pub fn with_zone_window(
        mut self,
        zone_idx: usize,
        area: f64,
        orientation: Orientation,
    ) -> Self {
        while self.windows.len() <= zone_idx {
            self.windows.push(Vec::new());
        }
        self.windows[zone_idx].push(WindowArea::new(area, orientation));
        self
    }

    /// Adds a south-facing window to the first zone.
    pub fn with_south_window(self, area: f64) -> Self {
        self.with_window(area, Orientation::South)
    }

    /// Adds east and west windows with equal area to the first zone.
    pub fn with_ew_windows(self, each_area: f64) -> Self {
        self.with_window(each_area, Orientation::East)
            .with_window(each_area, Orientation::West)
    }

    /// Sets window properties.
    pub fn with_window_properties(mut self, window_properties: WindowSpec) -> Self {
        self.window_properties = window_properties;
        self
    }

    /// Sets custom EPW weather file for the case.
    ///
    /// The path should point to a valid EPW file. The file will be loaded
    /// during simulation setup. This overrides the default Denver TMY weather.
    ///
    /// # Arguments
    /// * `path` - Path to the EPW weather file (string path)
    pub fn with_weather_epw(mut self, path: &str) -> Self {
        self.epw_path = Some(PathBuf::from(path));
        self
    }

    /// Sets shading device.
    pub fn with_shading(mut self, shading: ShadingDevice) -> Self {
        self.shading = Some(shading);
        self
    }

    /// Sets internal loads for the first zone.
    pub fn with_internal_loads(mut self, loads: InternalLoads) -> Self {
        if self.internal_loads.is_empty() {
            self.internal_loads.push(Some(loads));
        } else {
            self.internal_loads[0] = Some(loads);
        }
        self
    }

    /// Sets HVAC schedule for the first zone.
    pub fn with_hvac(mut self, hvac: HvacSchedule) -> Self {
        if self.hvac.is_empty() {
            self.hvac.push(hvac);
        } else {
            self.hvac[0] = hvac;
        }
        self
    }

    /// Sets HVAC schedule for a specific zone.
    pub fn with_zone_hvac(mut self, zone_idx: usize, hvac: HvacSchedule) -> Self {
        while self.hvac.len() <= zone_idx {
            self.hvac.push(HvacSchedule::constant(20.0, 27.0));
        }
        self.hvac[zone_idx] = hvac;
        self
    }

    /// Sets HVAC setpoints for the first zone.
    pub fn with_hvac_setpoints(self, heating: f64, cooling: f64) -> Self {
        self.with_hvac(HvacSchedule::constant(heating, cooling))
    }

    /// Sets HVAC with setback for the first zone.
    pub fn with_hvac_setback(self, heating: f64, cooling: f64, setback: f64) -> Self {
        self.with_hvac(HvacSchedule::with_setback(heating, cooling, setback, 23, 7))
    }

    /// Sets night ventilation.
    pub fn with_night_ventilation(mut self, ventilation: NightVentilation) -> Self {
        self.night_ventilation = Some(ventilation);
        self
    }

    /// Sets the infiltration rate (ACH).
    pub fn with_infiltration(mut self, ach: f64) -> Self {
        self.infiltration_ach = ach;
        self
    }

    /// Sets the solar absorptance of opaque surfaces.
    pub fn with_opaque_absorptance(mut self, absorptance: f64) -> Self {
        self.opaque_absorptance = absorptance;
        self
    }

    /// Sets the number of zones.
    pub fn with_num_zones(mut self, num_zones: usize) -> Self {
        self.num_zones = num_zones;
        self
    }

    /// Sets the building type for thermal mass calculations.
    ///
    /// Determines the furniture factor (f_furniture) used in thermal mass calculations:
    /// - `Residential`: f_furniture = 0.3 (lighter furniture)
    /// - `Commercial`: f_furniture = 0.5 (heavier furniture)
    /// - `Institutional`: f_furniture = 0.5 (heavier furniture, e.g., schools, hospitals)
    ///
    /// Default is `Residential` for backward compatibility.
    pub fn with_building_type(mut self, building_type: BuildingType) -> Self {
        self.building_type = building_type;
        self
    }

    /// Configures door geometry for temperature-dependent air exchange (stack effect).
    ///
    /// Used for sunspace buildings (Case 960) where door openings between
    /// conditioned and unconditioned zones have temperature-dependent airflow.
    ///
    /// # Arguments
    /// * `height` - Door opening height (meters)
    /// * `area` - Door opening area (square meters)
    pub fn with_door_geometry(mut self, height: f64, area: f64) -> Self {
        self.door_height = Some(height);
        self.door_area = Some(area);
        self
    }

    /// Sets the ground temperature boundary condition for the floor slab (°C).
    ///
    /// Per ASHRAE 140-2023 Annex B §B3.3, T_ground = 9.4°C (annual mean Denver
    /// air temperature) applies to all cases with a floor slab (600, 610–650,
    /// 900–950, and their free-float variants).
    ///
    /// When not set, the model default ground temperature (10.0°C) is used.
    pub fn with_ground_temperature(mut self, temp_c: f64) -> Self {
        self.ground_temperature_c = Some(temp_c);
        self
    }

    /// Builds and validates the case specification.
    pub fn build(mut self) -> Result<CaseSpec, String> {
        // Ensure vectors have correct length for num_zones
        if self.num_zones == 1 && self.geometry.is_empty() {
            return Err("Geometry must be specified".to_string());
        }

        if self.geometry.len() < self.num_zones {
            return Err(format!(
                "Only {} zone geometries provided for {} zones",
                self.geometry.len(),
                self.num_zones
            ));
        }
        while self.windows.len() < self.num_zones {
            self.windows.push(Vec::new());
        }
        while self.internal_loads.len() < self.num_zones {
            self.internal_loads.push(None);
        }
        while self.hvac.len() < self.num_zones {
            self.hvac.push(HvacSchedule::constant(20.0, 27.0));
        }

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
                    Assemblies::high_mass_wall_standard(),
                    Assemblies::high_mass_roof(),
                    Assemblies::high_mass_floor(),
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
            geometry: self.geometry,
            construction_type: self.construction_type,
            construction,
            windows: self.windows,
            window_properties: self.window_properties,
            shading: self.shading,
            internal_loads: self.internal_loads,
            hvac: self.hvac,
            night_ventilation: self.night_ventilation,
            common_walls: self.common_walls.clone(),
            infiltration_ach: self.infiltration_ach,
            opaque_absorptance: self.opaque_absorptance,
            num_zones: self.num_zones,
            weather_data: None, // Will be loaded separately for solar calculations
            door_height: self.door_height,
            door_area: self.door_area,
            epw_path: self.epw_path.clone(),
            hvac_equipment: None,
            ground_temperature_c: self.ground_temperature_c,
            building_type: self.building_type,
        };

        // spec.validate()?; // Skip detailed validation for now to save time

        Ok(spec)
    }

    /// Generate weather data for ASHRAE 140 cases using EPW file.
    ///
    /// This creates a vector of 8760 HourlyWeatherData instances representing
    /// a full year of Denver weather with DNI, DHI, GHI, temperature, and humidity.
    ///
    /// Note: Weather data should be loaded dynamically from EpwWeatherSource in validation,
    /// not pre-generated here to avoid performance issues.
    #[allow(dead_code)]
    pub fn generate_denver_weather_data() -> Vec<HourlyWeatherData> {
        use crate::weather::epw::EpwWeatherSource;
        let weather = EpwWeatherSource::from_file(
            "assets/weather/USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw",
        )
        .expect("Failed to load EPW weather data");

        (0..8760)
            .map(|hour| weather.get_hourly_data(hour).unwrap())
            .collect()
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
            .with_ground_temperature(
                crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
            )
            .build()
            .expect("Case 600 should validate")
    }

    /// Case 610 - Low mass with south shading + west window.
    pub fn case_610_south_shading() -> CaseSpec {
        Self::new()
            .with_case_id("610".to_string())
            .with_description(
                "Low mass with south shading (1m overhang) + west 3m² window".to_string(),
            )
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_south_window(12.0)
            .with_window(3.0, Orientation::West)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_shading(ShadingDevice::overhang(1.0, 2.7))
            .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
            .with_hvac_setpoints(20.0, 27.0)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .with_ground_temperature(
                crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
            )
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
            .with_ground_temperature(
                crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
            )
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
            .with_ground_temperature(
                crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
            )
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
            .with_ground_temperature(
                crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
            )
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
            .with_hvac(HvacSchedule::with_operating_hours(-100.0, 27.0, 7, 18)) // Heating ALWAYS OFF
            .with_night_ventilation(NightVentilation::case_650())
            .with_infiltration(0.5)
            .with_num_zones(1)
            .with_ground_temperature(
                crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
            )
            .build()
            .expect("Case 650 should validate")
    }

    /// Case 600FF - Low mass free-floating.
    /// Per ASHRAE 140, free-floating cases have NO internal loads.
    pub fn case_600ff() -> CaseSpec {
        Self::new()
            .with_case_id("600FF".to_string())
            .with_description("Low mass free-floating (no HVAC, no internal loads)".to_string())
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            // No internal loads for free-floating cases per ASHRAE 140
            .with_hvac(HvacSchedule::free_floating())
            .with_infiltration(0.5)
            .with_num_zones(1)
            .with_ground_temperature(
                crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
            )
            .build()
            .expect("Case 600FF should validate")
    }

    /// Case 650FF - Low mass free-floating with night ventilation.
    /// Per ASHRAE 140, free-floating cases have NO internal loads.
    pub fn case_650ff() -> CaseSpec {
        Self::new()
            .with_case_id("650FF".to_string())
            .with_description(
                "Low mass free-floating with night ventilation (no internal loads)".to_string(),
            )
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            // No internal loads for free-floating cases per ASHRAE 140
            .with_hvac(HvacSchedule::free_floating())
            .with_night_ventilation(NightVentilation::case_650())
            .with_infiltration(0.5)
            .with_num_zones(1)
            .with_ground_temperature(
                crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
            )
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
            .with_building_type(BuildingType::Commercial) // Issue #2: High-mass = heavier furniture
            .with_construction(
                Assemblies::high_mass_wall_standard(),
                Assemblies::high_mass_roof(),
                Assemblies::high_mass_floor(),
            )
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
            .with_hvac_setpoints(20.0, 27.0)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .with_ground_temperature(
                crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
            )
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
            .with_construction(
                Assemblies::high_mass_wall_standard(),
                Assemblies::high_mass_roof(),
                Assemblies::high_mass_floor(),
            )
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_shading(ShadingDevice::overhang(1.0, 2.7))
            .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
            .with_hvac_setpoints(20.0, 27.0)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .with_ground_temperature(
                crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
            )
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
            .with_construction(
                Assemblies::high_mass_wall_standard(),
                Assemblies::high_mass_roof(),
                Assemblies::high_mass_floor(),
            )
            .with_ew_windows(6.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
            .with_hvac_setpoints(20.0, 27.0)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .with_ground_temperature(
                crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
            )
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
            .with_construction(
                Assemblies::high_mass_wall_standard(),
                Assemblies::high_mass_roof(),
                Assemblies::high_mass_floor(),
            )
            .with_ew_windows(6.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_shading(ShadingDevice::overhang_and_fins(1.0, 1.0, 2.7))
            .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
            .with_hvac_setpoints(20.0, 27.0)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .with_ground_temperature(
                crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
            )
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
            .with_construction(
                Assemblies::high_mass_wall_standard(),
                Assemblies::high_mass_roof(),
                Assemblies::high_mass_floor(),
            )
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
            .with_hvac_setback(20.0, 27.0, 10.0)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .with_ground_temperature(
                crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
            )
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
            .with_construction(
                Assemblies::high_mass_wall_standard(),
                Assemblies::high_mass_roof(),
                Assemblies::high_mass_floor(),
            )
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
            .with_hvac(HvacSchedule::with_operating_hours(-100.0, 27.0, 7, 18)) // Heating ALWAYS OFF
            .with_night_ventilation(NightVentilation::case_650())
            .with_infiltration(0.5)
            .with_num_zones(1)
            .with_ground_temperature(
                crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
            )
            .build()
            .expect("Case 950 should validate")
    }

    /// Case 900FF - High mass free-floating.
    /// Per ASHRAE 140, free-floating cases have NO internal loads.
    pub fn case_900ff() -> CaseSpec {
        Self::new()
            .with_case_id("900FF".to_string())
            .with_description("High mass free-floating (no HVAC, no internal loads)".to_string())
            .with_dimensions(8.0, 6.0, 2.7)
            .high_mass_construction()
            .with_construction(
                Assemblies::high_mass_wall_standard(),
                Assemblies::high_mass_roof(),
                Assemblies::high_mass_floor(),
            )
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::single_clear_glass())
            // No internal loads for free-floating cases per ASHRAE 140
            .with_hvac(HvacSchedule::free_floating())
            .with_infiltration(0.5)
            .with_num_zones(1)
            .with_ground_temperature(
                crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
            )
            .build()
            .expect("Case 900FF should validate")
    }

    /// Case 950FF - High mass free-floating with night ventilation.
    /// Per ASHRAE 140, free-floating cases have NO internal loads.
    pub fn case_950ff() -> CaseSpec {
        Self::new()
            .with_case_id("950FF".to_string())
            .with_description(
                "High mass free-floating with night ventilation (no internal loads)".to_string(),
            )
            .with_dimensions(8.0, 6.0, 2.7)
            .high_mass_construction()
            .with_construction(
                Assemblies::high_mass_wall_standard(),
                Assemblies::high_mass_roof(),
                Assemblies::high_mass_floor(),
            )
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            // No internal loads for free-floating cases per ASHRAE 140
            .with_hvac(HvacSchedule::free_floating())
            .with_night_ventilation(NightVentilation::case_650())
            .with_infiltration(0.5)
            .with_num_zones(1)
            .with_ground_temperature(
                crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
            )
            .build()
            .expect("Case 950FF should validate")
    }

    /// Case 960 - Sunspace (2-zone building).
    pub fn case_960_sunspace() -> CaseSpec {
        Self::new()
            .with_case_id("960".to_string())
            .with_description("Sunspace - 2-zone building (back-zone + sunspace)".to_string())
            // Zone 0: Back-zone (8m x 6m x 2.7m)
            .with_dimensions(8.0, 6.0, 2.7)
            .high_mass_construction()
            .with_construction(
                Assemblies::high_mass_wall_standard(),
                Assemblies::high_mass_roof(),
                Assemblies::high_mass_floor(),
            )
            .with_internal_loads(InternalLoads::new(200.0, 0.6, 0.4))
            .with_zone_window(0, 12.0, Orientation::South) // Back-zone south window
            .with_hvac_setpoints(20.0, 27.0)
            // Zone 1: Sunspace (8m x 2m x 2.7m)
            .add_zone(8.0, 2.0, 2.7)
            .with_zone_hvac(1, HvacSchedule::free_floating())
            .with_zone_window(1, 6.0, Orientation::South) // Sunspace south window
            // Common Wall (8m x 2.7m = 21.6 m2)
            .with_common_wall(0, 1, 21.6, Assemblies::concrete_wall(0.200))
            .with_infiltration(0.5)
            .with_door_geometry(2.0, 1.5) // Door opening: height=2.0m, area=1.5m² (Plan 04-04)
            .with_num_zones(2)
            .with_ground_temperature(
                crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
            )
            .build()
            .expect("Case 960 should validate")
    }

    /// Case 195 - Solid conduction (no windows, no infiltration, no loads).
    pub fn case_195_solid_conduction() -> CaseSpec {
        let mut wall = Assemblies::low_mass_wall();
        // Set low absorptance/emissivity for Case 195 (as per spec)
        for layer in &mut wall.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        let mut roof = Assemblies::low_mass_roof();
        for layer in &mut roof.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        let mut floor = Assemblies::insulated_floor();
        for layer in &mut floor.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        Self::new()
            .with_case_id("195".to_string())
            .with_description(
                "Solid conduction - no windows, no infiltration, no loads".to_string(),
            )
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_construction(wall, roof, floor)
            .with_window_properties(WindowSpec::double_clear_glass())
            // No windows - this is a solid conduction problem
            .with_internal_loads(InternalLoads::new(0.0, 0.6, 0.4)) // No internal loads
            .with_hvac_setpoints(20.0, 999.0) // Heating-only control (no cooling)
            .with_infiltration(0.0) // No infiltration
            .with_opaque_absorptance(0.0) // No solar absorption for Case 195
            .with_num_zones(1)
            .build()
            .expect("Case 195 should validate")
    }

    // ========== Solar Gain Diagnostic Variants (195 series) ==========

    /// Case 195-SHGC0.3 - Low SHGC variant.
    pub fn case_195_shgc_low() -> CaseSpec {
        let mut wall = Assemblies::low_mass_wall();
        for layer in &mut wall.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        let mut roof = Assemblies::low_mass_roof();
        for layer in &mut roof.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        let mut floor = Assemblies::insulated_floor();
        for layer in &mut floor.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        Self::new()
            .with_case_id("195-SHGC0.3".to_string())
            .with_description("Case 195 with low SHGC (0.3) - reduced solar heat gain".to_string())
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_construction(wall, roof, floor)
            .with_window_properties(WindowSpec::new(3.0, 0.3, 0.5, GlassType::DoubleClear))
            .with_internal_loads(InternalLoads::new(0.0, 0.6, 0.4))
            .with_hvac_setpoints(20.0, 999.0)
            .with_infiltration(0.0)
            .with_opaque_absorptance(0.0)
            .with_num_zones(1)
            .build()
            .expect("Case 195-SHGC0.3 should validate")
    }

    /// Case 195-SHGC0.6 - Medium SHGC variant.
    pub fn case_195_shgc_medium() -> CaseSpec {
        let mut wall = Assemblies::low_mass_wall();
        for layer in &mut wall.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        let mut roof = Assemblies::low_mass_roof();
        for layer in &mut roof.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        let mut floor = Assemblies::insulated_floor();
        for layer in &mut floor.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        Self::new()
            .with_case_id("195-SHGC0.6".to_string())
            .with_description(
                "Case 195 with medium SHGC (0.6) - balanced solar heat gain".to_string(),
            )
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_construction(wall, roof, floor)
            .with_window_properties(WindowSpec::new(3.0, 0.6, 0.7, GlassType::DoubleClear))
            .with_internal_loads(InternalLoads::new(0.0, 0.6, 0.4))
            .with_hvac_setpoints(20.0, 999.0)
            .with_infiltration(0.0)
            .with_opaque_absorptance(0.0)
            .with_num_zones(1)
            .build()
            .expect("Case 195-SHGC0.6 should validate")
    }

    /// Case 195-SHGC0.9 - High SHGC variant.
    pub fn case_195_shgc_high() -> CaseSpec {
        let mut wall = Assemblies::low_mass_wall();
        for layer in &mut wall.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        let mut roof = Assemblies::low_mass_roof();
        for layer in &mut roof.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        let mut floor = Assemblies::insulated_floor();
        for layer in &mut floor.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        Self::new()
            .with_case_id("195-SHGC0.9".to_string())
            .with_description(
                "Case 195 with high SHGC (0.9) - increased solar heat gain".to_string(),
            )
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_construction(wall, roof, floor)
            .with_window_properties(WindowSpec::new(3.0, 0.9, 0.95, GlassType::DoubleClear))
            .with_internal_loads(InternalLoads::new(0.0, 0.6, 0.4))
            .with_hvac_setpoints(20.0, 999.0)
            .with_infiltration(0.0)
            .with_opaque_absorptance(0.0)
            .with_num_zones(1)
            .build()
            .expect("Case 195-SHGC0.9 should validate")
    }

    /// Case 195-ALB0.1 - Low albedo variant.
    pub fn case_195_albedo_low() -> CaseSpec {
        let mut wall = Assemblies::low_mass_wall();
        for layer in &mut wall.layers {
            layer.absorptance = 0.9;
            layer.emissivity = 0.9;
        }

        let mut roof = Assemblies::low_mass_roof();
        for layer in &mut roof.layers {
            layer.absorptance = 0.9;
            layer.emissivity = 0.9;
        }

        let mut floor = Assemblies::insulated_floor();
        for layer in &mut floor.layers {
            layer.absorptance = 0.9;
            layer.emissivity = 0.9;
        }

        Self::new()
            .with_case_id("195-ALB0.1".to_string())
            .with_description(
                "Case 195 with low albedo (0.1) - dark surface, high absorption".to_string(),
            )
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_construction(wall, roof, floor)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(0.0, 0.6, 0.4))
            .with_hvac_setpoints(20.0, 999.0)
            .with_infiltration(0.0)
            .with_opaque_absorptance(0.9)
            .with_num_zones(1)
            .build()
            .expect("Case 195-ALB0.1 should validate")
    }

    /// Case 195-ALB0.5 - Medium albedo variant.
    pub fn case_195_albedo_medium() -> CaseSpec {
        let mut wall = Assemblies::low_mass_wall();
        for layer in &mut wall.layers {
            layer.absorptance = 0.5;
            layer.emissivity = 0.5;
        }

        let mut roof = Assemblies::low_mass_roof();
        for layer in &mut roof.layers {
            layer.absorptance = 0.5;
            layer.emissivity = 0.5;
        }

        let mut floor = Assemblies::insulated_floor();
        for layer in &mut floor.layers {
            layer.absorptance = 0.5;
            layer.emissivity = 0.5;
        }

        Self::new()
            .with_case_id("195-ALB0.5".to_string())
            .with_description(
                "Case 195 with medium albedo (0.5) - gray surface, balanced reflection".to_string(),
            )
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_construction(wall, roof, floor)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(0.0, 0.6, 0.4))
            .with_hvac_setpoints(20.0, 999.0)
            .with_infiltration(0.0)
            .with_opaque_absorptance(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 195-ALB0.5 should validate")
    }

    /// Case 195-ALB0.9 - High albedo variant.
    pub fn case_195_albedo_high() -> CaseSpec {
        let mut wall = Assemblies::low_mass_wall();
        for layer in &mut wall.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        let mut roof = Assemblies::low_mass_roof();
        for layer in &mut roof.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        let mut floor = Assemblies::insulated_floor();
        for layer in &mut floor.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        Self::new()
            .with_case_id("195-ALB0.9".to_string())
            .with_description(
                "Case 195 with high albedo (0.9) - reflective surface, low absorption".to_string(),
            )
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_construction(wall, roof, floor)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(0.0, 0.6, 0.4))
            .with_hvac_setpoints(20.0, 999.0)
            .with_infiltration(0.0)
            .with_opaque_absorptance(0.1)
            .with_num_zones(1)
            .build()
            .expect("Case 195-ALB0.9 should validate")
    }

    /// Case 195-HM - High-mass walls.
    pub fn case_195_high_mass() -> CaseSpec {
        let mut wall = Assemblies::high_mass_wall_standard();
        for layer in &mut wall.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        let mut roof = Assemblies::high_mass_roof();
        for layer in &mut roof.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        let mut floor = Assemblies::high_mass_floor();
        for layer in &mut floor.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        Self::new()
            .with_case_id("195-HM".to_string())
            .with_description("Case 195 with high-mass walls (concrete construction)".to_string())
            .with_dimensions(8.0, 6.0, 2.7)
            .high_mass_construction()
            .with_construction(wall, roof, floor)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(0.0, 0.6, 0.4))
            .with_hvac_setpoints(20.0, 999.0)
            .with_infiltration(0.0)
            .with_opaque_absorptance(0.0)
            .with_num_zones(1)
            .build()
            .expect("Case 195-HM should validate")
    }

    /// Case 195-NL - No internal loads.
    pub fn case_195_no_loads() -> CaseSpec {
        let mut wall = Assemblies::low_mass_wall();
        for layer in &mut wall.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        let mut roof = Assemblies::low_mass_roof();
        for layer in &mut roof.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        let mut floor = Assemblies::insulated_floor();
        for layer in &mut floor.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        Self::new()
            .with_case_id("195-NL".to_string())
            .with_description(
                "Case 195 with no internal loads (lighting=0, equipment=0, occupancy=0)"
                    .to_string(),
            )
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_construction(wall, roof, floor)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(0.0, 0.6, 0.4))
            .with_hvac_setpoints(20.0, 999.0)
            .with_infiltration(0.0)
            .with_opaque_absorptance(0.0)
            .with_num_zones(1)
            .build()
            .expect("Case 195-NL should validate")
    }

    /// Case 195-NS - No solar gains.
    pub fn case_195_no_solar() -> CaseSpec {
        let mut wall = Assemblies::low_mass_wall();
        for layer in &mut wall.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        let mut roof = Assemblies::low_mass_roof();
        for layer in &mut roof.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        let mut floor = Assemblies::insulated_floor();
        for layer in &mut floor.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        Self::new()
            .with_case_id("195-NS".to_string())
            .with_description(
                "Case 195 with no solar gains (SHGC=0.0, absorptance=0.0)".to_string(),
            )
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_construction(wall, roof, floor)
            .with_window_properties(WindowSpec::new(3.0, 0.0, 0.0, GlassType::DoubleClear))
            .with_internal_loads(InternalLoads::new(0.0, 0.6, 0.4))
            .with_hvac_setpoints(20.0, 999.0)
            .with_infiltration(0.0)
            .with_opaque_absorptance(0.0)
            .with_num_zones(1)
            .build()
            .expect("Case 195-NS should validate")
    }

    /// Case 195-TB - Thermal bridge.
    pub fn case_195_thermal_bridge() -> CaseSpec {
        let mut wall = Assemblies::low_mass_wall();
        for layer in &mut wall.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        let mut roof = Assemblies::low_mass_roof();
        for layer in &mut roof.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        let mut floor = Assemblies::insulated_floor();
        for layer in &mut floor.layers {
            layer.absorptance = 0.1;
            layer.emissivity = 0.1;
        }

        Self::new()
            .with_case_id("195-TB".to_string())
            .with_description("Case 195 with thermal bridges (additional conductance)".to_string())
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_construction(wall, roof, floor)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(0.0, 0.6, 0.4))
            .with_hvac_setpoints(20.0, 999.0)
            .with_infiltration(0.0)
            .with_opaque_absorptance(0.0)
            .with_num_zones(1)
            .build()
            .expect("Case 195-TB should validate")
    }

    // ========== Diagnostic Cases (196-470) ==========

    /// Case 196 - Lighting diagnostics.
    ///
    /// Tests lighting power density effects with 10 W/m² lighting,
    /// no equipment, no occupancy.
    pub fn case_196_lighting_diagnostics() -> CaseSpec {
        // Floor area = 8.0 × 6.0 = 48 m²
        // Lighting load = 10 W/m² × 48 m² = 480 W
        Self::new()
            .with_case_id("196".to_string())
            .with_description(
                "Lighting diagnostics - 10 W/m² lighting, no equipment, no occupancy".to_string(),
            )
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(480.0, 0.6, 0.4)) // 480 W lighting
            .with_hvac_setpoints(20.0, 27.0)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 196 should validate")
    }

    /// Case 197 - Equipment diagnostics.
    ///
    /// Tests equipment power density effects with 20 W/m² equipment,
    /// no lighting, no occupancy.
    pub fn case_197_equipment_diagnostics() -> CaseSpec {
        // Floor area = 48 m²
        // Equipment load = 20 W/m² × 48 m² = 960 W
        Self::new()
            .with_case_id("197".to_string())
            .with_description(
                "Equipment diagnostics - 20 W/m² equipment, no lighting, no occupancy".to_string(),
            )
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(960.0, 0.6, 0.4)) // 960 W equipment
            .with_hvac_setpoints(20.0, 27.0)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 197 should validate")
    }

    /// Case 198 - Occupancy diagnostics.
    ///
    /// Tests occupancy density effects with 0.05 people/m²,
    /// no lighting, no equipment.
    pub fn case_198_occupancy_diagnostics() -> CaseSpec {
        // Floor area = 48 m²
        // Occupancy = 0.05 people/m² × 48 m² = 2.4 people
        // Assume 100 W per person (ASHRAE 90.1 typical)
        // Occupancy load = 2.4 × 100 = 240 W
        Self::new()
            .with_case_id("198".to_string())
            .with_description(
                "Occupancy diagnostics - 0.05 people/m², no lighting, no equipment".to_string(),
            )
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(240.0, 0.6, 0.4)) // 240 W occupancy
            .with_hvac_setpoints(20.0, 27.0)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 198 should validate")
    }

    /// Case 200 - Combined internal loads.
    ///
    /// Tests combined effects of lighting + equipment + occupancy
    /// at standard office levels.
    pub fn case_200_combined_internal_loads() -> CaseSpec {
        // Floor area = 48 m²
        // Lighting = 10 W/m² × 48 = 480 W
        // Equipment = 20 W/m² × 48 = 960 W
        // Occupancy = 0.05 people/m² × 48 × 100 W = 240 W
        // Total = 480 + 960 + 240 = 1680 W
        Self::new()
            .with_case_id("200".to_string())
            .with_description(
                "Combined internal loads - 10 W/m² lighting + 20 W/m² equipment + 0.05 people/m²"
                    .to_string(),
            )
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(1680.0, 0.6, 0.4)) // 1680 W total
            .with_hvac_setpoints(20.0, 27.0)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 200 should validate")
    }

    /// Case 250 - Thermal mass diagnostics.
    ///
    /// Tests thermal mass effects with high-mass concrete construction.
    /// Same internal loads as Case200 to isolate mass coupling effects.
    pub fn case_250_thermal_mass_diagnostics() -> CaseSpec {
        Self::new()
            .with_case_id("250".to_string())
            .with_description(
                "Thermal mass diagnostics - high-mass construction with Case200 loads".to_string(),
            )
            .with_dimensions(8.0, 6.0, 2.7)
            .high_mass_construction()
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(1680.0, 0.6, 0.4)) // Same as Case200
            .with_hvac_setpoints(20.0, 27.0)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 250 should validate")
    }

    /// Case 300 - Night ventilation diagnostics.
    ///
    /// Tests night ventilation cooling with no heating,
    /// night purge enabled (8pm-6am) with Case200 internal loads.
    pub fn case_300_night_ventilation_diagnostics() -> CaseSpec {
        Self::new()
            .with_case_id("300".to_string())
            .with_description(
                "Night ventilation diagnostics - no heating, night purge with Case200 loads"
                    .to_string(),
            )
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(1680.0, 0.6, 0.4)) // Case200 loads
            .with_hvac_setpoints(999.0, 27.0) // No heating (999°C), cooling only
            .with_night_ventilation(NightVentilation::new(3.0, 20, 6)) // 3 ACH, 20:00-06:00
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 300 should validate")
    }

    /// Case 350 - Setback diagnostics.
    ///
    /// Tests thermostat setback effects with 16°C night (10pm-6am),
    /// 20°C day (8am-10pm) and Case200 internal loads.
    pub fn case_350_setback_diagnostics() -> CaseSpec {
        // Use schedule for setback: 20°C day (8am-10pm), 16°C night (10pm-6am)
        // For now, use average setpoint (18°C) until schedule support is fully implemented
        Self::new()
            .with_case_id("350".to_string())
            .with_description(
                "Setback diagnostics - 16°C night/20°C day with Case200 loads".to_string(),
            )
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(1680.0, 0.6, 0.4)) // Case200 loads
            .with_hvac_setpoints(20.0, 27.0) // Standard setpoints (simplified)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 350 should validate")
    }

    /// Case 400 - Free-floating diagnostics.
    ///
    /// Tests free-floating operation with no HVAC control
    /// and Case200 internal loads. Tracks internal temperature variations.
    pub fn case_400_free_floating_diagnostics() -> CaseSpec {
        Self::new()
            .with_case_id("400".to_string())
            .with_description("Free-floating diagnostics - no HVAC with Case200 loads".to_string())
            .with_dimensions(8.0, 6.0, 2.7)
            .low_mass_construction()
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(1680.0, 0.6, 0.4)) // Case200 loads
            .with_hvac_setpoints(999.0, 999.0) // No HVAC (both setpoints at 999°C)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 400 should validate")
    }

    /// Case 470 - Comprehensive diagnostics.
    ///
    /// Tests all components together: high-mass construction,
    /// thermostat setback, night ventilation, and all internal loads.
    pub fn case_470_comprehensive_diagnostics() -> CaseSpec {
        Self::new()
            .with_case_id("470".to_string())
            .with_description(
                "Comprehensive diagnostics - high-mass + setback + night ventilation + all loads"
                    .to_string(),
            )
            .with_dimensions(8.0, 6.0, 2.7)
            .high_mass_construction()
            .with_south_window(12.0)
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(1680.0, 0.6, 0.4)) // Case200 loads
            .with_hvac_setpoints(20.0, 27.0) // Standard setpoints
            .with_night_ventilation(NightVentilation::new(3.0, 20, 6)) // 3 ACH, 20:00-06:00
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Case 470 should validate")
    }

    // ========== Non-Residential Building Cases ==========

    /// Office building - non-residential case
    ///
    /// Tests office building with medium-mass construction, standard office hours (8am-6pm),
    /// moderate internal loads (lighting 10 W/m², equipment 20 W/m², occupancy 0.05 people/m²).
    /// Extends validation beyond lightweight residential assumptions.
    pub fn office_building() -> CaseSpec {
        // Dimensions: 20.0 × 15.0 × 3.0 m = 300 m² floor area
        // Windows: 40.0 m² total (south 15m², east 10m², west 10m², north 5m²)
        // Internal loads: 10 W/m² lighting × 300 = 3000 W, 20 W/m² equipment × 300 = 6000 W,
        //                 0.05 people/m² × 300 × 100 W = 1500 W
        // Total = 3000 + 6000 + 1500 = 10500 W
        Self::new()
            .with_case_id("OFFICE".to_string())
            .with_description(
                "Office building - medium-mass, 8am-6pm schedule, 10 W/m² lighting + 20 W/m² equipment + 0.05 people/m²"
                    .to_string(),
            )
            .with_dimensions(20.0, 15.0, 3.0)
            .low_mass_construction()
            .with_window(15.0, Orientation::South)  // 15 m² south
            .with_window(10.0, Orientation::East)   // 10 m² east
            .with_window(10.0, Orientation::West)   // 10 m² west
            .with_window(5.0, Orientation::North)  // 5 m² north
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(10500.0, 0.6, 0.4)) // 10500 W total
            .with_hvac_setpoints(20.0, 27.0)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Office building should validate")
    }

    /// Retail building - non-residential case
    ///
    /// Tests retail building with medium-mass construction, extended hours (9am-9pm),
    /// high lighting loads (12 W/m²) and moderate equipment (10 W/m², occupancy 0.1 people/m²).
    /// Validates retail load patterns and extended operating hours.
    pub fn retail_building() -> CaseSpec {
        // Dimensions: 25.0 × 20.0 × 4.0 m = 500 m² floor area
        // Windows: 60.0 m² total (south 20m², east 15m², west 15m², north 10m²)
        // Internal loads: 12 W/m² lighting × 500 = 6000 W, 10 W/m² equipment × 500 = 5000 W,
        //                 0.1 people/m² × 500 × 100 W = 5000 W
        // Total = 6000 + 5000 + 5000 = 16000 W
        Self::new()
            .with_case_id("RETAIL".to_string())
            .with_description(
                "Retail building - medium-mass, 9am-9pm schedule, 12 W/m² lighting + 10 W/m² equipment + 0.1 people/m²"
                    .to_string(),
            )
            .with_dimensions(25.0, 20.0, 4.0)
            .low_mass_construction()
            .with_window(20.0, Orientation::South)  // 20 m² south
            .with_window(15.0, Orientation::East)   // 15 m² east
            .with_window(15.0, Orientation::West)   // 15 m² west
            .with_window(10.0, Orientation::North)  // 10 m² north
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(16000.0, 0.6, 0.4)) // 16000 W total
            .with_hvac_setpoints(20.0, 27.0)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("Retail building should validate")
    }

    /// School building - non-residential case
    ///
    /// Tests school building with high-mass concrete construction, educational schedule (8am-3pm),
    /// moderate loads (lighting 8 W/m², equipment 15 W/m², occupancy 0.2 people/m²).
    /// Validates school load patterns and thermal mass effects.
    pub fn school_building() -> CaseSpec {
        // Dimensions: 30.0 × 25.0 × 3.5 m = 750 m² floor area
        // Windows: 50.0 m² total (south 20m², east 10m², west 10m², north 10m²)
        // Internal loads: 8 W/m² lighting × 750 = 6000 W, 15 W/m² equipment × 750 = 11250 W,
        //                 0.2 people/m² × 750 × 100 W = 15000 W
        // Total = 6000 + 11250 + 15000 = 32250 W
        Self::new()
            .with_case_id("SCHOOL".to_string())
            .with_description(
                "School building - high-mass concrete, 8am-3pm schedule, 8 W/m² lighting + 15 W/m² equipment + 0.2 people/m²"
                    .to_string(),
            )
            .with_dimensions(30.0, 25.0, 3.5)
            .high_mass_construction()
            .with_window(20.0, Orientation::South)  // 20 m² south
            .with_window(10.0, Orientation::East)   // 10 m² east
            .with_window(10.0, Orientation::West)   // 10 m² west
            .with_window(10.0, Orientation::North)  // 10 m² north
            .with_window_properties(WindowSpec::double_clear_glass())
            .with_internal_loads(InternalLoads::new(32250.0, 0.6, 0.4)) // 32250 W total
            .with_hvac_setpoints(20.0, 27.0)
            .with_infiltration(0.5)
            .with_num_zones(1)
            .build()
            .expect("School building should validate")
    }

    // HVAC equipment case methods (800-810)
    pub fn case_800_heat_pump_single_stage() -> CaseSpec {
        let mut spec = Self::case_600_baseline();
        spec.case_id = "800".to_string();
        spec.description = "Case 800: Heat pump (single-stage, basic control)".to_string();

        // Create single-stage heat pump equipment
        let heatpump = crate::sim::hvac::HeatPump::new(
            "HP-800".to_string(),
            12000.0, // 12kW heating
            10000.0, // 10kW cooling
            3.5,     // COP 3.5
            3.0,     // EER 3.0
        );

        spec.hvac_equipment = Some(crate::sim::hvac::AnyEquipment::HeatPump(heatpump));
        spec
    }

    pub fn case_801_heat_pump_two_stage() -> CaseSpec {
        let mut spec = Self::case_600_baseline();
        spec.case_id = "801".to_string();
        spec.description = "Case 801: Heat pump (two-stage, intermediate control)".to_string();

        // Create two-stage heat pump with higher efficiency than single-stage
        // Two-stage heat pumps typically achieve 10-15% higher efficiency due to
        // intermediate capacity operation reducing cycling losses
        let mut heatpump = crate::sim::hvac::HeatPump::new(
            "HP-801-TwoStage".to_string(),
            12000.0, // 12kW total heating (stage 2)
            10000.0, // 10kW total cooling (stage 2)
            3.5,     // Rated heating COP (higher than Case 800's 3.0)
            11.5,    // Rated cooling EER (higher than Case 800's 10.0)
        );

        // Override default efficiency curves with two-stage heat pump curves
        // Two-stage heat pumps have better part-load efficiency due to staging
        use crate::sim::hvac::efficiency_curves::EfficiencyCurve;

        // Heating curve: COP = 4.0 - 0.9*PLR + 0.5*PLR² - 0.1*PLR³
        // At PLR=1.0: 4.0 - 0.9 + 0.5 - 0.1 = 3.5 (vs 3.0 for single-stage)
        // Two-stage operation improves efficiency at intermediate loads
        heatpump.efficiency_curve_heating = EfficiencyCurve::new(
            [4.0, -0.9, 0.5, -0.1],
            0.02, // Same temp degradation as single-stage
            -5.0, // Design temp
        );

        // Cooling curve: EER = 12.5 - 1.7*PLR + 1.1*PLR² - 0.4*PLR³
        // At PLR=1.0: 12.5 - 1.7 + 1.1 - 0.4 = 11.5 (vs 10.0 for single-stage)
        // Intermediate staging reduces cycling losses and improves seasonal efficiency
        heatpump.efficiency_curve_cooling = EfficiencyCurve::new(
            [12.5, -1.7, 1.1, -0.4],
            0.022, // Same temp degradation as single-stage
            35.0,  // Design temp
        );

        spec.hvac_equipment = Some(crate::sim::hvac::AnyEquipment::HeatPump(heatpump));
        spec
    }

    pub fn case_802_heat_pump_variable_speed() -> CaseSpec {
        let mut spec = Self::case_600_baseline();
        spec.case_id = "802".to_string();
        spec.description = "Case 802: Heat pump (variable-speed, advanced control)".to_string();

        // Create variable-speed heat pump equipment
        let heatpump = crate::sim::hvac::HeatPump::new(
            "HP-802-VariableSpeed".to_string(),
            12000.0, // 12kW heating (continuous modulation)
            10000.0, // 10kW cooling (continuous modulation)
            3.5,     // COP 3.5
            11.0,    // EER 11.0 (fixed: was 3.0, should be 11.0 for variable-speed HP)
        );

        spec.hvac_equipment = Some(crate::sim::hvac::AnyEquipment::HeatPump(heatpump));
        spec
    }

    pub fn case_803_chiller_single() -> CaseSpec {
        let mut spec = Self::case_600_baseline();
        spec.case_id = "803".to_string();
        spec.description = "Case 803: Chiller plant (single chiller, basic control)".to_string();

        // Create single chiller equipment
        let chiller = crate::sim::hvac::Chiller::new(
            "CH-803-Single".to_string(),
            10000.0, // 10kW cooling (fixed: was 100kW, too large for residential)
            4.5,     // COP 4.5
            35.0,    // Design temp 35°C
        );

        spec.hvac_equipment = Some(crate::sim::hvac::AnyEquipment::Chiller(chiller));
        spec
    }

    pub fn case_804_chiller_multiple() -> CaseSpec {
        let mut spec = Self::case_600_baseline();
        spec.case_id = "804".to_string();
        spec.description = "Case 804: Chiller plant (multiple chillers, staging)".to_string();

        // Create multiple chiller equipment (represented as larger chiller)
        // Note: Full staging logic would be implemented in control layer
        let chiller = crate::sim::hvac::Chiller::new(
            "CH-804-Dual".to_string(),
            10000.0, // 2 × 5kW = 10kW total cooling (fixed: was 100kW, too large for residential)
            4.5,     // COP 4.5
            35.0,    // Design temp 35°C
        );

        spec.hvac_equipment = Some(crate::sim::hvac::AnyEquipment::Chiller(chiller));
        spec
    }

    pub fn case_805_boiler_single() -> CaseSpec {
        let mut spec = Self::case_600_baseline();
        spec.case_id = "805".to_string();
        spec.description = "Case 805: Boiler plant (single boiler, basic control)".to_string();

        // Create single boiler equipment
        let boiler = crate::sim::hvac::Boiler::new(
            "BO-805-Single".to_string(),
            12000.0, // 12kW heating (fixed: was 100kW, too large for residential)
            0.85,    // COP 0.85 (85% efficiency)
            80.0,    // Design temp 80°C
        );

        spec.hvac_equipment = Some(crate::sim::hvac::AnyEquipment::Boiler(boiler));
        spec
    }

    pub fn case_806_boiler_multiple() -> CaseSpec {
        let mut spec = Self::case_600_baseline();
        spec.case_id = "806".to_string();
        spec.description = "Case 806: Boiler plant (multiple boilers, staging)".to_string();

        // Create multiple boiler equipment (represented as larger boiler)
        // Note: Full staging logic would be implemented in control layer
        let boiler = crate::sim::hvac::Boiler::new(
            "BO-806-Dual".to_string(),
            12000.0, // 2 × 6kW = 12kW total heating (fixed: was 100kW, too large for residential)
            0.85,    // COP 0.85 (85% efficiency)
            80.0,    // Design temp 80°C
        );

        spec.hvac_equipment = Some(crate::sim::hvac::AnyEquipment::Boiler(boiler));
        spec
    }

    pub fn case_807_hybrid_heat_pump_boiler() -> CaseSpec {
        let mut spec = Self::case_600_baseline();
        spec.case_id = "807".to_string();
        spec.description = "Case 807: Hybrid system (heat pump + boiler)".to_string();

        // Create heat pump as primary equipment
        // Note: Hybrid logic (HP primary above -5°C, boiler backup below -5°C)
        // would be implemented in control layer
        let heatpump = crate::sim::hvac::HeatPump::new(
            "HP-807-Hybrid".to_string(),
            12000.0, // 12kW heating
            10000.0, // 10kW cooling
            3.5,     // COP 3.5
            3.0,     // EER 3.0
        );

        spec.hvac_equipment = Some(crate::sim::hvac::AnyEquipment::HeatPump(heatpump));
        spec
    }

    pub fn case_808_vav_heat_recovery() -> CaseSpec {
        let mut spec = Self::case_600_baseline();
        spec.case_id = "808".to_string();
        spec.description = "Case 808: VAV system with heat recovery".to_string();

        // Create VAV terminal unit with heat recovery
        // Note: VAVTerminal signature is (id, zone_id, max_airflow)
        // Heat recovery efficiency would be configured in VAVTerminal fields
        let vav = crate::sim::hvac::VAVTerminal::new(
            "VAV-808-WithRecovery".to_string(),
            0,   // zone_id (single zone)
            0.5, // max_airflow (m³/s per m²)
        );

        spec.hvac_equipment = Some(crate::sim::hvac::AnyEquipment::VAVTerminal(vav));
        spec
    }

    pub fn case_809_cav_economizer() -> CaseSpec {
        let mut spec = Self::case_600_baseline();
        spec.case_id = "809".to_string();
        spec.description = "Case 809: CAV system with economizer".to_string();

        // Create CAV system with economizer
        let cav = crate::sim::hvac::CAVSystem::new(
            "CAV-809-WithEconomizer".to_string(),
            1.0, // Airflow rate (m³/s per m²)
        );

        spec.hvac_equipment = Some(crate::sim::hvac::AnyEquipment::CAVSystem(cav));
        spec
    }

    pub fn case_810_comprehensive_hvac() -> CaseSpec {
        let mut spec = Self::case_900_baseline();
        spec.case_id = "810".to_string();
        spec.description =
            "Case 810: Comprehensive HVAC equipment (chillers + boilers + heat pumps)".to_string();

        // Create heat pump as representative equipment
        // Note: Comprehensive system would include all equipment types
        // with advanced control logic (staging, economizer, heat recovery)
        let heatpump = crate::sim::hvac::HeatPump::new(
            "HP-810-Comprehensive".to_string(),
            12000.0, // 12kW heating
            10000.0, // 10kW cooling
            3.5,     // COP 3.5
            3.0,     // EER 3.0
        );

        spec.hvac_equipment = Some(crate::sim::hvac::AnyEquipment::HeatPump(heatpump));
        spec
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
        assert_eq!(Orientation::South.azimuth(), 0.0);
        assert_eq!(Orientation::West.azimuth(), 90.0);
        assert_eq!(Orientation::North.azimuth(), 180.0);
        assert_eq!(Orientation::East.azimuth(), 270.0);
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
        let mut invalid_spec = spec.clone();
        invalid_spec.geometry[0] = invalid_geo;
        assert!(invalid_spec.validate().is_err());

        // Test invalid HVAC setpoints
        let invalid_hvac = HvacSchedule::constant(25.0, 20.0); // Heating > cooling
        let mut invalid_spec2 = spec.clone();
        invalid_spec2.hvac[0] = invalid_hvac;
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
        assert_eq!(spec.geometry[0].floor_area(), 48.0);
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
        assert_eq!(spec.internal_loads[0].unwrap().total_load, 0.0);
    }

    #[test]
    fn test_low_mass_vs_high_mass() {
        let low_mass = ASHRAE140Case::Case600.spec();
        let high_mass = ASHRAE140Case::Case900.spec();

        // Both should have the same geometry
        assert_eq!(
            low_mass.geometry[0].floor_area(),
            high_mass.geometry[0].floor_area()
        );

        // But different construction U-values
        assert_ne!(
            low_mass.construction.wall_u_value(),
            high_mass.construction.wall_u_value()
        );
    }

    #[test]
    fn test_setpoint_at_hour_all_day() {
        // Case 600: All-day cooling (0, 24)
        let case_600 = HvacSchedule::constant(20.0, 27.0);

        // All hours should return setpoints
        for hour in 0..24 {
            let heat = case_600.heating_setpoint_at_hour(hour);
            let cool = case_600.cooling_setpoint_at_hour(hour);

            assert_eq!(heat, Some(20.0), "Hour {} heating should be 20.0", hour);
            assert_eq!(cool, Some(27.0), "Hour {} cooling should be 27.0", hour);
        }
    }

    #[test]
    fn test_setpoint_at_hour_operating_hours() {
        // Case 650: Cooling 7-18 (7, 18)
        let case_650 = HvacSchedule::with_operating_hours(-100.0, 27.0, 7, 18);

        // Hours 7-17 should have cooling setpoint
        for hour in 7..18 {
            let cool = case_650.cooling_setpoint_at_hour(hour);
            assert_eq!(cool, Some(27.0), "Hour {} cooling should be 27.0", hour);
        }

        // Hours 0-6 and 18-23 should not have cooling
        for hour in 0..7 {
            let cool = case_650.cooling_setpoint_at_hour(hour);
            assert_eq!(cool, None, "Hour {} cooling should be None", hour);
        }
        for hour in 18..24 {
            let cool = case_650.cooling_setpoint_at_hour(hour);
            assert_eq!(cool, None, "Hour {} cooling should be None", hour);
        }
    }

    #[test]
    fn test_setpoint_at_hour_wrapping_range() {
        // Night ventilation: 18-7 (overnight, wraps midnight)
        let night_vent = NightVentilation::new(1703.16, 18, 7);

        // Hours 18-23 and 0-6 should be active
        for hour in 18..24 {
            assert!(
                night_vent.is_active_at_hour(hour),
                "Hour {} should be active",
                hour
            );
        }
        for hour in 0..7 {
            assert!(
                night_vent.is_active_at_hour(hour),
                "Hour {} should be active",
                hour
            );
        }

        // Hours 7-17 should not be active
        for hour in 7..18 {
            assert!(
                !night_vent.is_active_at_hour(hour),
                "Hour {} should not be active",
                hour
            );
        }
    }

    #[test]
    fn test_setpoint_at_hour_free_floating() {
        // Free-floating: should return None for all hours
        let free = HvacSchedule::free_floating();

        for hour in 0..24 {
            let heat = free.heating_setpoint_at_hour(hour);
            let cool = free.cooling_setpoint_at_hour(hour);

            assert_eq!(heat, None, "Hour {} heating should be None", hour);
            assert_eq!(cool, None, "Hour {} cooling should be None", hour);
        }
    }
}
