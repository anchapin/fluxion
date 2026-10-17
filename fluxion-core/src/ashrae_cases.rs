//! ASHRAE Standard 140 leaf data types — moved from `fluxion::validation::ashrae_140_cases`.
//!
//! Issue #1441: These types are pure-data leaf structs/enums with **no upward
//! dependencies on `sim`, `physics`, `ai`, or any other non-leaf module**. They
//! were hoisted into `fluxion-core` to break the `sim ↔ validation` dependency
//! cycle documented in `ARCHITECTURE.md` §"Remaining cycles".
//!
//! ## Why a separate crate?
//!
//! `cargo-mutants` recompiles the *target* crate for every generated mutant.
//! `fluxion-core` is built once and cached, while cargo-mutants mutates only
//! `fluxion`. By moving these leaf types out of the main crate, the per-mutant
//! compile no longer pulls in the 208 KB `validation::ashrae_140_cases` module
//! (and its upward dep chain into `sim::construction`).
//!
//! ## Re-export shim
//!
//! `fluxion` re-exports this module at top level (`pub use fluxion_core::ashrae_cases;`)
//! and `fluxion::validation::ashrae_140_cases` re-exports each type, so existing
//! `crate::validation::ashrae_140_cases::Orientation` and
//! `fluxion::validation::ashrae_140_cases::Orientation` paths resolve unchanged.
//!
//! ## Contents
//!
//! | Type | Lines | Notes |
//! |------|-------|-------|
//! | `Orientation`        | pure enum + impl | azimuth/prefix helpers |
//! | `WindowArea`         | struct + impl    | uses `Orientation` |
//! | `ConstructionType`   | enum            | low/high/special mass |
//! | `ShadingType`        | enum            | none/overhang/fins |
//! | `ShadingDevice`      | struct + impl   | uses `ShadingType` |
//! | `GlassType`          | enum + impl     | emissivity table |
//! | `WindowSpec`         | struct + impl   | uses `GlassType` |
//! | `InternalLoads`      | struct + impl   | radiative/convective split |
//! | `HvacSchedule`       | struct + impl   | setpoints + operating hours |
//! | `NightVentilation`   | struct + impl   | Case 650 schedule |
//! | `BuildingType`       | enum + Default  | furniture factor categories |
//! | `GeometrySpec`       | struct + impl   | zone width/depth/height |
//! | `ConductanceReferences` | struct       | Case 600 reference W/K |
//!
//! ## What is NOT here (stays in `fluxion::validation::ashrae_140_cases`)
//!
//! - `ASHRAE140Case` — large enum with hundreds of body methods (depends on sim)
//! - `CaseSpec` — carries `Option<EquipmentSelection>` (data-only, slice 2 of
//!   issue #4157) — still lives in the validation layer pending slice 3
//! - `CaseBuilder` — builder that calls into sim/physics
//! - `CommonWall`, `ConstructionSpec` — depend on `crate::sim::construction::Construction`
//! - `WindowSpec::with_dimensions`-style high-level helpers that compose
//!   simulation state.

#![allow(clippy::option_as_ref_deref)]

use serde::{Deserialize, Serialize};

// =============================================================================
// EquipmentSelection (issue #4157 slice 2)
// =============================================================================

/// Polynomial efficiency-curve parameters (issue #4157 slice 2).
///
/// Data-only mirror of `fluxion::sim::hvac::efficiency_curves::EfficiencyCurve`:
/// the same three facts its `new` constructor takes, carried as parameters so a
/// [`EquipmentSelection`] stays free of any `sim` dependency. The sim-side
/// mapping site (issue #4157 slice 2) converts this to a real
/// `EfficiencyCurve` with identical values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EfficiencyCurveSpec {
    /// Part-load-ratio polynomial coefficients (order 3 down to constant, as
    /// `EfficiencyCurve::new` takes them).
    pub plr_coefficients: [f64; 4],
    /// Temperature degradation coefficient.
    pub temp_coefficient: f64,
    /// Design temperature (°C).
    pub design_temp: f64,
}

/// Data-only HVAC equipment selection carried on `CaseSpec` (issue #4157
/// slice 2).
///
/// Variant payloads are parameters only — capacities, efficiency ratings,
/// design temperatures, curve coefficients, airflow rates — the same facts the
/// `CaseBuilder` sites write today. `fluxion::sim::ThermalModel::from_spec*`
/// maps this selection to the behavioral `AnyEquipment` family at a single
/// site, so `CaseSpec` describes equipment without depending on `sim`.
///
/// Values are bit-identical to the pre-slice-2 builder output: the mapping
/// calls the same constructors with the same numbers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EquipmentSelection {
    /// Air-to-air heat pump (heating + cooling).
    HeatPump {
        /// System identifier.
        id: String,
        /// Rated heating capacity at design conditions (W).
        heating_capacity_w: f64,
        /// Rated cooling capacity at design conditions (W).
        cooling_capacity_w: f64,
        /// Rated heating COP at design conditions.
        heating_cop: f64,
        /// Rated cooling COP (EER) at design conditions.
        cooling_eer: f64,
        /// Overrides the default heating efficiency curve when present.
        heating_curve: Option<EfficiencyCurveSpec>,
        /// Overrides the default cooling efficiency curve when present.
        cooling_curve: Option<EfficiencyCurveSpec>,
    },
    /// Chiller (cooling-only plant).
    Chiller {
        /// System identifier.
        id: String,
        /// Rated cooling capacity (W).
        cooling_capacity_w: f64,
        /// Rated cooling COP.
        cop: f64,
        /// Design temperature (°C).
        design_temp_c: f64,
    },
    /// Boiler (heating-only plant).
    Boiler {
        /// System identifier.
        id: String,
        /// Rated heating capacity (W).
        heating_capacity_w: f64,
        /// Thermal efficiency (fraction of 1).
        efficiency: f64,
        /// Design supply temperature (°C).
        design_temp_c: f64,
    },
    /// VAV terminal unit.
    VavTerminal {
        /// Unit identifier.
        id: String,
        /// Served zone index.
        zone_id: usize,
        /// Maximum airflow (m³/s).
        max_airflow_m3_per_s: f64,
    },
    /// CAV (constant-volume) system.
    CavSystem {
        /// System identifier.
        id: String,
        /// Design airflow (m³/s).
        design_airflow_m3_per_s: f64,
    },
}

// =============================================================================
// Orientation
// =============================================================================

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

    /// Returns a short prefix identifier for surface naming (e.g., "N", "S", "E", "W", "Up", "Down", "H").
    pub fn prefix(&self) -> &'static str {
        match self {
            Orientation::North => "N",
            Orientation::East => "E",
            Orientation::South => "S",
            Orientation::West => "W",
            Orientation::Up => "Up",
            Orientation::Down => "Down",
            Orientation::Horizontal => "H",
        }
    }
}

// =============================================================================
// WindowArea
// =============================================================================

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

// =============================================================================
// ConstructionType
// =============================================================================

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

// =============================================================================
// ShadingDevice + ShadingType
// =============================================================================

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

// =============================================================================
// GlassType + WindowSpec
// =============================================================================

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
    /// Additive extra U-value contribution from the frame (W/m²K).
    /// Issue #2889 — frame-to-glazing thermal bridge. Default 0.1 W/m²K
    /// per ASHRAE 140 §5.2.4 (low end of the "5–15 % additional U-value
    /// on perimeter" range). The effective window U-value used in
    /// `h_tr_w = U_eff × A_win` adds `frame_u_value` on top of the
    /// center-of-glass U.
    pub frame_u_value: f64,
    /// Fraction of total window area that is frame (0–1). Issue #2889 —
    /// default 0.15 per ASHRAE 140 Bestest framing schedule. Gating: when
    /// 0.0 the frame bridge is fully suppressed.
    pub frame_area_fraction: f64,
    /// Frame perimeter in metres. Issue #2889 — used with the linear edge
    /// conductance coefficient (0.2 W/(m·K) by default) to model the
    /// frame-to-glazing transition. If 0.0, the linear term is omitted.
    pub frame_perimeter: f64,
}

/// ASHRAE 140 §5.2 / BESTEST window conductance for the 600/900-series
/// double-clear-glazing cases: 3.0 W/m²K (36 W/K over the 12 m² south
/// window). WINDOW-01 (loop round 7): this is the single source of truth for
/// the spec-level window U-value; the engine's `physics::constants` table
/// re-exports it, and the 600/900-series CaseSpec builders thread it through
/// `window_u_value_override` so both engine entry points agree. Lives here
/// (downward) so validation-side builders can reference it without adding a
/// validation→physics cycle edge (issue #1441/#2495 guards).
pub const ASHRAE140_WINDOW_U_VALUE: f64 = 3.0;

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
            frame_u_value: 0.1,
            frame_area_fraction: 0.15,
            frame_perimeter: 0.0,
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

    /// ASHRAE 140 §5.2.4 — effective window U-value including the frame
    /// thermal bridge (W/m²K).
    ///
    /// Combines:
    /// 1. The center-of-glass U-value (`self.u_value`).
    /// 2. The additive frame contribution (`self.frame_u_value`, per unit
    ///    total window area). This is the area-weighted uplift from the
    ///    frame vs glass delta, applied to the whole window — within
    ///    the ASHRAE 140 §5.2.4 "5–15 % additional U-value" range.
    /// 3. The linear edge conductance at the frame-to-glazing transition:
    ///    `psi × perimeter / total_area`. The `total_area` argument is the
    ///    total window area; the linear term is omitted when `self.frame_perimeter == 0.0`.
    ///
    /// `psi` defaults to 0.2 W/(m·K) per ASHRAE 140 §5.2.4 (Bestest
    /// convention). This is the value used by the engine when wiring
    /// `WindowSpec` into `h_tr_w`. The frame bridge is gated on
    /// `frame_area_fraction > 0.0` so that "fully glazed" windows (no
    /// frame) skip it entirely.
    pub fn effective_u_value_with_frame(&self, total_area: f64, linear_edge_psi: f64) -> f64 {
        let f_frame = self.frame_area_fraction.clamp(0.0, 1.0);
        if f_frame <= 0.0 {
            return self.u_value;
        }
        let area_delta = self.frame_u_value.max(0.0);
        let edge_delta = if self.frame_perimeter > 0.0 && total_area > 0.0 {
            linear_edge_psi * self.frame_perimeter / total_area
        } else {
            0.0
        };
        self.u_value + area_delta + edge_delta
    }
}

// =============================================================================
// InternalLoads
// =============================================================================

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

// =============================================================================
// HvacSchedule
// =============================================================================

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

    /// Creates an HVAC schedule with BOTH operating hours restriction AND
    /// a setback window. Used by ASHRAE 140 Case 950 (issue #1347).
    ///
    /// # Arguments
    /// * `heating_setpoint` - Normal heating setpoint in °C
    /// * `cooling_setpoint` - Cooling setpoint in °C
    /// * `operating_start` - Hour when HVAC turns on (0-23)
    /// * `operating_end` - Hour when HVAC turns off (0-23)
    /// * `setback_setpoint` - Setpoint applied during the setback window
    /// * `setback_start` - Hour when setback starts (0-23)
    /// * `setback_end` - Hour when setback ends (0-23)
    pub fn with_operating_hours_and_setback(
        heating_setpoint: f64,
        cooling_setpoint: f64,
        operating_start: u8,
        operating_end: u8,
        setback_setpoint: f64,
        setback_start: u8,
        setback_end: u8,
    ) -> Self {
        HvacSchedule {
            heating_setpoint,
            cooling_setpoint,
            operating_hours: (operating_start, operating_end),
            setback_setpoint: Some(setback_setpoint),
            setback_hours: Some((setback_start, setback_end)),
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
            true
        };

        if is_operating {
            return Some(current_setpoint);
        }

        None
    }

    /// Returns the heating setpoint at a fractional hour of day in `[0.0, 24.0)`.
    ///
    /// Returns the discrete scheduled setpoint at the floor hour, mirroring
    /// `heating_setpoint_at_hour`. (Issue #2870 originally added a 2-hour
    /// linear ramp from the setback setpoint to the occupied setpoint over
    /// wraparound morning setback ends; removed in round 16 per issue #4226 —
    /// ASHRAE 140-2023 §5.5 specifies a step change at the setback-end hour,
    /// and the ramp was a specified-boundary deviation recorded in
    /// docs/investigations/issue-4196-setback-ramp-deviation.md.)
    ///
    /// `None` is returned when the schedule is disabled (free-floating), mirroring
    /// `heating_setpoint_at_hour`.
    pub fn heating_setpoint_at_fractional_hour(&self, fractional_hour: f64) -> Option<f64> {
        if !self.is_enabled() {
            return None;
        }

        // Normalize to [0.0, 24.0); negative inputs and > 24 wrap accordingly.
        let fh = fractional_hour.rem_euclid(24.0);
        let occupied = self.heating_setpoint;
        let setback = self.setback_setpoint.unwrap_or(occupied);

        // No setback window configured: discrete occupied setpoint at any hour.
        let Some((sb_start, sb_end)) = self.setback_hours else {
            return Some(occupied);
        };

        // Discrete lookup at integer floor hour — mirrors `heating_setpoint_at_hour`.
        let h_floor = fh.floor().clamp(0.0, 23.0) as u8;
        let in_discrete_setback_window = if sb_start <= sb_end {
            h_floor >= sb_start && h_floor < sb_end
        } else {
            h_floor >= sb_start || h_floor < sb_end
        };
        let discrete_value = if in_discrete_setback_window {
            setback
        } else {
            occupied
        };

        // Respect operating-hours gate (mirrors `heating_setpoint_at_hour`).
        let (start, end) = self.operating_hours;
        let is_operating = if start < end {
            h_floor >= start && h_floor < end
        } else if start > end {
            h_floor >= start || h_floor < end
        } else {
            true
        };

        if is_operating {
            Some(discrete_value)
        } else {
            None
        }
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
            true
        };

        if is_operating {
            return Some(self.cooling_setpoint);
        }

        None
    }
}

// =============================================================================
// NightVentilation
// =============================================================================

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
    ///
    /// Fan capacity 1400 standard m³/h (10.80 ach on the 129.6 m³ test cell),
    /// per ASHRAE 140 Table 5-160: the 1400 m³/h input is the value specified
    /// for "programs with fixed assumption that site is at sea level (no
    /// automatic correction of fan mass flow)". This engine applies a fixed
    /// ρ = 1.2 kg/m³ everywhere (infiltration included), so the
    /// altitude-correcting route's 1703.16 m³/h input (13.14 ach, density
    /// 1.0156 kg/m³ at the table's 1650.9 m reference altitude) does NOT
    /// apply to it: pairing 1703.16 with ρ = 1.2 ventilated at
    /// 2043.8 kg/h — 21.7 % more mass flow than this route's 1400 × 1.2 =
    /// 1680 kg/h and 18 % more than the altitude-correcting route's
    /// 1703.16 × 1.0156 = 1730 kg/h. Corrected loop round 26 (2026-10-13).
    pub fn case_650() -> Self {
        NightVentilation {
            fan_capacity: 1400.0, // standard m³/h (ASHRAE 140 Table 5-160, sea-level route)
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

// =============================================================================
// BuildingType
// =============================================================================

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
    /// Warehouse / light-commercial buildings - f_furniture = 0.5
    Warehouse,
}

impl Default for BuildingType {
    fn default() -> Self {
        BuildingType::Residential
    }
}

// =============================================================================
// GeometrySpec
// =============================================================================

/// Geometry specification for a single zone.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

// =============================================================================
// ConductanceReferences
// =============================================================================

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

// ---------------------------------------------------------------------------
// Composite case types — #4157 slice 3 (loop round 50).
//
// `CaseSpec`, `ConstructionSpec`, `CommonWall` and their helper `ZoneOrigin`
// move here from `fluxion::validation::ashrae_140_cases` (issue #2495 called
// these the composite types that keep the sim↔validation cycle alive). After
// slice 2 (#4458) removed the last upward dependency (`hvac_equipment` became
// `Option<EquipmentSelection>`), every remaining dependency of these types is
// already in fluxion-core: the leaf types in this module, `Construction` /
// `Materials` in `fluxion_core::construction`, and `HourlyWeatherData` in
// `fluxion_core::weather`. The physics constants used by
// `CaseSpec::calculate_iso13790_conductances` are the identical values from
// `fluxion_core::construction` (the `crate::physics::constants` copies are
// value-identical re-exports). `fluxion::validation::ashrae_140_cases`
// re-exports them, so every existing path keeps compiling.
// ---------------------------------------------------------------------------

use std::path::PathBuf;

use crate::construction::{
    Construction, SurfaceType, AIR_DENSITY_SEA_LEVEL, AIR_SPECIFIC_HEAT,
    INTERIOR_FILM_COEFF_CEILING, INTERIOR_FILM_COEFF_FLOOR, INTERIOR_FILM_COEFF_WALL,
};

use super::weather::HourlyWeatherData;

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

/// Zone plan origin (meters) for perimeter-adjacency derivation.
///
/// Round 21 (§LIMIT-23b): multi-zone cases whose zones are laid out on a
/// shared plan (origin + width/depth per zone) can declare these origins so
/// the engine derives which facades lie on the building perimeter. Without
/// origins every zone keeps the legacy behaviour (all six surfaces treated
/// as outdoor-exposed), so existing specs parse and simulate unchanged.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ZoneOrigin {
    /// West-edge x coordinate (meters)
    pub x: f64,
    /// South-edge y coordinate (meters, y grows northward)
    pub y: f64,
}

impl ZoneOrigin {
    /// Creates a zone plan origin.
    pub fn new(x: f64, y: f64) -> Self {
        ZoneOrigin { x, y }
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

    /// Optional per-zone plan origins (meters), indexed by zone.
    /// When present, the engine derives perimeter-only exterior facade areas
    /// from zone adjacency (§LIMIT-23b, round 21) instead of treating every
    /// zone's walls as outdoor-exposed. `None` keeps the legacy behaviour.
    #[serde(default)]
    pub zone_origins: Option<Vec<ZoneOrigin>>,

    /// Per-zone construction assemblies (round 24, §LIMIT-14).
    ///
    /// `None` (serde default, every existing case) keeps the legacy single
    /// `construction` for all zones — bit-identical behaviour. When present,
    /// length must equal `num_zones`, and each zone builds from its own
    /// `ConstructionSpec` (140-2017 §5.2.2.2.7: Case 960's back zone is
    /// Case 600 lightweight while the sun zone is Case 900 heavyweight —
    /// not expressible with the single global construction).
    pub zone_constructions: Option<Vec<ConstructionSpec>>,

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
    pub hvac_equipment: Option<EquipmentSelection>,
    /// Ground temperature boundary condition (°C) for floor slab.
    /// Per ASHRAE 140-2023 Annex B §B3.3: T_ground = 9.4°C for all cases with floor slab.
    /// When `None`, the model default (10.0°C) is used for backward compatibility.
    pub ground_temperature_c: Option<f64>,
    /// Optional override for the floor U-value (W/m²K) used in ground coupling.
    /// When `Some`, replaces the construction-derived floor U-value in both
    /// `setpoints.floor_u_value` and the `h_tr_floor` conductance. Used for
    /// specified test conditions such as the ASHRAE 140 Case 195 solid
    /// conduction test (U = 0.039 W/m²K).
    pub floor_u_value_override: Option<f64>,
    /// Optional override for the window U-value (W/m²K).
    /// WINDOW-01 (loop round 7): the ASHRAE 140 spec defines the Case
    /// 600/900-series window conductance as 3.0 W/m²K (WINDOW_U_VALUE, §5.2),
    /// which the legacy validator setup path sets directly; the WindowSpec
    /// glazing U (2.10) + frame bridge composes to ~2.27 W/m²K instead. When
    /// `Some`, the effective window U-value — `solar.window_u_value` and the
    /// `h_tr_w` conductance — is this value, so both engine entry points agree.
    /// Data-driven, no case_id check.
    #[serde(default)]
    pub window_u_value_override: Option<f64>,
    /// Building usage type for thermal mass calculations (default: Residential)
    pub building_type: BuildingType,
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
        self.wall.u_value(Some(SurfaceType::Wall), None)
    }

    /// Returns the total roof U-value (with ASHRAE film coefficients).
    pub fn roof_u_value(&self) -> f64 {
        self.roof.u_value(Some(SurfaceType::Ceiling), None)
    }

    /// Returns the total floor U-value (with ASHRAE film coefficients).
    pub fn floor_u_value(&self) -> f64 {
        self.floor.u_value(Some(SurfaceType::Floor), None)
    }
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

        // Round 21 (§LIMIT-23b): a malformed zone-origins vector must not
        // silently fall back to the every-wall-exterior legacy behaviour —
        // the adjacency helpers return `None` on a length mismatch, so
        // surface derivation would quietly revert. Reject it instead.
        if let Some(cons) = &self.zone_constructions {
            if cons.len() != self.num_zones {
                return Err(format!(
                    "zone_constructions length ({}) must match num_zones ({})",
                    cons.len(),
                    self.num_zones
                ));
            }
        }
        if let Some(origins) = &self.zone_origins {
            if origins.len() != self.num_zones {
                return Err(format!(
                    "zone_origins length ({}) must match num_zones ({})",
                    origins.len(),
                    self.num_zones
                ));
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

    /// Per-zone exterior facade areas `[South, West, North, East]` in m²,
    /// derived from zone plan origins and adjacency (round 21, §LIMIT-23b).
    ///
    /// Returns `None` when `zone_origins` is absent (legacy behaviour: every
    /// zone wall treated as outdoor-exposed) or malformed. A facade is
    /// exterior where no adjacent zone covers it: the covered length of a
    /// face is the sum of the overlaps of neighbouring zones that share the
    /// face plane, so partially covered faces keep their exposed remainder.
    pub fn exterior_wall_areas_by_zone(&self) -> Option<Vec<[f64; 4]>> {
        const ADJACENCY_EPS: f64 = 1e-6;
        let origins = self.zone_origins.as_ref()?;
        if origins.len() < self.num_zones || self.geometry.len() < self.num_zones {
            return None;
        }
        let extent = |i: usize, vertical: bool| -> (f64, f64) {
            let g = &self.geometry[i];
            let o = &origins[i];
            if vertical {
                (o.x, o.x + g.width)
            } else {
                (o.y, o.y + g.depth)
            }
        };
        let mut out = Vec::with_capacity(self.num_zones);
        for i in 0..self.num_zones {
            let covered_len =
                |span_along_x: bool, span: (f64, f64), touches: &dyn Fn(usize) -> bool| -> f64 {
                    let mut covered = 0.0;
                    for j in 0..self.num_zones {
                        if j == i || !touches(j) {
                            continue;
                        }
                        let (t0, t1) = extent(j, span_along_x);
                        let overlap = (span.1.min(t1) - span.0.max(t0)).max(0.0);
                        covered += overlap;
                    }
                    covered
                };
            let g = &self.geometry[i];
            let o = &origins[i];
            let (x0, x1) = (o.x, o.x + g.width);
            let (y0, y1) = (o.y, o.y + g.depth);
            // South face: plane y = y0; a neighbour covers it when the
            // neighbour's north edge (y + depth) coincides with y0.
            let s_covered = covered_len(true, (x0, x1), &|j| {
                (origins[j].y + self.geometry[j].depth - y0).abs() < ADJACENCY_EPS
            });
            // West face: plane x = x0; neighbour's east edge coincides.
            let w_covered = covered_len(false, (y0, y1), &|j| {
                (origins[j].x + self.geometry[j].width - x0).abs() < ADJACENCY_EPS
            });
            // North face: plane y = y1; neighbour's south edge coincides.
            let n_covered = covered_len(true, (x0, x1), &|j| {
                (origins[j].y - y1).abs() < ADJACENCY_EPS
            });
            // East face: plane x = x1; neighbour's west edge coincides.
            let e_covered = covered_len(false, (y0, y1), &|j| {
                (origins[j].x - x1).abs() < ADJACENCY_EPS
            });
            out.push([
                ((x1 - x0) - s_covered).max(0.0) * g.height,
                ((y1 - y0) - w_covered).max(0.0) * g.height,
                ((x1 - x0) - n_covered).max(0.0) * g.height,
                ((y1 - y0) - e_covered).max(0.0) * g.height,
            ]);
        }
        Some(out)
    }

    /// Per-zone share of interior partition wall area in m², derived from
    /// `common_walls` (round 21, §LIMIT-23b): each partition's area is split
    /// evenly between the two zones it separates, so per-zone fabric mass is
    /// not double-counted. Returns `None` when `zone_origins` is absent.
    pub fn partition_wall_area_by_zone(&self) -> Option<Vec<f64>> {
        self.zone_origins.as_ref()?;
        let mut out = vec![0.0; self.num_zones];
        for wall in &self.common_walls {
            let half = wall.area / 2.0;
            if wall.zone_a < self.num_zones {
                out[wall.zone_a] += half;
            }
            if wall.zone_b < self.num_zones {
                out[wall.zone_b] += half;
            }
        }
        Some(out)
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
        CaseSpec::calculate_iso13790_conductances(self)
    }

    /// Calculate ISO 13790 / ASHRAE 140 compliant conductances from case geometry and materials.
    ///
    /// This function replaces placeholder values with physics-based calculations per the
    /// ISO 13790 5R1C thermal network model:
    ///
    /// - **H_tr_em** (opaque envelope to external): Series combination of opaque U-values
    ///   using ISO 13790 Eq. 64: h_em = 1 / (1/h_op - 1/h_ms)
    /// - **H_tr_w** (windows): U_window × A_window
    /// - **H_tr_ms** (mass-to-surface coupling): h_ms_coeff × A_m where h_ms_coeff=9.1 W/m²K (ISO 13790)
    ///   and A_m = 2.5 × A_f (effective mass area per ISO 13790 Table C.2)
    /// - **H_tr_is** (surface-to-indoor air): Σ(h_is × A_surface) for all interior surfaces
    /// - **H_ve** (ventilation): ρ_air × c_p × V̇ where V̇ = ACH/3600 × V_zone
    ///
    /// # Arguments
    /// * `case_spec` - The ASHRAE 140 case specification with geometry and construction
    ///
    /// # Returns
    /// `ConductanceReferences` with all conductances in W/K
    ///
    /// # Calculation Details
    ///
    /// For Case 600 (low-mass baseline):
    /// - Geometry: 8.0m × 6.0m × 2.7m (W×D×H)
    /// - Floor area: 48.0 m², Volume: 129.6 m³
    /// - Wall area: 75.6 m² (gross), 63.6 m² (opaque, 12 m² windows)
    /// - Roof area: 48.0 m²
    /// - Windows: 12.0 m² south-facing (double clear glass, U=2.10 W/m²K)
    /// - ACH: 0.5
    ///
    /// Construction U-values (from material properties):
    /// - Wall: 0.5119 W/m²K (plasterboard + fiberglass + wood siding)
    /// - Roof: 0.3198 W/m²K (plasterboard + fiberglass + roof deck)
    /// - Floor: 0.1837 W/m²K (timber + fiberglass, with ground coupling)
    pub fn calculate_iso13790_conductances(case_spec: &CaseSpec) -> ConductanceReferences {
        let geo = &case_spec.geometry[0];
        let width = geo.width;
        let depth = geo.depth;
        let height = geo.height;
        let floor_area = width * depth;
        let volume = floor_area * height;
        let gross_wall_area = 2.0 * (width + depth) * height;
        let window_area = case_spec.total_window_area();
        let opaque_wall_area = (gross_wall_area - window_area).max(0.0);
        let roof_area = floor_area;

        let rho_air = AIR_DENSITY_SEA_LEVEL;
        let cp_air = AIR_SPECIFIC_HEAT;
        let h_ve = rho_air * cp_air * (case_spec.infiltration_ach / 3600.0) * volume;

        // WINDOW-01: honour the spec-level window U override when present so
        // the audit agrees with the engine's h_tr_w.
        let window_u = case_spec
            .window_u_value_override
            .unwrap_or(case_spec.window_properties.u_value);
        let h_tr_w = window_u * window_area;

        let wall_u = case_spec.construction.wall.u_value(None, None);
        let roof_u = case_spec.construction.roof.u_value(None, None);

        let h_op_walls = wall_u * opaque_wall_area;
        let h_op_roof = roof_u * roof_area;
        let h_op_total = h_op_walls + h_op_roof;

        let a_m = 2.5 * floor_area;
        let h_ms_coeff = 9.1;
        let h_ms_iso = h_ms_coeff * a_m;

        let h_tr_em = if h_op_total > 0.0 && h_op_total < h_ms_iso {
            1.0 / (1.0 / h_op_total - 1.0 / h_ms_iso)
        } else {
            h_op_total + h_tr_w
        };

        let h_is_wall = INTERIOR_FILM_COEFF_WALL;
        let h_is_ceiling = INTERIOR_FILM_COEFF_CEILING;
        let h_is_floor = INTERIOR_FILM_COEFF_FLOOR;
        let h_tr_is =
            h_is_wall * gross_wall_area + h_is_ceiling * roof_area + h_is_floor * floor_area;

        let h_tr_ms = h_ms_iso;

        ConductanceReferences {
            h_tr_em: h_tr_em.max(0.1),
            h_tr_w,
            h_tr_ms,
            h_tr_is,
            h_ve,
        }
    }
}

// Issue #4157 (round 52): `CaseBuilder` moved to `ashrae_case_builder`; re-exported
// here so historical `fluxion_core::ashrae_cases::CaseBuilder` paths resolve.
pub use crate::ashrae_case_builder::CaseBuilder;
