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
//! - `CaseSpec` — has `Option<crate::sim::hvac::AnyEquipment>` field
//! - `CaseBuilder` — builder that calls into sim/physics
//! - `CommonWall`, `ConstructionSpec` — depend on `crate::sim::construction::Construction`
//! - `WindowSpec::with_dimensions`-style high-level helpers that compose
//!   simulation state.

#![allow(clippy::option_as_ref_deref)]

use serde::{Deserialize, Serialize};

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
    /// Implements sub-hour HVAC mode interpolation (Issue #2870 — Case 940
    /// morning overshoot). For wraparound setback schedules (start > end, e.g.,
    /// `23→7`) the discrete `setback → occupied` jump at the setback-end hour is
    /// replaced with a 2-hour linear ramp from the setback setpoint up to the
    /// occupied setpoint. Outside the ramp window the function degrades to the
    /// discrete `heating_setpoint_at_hour` floor-hour lookup.
    ///
    /// The ramp applies only when:
    ///   * the schedule has a setback window with `setback_start > setback_end`
    ///     (wraparound overnight setback — covers ASHRAE 140 Cases 640/940),
    ///   * `setback_end` is in `[1, 12]` so the ramp falls inside the morning
    ///     window between wake-up time and midday occupied heating, and
    ///   * the setback setpoint is distinct from the occupied setpoint (no ramp
    ///     is added when the two values are equal).
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

        // Sub-hour ramp: linear blend between setback and occupied spanning
        // the 2-hour window from `setback_end` to `setback_end + RAMP_HOURS`.
        const RAMP_HOURS: f64 = 2.0;
        let ramp_eligible =
            sb_start > sb_end && sb_end >= 1 && sb_end <= 12 && (setback - occupied).abs() > 1e-9;
        if ramp_eligible {
            let ramp_start = f64::from(sb_end);
            let ramp_end = (f64::from(sb_end) + RAMP_HOURS).min(24.0);
            if fh >= ramp_start && fh < ramp_end {
                let t = (fh - ramp_start) / (ramp_end - ramp_start);
                let ramped = setback + t * (occupied - setback);
                // Respect operating-hours gate (e.g., Case 950 disables heat entirely).
                let (start, end) = self.operating_hours;
                let is_operating = if start < end {
                    h_floor >= start && h_floor < end
                } else if start > end {
                    h_floor >= start || h_floor < end
                } else {
                    true
                };
                return if is_operating { Some(ramped) } else { None };
            }
        }

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
