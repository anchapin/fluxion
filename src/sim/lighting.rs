//! Daylighting and Lighting Control Models
//!
//! This module provides daylighting modeling and automated lighting controls
//! for commercial building energy simulations.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Window optical properties (Phase 1 — per-layer transmittance/absorptance)
// ---------------------------------------------------------------------------

/// A single glazing layer (pane + gas gap).
///
/// Multiplicative transmission across layers follows:
/// `T_system = ∏ T_layer`, `α_system = 1 - T_system - R_system`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WindowLayer {
    /// Solar transmittance of this layer (0–1).
    pub solar_transmittance: f64,
    /// Visible transmittance of this layer (0–1).
    pub visible_transmittance: f64,
    /// Solar absorptance of this layer (0–1).
    pub solar_absorptance: f64,
    /// Solar reflectance of this layer (0–1).
    pub solar_reflectance: f64,
}

impl WindowLayer {
    /// Single clear pane (6 mm glass).
    pub fn clear_glass() -> Self {
        Self {
            solar_transmittance: 0.837,
            visible_transmittance: 0.898,
            solar_absorptance: 0.095,
            solar_reflectance: 0.068,
        }
    }

    /// Low-e coated pane.
    pub fn low_e() -> Self {
        Self {
            solar_transmittance: 0.603,
            visible_transmittance: 0.745,
            solar_absorptance: 0.140,
            solar_reflectance: 0.257,
        }
    }

    /// Opaque shading layer (e.g. fully closed roller shade).
    pub fn opaque_shade() -> Self {
        Self {
            solar_transmittance: 0.05,
            visible_transmittance: 0.10,
            solar_absorptance: 0.45,
            solar_reflectance: 0.50,
        }
    }
}

/// Aggregate optical properties for a complete window assembly (one or more layers).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WindowOpticalProperties {
    /// System solar transmittance (product of layer transmittances).
    pub solar_transmittance: f64,
    /// System visible transmittance (product of layer visible transmittances).
    pub visible_transmittance: f64,
    /// System solar absorptance.
    pub solar_absorptance: f64,
    /// System solar reflectance.
    pub solar_reflectance: f64,
}

impl WindowOpticalProperties {
    /// Compute system-level optics from individual layers using Beer-Lambert multiplication.
    pub fn from_layers(layers: &[WindowLayer]) -> Self {
        let mut ts = 1.0;
        let mut tv = 1.0;
        let mut absorptance_sum = 0.0;

        for layer in layers {
            ts *= layer.solar_transmittance;
            tv *= layer.visible_transmittance;
            // Absorbed fraction of incident radiation at each layer (first-order model).
            absorptance_sum += layer.solar_absorptance * ts;
        }

        let reflectance = 1.0 - ts - absorptance_sum;

        Self {
            solar_transmittance: ts,
            visible_transmittance: tv,
            solar_absorptance: absorptance_sum.min(1.0),
            solar_reflectance: reflectance.max(0.0),
        }
    }

    /// Default double-clear ASHRAE 140 window.
    pub fn double_clear() -> Self {
        Self::from_layers(&[WindowLayer::clear_glass(), WindowLayer::clear_glass()])
    }
}

/// Lighting control types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LightingControlType {
    /// Manual on/off switching
    Manual,
    /// Continuous dimming based on daylight
    ContinuousDimming,
    /// Stepped dimming (multiple levels)
    SteppedDimming,
    /// Occupancy-based on/off
    OccupancySensing,
}

/// Represents a daylight zone for lighting control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaylightZone {
    /// Zone identifier
    pub id: String,
    /// Associated thermal zone
    pub thermal_zone_id: usize,
    /// Window area contributing daylight (m²)
    pub window_area: f64,
    /// Window height for daylight calculations (m)
    pub window_height: f64,
    /// Depth of daylight zone (m)
    pub daylight_zone_depth: f64,
    /// Average daylight factor (%) — legacy simple model
    pub daylight_factor: f64,
    /// Illuminance threshold for dimming (lux)
    pub dimming_threshold: f64,
    /// Minimum lighting level when dimming (fraction 0-1)
    pub min_dimming_level: f64,
    /// Zone floor area (m²). Used in split-flux illuminance model.
    pub zone_area: f64,
    /// Height of the workplane above the floor (m). Default 0.8 m.
    pub workplane_height: f64,
    /// Window visible transmittance. When set, the split-flux model uses
    /// this instead of the legacy daylight factor.
    pub visible_transmittance: Option<f64>,
    /// Room cavity ratio (floor-to-ceiling / window-to-ceiling), used for
    /// flux transfer factor. 0 = auto-compute from geometry.
    pub room_cavity_ratio: f64,
}

impl DaylightZone {
    /// Create a new daylight zone with default geometry.
    pub fn new(id: String, thermal_zone_id: usize, window_area: f64, window_height: f64) -> Self {
        Self {
            id,
            thermal_zone_id,
            window_area,
            window_height,
            daylight_zone_depth: window_height * 1.5,
            daylight_factor: 5.0,
            dimming_threshold: 300.0,
            min_dimming_level: 0.1,
            zone_area: window_area * 1.5,
            workplane_height: 0.8,
            visible_transmittance: None,
            room_cavity_ratio: 0.0,
        }
    }

    /// Builder-style setter for window visible transmittance (enables split-flux model).
    pub fn with_visible_transmittance(mut self, vt: f64) -> Self {
        self.visible_transmittance = Some(vt);
        self
    }

    /// Builder-style setter for zone area.
    pub fn with_zone_area(mut self, area: f64) -> Self {
        self.zone_area = area;
        self
    }

    /// Auto-compute room cavity ratio from zone geometry.
    ///
    /// RCR = (5 * floor_area * workplane_height) / (window_area * ceiling_height)
    /// where ceiling_height ≈ window_height + workplane_height.
    fn compute_rcr(&self) -> f64 {
        if self.room_cavity_ratio > 0.0 {
            return self.room_cavity_ratio;
        }
        let ceiling_height = self.window_height + self.workplane_height;
        if self.zone_area <= 0.0 || self.window_area <= 0.0 || ceiling_height <= 0.0 {
            return 2.5; // mid-range default
        }
        (5.0 * self.zone_area * self.workplane_height) / (self.window_area * ceiling_height)
    }

    /// Flux transfer factor as a function of room cavity ratio.
    ///
    /// Empirical fit: FTF = 1 / (1 + 0.5 * RCR^0.7).
    /// This captures the attenuation of daylight with depth.
    fn flux_transfer_factor(&self) -> f64 {
        let rcr = self.compute_rcr();
        1.0 / (1.0 + 0.5 * rcr.powf(0.7))
    }

    /// Split-flux interior illuminance model.
    ///
    /// Replaces the simple daylight-factor shortcut (SOLAR-03 / Phase 2).
    /// Uses window visible transmittance, sky illuminance, and room geometry
    /// to produce a reference-point illuminance.
    ///
    /// `sky_illuminance` — horizontal exterior illuminance (lux).
    /// `sky_condition` — clearness factor (0 = overcast … 1 = clear).
    pub fn interior_illuminance(&self, sky_illuminance: f64, sky_condition: f64) -> f64 {
        if let Some(vt) = self.visible_transmittance {
            // Split-flux model: beam + diffuse contributions.
            //
            // Beam contribution: depends on direct sun on window and interior
            // flux transfer to the workplane.
            let beam_fraction = sky_condition; // clear → more beam
            let diffuse_fraction = 1.0 - beam_fraction * 0.5;

            let ftf = self.flux_transfer_factor();

            // Vertical illuminance on the window from sky hemisphere.
            let vertical_beam = sky_illuminance * beam_fraction * 0.6; // ~cos-weighted
            let vertical_diffuse = sky_illuminance * diffuse_fraction * 0.35; // CIE overcast ≈0.35

            // Interior illuminance at workplane.
            let interior =
                (vertical_beam + vertical_diffuse) * vt * ftf * (self.window_area / self.zone_area);

            interior.max(0.0)
        } else {
            // Legacy daylight-factor path (backward compatible).
            sky_illuminance * (self.daylight_factor / 100.0) * sky_condition
        }
    }

    /// Calculate dimming level based on illuminance.
    /// Returns fraction (0-1) of maximum lighting output.
    pub fn dimming_level(&self, interior_illuminance: f64) -> f64 {
        if interior_illuminance >= self.dimming_threshold {
            self.min_dimming_level
        } else {
            let fraction = interior_illuminance / self.dimming_threshold;
            self.min_dimming_level + fraction * (1.0 - self.min_dimming_level)
        }
    }

    /// Calculate energy savings from daylighting.
    /// Returns savings in kWh.
    pub fn annual_energy_savings(
        &self,
        baseline_power: f64,
        hours_per_day: f64,
        days_per_year: f64,
        average_illuminance: f64,
    ) -> f64 {
        let dimming = self.dimming_level(average_illuminance);
        let energy_reduction = 1.0 - dimming;
        baseline_power * hours_per_day * days_per_year * energy_reduction / 1000.0
    }
}

/// Optical properties of a shading device.
///
/// When present, `ShadingControl::shgc_reduction()` uses these instead of
/// the legacy hard-coded factors per `ShadingType`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ShadeOpticalProperties {
    /// Solar transmittance of the shade fabric (0–1).
    pub solar_transmittance: f64,
    /// Visible transmittance of the shade fabric (0–1).
    pub visible_transmittance: f64,
    /// Solar absorptance of the shade fabric (0–1).
    pub solar_absorptance: f64,
    /// Solar reflectance of the shade fabric (0–1).
    pub solar_reflectance: f64,
    /// Whether the shade is exterior-mounted (true) or interior (false).
    pub is_exterior: bool,
}

impl ShadeOpticalProperties {
    /// Medium-tint roller shade (interior).
    pub fn interior_roller_tint() -> Self {
        Self {
            solar_transmittance: 0.20,
            visible_transmittance: 0.25,
            solar_absorptance: 0.55,
            solar_reflectance: 0.25,
            is_exterior: false,
        }
    }

    /// Exterior venetian blinds (aluminum).
    pub fn exterior_venetian() -> Self {
        Self {
            solar_transmittance: 0.05,
            visible_transmittance: 0.10,
            solar_absorptance: 0.45,
            solar_reflectance: 0.50,
            is_exterior: true,
        }
    }

    /// Interior fabric shade.
    pub fn interior_fabric() -> Self {
        Self {
            solar_transmittance: 0.30,
            visible_transmittance: 0.35,
            solar_absorptance: 0.50,
            solar_reflectance: 0.20,
            is_exterior: false,
        }
    }
}

/// Represents an automated shading system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadingControl {
    /// Shading device type
    pub shading_type: ShadingType,
    /// Position (0 = fully open, 1 = fully closed)
    pub position: f64,
    /// Solar irradiance threshold to deploy shading (W/m²)
    pub deployment_threshold: f64,
    /// Minimum outdoor temperature to allow shading (°C)
    pub min_temp_deployment: f64,
    /// Whether shading is currently deployed
    pub is_deployed: bool,
    /// Shade optical properties (when set, overrides hard-coded shgc_reduction).
    pub shade_optics: Option<ShadeOpticalProperties>,
}

/// Types of shading devices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShadingType {
    /// Interior blinds
    InteriorBlinds,
    /// Exterior blinds
    ExteriorBlinds,
    /// Roller shades
    RollerShades,
    /// Light shelves
    LightShelves,
}

impl ShadingControl {
    /// Create a new shading control
    pub fn new(shading_type: ShadingType) -> Self {
        Self {
            shading_type,
            position: 0.0,
            deployment_threshold: 300.0,
            min_temp_deployment: 15.0,
            is_deployed: false,
            shade_optics: None,
        }
    }

    /// Create a shading control with explicit optical properties.
    pub fn with_optics(shading_type: ShadingType, optics: ShadeOpticalProperties) -> Self {
        Self {
            shading_type,
            position: 0.0,
            deployment_threshold: 300.0,
            min_temp_deployment: 15.0,
            is_deployed: false,
            shade_optics: Some(optics),
        }
    }

    /// Determine shading deployment based on conditions.
    pub fn update(&mut self, solar_irradiance: f64, outdoor_temp: f64) {
        if solar_irradiance > self.deployment_threshold && outdoor_temp > self.min_temp_deployment {
            self.is_deployed = true;
            self.position = 1.0;
        } else {
            self.is_deployed = false;
            self.position = 0.0;
        }
    }

    /// Calculate solar heat gain coefficient reduction from shading.
    ///
    /// When `shade_optics` is set, the reduction is derived from the
    /// transmittance and absorptance of the shade material, replacing the
    /// legacy hard-coded factors.
    pub fn shgc_reduction(&self) -> f64 {
        if !self.is_deployed {
            return 0.0;
        }

        if let Some(optics) = self.shade_optics {
            // Exterior shade: blocks incident radiation before it reaches the glass,
            // so the SHGC reduction equals the incident radiation intercepted.
            // Interior shade: solar gain already passed through glass; the shade
            // absorbs and reflects a fraction of the interior-side irradiance.
            let intercept_fraction = if optics.is_exterior {
                1.0 - optics.solar_transmittance - optics.solar_absorptance
            } else {
                // Interior shade: the glass SHGC already accounts for window losses;
                // shade reduces the portion that reaches the room.
                optics.solar_absorptance + optics.solar_reflectance
            };

            return intercept_fraction * self.position;
        }

        // Legacy hard-coded factors.
        match self.shading_type {
            ShadingType::InteriorBlinds => 0.3 * self.position,
            ShadingType::ExteriorBlinds => 0.6 * self.position,
            ShadingType::RollerShades => 0.5 * self.position,
            ShadingType::LightShelves => 0.2 * self.position,
        }
    }

    /// Visible transmittance reduction factor from shading (0 = fully blocked, 1 = no shade).
    pub fn visible_transmittance_factor(&self) -> f64 {
        if !self.is_deployed {
            return 1.0;
        }

        if let Some(optics) = self.shade_optics {
            return 1.0 - (optics.visible_transmittance * self.position);
        }

        // Legacy: use same factors as shgc as a proxy.
        1.0 - self.shgc_reduction()
    }
}

/// Represents an artificial lighting schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightingSchedule {
    /// Hourly lighting schedule (0-23), values 0-1
    pub hourly_schedule: [f64; 24],
    /// Lighting power density (W/m²)
    pub power_density: f64,
    /// Zone area this schedule applies to (m²)
    pub zone_area: f64,
    /// Fraction of lighting heat that is convective (0-1)
    pub convective_fraction: f64,
    /// Fraction of lighting heat that is radiative (0-1)
    pub radiative_fraction: f64,
}

impl LightingSchedule {
    /// Create a new lighting schedule with default (off) values
    pub fn new(power_density: f64, zone_area: f64) -> Self {
        Self {
            hourly_schedule: [0.0; 24],
            power_density,
            zone_area,
            convective_fraction: 0.2,
            radiative_fraction: 0.8,
        }
    }

    /// Create an office lighting schedule (8am - 6pm)
    pub fn office_schedule(power_density: f64, zone_area: f64) -> Self {
        let mut schedule = Self::new(power_density, zone_area);
        for hour in 8..=17 {
            schedule.hourly_schedule[hour] = 1.0;
        }
        schedule
    }

    /// Create a retail lighting schedule
    pub fn retail_schedule(power_density: f64, zone_area: f64) -> Self {
        let mut schedule = Self::new(power_density, zone_area);
        for hour in 9..=20 {
            schedule.hourly_schedule[hour] = 1.0;
        }
        schedule
    }

    /// Get lighting power for a specific hour
    pub fn lighting_power(&self, hour: usize) -> f64 {
        let h = hour % 24;
        self.power_density * self.zone_area * self.hourly_schedule[h]
    }

    /// Get convective heat gains from lighting for a specific hour
    pub fn convective_heat_gains(&self, hour: usize) -> f64 {
        self.lighting_power(hour) * self.convective_fraction
    }

    /// Get radiative heat gains from lighting for a specific hour
    pub fn radiative_heat_gains(&self, hour: usize) -> f64 {
        self.lighting_power(hour) * self.radiative_fraction
    }

    /// Calculate annual lighting energy consumption (kWh)
    pub fn annual_energy(&self, operating_days: usize) -> f64 {
        let daily_energy: f64 = (0..24).map(|h| self.lighting_power(h)).sum::<f64>() / 1000.0;
        daily_energy * operating_days as f64
    }
}

/// Combined lighting system with controls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightingSystem {
    /// Artificial lighting schedule
    pub schedule: LightingSchedule,
    /// Daylight zones
    pub daylight_zones: Vec<DaylightZone>,
    /// Shading controls per orientation
    pub shading_controls: Vec<ShadingControl>,
    /// Control type
    pub control_type: LightingControlType,
}

/// Timestep-level lighting result — couples dimmed lighting into the zone energy balance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightingTimestepResult {
    /// Effective lighting power after daylighting dimming (W).
    pub effective_power_w: f64,
    /// Convective heat gain from lighting (W).
    pub convective_gain_w: f64,
    /// Radiative heat gain from lighting (W).
    pub radiative_gain_w: f64,
    /// Fraction of baseline power consumed (0–1).
    pub dimming_fraction: f64,
    /// Average workplane illuminance across daylight zones (lux).
    pub average_illuminance_lux: f64,
}

impl LightingSystem {
    /// Create a new lighting system
    pub fn new(power_density: f64, zone_area: f64) -> Self {
        Self {
            schedule: LightingSchedule::office_schedule(power_density, zone_area),
            daylight_zones: Vec::new(),
            shading_controls: Vec::new(),
            control_type: LightingControlType::ContinuousDimming,
        }
    }

    /// Add a daylight zone
    pub fn add_daylight_zone(&mut self, zone: DaylightZone) {
        self.daylight_zones.push(zone);
    }

    /// Add shading control for an orientation
    pub fn add_shading(&mut self, shading: ShadingControl) {
        self.shading_controls.push(shading);
    }

    /// Calculate effective lighting power with controls.
    ///
    /// Uses the split-flux illuminance model when daylight zones have
    /// `visible_transmittance` set, falling back to the legacy daylight factor.
    ///
    /// # Arguments
    /// * `hour` - Hour of day (0-23)
    /// * `exterior_illuminance` - Exterior horizontal illuminance (lux)
    /// * `sky_condition` - Sky factor (0-1)
    pub fn effective_lighting_power(
        &self,
        hour: usize,
        exterior_illuminance: f64,
        sky_condition: f64,
    ) -> f64 {
        let base_power = self.schedule.lighting_power(hour);

        if self.control_type == LightingControlType::ContinuousDimming
            && !self.daylight_zones.is_empty()
        {
            let avg_dimming: f64 = self
                .daylight_zones
                .iter()
                .map(|dz| {
                    dz.dimming_level(dz.interior_illuminance(exterior_illuminance, sky_condition))
                })
                .sum::<f64>()
                / self.daylight_zones.len() as f64;

            return base_power * avg_dimming;
        }

        base_power
    }

    /// Compute timestep-level lighting result with heat-gain coupling.
    ///
    /// This is the primary interface for the thermal model solver: it returns
    /// dimmed convective and radiative gains that should be added to the zone
    /// energy balance on each timestep (Phase 2 acceptance criterion 3).
    ///
    /// The shading controls' `visible_transmittance_factor` is applied to the
    /// exterior illuminance before the dimming calculation, coupling blind
    /// deployment with daylight availability.
    pub fn timestep_result(
        &self,
        hour: usize,
        exterior_illuminance: f64,
        sky_condition: f64,
    ) -> LightingTimestepResult {
        let base_power = self.schedule.lighting_power(hour);

        // Apply shading reductions to exterior illuminance.
        let shading_vt_factor: f64 = if self.shading_controls.is_empty() {
            1.0
        } else {
            self.shading_controls
                .iter()
                .map(|s| s.visible_transmittance_factor())
                .sum::<f64>()
                / self.shading_controls.len() as f64
        };

        let effective_exterior_ill = exterior_illuminance * shading_vt_factor;

        if self.control_type == LightingControlType::ContinuousDimming
            && !self.daylight_zones.is_empty()
        {
            let mut total_dimming = 0.0;
            let mut total_illuminance = 0.0;

            for dz in &self.daylight_zones {
                let illum = dz.interior_illuminance(effective_exterior_ill, sky_condition);
                total_illuminance += illum;
                total_dimming += dz.dimming_level(illum);
            }

            let avg_dimming = total_dimming / self.daylight_zones.len() as f64;
            let avg_illuminance = total_illuminance / self.daylight_zones.len() as f64;

            let effective_power = base_power * avg_dimming;
            let convective = effective_power * self.schedule.convective_fraction;
            let radiative = effective_power * self.schedule.radiative_fraction;

            return LightingTimestepResult {
                effective_power_w: effective_power,
                convective_gain_w: convective,
                radiative_gain_w: radiative,
                dimming_fraction: avg_dimming,
                average_illuminance_lux: avg_illuminance,
            };
        }

        let convective = base_power * self.schedule.convective_fraction;
        let radiative = base_power * self.schedule.radiative_fraction;

        LightingTimestepResult {
            effective_power_w: base_power,
            convective_gain_w: convective,
            radiative_gain_w: radiative,
            dimming_fraction: 1.0,
            average_illuminance_lux: 0.0,
        }
    }

    /// Simulate a full year (8760 hourly timesteps) and return total energy (kWh).
    ///
    /// Uses per-hour exterior illuminance profile and sky condition profile.
    pub fn annual_energy_kwh(
        &self,
        exterior_illuminance_profile: &[f64; 8760],
        sky_condition_profile: &[f64; 8760],
    ) -> f64 {
        let mut total_wh = 0.0;
        for hour in 0..8760 {
            let h = hour % 24;
            let result = self.timestep_result(
                h,
                exterior_illuminance_profile[hour],
                sky_condition_profile[hour],
            );
            total_wh += result.effective_power_w;
        }
        total_wh / 1000.0
    }

    /// Compute annual lighting energy without daylighting controls (baseline).
    pub fn baseline_annual_energy_kwh(&self) -> f64 {
        self.schedule.annual_energy(250)
    }

    /// Compute annual lighting energy with daylighting controls.
    pub fn controlled_annual_energy_kwh(
        &self,
        exterior_illuminance_profile: &[f64; 8760],
        sky_condition_profile: &[f64; 8760],
    ) -> f64 {
        self.annual_energy_kwh(exterior_illuminance_profile, sky_condition_profile)
    }
}

// ---------------------------------------------------------------------------
// Shading schedule and blind control (Phase 3 — window management)
// ---------------------------------------------------------------------------

/// Time-of-day shading schedule for automated blind/shade control.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadingSchedule {
    /// Hourly position targets (0–1) for 24 hours.
    pub hourly_position: [f64; 24],
    /// If true, override hourly schedule with solar-irradiance-based deployment.
    pub use_solar_override: bool,
    /// Solar threshold (W/m²) for override deployment.
    pub solar_threshold: f64,
}

impl ShadingSchedule {
    /// Create an always-retracted schedule (no shading).
    pub fn always_open() -> Self {
        Self {
            hourly_position: [0.0; 24],
            use_solar_override: false,
            solar_threshold: 300.0,
        }
    }

    /// Create an always-deployed schedule (full shading).
    pub fn always_closed() -> Self {
        Self {
            hourly_position: [1.0; 24],
            use_solar_override: false,
            solar_threshold: 300.0,
        }
    }

    /// Create a business-hours shading schedule (deployed during 9am–5pm).
    pub fn business_hours() -> Self {
        let mut schedule = Self::always_open();
        for h in 9..17 {
            schedule.hourly_position[h] = 1.0;
        }
        schedule
    }

    /// Get the scheduled position for a given hour.
    pub fn position_at_hour(&self, hour: usize, solar_irradiance: f64) -> f64 {
        let h = hour % 24;
        if self.use_solar_override && solar_irradiance > self.solar_threshold {
            1.0
        } else {
            self.hourly_position[h]
        }
    }
}

/// Blind slat angle control for venetian blinds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlindControl {
    /// Current slat angle in degrees (0 = horizontal/open, 90 = fully closed).
    pub slat_angle_deg: f64,
    /// Target slat angle (set by control algorithm).
    pub target_angle_deg: f64,
    /// Solar altitude threshold (deg) below which slats open to admit daylight.
    pub min_altitude_threshold_deg: f64,
    /// Solar altitude threshold (deg) above which slats close to block glare.
    pub max_altitude_threshold_deg: f64,
    /// Maximum rate of slat angle change per minute (deg/min).
    pub slew_rate_deg_per_min: f64,
}

impl Default for BlindControl {
    fn default() -> Self {
        Self::new()
    }
}

impl BlindControl {
    /// Create a new blind controller with default settings.
    pub fn new() -> Self {
        Self {
            slat_angle_deg: 0.0,
            target_angle_deg: 0.0,
            min_altitude_threshold_deg: 15.0,
            max_altitude_threshold_deg: 60.0,
            slew_rate_deg_per_min: 2.0,
        }
    }

    /// Update slat angle based on solar altitude.
    ///
    /// - Below `min_altitude_threshold_deg`: slats fully open (0°).
    /// - Above `max_altitude_threshold_deg`: slats fully closed (90°).
    /// - Between: linear interpolation.
    pub fn update_for_altitude(&mut self, solar_altitude_deg: f64) {
        self.target_angle_deg = if solar_altitude_deg <= self.min_altitude_threshold_deg {
            0.0
        } else if solar_altitude_deg >= self.max_altitude_threshold_deg {
            90.0
        } else {
            let t = (solar_altitude_deg - self.min_altitude_threshold_deg)
                / (self.max_altitude_threshold_deg - self.min_altitude_threshold_deg);
            t * 90.0
        };
    }

    /// Advance slat angle toward the target, respecting slew rate.
    ///
    /// `dt_minutes` — timestep duration in minutes.
    pub fn advance(&mut self, dt_minutes: f64) {
        let max_change = self.slew_rate_deg_per_min * dt_minutes;
        let diff = self.target_angle_deg - self.slat_angle_deg;
        if diff.abs() <= max_change {
            self.slat_angle_deg = self.target_angle_deg;
        } else {
            self.slat_angle_deg += diff.signum() * max_change;
        }
    }

    /// Effective solar transmittance fraction based on slat angle.
    ///
    /// Simple model: transmittance = cos(slat_angle) when slats are horizontal,
    /// decreasing to near-zero at 90°.
    pub fn effective_transmittance_factor(&self) -> f64 {
        let angle_rad = self.slat_angle_deg.to_radians();
        angle_rad.cos().max(0.0)
    }
}

/// Daylighting control system — integrates sensors, dimming, blinds, and shading.
///
/// This is the top-level coordinator that a thermal model or HVAC controller
/// uses to apply daylight-responsive controls on each timestep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaylightingControls {
    /// Per-orientation shading schedules.
    pub shading_schedules: Vec<ShadingSchedule>,
    /// Per-orientation blind controllers.
    pub blind_controls: Vec<BlindControl>,
    /// Target illuminance at the workplane (lux). The dimming system tries
    /// to maintain this level by supplementing with artificial light.
    pub target_illuminance_lux: f64,
    /// Maximum lighting power density (W/m²) for the controlled zones.
    pub max_power_density: f64,
}

impl DaylightingControls {
    /// Create a new daylighting control system.
    pub fn new(target_illuminance_lux: f64, max_power_density: f64) -> Self {
        Self {
            shading_schedules: Vec::new(),
            blind_controls: Vec::new(),
            target_illuminance_lux,
            max_power_density,
        }
    }

    /// Add a shading schedule for an orientation.
    pub fn add_shading_schedule(&mut self, schedule: ShadingSchedule) {
        self.shading_schedules.push(schedule);
    }

    /// Add a blind controller for an orientation.
    pub fn add_blind_controller(&mut self, controller: BlindControl) {
        self.blind_controls.push(controller);
    }

    /// Compute the net shading position for a given hour.
    ///
    /// Averages the scheduled positions across all orientations.
    pub fn net_shading_position(&self, hour: usize, solar_irradiance: f64) -> f64 {
        if self.shading_schedules.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .shading_schedules
            .iter()
            .map(|s| s.position_at_hour(hour, solar_irradiance))
            .sum();
        sum / self.shading_schedules.len() as f64
    }

    /// Compute average blind transmittance factor across all orientations.
    pub fn average_blind_transmittance(&self) -> f64 {
        if self.blind_controls.is_empty() {
            return 1.0;
        }
        let sum: f64 = self
            .blind_controls
            .iter()
            .map(|b| b.effective_transmittance_factor())
            .sum();
        sum / self.blind_controls.len() as f64
    }

    /// Update all blind controllers for a given solar altitude.
    pub fn update_blinds_for_altitude(&mut self, solar_altitude_deg: f64) {
        for bc in &mut self.blind_controls {
            bc.update_for_altitude(solar_altitude_deg);
        }
    }

    /// Advance all blind controllers by `dt_minutes`.
    pub fn advance_blinds(&mut self, dt_minutes: f64) {
        for bc in &mut self.blind_controls {
            bc.advance(dt_minutes);
        }
    }

    /// Compute the effective illuminance reaching the workplane after
    /// shading and blind attenuation.
    pub fn effective_exterior_illuminance(
        &self,
        exterior_illuminance: f64,
        hour: usize,
        solar_irradiance: f64,
    ) -> f64 {
        let shade_factor = self.net_shading_position(hour, solar_irradiance);
        let blind_factor = self.average_blind_transmittance();
        // shade_factor is position (0=open, 1=closed) — attenuation = 1 - shade_factor * absorption
        let attenuation = 1.0 - shade_factor * 0.5; // 50% max attenuation from shading
        exterior_illuminance * attenuation * blind_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daylight_zone() {
        let zone = DaylightZone::new("DZ-1".to_string(), 0, 10.0, 2.0);

        let illuminance = zone.interior_illuminance(10000.0, 0.8);
        assert!(illuminance > 0.0);

        let dimming = zone.dimming_level(500.0);
        assert!((0.1..=1.0).contains(&dimming));
    }

    #[test]
    fn test_shading_control() {
        let mut shading = ShadingControl::new(ShadingType::InteriorBlinds);

        shading.update(500.0, 25.0);
        assert!(shading.is_deployed);

        shading.update(100.0, 25.0);
        assert!(!shading.is_deployed);
    }

    #[test]
    fn test_lighting_schedule() {
        let schedule = LightingSchedule::office_schedule(10.0, 100.0);

        assert!(schedule.lighting_power(10) > 0.0);

        assert_eq!(schedule.lighting_power(2), 0.0);
    }

    #[test]
    fn test_lighting_heat_gains() {
        let schedule = LightingSchedule::office_schedule(10.0, 100.0);

        assert_eq!(schedule.convective_fraction, 0.2);
        assert_eq!(schedule.radiative_fraction, 0.8);

        let hour = 10;
        let total_power = schedule.lighting_power(hour);
        let convective = schedule.convective_heat_gains(hour);
        let radiative = schedule.radiative_heat_gains(hour);

        assert_eq!(convective, total_power * 0.2);
        assert_eq!(radiative, total_power * 0.8);
        assert!((total_power - (convective + radiative)).abs() < 1e-10);
    }

    #[test]
    fn test_lighting_system() {
        let mut system = LightingSystem::new(10.0, 100.0);
        let mut dz = DaylightZone::new("DZ-1".to_string(), 0, 10.0, 2.0);
        dz.dimming_threshold = 500.0;
        system.add_daylight_zone(dz);
        let power = system.effective_lighting_power(12, 10000.0, 0.8);
        assert!(power < 1000.0);
    }

    #[test]
    fn test_daylight_zone_default_values() {
        let zone = DaylightZone::new("test".to_string(), 1, 15.0, 3.0);
        assert_eq!(zone.daylight_zone_depth, 4.5);
        assert_eq!(zone.daylight_factor, 5.0);
        assert_eq!(zone.dimming_threshold, 300.0);
        assert_eq!(zone.min_dimming_level, 0.1);
    }

    #[test]
    fn test_daylight_zone_interior_illuminance_clear_sky() {
        let zone = DaylightZone::new("test".to_string(), 0, 10.0, 2.0);
        assert_eq!(zone.interior_illuminance(50000.0, 1.0), 2500.0);
    }

    #[test]
    fn test_daylight_zone_interior_illuminance_overcast() {
        let zone = DaylightZone::new("test".to_string(), 0, 10.0, 2.0);
        assert_eq!(zone.interior_illuminance(20000.0, 0.5), 500.0);
    }

    #[test]
    fn test_daylight_zone_dimming_above_threshold() {
        let zone = DaylightZone::new("test".to_string(), 0, 10.0, 2.0);
        assert_eq!(zone.dimming_level(500.0), zone.min_dimming_level);
    }

    #[test]
    fn test_daylight_zone_dimming_below_threshold() {
        let zone = DaylightZone::new("test".to_string(), 0, 10.0, 2.0);
        let dimming = zone.dimming_level(150.0);
        assert!(dimming > zone.min_dimming_level && dimming < 1.0);
    }

    #[test]
    fn test_daylight_zone_dimming_zero_illuminance() {
        let zone = DaylightZone::new("test".to_string(), 0, 10.0, 2.0);
        assert_eq!(zone.dimming_level(0.0), zone.min_dimming_level);
    }

    #[test]
    fn test_daylight_zone_annual_energy_savings() {
        let zone = DaylightZone::new("test".to_string(), 0, 10.0, 2.0);
        assert!(zone.annual_energy_savings(1000.0, 10.0, 250.0, 600.0) > 0.0);
    }

    #[test]
    fn test_shading_control_default_values() {
        let shading = ShadingControl::new(ShadingType::ExteriorBlinds);
        assert_eq!(shading.position, 0.0);
        assert_eq!(shading.deployment_threshold, 300.0);
        assert_eq!(shading.min_temp_deployment, 15.0);
        assert!(!shading.is_deployed);
    }

    #[test]
    fn test_shading_update_deploy() {
        let mut shading = ShadingControl::new(ShadingType::InteriorBlinds);
        shading.update(400.0, 25.0);
        assert!(shading.is_deployed);
        assert_eq!(shading.position, 1.0);
    }

    #[test]
    fn test_shading_update_retract_low_solar() {
        let mut shading = ShadingControl::new(ShadingType::InteriorBlinds);
        shading.is_deployed = true;
        shading.update(100.0, 25.0);
        assert!(!shading.is_deployed);
    }

    #[test]
    fn test_shading_update_retract_low_temp() {
        let mut shading = ShadingControl::new(ShadingType::InteriorBlinds);
        shading.is_deployed = true;
        shading.update(400.0, 10.0);
        assert!(!shading.is_deployed);
    }

    #[test]
    fn test_shading_shgc_reduction_not_deployed() {
        assert_eq!(
            ShadingControl::new(ShadingType::InteriorBlinds).shgc_reduction(),
            0.0
        );
    }

    #[test]
    fn test_shading_shgc_reduction_interior_blinds() {
        let mut s = ShadingControl::new(ShadingType::InteriorBlinds);
        s.is_deployed = true;
        s.position = 1.0;
        assert_eq!(s.shgc_reduction(), 0.3);
    }

    #[test]
    fn test_shading_shgc_reduction_exterior_blinds() {
        let mut s = ShadingControl::new(ShadingType::ExteriorBlinds);
        s.is_deployed = true;
        s.position = 1.0;
        assert_eq!(s.shgc_reduction(), 0.6);
    }

    #[test]
    fn test_shading_shgc_reduction_roller_shades() {
        let mut s = ShadingControl::new(ShadingType::RollerShades);
        s.is_deployed = true;
        s.position = 1.0;
        assert_eq!(s.shgc_reduction(), 0.5);
    }

    #[test]
    fn test_shading_shgc_reduction_light_shelves() {
        let mut s = ShadingControl::new(ShadingType::LightShelves);
        s.is_deployed = true;
        s.position = 1.0;
        assert_eq!(s.shgc_reduction(), 0.2);
    }

    #[test]
    fn test_shading_shgc_reduction_partial_position() {
        let mut s = ShadingControl::new(ShadingType::ExteriorBlinds);
        s.is_deployed = true;
        s.position = 0.5;
        assert_eq!(s.shgc_reduction(), 0.3);
    }

    #[test]
    fn test_lighting_schedule_default() {
        let schedule = LightingSchedule::new(10.0, 100.0);
        assert_eq!(schedule.power_density, 10.0);
        assert_eq!(schedule.convective_fraction, 0.2);
        assert_eq!(schedule.radiative_fraction, 0.8);
        assert!(schedule.hourly_schedule.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_lighting_schedule_office_hours() {
        let schedule = LightingSchedule::office_schedule(10.0, 100.0);
        assert_eq!(schedule.lighting_power(8), 1000.0);
        assert_eq!(schedule.lighting_power(12), 1000.0);
        assert_eq!(schedule.lighting_power(17), 1000.0);
        assert_eq!(schedule.lighting_power(7), 0.0);
        assert_eq!(schedule.lighting_power(18), 0.0);
    }

    #[test]
    fn test_lighting_schedule_retail_hours() {
        let schedule = LightingSchedule::retail_schedule(10.0, 100.0);
        assert_eq!(schedule.lighting_power(9), 1000.0);
        assert_eq!(schedule.lighting_power(15), 1000.0);
        assert_eq!(schedule.lighting_power(20), 1000.0);
        assert_eq!(schedule.lighting_power(8), 0.0);
        assert_eq!(schedule.lighting_power(21), 0.0);
    }

    #[test]
    fn test_lighting_schedule_hour_wraparound() {
        let schedule = LightingSchedule::office_schedule(10.0, 100.0);
        assert_eq!(schedule.lighting_power(24), 0.0);
        assert_eq!(schedule.lighting_power(30), schedule.lighting_power(6));
    }

    #[test]
    fn test_lighting_schedule_annual_energy() {
        let schedule = LightingSchedule::office_schedule(10.0, 100.0);
        assert!((schedule.annual_energy(250) - 2500.0).abs() < 1.0);
    }

    #[test]
    fn test_lighting_schedule_annual_energy_zero_days() {
        assert_eq!(
            LightingSchedule::office_schedule(10.0, 100.0).annual_energy(0),
            0.0
        );
    }

    #[test]
    fn test_lighting_system_no_daylight_zones() {
        let system = LightingSystem::new(10.0, 100.0);
        assert_eq!(system.effective_lighting_power(12, 10000.0, 0.8), 1000.0);
    }

    #[test]
    fn test_lighting_system_multiple_zones() {
        let mut system = LightingSystem::new(10.0, 100.0);
        system.add_daylight_zone(DaylightZone::new("DZ-1".to_string(), 0, 10.0, 2.0));
        system.add_daylight_zone(DaylightZone::new("DZ-2".to_string(), 0, 15.0, 3.0));
        let power = system.effective_lighting_power(12, 10000.0, 0.8);
        assert!(power > 0.0 && power < 1000.0);
    }

    #[test]
    fn test_lighting_control_type_equality() {
        assert_eq!(LightingControlType::Manual, LightingControlType::Manual);
        assert_ne!(
            LightingControlType::Manual,
            LightingControlType::ContinuousDimming
        );
    }

    #[test]
    fn test_shading_type_equality() {
        assert_eq!(ShadingType::InteriorBlinds, ShadingType::InteriorBlinds);
        assert_ne!(ShadingType::InteriorBlinds, ShadingType::ExteriorBlinds);
    }

    // ─── WindowOpticalProperties ───────────────────────────────────────────────

    #[test]
    fn test_window_optical_properties_single_layer() {
        // Single clear glass: solar_transmittance=0.837, layer_absorptance=0.095
        // System absorptance = layer_absorptance * ts = 0.095 * 0.837 = 0.0795
        let single = WindowOpticalProperties::from_layers(&[WindowLayer::clear_glass()]);
        assert_eq!(single.solar_transmittance, 0.837);
        assert!((single.solar_absorptance - 0.0795).abs() < 0.001);
        // Reflectance = 1 - ts - absorptance = 1 - 0.837 - 0.0795 = 0.0835
        assert!((single.solar_reflectance - 0.0835).abs() < 0.001);
    }

    #[test]
    fn test_window_optical_properties_double_clear() {
        let optics = WindowOpticalProperties::double_clear();
        // Two clear glasses: ts=0.837^2≈0.7006
        assert_eq!(optics.solar_transmittance, 0.837_f64 * 0.837);
        assert!(optics.solar_absorptance > 0.0);
        assert!(optics.solar_reflectance > 0.0);
    }

    #[test]
    fn test_window_optical_properties_three_layers() {
        let layers = [
            WindowLayer::clear_glass(),
            WindowLayer::clear_glass(),
            WindowLayer::clear_glass(),
        ];
        let optics = WindowOpticalProperties::from_layers(&layers);
        assert_eq!(optics.solar_transmittance, 0.837_f64.powi(3));
        assert!(
            optics.solar_transmittance
                < WindowOpticalProperties::double_clear().solar_transmittance
        );
    }

    // ─── ShadeOpticalProperties ────────────────────────────────────────────────

    #[test]
    fn test_shade_optical_properties_interior_fabric() {
        let fabric = ShadeOpticalProperties::interior_fabric();
        assert_eq!(fabric.solar_transmittance, 0.30);
        assert_eq!(fabric.visible_transmittance, 0.35);
        assert_eq!(fabric.solar_absorptance, 0.50);
        assert_eq!(fabric.solar_reflectance, 0.20);
        assert!(!fabric.is_exterior);
    }

    #[test]
    fn test_shade_optical_properties_exterior_venetian() {
        let blinds = ShadeOpticalProperties::exterior_venetian();
        assert_eq!(blinds.solar_transmittance, 0.05);
        assert_eq!(blinds.visible_transmittance, 0.10);
        assert!(blinds.is_exterior);
    }

    // ─── ShadingControl with optics ──────────────────────────────────────────

    #[test]
    fn test_shading_control_with_optics() {
        let optics = ShadeOpticalProperties::interior_roller_tint();
        let shading = ShadingControl::with_optics(ShadingType::RollerShades, optics);
        assert!(shading.shade_optics.is_some());
        assert_eq!(shading.position, 0.0);
        assert!(!shading.is_deployed);
    }

    #[test]
    fn test_shading_control_shgc_reduction_exterior_shade_optics() {
        let optics = ShadeOpticalProperties::exterior_venetian();
        let mut s = ShadingControl::with_optics(ShadingType::ExteriorBlinds, optics);
        s.is_deployed = true;
        s.position = 1.0;
        // Exterior: intercept = 1 - ts - absorptance = 1 - 0.05 - 0.45 = 0.50
        assert!((s.shgc_reduction() - 0.50).abs() < 1e-9);
    }

    #[test]
    fn test_shading_control_shgc_reduction_interior_shade_optics() {
        let optics = ShadeOpticalProperties::interior_fabric();
        let mut s = ShadingControl::with_optics(ShadingType::RollerShades, optics);
        s.is_deployed = true;
        s.position = 1.0;
        // Interior: intercept = absorptance + reflectance = 0.50 + 0.20 = 0.70
        assert_eq!(s.shgc_reduction(), 0.70);
    }

    #[test]
    fn test_shading_control_visible_transmittance_factor_deployed() {
        let optics = ShadeOpticalProperties::interior_roller_tint();
        let mut s = ShadingControl::with_optics(ShadingType::RollerShades, optics);
        s.is_deployed = true;
        s.position = 1.0;
        // factor = 1 - (0.25 * 1.0) = 0.75
        assert_eq!(s.visible_transmittance_factor(), 0.75);
    }

    #[test]
    fn test_shading_control_visible_transmittance_factor_partial_position() {
        let optics = ShadeOpticalProperties::interior_roller_tint();
        let mut s = ShadingControl::with_optics(ShadingType::RollerShades, optics);
        s.is_deployed = true;
        s.position = 0.5;
        // factor = 1 - (0.25 * 0.5) = 0.875
        assert_eq!(s.visible_transmittance_factor(), 0.875);
    }

    // ─── DaylightZone with split-flux model ───────────────────────────────────

    #[test]
    fn test_daylight_zone_with_visible_transmittance() {
        let zone = DaylightZone::new("test".to_string(), 0, 10.0, 2.0)
            .with_visible_transmittance(0.10)
            .with_zone_area(15.0);
        assert_eq!(zone.visible_transmittance, Some(0.10));
        assert_eq!(zone.zone_area, 15.0);
    }

    #[test]
    fn test_daylight_zone_split_flux_interior_illuminance() {
        let zone = DaylightZone::new("test".to_string(), 0, 10.0, 2.0)
            .with_visible_transmittance(0.10)
            .with_zone_area(15.0);
        let illum = zone.interior_illuminance(50000.0, 1.0);
        assert!(illum > 0.0);
        // Split-flux produces a positive value; verify it's in a reasonable range.
        // With sky_condition=1.0 (clear), vt=0.10, the interior illuminance
        // should be proportional to sky_illuminance * vt * FTF * geometry_factor.
        assert!(illum < 50000.0); // Cannot exceed exterior illuminance
    }

    #[test]
    fn test_daylight_zone_flux_transfer_factor() {
        let zone = DaylightZone::new("test".to_string(), 0, 10.0, 2.0);
        let ftf = zone.flux_transfer_factor();
        assert!(ftf > 0.0 && ftf <= 1.0);
        // RCR formula: (5 * zone_area * workplane_height) / (window_area * ceiling_height)
        // ceiling_height = window_height + workplane_height = 2.0 + 0.8 = 2.8
        // RCR = (5 * 15 * 0.8) / (10 * 2.8) = 60 / 28 ≈ 2.143
        // ftf = 1 / (1 + 0.5 * 2.143_f64.powf(0.7)) ≈ 1 / (1 + 0.5 * 1.73) ≈ 0.536
        assert!((ftf - 0.536).abs() < 0.01);
    }

    #[test]
    fn test_daylight_zone_compute_rcr_with_explicit_value() {
        let mut zone = DaylightZone::new("test".to_string(), 0, 10.0, 2.0);
        zone.room_cavity_ratio = 5.0;
        assert_eq!(zone.compute_rcr(), 5.0); // Explicit value returned as-is
    }

    #[test]
    fn test_daylight_zone_compute_rcr_zero_area_fallback() {
        let zone = DaylightZone::new("test".to_string(), 0, 0.0, 2.0); // zero window_area
        assert_eq!(zone.compute_rcr(), 2.5); // default fallback
    }

    // ─── LightingSchedule ────────────────────────────────────────────────────

    #[test]
    fn test_lighting_schedule_convective_radiative_sum() {
        let schedule = LightingSchedule::new(10.0, 100.0);
        assert!((schedule.convective_fraction + schedule.radiative_fraction - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lighting_schedule_convective_radiative_gains() {
        let schedule = LightingSchedule::office_schedule(10.0, 100.0);
        let hour = 12;
        let total = schedule.lighting_power(hour);
        let convective = schedule.convective_heat_gains(hour);
        let radiative = schedule.radiative_heat_gains(hour);
        assert!((total - convective - radiative).abs() < 1e-9);
    }

    // ─── LightingSystem timestep_result ──────────────────────────────────────

    #[test]
    fn test_lighting_system_timestep_result_no_controls() {
        let system = LightingSystem::new(10.0, 100.0);
        let result = system.timestep_result(12, 10000.0, 0.8);
        assert_eq!(result.effective_power_w, 1000.0);
        assert_eq!(result.dimming_fraction, 1.0);
    }

    #[test]
    fn test_lighting_system_timestep_result_with_daylight_zones() {
        let mut system = LightingSystem::new(10.0, 100.0);
        let mut dz = DaylightZone::new("DZ-1".to_string(), 0, 10.0, 2.0);
        dz.dimming_threshold = 500.0;
        system.add_daylight_zone(dz);
        let result = system.timestep_result(12, 50000.0, 0.8);
        assert!(result.effective_power_w < 1000.0);
        assert!(result.average_illuminance_lux > 0.0);
    }

    // ─── ShadingSchedule ──────────────────────────────────────────────────────

    #[test]
    fn test_shading_schedule_always_open() {
        let schedule = ShadingSchedule::always_open();
        assert!(schedule.hourly_position.iter().all(|&p| p == 0.0));
        assert_eq!(schedule.position_at_hour(12, 500.0), 0.0);
    }

    #[test]
    fn test_shading_schedule_always_closed() {
        let schedule = ShadingSchedule::always_closed();
        assert!(schedule.hourly_position.iter().all(|&p| p == 1.0));
        assert_eq!(schedule.position_at_hour(12, 500.0), 1.0);
    }

    #[test]
    fn test_shading_schedule_business_hours() {
        let schedule = ShadingSchedule::business_hours();
        assert_eq!(schedule.position_at_hour(9, 0.0), 1.0);
        assert_eq!(schedule.position_at_hour(12, 0.0), 1.0);
        assert_eq!(schedule.position_at_hour(16, 0.0), 1.0);
        assert_eq!(schedule.position_at_hour(8, 0.0), 0.0);
        assert_eq!(schedule.position_at_hour(17, 0.0), 0.0);
    }

    #[test]
    fn test_shading_schedule_solar_override() {
        let mut schedule = ShadingSchedule::always_open();
        schedule.use_solar_override = true;
        // Threshold is 300.0 W/m² (default from always_open())
        assert_eq!(schedule.position_at_hour(12, 200.0), 0.0); // below threshold
        assert_eq!(schedule.position_at_hour(12, 400.0), 1.0); // above threshold
    }

    // ─── BlindControl ────────────────────────────────────────────────────────

    #[test]
    fn test_blind_control_update_below_min_altitude() {
        let mut blind = BlindControl::new();
        blind.update_for_altitude(10.0); // below min threshold of 15°
        assert_eq!(blind.target_angle_deg, 0.0);
    }

    #[test]
    fn test_blind_control_update_above_max_altitude() {
        let mut blind = BlindControl::new();
        blind.update_for_altitude(70.0); // above max threshold of 60°
        assert_eq!(blind.target_angle_deg, 90.0);
    }

    #[test]
    fn test_blind_control_update_between_thresholds() {
        let mut blind = BlindControl::new();
        blind.update_for_altitude(37.5); // midpoint of 15° and 60°
                                         // t = (37.5 - 15) / (60 - 15) = 0.5, so target = 0.5 * 90 = 45°
        assert_eq!(blind.target_angle_deg, 45.0);
    }

    #[test]
    fn test_blind_control_advance_slew_rate_limiting() {
        let mut blind = BlindControl::new();
        blind.slew_rate_deg_per_min = 5.0;
        blind.target_angle_deg = 60.0;
        blind.advance(5.0); // 5 min * 5 deg/min = 25° max change
        assert_eq!(blind.slat_angle_deg, 25.0);
    }

    #[test]
    fn test_blind_control_advance_within_slew_rate() {
        let mut blind = BlindControl::new();
        blind.target_angle_deg = 10.0;
        blind.advance(60.0); // 60 min * 2 deg/min = 120° max; diff = 10°
        assert_eq!(blind.slat_angle_deg, 10.0); // Reaches target directly
    }

    #[test]
    fn test_blind_control_effective_transmittance_open() {
        let blind = BlindControl::new(); // slat_angle_deg = 0
        assert!((blind.effective_transmittance_factor() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_blind_control_effective_transmittance_closed() {
        let mut blind = BlindControl::new();
        blind.slat_angle_deg = 90.0;
        assert!((blind.effective_transmittance_factor() - 0.0).abs() < 1e-10);
    }

    // ─── DaylightingControls ─────────────────────────────────────────────────

    #[test]
    fn test_daylighting_controls_net_shading_position_empty() {
        let controls = DaylightingControls::new(500.0, 10.0);
        assert_eq!(controls.net_shading_position(12, 500.0), 0.0);
    }

    #[test]
    fn test_daylighting_controls_net_shading_position_single_schedule() {
        let mut controls = DaylightingControls::new(500.0, 10.0);
        controls.add_shading_schedule(ShadingSchedule::always_closed());
        assert_eq!(controls.net_shading_position(12, 500.0), 1.0);
    }

    #[test]
    fn test_daylighting_controls_effective_exterior_illuminance() {
        let mut controls = DaylightingControls::new(500.0, 10.0);
        let mut blind = BlindControl::new();
        blind.slat_angle_deg = 0.0; // fully open
        controls.add_blind_controller(blind);
        let illum = controls.effective_exterior_illuminance(10000.0, 12, 500.0);
        // With open blinds (transmittance factor = 1.0) and no shading (position = 0),
        // effective = 10000 * 1.0 * 1.0 = 10000
        assert_eq!(illum, 10000.0);
    }

    #[test]
    fn test_daylighting_controls_effective_exterior_illuminance_with_shading() {
        let mut controls = DaylightingControls::new(500.0, 10.0);
        let mut blind = BlindControl::new();
        blind.slat_angle_deg = 45.0; // partially closed
        controls.add_blind_controller(blind);
        let illum = controls.effective_exterior_illuminance(10000.0, 12, 500.0);
        // With cos(45°) transmittance ≈ 0.707 and position 0, attenuation = 1 - 0*0.5 = 1.0
        // effective = 10000 * 1.0 * 0.707 ≈ 7071
        assert!((illum - 7071.0).abs() < 1.0);
    }
}
