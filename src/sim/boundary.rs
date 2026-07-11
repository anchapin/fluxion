//! Ground boundary condition models for building energy simulation.
//!
//! This module provides traits and implementations for modeling ground temperatures
//! as thermal boundary conditions. Ground coupling is a critical component of building
//! heat transfer, affecting annual heating loads significantly.
//!
//! # ASHRAE 140 Specification
//!
//! Per ASHRAE Standard 140, the ground temperature is specified as a constant 10°C
//! for baseline test cases. This simplification is appropriate for annual simulations
//! where ground temperature varies minimally at typical foundation depths.
//!
//! # Models Provided
//!
//! - [`ConstantGroundTemperature`](struct@ConstantGroundTemperature): Fixed temperature model
//! - [`DynamicGroundTemperature`](struct@DynamicGroundTemperature): Time-varying model using Kusuda formula
//!
//! # Example
//!
//! ```rust
//! use fluxion::sim::boundary::{GroundTemperature, ConstantGroundTemperature};
//!
//! // Create constant ground temperature (ASHRAE 140 default)
//! let ground = ConstantGroundTemperature::new(10.0);
//! let temp = ground.ground_temperature(1000); // Hour 1000 of year
//! assert_eq!(temp, 10.0);
//! ```

use std::boxed::Box;
use std::f64::consts::PI;

/// Trait for ground temperature models in building energy simulation.
///
/// Implementors provide ground temperature at specific timesteps, enabling
/// different modeling approaches from simple constant temperatures to
/// sophisticated time-varying models based on soil physics.
///
/// # Requirements
///
/// Implementors must be [`Send`] + [`Sync`] to enable thread-safe parallel
/// simulation of building populations.
///
/// # Example Implementation
///
/// ```rust
/// use fluxion::sim::boundary::GroundTemperature;
///
/// struct SimpleGround {
///     temp: f64,
/// }
///
/// impl GroundTemperature for SimpleGround {
///     fn clone_box(&self) -> Box<dyn GroundTemperature> {
///         Box::new(SimpleGround { temp: self.temp })
///     }
///
///     fn ground_temperature(&self, _hour_of_year: usize) -> f64 {
///         self.temp
///     }
/// }
/// ```
pub trait GroundTemperature: Send + Sync {
    /// Clone this trait object into a new Box.
    ///
    /// This is needed for implementing Clone on ThermalModel.
    fn clone_box(&self) -> Box<dyn GroundTemperature>;

    /// Get ground temperature at a given hour of the year.
    ///
    /// # Arguments
    ///
    /// * `hour_of_year` - Hour index (0-8759) in the annual cycle
    ///
    /// # Returns
    ///
    /// Ground temperature in degrees Celsius.
    ///
    /// # Notes
    ///
    /// - Hour values wrap around using modulo for year-to-year continuity
    /// - Implementations may ignore `hour_of_year` for constant models
    fn ground_temperature(&self, hour_of_year: usize) -> f64;
}

/// Constant ground temperature model.
///
/// This is the simplest ground model, suitable for annual simulations where
/// ground temperature variation at foundation depth is minimal. Per ASHRAE 140
/// specification, the default value is 10°C.
///
/// # Advantages
///
/// - Simple and fast (no calculation needed)
/// - Matches ASHRAE 140 baseline specification
/// - Appropriate for well-insulated slabs at typical depths (>1m)
///
/// # Limitations
///
/// - Ignores seasonal ground temperature variation
/// - Not suitable for shallow foundations or crawl spaces
/// - May underpredict summer heat gain/loss in some climates
///
/// # Example
///
/// ```rust
/// use fluxion::sim::boundary::{GroundTemperature, ConstantGroundTemperature};
///
/// // ASHRAE 140 default
/// let ground = ConstantGroundTemperature::new(10.0);
/// assert_eq!(ground.ground_temperature(0), 10.0);
/// assert_eq!(ground.ground_temperature(4380), 10.0);
/// ```
#[derive(Debug, Clone)]
pub struct ConstantGroundTemperature {
    /// Constant soil temperature in degrees Celsius
    temperature: f64,
}

impl ConstantGroundTemperature {
    /// Create a new constant ground temperature model.
    ///
    /// # Arguments
    ///
    /// * `temperature` - Ground temperature in °C (typical range: 5-15°C for mid-latitude climates)
    ///
    /// # Example
    ///
    /// ```rust
    /// use fluxion::sim::boundary::ConstantGroundTemperature;
    ///
    /// // ASHRAE 140 specification
    /// let ground = ConstantGroundTemperature::new(10.0);
    /// ```
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }

    /// Get the constant temperature value.
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Update the constant temperature value.
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature;
    }
}

impl GroundTemperature for ConstantGroundTemperature {
    fn clone_box(&self) -> Box<dyn GroundTemperature> {
        Box::new(self.clone())
    }
    fn ground_temperature(&self, _hour_of_year: usize) -> f64 {
        self.temperature
    }
}

/// Dynamic ground temperature model using the Kusuda formula.
///
/// This model calculates time-varying soil temperature based on annual climate
/// cycles, soil thermal properties, and depth below surface. The Kusuda formula
/// is widely used in building simulation for ground coupling calculations.
///
/// # Mathematical Model
///
/// The Kusuda formula for soil temperature at depth `z` and time `t`:
///
/// ```text
/// T(z,t) = T_mean - T_amp × exp(-d√(π/365α)) × cos(ωt - d√(π/365α))
/// ```
///
/// Where:
/// - `T_mean`: Mean annual soil temperature (°C)
/// - `T_amp`: Annual temperature amplitude (°C)
/// - `d`: Depth below surface (m)
/// - `α`: Soil thermal diffusivity (m²/day)
/// - `ω`: Angular frequency = 2π/365 (rad/day)
/// - `t`: Day of year (0-364)
///
/// # Physical Basis
///
/// - Temperature amplitude decays exponentially with depth
/// - Phase shift increases with depth (peak temp occurs later in year)
/// - Thermal diffusivity controls penetration depth and lag time
///
/// # Typical Parameter Values
///
/// | Parameter | Typical Range | Notes |
/// |-----------|--------------|-------|
/// | T_mean | 5-15°C | Varies by climate (Denver: ~10-12°C) |
/// | T_amp | 8-15°C | Half of annual air temp swing |
/// | Depth | 0.5-2.0m | Slab thickness + insulation |
/// | Diffusivity | 0.05-0.1 m²/day | Dry sand: 0.05, moist clay: 0.08 |
///
/// # When to Use
///
/// - Shallow foundations (<1m depth)
/// - Basements or crawl spaces
/// - High-precision modeling where ground coupling is significant
/// - Climate zones with large annual temperature swings
///
/// # Example
///
/// ```rust
/// use fluxion::sim::boundary::{GroundTemperature, DynamicGroundTemperature};
///
/// // Denver-like climate parameters
/// let ground = DynamicGroundTemperature::new(
///     11.0,  // Mean annual temperature (°C)
///     12.0,  // Annual amplitude (°C)
///     1.0,   // Depth (m)
///     0.07,  // Diffusivity (m²/day)
/// );
///
/// // Temperature varies by hour and depth
/// let temp_winter = ground.ground_temperature(0);      // ~Jan 1
/// let temp_summer = ground.ground_temperature(4380); // ~Jul 1
/// assert!(temp_summer > temp_winter);
/// ```
#[derive(Debug, Clone)]
pub struct DynamicGroundTemperature {
    /// Mean annual soil temperature (°C)
    t_mean: f64,
    /// Annual temperature amplitude (°C)
    t_amplitude: f64,
    /// Depth below ground surface (m)
    depth: f64,
    /// Soil thermal diffusivity (m²/day)
    diffusivity: f64,
}

impl DynamicGroundTemperature {
    /// Create a new dynamic ground temperature model using the Kusuda formula.
    ///
    /// # Arguments
    ///
    /// * `t_mean` - Mean annual soil temperature in °C (typical: 10-12°C)
    /// * `t_amplitude` - Annual temperature amplitude in °C (typical: 8-15°C)
    /// * `depth` - Depth below surface in meters (typical: 0.5-2.0m)
    /// * `diffusivity` - Soil thermal diffusivity in m²/day (typical: 0.05-0.1)
    ///
    /// # Physical Constraints
    ///
    /// - `depth` must be positive
    /// - `diffusivity` must be positive
    /// - `t_amplitude` must be non-negative
    ///
    /// # Panics
    ///
    /// Will panic if physical constraints are violated.
    ///
    /// # Example
    ///
    /// ```rust
    /// use fluxion::sim::boundary::DynamicGroundTemperature;
    ///
    /// // Denver climate parameters
    /// let ground = DynamicGroundTemperature::new(
    ///     11.0,  // Mean temperature
    ///     12.0,  // Amplitude
    ///     1.0,   // Depth
    ///     0.07,  // Diffusivity
    /// );
    /// ```
    pub fn new(t_mean: f64, t_amplitude: f64, depth: f64, diffusivity: f64) -> Self {
        assert!(depth > 0.0, "Depth must be positive");
        assert!(diffusivity > 0.0, "Diffusivity must be positive");
        assert!(t_amplitude >= 0.0, "Amplitude must be non-negative");

        Self {
            t_mean,
            t_amplitude,
            depth,
            diffusivity,
        }
    }

    /// Get the mean annual soil temperature.
    pub fn t_mean(&self) -> f64 {
        self.t_mean
    }

    /// Get the annual temperature amplitude.
    pub fn t_amplitude(&self) -> f64 {
        self.t_amplitude
    }

    /// Get the depth below surface.
    pub fn depth(&self) -> f64 {
        self.depth
    }

    /// Get the soil thermal diffusivity.
    pub fn diffusivity(&self) -> f64 {
        self.diffusivity
    }

    /// Calculate the temperature damping factor at current depth.
    ///
    /// This factor represents how much the annual temperature amplitude
    /// is attenuated at the specified depth.
    ///
    /// # Returns
    ///
    /// Damping factor (dimensionless, range 0-1).
    pub fn damping_factor(&self) -> f64 {
        let decay = self.depth * (PI / (365.0 * self.diffusivity)).sqrt();
        (-decay).exp()
    }

    /// Calculate the phase shift in days.
    ///
    /// This represents how many days the ground temperature lags
    /// behind the surface temperature cycle.
    ///
    /// # Returns
    ///
    /// Phase shift in days.
    pub fn phase_shift(&self) -> f64 {
        let decay = self.depth * (PI / (365.0 * self.diffusivity)).sqrt();
        decay * 365.0 / PI
    }
}

impl GroundTemperature for DynamicGroundTemperature {
    fn clone_box(&self) -> Box<dyn GroundTemperature> {
        Box::new(self.clone())
    }

    fn ground_temperature(&self, hour_of_year: usize) -> f64 {
        // Convert hour to day (0-364)
        let day = (hour_of_year / 24) as f64 % 365.0;

        // Angular frequency (rad/day)
        let omega = 2.0 * PI / 365.0;

        // Decay parameter (dimensionless)
        let decay = self.depth * (PI / (365.0 * self.diffusivity)).sqrt();

        // Kusuda formula
        let damping = (-decay).exp();
        let phase = decay;
        let annual_cycle = (omega * day - phase).cos();

        self.t_mean - self.t_amplitude * damping * annual_cycle
    }
}

/// Monthly ground temperature model — Issue #1435 (IDF import support).
///
/// Holds 12 monthly ground temperature values (°C) as used by EnergyPlus'
/// `Site:GroundTemperature:BuildingSurface` object. The ground temperature
/// at hour `h` of the year is the value for the corresponding month
/// (interpolated as a step function between the 1st of each month).
///
/// `monthly[0]` is January, `monthly[11]` is December. Leap years are
/// treated as 365-day years for the lookup.
#[derive(Debug, Clone)]
pub struct MonthlyGroundTemperature {
    monthly: [f64; 12],
}

impl MonthlyGroundTemperature {
    /// Build a new monthly ground temperature model. The slice must contain
    /// exactly 12 entries (Jan…Dec); values are stored verbatim.
    ///
    /// # Panics
    /// Panics if `monthly.len() != 12`.
    pub fn new(monthly: [f64; 12]) -> Self {
        Self { monthly }
    }

    /// Try to build from a slice of any length. Returns `None` if the
    /// length is not 12.
    pub fn from_slice(monthly: &[f64]) -> Option<Self> {
        if monthly.len() != 12 {
            return None;
        }
        let mut arr = [0.0_f64; 12];
        arr.copy_from_slice(monthly);
        Some(Self { monthly: arr })
    }

    /// Monthly ground temperature values, January → December.
    pub fn monthly(&self) -> &[f64; 12] {
        &self.monthly
    }

    /// First hour of each month in a non-leap year (Jan=0, Feb=744,
    /// Mar=1416, …, Dec=8016). Used to map `hour_of_year` to the
    /// appropriate monthly value.
    const MONTH_START_HOURS: [usize; 12] = [
        0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016,
    ];
}

impl GroundTemperature for MonthlyGroundTemperature {
    fn clone_box(&self) -> Box<dyn GroundTemperature> {
        Box::new(self.clone())
    }

    fn ground_temperature(&self, hour_of_year: usize) -> f64 {
        // Find the most recent month start ≤ hour_of_year.
        let h = hour_of_year.min(Self::MONTH_START_HOURS[11]);
        let mut month = 0_usize;
        for (idx, start) in Self::MONTH_START_HOURS.iter().enumerate() {
            if *start <= h {
                month = idx;
            } else {
                break;
            }
        }
        self.monthly[month]
    }
}

// === Issue #864: Per-surface solar and internal gain distribution ===

/// Per-surface solar gain distribution result (Issue #864).
///
/// Contains the portion of total opaque solar gains distributed to each
/// envelope surface mass node (wall, roof, floor) in Watts.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SurfaceSolarGains {
    /// Opaque solar gain to wall mass node [W]
    pub phi_m_wall: f64,
    /// Opaque solar gain to roof mass node [W]
    pub phi_m_roof: f64,
    /// Opaque solar gain to floor mass node [W]
    pub phi_m_floor: f64,
}

/// Distribute total opaque solar gains to per-surface mass nodes (Issue #864).
///
/// Gains are distributed proportional to irradiance-weighted area fraction.
/// If all irradiance values are zero, distributes by area fraction alone.
pub fn distribute_opaque_solar_gains(
    total_opaque_solar: f64,
    wall_area: f64,
    roof_area: f64,
    floor_area: f64,
    wall_irradiance: f64,
    roof_irradiance: f64,
    floor_irradiance: f64,
) -> SurfaceSolarGains {
    let wall_weight = wall_area * wall_irradiance;
    let roof_weight = roof_area * roof_irradiance;
    let floor_weight = floor_area * floor_irradiance;
    let total_weight = wall_weight + roof_weight + floor_weight;

    if total_weight > 1e-10 {
        SurfaceSolarGains {
            phi_m_wall: total_opaque_solar * wall_weight / total_weight,
            phi_m_roof: total_opaque_solar * roof_weight / total_weight,
            phi_m_floor: total_opaque_solar * floor_weight / total_weight,
        }
    } else if wall_area + roof_area + floor_area > 1e-10 {
        let total_area = wall_area + roof_area + floor_area;
        SurfaceSolarGains {
            phi_m_wall: total_opaque_solar * wall_area / total_area,
            phi_m_roof: total_opaque_solar * roof_area / total_area,
            phi_m_floor: total_opaque_solar * floor_area / total_area,
        }
    } else {
        SurfaceSolarGains::default()
    }
}

/// Distribute radiative internal gains across surfaces by area fraction (Issue #864).
///
/// Returns a tuple `(wall_share, roof_share, floor_share)` in Watts.
pub fn distribute_radiative_gains(
    radiative_gains: f64,
    wall_area: f64,
    roof_area: f64,
    floor_area: f64,
) -> (f64, f64, f64) {
    let total_area = wall_area + roof_area + floor_area;
    if total_area > 1e-10 {
        (
            radiative_gains * wall_area / total_area,
            radiative_gains * roof_area / total_area,
            radiative_gains * floor_area / total_area,
        )
    } else {
        (0.0, 0.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_ground_temperature() {
        let ground = ConstantGroundTemperature::new(10.0);

        // Temperature should be constant regardless of time
        assert_eq!(ground.ground_temperature(0), 10.0);
        assert_eq!(ground.ground_temperature(4380), 10.0); // Mid-year
        assert_eq!(ground.ground_temperature(8759), 10.0); // End of year
    }

    #[test]
    fn test_constant_temperature_getters() {
        let mut ground = ConstantGroundTemperature::new(15.0);
        assert_eq!(ground.temperature(), 15.0);

        ground.set_temperature(12.0);
        assert_eq!(ground.temperature(), 12.0);
        assert_eq!(ground.ground_temperature(100), 12.0);
    }

    #[test]
    fn test_ashrae_140_default() {
        // ASHRAE 140 specifies 10°C constant ground temperature
        let ground = ConstantGroundTemperature::new(10.0);
        assert_eq!(ground.ground_temperature(0), 10.0);
    }

    #[test]
    fn test_dynamic_ground_temperature_creation() {
        let ground = DynamicGroundTemperature::new(11.0, 12.0, 1.0, 0.07);

        assert_eq!(ground.t_mean(), 11.0);
        assert_eq!(ground.t_amplitude(), 12.0);
        assert_eq!(ground.depth(), 1.0);
        assert_eq!(ground.diffusivity(), 0.07);
    }

    #[test]
    #[should_panic(expected = "Depth must be positive")]
    fn test_dynamic_ground_temperature_zero_depth() {
        DynamicGroundTemperature::new(11.0, 12.0, 0.0, 0.07);
    }

    #[test]
    #[should_panic(expected = "Diffusivity must be positive")]
    fn test_dynamic_ground_temperature_zero_diffusivity() {
        DynamicGroundTemperature::new(11.0, 12.0, 1.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "Amplitude must be non-negative")]
    fn test_dynamic_ground_temperature_negative_amplitude() {
        DynamicGroundTemperature::new(11.0, -1.0, 1.0, 0.07);
    }

    #[test]
    fn test_dynamic_ground_temperature_varies_with_time() {
        let ground = DynamicGroundTemperature::new(11.0, 12.0, 1.0, 0.07);

        let temp_winter = ground.ground_temperature(0); // ~Jan 1
        let temp_summer = ground.ground_temperature(4380); // ~Jul 1

        // Summer should be warmer than winter
        assert!(temp_summer > temp_winter);
    }

    #[test]
    fn test_dynamic_ground_temperature_mean() {
        let ground = DynamicGroundTemperature::new(11.0, 12.0, 1.0, 0.07);

        // Average over a full year should approximate the mean
        let mut sum = 0.0;
        for h in (0..8760).step_by(24) {
            sum += ground.ground_temperature(h);
        }
        let avg = sum / 365.0;

        // Should be close to mean (within 0.5°C)
        assert!((avg - ground.t_mean()).abs() < 0.5);
    }

    #[test]
    fn test_dynamic_ground_temperature_damping_factor() {
        // Shallow depth: more variation
        let shallow = DynamicGroundTemperature::new(11.0, 12.0, 0.5, 0.07);
        let damping_shallow = shallow.damping_factor();

        // Deep: less variation
        let deep = DynamicGroundTemperature::new(11.0, 12.0, 2.0, 0.07);
        let damping_deep = deep.damping_factor();

        // Shallow should have higher damping factor (less attenuation)
        assert!(damping_shallow > damping_deep);

        // Both should be between 0 and 1
        assert!(damping_shallow > 0.0 && damping_shallow < 1.0);
        assert!(damping_deep > 0.0 && damping_deep < 1.0);
    }

    #[test]
    fn test_dynamic_ground_temperature_phase_shift() {
        let ground = DynamicGroundTemperature::new(11.0, 12.0, 1.0, 0.07);

        let phase_shift = ground.phase_shift();

        // Phase shift should be positive
        assert!(phase_shift > 0.0);

        // At 1m depth, shift should be significant (several days)
        assert!(phase_shift > 10.0);
    }

    #[test]
    fn test_dynamic_ground_temperature_amplitude_reduces_with_depth() {
        let shallow = DynamicGroundTemperature::new(11.0, 12.0, 0.5, 0.07);
        let deep = DynamicGroundTemperature::new(11.0, 12.0, 2.0, 0.07);

        let temp_shallow_min = (0..365)
            .map(|d| shallow.ground_temperature(d * 24))
            .fold(f64::INFINITY, f64::min);
        let temp_shallow_max = (0..365)
            .map(|d| shallow.ground_temperature(d * 24))
            .fold(f64::NEG_INFINITY, f64::max);
        let amp_shallow = temp_shallow_max - temp_shallow_min;

        let temp_deep_min = (0..365)
            .map(|d| deep.ground_temperature(d * 24))
            .fold(f64::INFINITY, f64::min);
        let temp_deep_max = (0..365)
            .map(|d| deep.ground_temperature(d * 24))
            .fold(f64::NEG_INFINITY, f64::max);
        let amp_deep = temp_deep_max - temp_deep_min;

        // Shallow depth should have larger amplitude
        assert!(amp_shallow > amp_deep);
    }

    #[test]
    fn test_dynamic_ground_temperature_high_diffusivity() {
        // High diffusivity: temperature penetrates deeper
        let low_diff = DynamicGroundTemperature::new(11.0, 12.0, 1.0, 0.05);
        let high_diff = DynamicGroundTemperature::new(11.0, 12.0, 1.0, 0.10);

        let damping_low = low_diff.damping_factor();
        let damping_high = high_diff.damping_factor();

        // High diffusivity should have lower damping (more variation)
        assert!(damping_high > damping_low);
    }

    #[test]
    fn test_ground_temperature_trait_bounds() {
        // Verify that our implementations satisfy Send + Sync
        fn is_send_sync<T: Send + Sync>() {}

        is_send_sync::<ConstantGroundTemperature>();
        is_send_sync::<DynamicGroundTemperature>();

        // Can create boxed trait objects
        let _: Box<dyn GroundTemperature> = Box::new(ConstantGroundTemperature::new(10.0));
        let _: Box<dyn GroundTemperature> =
            Box::new(DynamicGroundTemperature::new(11.0, 12.0, 1.0, 0.07));
    }

    #[test]
    fn test_ground_temperature_continuity_across_year() {
        let ground = DynamicGroundTemperature::new(11.0, 12.0, 1.0, 0.07);

        // Temperature at end of year should be close to start of next year
        let temp_end = ground.ground_temperature(8759);
        let temp_start = ground.ground_temperature(0);

        // Within 0.1°C
        assert!((temp_end - temp_start).abs() < 0.1);
    }

    // === Issue #1435: MonthlyGroundTemperature ===

    #[test]
    fn test_monthly_ground_temperature_lookup() {
        let monthly = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let ground = MonthlyGroundTemperature::new(monthly);
        // January
        assert_eq!(ground.ground_temperature(0), 1.0);
        assert_eq!(ground.ground_temperature(743), 1.0);
        // February (starts at hour 744)
        assert_eq!(ground.ground_temperature(744), 2.0);
        // June (starts at hour 3624)
        assert_eq!(ground.ground_temperature(3624), 6.0);
        // December (starts at hour 8016)
        assert_eq!(ground.ground_temperature(8016), 12.0);
        assert_eq!(ground.ground_temperature(8759), 12.0);
    }

    #[test]
    fn test_monthly_ground_temperature_from_slice_wrong_length() {
        assert!(MonthlyGroundTemperature::from_slice(&[1.0; 11]).is_none());
        assert!(MonthlyGroundTemperature::from_slice(&[1.0; 13]).is_none());
        let ok = MonthlyGroundTemperature::from_slice(&[1.0; 12]).unwrap();
        assert_eq!(ok.monthly()[0], 1.0);
    }

    #[test]
    fn test_constant_ground_temperature_clone() {
        let ground1 = ConstantGroundTemperature::new(10.0);
        let ground2 = ground1.clone();

        assert_eq!(ground1.temperature(), ground2.temperature());
        assert_eq!(ground2.ground_temperature(100), 10.0);
    }

    #[test]
    fn test_dynamic_ground_temperature_clone() {
        let ground1 = DynamicGroundTemperature::new(11.0, 12.0, 1.0, 0.07);
        let ground2 = ground1.clone();

        assert_eq!(ground1.t_mean(), ground2.t_mean());
        assert_eq!(ground1.t_amplitude(), ground2.t_amplitude());
        assert_eq!(ground1.depth(), ground2.depth());
        assert_eq!(ground1.diffusivity(), ground2.diffusivity());

        // Same temperature at same time
        assert_eq!(
            ground1.ground_temperature(1000),
            ground2.ground_temperature(1000)
        );
    }

    // === Issue #864: Distribution function tests ===

    #[test]
    fn test_distribute_opaque_solar_equal_irradiance() {
        let result = distribute_opaque_solar_gains(300.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0);
        assert!((result.phi_m_wall - 100.0).abs() < 1e-10);
        assert!((result.phi_m_roof - 100.0).abs() < 1e-10);
        assert!((result.phi_m_floor - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_distribute_opaque_solar_weighted() {
        let result = distribute_opaque_solar_gains(600.0, 20.0, 20.0, 10.0, 50.0, 200.0, 0.0);
        assert!((result.phi_m_wall - 120.0).abs() < 1e-10);
        assert!((result.phi_m_roof - 480.0).abs() < 1e-10);
        assert!(result.phi_m_floor.abs() < 1e-10);
    }

    #[test]
    fn test_distribute_opaque_solar_zero_irradiance_fallback() {
        let result = distribute_opaque_solar_gains(300.0, 20.0, 10.0, 10.0, 0.0, 0.0, 0.0);
        assert!((result.phi_m_wall - 150.0).abs() < 1e-10);
        assert!((result.phi_m_roof - 75.0).abs() < 1e-10);
        assert!((result.phi_m_floor - 75.0).abs() < 1e-10);
    }

    #[test]
    fn test_distribute_opaque_solar_zero_total() {
        let result = distribute_opaque_solar_gains(0.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0);
        assert!(result.phi_m_wall.abs() < 1e-10);
        assert!(result.phi_m_roof.abs() < 1e-10);
        assert!(result.phi_m_floor.abs() < 1e-10);
    }

    #[test]
    fn test_distribute_radiative_gains_by_area() {
        let (w, r, f) = distribute_radiative_gains(400.0, 30.0, 10.0, 10.0);
        assert!((w - 240.0).abs() < 1e-10);
        assert!((r - 80.0).abs() < 1e-10);
        assert!((f - 80.0).abs() < 1e-10);
    }

    #[test]
    fn test_distribute_radiative_gains_zero() {
        let (w, r, f) = distribute_radiative_gains(0.0, 10.0, 10.0, 10.0);
        assert!(w.abs() < 1e-10);
        assert!(r.abs() < 1e-10);
        assert!(f.abs() < 1e-10);
    }

    #[test]
    fn test_distribute_radiative_gains_zero_area() {
        let (w, r, f) = distribute_radiative_gains(400.0, 0.0, 0.0, 0.0);
        assert!(w.abs() < 1e-10);
        assert!(r.abs() < 1e-10);
        assert!(f.abs() < 1e-10);
    }
}
