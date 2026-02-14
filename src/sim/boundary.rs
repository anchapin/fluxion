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
}
