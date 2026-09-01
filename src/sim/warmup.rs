//! Warm-up / pre-conditioning period for annual simulations per ASHRAE 140 §B2.
//!
//! ASHRAE 140 requires that annual simulations reach periodic steady state before
//! results are collected. Without warm-up, the initial conditions (e.g., all zones
//! at 20°C) introduce a transient that biases early timesteps, particularly for
//! free-floating (FF) cases where there is no HVAC to clamp temperatures.
//!
//! # Strategy
//!
//! Two approaches are supported:
//!
//! 1. **Fixed-duration warm-up** (default): Run N days of simulation before
//!    recording results. The converged state at the end of warm-up becomes the
//!    initial state for the recording period.
//!
//! 2. **Convergence-based warm-up**: Run full-year iterations until the
//!    temperature profile converges between iterations (max ΔT < threshold),
//!    up to a maximum number of iterations.
//!
//! # Weather Data Wrapping
//!
//! The weather data is periodic — 8760 hours wrapping around — so warm-up timesteps
//! use `(hour % 8760)` to index into the weather array.

use crate::physics::cta::ContinuousTensor;
use crate::sim::thermal_model_core::ThermalModel;
use crate::weather::WeatherSource;

/// Hours per year in a TMY weather file.
const HOURS_PER_YEAR: usize = 8760;

/// Default number of warm-up days (14 days per ASHRAE 140 §B2 guidance).
pub const DEFAULT_WARMUP_DAYS: usize = 14;

/// Default maximum warm-up iterations for convergence-based warm-up.
pub const DEFAULT_MAX_WARMUP_ITERATIONS: usize = 4;

/// Default convergence threshold for temperature [°C].
/// Per ASHRAE 140 §B2, convergence to within 0.01°C is typical.
pub const DEFAULT_CONVERGENCE_THRESHOLD: f64 = 0.01;

/// Warm-up configuration for annual simulations.
#[derive(Debug, Clone)]
pub struct WarmupConfig {
    /// Number of warm-up days for fixed-duration warm-up.
    /// Default: 14 days (336 hours).
    pub warmup_days: usize,

    /// Whether to use convergence-based warm-up instead of fixed-duration.
    /// If true, runs full-year iterations until temperatures converge.
    /// If false, runs `warmup_days` of warm-up only.
    pub use_convergence: bool,

    /// Maximum number of full-year iterations for convergence-based warm-up.
    /// Default: 4 (i.e., up to 4 × 8760 hours).
    pub max_iterations: usize,

    /// Convergence threshold [°C]. Warm-up stops when the maximum temperature
    /// change between consecutive full-year runs is below this value.
    /// Default: 0.01°C.
    pub convergence_threshold: f64,

    /// Whether warm-up is enabled at all.
    /// Set to false to skip warm-up entirely (legacy behavior).
    pub enabled: bool,
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self {
            warmup_days: DEFAULT_WARMUP_DAYS,
            use_convergence: false,
            max_iterations: DEFAULT_MAX_WARMUP_ITERATIONS,
            convergence_threshold: DEFAULT_CONVERGENCE_THRESHOLD,
            enabled: true,
        }
    }
}

impl WarmupConfig {
    /// Create a new warm-up config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a disabled warm-up config (no warm-up, legacy behavior).
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Create a fixed-duration warm-up config with the specified number of days.
    pub fn fixed_days(days: usize) -> Self {
        Self {
            warmup_days: days,
            use_convergence: false,
            enabled: true,
            ..Self::default()
        }
    }

    /// Create a convergence-based warm-up config.
    pub fn convergence() -> Self {
        Self {
            use_convergence: true,
            enabled: true,
            ..Self::default()
        }
    }

    /// Set the warm-up duration in days.
    pub fn with_warmup_days(mut self, days: usize) -> Self {
        self.warmup_days = days;
        self
    }

    /// Set the convergence threshold [°C].
    pub fn with_convergence_threshold(mut self, threshold: f64) -> Self {
        self.convergence_threshold = threshold;
        self
    }

    /// Set the maximum number of full-year iterations for convergence.
    pub fn with_max_iterations(mut self, iterations: usize) -> Self {
        self.max_iterations = iterations;
        self
    }

    /// Number of warm-up hours.
    pub fn warmup_hours(&self) -> usize {
        self.warmup_days * 24
    }
}

/// Result of the warm-up phase.
#[derive(Debug, Clone)]
pub struct WarmupResult {
    /// Total number of warm-up timesteps executed.
    pub timesteps: usize,

    /// Maximum temperature change in the last warm-up iteration [°C].
    /// For fixed-duration warm-up, this is the change over the last 24 hours.
    /// For convergence-based, this is the max ΔT between the last two full-year runs.
    pub max_temperature_change: f64,

    /// Whether convergence was achieved (always true for fixed-duration).
    pub converged: bool,

    /// Number of full-year iterations completed (1 for fixed-duration).
    pub iterations: usize,
}

impl std::fmt::Display for WarmupResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "WarmupResult(timesteps={}, max_dT={:.4}°C, converged={}, iterations={})",
            self.timesteps, self.max_temperature_change, self.converged, self.iterations
        )
    }
}

/// Run the warm-up phase on a ThermalModel.
///
/// This advances the model through a warm-up period using the provided weather source,
/// allowing temperatures to converge toward periodic steady state before the recording
/// period begins.
///
/// # Type Parameters
///
/// * `T` - Tensor type (typically `VectorField`)
///
/// # Arguments
///
/// * `model` - The thermal model to warm up (modified in-place)
/// * `weather` - Weather data source (must support `get_hourly_data`)
/// * `config` - Warm-up configuration
///
/// # Returns
///
/// A `WarmupResult` describing what happened during warm-up.
///
/// # Weather Wrapping
///
/// Warm-up timesteps use `(hour_index % 8760)` to wrap around the weather data.
/// For a 14-day warm-up, hours 0–335 of the weather year are used.
/// For convergence mode, full years are simulated using the same weather data repeatedly.
pub fn run_warmup<T>(
    model: &mut ThermalModel<T>,
    weather: &dyn WeatherSource,
    config: &WarmupConfig,
) -> WarmupResult
where
    T: ContinuousTensor<f64>
        + Clone
        + AsRef<[f64]>
        + AsMut<[f64]>
        + From<crate::physics::cta::VectorField>,
{
    if !config.enabled {
        return WarmupResult {
            timesteps: 0,
            max_temperature_change: 0.0,
            converged: true,
            iterations: 0,
        };
    }

    if config.use_convergence {
        run_convergence_warmup(model, weather, config)
    } else {
        run_fixed_warmup(model, weather, config)
    }
}

/// Run a fixed-duration warm-up period.
///
/// Advances the model by `warmup_days` days using wrapping weather data.
fn run_fixed_warmup<T>(
    model: &mut ThermalModel<T>,
    weather: &dyn WeatherSource,
    config: &WarmupConfig,
) -> WarmupResult
where
    T: ContinuousTensor<f64>
        + Clone
        + AsRef<[f64]>
        + AsMut<[f64]>
        + From<crate::physics::cta::VectorField>,
{
    let warmup_hours = config.warmup_hours();

    // Snapshot temperatures before last day of warm-up to measure convergence
    let snapshot_start = warmup_hours.saturating_sub(24);
    let mut temps_before_last_day: Vec<f64> = Vec::new();

    for step in 0..warmup_hours {
        // Snapshot temperatures 24 hours before end of warm-up
        if step == snapshot_start {
            temps_before_last_day = model.setpoints.temperatures.as_ref().to_vec();
        }

        // Wrap weather data for periodic year
        let weather_hour = step % HOURS_PER_YEAR;
        let weather_data = match weather.get_hourly_data(weather_hour) {
            Ok(data) => data,
            Err(_) => {
                // If weather data unavailable, skip this timestep
                continue;
            }
        };

        // Extract the only field used downstream (f64 is Copy) so we can move
        // weather_data into model.solar.weather without an extra clone (Issue #2893).
        let dry_bulb_temp = weather_data.dry_bulb_temp;

        // Set weather on model for solar gain calculation
        model.solar.weather = Some(weather_data);

        // Advance physics
        model.step_physics(step, dry_bulb_temp, 3600.0);
    }

    // Calculate max temperature change over the last 24 hours of warm-up
    let current_temps = model.setpoints.temperatures.as_ref();
    let max_temp_change = if !temps_before_last_day.is_empty() {
        current_temps
            .iter()
            .zip(temps_before_last_day.iter())
            .map(|(curr, prev)| (curr - prev).abs())
            .fold(0.0_f64, f64::max)
    } else {
        0.0
    };

    WarmupResult {
        timesteps: warmup_hours,
        max_temperature_change: max_temp_change,
        converged: true,
        iterations: 1,
    }
}

/// Run convergence-based warm-up.
///
/// Iterates full years until the maximum temperature change between
/// consecutive years is below the convergence threshold.
fn run_convergence_warmup<T>(
    model: &mut ThermalModel<T>,
    weather: &dyn WeatherSource,
    config: &WarmupConfig,
) -> WarmupResult
where
    T: ContinuousTensor<f64>
        + Clone
        + AsRef<[f64]>
        + AsMut<[f64]>
        + From<crate::physics::cta::VectorField>,
{
    let mut iteration = 0;
    let mut max_temp_change = f64::INFINITY;
    let mut total_timesteps = 0;

    // Start with an initial fixed warm-up to stabilize
    let initial_warmup_hours = config.warmup_hours();
    for step in 0..initial_warmup_hours {
        let weather_hour = step % HOURS_PER_YEAR;
        if let Ok(weather_data) = weather.get_hourly_data(weather_hour) {
            // Extract the only field used downstream (f64 is Copy) so we can move
            // weather_data into model.solar.weather without an extra clone (Issue #2893).
            let dry_bulb_temp = weather_data.dry_bulb_temp;
            model.solar.weather = Some(weather_data);
            model.step_physics(step, dry_bulb_temp, 3600.0);
            total_timesteps += 1;
        }
    }
    iteration += 1;

    // Now iterate full years until convergence
    while iteration < config.max_iterations && max_temp_change > config.convergence_threshold {
        // Snapshot temperatures at start of this iteration
        let temps_start: Vec<f64> = model.setpoints.temperatures.as_ref().to_vec();

        // Run one full year
        for step in 0..HOURS_PER_YEAR {
            if let Ok(weather_data) = weather.get_hourly_data(step) {
                // Extract the only field used downstream (f64 is Copy) so we can move
                // weather_data into model.solar.weather without an extra clone (Issue #2893).
                let dry_bulb_temp = weather_data.dry_bulb_temp;
                model.solar.weather = Some(weather_data);
                model.step_physics(step, dry_bulb_temp, 3600.0);
                total_timesteps += 1;
            }
        }

        // Calculate max temperature change between start and end of this year
        let temps_end = model.setpoints.temperatures.as_ref();
        max_temp_change = temps_start
            .iter()
            .zip(temps_end.iter())
            .map(|(start, end)| (start - end).abs())
            .fold(0.0_f64, f64::max);

        iteration += 1;
    }

    let converged = max_temp_change <= config.convergence_threshold;

    WarmupResult {
        timesteps: total_timesteps,
        max_temperature_change: max_temp_change,
        converged,
        iterations: iteration,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::cta::VectorField;
    use crate::sim::thermal_model_core::ThermalModel;
    use crate::sim::thermal_selector::ThermalSelector;
    use crate::validation::ashrae_140_cases::ASHRAE140Case;
    use crate::weather::denver::DenverTmyWeather;

    /// Helper to create a properly initialized free-floating thermal model.
    fn create_free_float_model() -> ThermalModel<VectorField> {
        let spec = ASHRAE140Case::Case600FF.spec();
        let mut model = ThermalModel::<VectorField>::from_spec_with_selector(
            &spec,
            &ThermalSelector::default(),
        )
        .expect("default selector must initialize");
        // Disable HVAC for free-floating mode
        model.setpoints.heating_setpoint = -999.0;
        model.setpoints.cooling_setpoint = 999.0;
        model.hvac.hvac_heating_capacity = 0.0;
        model.hvac.hvac_cooling_capacity = 0.0;
        model
    }

    #[test]
    fn test_warmup_config_default() {
        let config = WarmupConfig::default();
        assert!(config.enabled);
        assert_eq!(config.warmup_days, 14);
        assert_eq!(config.warmup_hours(), 336);
        assert!(!config.use_convergence);
    }

    #[test]
    fn test_warmup_config_disabled() {
        let config = WarmupConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_warmup_config_fixed_days() {
        let config = WarmupConfig::fixed_days(7);
        assert!(config.enabled);
        assert_eq!(config.warmup_days, 7);
        assert_eq!(config.warmup_hours(), 168);
    }

    #[test]
    fn test_warmup_changes_temperatures() {
        let mut model = create_free_float_model();
        let weather = DenverTmyWeather::new();
        let config = WarmupConfig::fixed_days(14);

        // Temperatures start at 20°C
        let initial_temp = model.setpoints.temperatures.as_ref()[0];
        assert!(
            (initial_temp - 20.0).abs() < 1e-9,
            "Initial temp should be 20°C"
        );

        let result = run_warmup(&mut model, &weather, &config);

        // After 14 days of Denver January weather, temperatures should have changed
        assert_eq!(result.timesteps, 336);
        assert!(result.converged);

        let final_temp = model.setpoints.temperatures.as_ref()[0];
        // Temperature should have moved from the initial 20°C
        // (Denver January is cold, so free-floating should drift)
        assert!(
            (final_temp - initial_temp).abs() > 0.01,
            "Temperature should change during warm-up: initial={}, final={}",
            initial_temp,
            final_temp
        );
    }

    #[test]
    fn test_disabled_warmup_does_nothing() {
        let mut model = create_free_float_model();
        let weather = DenverTmyWeather::new();
        let config = WarmupConfig::disabled();

        let initial_temp = model.setpoints.temperatures.as_ref()[0];
        let result = run_warmup(&mut model, &weather, &config);

        assert_eq!(result.timesteps, 0);
        let final_temp = model.setpoints.temperatures.as_ref()[0];
        assert!((final_temp - initial_temp).abs() < 1e-9);
    }

    #[test]
    fn test_convergence_warmup() {
        let mut model = create_free_float_model();
        let weather = DenverTmyWeather::new();
        let config = WarmupConfig::convergence()
            .with_warmup_days(7)
            .with_max_iterations(2);

        let result = run_warmup(&mut model, &weather, &config);

        // Should have run at least the initial warm-up + 1 full year
        assert!(result.timesteps >= 168 + 8760);
        assert!(result.iterations >= 2);
        // Max temperature change should be finite
        assert!(result.max_temperature_change.is_finite());
    }

    #[test]
    fn test_warmup_result_display() {
        let result = WarmupResult {
            timesteps: 336,
            max_temperature_change: 0.005,
            converged: true,
            iterations: 1,
        };
        let display = format!("{}", result);
        assert!(display.contains("336"));
        assert!(display.contains("0.0050"));
        assert!(display.contains("converged=true"));
    }
}
