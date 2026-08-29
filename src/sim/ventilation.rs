//! Ventilation and infiltration modeling.
//!
//! This module provides tools for defining ventilation schedules and calculating
//! time-varying air change rates.
//!
//! # Ventilation Modes
//!
//! - **Constant**: Fixed ACH regardless of conditions
//! - **Scheduled**: Time-based ACH changes (e.g., night ventilation)
//! - **Weather-Responsive**: ACH varies with outdoor temperature and wind speed
//!
//! # `ach_to_conductance` Formula
//!
//! The [`ach_to_conductance`] function converts an air change rate (ACH) to a
//! thermal conductance `h_ve` [W/K] representing ventilation heat transfer:
//!
//! ```text
//! h_ve = (ACH × V × ρ × c_p) / 3600
//! ```
//!
//! Where:
//! - `ACH` — air changes per hour [1/h]
//! - `V`   — zone volume [m³]
//! - `ρ`   — air density [kg/m³] (standard: 1.2)
//! - `c_p` — specific heat of air [J/kg·K] (standard: 1005)
//! - `3600` — seconds per hour conversion factor
//!
//! **Validation**: For `ACH=0.5`, `V=129.6 m³`, `ρ=1.2`, `c_p=1005`:
//! Fluxion ≈ 21.71 W/K vs EnergyPlus ≈ 21.6 W/K (Δ < 0.5%). See Issue #918.

use crate::physics::units::{FromF64, ThermalConductance};
use fluxion_core::earth_tube::EarthTube;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Stack effect coefficient for buoyancy-driven ventilation.
/// Based on ASHRAE 140 natural ventilation model.
pub const STACK_COEFFICIENT: f64 = 0.025;

/// Air density at standard conditions (kg/m³).
pub const AIR_DENSITY: f64 = 1.2;

/// Air specific heat capacity (J/kg·K).
pub const AIR_SPECIFIC_HEAT: f64 = 1000.0;

/// Calculates infiltration ACH from wind speed using the ASHRAE simple method.
///
/// # Arguments
/// * `wind_speed` - Wind speed in m/s
/// * `building_height` - Building height in meters (affects shielding coefficient)
/// * `shielding_factor` - Shielding factor (0 = very sheltered, 1 = no shielding)
///
/// # Returns
/// Wind-driven infiltration rate in ACH
pub fn calculate_wind_infiltration_ach(
    wind_speed: f64,
    building_height: f64,
    shielding_factor: f64,
) -> f64 {
    let shelter_coefficient = 0.0 + (1.0 - shielding_factor) * 0.4;
    let height_factor = (building_height / 3.0).powf(0.5);
    let base_wind_speed = 3.0;
    let n_factor = shelter_coefficient * height_factor;
    n_factor * (wind_speed / base_wind_speed)
}

/// Calculates stack effect ACH for temperature-driven natural ventilation.
///
/// # Arguments
/// * `indoor_temp` - Indoor air temperature (°C)
/// * `outdoor_temp` - Outdoor air temperature (°C)
/// * `height_diff` - Vertical distance between inlet and outlet (m)
/// * `opening_area` - Opening area (m²)
/// * `zone_volume` - Zone volume (m³)
///
/// # Returns
/// Stack-driven infiltration rate in ACH
pub fn calculate_stack_infiltration_ach(
    indoor_temp: f64,
    outdoor_temp: f64,
    height_diff: f64,
    opening_area: f64,
    zone_volume: f64,
) -> f64 {
    if zone_volume <= 0.0 || height_diff <= 0.0 {
        return 0.0;
    }
    let delta_t = (indoor_temp - outdoor_temp).abs();
    if delta_t < 0.5 {
        return 0.0;
    }
    let flow_arg = delta_t / height_diff;
    let flow_sqrt = if flow_arg > 0.0 { flow_arg.sqrt() } else { 0.0 };
    let q_vent = STACK_COEFFICIENT * opening_area * flow_sqrt;
    q_vent / zone_volume
}

/// Calculates combined infiltration ACH using simplified ASHRAE method.
///
/// Combines wind-driven and stack-driven effects for realistic infiltration modeling.
/// Used for free-floating and mixed-mode buildings without mechanical ventilation.
///
/// # Arguments
/// * `outdoor_temp` - Outdoor air temperature (°C)
/// * `indoor_temp` - Indoor air temperature (°C)
/// * `wind_speed` - Wind speed in m/s
/// * `height_diff` - Building height in meters
/// * `opening_area` - Effective opening area for stack effect (m²)
/// * `zone_volume` - Zone volume (m³)
/// * `shielding_factor` - Shielding factor (0-1)
pub fn calculate_combined_infiltration_ach(
    outdoor_temp: f64,
    indoor_temp: f64,
    wind_speed: f64,
    height_diff: f64,
    opening_area: f64,
    zone_volume: f64,
    shielding_factor: f64,
) -> f64 {
    let wind_ach = calculate_wind_infiltration_ach(wind_speed, height_diff, shielding_factor);
    let stack_ach = calculate_stack_infiltration_ach(
        indoor_temp,
        outdoor_temp,
        height_diff,
        opening_area,
        zone_volume,
    );
    let total_ach = wind_ach + stack_ach;
    total_ach.max(0.0)
}

/// Trait for defining air change rate (ACH) schedules.
///
/// All implementations should return weather-dependent ACH when applicable,
/// using the provided weather parameters. The returned ACH is multiplied by
/// zone volume and air properties in [`ach_to_conductance`] to produce the
/// ventilation heat transfer coefficient `h_ve` [W/K] used in the zone energy
/// balance.
pub trait VentilationSchedule: Debug + Send + Sync {
    /// Returns the air change rate (ACH) for a given hour.
    ///
    /// # Arguments
    /// * `hour` — Hour of day (0–23)
    /// * `T_outdoor` — Outdoor dry-bulb temperature [°C]
    /// * `T_indoor` — Indoor air temperature [°C]
    /// * `wind_speed` — Wind speed at building height [m/s]
    /// * `volume` — Zone volume [m³]
    ///
    /// # Returns
    /// Air change rate [1/h]. Implementations may ignore weather arguments
    /// (e.g. [`ConstantVentilation`]) or use them to modulate the rate
    /// (e.g. [`WeatherDependentVentilation`]).
    fn get_ach(
        &self,
        hour: usize,
        T_outdoor: f64,
        T_indoor: f64,
        wind_speed: f64,
        volume: f64,
    ) -> f64;

    /// Clones the schedule into a boxed trait object.
    fn clone_box(&self) -> Box<dyn VentilationSchedule>;
}

/// Computes the forced-convection multiplier for h_tr_is based on ACH.
///
/// Uses the ASHRAE/EnergyPlus empirical correlation for interior forced convection:
/// `h_c = h_c_still + 0.84 * ACH^0.8` [W/m²K]
///
/// Where:
/// - `h_c_still = 3.45 W/m²K` (ASHRAE 140 simplified 5R1C still-air value)
/// - ACH is in air changes per hour
///
/// This gives approximately:
/// - ACH=0.5: ratio ≈ 1.14× (daytime baseline)
/// - ACH=3:   ratio ≈ 1.59× (Case 950 night vent threshold)
/// - ACH=13.14: ratio ≈ 2.91× (Case 650/950 spec night vent ACH=13.14)
/// - ACH=40:  ratio ≈ 5.66× (theoretical high-ACH night vent)
///
/// Reference: ASHRAE Handbook — Fundamentals (ch. 4), EnergyPlus Engineering Reference.
/// Issue #1279, Issue #1624.
pub fn h_tr_is_ach_multiplier(ach: f64) -> f64 {
    const H_C_STILL: f64 = 3.45; // W/m²K - ASHRAE 140 simplified still-air value
    if ach <= 0.0 {
        1.0
    } else {
        // h_c_forced = h_c_still + 0.84 * ACH^0.8
        let h_c_forced = H_C_STILL + 0.84 * ach.powf(0.8);
        h_c_forced / H_C_STILL
    }
}

/// Upper bound on the ACH-driven forced-convection multiplier applied to the
/// air-surface coupling during COOLING-mode operation (Issue #2871).
///
/// Without this cap, the Case 650/950 night-ventilation ACH (13.14) yields a
/// natural multiplier of ≈ 2.91× and very high theoretical ACH (≥ 30) yields
/// multipliers above 4×.  When the morning ramp begins and the night-charged
/// mass node dumps to the still-cool morning air, the unbounded multiplier
/// causes the air node to overshoot the cooling setpoint, inflating the
/// peak-cooling load by 48–92 % across Cases 610/620/630/640/650.
///
/// The cap preserves the natural ASHRAE correlation at low ACH (the natural
/// value is monotone in ACH and stays below `MAX_CONVECTIVE_TO_AIR_MULTIPLIER`
/// for ACH ≲ 7.5 — i.e. all ASHRAE 140 default infiltration schedules and the
/// Case 950 night vent), while preventing the runaway mass-to-air pulse during
/// the morning ramp in the very-high-ACH night flush.
///
/// Value: 2.0×  (≈ 6.9 W/m²K effective h_c vs the 3.45 W/m²K still-air
/// baseline).  This corresponds to ACH ≈ 4.3 and is large enough to deliver
/// the morning cooling benefit, but small enough to bound the peak-cooling
/// overshoot.
///
/// Issue #2871 — Case 600-series peak-cooling OVER prediction.
pub const MAX_CONVECTIVE_TO_AIR_MULTIPLIER: f64 = 2.0;

/// Forced-convection contribution to the air-surface coupling, capped to
/// `MAX_CONVECTIVE_TO_AIR_MULTIPLIER` so high-ACH night flush cannot drive the
/// mass-node pulsed-charging dump during the morning cooling ramp (Issue
/// #2871).
///
/// The natural correlation `h_c = 3.45 + 0.84·ACH^0.8` is preserved at low
/// ACH; only values exceeding `MAX_CONVECTIVE_TO_AIR_MULTIPLIER` are clamped.
/// The cap corresponds to ≈ 6.9 W/m²K effective interior film coefficient
/// (vs the 3.45 W/m²K still-air baseline).
///
/// # Arguments
/// * `ach` — air changes per hour for the active ventilation schedule
///   (typically the night-ventilation fan capacity ÷ zone volume).
///
/// # Returns
/// The dimensionless multiplier to apply to `h_tr_is` during active cooling.
/// Returns `1.0` (no boost) when `ach <= 0`.
pub fn capped_h_tr_is_ach_multiplier(ach: f64) -> f64 {
    let natural = h_tr_is_ach_multiplier(ach);
    if natural <= MAX_CONVECTIVE_TO_AIR_MULTIPLIER {
        natural
    } else {
        MAX_CONVECTIVE_TO_AIR_MULTIPLIER
    }
}

/// A constant ventilation schedule.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ConstantVentilation {
    pub ach: f64,
}

impl ConstantVentilation {
    pub fn new(ach: f64) -> Self {
        Self { ach }
    }
}

impl VentilationSchedule for ConstantVentilation {
    fn get_ach(
        &self,
        _hour: usize,
        _T_outdoor: f64,
        _T_indoor: f64,
        _wind_speed: f64,
        _volume: f64,
    ) -> f64 {
        self.ach
    }
    fn clone_box(&self) -> Box<dyn VentilationSchedule> {
        Box::new(*self)
    }
}

/// A scheduled ventilation system with base infiltration and a timed fan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledVentilation {
    /// Base infiltration rate (ACH) always present.
    pub base_ach: f64,
    /// Additional ACH when fan is ON.
    pub fan_ach: f64,
    /// 24-hour binary schedule (true = fan ON, false = fan OFF).
    pub schedule: [bool; 24],
}

impl ScheduledVentilation {
    /// Creates a new scheduled ventilation.
    pub fn new(base_ach: f64, fan_ach: f64) -> Self {
        Self {
            base_ach,
            fan_ach,
            schedule: [false; 24],
        }
    }

    /// Creates a night ventilation schedule (ON during specified range).
    pub fn night_ventilation(
        base_ach: f64,
        fan_ach: f64,
        start_hour: usize,
        end_hour: usize,
    ) -> Self {
        let mut vent = Self::new(base_ach, fan_ach);
        if start_hour == end_hour {
            vent.schedule = [true; 24];
        } else if start_hour < end_hour {
            for i in start_hour..end_hour {
                vent.schedule[i] = true;
            }
        } else {
            for i in start_hour..24 {
                vent.schedule[i] = true;
            }
            for i in 0..end_hour {
                vent.schedule[i] = true;
            }
        }
        vent
    }
}

impl VentilationSchedule for ScheduledVentilation {
    fn get_ach(
        &self,
        hour: usize,
        _T_outdoor: f64,
        _T_indoor: f64,
        _wind_speed: f64,
        _volume: f64,
    ) -> f64 {
        if self.schedule[hour] {
            self.base_ach + self.fan_ach
        } else {
            self.base_ach
        }
    }
    fn clone_box(&self) -> Box<dyn VentilationSchedule> {
        Box::new(self.clone())
    }
}

/// Weather-dependent ventilation that responds to outdoor temperature.
///
/// This ventilation mode is particularly important for mixed-mode buildings
/// where natural ventilation (windows, vents) responds to weather conditions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherDependentVentilation {
    /// Base infiltration rate (ACH) always present.
    pub base_ach: f64,
    /// Minimum ACH when ventilation is OFF (natural ventilation closed).
    pub min_ach: f64,
    /// Maximum ACH when ventilation is fully ON (windows open).
    pub max_ach: f64,
    /// Outdoor temperature threshold to START ventilation (°C).
    pub start_temp: f64,
    /// Outdoor temperature threshold to fully OPEN ventilation (°C).
    pub full_open_temp: f64,
    /// Indoor temperature threshold for cooling mode (°C).
    pub indoor_cooling_setpoint: f64,
    /// Building height for wind calculation (m).
    pub building_height: f64,
    /// Effective opening area fraction (0-1).
    pub opening_fraction: f64,
}

impl WeatherDependentVentilation {
    pub fn new(
        base_ach: f64,
        min_ach: f64,
        max_ach: f64,
        start_temp: f64,
        full_open_temp: f64,
    ) -> Self {
        Self {
            base_ach,
            min_ach,
            max_ach,
            start_temp,
            full_open_temp: if full_open_temp <= start_temp {
                start_temp + 5.0
            } else {
                full_open_temp
            },
            indoor_cooling_setpoint: 26.0,
            building_height: 2.7,
            opening_fraction: 0.3,
        }
    }

    pub fn mixed_mode(
        base_ach: f64,
        max_ach: f64,
        start_temp: f64,
        full_open_temp: f64,
        indoor_cooling_setpoint: f64,
    ) -> Self {
        Self {
            base_ach,
            min_ach: base_ach,
            max_ach,
            start_temp,
            full_open_temp: if full_open_temp <= start_temp {
                start_temp + 5.0
            } else {
                full_open_temp
            },
            indoor_cooling_setpoint,
            building_height: 2.7,
            opening_fraction: 0.3,
        }
    }

    /// Calculate ventilation ACH based on weather and indoor conditions.
    ///
    /// Returns the actual ACH considering both outdoor temperature and wind.
    ///
    /// # ASHRAE 140-2023 §5.5.3.6 default-infiltration lock-in
    ///
    /// ASHRAE Standard 140 (BESTEST) specifies a default infiltration rate of
    /// **0.5 ACH** for the Case 900 / 920 / 940 / 950 / 960 reference models.
    /// When this struct is constructed with `min_ach == max_ach == 0.5` and the
    /// spec inputs (T_in = 20 °C, building height = 2.7 m, shielding = 0.5,
    /// volume = 129.6 m³, Denver TMY3 wind), this method returns exactly
    /// `0.5 ACH` for every of the 8760 hours — i.e. the wind/temperature
    /// blending does not perturb the spec value.
    ///
    /// See `tests/ventilation_isolation.rs::test_ashrae_140_0p5_ach_default`
    /// for the lock-in assertion (Issue #1327), and PRs #1278/#1279 for the
    /// upstream wind + forced-convection fixes that this lock-in guards.
    pub fn get_ach_weather(
        &self,
        outdoor_temp: f64,
        indoor_temp: f64,
        wind_speed: f64,
        zone_volume: f64,
    ) -> f64 {
        let temp_benefit = self.outdoor_temp_benefit(outdoor_temp, indoor_temp);
        let wind_benefit = self.wind_benefit(wind_speed, outdoor_temp, indoor_temp, zone_volume);
        let combined = (temp_benefit + wind_benefit) / 2.0;
        (self.min_ach + (self.max_ach - self.min_ach) * combined).max(self.min_ach)
    }

    /// Calculate outdoor temperature benefit (0-1).
    ///
    /// Returns 0 when outdoor temp <= start_temp (no benefit),
    /// Returns 1 when outdoor temp >= full_open_temp (full benefit).
    fn outdoor_temp_benefit(&self, outdoor_temp: f64, indoor_temp: f64) -> f64 {
        if outdoor_temp <= self.start_temp {
            return 0.0;
        }
        if outdoor_temp >= self.full_open_temp {
            return 1.0;
        }
        if indoor_temp <= self.indoor_cooling_setpoint {
            return 0.0;
        }
        let delta_t_out = self.full_open_temp - self.start_temp;
        if delta_t_out <= 0.0 {
            return 0.0;
        }
        ((outdoor_temp - self.start_temp) / delta_t_out).clamp(0.0, 1.0)
    }

    /// Calculate wind benefit (0-1) using combined wind + stack infiltration.
    ///
    /// Uses `calculate_combined_infiltration_ach` which properly accounts for:
    /// - Wind-driven infiltration via `calculate_wind_infiltration_ach`
    /// - Stack-driven infiltration via `calculate_stack_infiltration_ach`
    fn wind_benefit(
        &self,
        wind_speed: f64,
        outdoor_temp: f64,
        indoor_temp: f64,
        zone_volume: f64,
    ) -> f64 {
        let opening_area = self.opening_fraction * 2.0 * (self.building_height * 3.0);
        // Shielding factor 0.5 is appropriate for weather-responsive ventilation
        // (windows/vents have partial shielding vs. whole-building tightness)
        let shielding_factor = 0.5;
        let ach = calculate_combined_infiltration_ach(
            outdoor_temp,
            indoor_temp,
            wind_speed,
            self.building_height,
            opening_area,
            zone_volume,
            shielding_factor,
        );
        (ach / self.max_ach).clamp(0.0, 1.0)
    }
}

impl VentilationSchedule for WeatherDependentVentilation {
    fn get_ach(
        &self,
        _hour: usize,
        T_outdoor: f64,
        T_indoor: f64,
        wind_speed: f64,
        volume: f64,
    ) -> f64 {
        self.get_ach_weather(T_outdoor, T_indoor, wind_speed, volume)
    }
    fn clone_box(&self) -> Box<dyn VentilationSchedule> {
        Box::new(self.clone())
    }
}

/// Utility to calculate thermal conductance (W/K) from air change rate (ACH).
///
/// This computes the h_ve ventilation heat transfer coefficient, which represents
/// the rate of heat gain/loss due to ventilation air exchange.
///
/// # Validation
/// The h_ve calculation is validated against EnergyPlus reference data.
/// For a reference case with ACH=0.5, volume=129.6 m³, rho=1.2 kg/m³, cp=1005 J/kg·K:
/// - Fluxion result: ~21.71 W/K
/// - EnergyPlus result: 21.6 W/K
/// - Difference: 0.5% (within acceptable tolerance)
///
/// See GitHub Issue #918 for the full validation study.
/// Issue concluded that h_ve values are correct and the ventilation module is working as expected.
///
/// # Arguments
/// * `ach` - Air changes per hour (1/h)
/// * `volume` - Zone volume (m³)
/// * `rho` - Air density (kg/m³), typically 1.2
/// * `cp` - Specific heat capacity of air (J/kg·K), typically 1005
pub fn ach_to_conductance(ach: f64, volume: f64, rho: f64, cp: f64) -> ThermalConductance {
    ThermalConductance::from_value((ach * volume * rho * cp) / 3600.0)
}

/// A ventilation schedule decorator that applies earth tube (ground-air heat exchanger)
/// pre-conditioning to incoming outdoor air.
///
/// The earth tube pre-heats winter intake air and pre-cools summer intake air using
/// the ground's stable thermal mass, reducing HVAC heating and cooling loads.
///
/// # Example
///
/// ```
/// use fluxion::sim::ventilation::{ConstantVentilation, EarthTubeVentilation, VentilationSchedule};
/// use fluxion_core::earth_tube::EarthTube;
///
/// // Base ventilation schedule
/// let base = ConstantVentilation::new(0.5);
///
/// // Earth tube with typical parameters
/// let earth_tube = EarthTube::new()
///     .soil_temperature_K(285.15)  // ~12°C ground temperature
///     .flow_rate_m3_s(0.05);
///
/// // Decorated schedule with earth tube pre-conditioning
/// let vent = EarthTubeVentilation::new(base, earth_tube);
///
/// // ACH is unchanged from base schedule
/// let ach = vent.get_ach(12, 30.0, 25.0, 2.0, 100.0);
/// assert_eq!(ach, 0.5);
///
/// // But supply temperature is pre-conditioned
/// let supply = vent.supply_temperature(35.0);  // Hot summer day
/// assert!(supply < 35.0);  // Pre-cooled
/// assert!(supply > 12.0);  // Above ground temperature
/// ```
#[derive(Debug)]
pub struct EarthTubeVentilation<S: VentilationSchedule + Clone> {
    inner: S,
    earth_tube: EarthTube,
}

impl<S: VentilationSchedule + Clone> Clone for EarthTubeVentilation<S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            earth_tube: self.earth_tube.clone(),
        }
    }
}

impl<S: VentilationSchedule + Clone> EarthTubeVentilation<S> {
    /// Creates a new earth tube decorated ventilation schedule.
    pub fn new(inner: S, earth_tube: EarthTube) -> Self {
        Self { inner, earth_tube }
    }

    /// Returns the earth tube's supply air temperature after pre-conditioning.
    ///
    /// This is the temperature of ventilation air after passing through the earth tube,
    /// which has been pre-heated (winter) or pre-cooled (summer) by the ground.
    ///
    /// # Arguments
    ///
    /// * `outdoor_temp_K` - Outdoor air temperature in Kelvin entering the earth tube
    ///
    /// # Returns
    ///
    /// Supply air temperature after earth tube pre-conditioning (Kelvin)
    pub fn supply_temperature(&self, outdoor_temp_K: f64) -> f64 {
        self.earth_tube.supply_temperature(outdoor_temp_K)
    }

    /// Returns the heat transfer rate through the earth tube (Watts).
    ///
    /// Positive = heat gain (pre-heating), Negative = heat loss (pre-cooling).
    pub fn heat_transfer_rate(&self, outdoor_temp_K: f64) -> f64 {
        self.earth_tube.heat_transfer_rate(outdoor_temp_K)
    }

    /// Returns the temperature difference (K) due to earth tube pre-conditioning.
    ///
    /// Positive = pre-heating (outdoor colder than ground)
    /// Negative = pre-cooling (outdoor warmer than ground)
    pub fn temperature_difference(&self, outdoor_temp_K: f64) -> f64 {
        self.earth_tube.temperature_difference(outdoor_temp_K)
    }

    /// Returns a reference to the inner ventilation schedule.
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Returns a reference to the earth tube.
    pub fn earth_tube(&self) -> &EarthTube {
        &self.earth_tube
    }

    /// Consumes the decorator and returns the inner ventilation schedule.
    pub fn into_inner(self) -> S {
        self.inner
    }
}

impl<S: VentilationSchedule + Clone + 'static> VentilationSchedule for EarthTubeVentilation<S> {
    fn get_ach(
        &self,
        hour: usize,
        T_outdoor: f64,
        T_indoor: f64,
        wind_speed: f64,
        volume: f64,
    ) -> f64 {
        self.inner
            .get_ach(hour, T_outdoor, T_indoor, wind_speed, volume)
    }

    fn clone_box(&self) -> Box<dyn VentilationSchedule> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ach_to_conductance, calculate_combined_infiltration_ach, calculate_stack_infiltration_ach,
        calculate_wind_infiltration_ach, h_tr_is_ach_multiplier, ConstantVentilation,
        EarthTubeVentilation, ScheduledVentilation, VentilationSchedule,
        WeatherDependentVentilation,
    };
    use fluxion_core::earth_tube::EarthTube;

    #[test]
    fn test_constant_ventilation() {
        let vent = ConstantVentilation::new(0.5);
        assert_eq!(vent.ach, 0.5);
        assert_eq!(vent.get_ach(0, 20.0, 22.0, 2.0, 100.0), 0.5);
        assert_eq!(vent.get_ach(12, 20.0, 22.0, 2.0, 100.0), 0.5);
        assert_eq!(vent.get_ach(23, 20.0, 22.0, 2.0, 100.0), 0.5);
    }

    #[test]
    fn test_constant_ventilation_clone() {
        let vent = ConstantVentilation::new(1.0);
        let cloned = vent.clone_box();
        assert_eq!(cloned.get_ach(5, 20.0, 22.0, 2.0, 100.0), 1.0);
    }

    #[test]
    fn test_scheduled_ventilation_default() {
        let vent = ScheduledVentilation::new(0.3, 2.0);
        assert_eq!(vent.base_ach, 0.3);
        assert_eq!(vent.fan_ach, 2.0);
        assert!(!vent.schedule.iter().any(|&x| x)); // all false
                                                    // Should return base_ach for all hours
        for hour in 0..24 {
            assert_eq!(vent.get_ach(hour, 20.0, 22.0, 2.0, 100.0), 0.3);
        }
    }

    #[test]
    fn test_night_ventilation_normal_range() {
        let vent = ScheduledVentilation::night_ventilation(0.3, 2.0, 22, 6);
        // Fan ON from hour 22 to 23, 0 to 5
        assert_eq!(vent.get_ach(21, 20.0, 22.0, 2.0, 100.0), 0.3); // before start
        assert_eq!(vent.get_ach(22, 20.0, 22.0, 2.0, 100.0), 2.3); // fan on
        assert_eq!(vent.get_ach(23, 20.0, 22.0, 2.0, 100.0), 2.3); // fan on
        assert_eq!(vent.get_ach(0, 20.0, 22.0, 2.0, 100.0), 2.3); // fan on (next day)
        assert_eq!(vent.get_ach(5, 20.0, 22.0, 2.0, 100.0), 2.3); // fan on
        assert_eq!(vent.get_ach(6, 20.0, 22.0, 2.0, 100.0), 0.3); // fan off
        assert_eq!(vent.get_ach(12, 20.0, 22.0, 2.0, 100.0), 0.3); // fan off
    }

    #[test]
    fn test_night_ventilation_same_start_end() {
        let vent = ScheduledVentilation::night_ventilation(0.3, 2.0, 10, 10);
        // When start == end, fan is on all 24 hours
        for hour in 0..24 {
            assert_eq!(vent.get_ach(hour, 20.0, 22.0, 2.0, 100.0), 2.3);
        }
    }

    #[test]
    fn test_night_ventilation_single_hour() {
        let vent = ScheduledVentilation::night_ventilation(0.5, 1.5, 14, 15);
        assert_eq!(vent.get_ach(13, 20.0, 22.0, 2.0, 100.0), 0.5);
        assert_eq!(vent.get_ach(14, 20.0, 22.0, 2.0, 100.0), 2.0); // fan on
        assert_eq!(vent.get_ach(15, 20.0, 22.0, 2.0, 100.0), 0.5); // fan off
    }

    #[test]
    fn test_scheduled_ventilation_clone() {
        let vent = ScheduledVentilation::night_ventilation(0.3, 2.0, 20, 8);
        let cloned = vent.clone_box();
        assert_eq!(cloned.get_ach(21, 20.0, 22.0, 2.0, 100.0), 2.3);
        assert_eq!(cloned.get_ach(10, 20.0, 22.0, 2.0, 100.0), 0.3);
    }

    #[test]
    fn test_ach_to_conductance() {
        use crate::physics::units::ToF64;
        // Standard values: ach=1.0, volume=100m³, rho=1.2, cp=1005
        let conductance = ach_to_conductance(1.0, 100.0, 1.2, 1005.0);
        assert!((conductance.to_value() - 33.5).abs() < 0.01); // (1*100*1.2*1005)/3600 = 33.5
    }

    #[test]
    fn test_ach_to_conductance_zero() {
        use crate::physics::units::ToF64;
        assert_eq!(ach_to_conductance(0.0, 100.0, 1.2, 1005.0).to_value(), 0.0);
    }

    #[test]
    fn test_ach_to_conductance_scaling() {
        use crate::physics::units::ToF64;
        // Doubling ACH should double conductance
        let c1 = ach_to_conductance(0.5, 100.0, 1.2, 1005.0);
        let c2 = ach_to_conductance(1.0, 100.0, 1.2, 1005.0);
        assert!((c2.to_value() - 2.0 * c1.to_value()).abs() < 0.001);
    }

    #[test]
    fn test_ventilation_schedule_trait_object() {
        let vent1: Box<dyn VentilationSchedule> = Box::new(ConstantVentilation::new(0.5));
        let vent2: Box<dyn VentilationSchedule> =
            Box::new(ScheduledVentilation::night_ventilation(0.3, 2.0, 22, 6));

        assert_eq!(vent1.get_ach(10, 20.0, 22.0, 2.0, 100.0), 0.5);
        assert_eq!(vent2.get_ach(23, 20.0, 22.0, 2.0, 100.0), 2.3);
    }

    #[test]
    fn test_wind_infiltration_basic() {
        let ach = calculate_wind_infiltration_ach(3.0, 3.0, 0.0);
        assert!(ach >= 0.0);
        assert!(ach < 1.0);
    }

    #[test]
    fn test_wind_infiltration_high_wind() {
        let ach_calm = calculate_wind_infiltration_ach(2.0, 3.0, 0.0);
        let ach_windy = calculate_wind_infiltration_ach(6.0, 3.0, 0.0);
        assert!(ach_windy > ach_calm);
    }

    #[test]
    fn test_stack_infiltration_no_delta_t() {
        let ach = calculate_stack_infiltration_ach(25.0, 25.0, 2.0, 1.0, 100.0);
        assert_eq!(ach, 0.0);
    }

    #[test]
    fn test_stack_infiltration_with_delta_t() {
        let ach = calculate_stack_infiltration_ach(25.0, 20.0, 2.0, 1.0, 100.0);
        assert!(ach > 0.0);
    }

    #[test]
    fn test_combined_infiltration_ach() {
        let ach = calculate_combined_infiltration_ach(
            25.0,  // outdoor
            28.0,  // indoor
            3.0,   // wind
            2.7,   // height
            1.0,   // opening
            129.6, // volume
            0.3,   // shielding
        );
        assert!(ach >= 0.0);
    }

    #[test]
    fn test_weather_dependent_ventilation_creation() {
        let vent = WeatherDependentVentilation::new(0.3, 0.3, 2.0, 18.0, 26.0);
        assert_eq!(vent.base_ach, 0.3);
        assert_eq!(vent.min_ach, 0.3);
        assert_eq!(vent.max_ach, 2.0);
        assert_eq!(vent.start_temp, 18.0);
    }

    #[test]
    fn test_weather_dependent_ventilation_outdoor_temp_benefit() {
        let vent = WeatherDependentVentilation::new(0.3, 0.3, 2.0, 18.0, 26.0);
        assert_eq!(vent.outdoor_temp_benefit(15.0, 28.0), 0.0);
        assert_eq!(vent.outdoor_temp_benefit(18.0, 28.0), 0.0);
        assert!(vent.outdoor_temp_benefit(23.0, 28.0) > 0.0);
        assert_eq!(vent.outdoor_temp_benefit(26.0, 28.0), 1.0);
    }

    #[test]
    fn test_weather_dependent_ventilation_indoor_not_cooling() {
        let vent = WeatherDependentVentilation::new(0.3, 0.3, 2.0, 18.0, 26.0);
        assert_eq!(vent.outdoor_temp_benefit(25.0, 24.0), 0.0);
    }

    #[test]
    fn test_weather_dependent_ventilation_fallback_full_open() {
        let vent = WeatherDependentVentilation::new(0.3, 0.3, 2.0, 26.0, 18.0);
        assert_eq!(vent.full_open_temp, 31.0);
    }

    // =============================================================================
    // Issue #1624: Forced-convection h_tr_is boost during high ACH night flush
    // =============================================================================

    #[test]
    fn test_h_tr_is_ach_multiplier_zero_ach() {
        // Zero or negative ACH should return 1.0 (no boost)
        assert_eq!(h_tr_is_ach_multiplier(0.0), 1.0);
        assert_eq!(h_tr_is_ach_multiplier(-1.0), 1.0);
    }

    #[test]
    fn test_h_tr_is_ach_multiplier_baseline() {
        // ACH=0.5: ratio ≈ 1.14× (daytime baseline, below night flush threshold)
        let multiplier = h_tr_is_ach_multiplier(0.5);
        assert!((multiplier - 1.14).abs() < 0.01);
    }

    #[test]
    fn test_h_tr_is_ach_multiplier_night_flush_threshold() {
        // ACH=3.0: ratio ≈ 1.59× (Case 950 night vent threshold)
        let multiplier = h_tr_is_ach_multiplier(3.0);
        assert!((multiplier - 1.59).abs() < 0.01);
    }

    #[test]
    fn test_h_tr_is_ach_multiplier_high_ach() {
        // ACH=13.14: ratio ≈ 2.91× (Case 650/950 spec night vent ACH=13.14)
        let multiplier = h_tr_is_ach_multiplier(13.14);
        assert!((multiplier - 2.91).abs() < 0.02);
    }

    #[test]
    fn test_h_tr_is_ach_multiplier_very_high_ach() {
        // ACH=40: ratio ≈ 5.66× (theoretical high-ACH night vent)
        let multiplier = h_tr_is_ach_multiplier(40.0);
        assert!((multiplier - 5.66).abs() < 0.02);
    }

    #[test]
    fn test_h_tr_is_ach_multiplier_monotonic_increase() {
        // Multiplier should increase monotonically with ACH
        let m1 = h_tr_is_ach_multiplier(1.0);
        let m2 = h_tr_is_ach_multiplier(5.0);
        let m3 = h_tr_is_ach_multiplier(10.0);
        let m4 = h_tr_is_ach_multiplier(20.0);
        assert!(m1 < m2);
        assert!(m2 < m3);
        assert!(m3 < m4);
    }

    #[test]
    fn test_h_tr_is_ach_multiplier_ach_3_threshold_boost() {
        // Verify boost activates at ACH >= 3.0 (night flush threshold)
        let multiplier_below = h_tr_is_ach_multiplier(2.9);
        let multiplier_at = h_tr_is_ach_multiplier(3.0);
        let multiplier_above = h_tr_is_ach_multiplier(4.0);

        // Below threshold: multiplier < 1.59
        assert!(multiplier_below < 1.59);
        // At threshold: multiplier ≈ 1.59
        assert!((multiplier_at - 1.59).abs() < 0.01);
        // Above threshold: multiplier > 1.59
        assert!(multiplier_above > 1.59);
    }

    // =============================================================================
    // EarthTubeVentilation integration tests (Issue #2276)
    // =============================================================================

    #[test]
    fn test_earth_tube_ventilation_winter_preheating() {
        use super::EarthTubeVentilation;

        let base = ConstantVentilation::new(0.5);
        let earth_tube = EarthTube::new()
            .soil_temperature_K(285.15) // ~12°C
            .flow_rate_m3_s(0.05);

        let vent = EarthTubeVentilation::new(base, earth_tube);

        // ACH is unchanged from base schedule
        let ach = vent.get_ach(8, -5.0, 20.0, 2.0, 129.6);
        assert_eq!(ach, 0.5);

        // Supply temperature is pre-heated
        // Outdoor -5°C (268.15K) → supply should be warmer
        let supply_K = vent.supply_temperature(268.15);
        let delta = supply_K - 268.15;

        println!(
            "Winter: outdoor=-5°C, supply={:.1}°C, preheat={:.1}°C",
            supply_K - 273.15,
            delta
        );

        assert!(
            delta >= 5.0,
            "Winter pre-heating should be at least 5°C, got {:.1}°C",
            delta
        );
    }

    #[test]
    fn test_earth_tube_ventilation_summer_precooling() {
        use super::EarthTubeVentilation;

        let base = ConstantVentilation::new(0.5);
        let earth_tube = EarthTube::new()
            .soil_temperature_K(291.15) // ~18°C
            .flow_rate_m3_s(0.05);

        let vent = EarthTubeVentilation::new(base, earth_tube);

        // ACH is unchanged
        let ach = vent.get_ach(14, 35.0, 25.0, 2.0, 129.6);
        assert_eq!(ach, 0.5);

        // Supply temperature is pre-cooled
        // Outdoor 35°C (308.15K) → supply should be cooler
        let supply_K = vent.supply_temperature(308.15);
        let delta = supply_K - 308.15; // negative

        println!(
            "Summer: outdoor=35°C, supply={:.1}°C, precool={:.1}°C",
            supply_K - 273.15,
            delta
        );

        assert!(
            delta <= -5.0,
            "Summer pre-cooling should be at least 5°C, got {:.1}°C",
            delta.abs()
        );
    }

    #[test]
    fn test_earth_tube_ventilation_clone_box() {
        use super::EarthTubeVentilation;

        let base = ConstantVentilation::new(0.5);
        let earth_tube = EarthTube::new();
        let vent = EarthTubeVentilation::new(base, earth_tube);

        let cloned = vent.clone_box();
        let ach = cloned.get_ach(12, 20.0, 25.0, 2.0, 100.0);
        assert_eq!(ach, 0.5);
    }

    #[test]
    fn test_earth_tube_ventilation_heat_transfer_rate() {
        use super::EarthTubeVentilation;

        let base = ConstantVentilation::new(0.5);
        let earth_tube = EarthTube::new()
            .soil_temperature_K(285.15)
            .flow_rate_m3_s(0.05);

        let vent = EarthTubeVentilation::new(base, earth_tube);

        // Winter: heating mode (positive heat transfer)
        let Q_winter = vent.heat_transfer_rate(268.15); // -5°C
        assert!(
            Q_winter > 0.0,
            "Winter should have positive heat transfer (pre-heating)"
        );

        // Summer: cooling mode (negative heat transfer)
        let Q_summer = vent.heat_transfer_rate(308.15); // 35°C
        assert!(
            Q_summer < 0.0,
            "Summer should have negative heat transfer (pre-cooling)"
        );

        println!("Q_winter = {:.1} W, Q_summer = {:.1} W", Q_winter, Q_summer);
    }

    // ─── Constants ─────────────────────────────────────────────────────────────

    #[test]
    fn test_ventilation_constants() {
        assert_eq!(super::STACK_COEFFICIENT, 0.025);
        assert_eq!(super::AIR_DENSITY, 1.2);
        assert_eq!(super::AIR_SPECIFIC_HEAT, 1000.0);
    }

    // ─── calculate_wind_infiltration_ach edge cases ─────────────────────────────

    #[test]
    fn test_wind_infiltration_zero_wind() {
        // Zero wind speed → no wind-driven infiltration
        let ach = calculate_wind_infiltration_ach(0.0, 3.0, 0.5);
        assert_eq!(ach, 0.0);
    }

    #[test]
    fn test_wind_infiltration_max_shielding() {
        // shielding_factor=1.0 → maximum shelter
        let ach_no_shield = calculate_wind_infiltration_ach(3.0, 3.0, 0.0);
        let ach_shielded = calculate_wind_infiltration_ach(3.0, 3.0, 1.0);
        assert!(ach_shielded < ach_no_shield);
    }

    #[test]
    fn test_wind_infiltration_building_height_effect() {
        // Taller building → greater height factor → higher infiltration
        let ach_short = calculate_wind_infiltration_ach(3.0, 2.0, 0.5);
        let ach_tall = calculate_wind_infiltration_ach(3.0, 10.0, 0.5);
        assert!(ach_tall > ach_short);
    }

    // ─── calculate_stack_infiltration_ach edge cases ───────────────────────────

    #[test]
    fn test_stack_infiltration_zero_volume() {
        // Zero volume → no infiltration
        let ach = calculate_stack_infiltration_ach(25.0, 20.0, 2.0, 1.0, 0.0);
        assert_eq!(ach, 0.0);
    }

    #[test]
    fn test_stack_infiltration_zero_height() {
        // Zero height diff → no stack effect
        let ach = calculate_stack_infiltration_ach(25.0, 20.0, 0.0, 1.0, 100.0);
        assert_eq!(ach, 0.0);
    }

    #[test]
    fn test_stack_infiltration_below_delta_t_threshold() {
        // delta_t < 0.5 → no stack effect
        let ach = calculate_stack_infiltration_ach(25.0, 24.6, 2.0, 1.0, 100.0);
        assert_eq!(ach, 0.0);
    }

    // ─── calculate_combined_infiltration_ach ─────────────────────────────────

    #[test]
    fn test_combined_infiltration_ach_non_negative() {
        // Total ACH should never be negative
        let ach = calculate_combined_infiltration_ach(
            30.0, 20.0, 5.0, 2.7, 1.0, 129.6, 0.5,
        );
        assert!(ach >= 0.0);
    }

    // ─── capped_h_tr_is_ach_multiplier ────────────────────────────────────────

    #[test]
    fn test_capped_h_tr_is_ach_multiplier_below_cap() {
        // Below cap: natural multiplier applies
        let capped = super::capped_h_tr_is_ach_multiplier(3.0);
        let natural = h_tr_is_ach_multiplier(3.0);
        assert!((capped - natural).abs() < 1e-10);
    }

    #[test]
    fn test_capped_h_tr_is_ach_multiplier_at_cap() {
        // Near the cap value, should equal cap
        // The natural multiplier hits 2.0 around ACH ≈ 4.3
        let capped = super::capped_h_tr_is_ach_multiplier(10.0);
        assert!((capped - super::MAX_CONVECTIVE_TO_AIR_MULTIPLIER).abs() < 0.01);
    }

    #[test]
    fn test_capped_h_tr_is_ach_multiplier_above_cap() {
        // Above cap: should be clamped to 2.0
        let capped = super::capped_h_tr_is_ach_multiplier(40.0);
        assert!((capped - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_capped_h_tr_is_ach_multiplier_zero_ach() {
        assert_eq!(super::capped_h_tr_is_ach_multiplier(0.0), 1.0);
    }

    // ─── WeatherDependentVentilation get_ach_weather ─────────────────────────

    #[test]
    fn test_weather_dependent_get_ach_outdoor_below_start_temp_still_has_wind() {
        let vent = WeatherDependentVentilation::new(0.3, 0.3, 2.0, 18.0, 26.0);
        // Even when outdoor is cold (below start_temp), wind benefit can still contribute
        let ach = vent.get_ach_weather(15.0, 28.0, 2.0, 129.6);
        // Wind benefit adds to the ACH
        assert!(ach >= 0.3);
        assert!(ach <= 2.0);
    }

    #[test]
    fn test_weather_dependent_get_ach_indoor_not_cooling_wind_benefit_still_applies() {
        let vent = WeatherDependentVentilation::new(0.3, 0.3, 2.0, 18.0, 26.0);
        // When indoor is not in cooling mode, wind benefit still applies
        let ach = vent.get_ach_weather(25.0, 24.0, 2.0, 129.6);
        assert!(ach >= 0.3);
        assert!(ach <= 2.0);
    }

    #[test]
    fn test_weather_dependent_mixed_mode_factory() {
        let vent = WeatherDependentVentilation::mixed_mode(0.3, 2.0, 18.0, 26.0, 26.0);
        assert_eq!(vent.base_ach, 0.3);
        assert_eq!(vent.min_ach, 0.3); // min_ach = base_ach in mixed_mode
        assert_eq!(vent.max_ach, 2.0);
        assert_eq!(vent.start_temp, 18.0);
    }

    #[test]
    fn test_weather_dependent_get_ach_ach_values_bounded() {
        let vent = WeatherDependentVentilation::new(0.3, 0.3, 2.0, 18.0, 26.0);
        // ACH should always be within [min_ach, max_ach]
        for tout in 0..40 {
            let ach = vent.get_ach_weather(tout as f64, 28.0, 2.0, 129.6);
            assert!(ach >= 0.3 - 1e-9);
            assert!(ach <= 2.0 + 1e-9);
        }
    }

    // ─── ScheduledVentilation cross-midnight wrap ────────────────────────────

    #[test]
    fn test_night_ventilation_cross_midnight_wrap() {
        // Verify hours 22..24 and 0..6 are ON, others are OFF
        let vent = ScheduledVentilation::night_ventilation(0.3, 1.0, 22, 6);
        for hour in 0..24 {
            let ach = vent.get_ach(hour, 20.0, 22.0, 0.0, 100.0);
            if hour >= 22 || hour < 6 {
                assert_eq!(ach, 1.3, "hour {} should be fan ON", hour);
            } else {
                assert_eq!(ach, 0.3, "hour {} should be fan OFF", hour);
            }
        }
    }

    // ─── EarthTubeVentilation temperature_difference ───────────────────────────

    #[test]
    fn test_earth_tube_temperature_difference() {
        let base = ConstantVentilation::new(0.5);
        let earth_tube = EarthTube::new().soil_temperature_K(285.15);
        let vent = EarthTubeVentilation::new(base, earth_tube);

        // Outdoor warmer than ground → negative difference (pre-cooling)
        let delta_hot = vent.temperature_difference(308.15); // 35°C
        assert!(delta_hot < 0.0);

        // Outdoor colder than ground → positive difference (pre-heating)
        let delta_cold = vent.temperature_difference(268.15); // -5°C
        assert!(delta_cold > 0.0);
    }
}
