//! HVAC Predictive Control
//!
//! This module provides predictive control strategies using thermal inertia
//! to smooth response and prevent oscillation in high-thermal-mass buildings.

use crate::sim::hvac::HVACMode;
use serde::{Deserialize, Serialize};

/// Predictive HVAC controller using thermal inertia.
///
/// This controller considers thermal mass state and temperature rate of change
/// to smooth response and prevent oscillation, more realistic than simple
/// setpoint hysteresis for high-thermal-mass buildings.
///
/// Physics-based gain derivation (Issue #1614):
/// - α = 1 - exp(-dt/τ) where τ = Cm / h_ms (first-order thermal response)
/// - β = dt · α_diff where α_diff = k / (ρ · cp · L²) (thermal diffusion timescale)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveController {
    /// Heating setpoint (°C)
    pub heating_setpoint: f64,
    /// Cooling setpoint (°C)
    pub cooling_setpoint: f64,
    /// Deadband tolerance (°C) - prevents rapid cycling near setpoints
    pub deadband_tolerance: f64,
    /// Thermal inertia gain factor (α) - derived from τ = Cm/h_ms
    pub thermal_inertia_gain: f64,
    /// Temperature rate gain factor (β) - derived from thermal diffusion
    pub temp_rate_gain: f64,
    /// Previous zone temperature (for calculating dT/dt)
    pub previous_zone_temp: f64,
    /// Thermal capacitance (J/K) - used for gain derivation
    pub cm: f64,
    /// Mass-to-surface heat transfer coefficient (W/K) - used for gain derivation
    pub h_ms: f64,
    /// Timestep (seconds) - used for gain derivation
    pub dt: f64,
}

impl PredictiveController {
    /// Create a new predictive controller with physics-based gains (Issue #1614).
    ///
    /// Gains are derived from thermal parameters rather than empirical tuning:
    /// - α = 1 - exp(-dt/τ) where τ = Cm / h_ms (first-order thermal response)
    /// - β = dt · k / (ρ · cp · L²) (thermal diffusion timescale)
    ///
    /// # Arguments
    /// * `heating_setpoint` - Heating setpoint (°C)
    /// * `cooling_setpoint` - Cooling setpoint (°C)
    /// * `cm` - Thermal capacitance (J/K)
    /// * `h_ms` - Mass-to-surface heat transfer coefficient (W/K)
    /// * `dt` - Timestep (seconds)
    /// * `k` - Thermal conductivity (W/m·K)
    /// * `rho` - Material density (kg/m³)
    /// * `cp` - Specific heat capacity (J/kg·K)
    /// * `l` - Characteristic length/thickness (m)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        heating_setpoint: f64,
        cooling_setpoint: f64,
        cm: f64,
        h_ms: f64,
        dt: f64,
        k: f64,
        rho: f64,
        cp: f64,
        l: f64,
    ) -> Self {
        let (alpha, beta) = Self::compute_gains(cm, h_ms, dt, k, rho, cp, l);
        Self {
            heating_setpoint,
            cooling_setpoint,
            deadband_tolerance: 0.5,
            thermal_inertia_gain: alpha,
            temp_rate_gain: beta,
            previous_zone_temp: 20.0,
            cm,
            h_ms,
            dt,
        }
    }

    /// Compute physics-based gain factors from thermal parameters.
    ///
    /// α = 1 - exp(-dt/τ) where τ = Cm / h_ms is the thermal time constant.
    /// β = dt · α_diff where α_diff = k / (ρ · cp · L²) is the thermal diffusion rate.
    ///
    /// # Arguments
    /// * `cm` - Thermal capacitance (J/K)
    /// * `h_ms` - Mass-to-surface heat transfer coefficient (W/K)
    /// * `dt` - Timestep (seconds)
    /// * `k` - Thermal conductivity (W/m·K)
    /// * `rho` - Material density (kg/m³)
    /// * `cp` - Specific heat capacity (J/kg·K)
    /// * `l` - Characteristic length (m)
    ///
    /// # Returns
    /// Tuple of (α, β) gain factors
    fn compute_gains(cm: f64, h_ms: f64, dt: f64, k: f64, rho: f64, cp: f64, l: f64) -> (f64, f64) {
        // Thermal time constant τ = Cm / h_ms (seconds)
        // For first-order system response over timestep dt:
        // α = 1 - exp(-dt/τ) (discrete-time first-order lag)
        let tau = if h_ms > 0.0 { cm / h_ms } else { f64::INFINITY };
        let alpha = if tau.is_infinite() || tau <= 0.0 {
            0.0
        } else {
            1.0_f64 - (-dt / tau).exp()
        };

        // Thermal diffusion rate α_diff = k / (ρ · cp · L²) [1/s]
        // This represents how quickly heat diffuses through the material.
        // When Cm = 0 or h_ms = 0, there's no thermal coupling, so both gains are zero.
        let alpha_diff = if cm <= 0.0 || h_ms <= 0.0 || rho <= 0.0 || cp <= 0.0 || l <= 0.0 {
            0.0
        } else {
            k / (rho * cp * l * l)
        };
        // β = dt · α_diff (dimensionless rate gain)
        let beta = dt * alpha_diff;

        (alpha, beta)
    }

    /// Create a predictive controller with custom tuning parameters (backward compatibility).
    ///
    /// # Arguments
    /// * `heating_setpoint` - Heating setpoint (°C)
    /// * `cooling_setpoint` - Cooling setpoint (°C)
    /// * `thermal_inertia_gain` - Thermal inertia gain factor (α)
    /// * `temp_rate_gain` - Temperature rate gain factor (β)
    #[allow(dead_code)]
    pub fn with_tuning(
        heating_setpoint: f64,
        cooling_setpoint: f64,
        thermal_inertia_gain: f64,
        temp_rate_gain: f64,
    ) -> Self {
        Self {
            heating_setpoint,
            cooling_setpoint,
            deadband_tolerance: 0.5,
            thermal_inertia_gain,
            temp_rate_gain,
            previous_zone_temp: 20.0,
            cm: 0.0,
            h_ms: 0.0,
            dt: 3600.0,
        }
    }

    /// Compute the effective heating and cooling setpoints after applying the
    /// inertia and predictive corrections.
    ///
    /// Canonical sign convention (issue #1412, EnergyPlus IO Reference
    /// "Zone Thermostat / Predictive Controller"):
    ///
    ///   `inertia_factor  = α · (T_zone − T_mass)`
    ///   `predictive_factor = β · dT/dt`
    ///   `eff_heating_sp  = heating_setpoint  − inertia_factor − predictive_factor`
    ///   `eff_cooling_sp  = cooling_setpoint  − inertia_factor − predictive_factor`
    ///
    /// When the mass is cooler than the zone (`inertia_factor > 0`), the
    /// effective setpoints are lowered: heating triggers earlier (anticipating
    /// the mass absorbing heat and cooling the zone) and cooling defers
    /// (anticipating the mass helping the zone cool). The opposite holds when
    /// the mass is warmer than the zone.
    ///
    /// Both `calculate_modulation` and `calculate_modulation_with_setpoints`
    /// route through this helper, so the two overloads cannot drift apart
    /// (the bug fixed by issue #1412 was the dynamic-setpoint overload using
    /// `+ inertia_factor`, the opposite of the static-setpoint overload).
    fn effective_setpoints(
        &self,
        zone_temp: f64,
        mass_temp: f64,
        temp_rate: f64,
        heating_setpoint: f64,
        cooling_setpoint: f64,
    ) -> (f64, f64) {
        let inertia_factor = self.thermal_inertia_gain * (zone_temp - mass_temp);
        let predictive_factor = self.temp_rate_gain * temp_rate;
        (
            heating_setpoint - inertia_factor - predictive_factor,
            cooling_setpoint - inertia_factor - predictive_factor,
        )
    }

    /// Calculate control signal (mode and modulation factor).
    ///
    /// Uses thermal inertia to predict thermal response and adjust control signal.
    ///
    /// # Arguments
    /// * `zone_temp` - Current zone air temperature (°C)
    /// * `mass_temp` - Thermal mass temperature (°C) from 5R1C network
    /// * `temp_rate` - Rate of temperature change (°C/s), dT/dt
    ///
    /// # Returns
    /// Tuple of (HVACMode, modulation_factor) where:
    /// - `mode`: Heating, Cooling, or Off
    /// - `modulation_factor`: 0.0 to 1.0 (0% to 100% capacity)
    ///
    /// # Control Logic
    /// 1. Calculate inertia factor based on zone temp vs mass temp offset
    /// 2. Calculate predictive factor based on temperature rate
    /// 3. Adjust effective setpoints by inertia and prediction
    /// 4. Determine mode based on zone temp vs adjusted setpoints
    /// 5. Calculate modulation factor based on temperature error
    pub fn calculate_modulation(
        &mut self,
        zone_temp: f64,
        mass_temp: f64,
        temp_rate: f64,
    ) -> (HVACMode, f64) {
        // Step 0: Handle edge cases
        if zone_temp.is_infinite() || zone_temp.is_nan() {
            return (HVACMode::Off, 0.0);
        }

        // Steps 1-3: effective setpoints (canonical sign — see `effective_setpoints`)
        let (effective_heating_sp, effective_cooling_sp) = self.effective_setpoints(
            zone_temp,
            mass_temp,
            temp_rate,
            self.heating_setpoint,
            self.cooling_setpoint,
        );

        // Step 4: Determine mode based on zone temp vs adjusted setpoints
        // Apply deadband tolerance to prevent cycling
        let heating_threshold = effective_heating_sp - self.deadband_tolerance;
        let cooling_threshold = effective_cooling_sp + self.deadband_tolerance;

        let mode = if zone_temp < heating_threshold {
            HVACMode::Heating
        } else if zone_temp > cooling_threshold {
            HVACMode::Cooling
        } else {
            HVACMode::Off
        };

        // Step 5: Calculate modulation factor based on temperature error
        let temp_error = match mode {
            HVACMode::Heating => effective_heating_sp - zone_temp,
            HVACMode::Cooling => zone_temp - effective_cooling_sp,
            HVACMode::Off => 0.0,
        };

        // Modulation factor: 0-1 based on temperature error
        // Sensitivity: 10.0 means 1°C error = 10% modulation
        let modulation = (temp_error * 10.0).clamp(0.0, 1.0);

        // Update previous zone temp for next timestep's dT/dt calculation
        self.previous_zone_temp = zone_temp;

        (mode, modulation)
    }

    /// Calculate control signal with dynamic setpoints (supports setback schedules).
    ///
    /// This variant allows passing time-varying setpoints, which is needed for
    /// setback schedules where the setpoint changes at different hours.
    ///
    /// Sign convention on the inertia and predictive contributions is identical
    /// to `calculate_modulation` — both overloads route through the private
    /// `effective_setpoints` helper, so the two cannot drift apart (issue #1412).
    pub fn calculate_modulation_with_setpoints(
        &mut self,
        zone_temp: f64,
        mass_temp: f64,
        temp_rate: f64,
        heating_setpoint: f64,
        cooling_setpoint: f64,
    ) -> (HVACMode, f64) {
        if zone_temp.is_infinite() || zone_temp.is_nan() {
            return (HVACMode::Off, 0.0);
        }

        // Steps 1-3: effective setpoints (canonical sign — see `effective_setpoints`)
        let (effective_heating_sp, effective_cooling_sp) = self.effective_setpoints(
            zone_temp,
            mass_temp,
            temp_rate,
            heating_setpoint,
            cooling_setpoint,
        );

        // Step 4: Determine mode based on zone temp vs adjusted setpoints
        let heating_threshold = effective_heating_sp - self.deadband_tolerance;
        let cooling_threshold = effective_cooling_sp + self.deadband_tolerance;

        let mode = if zone_temp < heating_threshold {
            HVACMode::Heating
        } else if zone_temp > cooling_threshold {
            HVACMode::Cooling
        } else {
            HVACMode::Off
        };

        // Step 5: Calculate modulation factor based on temperature error
        let temp_error = match mode {
            HVACMode::Heating => effective_heating_sp - zone_temp,
            HVACMode::Cooling => zone_temp - effective_cooling_sp,
            HVACMode::Off => 0.0,
        };

        let modulation = (temp_error * 10.0).clamp(0.0, 1.0);

        // Update previous zone temp for next timestep's dT/dt calculation
        self.previous_zone_temp = zone_temp;

        (mode, modulation)
    }

    /// Reset controller state (for new simulation or year boundary).
    pub fn reset(&mut self) {
        self.previous_zone_temp = 20.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =============================================================================
    // Physics-based gain derivation tests (Issue #1614)
    // =============================================================================

    #[test]
    fn test_physics_rc_circuit_tau_1h() {
        // Issue #1614: RC circuit test for τ=1h, R=100 K/W, C=36 kJ/K
        // τ = R × C = 100 × 36000 = 3,600,000 s = 1000 h
        // But the issue says τ=1h, so we use Cm/h_ms to get τ=1h directly
        //
        // For τ=1h = 3600s with dt=1h = 3600s:
        // α = 1 - exp(-dt/τ) = 1 - exp(-1) ≈ 0.632
        let dt = 3600.0; // 1 hour in seconds
        let tau = 3600.0; // 1 hour in seconds

        // Compute gains using physics formulas
        let cm = 36000.0; // 36 kJ/K = 36000 J/K
        let h_ms = cm / tau; // h_ms = Cm/τ = 10 W/K

        // Concrete properties for β calculation
        let k = 1.7; // W/m·K (concrete thermal conductivity)
        let rho = 2300.0; // kg/m³ (concrete density)
        let cp = 1000.0; // J/kg·K (concrete specific heat)
        let l = 0.2; // m (characteristic length)

        let (alpha, beta) = PredictiveController::compute_gains(cm, h_ms, dt, k, rho, cp, l);

        // α = 1 - exp(-dt/τ) = 1 - exp(-1) ≈ 0.632
        let expected_alpha = 1.0_f64 - (-1.0_f64).exp();
        assert!(
            (alpha - expected_alpha).abs() < 0.001,
            "α = {} expected ≈ {}",
            alpha,
            expected_alpha
        );

        // Verify the controller produces correct anticipation factor
        let controller = PredictiveController::new(20.0, 27.0, cm, h_ms, dt, k, rho, cp, l);
        assert!((controller.thermal_inertia_gain - expected_alpha).abs() < 0.001);
    }

    #[test]
    fn test_physics_thermal_diffusion_timescale() {
        // Issue #1614: Thermal diffusion timescale test
        // α_diff = k / (ρ · cp · L²)
        // For concrete: α_diff = 1.7 / (2300 × 1000 × 0.2²) ≈ 1.85e-5 s⁻¹
        let k = 1.7; // W/m·K
        let rho = 2300.0; // kg/m³
        let cp = 1000.0; // J/kg·K
        let l = 0.2; // m
        let dt = 3600.0; // 1 hour

        let alpha_diff_expected = k / (rho * cp * l * l);
        let beta_expected = dt * alpha_diff_expected;

        let cm = 36000.0; // 36 kJ/K
        let h_ms = cm / 3600.0; // 10 W/K for τ=1h

        let (_alpha, beta) = PredictiveController::compute_gains(cm, h_ms, dt, k, rho, cp, l);

        // Verify thermal diffusion coefficient
        let alpha_diff_actual = beta / dt;
        assert!(
            (alpha_diff_actual - alpha_diff_expected).abs() < 1e-10,
            "α_diff = {} expected ≈ {}",
            alpha_diff_actual,
            alpha_diff_expected
        );

        // Verify β = dt × α_diff
        assert!(
            (beta - beta_expected).abs() < 1e-10,
            "β = {} expected ≈ {}",
            beta,
            beta_expected
        );
    }

    #[test]
    fn test_physics_gain_derivation_known_rc() {
        // Test with known RC parameters: τ=2h, R=50 K/W, C=72 kJ/K
        // τ = R × C = 50 × 72000 = 3,600,000 s = 1000 h (still doesn't match)
        // Let me use direct τ computation instead

        let dt = 3600.0; // 1 hour
        let tau = 7200.0; // 2 hours

        // For τ = Cm/h_ms, if we want τ=2h:
        let cm = 72000.0; // 72 kJ/K
        let h_ms = cm / tau; // h_ms = 10 W/K

        // Concrete properties
        let k = 1.7;
        let rho = 2300.0;
        let cp = 1000.0;
        let l = 0.2;

        let (alpha, _) = PredictiveController::compute_gains(cm, h_ms, dt, k, rho, cp, l);

        // α = 1 - exp(-dt/τ) = 1 - exp(-0.5) ≈ 0.393
        let expected_alpha = 1.0_f64 - (-0.5_f64).exp();
        assert!(
            (alpha - expected_alpha).abs() < 0.001,
            "α = {} expected ≈ {}",
            alpha,
            expected_alpha
        );
    }

    // =============================================================================
    // Backward compatibility tests (using with_tuning)
    // =============================================================================

    #[test]
    fn test_predictive_control() {
        // Use with_tuning for backward compatibility (no thermal params needed)
        let mut controller = PredictiveController::with_tuning(20.0, 27.0, 0.1, 0.01);

        // Test heating mode
        let (mode, modulation) = controller.calculate_modulation(18.0, 19.0, -0.001);
        assert_eq!(mode, HVACMode::Heating);
        assert!(modulation > 0.0); // Some modulation needed

        // Test cooling mode
        let (mode, modulation) = controller.calculate_modulation(28.0, 27.0, 0.001);
        assert_eq!(mode, HVACMode::Cooling);
        assert!(modulation > 0.0);

        // Test off mode (within deadband)
        let (mode, modulation) = controller.calculate_modulation(23.0, 22.0, 0.0);
        assert_eq!(mode, HVACMode::Off);
        assert_eq!(modulation, 0.0);

        // Test modulation factor limits (0-1)
        let (_, modulation_high) = controller.calculate_modulation(15.0, 20.0, -0.01);
        assert!(modulation_high <= 1.0);

        let (_, modulation_low) = controller.calculate_modulation(30.0, 25.0, 0.01);
        assert!(modulation_low <= 1.0);
    }

    #[test]
    fn test_thermal_inertia() {
        let mut controller = PredictiveController::with_tuning(20.0, 27.0, 0.1, 0.01);

        // Test with thermal inertia (mass temp cooler than zone temp)
        // This should anticipate cooling and adjust setpoint upward
        let (mode_no_inertia, _) = controller.calculate_modulation(29.0, 29.0, 0.0);

        let (mode_with_inertia, _) = controller.calculate_modulation(29.0, 19.0, 0.0); // Mass temp 10°C cooler

        // Both should be cooling mode
        assert_eq!(mode_no_inertia, HVACMode::Cooling);
        assert_eq!(mode_with_inertia, HVACMode::Cooling);
    }

    #[test]
    fn test_temperature_rate_prediction() {
        let mut controller = PredictiveController::with_tuning(20.0, 27.0, 0.1, 0.01);

        // Test with rising temperature (anticipate overshoot)
        // This should reduce modulation to prevent overshoot
        let (mode_stable, _) = controller.calculate_modulation(29.0, 29.0, 0.0);

        let (mode_rising, _) = controller.calculate_modulation(29.0, 29.0, 0.01); // Rising at 0.01°C/s

        // Both should be cooling mode
        assert_eq!(mode_stable, HVACMode::Cooling);
        assert_eq!(mode_rising, HVACMode::Cooling);
    }

    #[test]
    fn test_deadband_tolerance() {
        let mut controller = PredictiveController::with_tuning(20.0, 27.0, 0.1, 0.01);

        // At heating setpoint (within deadband)
        let (mode, modulation) = controller.calculate_modulation(20.0, 20.0, 0.0);
        assert_eq!(mode, HVACMode::Off);
        assert_eq!(modulation, 0.0);

        // At cooling setpoint (within deadband)
        let (mode, modulation) = controller.calculate_modulation(27.0, 27.0, 0.0);
        assert_eq!(mode, HVACMode::Off);
        assert_eq!(modulation, 0.0);

        // Below heating setpoint - deadband
        let (mode, modulation) = controller.calculate_modulation(19.0, 19.0, 0.0);
        assert_eq!(mode, HVACMode::Heating);
        assert!(modulation > 0.0);

        // Above cooling setpoint + deadband
        let (mode, modulation) = controller.calculate_modulation(28.0, 28.0, 0.0);
        assert_eq!(mode, HVACMode::Cooling);
        assert!(modulation > 0.0);
    }

    #[test]
    fn test_controller_reset() {
        let mut controller = PredictiveController::with_tuning(20.0, 27.0, 0.1, 0.01);

        // Run a few timesteps
        controller.calculate_modulation(22.0, 21.0, 0.001);
        controller.calculate_modulation(23.0, 22.0, 0.001);

        // Reset
        controller.reset();

        // Previous zone temp should be reset to initial value
        assert_eq!(controller.previous_zone_temp, 20.0);
    }

    #[test]
    fn test_custom_tuning() {
        // Create controller with custom tuning parameters
        let controller = PredictiveController::with_tuning(20.0, 27.0, 0.2, 0.02);

        assert_eq!(controller.thermal_inertia_gain, 0.2);
        assert_eq!(controller.temp_rate_gain, 0.02);
        assert_eq!(controller.heating_setpoint, 20.0);
        assert_eq!(controller.cooling_setpoint, 27.0);
    }

    #[test]
    fn test_inertia_factor_calculation() {
        let controller = PredictiveController::with_tuning(20.0, 27.0, 0.1, 0.01);

        let zone_temp = 22.0;
        let mass_temp = 18.0; // 4°C cooler

        // Inertia factor = α × (zone_temp - mass_temp)
        // = 0.1 × (22.0 - 18.0) = 0.4
        let expected_inertia_factor = controller.thermal_inertia_gain * (zone_temp - mass_temp);
        assert!((expected_inertia_factor - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_predictive_factor_calculation() {
        let controller = PredictiveController::with_tuning(20.0, 27.0, 0.1, 0.01);

        let temp_rate = 0.01; // Rising at 0.01°C/s

        // Predictive factor = β × temp_rate
        // = 0.01 × 0.01 = 0.0001
        let expected_predictive_factor = controller.temp_rate_gain * temp_rate;
        assert!((expected_predictive_factor - 0.0001).abs() < 0.00001);
    }

    #[test]
    fn test_calculate_modulation_with_setpoints() {
        let mut controller = PredictiveController::with_tuning(20.0, 27.0, 0.1, 0.01);

        // Test with dynamic setpoints (e.g. night setback)
        let (mode, modulation) =
            controller.calculate_modulation_with_setpoints(16.0, 17.0, 0.0, 15.0, 30.0);

        // zone_temp (16.0) is above heating threshold (15.0 - inertia - deadband)
        // inertia = 0.1 * (16-17) = -0.1
        // effective_heating_sp = 15.0 - 0.1 - 0.0 = 14.9
        // heating_threshold = 14.9 - 0.5 = 14.4
        // 16.0 > 14.4, so should be OFF
        assert_eq!(mode, HVACMode::Off);
        assert_eq!(modulation, 0.0);

        // Test heating with dynamic setpoint
        let (mode2, modulation2) =
            controller.calculate_modulation_with_setpoints(14.0, 14.0, 0.0, 15.0, 30.0);
        assert_eq!(mode2, HVACMode::Heating);
        assert!(modulation2 > 0.0);
    }

    #[test]
    fn test_physics_alpha_with_different_timestep() {
        // Test that α varies correctly with timestep
        let cm = 36000.0; // 36 kJ/K
        let h_ms = 10.0; // gives τ = 3600s = 1h
        let k = 1.7;
        let rho = 2300.0;
        let cp = 1000.0;
        let l = 0.2;

        // dt = 30 minutes = 1800s, τ = 3600s
        let (alpha_30min, _) = PredictiveController::compute_gains(cm, h_ms, 1800.0, k, rho, cp, l);
        // α = 1 - exp(-0.5) ≈ 0.393
        let expected_30min = 1.0_f64 - (-0.5_f64).exp();
        assert!((alpha_30min - expected_30min).abs() < 0.001);

        // dt = 1 hour = 3600s, τ = 3600s
        let (alpha_1h, _) = PredictiveController::compute_gains(cm, h_ms, 3600.0, k, rho, cp, l);
        // α = 1 - exp(-1) ≈ 0.632
        let expected_1h = 1.0_f64 - (-1.0_f64).exp();
        assert!((alpha_1h - expected_1h).abs() < 0.001);

        // dt = 2 hours = 7200s, τ = 3600s
        let (alpha_2h, _) = PredictiveController::compute_gains(cm, h_ms, 7200.0, k, rho, cp, l);
        // α = 1 - exp(-2) ≈ 0.865
        let expected_2h = 1.0_f64 - (-2.0_f64).exp();
        assert!((alpha_2h - expected_2h).abs() < 0.001);

        // Verify α increases with larger dt relative to τ
        assert!(alpha_30min < alpha_1h);
        assert!(alpha_1h < alpha_2h);
    }

    #[test]
    fn test_physics_degenerate_cases() {
        // Test handling of degenerate cases
        let k = 1.7;
        let rho = 2300.0;
        let cp = 1000.0;
        let l = 0.2;

        // h_ms = 0 (infinite time constant)
        let (alpha, beta) =
            PredictiveController::compute_gains(36000.0, 0.0, 3600.0, k, rho, cp, l);
        assert_eq!(alpha, 0.0); // No thermal response
        assert_eq!(beta, 0.0);

        // Cm = 0 (instantaneous thermal response)
        let (alpha, beta) = PredictiveController::compute_gains(0.0, 10.0, 3600.0, k, rho, cp, l);
        assert_eq!(alpha, 0.0); // Instant response
        assert_eq!(beta, 0.0);

        // Negative h_ms (physically impossible)
        let (alpha, beta) =
            PredictiveController::compute_gains(36000.0, -10.0, 3600.0, k, rho, cp, l);
        assert_eq!(alpha, 0.0);
        assert_eq!(beta, 0.0);
    }
}
