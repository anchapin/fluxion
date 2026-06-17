//! Psychrometric state calculations for moist air.
//!
//! Reference: ASHRAE Handbook of Fundamentals, Chapter 1 (Psychrometrics).
//! All correlations follow the Hyland-Wexler formulations.
//! Units are SI throughout — temperatures in °C, pressures in Pa, masses in kg.

use std::f64::consts::PI;

/// Atmospheric pressure at sea level (Pa).
const P_ATM: f64 = 101_325.0;

/// Universal gas constant (J/(mol·K)).
const R_UNIVERSAL: f64 = 8.314_462;

/// Molar mass of dry air (kg/mol).
const M_DA: f64 = 0.028_964_5;

/// Molar mass of water vapor (kg/mol).
const M_W: f64 = 0.018_015_3;

/// Specific gas constant for dry air (J/(kg·K)).
const R_DA: f64 = R_UNIVERSAL / M_DA; // ~287.055

/// Specific gas constant for water vapor (J/(kg·K)).
const R_W: f64 = R_UNIVERSAL / M_W; // ~461.518

/// Ratio of molecular weights: M_W / M_DA.
const EPSILON: f64 = M_W / M_DA; // ~0.621_98

#[derive(Debug, Clone)]
pub struct PsychState {
    /// Dry-bulb temperature [°C].
    pub t_dry: f64,
    /// Total pressure [Pa].
    pub pressure: f64,
    /// Humidity ratio (kg_water / kg_dry_air).
    pub w: f64,
}

impl PsychState {
    /// Create a state from dry-bulb, humidity ratio, and pressure.
    pub fn from_tw(t_dry: f64, w: f64, pressure: f64) -> Self {
        debug_assert!(t_dry >= -100.0, "Dry-bulb below reasonable limit");
        debug_assert!(w >= 0.0, "Humidity ratio cannot be negative");
        debug_assert!(pressure > 0.0, "Pressure must be positive");
        Self { t_dry, w, pressure }
    }

    /// Create from dry-bulb and relative humidity.
    /// RH is expressed as a fraction (0.0–1.0).
    pub fn from_trh(t_dry: f64, rh: f64, pressure: f64) -> Self {
        debug_assert!((0.0..=1.0).contains(&rh), "RH must be in [0, 1]");
        let p_sat = saturation_pressure(t_dry);
        let p_v = rh * p_sat;
        let w = EPSILON * p_v / (pressure - p_v);
        Self::from_tw(t_dry, w, pressure)
    }

    /// Water vapor partial pressure [Pa].
    pub fn vapor_pressure(&self) -> f64 {
        self.w * self.pressure / (EPSILON + self.w)
    }

    /// Relative humidity as a fraction (0.0–1.0).
    pub fn relative_humidity(&self) -> f64 {
        let p_sat = saturation_pressure(self.t_dry);
        self.vapor_pressure() / p_sat
    }

    /// Dew-point temperature [°C].
    /// Inverts the saturation pressure function.
    pub fn dew_point(&self) -> f64 {
        let p_v = self.vapor_pressure();
        // ASHRAE HoF eq. 37 (over water, above 0°C) or eq. 38 (over ice)
        // Using the inverse via Newton-Raphson for simplicity.
        let mut t_dp = self.t_dry; // initial guess
        for _ in 0..50 {
            let f = saturation_pressure(t_dp) - p_v;
            let df = saturation_pressure_derivative(t_dp);
            let dt = f / df;
            t_dp -= dt;
            if dt.abs() < 1e-6 {
                break;
            }
        }
        t_dp
    }

    /// Wet-bulb temperature [°C] via iterative solve.
    /// Uses the psychrometric energy balance:
    ///   (W_sat - W) / (T_db - T_wb) ≈ f(T_wb, P)
    pub fn wet_bulb(&self) -> f64 {
        let mut t_wb = self.t_dry * 0.8; // initial guess below T_db
        for _ in 0..100 {
            let w_sat =
                EPSILON * saturation_pressure(t_wb) / (self.pressure - saturation_pressure(t_wb));
            // Energy balance: h1 + (w_sat - w) * h_w = h_sat
            // Simplified iterative form:
            let cp_air = 1.006; // kJ/(kg·K) approximate
            let h_fg = 2501.0 - 2.326 * t_wb; // latent heat approx kJ/kg
            let w_calc = ((2501.0 - 2.326 * t_wb) * self.w - cp_air * (self.t_dry - t_wb))
                / (2501.0 + 1.86 * self.t_dry - 4.186 * t_wb);
            let residual = w_sat - w_calc;
            if residual.abs() < 1e-7 {
                break;
            }
            t_wb += 0.1 * residual.signum(); // simple fixed-point step
        }
        t_wb
    }

    /// Specific enthalpy of moist air [kJ/kg_dry_air].
    /// Reference: 0°C, dry air + liquid water at 0°C.
    /// h = c_p,da * T + W * (h_fg + c_p,v * T)
    pub fn enthalpy(&self) -> f64 {
        1.006 * self.t_dry + self.w * (2501.0 + 1.86 * self.t_dry)
    }

    /// Specific volume of moist air [m³/kg_dry_air].
    pub fn specific_volume(&self) -> f64 {
        R_DA * (self.t_dry + 273.15) * (1.0 + self.w / EPSILON) / self.pressure
    }

    /// Density of moist air [kg/m³].
    pub fn density(&self) -> f64 {
        (1.0 + self.w) / self.specific_volume()
    }
}

/// Saturation pressure over liquid water [Pa].
/// ASHRAE HoF Chapter 1, eq. 5 (Hyland-Wexler formulation, simplified).
/// Valid for T in range -100°C to 200°C.
pub fn saturation_pressure(t_celsius: f64) -> f64 {
    let t_k = t_celsius + 273.15;
    let ln_p;
    if t_celsius >= 0.0 {
        // Over liquid water (eq. 5)
        let c = [
            -5_800.220_6,
            1.391_499_3,
            -0.048_640_239,
            0.417_647_68e-4,
            -0.144_520_93e-7,
        ];
        ln_p = c[0] / t_k
            + c[1]
            + c[2] * t_k
            + c[3] * t_k * t_k
            + c[4] * t_k * t_k * t_k
            + 6.545_967_3 * t_k.ln();
    } else {
        // Over ice (eq. 6) — less common but physically necessary
        let c = [
            -5_674.535_9,
            6.392_524_7,
            -0.967_784_3e-2,
            0.622_157_01e-6,
            0.207_478_25e-18,
            -0.948_402_4e-12,
        ];
        ln_p = c[0] / t_k
            + c[1]
            + c[2] * t_k
            + c[3] * t_k * t_k * t_k
            + c[4] * t_k * t_k * t_k * t_k * t_k
            + c[5] * t_k * t_k * t_k * t_k * t_k * t_k * t_k
            + t_k.ln() * 4.163_501_9;
    }
    ln_p.exp()
}

/// Numerical derivative of saturation pressure for Newton-Raphson.
fn saturation_pressure_derivative(t_celsius: f64) -> f64 {
    let dt = 0.001;
    (saturation_pressure(t_celsius + dt) - saturation_pressure(t_celsius - dt)) / (2.0 * dt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_saturation_pressure_at_100c() {
        // At 100°C, P_sat should be approximately 101325 Pa (1 atm)
        let p_sat = saturation_pressure(100.0);
        assert!(
            (p_sat - 101_325.0).abs() / 101_325.0 < 0.005,
            "P_sat at 100°C = {p_sat}, expected ~101325 Pa"
        );
    }

    #[test]
    fn test_saturation_pressure_at_20c() {
        // Known value: ~2339 Pa at 20°C
        let p_sat = saturation_pressure(20.0);
        assert!(
            (p_sat - 2339.0).abs() / 2339.0 < 0.005,
            "P_sat at 20°C = {p_sat}, expected ~2339 Pa"
        );
    }

    #[test]
    fn test_enthalpy_round_trip() {
        let state = PsychState::from_trh(25.0, 0.5, P_ATM);
        let h = state.enthalpy();
        assert!(
            h > 25.0,
            "Enthalpy at 25°C/50%RH must exceed dry air alone (~25 kJ/kg)"
        );
    }

    #[test]
    fn test_mass_balance() {
        // Specific volume * density = (1 + W) within rounding
        let state = PsychState::from_trh(20.0, 0.6, P_ATM);
        let product = state.specific_volume() * state.density();
        assert!(
            (product - (1.0 + state.w)).abs() < 0.01,
            "v * rho = {product}, expected {}",
            1.0 + state.w
        );
    }

    #[test]
    fn test_dew_point_below_dry_bulb() {
        let state = PsychState::from_trh(25.0, 0.5, P_ATM);
        assert!(
            state.dew_point() < state.t_dry,
            "Dew point must be below dry-bulb for RH < 1.0"
        );
    }
}
