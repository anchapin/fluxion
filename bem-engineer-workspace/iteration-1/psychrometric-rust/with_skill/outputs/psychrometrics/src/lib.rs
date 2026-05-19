//! Psychrometric state calculations for moist air.
//!
//! Reference: ASHRAE Handbook of Fundamentals, Chapter 1 (Psychrometrics).
//! All correlations follow the Hyland-Wexler formulations.
//! Units are SI throughout — temperatures in °C, pressures in Pa, masses in kg.

use std::f64::consts::PI;

const P_ATM: f64 = 101_325.0;
const R_UNIVERSAL: f64 = 8.314_462;
const M_DA: f64 = 0.028_964_5;
const M_W: f64 = 0.018_015_3;
const R_DA: f64 = R_UNIVERSAL / M_DA;
const R_W: f64 = R_UNIVERSAL / M_W;
const EPSILON: f64 = M_W / M_DA;

/// All derived psychrometric properties for a moist air state.
#[derive(Debug, Clone)]
pub struct PsychrometricResult {
    pub t_dry: f64,
    pub pressure: f64,
    pub rh: f64,
    pub p_sat: f64,
    pub p_vapor: f64,
    pub humidity_ratio: f64,
    pub enthalpy: f64,
    pub specific_volume: f64,
    pub dew_point: f64,
    pub wet_bulb: f64,
    pub density: f64,
}

/// Compute all psychrometric properties from dry-bulb temperature,
/// relative humidity, and total pressure.
///
/// # Arguments
/// * `t_db` — Dry-bulb temperature [°C]
/// * `rh` — Relative humidity as a fraction (0.0–1.0)
/// * `pressure` — Total atmospheric pressure [Pa]
pub fn psychrometric_state(t_db: f64, rh: f64, pressure: f64) -> PsychrometricResult {
    assert!((0.0..=1.0).contains(&rh), "RH must be in [0, 1]");
    assert!(pressure > 0.0, "Pressure must be positive");

    let p_sat = saturation_pressure(t_db);
    let p_vapor = rh * p_sat;
    let w = EPSILON * p_vapor / (pressure - p_vapor);
    let enthalpy = 1.006 * t_db + w * (2501.0 + 1.86 * t_db);
    let specific_volume = R_DA * (t_db + 273.15) * (1.0 + w / EPSILON) / pressure;
    let dew_point = compute_dew_point(p_vapor);
    let wet_bulb = compute_wet_bulb(t_db, w, pressure, enthalpy);
    let density = (1.0 + w) / specific_volume;

    PsychrometricResult {
        t_dry: t_db,
        pressure,
        rh,
        p_sat,
        p_vapor,
        humidity_ratio: w,
        enthalpy,
        specific_volume,
        dew_point,
        wet_bulb,
        density,
    }
}

/// ASHRAE HoF Chapter 1, eq. 5 (over liquid water) / eq. 6 (over ice).
/// Returns saturation pressure in Pa.
pub fn saturation_pressure(t_celsius: f64) -> f64 {
    let t_k = t_celsius + 273.15;
    let ln_p = if t_celsius >= 0.0 {
        let c = [
            -5_800.220_6,
            1.391_499_3,
            -0.048_640_239,
            0.417_647_68e-4,
            -0.144_520_93e-7,
        ];
        c[0] / t_k
            + c[1]
            + c[2] * t_k
            + c[3] * t_k * t_k
            + c[4] * t_k * t_k * t_k
            + 6.545_967_3 * t_k.ln()
    } else {
        let c = [
            -5_674.535_9,
            6.392_524_7,
            -0.967_784_3e-2,
            0.622_157_01e-6,
            0.207_478_25e-18,
            -0.948_402_4e-12,
        ];
        c[0] / t_k
            + c[1]
            + c[2] * t_k
            + c[3] * t_k * t_k * t_k
            + c[4] * t_k * t_k * t_k * t_k * t_k
            + c[5] * t_k * t_k * t_k * t_k * t_k * t_k * t_k
            + t_k.ln() * 4.163_501_9
    };
    ln_p.exp()
}

fn saturation_pressure_derivative(t_celsius: f64) -> f64 {
    let dt = 0.001;
    (saturation_pressure(t_celsius + dt) - saturation_pressure(t_celsius - dt)) / (2.0 * dt)
}

fn compute_dew_point(p_vapor: f64) -> f64 {
    let mut t_dp = 20.0_f64;
    for _ in 0..100 {
        let f = saturation_pressure(t_dp) - p_vapor;
        let df = saturation_pressure_derivative(t_dp);
        let dt = f / df;
        t_dp -= dt;
        if dt.abs() < 1e-8 {
            break;
        }
    }
    t_dp
}

fn compute_wet_bulb(t_db: f64, w: f64, pressure: f64, h_target: f64) -> f64 {
    let mut lo = -50.0_f64;
    let mut hi = t_db;
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        let p_sat_wb = saturation_pressure(mid);
        let w_sat = EPSILON * p_sat_wb / (pressure - p_sat_wb);
        let h_sat = 1.006 * mid + w_sat * (2501.0 + 1.86 * mid);
        if h_sat < h_target {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo).abs() < 1e-8 {
            break;
        }
    }
    (lo + hi) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    const T_DB: f64 = 25.0;
    const RH: f64 = 0.50;
    const P: f64 = 101_325.0;

    fn reference_state() -> PsychrometricResult {
        psychrometric_state(T_DB, RH, P)
    }

    #[test]
    fn test_saturation_pressure_100c() {
        let p_sat = saturation_pressure(100.0);
        assert!(
            (p_sat - 101_325.0).abs() / 101_325.0 < 0.005,
            "P_sat at 100°C = {p_sat}, expected ~101325 Pa"
        );
    }

    #[test]
    fn test_saturation_pressure_25c() {
        let p_sat = saturation_pressure(25.0);
        assert!(
            (p_sat - 3169.22).abs() / 3169.22 < 0.005,
            "P_sat at 25°C = {p_sat}, expected ~3169 Pa"
        );
    }

    #[test]
    fn test_saturation_pressure_20c() {
        let p_sat = saturation_pressure(20.0);
        assert!(
            (p_sat - 2339.0).abs() / 2339.0 < 0.005,
            "P_sat at 20°C = {p_sat}, expected ~2339 Pa"
        );
    }

    #[test]
    fn test_vapor_pressure() {
        let s = reference_state();
        let expected_pv = RH * saturation_pressure(T_DB);
        assert!(
            (s.p_vapor - expected_pv).abs() < 0.01,
            "P_v = {}, expected {}",
            s.p_vapor,
            expected_pv
        );
    }

    #[test]
    fn test_humidity_ratio() {
        let s = reference_state();
        assert!(
            (s.humidity_ratio - 0.00988).abs() < 0.0002,
            "W = {}, expected ~0.00988",
            s.humidity_ratio
        );
    }

    #[test]
    fn test_enthalpy() {
        let s = reference_state();
        assert!(
            (s.enthalpy - 50.32).abs() < 0.1,
            "h = {}, expected ~50.32 kJ/kg_da",
            s.enthalpy
        );
    }

    #[test]
    fn test_specific_volume() {
        let s = reference_state();
        assert!(
            (s.specific_volume - 0.858).abs() < 0.005,
            "v = {}, expected ~0.858 m³/kg_da",
            s.specific_volume
        );
    }

    #[test]
    fn test_dew_point() {
        let s = reference_state();
        assert!(
            (s.dew_point - 13.86).abs() < 0.1,
            "T_dp = {}, expected ~13.86°C",
            s.dew_point
        );
        assert!(
            s.dew_point < s.t_dry,
            "Dew point must be below dry-bulb for RH < 1.0"
        );
    }

    #[test]
    fn test_wet_bulb() {
        let s = reference_state();
        assert!(
            (s.wet_bulb - 17.82).abs() < 0.15,
            "T_wb = {}, expected ~17.82°C",
            s.wet_bulb
        );
        assert!(
            s.wet_bulb < s.t_dry,
            "Wet-bulb must be below dry-bulb for RH < 1.0"
        );
        assert!(
            s.wet_bulb > s.dew_point,
            "Wet-bulb must be above dew-point for RH < 1.0"
        );
    }

    #[test]
    fn test_density() {
        let s = reference_state();
        assert!(
            (s.density - 1.177).abs() < 0.01,
            "rho = {}, expected ~1.177 kg/m³",
            s.density
        );
    }

    #[test]
    fn test_mass_balance() {
        let s = reference_state();
        let product = s.specific_volume * s.density;
        assert!(
            (product - (1.0 + s.humidity_ratio)).abs() < 0.01,
            "v * rho = {}, expected {}",
            product,
            1.0 + s.humidity_ratio
        );
    }

    #[test]
    fn test_dry_air_enthalpy_at_zero() {
        let s = psychrometric_state(0.0, 0.0, P);
        assert!(
            s.enthalpy.abs() < 0.001,
            "h(0°C, 0%RH) = {}, expected ~0",
            s.enthalpy
        );
    }

    #[test]
    fn test_dry_air_specific_volume() {
        let s = psychrometric_state(25.0, 0.0, P);
        let v_expected = R_DA * (25.0 + 273.15) / P;
        assert!(
            (s.specific_volume - v_expected).abs() / v_expected < 0.001,
            "v(dry) = {}, expected {}",
            s.specific_volume,
            v_expected
        );
    }

    #[test]
    fn test_saturated_conditions() {
        let s = psychrometric_state(25.0, 1.0, P);
        assert!((s.rh - 1.0).abs() < 0.001, "RH at saturation = {}", s.rh);
        assert!(
            (s.dew_point - 25.0).abs() < 0.05,
            "T_dp at saturation = {}, expected ~25.0",
            s.dew_point
        );
    }
}
