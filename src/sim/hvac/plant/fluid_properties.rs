//! Fluid property tables for plant-loop working fluids.
//!
//! Provides temperature-dependent density and specific heat for water and
//! 30 % propylene-glycol solution (common antifreeze for chilled-water
//! systems).  All values use SI units and are sourced from ASHRAE
//! Handbook — Fundamentals 2021, Chapter 32.

/// Density of water at 4 °C (maximum).
pub const WATER_DENSITY_KG_PER_M3: f64 = 999.9;

/// Specific heat of water at 25 °C.
pub const WATER_CP_J_PER_KG_K: f64 = 4_186.0;

/// Thermal conductivity of water at 25 °C [W/(m·K)].
pub const WATER_K_W_PER_M_K: f64 = 0.606;

/// Dynamic viscosity of water at 25 °C [Pa·s].
pub const WATER_MU_PA_S: f64 = 8.9e-4;

/// Density of 30 % propylene-glycol solution at 25 °C [kg/m³].
pub const PG30_DENSITY_KG_PER_M3: f64 = 1_038.0;

/// Specific heat of 30 % propylene-glycol solution at 25 °C [J/(kg·K)].
pub const PG30_CP_J_PER_KG_K: f64 = 3_849.0;

/// Evaluate water density as a function of temperature (polynomial fit,
/// ASHRAE HOF 2021 Table 32, range 0–100 °C).
///
/// `ρ(T) = a₀ + a₁T + a₂T²`  where T is in °C.
pub fn water_density(temp_c: f64) -> f64 {
    // Quadratic fit to ASHRAE HOF 2021 Table 32 values
    let a0 = 1_000.18;
    let a1 = -0.00436;
    let a2 = -0.000_008_2;
    (a0 + a1 * temp_c + a2 * temp_c * temp_c).max(958.0)
}

/// Evaluate water specific heat as a function of temperature (polynomial
/// fit, ASHRAE HOF 2021 Table 32, range 0–100 °C).
pub fn water_cp(temp_c: f64) -> f64 {
    let a0 = 4_217.0;
    let a1 = -2.81;
    let a2 = 0.065;
    (a0 + a1 * temp_c + a2 * temp_c * temp_c).min(4_220.0)
}

/// Evaluate 30 % propylene-glycol density [kg/m³].
pub fn pg30_density(temp_c: f64) -> f64 {
    let a0 = 1_053.0;
    let a1 = -0.235;
    (a0 + a1 * temp_c).max(990.0)
}

/// Evaluate 30 % propylene-glycol specific heat [J/(kg·K)].
pub fn pg30_cp(temp_c: f64) -> f64 {
    let a0 = 3_920.0;
    let a1 = -2.93;
    (a0 + a1 * temp_c).max(3_600.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_water_density_at_4c() {
        let rho = water_density(4.0);
        assert!(
            (rho - WATER_DENSITY_KG_PER_M3).abs() < 2.0,
            "water_density(4) = {rho}, expected ~{WATER_DENSITY_KG_PER_M3}"
        );
    }

    #[test]
    fn test_water_cp_at_25c() {
        let cp = water_cp(25.0);
        assert!(
            (cp - WATER_CP_J_PER_KG_K).abs() < 20.0,
            "water_cp(25) = {cp}, expected ~{WATER_CP_J_PER_KG_K}"
        );
    }

    #[test]
    fn test_water_density_positive() {
        for t in 0..=100 {
            let rho = water_density(t as f64);
            assert!(rho > 900.0 && rho < 1100.0, "ρ({t}) = {rho}");
        }
    }

    #[test]
    fn test_pg30_density_finite() {
        for t in (-10)..=60 {
            let rho = pg30_density(t as f64);
            assert!(rho.is_finite() && rho > 900.0, "pg30_density({t}) = {rho}");
        }
    }

    #[test]
    fn test_pg30_cp_finite() {
        for t in (-10)..=60 {
            let cp = pg30_cp(t as f64);
            assert!(cp.is_finite() && cp > 3000.0, "pg30_cp({t}) = {cp}");
        }
    }
}
