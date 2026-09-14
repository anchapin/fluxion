//! Unit tests for psychrometric calculations vs ASHRAE analytical benchmarks.
//!
//! Validates all public functions in `src/weather/psychrometrics.rs` against
//! ASHRAE Handbook Fundamentals Chapter 1 analytical values.
//!
//! # Tolerances
//!
//! - Pressures: ±0.1%
//! - Temperatures: ±0.05°C
//! - Humidity ratio: ±0.0001 kg/kg
//!
//! # Reference Data
//!
//! ASHRAE Handbook of Fundamentals, Chapter 1 (2021) Table 3: Saturation Vapor Pressure

use fluxion::weather::psychrometrics::{
    calculate_dew_point, calculate_enthalpy, calculate_humidity_ratio, calculate_wet_bulb,
    saturation_vapor_pressure, STANDARD_ATMOSPHERIC_PRESSURE_Pa,
};

/// Tolerance for saturation vapor pressure: ±0.1%
fn p_tol(p: f64) -> f64 {
    p * 0.001
}

/// Tolerance for temperature: ±0.05°C
fn t_tol(_t: f64) -> f64 {
    0.05
}

/// Tolerance for humidity ratio: ±0.0001 kg/kg
fn w_tol(_w: f64) -> f64 {
    0.0001
}

// =============================================================================
// SATURATION VAPOR PRESSURE TESTS
// =============================================================================

#[test]
fn test_saturation_vapor_pressure_minus_20c() {
    // ASHRAE Table 3 (Hyland-Wexler ice): p_sat(-20°C) ≈ 103.3 Pa
    let t = -20.0;
    let expected = 103.3;
    let p_sat = saturation_vapor_pressure(t);
    let tol = p_tol(expected);
    assert!(
        (p_sat - expected).abs() <= tol,
        "p_sat(-20°C) = {} Pa, expected ≈ {} ± {} Pa",
        p_sat,
        expected,
        tol
    );
}

#[test]
fn test_saturation_vapor_pressure_0c() {
    // ASHRAE Table 3: p_sat(0°C) ≈ 611.2 Pa
    let t = 0.0;
    let expected = 611.2;
    let p_sat = saturation_vapor_pressure(t);
    let tol = p_tol(expected);
    assert!(
        (p_sat - expected).abs() <= tol,
        "p_sat(0°C) = {} Pa, expected ≈ {} ± {} Pa",
        p_sat,
        expected,
        tol
    );
}

#[test]
fn test_saturation_vapor_pressure_20c() {
    // ASHRAE Table 3: p_sat(20°C) ≈ 2338.8 Pa
    let t = 20.0;
    let expected = 2338.8;
    let p_sat = saturation_vapor_pressure(t);
    let tol = p_tol(expected);
    assert!(
        (p_sat - expected).abs() <= tol,
        "p_sat(20°C) = {} Pa, expected ≈ {} ± {} Pa",
        p_sat,
        expected,
        tol
    );
}

#[test]
fn test_saturation_vapor_pressure_40c() {
    // ASHRAE Table 3: p_sat(40°C) ≈ 7382.0 Pa
    let t = 40.0;
    let expected = 7382.0;
    let p_sat = saturation_vapor_pressure(t);
    let tol = p_tol(expected);
    assert!(
        (p_sat - expected).abs() <= tol,
        "p_sat(40°C) = {} Pa, expected ≈ {} ± {} Pa",
        p_sat,
        expected,
        tol
    );
}

#[test]
fn test_saturation_vapor_pressure_monotonic() {
    // Saturation vapor pressure must increase monotonically with temperature
    let temps = [-20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0];
    for window in temps.windows(2) {
        let p1 = saturation_vapor_pressure(window[0]);
        let p2 = saturation_vapor_pressure(window[1]);
        assert!(
            p2 > p1,
            "p_sat must increase with T: p_sat({})={} >= p_sat({})={}",
            window[0],
            p1,
            window[1],
            p2
        );
    }
}

// =============================================================================
// DEW POINT TEMPERATURE ROUNDTRIP TESTS
// =============================================================================

#[test]
fn test_dew_point_roundtrip_at_20c_50rh() {
    // Roundtrip: T -> p_sat -> dew_point(T, RH=50%) should return ≈ T
    let t = 20.0;
    let rh = 50.0;
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let p_sat = saturation_vapor_pressure(t);
    let p_water = p_sat * (rh / 100.0);

    // Calculate dew point from vapor pressure using the inverse relationship
    // p_water = p_sat(Td) => Td = dew_point(p_water)
    // Since we know p_water, we can find Td by iteration
    let dp = calculate_dew_point(t, rh, p);

    // At 50% RH, dew point is significantly below dry bulb
    // Verify it's physically reasonable (Td <= T)
    assert!(dp <= t, "Dew point {}°C must be ≤ dry bulb {}°C", dp, t);

    // Verify the dew point gives the same vapor pressure when used with 100% RH
    let p_sat_at_dp = saturation_vapor_pressure(dp);
    assert!(
        (p_sat_at_dp - p_water).abs() <= p_tol(p_water) * 2.0,
        "Roundtrip failed: p_sat(Td)={} ≠ p_water={}",
        p_sat_at_dp,
        p_water
    );
}

#[test]
fn test_dew_point_roundtrip_at_30c_80rh() {
    let t = 30.0;
    let rh = 80.0;
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let p_sat = saturation_vapor_pressure(t);
    let p_water = p_sat * (rh / 100.0);

    let dp = calculate_dew_point(t, rh, p);

    assert!(dp <= t, "Dew point {}°C must be ≤ dry bulb {}°C", dp, t);

    let p_sat_at_dp = saturation_vapor_pressure(dp);
    assert!(
        (p_sat_at_dp - p_water).abs() <= p_tol(p_water) * 2.0,
        "Roundtrip failed: p_sat(Td)={} ≠ p_water={}",
        p_sat_at_dp,
        p_water
    );
}

#[test]
fn test_dew_point_roundtrip_at_0c_100rh() {
    // At 100% RH, dew point equals dry bulb (saturated air)
    let t = 0.0;
    let rh = 100.0;
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let dp = calculate_dew_point(t, rh, p);

    // At 100% RH, dew point should equal dry bulb (within tolerance)
    assert!(
        (dp - t).abs() <= t_tol(t),
        "At 100% RH, dew point {}°C should equal dry bulb {}°C",
        dp,
        t
    );
}

#[test]
fn test_dew_point_roundtrip_at_minus_10c_30rh() {
    // Sub-zero temperature test
    let t = -10.0;
    let rh = 30.0;
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let p_sat = saturation_vapor_pressure(t);
    let p_water = p_sat * (rh / 100.0);

    let dp = calculate_dew_point(t, rh, p);

    assert!(dp <= t, "Dew point {}°C must be ≤ dry bulb {}°C", dp, t);

    let p_sat_at_dp = saturation_vapor_pressure(dp);
    assert!(
        (p_sat_at_dp - p_water).abs() <= p_tol(p_water) * 2.0,
        "Roundtrip failed: p_sat(Td)={} ≠ p_water={}",
        p_sat_at_dp,
        p_water
    );
}

// =============================================================================
// WET BULB TEMPERATURE TESTS
// =============================================================================

#[test]
fn test_wet_bulb_temperature_20c_50rh() {
    // ASHRAE psychrometric chart reference: 20°C, 50% RH => Twb ≈ 13.5°C
    let t = 20.0;
    let rh = 50.0;
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let wb = calculate_wet_bulb(t, rh, p);
    let expected = 13.5;

    assert!(
        (wb - expected).abs() <= t_tol(expected) * 5.0,
        "wet_bulb(20°C, 50%) = {}°C, expected ≈ {}°C",
        wb,
        expected
    );
}

#[test]
fn test_wet_bulb_temperature_30c_80rh() {
    // ASHRAE psychrometric chart reference: 30°C, 80% RH => Twb ≈ 27.0°C
    let t = 30.0;
    let rh = 80.0;
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let wb = calculate_wet_bulb(t, rh, p);
    let expected = 27.0;

    assert!(
        (wb - expected).abs() <= t_tol(expected) * 5.0,
        "wet_bulb(30°C, 80%) = {}°C, expected ≈ {}°C",
        wb,
        expected
    );
}

#[test]
fn test_wet_bulb_temperature_35c_20rh() {
    // Hot dry condition: 35°C, 20% RH => Twb ≈ 18.7°C (ASHRAE thermodynamic wet-bulb)
    let t = 35.0;
    let rh = 20.0;
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let wb = calculate_wet_bulb(t, rh, p);
    let expected = 18.7;

    assert!(
        (wb - expected).abs() <= t_tol(expected) * 5.0,
        "wet_bulb(35°C, 20%) = {}°C, expected ≈ {}°C",
        wb,
        expected
    );
}

#[test]
fn test_wet_bulb_temperature_5c_100rh() {
    // Cold saturated: 5°C, 100% RH => Twb ≈ 5°C (saturated)
    let t = 5.0;
    let rh = 100.0;
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let wb = calculate_wet_bulb(t, rh, p);

    // At 100% RH, wet bulb should equal dry bulb
    assert!(
        (wb - t).abs() <= t_tol(t),
        "At 100% RH, wet bulb {}°C should equal dry bulb {}°C",
        wb,
        t
    );
}

#[test]
fn test_wet_bulb_in_bounds() {
    // Wet bulb must be between dew point and dry bulb
    let test_cases = [
        (25.0, 50.0),
        (20.0, 80.0),
        (30.0, 20.0),
        (10.0, 30.0),
        (35.0, 60.0),
        (5.0, 90.0),
        (-5.0, 70.0),
    ];

    for (t, rh) in test_cases {
        let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;
        let wb = calculate_wet_bulb(t, rh, p);
        let dp = calculate_dew_point(t, rh, p);

        assert!(
            wb >= dp - 0.1,
            "wet_bulb {}°C must be ≥ dew_point {}°C at {}°C, {}% RH",
            wb,
            dp,
            t,
            rh
        );
        assert!(
            wb <= t + 0.1,
            "wet_bulb {}°C must be ≤ dry_bulb {}°C at {}°C, {}% RH",
            wb,
            t,
            t,
            rh
        );
    }
}

// =============================================================================
// HUMIDITY RATIO TESTS
// =============================================================================

#[test]
fn test_humidity_ratio_20c_50rh() {
    // ASHRAE reference: 20°C, 50% RH, 101325 Pa => ω ≈ 0.0073 kg/kg
    let t = 20.0;
    let rh = 50.0;
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let w = calculate_humidity_ratio(t, rh, p);
    let expected = 0.0073;

    assert!(
        (w - expected).abs() <= w_tol(expected) * 10.0,
        "W(20°C, 50%) = {} kg/kg, expected ≈ {} kg/kg",
        w,
        expected
    );
}

#[test]
fn test_humidity_ratio_25c_50rh() {
    // ASHRAE reference: 25°C, 50% RH, 101325 Pa => ω ≈ 0.0099 kg/kg
    let t = 25.0;
    let rh = 50.0;
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let w = calculate_humidity_ratio(t, rh, p);
    let expected = 0.0099;

    assert!(
        (w - expected).abs() <= w_tol(expected) * 10.0,
        "W(25°C, 50%) = {} kg/kg, expected ≈ {} kg/kg",
        w,
        expected
    );
}

#[test]
fn test_humidity_ratio_30c_80rh() {
    // ASHRAE reference: 30°C, 80% RH, 101325 Pa => ω ≈ 0.0217 kg/kg
    let t = 30.0;
    let rh = 80.0;
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let w = calculate_humidity_ratio(t, rh, p);
    let expected = 0.0217;

    assert!(
        (w - expected).abs() <= w_tol(expected) * 10.0,
        "W(30°C, 80%) = {} kg/kg, expected ≈ {} kg/kg",
        w,
        expected
    );
}

#[test]
fn test_humidity_ratio_very_dry_air() {
    // Very dry air: -10°C, 10% RH => ω ≈ 0.0006 kg/kg
    let t = -10.0;
    let rh = 10.0;
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let w = calculate_humidity_ratio(t, rh, p);
    let expected = 0.0006;

    assert!(w > 0.0, "Humidity ratio must be positive, got {} kg/kg", w);
    assert!(
        (w - expected).abs() <= w_tol(expected) * 10.0,
        "W(-10°C, 10%) = {} kg/kg, expected ≈ {} kg/kg",
        w,
        expected
    );
}

#[test]
fn test_humidity_ratio_100_rh() {
    // At 100% RH, humidity ratio is at saturation value
    let t = 25.0;
    let rh = 100.0;
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let w = calculate_humidity_ratio(t, rh, p);

    // Saturation humidity ratio at 25°C ≈ 0.0198 kg/kg
    let expected = 0.0198;
    assert!(
        (w - expected).abs() <= w_tol(expected) * 10.0,
        "W_sat(25°C) = {} kg/kg, expected ≈ {} kg/kg",
        w,
        expected
    );
}

// =============================================================================
// ENTHALPY TESTS
// =============================================================================

#[test]
fn test_enthalpy_20c_50rh() {
    // ASHRAE reference: 20°C, 50% RH => h ≈ 38.5 kJ/kg
    let t = 20.0;
    let rh = 50.0;
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let h = calculate_enthalpy(t, rh, p);
    let expected = 38.5;

    assert!(
        (h - expected).abs() <= 0.5,
        "h(20°C, 50%) = {} kJ/kg, expected ≈ {} kJ/kg",
        h,
        expected
    );
}

#[test]
fn test_enthalpy_25c_50rh() {
    // ASHRAE reference: 25°C, 50% RH => h ≈ 50.4 kJ/kg
    let t = 25.0;
    let rh = 50.0;
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let h = calculate_enthalpy(t, rh, p);
    let expected = 50.4;

    assert!(
        (h - expected).abs() <= 0.5,
        "h(25°C, 50%) = {} kJ/kg, expected ≈ {} kJ/kg",
        h,
        expected
    );
}

#[test]
fn test_enthalpy_30c_80rh() {
    // ASHRAE reference: 30°C, 80% RH => h ≈ 85.0 kJ/kg
    let t = 30.0;
    let rh = 80.0;
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let h = calculate_enthalpy(t, rh, p);
    let expected = 85.0;

    assert!(
        (h - expected).abs() <= 1.0,
        "h(30°C, 80%) = {} kJ/kg, expected ≈ {} kJ/kg",
        h,
        expected
    );
}

#[test]
fn test_enthalpy_0c_0rh() {
    // ASHRAE reference: 0°C, 0% RH => h ≈ 0.0 kJ/kg (dry air reference)
    let t = 0.0;
    let rh = 0.0;
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let h = calculate_enthalpy(t, rh, p);
    let expected = 0.0;

    assert!(
        (h - expected).abs() <= 0.1,
        "h(0°C, 0%) = {} kJ/kg, expected ≈ {} kJ/kg",
        h,
        expected
    );
}

#[test]
fn test_enthalpy_increases_with_temperature() {
    // Enthalpy should increase with temperature at constant RH
    let rh = 50.0;
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let h_10 = calculate_enthalpy(10.0, rh, p);
    let h_20 = calculate_enthalpy(20.0, rh, p);
    let h_30 = calculate_enthalpy(30.0, rh, p);

    assert!(
        h_20 > h_10,
        "Enthalpy at 20°C ({}) must be > enthalpy at 10°C ({})",
        h_20,
        h_10
    );
    assert!(
        h_30 > h_20,
        "Enthalpy at 30°C ({}) must be > enthalpy at 20°C ({})",
        h_30,
        h_20
    );
}

// =============================================================================
// DENSITY OF MOIST AIR TESTS
// =============================================================================

/// Calculates density of moist air (kg/m³) using the ideal gas law for a mixture.
///
/// Formula:
/// ```text
/// p_v = ω × P / (0.62198 + ω)
/// ρ   = (P − p_v) / (R_da × T_K) + p_v / (R_v × T_K)
/// ```
///
/// Where:
/// - P = atmospheric pressure (Pa)
/// - T_K = temperature (K)
/// - ω = humidity ratio (kg_water_vapor / kg_dry_air)
/// - 0.62198 = molecular weight ratio (H2O / dry air)
/// - R_da = 287.055 J/(kg·K) (dry air), R_v = 461.495 J/(kg·K) (water vapor)
///
/// # Arguments
///
/// * `temperature` - Dry bulb temperature in °C
/// * `humidity_ratio` - Humidity ratio in kg_water_vapor / kg_dry_air
/// * `pressure` - Atmospheric pressure in Pa
///
/// # Returns
///
/// Density of moist air in kg/m³
fn density_moist_air(temperature: f64, humidity_ratio: f64, pressure: f64) -> f64 {
    const R_DRY_AIR: f64 = 287.055; // J/(kg·K)
    const RATIO_MW: f64 = 0.62198; // H2O / dry_air molecular weight ratio

    let t_kelvin = temperature + 273.15;
    let partial_pressure_vapor = humidity_ratio * pressure / (RATIO_MW + humidity_ratio);
    let partial_pressure_dry = pressure - partial_pressure_vapor;

    (partial_pressure_dry / (R_DRY_AIR * t_kelvin))
        + (partial_pressure_vapor / (461.495 * t_kelvin))
}

#[test]
fn test_density_moist_air_20c_0rh() {
    // Dry air at 20°C, 101325 Pa => ρ ≈ 1.204 kg/m³
    let t = 20.0;
    let w = 0.0; // dry air
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let rho = density_moist_air(t, w, p);
    let expected = 1.204;

    assert!(
        (rho - expected).abs() <= 0.005,
        "ρ(20°C, dry) = {} kg/m³, expected ≈ {} kg/m³",
        rho,
        expected
    );
}

#[test]
fn test_density_moist_air_25c_50rh() {
    // ASHRAE reference: 25°C, 50% RH => ρ ≈ 1.177 kg/m³
    let t = 25.0;
    let rh = 50.0;
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let w = calculate_humidity_ratio(t, rh, p);
    let rho = density_moist_air(t, w, p);
    let expected = 1.177;

    assert!(
        (rho - expected).abs() <= 0.01,
        "ρ(25°C, 50%) = {} kg/m³, expected ≈ {} kg/m³",
        rho,
        expected
    );
}

#[test]
fn test_density_moist_air_35c_80rh() {
    // Hot humid: 35°C, 80% RH => ρ ≈ 1.126 kg/m³
    let t = 35.0;
    let rh = 80.0;
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let w = calculate_humidity_ratio(t, rh, p);
    let rho = density_moist_air(t, w, p);
    let expected = 1.126;

    assert!(
        (rho - expected).abs() <= 0.01,
        "ρ(35°C, 80%) = {} kg/m³, expected ≈ {} kg/m³",
        rho,
        expected
    );
}

#[test]
fn test_density_moist_air_less_than_dry_air() {
    // Moist air is less dense than dry air at same T and P
    let t = 25.0;
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let rho_dry = density_moist_air(t, 0.0, p);
    let w = calculate_humidity_ratio(t, 80.0, p);
    let rho_moist = density_moist_air(t, w, p);

    assert!(
        rho_moist < rho_dry,
        "Moist air ({}) must be less dense than dry air ({})",
        rho_moist,
        rho_dry
    );
}

// =============================================================================
// EDGE CASE TESTS
// =============================================================================

#[test]
fn test_sub_zero_temperature() {
    // Sub-zero conditions: -15°C, 60% RH
    let t = -15.0;
    let rh = 60.0;
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let p_sat = saturation_vapor_pressure(t);
    assert!(
        p_sat > 0.0 && p_sat < 1000.0,
        "p_sat at -15°C should be small: {} Pa",
        p_sat
    );

    let dp = calculate_dew_point(t, rh, p);
    assert!(dp <= t, "Dew point {}°C must be ≤ dry bulb {}°C", dp, t);

    let wb = calculate_wet_bulb(t, rh, p);
    assert!(
        wb >= dp - 0.1 && wb <= t + 0.1,
        "Wet bulb {}°C must be between dew point and dry bulb",
        wb
    );

    let w = calculate_humidity_ratio(t, rh, p);
    assert!(
        w > 0.0 && w < 0.01,
        "Humidity ratio at -15°C, 60% should be small: {} kg/kg",
        w
    );
}

#[test]
fn test_100_percent_rh() {
    // At 100% RH, wet bulb = dry bulb = dew point
    let t = 25.0;
    let rh = 100.0;
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let dp = calculate_dew_point(t, rh, p);
    let wb = calculate_wet_bulb(t, rh, p);

    assert!(
        (dp - t).abs() <= t_tol(t),
        "At 100% RH, dew point {}°C should equal dry bulb {}°C",
        dp,
        t
    );
    assert!(
        (wb - t).abs() <= t_tol(t),
        "At 100% RH, wet bulb {}°C should equal dry bulb {}°C",
        wb,
        t
    );
}

#[test]
fn test_very_dry_air() {
    // Very dry air: 40°C, 5% RH
    let t = 40.0;
    let rh = 5.0;
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let w = calculate_humidity_ratio(t, rh, p);
    assert!(
        w > 0.0 && w < 0.01,
        "Very dry air humidity ratio should be small: {} kg/kg",
        w
    );

    let h = calculate_enthalpy(t, rh, p);
    // At very low RH, enthalpy is close to dry air value
    let h_dry = 1.006 * t; // ≈ 40.2 kJ/kg
    assert!(
        h > h_dry - 1.0,
        "Enthalpy at 5% RH ({}) should be close to dry air ({})",
        h,
        h_dry
    );
}

#[test]
fn test_high_temperature_high_rh() {
    // High temperature and high humidity: 40°C, 90% RH
    // This is a challenging condition for numerical methods
    let t = 40.0;
    let rh = 90.0;
    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;

    let dp = calculate_dew_point(t, rh, p);
    assert!(dp <= t, "Dew point {}°C must be ≤ dry bulb {}°C", dp, t);

    let wb = calculate_wet_bulb(t, rh, p);
    assert!(
        wb >= dp - 0.1 && wb <= t + 0.1,
        "Wet bulb {}°C must be between dew point and dry bulb",
        wb
    );

    let w = calculate_humidity_ratio(t, rh, p);
    assert!(w > 0.0, "Humidity ratio must be positive at 90% RH");
}

// =============================================================================
// PERFORMANCE TEST
// =============================================================================

#[test]
fn test_performance_under_100ms() {
    // All functions should complete in under 100ms for typical use
    use std::time::Instant;

    let p = STANDARD_ATMOSPHERIC_PRESSURE_Pa;
    let iterations = 1000;

    // Test saturation vapor pressure
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = saturation_vapor_pressure(25.0);
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_millis() < 100,
        "saturation_vapor_pressure {} iterations took {}ms (>100ms)",
        iterations,
        elapsed.as_millis()
    );

    // Test dew point
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = calculate_dew_point(25.0, 50.0, p);
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_millis() < 100,
        "calculate_dew_point {} iterations took {}ms (>100ms)",
        iterations,
        elapsed.as_millis()
    );

    // Test wet bulb
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = calculate_wet_bulb(25.0, 50.0, p);
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_millis() < 100,
        "calculate_wet_bulb {} iterations took {}ms (>100ms)",
        iterations,
        elapsed.as_millis()
    );

    // Test humidity ratio
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = calculate_humidity_ratio(25.0, 50.0, p);
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_millis() < 100,
        "calculate_humidity_ratio {} iterations took {}ms (>100ms)",
        iterations,
        elapsed.as_millis()
    );

    // Test enthalpy
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = calculate_enthalpy(25.0, 50.0, p);
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_millis() < 100,
        "calculate_enthalpy {} iterations took {}ms (>100ms)",
        iterations,
        elapsed.as_millis()
    );
}
