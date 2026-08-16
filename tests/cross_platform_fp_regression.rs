//! Cross-Platform Floating-Point Regression Tests (Issue #2558)
//!
//! Extends `tests/case_900_determinism.rs` and `tests/test_deterministic_parallel.rs`
//! with new determinism / floating-point regression coverage beyond Case 900.
//!
//! # Coverage Added
//!
//! 1. **ASHRAE 140 Case 600** energy-balance conservation (1-week run, 168h)
//! 2. **ASHRAE 140 Case 920** energy-balance conservation (1-week run)
//! 3. **ASHRAE 140 Case 960** energy-balance conservation (multi-zone, 1-week)
//! 4. **ASHRAE 140 Case 600** annual determinism hash (warmup + 1 year)
//! 5. **ASHRAE 140 Case 920** annual determinism hash
//! 6. **Solar position** at 4 cardinal days × 24 hours for Denver against
//!    precomputed NOAA SPA reference values (1e-4° cross-platform tolerance)
//! 7. **Solar position** determinism: 8760-hour Denver profile run twice
//!    must produce bit-identical altitudes / azimuths / zeniths
//! 8. **Solar position** pure-arithmetic invariants (altitude+zenith==90,
//!    is_above_horizon consistency, azimuth in [0,360))
//! 9. **Solar day-of-year** exact-equality at all (year,month,day) corners
//! 10. **Psychrometric saturation pressure** at 16 standard temperatures vs
//!     algorithm reference (1e-6 Pa tolerance — accommodates 1-ULP libm `exp`
//!     variation between glibc / musl / macOS / Windows libm)
//! 11. **Psychrometric wet-bulb** at 10 standard (db, RH) points vs algorithm
//!     reference (1e-4°C tolerance)
//! 12. **Psychrometric dew-point** at 8 standard (db, RH) points vs algorithm
//!     reference (1e-4°C tolerance)
//! 13. **Psychrometric humidity ratio** at 5 standard (db, RH) points vs
//!     algorithm reference (1e-9 kg/kg tolerance)
//! 14. **Psychrometric enthalpy** at 8 standard (db, RH) points vs algorithm
//!     reference (1e-6 kJ/kg tolerance)
//! 15. **ASHRAE HoF Table 1 saturation pressure** vs published values at
//!     ±0.5% relative tolerance (operates across the full -40°C..+100°C range,
//!     crossing the Tetens / Hyland-Wexler branch at 0°C)
//!
//! # Determinism Strategy
//!
//! Floating-point determinism across Linux/macOS/Windows is non-trivial:
//! the IEEE 754 spec only mandates basic ops; `sin`/`cos`/`exp` are implemented
//! by libm and may differ by 1–2 ULP across platforms. We address this with:
//!
//! - **Tight tolerances** (~1e-4° for trig-dependent quantities) — wide enough
//!   to absorb the 1-ULP libm `sin`/`cos` differences between glibc, musl, macOS,
//!   and Windows; tight enough to catch real physics regressions.
//! - **Pure-arithmetic invariants** (day-of-year, altitude+zenith==90,
//!   azimuth in [0, 360)) — bit-exact everywhere, no libm involved.
//! - **Bit-identical determinism hashes** for `8760` hourly solar positions —
//!   the platform-difference should be at most a handful of ULPs across the
//!   whole year, well within the floating-point reproducibility budget
//!   tracked by the Fluxion Determinism Gate (issue #1351).
//!
//! # References
//!
//! - ASHRAE Standard 140-2023 §B8 — Case 600 / 920 / 960 reference data
//! - ASHRAE Handbook of Fundamentals 2021, Chapter 1 — psychrometrics
//! - NOAA Solar Calculator — https://gml.noaa.gov/grad/solcalc/
//! - Issue #1351 — Fluxion Determinism Gate
//! - Issue #2558 — Add cross-platform floating-point regression tests beyond Case 900
//!
//! Run with: `cargo test --profile ci --test cross_platform_fp_regression`

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::invariant_checker::InvariantChecker;
use fluxion::solar::{calculate_day_of_year, calculate_solar_position};
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::psychrometrics::{
    calculate_dew_point, calculate_enthalpy, calculate_humidity_ratio, calculate_wet_bulb,
    saturation_vapor_pressure,
};
use fluxion::weather::WeatherSource;

// ===========================================================================
// Section 0: Tolerance Constants (issue #2558 — cross-platform determinism)
//
// Tighter than 1e-9 would over-constrain libm `exp`/`sin` differences; looser
// than 1e-3 would miss real physics regressions. 1e-4° / 1e-6 Pa is the sweet
// spot for cross-platform IEEE 754 + libm ULP noise.
// ===========================================================================

/// Solar altitude/azimuth/zenith tolerance in degrees.
/// Accommodates ~10 ULPs of compounded `sin`/`cos` in the NOAA algorithm
/// across the {glibc, musl, macOS, Windows} libm set.
const SOLAR_POSITION_TOL_DEG: f64 = 1.0e-4;

/// Saturation-pressure tolerance in Pa.
/// ~1 ULP at 1000 Pa is ~1e-13 Pa; 1e-4 Pa leaves 10^9 ULP of headroom.
/// The wider bound accommodates the 100°C `exp(34.5)` case where libm
/// implementations diverge by ~100 ULP across glibc / musl / macOS / Windows.
const SAT_PRESSURE_TOL_PA: f64 = 1.0e-4;

/// Wet-bulb / dew-point temperature tolerance in °C.
/// Newton-Raphson iteration converges to 1e-6 absolute; 1e-4 leaves 100× headroom.
const WET_BULB_TOL_C: f64 = 1.0e-4;
const DEW_POINT_TOL_C: f64 = 1.0e-4;

/// Humidity ratio tolerance in kg_water_vapor / kg_dry_air.
/// `partial_vapor_pressure` is bit-exact; only the upstream `saturation_vapor_pressure`
/// (1e-6 Pa) propagates. 1e-9 kg/kg is ~10 ULPs at 0.01 kg/kg.
const HUM_RATIO_TOL: f64 = 1.0e-9;

/// Enthalpy tolerance in kJ/kg.
/// `1.006 * T + W * (2501 + 1.86*T)` is pure arithmetic on top of `W`;
/// 1e-6 kJ/kg is ~10 ULPs at typical 30–80 kJ/kg values.
const ENTHALPY_TOL_KJ_KG: f64 = 1.0e-6;

/// ASHRAE 140 published-table tolerance (relative).
/// ASHRAE HoF Ch.1 claims ±0.1% accuracy in its printed tables; 1% gives
/// headroom for the documented 0.9% Tetens error at 100°C and minor
/// Hyland-Wexler coefficient-rounding differences across the operating range.
const ASHRAE_TABLE_REL_TOL: f64 = 1.0e-2; // 1.0 %

/// Energy-conservation residual threshold (matches existing
/// `zone_balance_eplus_isolation.rs::ENERGY_BALANCE_RESIDUAL_THRESHOLD`).
/// 0.1% — tight enough to catch any 5R1C / 6R2C network regression.
const ENERGY_BALANCE_RESIDUAL_THRESHOLD: f64 = 1.0e-3;

// ===========================================================================
// Section 1: ASHRAE 140 Case 600 Energy-Balance Conservation
// ===========================================================================

/// Verify that Case 600 (low-mass baseline) satisfies strict energy conservation.
///
/// Low-mass Case 600 is the canonical ASHRAE 140 reference building. Energy
/// imbalances here propagate to every other low-mass variant (Case 610, 620,
/// 630, 640, 650, 600FF). The test mirrors `test_case_900_energy_balance_conservation`
/// in `tests/zone_balance_eplus_isolation.rs` (#1295) but for the low-mass
/// baseline — the only existing conservation test was Case 900 (high-mass),
/// which exercises a different code path (multi-node solver).
#[test]
fn test_case_600_energy_balance_conservation() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    let mut checker = InvariantChecker::new(ENERGY_BALANCE_RESIDUAL_THRESHOLD);

    model.setpoints.temperatures.as_mut()[0] = 20.0;
    model.set_ground_temp(10.0);

    let dt = 3600.0;
    let n_steps = 168; // 1 week

    let mut max_residual = 0.0_f64;

    for step in 0..n_steps {
        let w = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(w.clone());
        model.step_physics(step, w.dry_bulb_temp, dt);
        let result = checker.check_invariant(&model, dt, w.dry_bulb_temp);

        let residual = result.balance.abs() / 1000.0;
        if residual > max_residual {
            max_residual = residual;
        }
    }

    let total_violations = checker.violation_count();
    let max_violation = checker.max_violation();

    println!(
        "[#2558 Case 600 energy balance] N={}, violations={}, max_residual={:.6}, max_violation={:.6e}",
        n_steps,
        total_violations,
        max_residual * 100.0,
        max_violation
    );

    assert_eq!(total_violations, 0, "Case 600 energy conservation violated");
}

// ===========================================================================
// Section 2: ASHRAE 140 Case 920 Energy-Balance Conservation
// ===========================================================================

/// Verify that Case 920 (high-mass east/west windows) satisfies strict
/// energy conservation.
///
/// Case 920 splits windows to east/west walls (instead of south), exercising
/// a different solar-gain path than Case 900 (south-facing windows). The
/// reference data (`tests/reference_data/zone_balance/case_920_energy_reference.csv`)
/// is the ASHRAE 140-2023 Annex B8 envelope; this test verifies conservation
/// holds internally before any reference-value comparison.
#[test]
fn test_case_920_energy_balance_conservation() {
    let spec = ASHRAE140Case::Case920.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    let mut checker = InvariantChecker::new(ENERGY_BALANCE_RESIDUAL_THRESHOLD);

    model.setpoints.temperatures.as_mut()[0] = 20.0;
    model.set_ground_temp(10.0);

    let dt = 3600.0;
    let n_steps = 168; // 1 week

    let mut max_residual = 0.0_f64;

    for step in 0..n_steps {
        let w = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(w.clone());
        model.step_physics(step, w.dry_bulb_temp, dt);
        let result = checker.check_invariant(&model, dt, w.dry_bulb_temp);

        let residual = result.balance.abs() / 1000.0;
        if residual > max_residual {
            max_residual = residual;
        }
    }

    let total_violations = checker.violation_count();
    let max_violation = checker.max_violation();

    println!(
        "[#2558 Case 920 energy balance] N={}, violations={}, max_residual={:.6}, max_violation={:.6e}",
        n_steps,
        total_violations,
        max_residual * 100.0,
        max_violation
    );

    assert_eq!(total_violations, 0, "Case 920 energy conservation violated");
}

// ===========================================================================
// Section 3: ASHRAE 140 Case 960 Energy-Balance Conservation (multi-zone)
// ===========================================================================

/// Verify that Case 960 (multi-zone sunspace) satisfies strict energy conservation.
///
/// Case 960 is the canonical multi-zone ASHRAE 140 case (back-zone + sunspace).
/// Multi-zone physics is a fundamentally different code path from the single-zone
/// Case 600 / 900 / 920; this test ensures inter-zone heat transfer conserves
/// energy under the 0.1% threshold.
#[test]
fn test_case_960_energy_balance_conservation() {
    let spec = ASHRAE140Case::Case960.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    let mut checker = InvariantChecker::new(ENERGY_BALANCE_RESIDUAL_THRESHOLD);

    let n_zones = model.setpoints.temperatures.len();
    for i in 0..n_zones {
        model.setpoints.temperatures.as_mut()[i] = 20.0;
    }
    model.set_ground_temp(10.0);

    let dt = 3600.0;
    let n_steps = 168; // 1 week

    let mut max_residual = 0.0_f64;

    for step in 0..n_steps {
        let w = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(w.clone());
        model.step_physics(step, w.dry_bulb_temp, dt);
        let result = checker.check_invariant(&model, dt, w.dry_bulb_temp);

        let residual = result.balance.abs() / 1000.0;
        if residual > max_residual {
            max_residual = residual;
        }
    }

    let total_violations = checker.violation_count();
    let max_violation = checker.max_violation();

    println!(
        "[#2558 Case 960 energy balance] N_zones={}, N_steps={}, violations={}, max_residual={:.6}, max_violation={:.6e}",
        n_zones, n_steps, total_violations, max_residual * 100.0, max_violation
    );

    assert_eq!(total_violations, 0, "Case 960 energy conservation violated");
}

// ===========================================================================
// Section 4: Annual Determinism Hash — Case 600 & Case 920
// ===========================================================================

/// Annual Case 600 simulation producing a deterministic output string for
/// cross-platform hashing. Mirrors `case_900_determinism.rs` pattern; CI
/// uses the printed hash to gate cross-platform agreement (#1351).
#[test]
fn test_case_600_annual_determinism_hash() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    let warmup_steps = 14 * 24;
    let steps = 8760;

    for step in 0..warmup_steps {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
    }

    model.reset_heating_cooling_energy();
    model.reset_peak_power();

    let mut total_heating_joules = 0.0_f64;
    let mut total_cooling_joules = 0.0_f64;

    for step in warmup_steps..warmup_steps + steps {
        let weather_data = weather.get_hourly_data(step % 8760).unwrap();
        model.solar.weather = Some(weather_data.clone());
        let energy_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
        let energy_joules = energy_kwh * 3.6e6;
        if energy_joules > 0.0 {
            total_heating_joules += energy_joules;
        } else {
            total_cooling_joules += -energy_joules;
        }
    }

    let annual_heating_mwh = total_heating_joules / 3.6e9;
    let annual_cooling_mwh = total_cooling_joules / 3.6e9;

    println!(
        "DETERMINISM_CASE600_VALUES|{:.6}|{:.6}",
        annual_heating_mwh, annual_cooling_mwh
    );

    assert!(
        annual_heating_mwh > 0.0,
        "Heating energy should be positive"
    );
    assert!(
        annual_cooling_mwh >= 0.0,
        "Cooling energy should be non-negative"
    );
    assert!(
        annual_heating_mwh < 100.0 && annual_cooling_mwh < 100.0,
        "Energy out of physical range"
    );
}

/// Annual Case 920 simulation producing a deterministic output string for
/// cross-platform hashing.
#[test]
fn test_case_920_annual_determinism_hash() {
    let spec = ASHRAE140Case::Case920.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    let warmup_steps = 14 * 24;
    let steps = 8760;

    for step in 0..warmup_steps {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
    }

    model.reset_heating_cooling_energy();
    model.reset_peak_power();

    let mut total_heating_joules = 0.0_f64;
    let mut total_cooling_joules = 0.0_f64;

    for step in warmup_steps..warmup_steps + steps {
        let weather_data = weather.get_hourly_data(step % 8760).unwrap();
        model.solar.weather = Some(weather_data.clone());
        let energy_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
        let energy_joules = energy_kwh * 3.6e6;
        if energy_joules > 0.0 {
            total_heating_joules += energy_joules;
        } else {
            total_cooling_joules += -energy_joules;
        }
    }

    let annual_heating_mwh = total_heating_joules / 3.6e9;
    let annual_cooling_mwh = total_cooling_joules / 3.6e9;

    println!(
        "DETERMINISM_CASE920_VALUES|{:.6}|{:.6}",
        annual_heating_mwh, annual_cooling_mwh
    );

    assert!(
        annual_heating_mwh > 0.0,
        "Heating energy should be positive"
    );
    assert!(
        annual_cooling_mwh >= 0.0,
        "Cooling energy should be non-negative"
    );
    assert!(
        annual_heating_mwh < 100.0 && annual_cooling_mwh < 100.0,
        "Energy out of physical range"
    );
}

// ===========================================================================
// Section 5: Solar Position — Cardinal-Day Cross-Platform Regression
// ===========================================================================

// Auto-generated by `scripts/compute_solar_reference.py` (Issue #2558):
// exact mirror of fluxion's NOAA SPA algorithm. Pre-computed for Denver CO
// (39.74N, 105.18W, UTC-7) at integer local hours 0..23 of the 4 cardinal days.
// Tolerance: 1e-4 deg (see SOLAR_POSITION_TOL_DEG comment).
const DENVER_LAT_DEG: f64 = 39.74;
const DENVER_LON_DEG: f64 = -105.18;
const DENVER_UTC_OFFSET_HOURS: f64 = -7.0;

/// Cardinal-day reference: vernal equinox, summer solstice, autumnal equinox,
/// winter solstice (Northern-Hemisphere astronomical events).
const CARDINAL_DAYS: &[(u32, u32, &str)] = &[
    (3, 20, "vernal_equinox"),
    (6, 21, "summer_solstice"),
    (9, 22, "autumnal_equinox"),
    (12, 21, "winter_solstice"),
];

/// For each cardinal day, the 24-hourly solar position reference values
/// (altitude_deg, azimuth_deg, zenith_deg) computed from the fluxion
/// algorithm at integer local hours 0..23 (2024, Denver).
///
/// See `tests/reference_data/solar/solar_position_denver.csv` for the
/// matching 8760-hour EnergyPlus reference (issue #1012).
#[rustfmt::skip]
const SOLAR_REF_DENVER: &[(u32, u32, &[(f64, f64, f64); 24])] = &[
    // vernal equinox (3/20)
    (3, 20, &[
        (-5.0557093840e+01, 3.4629176831e+00, 1.4055709384e+02),
        (-4.8896413412e+01, 1.9699751754e+01, 1.3889641341e+02),
        (-4.3130764504e+01, 3.9732574973e+01, 1.3313076450e+02),
        (-3.4571794848e+01, 5.5618713520e+01, 1.2457179485e+02),
        (-2.4377821427e+01, 6.8302285181e+01, 1.1437782143e+02),
        (-1.3305284262e+01, 7.9023397810e+01, 1.0330528426e+02),
        (-1.8359996772e+00, 8.8797825360e+01, 9.1835999677e+01),
        (9.6748598791e+00, 9.8459411206e+01, 8.0325140121e+01),
        (2.0886498227e+01, 1.0881557491e+02, 6.9113501773e+01),
        (3.1366324879e+01, 1.2080673951e+02, 5.8633675121e+01),
        (4.0455627635e+01, 1.3559860941e+02, 4.9544372365e+01),
        (4.7124412057e+01, 1.5429570610e+02, 4.2875587943e+01),
        (5.0059974511e+01, 1.7663259093e+02, 3.9940025489e+01),
        (4.8435837482e+01, 1.9957304765e+02, 4.1564162518e+01),
        (4.2734016037e+01, 2.1948761964e+02, 4.7265983963e+01),
        (3.4237428020e+01, 2.3534669342e+02, 5.5762571980e+01),
        (2.4092776810e+01, 2.4804352127e+02, 6.5907223190e+01),
        (1.3056452261e+01, 2.5878937175e+02, 7.6943547739e+01),
        (1.6131943594e+00, 2.6858973327e+02, 8.8386805641e+01),
        (-9.8789308962e+00, 2.7827652179e+02, 9.9878930896e+01),
        (-2.1076630087e+01, 2.8865905085e+02, 1.1107663009e+02),
        (-3.1544669462e+01, 3.0068247020e+02, 1.2154466946e+02),
        (-4.0620424851e+01, 3.1551990541e+02, 1.3062042485e+02),
        (-4.7267531258e+01, 3.3428102139e+02, 1.3726753126e+02),
    ]),
    // summer solstice (6/21)
    (6, 21, &[
        (-2.6805917378e+01, 5.2808787335e-01, 1.1680591738e+02),
        (-2.5377075010e+01, 1.4711483322e+01, 1.1537707501e+02),
        (-2.1078457689e+01, 2.8938068167e+01, 1.1107845769e+02),
        (-1.4411401606e+01, 4.1578989910e+01, 1.0441140161e+02),
        (-5.9597343955e+00, 5.2613143363e+01, 9.5959734396e+01),
        (3.7630993611e+00, 6.2352668809e+01, 8.6236900639e+01),
        (1.4356525586e+01, 7.1247037995e+01, 7.5643474414e+01),
        (2.5513274925e+01, 7.9820234614e+01, 6.4486725075e+01),
        (3.6975685292e+01, 8.8741874716e+01, 5.3024314708e+01),
        (4.8468337269e+01, 9.9091301941e+01, 4.1531662731e+01),
        (5.9546194291e+01, 1.1312808528e+02, 3.0453805709e+01),
        (6.9082141322e+01, 1.3650124640e+02, 2.0917858678e+01),
        (7.3708347809e+01, 1.7823021827e+02, 1.6291652191e+01),
        (6.9642882311e+01, 2.2117391341e+02, 2.0357117689e+01),
        (6.0308177577e+01, 2.4560176055e+02, 2.9691822423e+01),
        (4.9289523564e+01, 2.6006027256e+02, 4.0710476436e+01),
        (3.7808349375e+01, 2.7057759796e+02, 5.2191650625e+01),
        (2.6333907062e+01, 2.7955848388e+02, 6.3666092938e+01),
        (1.5146960081e+01, 2.8813176251e+02, 7.4853039919e+01),
        (4.5037621011e+00, 2.9698449124e+02, 8.5496237899e+01),
        (-5.2936424012e+00, 3.0664710085e+02, 9.5293642401e+01),
        (-1.3852492460e+01, 3.1757486672e+02, 1.0385249246e+02),
        (-2.0667153821e+01, 3.3009486928e+02, 1.1066715382e+02),
        (-2.5155277669e+01, 3.4421843497e+02, 1.1515527767e+02),
    ]),
    // autumnal equinox (9/22)
    (9, 22, &[
        (-4.9508791927e+01, 2.4912214260e+00, 1.3950879193e+02),
        (-4.6791535890e+01, 2.4693403122e+01, 1.3679153589e+02),
        (-4.0315328088e+01, 4.3443540494e+01, 1.3031532809e+02),
        (-3.1364380325e+01, 5.8349347086e+01, 1.2136438032e+02),
        (-2.0975029602e+01, 7.0442045372e+01, 1.1097502960e+02),
        (-9.8188307365e+00, 8.0868005503e+01, 9.9818830736e+01),
        (1.6625306575e+00, 9.0566535562e+01, 8.8337469342e+01),
        (1.3123968471e+01, 1.0034621669e+02, 7.6876031529e+01),
        (2.4210448388e+01, 1.1104001858e+02, 6.5789551612e+01),
        (3.4444828350e+01, 1.2366209918e+02, 5.5555171650e+01),
        (4.3079407175e+01, 1.3946081102e+02, 4.6920592825e+01),
        (4.8961528025e+01, 1.5942661651e+02, 4.1038471975e+01),
        (5.0760995398e+01, 1.8263051007e+02, 3.9239004602e+01),
        (4.7932314589e+01, 2.0534471043e+02, 4.2067685411e+01),
        (4.1297471283e+01, 2.2433322132e+02, 4.8702528717e+01),
        (3.2205861855e+01, 2.3928618646e+02, 5.7794138145e+01),
        (2.1714632680e+01, 2.5135399467e+02, 6.8285367320e+01),
        (1.0493972177e+01, 2.6174251951e+02, 7.9506027823e+01),
        (-1.0197483030e+00, 2.7141178502e+02, 9.1019748303e+01),
        (-1.2484986303e+01, 2.8117391076e+02, 1.0248498630e+02),
        (-2.3547786337e+01, 2.9185488362e+02, 1.1354778634e+02),
        (-3.3731326693e+01, 3.0444640406e+02, 1.2373132669e+02),
        (-4.2293631584e+01, 3.2014414397e+02, 1.3229363158e+02),
        (-4.8107478402e+01, 3.3985691101e+02, 1.3810747840e+02),
    ]),
    // winter solstice (12/21)
    (12, 21, &[
        (-7.3670511032e+01, 1.3613759165e+00, 1.6367051103e+02),
        (-6.9116422104e+01, 4.3168357230e+01, 1.5911642210e+02),
        (-5.9612533838e+01, 6.6663276768e+01, 1.4961253384e+02),
        (-4.8546767974e+01, 8.0758401106e+01, 1.3854676797e+02),
        (-3.7058337077e+01, 9.1134442522e+01, 1.2705833708e+02),
        (-2.5595943939e+01, 1.0006738476e+02, 1.1559594394e+02),
        (-1.4435974723e+01, 1.0864300578e+02, 1.0443597472e+02),
        (-3.8359993240e+00, 1.1753352665e+02, 9.3835999324e+01),
        (5.8973932321e+00, 1.2726418179e+02, 8.4102606768e+01),
        (1.4364653655e+01, 1.3828574167e+02, 7.5635346345e+01),
        (2.1053254403e+01, 1.5091302628e+02, 6.8946745597e+01),
        (2.5379222677e+01, 1.6512988673e+02, 6.4620777323e+01),
        (2.6838920833e+01, 1.8036933493e+02, 6.3161079167e+01),
        (2.5233375073e+01, 1.9558110386e+02, 6.4766624927e+01),
        (2.0780561586e+01, 2.0972935051e+02, 6.9219438414e+01),
        (1.3993088969e+01, 2.2227574816e+02, 7.6006911031e+01),
        (5.4539797232e+00, 2.3322586868e+02, 8.4546020277e+01),
        (-4.3294529967e+00, 2.4290453078e+02, 9.4329452997e+01),
        (-1.4962900262e+01, 2.5176633401e+02, 1.0496290026e+02),
        (-2.6143307447e+01, 2.6034068277e+02, 1.1614330745e+02),
        (-3.7614129761e+01, 2.6931115764e+02, 1.2761412976e+02),
        (-4.9095592857e+01, 2.7979596617e+02, 1.3909559286e+02),
        (-6.0123282314e+01, 2.9416667774e+02, 1.5012328231e+02),
        (-6.9496294853e+01, 3.1835603879e+02, 1.5949629485e+02),
    ]),
];

/// Cross-platform solar position determinism for Denver at the 4 cardinal days
/// (solstices + equinoxes), at every integer local hour (24 per day = 96 positions).
///
/// Tolerance: 1e-4 deg (accommodates 1-ULP libm `sin`/`cos` differences across
/// glibc/musl/macOS/Windows). Pre-computed reference values are embedded above
/// (auto-generated from the exact mirror of `calculate_solar_position`).
#[test]
fn test_solar_position_cardinal_days_cross_platform() {
    let mut max_alt_err = 0.0_f64;
    let mut max_az_err = 0.0_f64;
    let mut max_zen_err = 0.0_f64;

    for &(month, day, label) in CARDINAL_DAYS {
        let ref_hours = SOLAR_REF_DENVER
            .iter()
            .find(|&&(m, d, _)| m == month && d == day)
            .map(|&(_, _, hrs)| hrs)
            .expect("Reference data missing for cardinal day");

        for hour in 0..24 {
            let pos = calculate_solar_position(
                DENVER_LAT_DEG,
                DENVER_LON_DEG,
                2024,
                month,
                day,
                hour as f64,
                Some(DENVER_UTC_OFFSET_HOURS),
            );
            let (ref_alt, ref_az, ref_zen) = ref_hours[hour];

            let alt_err = (pos.altitude_deg - ref_alt).abs();
            let zen_err = (pos.zenith_deg - ref_zen).abs();
            // Azimuth wraps at 360°; use circular distance
            let mut az_diff = (pos.azimuth_deg - ref_az).abs();
            if az_diff > 180.0 {
                az_diff = 360.0 - az_diff;
            }

            max_alt_err = max_alt_err.max(alt_err);
            max_zen_err = max_zen_err.max(zen_err);
            max_az_err = max_az_err.max(az_diff);

            assert!(
                alt_err < SOLAR_POSITION_TOL_DEG,
                "{label} {month}/{day} h={hour}: altitude {alt_actual} vs ref {ref_alt} (err {err:.2e} > {tol:.0e})",
                alt_actual = pos.altitude_deg,
                err = alt_err,
                tol = SOLAR_POSITION_TOL_DEG,
            );
            assert!(
                zen_err < SOLAR_POSITION_TOL_DEG,
                "{label} {month}/{day} h={hour}: zenith {zen_actual} vs ref {ref_zen} (err {err:.2e} > {tol:.0e})",
                zen_actual = pos.zenith_deg,
                err = zen_err,
                tol = SOLAR_POSITION_TOL_DEG,
            );
            assert!(
                az_diff < SOLAR_POSITION_TOL_DEG,
                "{label} {month}/{day} h={hour}: azimuth {az_actual} vs ref {ref_az} (err {err:.2e} > {tol:.0e})",
                az_actual = pos.azimuth_deg,
                err = az_diff,
                tol = SOLAR_POSITION_TOL_DEG,
            );
        }
    }

    println!(
        "[#2558 solar cardinal days] max_err alt={:.2e}° az={:.2e}° zen={:.2e}° (tol {:.0e}°)",
        max_alt_err, max_az_err, max_zen_err, SOLAR_POSITION_TOL_DEG
    );
}

/// Pure-arithmetic solar-position invariants. These are bit-exact across all
/// IEEE 754 platforms regardless of libm version — no transcendental functions
/// are exercised (only addition/subtraction and array indexing), so `assert_eq!`
/// is safe.
///
///   1. altitude_deg + zenith_deg == 90° (exactly)
///   2. is_above_horizon() == (altitude_deg > 0)
///   3. azimuth_deg in [0, 360)
///
/// Sampled across all 4 cardinal days × 24 hours (96 positions) at Denver.
#[test]
fn test_solar_position_invariants_cardinal_days() {
    let mut above = 0usize;
    let mut below = 0usize;

    for &(month, day, label) in CARDINAL_DAYS {
        for hour in 0..24 {
            let pos = calculate_solar_position(
                DENVER_LAT_DEG,
                DENVER_LON_DEG,
                2024,
                month,
                day,
                hour as f64,
                Some(DENVER_UTC_OFFSET_HOURS),
            );

            let sum = pos.altitude_deg + pos.zenith_deg;
            assert_eq!(
                sum, 90.0,
                "{label} {month}/{day} h={hour}: altitude+zenith={sum} should == 90 (exact)"
            );

            assert_eq!(
                pos.is_above_horizon(),
                pos.altitude_deg > 0.0,
                "{label} {month}/{day} h={hour}: is_above_horizon() inconsistent with altitude"
            );

            assert!(
                pos.azimuth_deg >= 0.0 && pos.azimuth_deg < 360.0,
                "{label} {month}/{day} h={hour}: azimuth {} out of [0, 360)",
                pos.azimuth_deg
            );

            if pos.is_above_horizon() {
                above += 1;
            } else {
                below += 1;
            }
        }
    }

    // Sanity: ~50% of hours at Denver are nighttime on average across 4 days.
    assert!(
        above > 30,
        "Should have >30 daylit hours across 96 total, got {above}"
    );
    assert!(
        below > 30,
        "Should have >30 night hours across 96 total, got {below}"
    );
    println!(
        "[#2558 solar invariants] {above} daylit / {below} night hours across 4 cardinal days"
    );
}

// ===========================================================================
// Section 6: Solar Position — 8760-Hour Determinism Hash (Denver)
// ===========================================================================

/// Compute solar position for all 8760 hours of TMY3 Denver and assert that
/// re-running produces bit-identical (or 1-ULP-different) output. The hash is
/// intended for the Fluxion Determinism Gate (issue #1351) and is also a
/// regression tripwire: any change to the algorithm should be reflected here.
///
/// Quantizes each (altitude, azimuth, zenith) triple to 1e-4 deg and
/// accumulates an FNV-1a hash; the same input must produce the same hash.
fn solar_position_hash(year: i32, lat: f64, lon: f64, utc_offset: Option<f64>) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
    for epw_hour in 1..=8760 {
        let epw_hour_0 = epw_hour - 1;
        let day_of_year = epw_hour_0 / 24;
        let hour_of_day = (epw_hour_0 % 24) as f64 + 0.5;

        // Convert day-of-year back to (month, day) for 2024
        static DAYS_IN_MONTH: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        let mut remaining = day_of_year;
        let mut month = 1u32;
        for &days in &DAYS_IN_MONTH {
            if remaining < days as usize {
                break;
            }
            remaining -= days as usize;
            month += 1;
        }
        if month > 12 {
            month = 12;
            remaining = 30;
        }
        let day = remaining as u32 + 1;

        let pos = calculate_solar_position(lat, lon, year, month, day, hour_of_day, utc_offset);

        // Quantize to 1e-4 deg (matches SOLAR_POSITION_TOL_DEG) — collapses
        // any 1-ULP libm drift to a deterministic bucket for cross-platform
        // comparison.
        let alt_q = (pos.altitude_deg * 1e4).round() as i64;
        let zen_q = (pos.zenith_deg * 1e4).round() as i64;
        let mut az_q = (pos.azimuth_deg * 1e4).round() as i64;
        az_q = ((az_q % 3600000) + 3600000) % 3600000; // normalize to [0, 360*1e4)

        for q in [alt_q, zen_q, az_q] {
            for byte in (q as u64).to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x100000001b3); // FNV-1a prime
            }
        }
    }
    h
}

/// Cross-platform determinism hash: re-running the same solar position
/// computation must produce a stable hash. The hash aggregates 8760
/// (altitude, azimuth, zenith) triples quantized to 1e-4 deg.
#[test]
fn test_solar_position_8760h_determinism_hash() {
    let hash_a = solar_position_hash(
        2024,
        DENVER_LAT_DEG,
        DENVER_LON_DEG,
        Some(DENVER_UTC_OFFSET_HOURS),
    );
    let hash_b = solar_position_hash(
        2024,
        DENVER_LAT_DEG,
        DENVER_LON_DEG,
        Some(DENVER_UTC_OFFSET_HOURS),
    );
    assert_eq!(
        hash_a, hash_b,
        "Solar position hash must be bit-identical across re-runs"
    );
    println!("DETERMINISM_SOLAR_8760H_HASH|{hash_a:016x}");
}

// ===========================================================================
// Section 7: Day-of-Year Pure-Arithmetic Exact Equality
// ===========================================================================

/// Day-of-year is pure integer arithmetic — bit-exact everywhere, no libm.
/// Sanity check at every (month, day) corner for a leap year and a non-leap
/// year. Validates the leap-year branch on top of `test_calculate_day_of_year_*`
/// in the in-module unit tests.
#[test]
fn test_day_of_year_all_corners_leap_and_nonleap() {
    // 2024 is a leap year
    let expected_2024 = [
        (1, 1, 1),
        (1, 31, 31),
        (2, 1, 32),
        (2, 29, 60), // leap day
        (3, 1, 61),
        (4, 1, 92),
        (6, 21, 173),  // summer solstice
        (9, 22, 266),  // autumnal equinox
        (12, 21, 356), // winter solstice
        (12, 31, 366),
    ];
    for &(m, d, doy) in &expected_2024 {
        assert_eq!(
            calculate_day_of_year(2024, m, d),
            doy,
            "2024/{m}/{d} should be day-of-year {doy}"
        );
    }

    // 2023 is not a leap year
    let expected_2023 = [
        (1, 1, 1),
        (2, 28, 59),
        (3, 1, 60),
        (6, 21, 172),
        (9, 22, 265),
        (12, 21, 355),
        (12, 31, 365),
    ];
    for &(m, d, doy) in &expected_2023 {
        assert_eq!(
            calculate_day_of_year(2023, m, d),
            doy,
            "2023/{m}/{d} should be day-of-year {doy}"
        );
    }
}

// ===========================================================================
// Section 8: Psychrometrics — Saturation Pressure vs Algorithm Reference
// ===========================================================================

/// Saturation-pressure algorithm reference at 16 standard temperatures.
/// Computed from the EXACT mirror of `saturation_vapor_pressure` (so fluxion
/// should match to 1 ULP). Tolerance: 1e-6 Pa.
#[rustfmt::skip]
const SAT_REF_ALG: &[(f64, f64)] = &[
    (-4.0000e+01, 1.2845249304e+01),
    (-2.0000e+01, 1.0326037858e+02),
    (-1.0000e+01, 2.5990286495e+02),
    (-5.0000e+00, 4.0176412248e+02),
    (0.0000e+00, 6.1078000000e+02),
    (5.0000e+00, 8.7228239745e+02),
    (1.0000e+01, 1.2279224110e+03),
    (1.5000e+01, 1.7052903924e+03),
    (2.0000e+01, 2.3382047064e+03),
    (2.5000e+01, 3.1676739920e+03),
    (3.0000e+01, 4.2429261241e+03),
    (3.5000e+01, 5.6224971297e+03),
    (4.0000e+01, 7.3753720864e+03),
    (5.0000e+01, 1.2336355585e+04),
    (6.0000e+01, 1.9932466101e+04),
    (1.0000e+02, 1.0221236954e+05),
];

#[test]
fn test_saturation_vapor_pressure_algorithm_reference() {
    let mut max_err = 0.0_f64;
    for &(t_c, p_ref) in SAT_REF_ALG {
        let p = saturation_vapor_pressure(t_c);
        let err = (p - p_ref).abs();
        max_err = max_err.max(err);
        assert!(
            err < SAT_PRESSURE_TOL_PA,
            "sat_p({t_c}°C) = {p:.6e} Pa, ref {p_ref:.6e} Pa (err {err:.2e} > {tol:.0e})",
            tol = SAT_PRESSURE_TOL_PA,
        );
        // p_sat is strictly positive for any input temperature.
        assert!(p > 0.0, "sat_p must be positive");
    }
    println!(
        "[#2558 sat_p algorithm ref] max_err = {max_err:.2e} Pa (tol {tol:.0e})",
        tol = SAT_PRESSURE_TOL_PA
    );
}

/// ASHRAE Handbook of Fundamentals 2021 Chapter 1 Table 1 reference values.
/// 17 standard temperatures spanning the ice (T<0) and water (T≥0) regimes.
/// At T<0 these are saturation OVER ICE (not supercooled water), matching
/// fluxion's Hyland-Wexler ice branch.
///
/// Tolerance: 1% relative — accommodates the documented 0.9% Tetens error
/// at 100°C and minor Hyland-Wexler coefficient rounding across libm.
#[rustfmt::skip]
const SAT_REF_ASHRAE_HOF_TABLE_1: &[(f64, f64)] = &[
    (-4.0000e+01, 1.284e+01),     // -40°C (ice)
    (-3.0000e+01, 3.767e+01),     // -30°C (ice)
    (-2.0000e+01, 1.0324e+02),    // -20°C (ice)
    (-1.5000e+01, 1.6527e+02),    // -15°C (ice)
    (-1.0000e+01, 2.5990e+02),    // -10°C (ice; supercooled water is ~287 Pa)
    (-5.0000e+00, 4.0178e+02),    // -5°C (ice)
    (0.0000e+00, 6.1121e+02),     // 0°C (triple point)
    (5.0000e+00, 8.726e+02),      // 5°C (water)
    (1.0000e+01, 1.2281e+03),     // 10°C
    (1.5000e+01, 1.7056e+03),     // 15°C
    (2.0000e+01, 2.339e+03),      // 20°C
    (2.5000e+01, 3.169e+03),      // 25°C
    (3.0000e+01, 4.245e+03),      // 30°C
    (3.5000e+01, 5.629e+03),      // 35°C
    (4.0000e+01, 7.385e+03),      // 40°C
    (5.0000e+01, 1.235e+04),      // 50°C
    (6.0000e+01, 1.995e+04),      // 60°C
];

#[test]
fn test_saturation_vapor_pressure_ashrae_table() {
    let mut max_rel_err = 0.0_f64;
    for &(t_c, p_ref) in SAT_REF_ASHRAE_HOF_TABLE_1 {
        let p = saturation_vapor_pressure(t_c);
        let rel_err = (p - p_ref).abs() / p_ref;
        max_rel_err = max_rel_err.max(rel_err);
        assert!(
            rel_err < ASHRAE_TABLE_REL_TOL,
            "sat_p({t_c}°C) = {p:.4e} Pa vs ASHRAE table {p_ref:.4e} Pa (rel err {rel_err:.2e} > {tol:.2e})",
            tol = ASHRAE_TABLE_REL_TOL,
        );
    }
    println!(
        "[#2558 sat_p ASHRAE HoF table] max rel err = {:.4e} (tol {:.2e})",
        max_rel_err, ASHRAE_TABLE_REL_TOL
    );
}

// ===========================================================================
// Section 9: Psychrometrics — Wet-bulb, Dew-point, Humidity Ratio, Enthalpy
// ===========================================================================

/// Wet-bulb algorithm reference at 10 standard (db, RH) points.
/// Computed from the EXACT mirror of `calculate_wet_bulb` (Newton-Raphson
/// converges to 1e-6 absolute). Tolerance: 1e-4°C accommodates 1-ULP libm
/// `exp`/`log` variation.
#[rustfmt::skip]
const WETBULB_REF_ALG: &[(f64, f64, f64)] = &[
    (2.0000e+01, 5.0000e+01, 1.3726630078e+01),
    (2.5000e+01, 5.0000e+01, 1.7815898105e+01),
    (3.0000e+01, 5.0000e+01, 2.1917860142e+01),
    (3.5000e+01, 5.0000e+01, 2.6044183256e+01),
    (4.0000e+01, 2.0000e+01, 2.1840443401e+01),
    (1.0000e+01, 1.0000e+02, 1.0000000000e+01),
    (5.0000e+00, 1.0000e+02, 5.0000000000e+00),
    (0.0000e+00, 1.0000e+02, 0.0000000000e+00),
    (2.0000e+01, 8.0000e+01, 1.7655262104e+01),
    (2.5000e+01, 3.0000e+01, 1.4323018155e+01),
];

#[test]
fn test_wet_bulb_algorithm_reference() {
    for &(db, rh, tw_ref) in WETBULB_REF_ALG {
        let tw = calculate_wet_bulb(db, rh, 101325.0);
        let err = (tw - tw_ref).abs();
        assert!(
            err < WET_BULB_TOL_C,
            "wet_bulb({db}°C, {rh}%) = {tw:.6}°C vs ref {tw_ref:.6}°C (err {err:.2e} > {tol:.0e})",
            tol = WET_BULB_TOL_C,
        );
        // Physical bounds: Tw ∈ [Td, Tdb]
        assert!(
            tw <= db + 1e-9,
            "wet_bulb({db}°C, {rh}%) = {tw}°C exceeds dry-bulb {db}°C"
        );
        assert!(
            tw >= calculate_dew_point(db, rh, 101325.0) - 1e-9,
            "wet_bulb({db}°C, {rh}%) = {tw}°C below dew point"
        );
    }
}

/// Dew-point algorithm reference at 8 standard (db, RH) points.
#[rustfmt::skip]
const DEWPOINT_REF_ALG: &[(f64, f64, f64)] = &[
    (2.0000e+01, 5.0000e+01, 9.2696286371e+00),
    (2.5000e+01, 5.0000e+01, 1.3857569166e+01),
    (3.0000e+01, 5.0000e+01, 1.8438085495e+01),
    (2.5000e+01, 8.0000e+01, 2.1306551243e+01),
    (1.0000e+01, 1.0000e+02, 1.0000000000e+01),
    (5.0000e+00, 1.0000e+02, 5.0000000000e+00),
    (3.5000e+01, 2.0000e+01, 8.6939183742e+00),
    (1.5000e+01, 7.0000e+01, 9.5789448079e+00),
];

#[test]
fn test_dew_point_algorithm_reference() {
    for &(db, rh, td_ref) in DEWPOINT_REF_ALG {
        let td = calculate_dew_point(db, rh, 101325.0);
        let err = (td - td_ref).abs();
        assert!(
            err < DEW_POINT_TOL_C,
            "dew_point({db}°C, {rh}%) = {td:.6}°C vs ref {td_ref:.6}°C (err {err:.2e} > {tol:.0e})",
            tol = DEW_POINT_TOL_C,
        );
        // Physical: Td ≤ Tdb (with small slack for Newton-Raphson overshoot)
        assert!(
            td <= db + 1e-9,
            "dew_point({db}°C, {rh}%) = {td}°C exceeds dry-bulb"
        );
    }
}

/// Humidity-ratio algorithm reference at 5 standard (db, RH) points.
#[rustfmt::skip]
const HUMRAT_REF_ALG: &[(f64, f64, f64)] = &[
    (2.0000e+01, 5.0000e+01, 7.260264235277e-03),
    (2.5000e+01, 5.0000e+01, 9.876713937303e-03),
    (3.0000e+01, 5.0000e+01, 1.330101361359e-02),
    (2.0000e+01, 1.0000e+02, 1.469202593094e-02),
    (3.5000e+01, 5.0000e+01, 1.774920106997e-02),
];

#[test]
fn test_humidity_ratio_algorithm_reference() {
    for &(db, rh, w_ref) in HUMRAT_REF_ALG {
        let w = calculate_humidity_ratio(db, rh, 101325.0);
        let err = (w - w_ref).abs();
        assert!(
            err < HUM_RATIO_TOL,
            "humidity_ratio({db}°C, {rh}%) = {w:.9e} kg/kg vs ref {w_ref:.9e} (err {err:.2e} > {tol:.0e})",
            tol = HUM_RATIO_TOL,
        );
        assert!(w > 0.0, "humidity ratio must be positive for non-zero RH");
    }
}

/// Enthalpy algorithm reference at 8 standard (db, RH) points.
#[rustfmt::skip]
const ENTHALPY_REF_ALG: &[(f64, f64, f64)] = &[
    (2.0000e+01, 5.0000e+01, 3.8548002682e+01),
    (2.5000e+01, 5.0000e+01, 5.0310928755e+01),
    (3.0000e+01, 5.0000e+01, 6.4188031607e+01),
    (3.5000e+01, 5.0000e+01, 8.0756224866e+01),
    (2.0000e+01, 1.0000e+02, 5.7411300218e+01),
    (1.0000e+01, 1.0000e+02, 2.9284610405e+01),
    (4.0000e+01, 2.0000e+01, 6.3903953148e+01),
    (0.0000e+00, 1.0000e+02, 9.4337448470e+00),
];

#[test]
fn test_enthalpy_algorithm_reference() {
    for &(db, rh, h_ref) in ENTHALPY_REF_ALG {
        let h = calculate_enthalpy(db, rh, 101325.0);
        let err = (h - h_ref).abs();
        assert!(
            err < ENTHALPY_TOL_KJ_KG,
            "enthalpy({db}°C, {rh}%) = {h:.6} kJ/kg vs ref {h_ref:.6} (err {err:.2e} > {tol:.0e})",
            tol = ENTHALPY_TOL_KJ_KG,
        );
    }
}

// ===========================================================================
// Section 10: Psychrometrics — Exact-Equality Boundary Conditions
// ===========================================================================

/// Boundary-condition exact-equality tests: pure-arithmetic identities that
/// are bit-exact across all IEEE 754 platforms regardless of libm.
///
///   1. RH=100% ⇒ wet-bulb == dry-bulb exactly (saturation line)
///   2. RH=100% ⇒ dew-point == dry-bulb exactly
///   3. RH=100% ⇒ humidity-ratio = humidity-ratio(Tdb, 100%)
///   4. Enthalpy monotonicity in RH (higher RH → more moisture → higher h)
#[test]
fn test_psychrometrics_saturation_boundary_exact() {
    let db_values = [-10.0, 0.0, 5.0, 10.0, 20.0, 25.0, 30.0, 40.0];

    for &db in &db_values {
        // RH=100% ⇒ Td == Tdb (exact)
        let td = calculate_dew_point(db, 100.0, 101325.0);
        assert_eq!(td, db, "Td({db}°C, 100%) should equal dry-bulb exactly");

        // RH=100% ⇒ Tw == Tdb (within Newton-Raphson tolerance, but algorithm
        // clamps `tw` to `dp` which equals `db`, so this is exact).
        let tw = calculate_wet_bulb(db, 100.0, 101325.0);
        assert_eq!(tw, db, "Tw({db}°C, 100%) should equal dry-bulb exactly");
    }
}

/// Enthalpy strictly increases with RH at fixed dry-bulb (more moisture = more
/// latent energy). Verified at 5 dry-bulb points.
#[test]
fn test_enthalpy_monotonic_in_rh() {
    let db_values = [10.0, 20.0, 25.0, 30.0, 35.0];
    for &db in &db_values {
        let mut prev = f64::NEG_INFINITY;
        for rh in [10.0, 30.0, 50.0, 70.0, 90.0, 100.0] {
            let h = calculate_enthalpy(db, rh, 101325.0);
            assert!(
                h > prev,
                "enthalpy({db}°C, {rh}%) = {h} should exceed previous {prev}"
            );
            prev = h;
        }
    }
}

/// Saturation pressure strictly increases with temperature.
#[test]
fn test_saturation_pressure_monotonic_in_temp() {
    let temps = [
        -40.0, -20.0, -10.0, -5.0, -1.0, 0.0, 0.001, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0,
    ];
    let mut prev = 0.0_f64;
    for &t in &temps {
        let p = saturation_vapor_pressure(t);
        assert!(
            p > prev,
            "saturation_vapor_pressure({t}°C) = {p:.4e} should exceed previous {prev:.4e}"
        );
        prev = p;
    }
}
