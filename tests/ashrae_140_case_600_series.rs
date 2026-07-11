//! ASHRAE 140 Case 600 Series Validation Tests
//!
//! This module provides validation tests for the 600-series (low-mass) cases:
//! - Case 610: South shading
//! - Case 620: East/West windows
//! - Case 630: East/West shading
//! - Case 640: Thermostat setback
//! - Case 650: Night ventilation
//! - Case 600FF: Free-floating
//! - Case 650FF: Free-floating with night ventilation
//!
//! These tests validate that fluxion's simulation results fall within the
//! ASHRAE 140 reference ranges for annual energy, peak loads, and
//! free-floating temperatures.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::WeatherSource;

const J_TO_MWH: f64 = 1.0 / 3.6e9;

struct CaseReference {
    #[allow(dead_code)]
    case_id: &'static str,
    annual_heating_min: f64,
    annual_heating_max: f64,
    annual_cooling_min: f64,
    annual_cooling_max: f64,
    peak_heating_min: f64,
    peak_heating_max: f64,
    peak_cooling_min: f64,
    peak_cooling_max: f64,
    min_free_float_min: f64,
    min_free_float_max: f64,
    max_free_float_min: f64,
    max_free_float_max: f64,
}

const CASE_610: CaseReference = CaseReference {
    case_id: "610",
    annual_heating_min: 4.36,
    annual_heating_max: 5.79,
    annual_cooling_min: 3.92,
    annual_cooling_max: 6.14,
    peak_heating_min: 4.30,
    peak_heating_max: 5.70,
    peak_cooling_min: 2.20,
    peak_cooling_max: 2.90,
    min_free_float_min: -19.2,
    min_free_float_max: -16.0,
    max_free_float_min: 60.2,
    max_free_float_max: 68.9,
};

const CASE_620: CaseReference = CaseReference {
    case_id: "620",
    annual_heating_min: 4.5,
    annual_heating_max: 6.5,
    annual_cooling_min: 3.2,
    annual_cooling_max: 5.0,
    peak_heating_min: 2.8,
    peak_heating_max: 3.8,
    peak_cooling_min: 2.5,
    peak_cooling_max: 3.5,
    min_free_float_min: -18.5,
    min_free_float_max: -15.3,
    max_free_float_min: 62.8,
    max_free_float_max: 71.5,
};

const CASE_630: CaseReference = CaseReference {
    case_id: "630",
    annual_heating_min: 5.05,
    annual_heating_max: 6.47,
    annual_cooling_min: 2.13,
    annual_cooling_max: 3.70,
    peak_heating_min: 4.70,
    peak_heating_max: 6.10,
    peak_cooling_min: 1.80,
    peak_cooling_max: 2.40,
    min_free_float_min: -18.0,
    min_free_float_max: -14.8,
    max_free_float_min: 58.5,
    max_free_float_max: 66.2,
};

const CASE_640: CaseReference = CaseReference {
    case_id: "640",
    annual_heating_min: 2.75,
    annual_heating_max: 3.80,
    annual_cooling_min: 5.95,
    annual_cooling_max: 8.10,
    peak_heating_min: 4.30,
    peak_heating_max: 5.70,
    peak_cooling_min: 2.80,
    peak_cooling_max: 3.70,
    min_free_float_min: -18.6,
    min_free_float_max: -15.4,
    max_free_float_min: 63.5,
    max_free_float_max: 72.8,
};

const CASE_650: CaseReference = CaseReference {
    case_id: "650",
    annual_heating_min: 0.0,
    annual_heating_max: 0.0,
    annual_cooling_min: 4.82,
    annual_cooling_max: 7.06,
    peak_heating_min: 0.0,
    peak_heating_max: 0.0,
    peak_cooling_min: 1.90,
    peak_cooling_max: 2.50,
    min_free_float_min: -23.0,
    min_free_float_max: -21.0,
    max_free_float_min: 58.8,
    max_free_float_max: 67.5,
};

const CASE_600FF: CaseReference = CaseReference {
    case_id: "600FF",
    annual_heating_min: 0.0,
    annual_heating_max: 0.0,
    annual_cooling_min: 0.0,
    annual_cooling_max: 0.0,
    peak_heating_min: 0.0,
    peak_heating_max: 0.0,
    peak_cooling_min: 0.0,
    peak_cooling_max: 0.0,
    min_free_float_min: -18.8,
    min_free_float_max: -15.6,
    max_free_float_min: 64.9,
    max_free_float_max: 75.1,
};

const CASE_650FF: CaseReference = CaseReference {
    case_id: "650FF",
    annual_heating_min: 0.0,
    annual_heating_max: 0.0,
    annual_cooling_min: 0.0,
    annual_cooling_max: 0.0,
    peak_heating_min: 0.0,
    peak_heating_max: 0.0,
    peak_cooling_min: 0.0,
    peak_cooling_max: 0.0,
    min_free_float_min: -23.0,
    min_free_float_max: -21.0,
    max_free_float_min: 63.2,
    max_free_float_max: 73.5,
};

fn run_annual_simulation(case_enum: ASHRAE140Case) -> (f64, f64, f64, f64) {
    let spec = case_enum.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = fluxion::weather::epw::EpwWeatherSource::from_file("assets/weather/WD600.epw")
        .expect("Failed to load EPW weather data");

    // ASHRAE 140 §B2 specifies a warm-up period before collecting annual metrics
    // (typically a full year repeat of the weather). The previous implementation
    // ran a single 8760-hour pass from default initial conditions, which produced
    // spurious "phantom" energy in the first year — the mass node started at 20°C
    // regardless of season and absorbed/released heat until it converged. The
    // warm-up below drops that phantom energy without changing the steady-state
    // physics, and aligns the test methodology with `case_900_multinode_validation`.
    //
    // Issue #1457 investigation: the warm-up also smooths the peak-load metric,
    // which had a single-timestep spike at hour 0 from the mass starting 20°C
    // above the winter outdoor temperature.
    const WARMUP_STEPS: usize = 14 * 24; // 14-day warm-up, matches Case 900 test
    for step in 0..WARMUP_STEPS {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.weather = Some(weather_data.clone());
        let _ = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
    }

    let mut total_heating = 0.0_f64;
    let mut total_cooling = 0.0_f64;
    let mut peak_heating = 0.0_f64;
    let mut peak_cooling = 0.0_f64;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.weather = Some(weather_data.clone());
        let _zone_temp_before = model
            .temperatures
            .as_slice()
            .first()
            .copied()
            .unwrap_or(20.0);

        let energy_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
        let energy_joules = energy_kwh * 3.6e6;

        // Classify energy based on HVAC output sign, not zone temperature
        // Positive = heating energy, Negative = cooling energy
        if energy_kwh > 0.0 {
            total_heating += energy_joules;
            let power_watts = energy_joules / 3600.0;
            peak_heating = peak_heating.max(power_watts);
        } else if energy_kwh < 0.0 {
            total_cooling += -energy_joules;
            let power_watts = -energy_joules / 3600.0;
            peak_cooling = peak_cooling.max(power_watts);
        }
    }

    (total_heating, total_cooling, peak_heating, peak_cooling)
}

fn run_free_floating_simulation(case_enum: ASHRAE140Case) -> (f64, f64) {
    let spec = case_enum.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = fluxion::weather::epw::EpwWeatherSource::from_file("assets/weather/WD600.epw")
        .expect("Failed to load EPW weather data");

    // Issue #806 fix: For low-mass cases (600FF, 650FF), DO NOT enable ctf_primary.
    // The 5R1C model provides correct free-floating temperatures.
    // ctf_primary=true was incorrectly added here and DISABLES the mass coupling
    // in the 6R2C path, causing temperatures to be damped incorrectly.
    // For 600FF/650FF, the standard 5R1C path should be used.
    // Note: 900FF/950FF have CTF solvers initialized in from_spec and correctly use ctf_primary.

    #[cfg(feature = "pr821-diag")]
    let mut diag = fluxion::sim::pr821_diag::DiagCollector::new(spec.case_id.clone());

    let mut min_temp = f64::MAX;
    let mut max_temp = f64::MIN;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.weather = Some(weather_data.clone());
        let outdoor_temp = weather_data.dry_bulb_temp;
        model.step_physics(step, outdoor_temp, 3600.0);

        if let Some(&zone_temp) = model.temperatures.as_slice().first() {
            min_temp = min_temp.min(zone_temp);
            max_temp = max_temp.max(zone_temp);
        }

        #[cfg(feature = "pr821-diag")]
        {
            // The model exposes solar_gains as W/m² of zone floor area; convert to
            // total window-attributable W for the diagnostic CSV.
            let solar_w_per_m2 = model.solar_gains.as_slice().first().copied().unwrap_or(0.0);
            let zone_area = model.zone_area.as_slice().first().copied().unwrap_or(1.0);
            let solar_window_w = solar_w_per_m2 * zone_area;

            let hour = (step % 24) as u8;
            let night_vent_active = spec
                .night_ventilation
                .as_ref()
                .is_some_and(|nv| nv.is_active_at_hour(hour));

            // Issue #825: phi_ia / phi_st / phi_m are now captured inside
            // `step_physics_5r1c` and exposed on `model.0` when the
            // `pr821-diag` feature is enabled. Read the most-recent zone-0
            // values directly from the model so the CSV row matches the same
            // timestep as the recorded temperatures.
            diag.record(
                step,
                &model,
                outdoor_temp,
                solar_window_w,
                model.0.last_phi_ia,
                model.0.last_phi_st,
                model.0.last_phi_m,
                night_vent_active,
                0.0, // hvac_out_w (always 0 for FF — guarded by free_float assert)
            );
        }
    }

    #[cfg(feature = "pr821-diag")]
    {
        // Issue #825 acceptance: at least one daytime row (10:00–16:00 local
        // hour) must have a non-zero phi_m (W to mass node) for cases that
        // see daylight solar gains (600FF, 650FF). A non-zero value proves
        // the solar/internal-gain routing is being captured in the diagnostic
        // CSV — a previous iteration left these as placeholder zeros.
        let any_daytime_phi_m = diag
            .rows()
            .iter()
            .any(|r| (10..=16).contains(&r.hour) && r.phi_m.abs() > f64::EPSILON);
        assert!(
            any_daytime_phi_m,
            "[pr821-diag] case={} has no daytime row with non-zero phi_m;              phi_* capture in step_physics_5r1c may have regressed (see #825)",
            spec.case_id
        );

        let path = diag
            .flush_to_csv()
            .expect("PR #821 diagnostic CSV write failed");
        eprintln!(
            "[pr821-diag] case={} rows={} -> {}",
            spec.case_id,
            diag.len(),
            path.display()
        );
    }

    (min_temp, max_temp)
}

// Case 610 tests
mod case_610 {
    use super::*;

    #[test]
    fn test_annual_heating() {
        let r = CASE_610;
        let (heating_j, _, _, _) = run_annual_simulation(ASHRAE140Case::Case610);
        let heating_mwh = heating_j * J_TO_MWH;
        println!(
            "Case 610 Annual Heating: {:.2} MWh (Ref: {:.2}-{:.2})",
            heating_mwh, r.annual_heating_min, r.annual_heating_max
        );
        assert!(heating_mwh >= r.annual_heating_min && heating_mwh <= r.annual_heating_max);
    }

    #[test]
    fn test_annual_cooling() {
        let r = CASE_610;
        let (_, cooling_j, _, _) = run_annual_simulation(ASHRAE140Case::Case610);
        let cooling_mwh = cooling_j * J_TO_MWH;
        println!(
            "Case 610 Annual Cooling: {:.2} MWh (Ref: {:.2}-{:.2})",
            cooling_mwh, r.annual_cooling_min, r.annual_cooling_max
        );
        assert!(cooling_mwh >= r.annual_cooling_min && cooling_mwh <= r.annual_cooling_max);
    }

    #[test]
    fn test_peak_heating() {
        let r = CASE_610;
        let (_, _, peak_h, _) = run_annual_simulation(ASHRAE140Case::Case610);
        let peak_h_kw = peak_h / 1000.0;
        println!(
            "Case 610 Peak Heating: {:.2} kW (Ref: {:.2}-{:.2})",
            peak_h_kw, r.peak_heating_min, r.peak_heating_max
        );
        assert!(peak_h_kw >= r.peak_heating_min && peak_h_kw <= r.peak_heating_max);
    }

    #[test]
    fn test_peak_cooling() {
        let r = CASE_610;
        let (_, _, _, peak_c) = run_annual_simulation(ASHRAE140Case::Case610);
        let peak_c_kw = peak_c / 1000.0;
        println!(
            "Case 610 Peak Cooling: {:.2} kW (Ref: {:.2}-{:.2})",
            peak_c_kw, r.peak_cooling_min, r.peak_cooling_max
        );
        assert!(peak_c_kw >= r.peak_cooling_min && peak_c_kw <= r.peak_cooling_max);
    }
}

// Case 620 tests
mod case_620 {
    use super::*;

    #[test]
    fn test_annual_heating() {
        let r = CASE_620;
        let (heating_j, _, _, _) = run_annual_simulation(ASHRAE140Case::Case620);
        let heating_mwh = heating_j * J_TO_MWH;
        println!(
            "Case 620 Annual Heating: {:.2} MWh (Ref: {:.2}-{:.2})",
            heating_mwh, r.annual_heating_min, r.annual_heating_max
        );
        assert!(heating_mwh >= r.annual_heating_min && heating_mwh <= r.annual_heating_max);
    }

    #[test]
    fn test_annual_cooling() {
        let r = CASE_620;
        let (_, cooling_j, _, _) = run_annual_simulation(ASHRAE140Case::Case620);
        let cooling_mwh = cooling_j * J_TO_MWH;
        println!(
            "Case 620 Annual Cooling: {:.2} MWh (Ref: {:.2}-{:.2})",
            cooling_mwh, r.annual_cooling_min, r.annual_cooling_max
        );
        assert!(cooling_mwh >= r.annual_cooling_min && cooling_mwh <= r.annual_cooling_max);
    }

    #[test]
    fn test_peak_heating() {
        let r = CASE_620;
        let (_, _, peak_h, _) = run_annual_simulation(ASHRAE140Case::Case620);
        let peak_h_kw = peak_h / 1000.0;
        println!(
            "Case 620 Peak Heating: {:.2} kW (Ref: {:.2}-{:.2})",
            peak_h_kw, r.peak_heating_min, r.peak_heating_max
        );
        assert!(peak_h_kw >= r.peak_heating_min && peak_h_kw <= r.peak_heating_max);
    }

    #[test]
    fn test_peak_cooling() {
        let r = CASE_620;
        let (_, _, _, peak_c) = run_annual_simulation(ASHRAE140Case::Case620);
        let peak_c_kw = peak_c / 1000.0;
        println!(
            "Case 620 Peak Cooling: {:.2} kW (Ref: {:.2}-{:.2})",
            peak_c_kw, r.peak_cooling_min, r.peak_cooling_max
        );
        assert!(peak_c_kw >= r.peak_cooling_min && peak_c_kw <= r.peak_cooling_max);
    }
}

// Case 630 tests
mod case_630 {
    use super::*;

    #[test]
    fn test_annual_heating() {
        let r = CASE_630;
        let (heating_j, _, _, _) = run_annual_simulation(ASHRAE140Case::Case630);
        let heating_mwh = heating_j * J_TO_MWH;
        println!(
            "Case 630 Annual Heating: {:.2} MWh (Ref: {:.2}-{:.2})",
            heating_mwh, r.annual_heating_min, r.annual_heating_max
        );
        assert!(heating_mwh >= r.annual_heating_min && heating_mwh <= r.annual_heating_max);
    }

    #[test]
    fn test_annual_cooling() {
        let r = CASE_630;
        let (_, cooling_j, _, _) = run_annual_simulation(ASHRAE140Case::Case630);
        let cooling_mwh = cooling_j * J_TO_MWH;
        println!(
            "Case 630 Annual Cooling: {:.2} MWh (Ref: {:.2}-{:.2})",
            cooling_mwh, r.annual_cooling_min, r.annual_cooling_max
        );
        assert!(cooling_mwh >= r.annual_cooling_min && cooling_mwh <= r.annual_cooling_max);
    }

    #[test]
    fn test_peak_heating() {
        let r = CASE_630;
        let (_, _, peak_h, _) = run_annual_simulation(ASHRAE140Case::Case630);
        let peak_h_kw = peak_h / 1000.0;
        println!(
            "Case 630 Peak Heating: {:.2} kW (Ref: {:.2}-{:.2})",
            peak_h_kw, r.peak_heating_min, r.peak_heating_max
        );
        assert!(peak_h_kw >= r.peak_heating_min && peak_h_kw <= r.peak_heating_max);
    }

    #[test]
    fn test_peak_cooling() {
        let r = CASE_630;
        let (_, _, _, peak_c) = run_annual_simulation(ASHRAE140Case::Case630);
        let peak_c_kw = peak_c / 1000.0;
        println!(
            "Case 630 Peak Cooling: {:.2} kW (Ref: {:.2}-{:.2})",
            peak_c_kw, r.peak_cooling_min, r.peak_cooling_max
        );
        assert!(peak_c_kw >= r.peak_cooling_min && peak_c_kw <= r.peak_cooling_max);
    }
}

// Case 640 tests
mod case_640 {
    use super::*;

    #[test]
    fn test_annual_heating() {
        let r = CASE_640;
        let (heating_j, _, _, _) = run_annual_simulation(ASHRAE140Case::Case640);
        let heating_mwh = heating_j * J_TO_MWH;
        println!(
            "Case 640 Annual Heating: {:.2} MWh (Ref: {:.2}-{:.2})",
            heating_mwh, r.annual_heating_min, r.annual_heating_max
        );
        assert!(heating_mwh >= r.annual_heating_min && heating_mwh <= r.annual_heating_max);
    }

    #[test]
    fn test_annual_cooling() {
        let r = CASE_640;
        let (_, cooling_j, _, _) = run_annual_simulation(ASHRAE140Case::Case640);
        let cooling_mwh = cooling_j * J_TO_MWH;
        println!(
            "Case 640 Annual Cooling: {:.2} MWh (Ref: {:.2}-{:.2})",
            cooling_mwh, r.annual_cooling_min, r.annual_cooling_max
        );
        assert!(cooling_mwh >= r.annual_cooling_min && cooling_mwh <= r.annual_cooling_max);
    }

    #[test]
    fn test_peak_heating() {
        let r = CASE_640;
        let (_, _, peak_h, _) = run_annual_simulation(ASHRAE140Case::Case640);
        let peak_h_kw = peak_h / 1000.0;
        println!(
            "Case 640 Peak Heating: {:.2} kW (Ref: {:.2}-{:.2})",
            peak_h_kw, r.peak_heating_min, r.peak_heating_max
        );
        assert!(peak_h_kw >= r.peak_heating_min && peak_h_kw <= r.peak_heating_max);
    }

    #[test]
    fn test_peak_cooling() {
        let r = CASE_640;
        let (_, _, _, peak_c) = run_annual_simulation(ASHRAE140Case::Case640);
        let peak_c_kw = peak_c / 1000.0;
        println!(
            "Case 640 Peak Cooling: {:.2} kW (Ref: {:.2}-{:.2})",
            peak_c_kw, r.peak_cooling_min, r.peak_cooling_max
        );
        assert!(peak_c_kw >= r.peak_cooling_min && peak_c_kw <= r.peak_cooling_max);
    }
}

// Case 650 tests
mod case_650 {
    use super::*;

    #[test]
    fn test_annual_heating_near_zero() {
        let r = CASE_650;
        let (heating_j, _, _, _) = run_annual_simulation(ASHRAE140Case::Case650);
        let heating_mwh = heating_j * J_TO_MWH;
        println!(
            "Case 650 Annual Heating: {:.2} MWh (Ref: {:.2}-{:.2})",
            heating_mwh, r.annual_heating_min, r.annual_heating_max
        );
        assert!(heating_mwh <= 0.01, "Case 650 heating should be near zero");
    }

    #[test]
    fn test_annual_cooling() {
        let r = CASE_650;
        let (_, cooling_j, _, _) = run_annual_simulation(ASHRAE140Case::Case650);
        let cooling_mwh = cooling_j * J_TO_MWH;
        println!(
            "Case 650 Annual Cooling: {:.2} MWh (Ref: {:.2}-{:.2})",
            cooling_mwh, r.annual_cooling_min, r.annual_cooling_max
        );
        assert!(cooling_mwh >= r.annual_cooling_min && cooling_mwh <= r.annual_cooling_max);
    }

    #[test]
    fn test_peak_cooling() {
        let r = CASE_650;
        let (_, _, _, peak_c) = run_annual_simulation(ASHRAE140Case::Case650);
        let peak_c_kw = peak_c / 1000.0;
        println!(
            "Case 650 Peak Cooling: {:.2} kW (Ref: {:.2}-{:.2})",
            peak_c_kw, r.peak_cooling_min, r.peak_cooling_max
        );
        assert!(peak_c_kw >= r.peak_cooling_min && peak_c_kw <= r.peak_cooling_max);
    }
}

// Case 600FF tests
mod case_600ff {
    use super::*;

    #[test]
    fn test_min_temperature() {
        let r = CASE_600FF;
        let (min_temp, _) = run_free_floating_simulation(ASHRAE140Case::Case600FF);
        println!(
            "Case 600FF Min Temp: {:.2}°C (Ref: {:.2} to {:.2})",
            min_temp, r.min_free_float_min, r.min_free_float_max
        );
        assert!(min_temp >= r.min_free_float_min && min_temp <= r.min_free_float_max);
    }

    #[test]
    fn test_max_temperature() {
        let r = CASE_600FF;
        let (_, max_temp) = run_free_floating_simulation(ASHRAE140Case::Case600FF);
        println!(
            "Case 600FF Max Temp: {:.2}°C (Ref: {:.2} to {:.2})",
            max_temp, r.max_free_float_min, r.max_free_float_max
        );
        assert!(max_temp >= r.max_free_float_min && max_temp <= r.max_free_float_max);
    }
}

// Case 650FF tests
mod case_650ff {
    use super::*;

    #[test]
    fn test_min_temperature() {
        let r = CASE_650FF;
        let (min_temp, _) = run_free_floating_simulation(ASHRAE140Case::Case650FF);
        println!(
            "Case 650FF Min Temp: {:.2}°C (Ref: {:.2} to {:.2})",
            min_temp, r.min_free_float_min, r.min_free_float_max
        );
        assert!(min_temp >= r.min_free_float_min && min_temp <= r.min_free_float_max);
    }

    #[test]
    fn test_max_temperature() {
        let r = CASE_650FF;
        let (_, max_temp) = run_free_floating_simulation(ASHRAE140Case::Case650FF);
        println!(
            "Case 650FF Max Temp: {:.2}°C (Ref: {:.2} to {:.2})",
            max_temp, r.max_free_float_min, r.max_free_float_max
        );
        assert!(max_temp >= r.max_free_float_min && max_temp <= r.max_free_float_max);
    }
}

// ===== Issue #821 / Issue #738 — Free-Float HVAC Zero-Output Regression Test =====
// Any case where `is_free_floating()` is true must report exactly zero annual
// heating energy and zero annual cooling energy. This guards against regressions
// where a hidden HVAC code path leaks into FF cases (the historical root cause
// of #725 / #738). It is paired with a hard `assert!` inside `step_physics_5r1c`
// and `step_physics_6r2c` (gated under `cfg(test)`) so the moment any code path
// produces non-zero HVAC under FF, this test panics with an actionable message.
mod free_float_hvac_guard {
    use super::*;
    use fluxion::validation::ashrae_140_cases::CaseBuilder;

    fn run_and_check_ff_zero(case_enum: ASHRAE140Case) {
        let spec = case_enum.spec();
        assert!(
            spec.is_free_floating(),
            "Case {} is not classified as free-floating; test misconfigured",
            spec.case_id
        );

        let mut model = ThermalModel::<VectorField>::from_spec(&spec);
        let weather =
            fluxion::weather::epw::EpwWeatherSource::from_file("assets/weather/WD600.epw")
                .expect("Failed to load EPW weather data");

        for step in 0..8760 {
            let weather_data = weather.get_hourly_data(step).unwrap();
            model.weather = Some(weather_data.clone());
            model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
        }

        // Both annual energy counters must be exactly zero (in the model's
        // internal kWh-equivalent accumulator). A non-zero value indicates an
        // HVAC code path bypassing the free_float guard.
        assert!(
            model.annual_heating_energy.abs() < 1e-9,
            "Free-float case {} produced non-zero annual heating energy: {} kWh",
            spec.case_id,
            model.annual_heating_energy
        );
        assert!(
            model.annual_cooling_energy.abs() < 1e-9,
            "Free-float case {} produced non-zero annual cooling energy: {} kWh",
            spec.case_id,
            model.annual_cooling_energy
        );
    }

    #[test]
    fn case_600ff_zero_hvac_energy() {
        run_and_check_ff_zero(ASHRAE140Case::Case600FF);
    }

    #[test]
    fn case_650ff_zero_hvac_energy() {
        run_and_check_ff_zero(ASHRAE140Case::Case650FF);
    }

    #[test]
    fn builders_match_free_float_classification() {
        // Sanity check the builder API agrees with the case enum mapping for
        // 600FF and 650FF. Catches future builder edits that could silently
        // turn an FF case into a setpoint case.
        let s_600 = CaseBuilder::case_600ff();
        assert!(
            s_600.is_free_floating(),
            "case_600ff() must be free-floating"
        );
        let s_650 = CaseBuilder::case_650ff();
        assert!(
            s_650.is_free_floating(),
            "case_650ff() must be free-floating"
        );
    }
}

// ============================================================================
// TEMPORARY DIAGNOSTIC: Case 640 Hourly HVAC Trace (July Peak Week)
// ============================================================================
// This test instruments Case 640 to capture hourly T_free, T_zone, Q_cooling
// during the peak summer week (July 21-27) to diagnose the annual cooling gap.
#[test]
fn test_case_640_hourly_peak_week_diagnostic() {
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::PathBuf;

    let spec = ASHRAE140Case::Case640.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = fluxion::weather::epw::EpwWeatherSource::from_file("assets/weather/WD600.epw")
        .expect("Failed to load EPW weather data");

    // Peak summer week: July 21 (DOY=202) 00:00 to July 27 (DOY=208) 23:00
    let peak_start = (202 - 1) * 24; // July 21 00:00 = timestep 4824
    let peak_end = 208 * 24; // July 28 00:00 = timestep 4992

    let mut heating_total_j = 0.0;
    let mut cooling_total_j = 0.0;

    // h_tr_is for zone 0 (approx 583 W/K for low-mass)
    let h_tr_is = model.h_tr_is.as_slice().first().copied().unwrap_or(583.0);

    // CSV output
    let dir: PathBuf = ["target", "diag"].iter().collect();
    fs::create_dir_all(&dir).ok();
    let csv_path = dir.join("case_640_peak_week.csv");
    let mut csv = File::create(&csv_path).unwrap();
    writeln!(
        csv,
        "step,hour,day_of_year,outdoor_C,solar_W,T_zone_C,hvac_out_W,Q_cooling_W,t_free_C,T_sp_heat,T_sp_cool,heating_j,cooling_j"
    )
    .unwrap();

    eprintln!(
        "\n=== Case 640 Peak Week Diagnostic (steps {} to {}) ===",
        peak_start, peak_end
    );
    eprintln!(
        "{:>6} {:>6} {:>6} {:>10} {:>10} {:>10} {:>10}",
        "step", "hour", "DOY", "T_out", "T_zone", "hvac_W", "Q_cool_W"
    );

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.weather = Some(weather_data.clone());
        let outdoor_temp = weather_data.dry_bulb_temp;

        let energy_kwh = model.step_physics(step, outdoor_temp, 3600.0);
        let energy_j = energy_kwh * 3.6e6;

        // Accumulate heating and cooling over ALL timesteps (annual totals)
        if energy_kwh > 0.0 {
            heating_total_j += energy_j;
        } else if energy_kwh < 0.0 {
            cooling_total_j += -energy_j;
        }

        // Get zone temperature after HVAC
        let t_zone = model
            .temperatures
            .as_slice()
            .first()
            .copied()
            .unwrap_or(20.0);

        // Solar gain (W/m² -> total window-attributable W)
        let solar_w_per_m2 = model.solar_gains.as_slice().first().copied().unwrap_or(0.0);
        let zone_area = model.zone_area.as_slice().first().copied().unwrap_or(129.6);
        let total_solar_w = solar_w_per_m2 * zone_area;

        // hvac_out_w = energy_kwh * 1000 (dt=3600s, so J/s=W = kWh*3600/3600*1000 = kWh*1000)
        // energy_kwh is NEGATIVE for cooling, positive for heating
        let hvac_out_w = energy_kwh * 1000.0;

        // t_free = t_zone - hvac_out_w / h_tr_is  (from t_i_act = t_free + hvac/h_tr_is)
        // hvac_out_w is negative for cooling, so t_free = t_zone - (negative)/h_tr_is = t_zone + |hvac|/h_tr_is
        let t_free = if hvac_out_w.abs() > 1.0 {
            t_zone - hvac_out_w / h_tr_is
        } else {
            t_zone
        };

        if step >= peak_start && step < peak_end {
            let hour = step % 24;
            let day_of_year = step / 24 + 1;
            let q_cooling_w = if energy_kwh < 0.0 {
                -energy_kwh * 1000.0
            } else {
                0.0
            };

            // Query the HVAC schedule for current setpoints
            let t_sp_heat = model.heating_schedule.value(hour);
            let t_sp_cool = model.cooling_schedule.value(hour);

            writeln!(
                csv,
                "{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
                step,
                hour,
                day_of_year,
                outdoor_temp,
                total_solar_w,
                t_zone,
                hvac_out_w,
                q_cooling_w,
                t_free,
                t_sp_heat,
                t_sp_cool,
                if energy_kwh > 0.0 { energy_j } else { 0.0 },
                if energy_kwh < 0.0 { -energy_j } else { 0.0 }
            )
            .unwrap();

            // Print daytime hours (07:00 - 20:00) with setpoints
            if hour >= 7 && hour <= 20 {
                eprintln!(
                    "{:>6} {:>6} {:>6} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
                    step, hour, day_of_year, outdoor_temp, t_zone, hvac_out_w, t_sp_heat
                );
                eprintln!("  -> T_sp_cool={:.1}", t_sp_cool);
            }
        }
    }

    let annual_heating_mwh = heating_total_j / 3.6e9;
    let annual_cooling_mwh = cooling_total_j / 3.6e9;
    let peak_week_cooling_mwh = cooling_total_j / 3.6e9 / 50.0;

    eprintln!(
        "\n=== Annual Totals ===\n\
        Total Annual Heating: {:.3} MWh\n\
        Total Annual Cooling: {:.3} MWh (Ref: 5.95-8.10 MWh)\n\
        Peak Week Cooling: {:.3} MWh\n\
        CSV written to: {}",
        annual_heating_mwh,
        annual_cooling_mwh,
        peak_week_cooling_mwh,
        csv_path.display()
    );

    // 9R4C gives ~4.0 MWh (vs ref 5.95-8.10 MWh) — improved over 5R1C's 2.88 MWh
    // Remaining gap (~67% of ref) is the next phase's target (Issue #533 + HVAC/solar fixes)
    assert!(
        annual_cooling_mwh > 3.5,
        "Annual cooling {:.3} MWh should be > 3.5 MWh with 9R4C (ref: 5.95-8.10 MWh)",
        annual_cooling_mwh
    );
}
