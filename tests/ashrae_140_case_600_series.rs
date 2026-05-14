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

    // Issue #806: Enable ctf_primary for free-floating cases to properly capture
    // thermal mass dynamics. This is the same approach used in ashrae_140_free_floating.rs.
    // Without this, the 5R1C model under-predicts max temperatures for low-mass buildings.
    let is_free_floating = spec.is_free_floating();
    if is_free_floating {
        model.ctf_primary = true;
    }

    let mut min_temp = f64::MAX;
    let mut max_temp = f64::MIN;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        if let Some(&zone_temp) = model.temperatures.as_slice().first() {
            min_temp = min_temp.min(zone_temp);
            max_temp = max_temp.max(zone_temp);
        }
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
