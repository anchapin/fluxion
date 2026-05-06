//! Blind Validation Test Suite for ASHRAE 140
//!
//! This test suite measures the baseline failure state when all corrections
//! are disabled (ValidationMode::Blind). This is part of the ASHRAE 140
//! Blind Validation Plan (v1.3) Phase A.2.
//!
//! # Expected Result
//! ~0% pass rate when corrections are disabled - the corrections are what
//! make the current numbers look acceptable.
//!
//! # Usage
//! ```bash
//! cargo test --test ashrae_140_blind_validation -- --nocapture
//! ```

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::validation::benchmark;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

struct BlindValidationResult {
    case_id: String,
    metric: String,
    simulated_value: f64,
    reference_min: f64,
    reference_max: f64,
    percent_error: f64,
    within_tolerance: bool,
}

fn simulate_case_blind(
    spec: &fluxion::validation::ashrae_140_cases::CaseSpec,
) -> CaseResultsBlinded {
    let mut model = ThermalModel::<VectorField>::from_spec(spec);

    model.reset_peak_power();
    model.reset_heating_cooling_energy();

    const STEPS: usize = 8760;
    let num_zones = model.num_zones;
    let is_free_floating = spec.is_free_floating();

    if is_free_floating {
        model.heating_setpoint = -999.0;
        model.cooling_setpoint = 999.0;
        model.hvac_heating_capacity = 0.0;
        model.hvac_cooling_capacity = 0.0;
    }

    let mut hvac_enabled_vals = vec![1.0; num_zones];
    if !spec.hvac.is_empty() {
        for (zone_idx, hvac) in spec.hvac.iter().enumerate() {
            if zone_idx < num_zones {
                hvac_enabled_vals[zone_idx] = if hvac.is_enabled() { 1.0 } else { 0.0 };
            }
        }
    }
    model.hvac_enabled = VectorField::new(hvac_enabled_vals);

    let mut min_temp_celsius: f64 = f64::INFINITY;
    let mut max_temp_celsius: f64 = f64::NEG_INFINITY;

    for step in 0..STEPS {
        let hour_of_day = step % 24;

        let weather_data = DenverTmyWeather::new().get_hourly_data(step).unwrap();
        model.weather = Some(weather_data.clone());

        if let Some(hvac_schedule) = spec.hvac.first() {
            let hour = hour_of_day as u8;
            let heating_sp = hvac_schedule
                .heating_setpoint_at_hour(hour)
                .unwrap_or(hvac_schedule.heating_setpoint);
            let cooling_sp = model.cooling_schedule.value(hour as usize);
            model.heating_setpoint = heating_sp;
            model.cooling_setpoint = cooling_sp;
        }

        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        if is_free_floating {
            if let Some(&zone_temp) = model.temperatures.as_slice().first() {
                min_temp_celsius = min_temp_celsius.min(zone_temp);
                max_temp_celsius = max_temp_celsius.max(zone_temp);
            }
        }
    }

    CaseResultsBlinded {
        min_temp_celsius: if is_free_floating && min_temp_celsius != f64::INFINITY {
            Some(min_temp_celsius)
        } else {
            None
        },
        max_temp_celsius: if is_free_floating && max_temp_celsius != f64::NEG_INFINITY {
            Some(max_temp_celsius)
        } else {
            None
        },
        annual_heating_mwh: model.annual_heating_energy / 1000.0,
        annual_cooling_mwh: model.annual_cooling_energy / 1000.0,
        peak_heating_kw: model.get_peak_heating_power_kw(),
        peak_cooling_kw: model.get_peak_cooling_power_kw(),
    }
}

struct CaseResultsBlinded {
    min_temp_celsius: Option<f64>,
    max_temp_celsius: Option<f64>,
    annual_heating_mwh: f64,
    annual_cooling_mwh: f64,
    peak_heating_kw: f64,
    peak_cooling_kw: f64,
}

fn run_blind_validation() -> Vec<BlindValidationResult> {
    let mut results = Vec::new();
    let benchmark_data = benchmark::get_all_benchmark_data();

    let cases = vec![
        ASHRAE140Case::Case600,
        ASHRAE140Case::Case610,
        ASHRAE140Case::Case620,
        ASHRAE140Case::Case630,
        ASHRAE140Case::Case640,
        ASHRAE140Case::Case650,
        ASHRAE140Case::Case600FF,
        ASHRAE140Case::Case650FF,
        ASHRAE140Case::Case900,
        ASHRAE140Case::Case910,
        ASHRAE140Case::Case920,
        ASHRAE140Case::Case930,
        ASHRAE140Case::Case940,
        ASHRAE140Case::Case950,
        ASHRAE140Case::Case900FF,
        ASHRAE140Case::Case950FF,
        ASHRAE140Case::Case960,
        ASHRAE140Case::Case195,
    ];

    for case in cases {
        let case_id = case.number();
        let spec = case.spec();
        let is_free_floating = spec.is_free_floating();

        let sim_results = simulate_case_blind(&spec);

        if let Some(data) = benchmark_data.get(&case_id) {
            if is_free_floating {
                if let Some(min_temp) = sim_results.min_temp_celsius {
                    let ref_mid = (data.min_free_float_min + data.min_free_float_max) / 2.0;
                    let percent_error = ((min_temp - ref_mid) / ref_mid * 100.0).abs();
                    let within_tolerance =
                        min_temp >= data.min_free_float_min && min_temp <= data.min_free_float_max;

                    results.push(BlindValidationResult {
                        case_id: case_id.clone(),
                        metric: "MinFreeFloat".to_string(),
                        simulated_value: min_temp,
                        reference_min: data.min_free_float_min,
                        reference_max: data.min_free_float_max,
                        percent_error,
                        within_tolerance,
                    });
                }

                if let Some(max_temp) = sim_results.max_temp_celsius {
                    let ref_mid = (data.max_free_float_min + data.max_free_float_max) / 2.0;
                    let percent_error = ((max_temp - ref_mid) / ref_mid * 100.0).abs();
                    let within_tolerance =
                        max_temp >= data.max_free_float_min && max_temp <= data.max_free_float_max;

                    results.push(BlindValidationResult {
                        case_id: case_id.clone(),
                        metric: "MaxFreeFloat".to_string(),
                        simulated_value: max_temp,
                        reference_min: data.max_free_float_min,
                        reference_max: data.max_free_float_max,
                        percent_error,
                        within_tolerance,
                    });
                }
            } else {
                let metrics = vec![
                    (
                        "AnnualHeating",
                        sim_results.annual_heating_mwh,
                        data.annual_heating_min,
                        data.annual_heating_max,
                    ),
                    (
                        "AnnualCooling",
                        sim_results.annual_cooling_mwh,
                        data.annual_cooling_min,
                        data.annual_cooling_max,
                    ),
                    (
                        "PeakHeating",
                        sim_results.peak_heating_kw,
                        data.peak_heating_min,
                        data.peak_heating_max,
                    ),
                    (
                        "PeakCooling",
                        sim_results.peak_cooling_kw,
                        data.peak_cooling_min,
                        data.peak_cooling_max,
                    ),
                ];

                for (metric_name, value, ref_min, ref_max) in metrics {
                    if ref_min > 0.0 || ref_max > 0.0 {
                        let ref_mid = (ref_min + ref_max) / 2.0;
                        let percent_error = if ref_mid.abs() > 1e-10 {
                            ((value - ref_mid) / ref_mid * 100.0).abs()
                        } else {
                            0.0
                        };
                        let within_tolerance = value >= ref_min && value <= ref_max;

                        results.push(BlindValidationResult {
                            case_id: case_id.clone(),
                            metric: metric_name.to_string(),
                            simulated_value: value,
                            reference_min: ref_min,
                            reference_max: ref_max,
                            percent_error,
                            within_tolerance,
                        });
                    }
                }
            }
        }
    }

    results
}

fn print_results_table(results: &[BlindValidationResult]) {
    println!("\n====================================================================================================");
    println!("ASHRAE 140 BLIND VALIDATION BASELINE RESULTS");
    println!("(All corrections disabled)");
    println!("====================================================================================================");
    println!();

    println!(
        "{:>8} {:>15} {:>15} {:>15} {:>15} {:>12} {:>10}",
        "Case", "Metric", "Simulated", "Ref Min", "Ref Max", "% Error", "Status"
    );
    println!("----------------------------------------------------------------------------------------------------");

    for r in results {
        let status = if r.within_tolerance { "PASS" } else { "FAIL" };
        println!(
            "{:>8} {:>15} {:>15.4} {:>15.4} {:>15.4} {:>12.2} {:>10}",
            r.case_id,
            r.metric,
            r.simulated_value,
            r.reference_min,
            r.reference_max,
            r.percent_error,
            status
        );
    }

    println!("----------------------------------------------------------------------------------------------------");
}

fn compute_summary_statistics(results: &[BlindValidationResult]) -> (usize, usize, f64, f64) {
    let total = results.len();
    let passed = results.iter().filter(|r| r.within_tolerance).count();
    let failed = total - passed;
    let pass_rate = if total > 0 {
        passed as f64 / total as f64 * 100.0
    } else {
        0.0
    };

    let mae = if !results.is_empty() {
        results.iter().map(|r| r.percent_error).sum::<f64>() / total as f64
    } else {
        0.0
    };

    (passed, failed, pass_rate, mae)
}

fn categorize_failures(
    results: &[BlindValidationResult],
) -> std::collections::HashMap<String, Vec<&BlindValidationResult>> {
    let mut categories: std::collections::HashMap<String, Vec<&BlindValidationResult>> =
        std::collections::HashMap::new();

    for r in results.iter().filter(|r| !r.within_tolerance) {
        let category = if r.case_id.ends_with("FF") {
            "free-float".to_string()
        } else if r.case_id.starts_with("9") {
            "high-mass".to_string()
        } else if r.case_id.starts_with("6") {
            "low-mass".to_string()
        } else {
            "special".to_string()
        };

        categories.entry(category).or_default().push(r);
    }

    categories
}

#[test]
fn test_blind_validation_baseline() {
    println!("\nStarting ASHRAE 140 Blind Validation Baseline Measurement");
    println!("Expected: ~0% pass rate (corrections make current numbers look acceptable)\n");

    let results = run_blind_validation();

    print_results_table(&results);

    let (passed, failed, pass_rate, mae) = compute_summary_statistics(&results);

    println!("\n====================================================================================================");
    println!("SUMMARY STATISTICS");
    println!("====================================================================================================");
    println!("Total metrics: {}", results.len());
    println!("Passed: {}", passed);
    println!("Failed: {}", failed);
    println!("Pass rate: {:.2}%", pass_rate);
    println!("Mean Absolute Error: {:.2}%", mae);
    println!();

    let categories = categorize_failures(&results);
    println!("Failure categories:");
    for (category, failures) in &categories {
        println!("  {}: {} failures", category, failures.len());
        for f in failures.iter().take(3) {
            println!(
                "    - Case {} {}: {:.2}% error",
                f.case_id, f.metric, f.percent_error
            );
        }
        if failures.len() > 3 {
            println!("    ... and {} more", failures.len() - 3);
        }
    }
    println!("====================================================================================================");

    if pass_rate > 50.0 {
        println!("\nWARNING: Pass rate is higher than expected for blind validation.");
        println!("This suggests corrections may not be fully disabled.");
    }
}
