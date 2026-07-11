//! Known Issues Regression Tests
//!
//! This module implements QG-03: Turn known issues into tracked regression cases.
//! Each test corresponds to an issue in docs/KNOWN_ISSUES.md and:
//! - Has a reproducer that demonstrates the issue
//! - Documents severity and hypothesis
//! - Uses #[test] with #[ignore] to quarantine known failures
//!
//! Definition of done: No high-severity issue exists only as prose;
//! every critical issue is machine-traceable.

use fluxion::validation::report::{MetricType, ValidationStatus};
use fluxion::validation::ASHRAE140Validator;

/// Issue #1457 tracking: the 14 Case 600-series metrics still out-of-band after
/// PR #1460 (which closed 6 of the original 16 via the ISO 13790 `h_coeff` fix).
///
/// These are NOT independent per-case bugs: grouped by metric they form a single
/// systematic signature — peak_cooling OVER (5/5), peak_heating UNDER (3/3),
/// annual_cooling UNDER (3/3), free-float min-temp too warm (2/2). That is the
/// discrete-node / 1-hour-timestep solar-injection limitation routed to the
/// GaugeSolver (#1465 / #1462) per the maintainer's #1457 direction update.
///
/// The test reproduces the exact metrics via the same `from_spec` + `step_physics`
/// path as `tests/ashrae_140_case_600_series.rs`, and is `#[ignore]`-quarantined.
/// It flips green when #1465 brings the 14 metrics into band, giving CI a concrete
/// close-out signal for #1457. Per the maintainer, forcing these into band with an
/// HVAC clamp / per-timestep bound is an anti-pattern and must NOT be used here.
mod issue_1457_case_600_series_tracking {
    use fluxion::physics::cta::VectorField;
    use fluxion::sim::engine::ThermalModel;
    use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
    use fluxion::weather::WeatherSource;

    const J_TO_MWH: f64 = 1.0 / 3.6e9;

    fn run_annual(case: ASHRAE140Case) -> (f64, f64, f64, f64) {
        let spec = case.spec();
        let mut model = ThermalModel::<VectorField>::from_spec(&spec);
        let weather =
            fluxion::weather::epw::EpwWeatherSource::from_file("assets/weather/WD600.epw")
                .expect("Failed to load EPW weather data");
        // 14-day warm-up (matches the post-#1457 fix in tests/ashrae_140_case_600_series.rs):
        // lets the 5R1C mass node settle from the 20°C default into the seasonal cycle
        // before we start collecting annual metrics, eliminating phantom first-year energy.
        const WARMUP_STEPS: usize = 14 * 24;
        for step in 0..WARMUP_STEPS {
            let w = weather.get_hourly_data(step).unwrap();
            model.weather = Some(w.clone());
            let _ = model.step_physics(step, w.dry_bulb_temp, 3600.0);
        }
        let (mut th, mut tc, mut ph, mut pc) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        for step in 0..8760 {
            let w = weather.get_hourly_data(step).unwrap();
            model.weather = Some(w.clone());
            let e_kwh = model.step_physics(step, w.dry_bulb_temp, 3600.0);
            let e_j = e_kwh * 3.6e6;
            if e_kwh > 0.0 {
                th += e_j;
                ph = ph.max(e_j / 3600.0);
            } else if e_kwh < 0.0 {
                tc += -e_j;
                pc = pc.max(-e_j / 3600.0);
            }
        }
        (th * J_TO_MWH, tc * J_TO_MWH, ph / 1000.0, pc / 1000.0)
    }

    fn run_min_temp(case: ASHRAE140Case) -> f64 {
        let spec = case.spec();
        let mut model = ThermalModel::<VectorField>::from_spec(&spec);
        let weather =
            fluxion::weather::epw::EpwWeatherSource::from_file("assets/weather/WD600.epw")
                .expect("Failed to load EPW weather data");
        let mut min_temp = f64::MAX;
        for step in 0..8760 {
            let w = weather.get_hourly_data(step).unwrap();
            model.weather = Some(w.clone());
            model.step_physics(step, w.dry_bulb_temp, 3600.0);
            if let Some(&t) = model.temperatures.as_slice().first() {
                min_temp = min_temp.min(t);
            }
        }
        min_temp
    }

    /// One quarantined guard covering all 14 currently-failing #1457 metrics.
    /// Assert-in-band on purpose: when #1465 lands, un-ignoring this proves closure.
    #[test]
    #[ignore = "#1457: 14 Case 600-series metrics await GaugeSolver #1465 (discrete-node solar injection)"]
    fn test_issue1457_remaining_600_series_metrics() {
        // (case, metric, band_lo, band_hi)
        let mut failures: Vec<String> = Vec::new();

        // (case, selector, lo, hi) for annual/peak metrics.
        // selector: 0=annual_heating 1=annual_cooling 2=peak_heating 3=peak_cooling
        let energy_metrics: [(ASHRAE140Case, &str, usize, f64, f64); 12] = [
            (ASHRAE140Case::Case610, "peak_heating", 2, 4.30, 5.70),
            (ASHRAE140Case::Case610, "peak_cooling", 3, 2.20, 2.90),
            (ASHRAE140Case::Case620, "annual_cooling", 1, 3.20, 5.00),
            (ASHRAE140Case::Case620, "peak_cooling", 3, 2.50, 3.50),
            (ASHRAE140Case::Case630, "peak_heating", 2, 4.70, 6.10),
            (ASHRAE140Case::Case630, "peak_cooling", 3, 1.80, 2.40),
            (ASHRAE140Case::Case640, "annual_heating", 0, 2.75, 3.80),
            (ASHRAE140Case::Case640, "annual_cooling", 1, 5.95, 8.10),
            (ASHRAE140Case::Case640, "peak_heating", 2, 4.30, 5.70),
            (ASHRAE140Case::Case640, "peak_cooling", 3, 2.80, 3.70),
            (ASHRAE140Case::Case650, "annual_cooling", 1, 4.82, 7.06),
            (ASHRAE140Case::Case650, "peak_cooling", 3, 1.90, 2.50),
        ];

        for (case, metric, sel, lo, hi) in energy_metrics {
            let (th, tc, ph, pc) = run_annual(case);
            let v = [th, tc, ph, pc][sel];
            let unit = if sel < 2 { "MWh" } else { "kW" };
            let ok = v >= lo && v <= hi;
            println!(
                "{:?} {}: {:.2} {} [ref {:.2}-{:.2}] {}",
                case,
                metric,
                v,
                unit,
                lo,
                hi,
                if ok { "PASS" } else { "FAIL" }
            );
            if !ok {
                failures.push(format!("{case:?}/{metric}={v:.2}{unit}"));
            }
        }

        // Free-float minimum temperatures (too warm — FREE-01/FREE-03 family).
        let ff_min: [(ASHRAE140Case, f64, f64); 2] = [
            (ASHRAE140Case::Case600FF, -18.80, -15.60),
            (ASHRAE140Case::Case650FF, -23.00, -21.00),
        ];
        for (case, lo, hi) in ff_min {
            let v = run_min_temp(case);
            let ok = v >= lo && v <= hi;
            println!(
                "{:?} min_free_float: {:.2}C [ref {:.2}..{:.2}] {}",
                case,
                v,
                lo,
                hi,
                if ok { "PASS" } else { "FAIL" }
            );
            if !ok {
                failures.push(format!("{case:?}/min_free_float={v:.2}C"));
            }
        }

        assert!(
            failures.is_empty(),
            "#1457: {} Case 600-series metrics still out-of-band pending GaugeSolver #1465: {:?}",
            failures.len(),
            failures
        );
    }
}

mod solar_issues {
    use super::*;

    /// SOLAR-01: Peak Cooling Load Under-Prediction (Low-Mass FIXED, High-Mass OPEN)
    ///
    /// Severity: Critical (low-mass FIXED), High (high-mass still OPEN)
    /// Hypothesis: Solar distribution parameters insufficient for high-mass thermal dynamics
    /// GitHub Issue: #274
    ///
    /// Status: LOW-MASS (600 series) PASSES after Phase 7A BASE-05 fix
    ///         HIGH-MASS (900 series) STILL FAILS - peak cooling 2x reference
    #[test]
    #[ignore = "LIMIT-05: Known 5R1C model limitation for high-mass peak cooling (h_tr_ms additive issue)"]
    fn test_solar01_high_mass_peak_cooling_regression() {
        let validator = ASHRAE140Validator::new();
        let report = validator.validate_analytical_engine();

        println!("\n=== SOLAR-01: High-Mass Peak Cooling Regression ===");

        let cases = ["900", "910", "920", "930", "940", "950"];
        let mut failures: Vec<&str> = Vec::new();

        for case_id in cases {
            if let Some(result) = report
                .results
                .iter()
                .find(|r| r.case_id == case_id && matches!(r.metric, MetricType::PeakCooling))
            {
                let passed = matches!(result.status, ValidationStatus::Pass);
                println!(
                    "Case {}: {:.2} kW [{:.2}, {:.2}] - {:?}",
                    case_id, result.fluxion_value, result.ref_min, result.ref_max, result.status
                );

                if !passed {
                    failures.push(case_id);
                }
            }
        }

        println!("Status: REGRESSION TEST - Currently FAILS");
        assert!(
            failures.is_empty(),
            "SOLAR-01: Cases {:?} peak cooling fail",
            failures
        );
    }

    /// SOLAR-02: Annual Cooling Energy Under-Prediction (High-Mass)
    ///
    /// Severity: High
    /// Hypothesis: Solar gain timing and thermal mass coupling incorrect for high-mass
    /// GitHub Issue: #275
    #[test]
    #[ignore = "Known issue - SOLAR-02: High-mass annual cooling under-prediction"]
    fn test_solar02_high_mass_annual_cooling_regression() {
        let validator = ASHRAE140Validator::new();
        let report = validator.validate_analytical_engine();

        println!("\n=== SOLAR-02: High-Mass Annual Cooling Regression ===");

        let cases = ["900", "910", "920", "930", "940", "950"];
        let mut failures: Vec<&str> = Vec::new();

        for case_id in cases {
            if let Some(result) = report
                .results
                .iter()
                .find(|r| r.case_id == case_id && matches!(r.metric, MetricType::AnnualCooling))
            {
                let passed = matches!(result.status, ValidationStatus::Pass);
                println!(
                    "Case {}: {:.2} MWh [{:.2}, {:.2}] - {:?}",
                    case_id, result.fluxion_value, result.ref_min, result.ref_max, result.status
                );

                if !passed {
                    failures.push(case_id);
                }
            }
        }

        println!("Status: REGRESSION TEST - Currently FAILS");
        assert!(
            failures.is_empty(),
            "SOLAR-02: Cases {:?} annual cooling fail",
            failures
        );
    }

    /// SOLAR-03: Solar Shading Cases Not Sensitive to Shading Changes
    ///
    /// Severity: Medium
    /// Hypothesis: Shading coefficient application or solar distribution incorrect
    #[test]
    #[ignore = "Known issue - SOLAR-03: Shading sensitivity insufficient"]
    fn test_solar03_shading_sensitivity_regression() {
        let validator = ASHRAE140Validator::new();
        let report = validator.validate_analytical_engine();

        println!("\n=== SOLAR-03: Shading Sensitivity Regression ===");

        let shade_cases: [(&str, &str, &str); 4] = [
            ("610", "600", "South shading (low-mass)"),
            ("630", "600", "E/W shading (low-mass)"),
            ("910", "900", "South shading (high-mass)"),
            ("930", "900", "E/W shading (high-mass)"),
        ];

        let mut failures: Vec<&str> = Vec::new();

        for (shade_id, base_id, desc) in shade_cases {
            let base_result = report
                .results
                .iter()
                .find(|r| r.case_id == base_id && matches!(r.metric, MetricType::AnnualCooling));
            let shade_result = report
                .results
                .iter()
                .find(|r| r.case_id == shade_id && matches!(r.metric, MetricType::AnnualCooling));

            if let (Some(base), Some(shade)) = (base_result, shade_result) {
                let reduction_pct =
                    (base.fluxion_value - shade.fluxion_value) / base.fluxion_value * 100.0;
                let status = if reduction_pct > 20.0 { "PASS" } else { "FAIL" };
                println!(
                    "{} ({}): {:.1}% reduction - {}",
                    shade_id, desc, reduction_pct, status
                );

                if reduction_pct <= 20.0 {
                    failures.push(shade_id);
                }
            }
        }

        println!("Status: REGRESSION TEST - Currently FAILS");
        assert!(
            failures.is_empty(),
            "SOLAR-03: Shading cases {:?} insufficient sensitivity",
            failures
        );
    }

    /// SOLAR-04: Night Ventilation Cooling Ineffective
    ///
    /// Severity: Medium
    /// Hypothesis: Ventilation air exchange not implemented correctly
    /// GitHub Issue: #276
    #[test]
    #[ignore = "Known issue - SOLAR-04: Night ventilation not effective"]
    fn test_solar04_night_ventilation_regression() {
        let validator = ASHRAE140Validator::new();
        let report = validator.validate_analytical_engine();

        println!("\n=== SOLAR-04: Night Ventilation Effectiveness Regression ===");

        let vent_cases: [(&str, &str, &str); 2] = [
            ("650", "600", "Low-mass night vent"),
            ("950", "900", "High-mass night vent"),
        ];

        let mut failures: Vec<&str> = Vec::new();

        for (vent_id, base_id, desc) in vent_cases {
            let base_result = report
                .results
                .iter()
                .find(|r| r.case_id == base_id && matches!(r.metric, MetricType::AnnualCooling));
            let vent_result = report
                .results
                .iter()
                .find(|r| r.case_id == vent_id && matches!(r.metric, MetricType::AnnualCooling));

            if let (Some(base), Some(vent)) = (base_result, vent_result) {
                let reduction_pct =
                    (base.fluxion_value - vent.fluxion_value) / base.fluxion_value * 100.0;
                let status = if reduction_pct > 15.0 { "PASS" } else { "FAIL" };
                println!(
                    "{} ({}): {:.1}% reduction - {}",
                    vent_id, desc, reduction_pct, status
                );

                if reduction_pct <= 15.0 {
                    failures.push(vent_id);
                }
            }
        }

        println!("Status: REGRESSION TEST - Currently FAILS");
        assert!(
            failures.is_empty(),
            "SOLAR-04: Night ventilation cases {:?} show minimal effect",
            failures
        );
    }
}

mod free_floating_issues {
    use super::*;

    /// FREE-01: Maximum Free-Floating Temperature Under-Prediction (Low-Mass)
    ///
    /// Severity: High
    /// Hypothesis: Solar gain distribution or heat loss coefficients incorrect
    #[test]
    #[ignore = "Known issue - FREE-01: Low-mass free-float max temp under-predicted"]
    fn test_free01_low_mass_max_temp_regression() {
        let validator = ASHRAE140Validator::new();
        let report = validator.validate_analytical_engine();

        println!("\n=== FREE-01: Low-Mass Max Free-Float Temp Regression ===");

        let cases = [("600FF", 64.90, 75.10), ("650FF", 63.20, 73.50)];
        let mut failures: Vec<&str> = Vec::new();

        for (case_id, ref_min, ref_max) in cases {
            if let Some(result) = report
                .results
                .iter()
                .find(|r| r.case_id == case_id && matches!(r.metric, MetricType::MaxFreeFloat))
            {
                let passed = result.fluxion_value >= ref_min && result.fluxion_value <= ref_max;
                println!(
                    "{}: {:.1}°C [ref: {:.1}-{:.1}] - {:?}",
                    case_id, result.fluxion_value, ref_min, ref_max, result.status
                );

                if !passed {
                    failures.push(case_id);
                }
            }
        }

        println!("Status: REGRESSION TEST - Currently FAILS");
        assert!(
            failures.is_empty(),
            "FREE-01: Low-mass cases {:?} max temp outside reference",
            failures
        );
    }

    /// FREE-02: Minimum Free-Floating Temperature Over-Prediction (High-Mass)
    ///
    /// Severity: Medium
    /// Hypothesis: Inadequate heat loss or insufficient thermal mass responsiveness
    ///
    /// NOTE: This test documents the 5R1C topological limitation. 950FF min temp
    /// (-20.84°C) is 0.64°C below the reference lower bound (-20.2°C). This is
    /// acceptable as it is a known structural limit of the simplified method.
    /// See ADR 0003: ISO 13790 5R1C High-Mass Free-Float Temperature Limitations.
    #[test]
    #[ignore = "Known limitation - ADR 0003: 5R1C cannot match EnergyPlus high-mass extremes"]
    fn test_free02_high_mass_min_temp_regression() {
        let validator = ASHRAE140Validator::new();
        let report = validator.validate_analytical_engine();

        println!("\n=== FREE-02: High-Mass Min Free-Float Temp Regression ===");

        let cases = [("900FF", -6.40, -1.60), ("950FF", -20.20, -17.80)];
        let mut failures: Vec<&str> = Vec::new();

        for (case_id, ref_min, ref_max) in cases {
            if let Some(result) = report
                .results
                .iter()
                .find(|r| r.case_id == case_id && matches!(r.metric, MetricType::MinFreeFloat))
            {
                let passed = result.fluxion_value >= ref_min && result.fluxion_value <= ref_max;
                println!(
                    "{}: {:.1}°C [ref: {:.1}-{:.1}] - {:?}",
                    case_id, result.fluxion_value, ref_min, ref_max, result.status
                );

                if !passed {
                    failures.push(case_id);
                }
            }
        }

        println!("Status: REGRESSION TEST - 950FF currently FAILS");
        assert!(
            failures.is_empty(),
            "FREE-02: High-mass cases {:?} min temp outside reference",
            failures
        );
    }

    /// FREE-03: Free-Floating Temperature Swings Reduced
    ///
    /// Severity: Medium
    /// Hypothesis: Thermal mass time constant too long or heat transfer coefficients too high
    #[test]
    #[ignore = "Known issue - FREE-03: Temperature swings reduced vs reference"]
    fn test_free03_temperature_swing_regression() {
        let validator = ASHRAE140Validator::new();
        let report = validator.validate_analytical_engine();

        println!("\n=== FREE-03: Temperature Swing Amplitude Regression ===");

        let cases = [
            ("600FF", 64.90, 75.10, -18.80, -15.60),
            ("650FF", 63.20, 73.50, -23.00, -21.00),
            ("900FF", 41.80, 46.40, -6.40, -1.60),
            ("950FF", 35.50, 38.50, -20.20, -17.80),
        ];

        let mut failures: Vec<&str> = Vec::new();

        for (case_id, max_lo, max_hi, min_lo, min_hi) in cases {
            let max_result = report
                .results
                .iter()
                .find(|r| r.case_id == case_id && matches!(r.metric, MetricType::MaxFreeFloat));
            let min_result = report
                .results
                .iter()
                .find(|r| r.case_id == case_id && matches!(r.metric, MetricType::MinFreeFloat));

            if let (Some(max_r), Some(min_r)) = (max_result, min_result) {
                let max_pass = max_r.fluxion_value >= max_lo && max_r.fluxion_value <= max_hi;
                let min_pass = min_r.fluxion_value >= min_lo && min_r.fluxion_value <= min_hi;

                println!(
                    "{}: max={:.1}°C min={:.1}°C swing={:.1}°C (max PASS={}, min PASS={})",
                    case_id,
                    max_r.fluxion_value,
                    min_r.fluxion_value,
                    max_r.fluxion_value - min_r.fluxion_value,
                    max_pass,
                    min_pass
                );

                if !max_pass || !min_pass {
                    failures.push(case_id);
                }
            }
        }

        println!("Status: REGRESSION TEST - Currently FAILS");
        assert!(
            failures.is_empty(),
            "FREE-03: Cases {:?} temperature swings outside reference",
            failures
        );
    }
}

mod model_limit_issues {
    use super::*;

    /// LIMIT-05: High-Mass Peak Cooling Overprediction (5R1C Model Limitation)
    ///
    /// Severity: Medium
    /// Hypothesis: 5R1C lumped thermal capacitance cannot handle high mass + large dt
    /// Root cause: h_tr_ms additive model overcounts thermal coupling
    #[test]
    #[ignore = "LIMIT-05: Known 5R1C model limitation - h_tr_ms additive overcounts thermal coupling"]
    fn test_limit05_high_mass_peak_cooling_model_limitation() {
        let validator = ASHRAE140Validator::new();
        let report = validator.validate_analytical_engine();

        println!("\n=== LIMIT-05: High-Mass Peak Cooling Model Limitation ===");
        println!(
            "Root cause: h_tr_ms = h_tr_ms_wall + h_tr_ms_roof + h_tr_ms_floor (not additive)"
        );
        println!();

        let cases = ["900", "910", "920", "930", "940", "950"];
        let mut failures: Vec<&str> = Vec::new();

        for case_id in cases {
            if let Some(result) = report
                .results
                .iter()
                .find(|r| r.case_id == case_id && matches!(r.metric, MetricType::PeakCooling))
            {
                let passed = matches!(result.status, ValidationStatus::Pass);
                println!(
                    "{}: {:.2} kW [{:.2}, {:.2}] - {:?}",
                    case_id, result.fluxion_value, result.ref_min, result.ref_max, result.status
                );

                if !passed {
                    failures.push(case_id);
                }
            }
        }

        println!("\nHigh-mass peak cooling failures: {}", failures.len());
        println!("Solution: Requires Phase 6+ multi-node thermal model redesign");
        assert!(
            failures.is_empty(),
            "LIMIT-05: High-mass peak cooling deviations: {:?}",
            failures
        );
    }

    /// LIMIT-06: 600-Series Annual Heating Correction (Empirical - FIXED)
    ///
    /// Severity: Medium (FIXED)
    /// GitHub Issue: #522
    #[test]
    fn test_limit06_600_series_heating_no_regression() {
        let validator = ASHRAE140Validator::new();
        let report = validator.validate_analytical_engine();

        println!("\n=== LIMIT-06: 600-Series Annual Heating Regression Guard ===");

        let cases = ["600", "610", "620", "630", "640"];
        let mut failures: Vec<&str> = Vec::new();

        for case_id in cases {
            if let Some(result) = report
                .results
                .iter()
                .find(|r| r.case_id == case_id && matches!(r.metric, MetricType::AnnualHeating))
            {
                let passed = matches!(
                    result.status,
                    ValidationStatus::Pass | ValidationStatus::Warning
                );
                println!(
                    "{}: {:.2} MWh [{:.2}, {:.2}] - {:?}",
                    case_id, result.fluxion_value, result.ref_min, result.ref_max, result.status
                );

                if !passed {
                    failures.push(case_id);
                }
            }
        }

        assert!(
            failures.is_empty(),
            "LIMIT-06 REGRESSION: 600-series heating cases {:?} outside reference",
            failures
        );
    }

    /// LIMIT-06 UPDATE: 600-Series Annual Cooling (FIXED in Phase 36-04)
    ///
    /// Severity: Medium (FIXED)
    /// GitHub Issue: #531
    #[test]
    fn test_limit06_600_series_cooling_no_regression() {
        let validator = ASHRAE140Validator::new();
        let report = validator.validate_analytical_engine();

        println!("\n=== LIMIT-06 UPDATE: 600-Series Annual Cooling Regression Guard ===");

        let cases = ["600", "610", "620", "630", "640", "650"];
        let mut failures: Vec<&str> = Vec::new();

        for case_id in cases {
            if let Some(result) = report
                .results
                .iter()
                .find(|r| r.case_id == case_id && matches!(r.metric, MetricType::AnnualCooling))
            {
                let passed = matches!(
                    result.status,
                    ValidationStatus::Pass | ValidationStatus::Warning
                );
                println!(
                    "{}: {:.2} MWh [{:.2}, {:.2}] - {:?}",
                    case_id, result.fluxion_value, result.ref_min, result.ref_max, result.status
                );

                if !passed {
                    failures.push(case_id);
                }
            }
        }

        assert!(
            failures.is_empty(),
            "LIMIT-06 UPDATE REGRESSION: 600-series cooling cases {:?} outside reference",
            failures
        );
    }
}

mod additional_regressions {
    use super::*;

    /// Issue #532: Case 195 producing zero annual energy
    ///
    /// Severity: High
    /// Hypothesis: Case 195 configuration error or validation bug
    #[test]
    #[ignore = "Issue #532: Case 195 zero annual energy investigation"]
    fn test_issue532_case195_energy_regression() {
        let validator = ASHRAE140Validator::new();
        let report = validator.validate_analytical_engine();

        println!("\n=== Issue #532: Case 195 Annual Energy Regression ===");

        let heating = report
            .results
            .iter()
            .find(|r| r.case_id == "195" && matches!(r.metric, MetricType::AnnualHeating));
        let cooling = report
            .results
            .iter()
            .find(|r| r.case_id == "195" && matches!(r.metric, MetricType::AnnualCooling));

        let h_val = heating.map(|r| r.fluxion_value).unwrap_or(0.0);
        let c_val = cooling.map(|r| r.fluxion_value).unwrap_or(0.0);

        println!("Case 195 Annual Heating: {:.2} MWh", h_val);
        println!("Case 195 Annual Cooling: {:.2} MWh", c_val);

        let is_zero = h_val < 0.01 && c_val < 0.01;

        assert!(
            !is_zero,
            "Issue #532 REGRESSION: Case 195 produces near-zero annual energy"
        );
        println!("Status: PASS - Case 195 produces valid energy values");
    }

    /// Issue #533: Case 600-series peak load underprediction
    ///
    /// Severity: High
    /// Hypothesis: Solar distribution or sensitivity calculation incorrect for 600 series
    #[test]
    #[ignore = "Issue #533: Case 600-series peak load underprediction investigation"]
    fn test_issue533_600_series_peak_load_regression() {
        let validator = ASHRAE140Validator::new();
        let report = validator.validate_analytical_engine();

        println!("\n=== Issue #533: 600-Series Peak Load Regression ===");

        let cases = ["600", "610", "620", "630", "640", "650"];
        let mut failures: Vec<(&str, &str)> = Vec::new();

        for case_id in cases {
            let peak_h = report
                .results
                .iter()
                .find(|r| r.case_id == case_id && matches!(r.metric, MetricType::PeakHeating));
            let peak_c = report
                .results
                .iter()
                .find(|r| r.case_id == case_id && matches!(r.metric, MetricType::PeakCooling));

            let hp = peak_h
                .map(|r| {
                    let passed = matches!(r.status, ValidationStatus::Pass);
                    if !passed {
                        failures.push((case_id, "PeakHeating"));
                    }
                    format!("{:.2}kW - {:?}", r.fluxion_value, r.status)
                })
                .unwrap_or("N/A".to_string());

            let cp = peak_c
                .map(|r| {
                    let passed = matches!(r.status, ValidationStatus::Pass);
                    if !passed {
                        failures.push((case_id, "PeakCooling"));
                    }
                    format!("{:.2}kW - {:?}", r.fluxion_value, r.status)
                })
                .unwrap_or("N/A".to_string());

            println!("{}: Heat={} Cool={}", case_id, hp, cp);
        }

        assert!(
            failures.is_empty(),
            "Issue #533 REGRESSION: 600-series peak load failures: {:?}",
            failures
        );
    }
}
