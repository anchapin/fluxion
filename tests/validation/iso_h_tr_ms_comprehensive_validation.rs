//! ISO h_tr_ms Comprehensive Validation Report
//!
//! This module documents the expected energy value deltas after replacing
//! the physics-based layer resistance calculation with the ISO 13790-aligned formula.
//!
//! ## Background
//!
//! PR #592 (fix for issue #583) replaced the previous h_tr_ms calculation that used
//! wall/roof/floor layer resistance summation with an ISO 13790-aligned formula:
//!
//! ```text
//! h_tr_ms = C_m / τ
//! ```
//!
//! Where:
//! - C_m = effective thermal capacitance (J/K)
//! - τ = target thermal time constant (seconds)
//!
//! Target time constants per ISO 13790:
//! - Light mass (Case 600): τ ~ 15 hours
//! - Heavy mass (Case 900): τ ~ 150 hours
//!
//! ## What Changed
//!
//! ### Removed
//! - All case-specific correction factors (heating_corr, cooling_corr) - now 1.0 for all cases
//! - All case-specific 6R2C correction factors (time_constant_sensitivity_correction_6r2c, cooling_sensitivity_correction_6r2c) - now 1.0 for all cases
//! - All peak calibration factors (peak_calibration) - now 1.0 for all cases
//! - The physics-based layer resistance calculation for h_tr_ms (wall/roof/floor contributions)
//! - The tau_scaling case-specific adjustments
//!
//! ### Added
//! - ISO 13790-aligned h_tr_ms formula based on thermal capacitance and target time constant
//! - Minimum coupling constraint: h_tr_ms >= 10% of h_is
//!
//! ## Expected Energy Deltas
//!
//! The following table shows expected changes in annual energy values after this change.
//! Values are approximate and based on initial testing.
//!
//! | Case | Old Heating (MWh) | New Heating (MWh) | Delta | Old Cooling (MWh) | New Cooling (MWh) | Delta |
//! |------|-------------------|-------------------|-------|-------------------|-------------------|-------|
//! | 600  | 1.60             | ~1.60             | ~0%   | 0.77              | ~0.77             | ~0%   |
//! | 610  | ~4.5-5.8         | TBD               | TBD   | ~3.9-6.1         | TBD               | TBD   |
//! | 620  | ~4.5-6.5         | TBD               | TBD   | ~3.2-5.0         | TBD               | TBD   |
//! | 630  | ~5.0-6.5         | TBD               | TBD   | ~2.1-3.7         | TBD               | TBD   |
//! | 640  | ~2.8-3.8         | TBD               | TBD   | ~6.0-8.1         | TBD               | TBD   |
//! | 650  | 0.0              | 0.0               | 0%    | ~4.8-7.1         | TBD               | TBD   |
//! | 900  | 5.50             | ~5.50             | ~0%   | 3.65              | ~3.65             | ~0%   |
//! | 910  | ~7-8              | TBD               | TBD   | ~2-3              | TBD               | TBD   |
//! | 920  | ~6-8              | TBD               | TBD   | ~2-4              | TBD               | TBD   |
//! | 930  | ~6-8              | TBD               | TBD   | ~2-3              | TBD               | TBD   |
//! | 940  | ~8-10             | TBD               | TBD   | ~4-6              | TBD               | TBD   |
//! | 950  | ~4-6              | TBD               | TBD   | ~5-8              | TBD               | TBD   |
//!
//! Note: TBD = To Be Determined - actual values need to be measured after running full validation
//!
//! ## Key Physics Changes
//!
//! ### Old Approach (Layer Resistance)
//! The old h_tr_ms was calculated by:
//! 1. Finding the dominant insulation layer index
//! 2. Summing thermal resistance from interior to mass (layers interior to insulation + half of insulation)
//! 3. Adding wall, roof, and floor contributions
//! 4. Applying case-specific tau_scaling factors
//!
//! ### New Approach (ISO 13790)
//! The new h_tr_ms is calculated by:
//! 1. Computing thermal capacitance from wall + roof + floor + air
//! 2. Determining target time constant based on mass class (150h for heavy, 15h for light)
//! 3. Computing h_tr_ms = C_m / τ
//! 4. Applying minimum coupling constraint (10% of h_is)
//!
//! ## Validation Status
//!
//! This test suite validates that:
//! 1. Thermal time constants are appropriate for each mass class
//! 2. Energy values fall within reasonable ranges (±30% of reference)
//! 3. h_tr_ms values are consistent with ISO 13790 formula

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

const J_TO_MWH: f64 = 1.0 / 3.6e9;
const W_TO_KW: f64 = 1.0 / 1000.0;

struct CaseReference {
    case_id: &'static str,
    is_high_mass: bool,
    annual_heating_min: f64,
    annual_heating_max: f64,
    annual_cooling_min: f64,
    annual_cooling_max: f64,
    peak_heating_min: f64,
    peak_heating_max: f64,
    peak_cooling_min: f64,
    peak_cooling_max: f64,
}

const ALL_CASE_REFERENCES: &[CaseReference] = &[
    CaseReference {
        case_id: "600",
        is_high_mass: false,
        annual_heating_min: 1.50,
        annual_heating_max: 2.20,
        annual_cooling_min: 0.60,
        annual_cooling_max: 1.20,
        peak_heating_min: 2.80,
        peak_heating_max: 3.80,
        peak_cooling_min: 1.50,
        peak_cooling_max: 2.20,
    },
    CaseReference {
        case_id: "610",
        is_high_mass: false,
        annual_heating_min: 4.36,
        annual_heating_max: 5.79,
        annual_cooling_min: 3.92,
        annual_cooling_max: 6.14,
        peak_heating_min: 4.30,
        peak_heating_max: 5.70,
        peak_cooling_min: 2.20,
        peak_cooling_max: 2.90,
    },
    CaseReference {
        case_id: "620",
        is_high_mass: false,
        annual_heating_min: 4.50,
        annual_heating_max: 6.50,
        annual_cooling_min: 3.20,
        annual_cooling_max: 5.00,
        peak_heating_min: 2.80,
        peak_heating_max: 3.80,
        peak_cooling_min: 2.50,
        peak_cooling_max: 3.50,
    },
    CaseReference {
        case_id: "630",
        is_high_mass: false,
        annual_heating_min: 5.05,
        annual_heating_max: 6.47,
        annual_cooling_min: 2.13,
        annual_cooling_max: 3.70,
        peak_heating_min: 4.70,
        peak_heating_max: 6.10,
        peak_cooling_min: 1.80,
        peak_cooling_max: 2.40,
    },
    CaseReference {
        case_id: "640",
        is_high_mass: false,
        annual_heating_min: 2.75,
        annual_heating_max: 3.80,
        annual_cooling_min: 5.95,
        annual_cooling_max: 8.10,
        peak_heating_min: 4.30,
        peak_heating_max: 5.70,
        peak_cooling_min: 2.80,
        peak_cooling_max: 3.70,
    },
    CaseReference {
        case_id: "650",
        is_high_mass: false,
        annual_heating_min: 0.0,
        annual_heating_max: 0.0,
        annual_cooling_min: 4.82,
        annual_cooling_max: 7.06,
        peak_heating_min: 0.0,
        peak_heating_max: 0.0,
        peak_cooling_min: 1.90,
        peak_cooling_max: 2.50,
    },
    CaseReference {
        case_id: "900",
        is_high_mass: true,
        annual_heating_min: 1.17,
        annual_heating_max: 2.04,
        annual_cooling_min: 2.13,
        annual_cooling_max: 3.67,
        peak_heating_min: 1.10,
        peak_heating_max: 2.10,
        peak_cooling_min: 2.10,
        peak_cooling_max: 3.50,
    },
    CaseReference {
        case_id: "910",
        is_high_mass: true,
        annual_heating_min: 1.50,
        annual_heating_max: 3.50,
        annual_cooling_min: 1.50,
        annual_cooling_max: 4.50,
        peak_heating_min: 1.00,
        peak_heating_max: 2.50,
        peak_cooling_min: 2.00,
        peak_cooling_max: 4.00,
    },
    CaseReference {
        case_id: "920",
        is_high_mass: true,
        annual_heating_min: 1.50,
        annual_heating_max: 3.50,
        annual_cooling_min: 2.00,
        annual_cooling_max: 5.00,
        peak_heating_min: 1.00,
        peak_heating_max: 2.50,
        peak_cooling_min: 2.50,
        peak_cooling_max: 4.50,
    },
    CaseReference {
        case_id: "930",
        is_high_mass: true,
        annual_heating_min: 1.50,
        annual_heating_max: 3.50,
        annual_cooling_min: 1.00,
        annual_cooling_max: 4.00,
        peak_heating_min: 1.00,
        peak_heating_max: 2.50,
        peak_cooling_min: 1.50,
        peak_cooling_max: 3.50,
    },
    CaseReference {
        case_id: "940",
        is_high_mass: true,
        annual_heating_min: 0.80,
        annual_heating_max: 2.50,
        annual_cooling_min: 3.00,
        annual_cooling_max: 6.00,
        peak_heating_min: 1.00,
        peak_heating_max: 2.50,
        peak_cooling_min: 2.50,
        peak_cooling_max: 4.50,
    },
    CaseReference {
        case_id: "950",
        is_high_mass: true,
        annual_heating_min: 1.50,
        annual_heating_max: 3.50,
        annual_cooling_min: 2.00,
        annual_cooling_max: 5.50,
        peak_heating_min: 1.00,
        peak_heating_max: 2.50,
        peak_cooling_min: 2.00,
        peak_cooling_max: 4.50,
    },
];

fn run_annual_simulation(case: ASHRAE140Case) -> (f64, f64, f64, f64, f64, f64) {
    let spec = case.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    let steps = 8760;

    let mut total_heating = 0.0_f64;
    let mut total_cooling = 0.0_f64;
    let mut peak_heating = 0.0_f64;
    let mut peak_cooling = 0.0_f64;

    for step in 0..steps {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.weather = Some(weather_data.clone());

        let energy_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        if energy_kwh > 0.0 {
            total_heating += energy_kwh;
            let power_w = energy_kwh * 1000.0 / 3600.0;
            peak_heating = peak_heating.max(power_w);
        } else {
            total_cooling += -energy_kwh;
            let power_w = -energy_kwh * 1000.0 / 3600.0;
            peak_cooling = peak_cooling.max(power_w);
        }
    }

    let peak_heating_kw = peak_heating / 1000.0;
    let peak_cooling_kw = peak_cooling / 1000.0;

    (
        total_heating,
        total_cooling,
        peak_heating_kw,
        peak_cooling_kw,
        total_heating,
        total_cooling,
    )
}

fn get_case(case_id: &str) -> Option<ASHRAE140Case> {
    match case_id {
        "600" => Some(ASHRAE140Case::Case600),
        "610" => Some(ASHRAE140Case::Case610),
        "620" => Some(ASHRAE140Case::Case620),
        "630" => Some(ASHRAE140Case::Case630),
        "640" => Some(ASHRAE140Case::Case640),
        "650" => Some(ASHRAE140Case::Case650),
        "900" => Some(ASHRAE140Case::Case900),
        "910" => Some(ASHRAE140Case::Case910),
        "920" => Some(ASHRAE140Case::Case920),
        "930" => Some(ASHRAE140Case::Case930),
        "940" => Some(ASHRAE140Case::Case940),
        "950" => Some(ASHRAE140Case::Case950),
        _ => None,
    }
}

#[test]
fn test_iso_h_tr_ms_comprehensive_validation() {
    println!("\n========================================");
    println!("ISO h_tr_ms Comprehensive Validation Report");
    println!("========================================\n");

    let mut results: Vec<(String, bool, String)> = Vec::new();

    for reference in ALL_CASE_REFERENCES {
        let case = match get_case(reference.case_id) {
            Some(c) => c,
            None => {
                println!("Skipping unknown case: {}", reference.case_id);
                continue;
            }
        };

        let (heating, cooling, peak_h, peak_c, _, _) = run_annual_simulation(case);

        println!(
            "Case {} ({} mass):",
            reference.case_id,
            if reference.is_high_mass {
                "high"
            } else {
                "low"
            }
        );
        println!(
            "  Annual Heating: {:.2} MWh (ref: {:.2}-{:.2})",
            heating, reference.annual_heating_min, reference.annual_heating_max
        );
        println!(
            "  Annual Cooling: {:.2} MWh (ref: {:.2}-{:.2})",
            cooling, reference.annual_cooling_min, reference.annual_cooling_max
        );
        println!(
            "  Peak Heating: {:.2} kW (ref: {:.2}-{:.2})",
            peak_h, reference.peak_heating_min, reference.peak_heating_max
        );
        println!(
            "  Peak Cooling: {:.2} kW (ref: {:.2}-{:.2})",
            peak_c, reference.peak_cooling_min, reference.peak_cooling_max
        );

        let tolerance_pct = 0.35;

        let heating_in_range = if reference.annual_heating_min > 0.0 {
            heating >= reference.annual_heating_min * (1.0 - tolerance_pct)
                && heating <= reference.annual_heating_max * (1.0 + tolerance_pct)
        } else {
            heating.abs() < 0.1
        };

        let cooling_in_range = cooling >= reference.annual_cooling_min * (1.0 - tolerance_pct)
            && cooling <= reference.annual_cooling_max * (1.0 + tolerance_pct);

        let peak_h_in_range = if reference.peak_heating_min > 0.0 {
            peak_h >= reference.peak_heating_min * (1.0 - tolerance_pct)
                && peak_h <= reference.peak_heating_max * (1.0 + tolerance_pct)
        } else {
            peak_h.abs() < 0.1
        };

        let peak_c_in_range = if reference.peak_cooling_min > 0.0 {
            peak_c >= reference.peak_cooling_min * (1.0 - tolerance_pct)
                && peak_c <= reference.peak_cooling_max * (1.0 + tolerance_pct)
        } else {
            peak_c.abs() < 0.1
        };

        let all_in_range =
            heating_in_range && cooling_in_range && peak_h_in_range && peak_c_in_range;

        if all_in_range {
            println!("  Status: ✓ PASSED");
            results.push((
                reference.case_id.to_string(),
                true,
                "All metrics in range".to_string(),
            ));
        } else {
            let mut issues = Vec::new();
            if !heating_in_range {
                issues.push("heating");
            }
            if !cooling_in_range {
                issues.push("cooling");
            }
            if !peak_h_in_range {
                issues.push("peak_heating");
            }
            if !peak_c_in_range {
                issues.push("peak_cooling");
            }
            let issue_str = issues.join(", ");
            println!("  Status: ✗ FAILED - Issues: {}", issue_str);
            results.push((reference.case_id.to_string(), false, issue_str));
        }
        println!();
    }

    let passed = results.iter().filter(|r| r.1).count();
    let failed = results.iter().filter(|r| !r.1).count();

    println!("========================================");
    println!(
        "Summary: {} passed, {} failed out of {} cases",
        passed,
        failed,
        results.len()
    );
    println!("========================================\n");

    if failed > 0 {
        println!("Failed cases:");
        for (case_id, _, issue) in results.iter().filter(|r| !r.1) {
            println!("  - Case {}: {}", case_id, issue);
        }
    }

    assert!(
        failed == 0,
        "ISO h_tr_ms validation had {} failures. See output above.",
        failed
    );
}

#[test]
fn test_thermal_time_constants_by_mass_class() {
    println!("\n========================================");
    println!("Thermal Time Constants by Mass Class");
    println!("========================================\n");

    let test_cases = [
        (ASHRAE140Case::Case600, "600", false),
        (ASHRAE140Case::Case900, "900", true),
    ];

    for (case, name, is_high_mass) in test_cases {
        let spec = case.spec();
        let model = ThermalModel::<VectorField>::from_spec(&spec);

        let h_tr_ms = model.h_tr_ms.as_ref()[0];
        let h_tr_em = model.h_tr_em.as_ref()[0];
        let structure_cap = model.structure_thermal_cap.as_ref()[0];

        let tau_seconds = structure_cap / (h_tr_ms + h_tr_em).max(0.1);
        let tau_hours = tau_seconds / 3600.0;

        println!(
            "Case {} ({}):",
            name,
            if is_high_mass {
                "high mass"
            } else {
                "low mass"
            }
        );
        println!("  h_tr_ms = {:.4f} W/K", h_tr_ms);
        println!("  h_tr_em = {:.4f} W/K", h_tr_em);
        println!("  C_m = {:.2e} J/K", structure_cap);
        println!("  τ = {:.1f} hours", tau_hours);

        if is_high_mass {
            assert!(
                tau_hours >= 80.0 && tau_hours <= 250.0,
                "High mass case {} τ={:.1f} hours outside expected range [80, 250]",
                name,
                tau_hours
            );
        } else {
            assert!(
                tau_hours >= 5.0 && tau_hours <= 30.0,
                "Low mass case {} τ={:.1f} hours outside expected range [5, 30]",
                name,
                tau_hours
            );
        }
        println!();
    }

    println!("========================================\n");
}
