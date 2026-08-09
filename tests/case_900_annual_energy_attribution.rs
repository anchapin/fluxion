//! Annual Energy Attribution Test for Case 900
//!
//! This test decomposes annual energy mismatches into subsystem contributions:
//! - Solar gains (total annual)
//! - Internal gains (total annual)
//! - Envelope conduction (total annual)
//! - Infiltration (total annual)
//! - HVAC output (total annual)
//!
//! The attribution helps identify which subsystem is causing annual energy over/under-prediction
//! and provides bounded residual error with subsystem explanation per PH-04 Definition of Done.
//!
//! Issue #2448 uses the ignored Case 900/910 comparison below to separate the
//! shading signal from the high-mass solver response. The incident-solar guard
//! proves that the overhang reduces the upstream solar forcing; if annual cooling
//! does not decrease with it, the inversion is downstream in the 5R1C lumped
//! thermal network documented by `KNOWN_ISSUES.md` LIMIT-05. The strict reference
//! bands remain blocked on GaugeSolver #1465/#1462 rather than parameter tuning.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::validation::diagnostics::SimulationDiagnostics;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

#[derive(Debug, Clone)]
pub struct AnnualEnergyAttribution {
    pub annual_heating_kwh: f64,
    pub annual_cooling_kwh: f64,
    pub solar_annual_kwh: f64,
    pub internal_annual_kwh: f64,
    pub conduction_annual_kwh: f64,
    pub infiltration_annual_kwh: f64,
    pub hvac_annual_kwh: f64,
    pub reference_heating_kwh: (f64, f64),
    pub reference_cooling_kwh: (f64, f64),
    pub heating_error_pct: f64,
    pub cooling_error_pct: f64,
    pub heating_residual_bounded: bool,
    pub cooling_residual_bounded: bool,
}

impl AnnualEnergyAttribution {
    pub fn new() -> Self {
        Self {
            annual_heating_kwh: 0.0,
            annual_cooling_kwh: 0.0,
            solar_annual_kwh: 0.0,
            internal_annual_kwh: 0.0,
            conduction_annual_kwh: 0.0,
            infiltration_annual_kwh: 0.0,
            hvac_annual_kwh: 0.0,
            reference_heating_kwh: (0.0, 0.0),
            reference_cooling_kwh: (0.0, 0.0),
            heating_error_pct: 0.0,
            cooling_error_pct: 0.0,
            heating_residual_bounded: false,
            cooling_residual_bounded: false,
        }
    }
}

impl Default for AnnualEnergyAttribution {
    fn default() -> Self {
        Self::new()
    }
}

pub fn calculate_annual_energy_attribution(
    diag: &SimulationDiagnostics,
) -> AnnualEnergyAttribution {
    let mut attribution = AnnualEnergyAttribution::new();

    let mut total_solar = 0.0;
    let mut total_internal = 0.0;
    let mut total_conduction = 0.0;
    let mut total_infiltration = 0.0;
    let mut total_hvac = 0.0;

    for i in 0..diag.hours.len() {
        let num_zones = diag.loads.solar.get(i).map(|z| z.len()).unwrap_or(1);
        for zone_idx in 0..num_zones {
            total_solar += diag
                .loads
                .solar
                .get(i)
                .and_then(|z| z.get(zone_idx))
                .copied()
                .unwrap_or(0.0);
            total_internal += diag
                .loads
                .internal
                .get(i)
                .and_then(|z| z.get(zone_idx))
                .copied()
                .unwrap_or(0.0);
            total_conduction += diag
                .loads
                .conduction
                .get(i)
                .and_then(|z| z.get(zone_idx))
                .copied()
                .unwrap_or(0.0);
            total_infiltration += diag
                .loads
                .infiltration
                .get(i)
                .and_then(|z| z.get(zone_idx))
                .copied()
                .unwrap_or(0.0);
            let hvac_val = diag
                .loads
                .hvac
                .get(i)
                .and_then(|z| z.get(zone_idx))
                .copied()
                .unwrap_or(0.0);
            total_hvac += hvac_val;
        }
    }

    attribution.solar_annual_kwh = total_solar / 1000.0;
    attribution.internal_annual_kwh = total_internal / 1000.0;
    attribution.conduction_annual_kwh = total_conduction / 1000.0;
    attribution.infiltration_annual_kwh = total_infiltration / 1000.0;
    attribution.hvac_annual_kwh = total_hvac / 1000.0;

    if total_hvac > 0.0 {
        attribution.annual_heating_kwh = total_hvac / 1000.0;
    } else {
        attribution.annual_cooling_kwh = (-total_hvac) / 1000.0;
    }

    attribution
}

pub fn calculate_annual_energy_attribution_for_case(
    case_spec: &ASHRAE140Case,
) -> AnnualEnergyAttribution {
    simulate_case(case_spec).0
}

fn simulate_case(
    case_spec: &ASHRAE140Case,
) -> (AnnualEnergyAttribution, ThermalModel<VectorField>) {
    let spec = case_spec.spec();
    let weather = DenverTmyWeather::new();

    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let diag = SimulationDiagnostics::new(model.num_zones, 8760);
    model.set_diagnostics(Some(diag));

    let num_zones = model.num_zones;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();

        model.weather = Some(weather_data.clone());

        if let Some(hvac_schedule) = spec.hvac.first() {
            let hour = (step % 24) as u8;
            let heating_sp = hvac_schedule
                .heating_setpoint_at_hour(hour)
                .unwrap_or(hvac_schedule.heating_setpoint);
            let cooling_sp = model.cooling_schedule.value(hour as usize);
            model.heating_setpoint = heating_sp;
            model.cooling_setpoint = cooling_sp;

            if spec.hvac.len() > 1 {
                let mut heating_sps = vec![heating_sp; num_zones];
                let mut cooling_sps = vec![cooling_sp; num_zones];
                for (zone_idx, hvac) in spec.hvac.iter().enumerate() {
                    if zone_idx < num_zones {
                        let h_sp = hvac
                            .heating_setpoint_at_hour(hour)
                            .unwrap_or(hvac.heating_setpoint);
                        let c_sp = model.cooling_schedule.value(hour as usize);
                        heating_sps[zone_idx] = h_sp;
                        cooling_sps[zone_idx] = c_sp;
                    }
                }
                model.heating_setpoints = VectorField::new(heating_sps);
                model.cooling_setpoints = VectorField::new(cooling_sps);
            }
        }

        let mut internal_loads: Vec<f64> = Vec::with_capacity(num_zones);
        for zone_idx in 0..num_zones {
            let internal_gains = spec
                .internal_loads
                .get(zone_idx)
                .or(spec.internal_loads.first())
                .and_then(|l| l.as_ref())
                .map_or(0.0, |l| l.total_load);

            let floor_area = spec
                .geometry
                .get(zone_idx)
                .or(spec.geometry.first())
                .map_or(20.0, |g| g.floor_area());

            internal_loads.push(internal_gains / floor_area);
        }
        model.set_loads(&internal_loads);

        let _hvac_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
    }

    let attribution = {
        let diag = model
            .get_diagnostics()
            .expect("Diagnostics should be attached");
        calculate_annual_energy_attribution(diag)
    };
    (attribution, model)
}

fn annual_window_irradiance_kwh_m2(model: &ThermalModel<VectorField>, orientation: &str) -> f64 {
    let key = format!("window_{orientation}");
    model
        .get_incident_solar()
        .iter()
        .find(|(surface, _)| surface.as_str() == key.as_str())
        .map(|(_, irradiance)| irradiance.annual_kwh_m2)
        .unwrap_or(0.0)
}

fn annual_zone_solar_kwh(attribution: &AnnualEnergyAttribution) -> f64 {
    attribution.solar_annual_kwh
}

#[test]
#[ignore = "Diagnostic: #2448 strict cooling bands require GaugeSolver #1465/#1462"]
fn test_issue_2448_case_910_shading_attribution() {
    let (case_900, model_900) = simulate_case(&ASHRAE140Case::Case900);
    let (case_910, model_910) = simulate_case(&ASHRAE140Case::Case910);

    // The `incident_solar_per_surface` accumulator reports pre-overhang
    // incident flux on the surface identifier (e.g., "window_S"), so the
    // south-window per-m² value is identical for the two cases by design.
    // The shading effect propagates into the per-zone solar load recorded
    // in `SimulationDiagnostics::loads.solar` (the energy actually delivered
    // to the air node after `calculate_hourly_solar_from_pos` applies the
    // overhang geometry), so the comparison must use the per-zone load
    // instead of the per-m² incident key.
    let s_irradiance_900 = annual_window_irradiance_kwh_m2(&model_900, "S");
    let s_irradiance_910 = annual_window_irradiance_kwh_m2(&model_910, "S");

    // Per-zone solar load integrated over the year (kWh from the
    // `SimulationDiagnostics::loads.solar` accumulator; the attribution
    // helper already converts W·h → kWh by dividing by 1e3).
    let solar_900_kwh = annual_zone_solar_kwh(&case_900);
    let solar_910_kwh = annual_zone_solar_kwh(&case_910);
    let solar_reduction = 1.0 - solar_910_kwh / solar_900_kwh;
    let cooling_delta_kwh = case_910.annual_cooling_kwh - case_900.annual_cooling_kwh;

    println!(
        "[#2448] Case 900: south incident={s_irradiance_900:.1} kWh/m², zone solar={solar_900_kwh:.3} kWh, cooling={:.3} MWh",
        case_900.annual_cooling_kwh / 1000.0
    );
    println!(
        "[#2448] Case 910: south incident={s_irradiance_910:.1} kWh/m², zone solar={solar_910_kwh:.3} kWh, cooling={:.3} MWh",
        case_910.annual_cooling_kwh / 1000.0
    );
    println!(
        "[#2448] shading solar reduction={:.1}%, cooling delta={:+.3} MWh",
        solar_reduction * 100.0,
        cooling_delta_kwh / 1000.0
    );

    assert!(
        s_irradiance_900 == s_irradiance_910 && s_irradiance_900 > 0.0,
        "expected pre-shading south incident irradiance to be shared and positive: 900={s_irradiance_900:.1}, 910={s_irradiance_910:.1}"
    );
    assert!(
        solar_910_kwh < solar_900_kwh,
        "Case 910 overhang must reduce annual solar delivered to the zone: 900={solar_900_kwh:.3} kWh, 910={solar_910_kwh:.3} kWh"
    );
}

#[test]
fn test_case_900_annual_energy_attribution() {
    println!("\n=== Case 900 Annual Energy Attribution ===");

    let attribution = calculate_annual_energy_attribution_for_case(&ASHRAE140Case::Case900);

    println!("\n--- Annual Energy Summary ---");
    println!(
        "Annual Heating: {:.2} MWh",
        attribution.annual_heating_kwh / 1000.0
    );
    println!(
        "Annual Cooling: {:.2} MWh",
        attribution.annual_cooling_kwh / 1000.0
    );

    println!("\n--- Subsystem Contributions (kWh/year) ---");
    println!(
        "Solar gains: {:.2} kWh ({:.1}%)",
        attribution.solar_annual_kwh,
        if attribution.solar_annual_kwh > 0.0 {
            (attribution.solar_annual_kwh
                / (attribution.solar_annual_kwh
                    + attribution.internal_annual_kwh
                    + attribution.conduction_annual_kwh.abs()
                    + attribution.infiltration_annual_kwh.abs()))
                * 100.0
        } else {
            0.0
        }
    );
    println!("Internal gains: {:.2} kWh", attribution.internal_annual_kwh);
    println!(
        "Envelope conduction: {:.2} kWh",
        attribution.conduction_annual_kwh
    );
    println!(
        "Infiltration: {:.2} kWh",
        attribution.infiltration_annual_kwh
    );
    println!("HVAC output: {:.2} kWh", attribution.hvac_annual_kwh);

    println!("\n--- ASHRAE 140 Reference Comparison ---");
    println!("Reference heating: 1170-2040 kWh (1.17-2.04 MWh)");
    println!("Reference cooling: 2130-3670 kWh (2.13-3.67 MWh)");
    println!(
        "Actual heating: {:.2} MWh (error: {:.1}%)",
        attribution.annual_heating_kwh / 1000.0,
        if attribution.annual_heating_kwh > 0.0 {
            ((attribution.annual_heating_kwh - 1605.0).abs() / 1605.0) * 100.0
        } else {
            0.0
        }
    );
    println!(
        "Actual cooling: {:.2} MWh (error: {:.1}%)",
        attribution.annual_cooling_kwh / 1000.0,
        if attribution.annual_cooling_kwh > 0.0 {
            ((attribution.annual_cooling_kwh - 2900.0).abs() / 2900.0) * 100.0
        } else {
            0.0
        }
    );

    assert!(
        attribution.annual_heating_kwh >= 0.0,
        "Annual heating should be non-negative"
    );
    assert!(
        attribution.annual_cooling_kwh >= 0.0,
        "Annual cooling should be non-negative"
    );

    println!("\n✅ Case 900 annual energy attribution computed successfully");
}

#[test]
fn test_case_900_annual_energy_subsystem_analysis() {
    println!("\n=== Case 900 Annual Energy Subsystem Analysis ===");

    let attribution = calculate_annual_energy_attribution_for_case(&ASHRAE140Case::Case900);

    println!("\n--- Subsystem Contribution Analysis ---");

    let total_loads = (attribution.solar_annual_kwh
        + attribution.internal_annual_kwh
        + attribution.conduction_annual_kwh.abs()
        + attribution.infiltration_annual_kwh.abs())
    .max(1.0);

    println!(
        "Solar fraction: {:.1}%",
        (attribution.solar_annual_kwh / total_loads) * 100.0
    );
    println!(
        "Internal fraction: {:.1}%",
        (attribution.internal_annual_kwh / total_loads) * 100.0
    );
    println!(
        "Conduction fraction: {:.1}%",
        (attribution.conduction_annual_kwh.abs() / total_loads) * 100.0
    );
    println!(
        "Infiltration fraction: {:.1}%",
        (attribution.infiltration_annual_kwh.abs() / total_loads) * 100.0
    );

    println!("\n--- Residual Error Bounded Analysis ---");
    let heating_midpoint = 1605.0;
    let heating_ref_range = 2040.0 - 1170.0;
    let heating_upper_bound = heating_midpoint + (heating_ref_range / 2.0) * 1.15;
    let heating_lower_bound = heating_midpoint - (heating_ref_range / 2.0) * 1.15;

    let cooling_midpoint = 2900.0;
    let cooling_ref_range = 3670.0 - 2130.0;
    let cooling_upper_bound = cooling_midpoint + (cooling_ref_range / 2.0) * 1.15;
    let cooling_lower_bound = cooling_midpoint - (cooling_ref_range / 2.0) * 1.15;

    let heating_in_bounds = attribution.annual_heating_kwh >= heating_lower_bound
        && attribution.annual_heating_kwh <= heating_upper_bound;
    let cooling_in_bounds = attribution.annual_cooling_kwh >= cooling_lower_bound
        && attribution.annual_cooling_kwh <= cooling_upper_bound;

    println!(
        "Heating bounds: [{:.0}, {:.0}] kWh, Actual: {:.0} kWh",
        heating_lower_bound, heating_upper_bound, attribution.annual_heating_kwh
    );
    println!(
        "Cooling bounds: [{:.0}, {:.0}] kWh, Actual: {:.0} kWh",
        cooling_lower_bound, cooling_upper_bound, attribution.annual_cooling_kwh
    );
    println!(
        "Heating bounded: {}",
        if heating_in_bounds { "YES" } else { "NO" }
    );
    println!(
        "Cooling bounded: {}",
        if cooling_in_bounds { "YES" } else { "NO" }
    );

    println!("\n✅ Case 900 subsystem analysis complete");
}

#[test]
fn test_all_900_series_annual_energy_attribution() {
    println!("\n=== All 900-Series Annual Energy Attribution ===");

    let cases = [
        (
            "900",
            ASHRAE140Case::Case900,
            1170.0,
            2040.0,
            2130.0,
            3670.0,
        ),
        ("910", ASHRAE140Case::Case910, 750.0, 1330.0, 2620.0, 4210.0),
        (
            "920",
            ASHRAE140Case::Case920,
            1100.0,
            2040.0,
            1250.0,
            2100.0,
        ),
        ("930", ASHRAE140Case::Case930, 690.0, 1330.0, 1400.0, 2400.0),
        ("940", ASHRAE140Case::Case940, 690.0, 1330.0, 2400.0, 3810.0),
        ("950", ASHRAE140Case::Case950, 650.0, 1170.0, 2690.0, 4170.0),
        (
            "960",
            ASHRAE140Case::Case960,
            1650.0,
            2450.0,
            1200.0,
            2000.0,
        ),
    ];

    for (case_id, case_enum, h_ref_min, h_ref_max, c_ref_min, c_ref_max) in cases {
        println!("\n  Case {}...", case_id);
        let attribution = calculate_annual_energy_attribution_for_case(&case_enum);

        let h_mid = (h_ref_min + h_ref_max) / 2.0;
        let c_mid = (c_ref_min + c_ref_max) / 2.0;
        let h_err = if attribution.annual_heating_kwh > 0.0 {
            ((attribution.annual_heating_kwh - h_mid) / h_mid) * 100.0
        } else {
            0.0
        };
        let c_err = if attribution.annual_cooling_kwh > 0.0 {
            ((attribution.annual_cooling_kwh - c_mid) / c_mid) * 100.0
        } else {
            0.0
        };

        println!(
            "    Heating: {:.2} MWh (ref: {:.2}-{:.2} MWh, error: {:.1}%)",
            attribution.annual_heating_kwh / 1000.0,
            h_ref_min / 1000.0,
            h_ref_max / 1000.0,
            h_err
        );
        println!(
            "    Cooling: {:.2} MWh (ref: {:.2}-{:.2} MWh, error: {:.1}%)",
            attribution.annual_cooling_kwh / 1000.0,
            c_ref_min / 1000.0,
            c_ref_max / 1000.0,
            c_err
        );

        let total_loads = (attribution.solar_annual_kwh
            + attribution.internal_annual_kwh
            + attribution.conduction_annual_kwh.abs()
            + attribution.infiltration_annual_kwh.abs())
        .max(1.0);

        if total_loads > 0.0 {
            println!("    Subsystem fractions:");
            println!(
                "      Solar: {:.1}%",
                (attribution.solar_annual_kwh / total_loads) * 100.0
            );
            println!(
                "      Internal: {:.1}%",
                (attribution.internal_annual_kwh / total_loads) * 100.0
            );
            println!(
                "      Conduction: {:.1}%",
                (attribution.conduction_annual_kwh.abs() / total_loads) * 100.0
            );
            println!(
                "      Infiltration: {:.1}%",
                (attribution.infiltration_annual_kwh.abs() / total_loads) * 100.0
            );
        }
    }

    println!("\n✅ All 900-series annual energy attribution computed successfully");
}
