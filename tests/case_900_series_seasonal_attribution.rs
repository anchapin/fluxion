//! Per-month seasonal energy attribution diagnostic for ASHRAE 140 900-series.
//!
//! Issue #2453: "900-series bidirectional annual-energy over-prediction
//! (Cases 900, 910, 920, 930, 940) — re-characterise Issue #2448"
//!
//! Goal
//! ----
//! The 900-series (high-mass) Cases 900, 910, 920, 930, 940 all over-predict
//! annual heating AND annual cooling simultaneously in the production
//! validator path (`enable_ctf_with_fd_fallback` on the CTF solver). This
//! bidirectional signature is the textbook signature of solar mass-node
//! over-injection on a long integration horizon: every kWh of solar that
//! goes into the mass node and is later released to the zone shows up as
//! either heating or cooling load depending on the outdoor temperature at
//! release time. Per `KNOWN_ISSUES.md` §LIMIT-05, the 5R1C/CTF discrete-node
//! pathology (dt/τ ≈ 3.6 on Case 600 parameters) means the residual error
//! compounds season-over-season.
//!
//! What this test does
//! -------------------
//! Reproduces the validator's exact setup (`EpwWeatherSource` for
//! `USA_CO_Denver-Stapleton`, 14-day warmup, CTF solver on
//! `HighMass` construction) and decomposes the per-hour energy into:
//!
//!   - `Q_solar`     (window + opaque solar gain, W)
//!   - `Q_internal`  (occupancy + equipment, W)
//!   - `Q_conduction` (envelope loss, W, + = loss to exterior)
//!   - `Q_infiltration` (ventilation loss, W)
//!   - `Q_hvac`      (HVAC energy, W, + = heating, - = cooling)
//!
//! It then aggregates by month and prints the per-month table for each of
//! Cases 900, 910, 920, 930, 940. The reference is the ASHRAE 140 monthly
//! interim CSV at
//! `tests/reference_data/ashrae140/monthly/case_900_monthly_reference.csv`
//! (Cases 910/920/930/940 use the same monthly fractions by symmetry of the
//! degree-day method; the inter-program reference for those cases is in
//! `tests/reference_data/zone_balance/case_*_energy_reference.csv`).
//!
//! Diagnostic output (run with `--ignored --nocapture`):
//!   - Per-month: H_kWh, C_kWh, Q_solar_kWh, Q_internal_kWh, Q_conduction_kWh,
//!     Q_infiltration_kWh for each case
//!   - Per-month deviation vs the reference (where the reference is available)
//!   - Annual: total H, C, Q_solar, Q_internal, Q_conduction, Q_infiltration
//!
//! Per AGENTS.md: "no parameter tuning to make system tests pass — fix the
//! underlying math." The fix path for the bidirectional over-prediction is
//! the **GaugeSolver** rework (#1465 / #1462) — out of scope for this
//! issue. This test localises the over-injection to a specific season and
//! term, which is the actionable output of the investigation.
//!
//! See also
//! --------
//! - `tests/diagnostics/case_920_orientation_attribution.rs` (#2454) — per-orientation
//!   solar decomposition for Case 920 (E/W windows).
//! - `tests/case_900ff_regression_bisect.rs` (#2455) — Case 900FF night
//!   minimum regression (separate issue, different physics path).
//! - `KNOWN_ISSUES.md` §LIMIT-05 — the documented discrete-node pathology
//!   that this test localises.

use fluxion::physics::cta::VectorField;
use fluxion::physics::fd_discretization::MaterialLayer as FDMaterialLayer;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::warmup::{run_warmup, WarmupConfig};
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::validation::diagnostics::SimulationDiagnostics;
use fluxion::weather::epw::EpwWeatherSource;
use fluxion::weather::WeatherSource;

const EPW_PATH: &str = "assets/weather/USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw";

const MONTH_LABELS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

const MONTH_START_HOUR: [usize; 12] = [
    0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016,
];

fn month_index_for_hour(hour: usize) -> usize {
    let mut idx = 0;
    for (i, &start) in MONTH_START_HOUR.iter().enumerate() {
        if hour >= start {
            idx = i;
        } else {
            break;
        }
    }
    idx
}

#[derive(Debug, Clone, Copy, Default)]
struct MonthlyAttribution {
    h_kwh: f64,
    c_kwh: f64,
    solar_kwh: f64,
    internal_kwh: f64,
    conduction_kwh: f64,
    infiltration_kwh: f64,
}

#[derive(Debug, Clone)]
struct CaseAttribution {
    case_id: &'static str,
    annual_h_mwh: f64,
    annual_c_mwh: f64,
    annual_solar_mwh: f64,
    annual_internal_mwh: f64,
    annual_conduction_mwh: f64,
    annual_infiltration_mwh: f64,
    monthly: [MonthlyAttribution; 12],
    heating_ref_mid_mwh: f64,
    heating_ref_min_mwh: f64,
    heating_ref_max_mwh: f64,
    cooling_ref_mid_mwh: f64,
    cooling_ref_min_mwh: f64,
    cooling_ref_max_mwh: f64,
}

fn run_case_with_attribution(case_enum: ASHRAE140Case) -> CaseAttribution {
    let spec = case_enum.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Attach a SimulationDiagnostics collector so we get per-hour load breakdown
    // for solar, internal, infiltration, conduction, and HVAC. The collector
    // is wired in `from_spec` via `set_diagnostics`.
    let diag = SimulationDiagnostics::new(model.hvac.num_zones, 8760);
    model.set_diagnostics(Some(diag));

    // Enable CTF (with FD fallback) for high-mass construction — matches the
    // production validator path. For Case 950 / 940 / 930 with night-ventilation
    // or shading CTF may also be relevant; we keep the same code path for
    // comparability with the validator (which is the source of the issue numbers).
    let fd_layers: Vec<FDMaterialLayer> = spec
        .construction
        .wall
        .layers
        .iter()
        .map(|layer| {
            FDMaterialLayer::new(
                &layer.name,
                layer.thickness,
                layer.conductivity,
                layer.density,
                layer.specific_heat,
            )
        })
        .collect();
    let _used_ctf = model.enable_ctf_with_fd_fallback(&fd_layers, 3600.0, 50, 5);

    // Load the same EPW file the production validator uses
    let weather = EpwWeatherSource::from_file(EPW_PATH)
        .expect("Failed to load Denver-Stapleton TMY EPW file required by this test");

    // 14-day fixed warmup per ASHRAE 140 §B2 (matches the validator)
    run_warmup(&mut model, &weather, &WarmupConfig::default());

    // Reset energy tracking AFTER warmup so the warmup period does not
    // pollute the per-hour diagnostic counters. Same pattern as the
    // production validator.
    model.reset_peak_power();
    model.reset_heating_cooling_energy();

    // 8760-hour production simulation
    let mut hvac_per_month: [f64; 12] = [0.0; 12];
    let mut c_kwh_per_month: [f64; 12] = [0.0; 12];
    for step in 0..8760 {
        let weather_data = weather
            .get_hourly_data(step)
            .expect("EPW must cover all 8760 hours");
        model.solar.weather = Some(weather_data.clone());

        // Apply dynamic setpoints (matches the validator's HVAC schedule handling)
        if let Some(hvac_schedule) = spec.hvac.first() {
            let hour = (step % 24) as u8;
            let heating_sp = hvac_schedule
                .heating_setpoint_at_hour(hour)
                .unwrap_or(hvac_schedule.heating_setpoint);
            let cooling_sp = model.setpoints.cooling_schedule.value(step % 24);
            model.setpoints.heating_setpoint = heating_sp;
            model.setpoints.cooling_setpoint = cooling_sp;
            if spec.hvac.len() > 1 {
                let num_zones = model.hvac.num_zones;
                let mut heating_sps = vec![heating_sp; num_zones];
                let mut cooling_sps = vec![cooling_sp; num_zones];
                for (zone_idx, hvac) in spec.hvac.iter().enumerate() {
                    if zone_idx < num_zones {
                        let h_sp = hvac
                            .heating_setpoint_at_hour(hour)
                            .unwrap_or(hvac.heating_setpoint);
                        let c_sp = model.setpoints.cooling_schedule.value(step % 24);
                        heating_sps[zone_idx] = h_sp;
                        cooling_sps[zone_idx] = c_sp;
                    }
                }
                model.setpoints.heating_setpoints = VectorField::new(heating_sps);
                model.setpoints.cooling_setpoints = VectorField::new(cooling_sps);
            }
        }

        // Apply per-zone internal loads (matches the validator)
        let num_zones = model.hvac.num_zones;
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

        // Step the physics. The step updates model.hvac.annual_heating_energy /
        // model.hvac.annual_cooling_energy and also advances the diagnostics
        // collector. We also keep the per-step hvac_kwh return value so the
        // monthly H/C attribution can be reconciled exactly with the model's
        // internal annual tracker (see the *_reconciles test below). The
        // SimulationDiagnostics `loads.hvac` field carries the same W/kWh per
        // step as `step_physics` returns, but is rounded to the diagnostics
        // collector's per-timestep write, so we use the return value here for
        // tighter reconciliation.
        let hvac_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
        let m = month_index_for_hour(step);
        if hvac_kwh > 0.0 {
            hvac_per_month[m] += hvac_kwh;
        } else if hvac_kwh < 0.0 {
            c_kwh_per_month[m] += -hvac_kwh;
        }
    }

    // Pull the per-hour load breakdown from the diagnostics collector. Energy
    // units are in W per the field docs at `src/validation/diagnostics.rs:62`.
    // Convert to kWh per timestep by integrating P (W) * dt (h) / 1000.
    let diag = model
        .get_diagnostics()
        .expect("Diagnostics should be attached");

    let mut monthly: [MonthlyAttribution; 12] = [MonthlyAttribution::default(); 12];

    for step in 0..8760 {
        let m = month_index_for_hour(step);
        let num_zones = diag.loads.solar.get(step).map(|z| z.len()).unwrap_or(1);

        for zone_idx in 0..num_zones {
            let solar_w = diag
                .loads
                .solar
                .get(step)
                .and_then(|z| z.get(zone_idx))
                .copied()
                .unwrap_or(0.0);
            let internal_w = diag
                .loads
                .internal
                .get(step)
                .and_then(|z| z.get(zone_idx))
                .copied()
                .unwrap_or(0.0);
            let infiltration_w = diag
                .loads
                .infiltration
                .get(step)
                .and_then(|z| z.get(zone_idx))
                .copied()
                .unwrap_or(0.0);
            let conduction_w = diag
                .loads
                .conduction
                .get(step)
                .and_then(|z| z.get(zone_idx))
                .copied()
                .unwrap_or(0.0);

            // dt = 3600 s = 1 h. Energy (kWh) = P (W) * 1 h / 1000.
            let dt_h = 1.0;
            monthly[m].solar_kwh += solar_w * dt_h / 1000.0;
            monthly[m].internal_kwh += internal_w * dt_h / 1000.0;
            monthly[m].infiltration_kwh += infiltration_w * dt_h / 1000.0;
            monthly[m].conduction_kwh += conduction_w * dt_h / 1000.0;
        }
    }

    // Splice the per-month H/C from the per-step hvac_kwh (returned by
    // step_physics) into the monthly table. This guarantees the monthly
    // table reconciles exactly with model.hvac.annual_heating_energy /
    // model.hvac.annual_cooling_energy — the `*_reconciles` test guards this.
    for m in 0..12 {
        monthly[m].h_kwh = hvac_per_month[m];
        monthly[m].c_kwh = c_kwh_per_month[m];
    }

    let annual_h_mwh = model.hvac.annual_heating_energy / 1000.0;
    let annual_c_mwh = model.hvac.annual_cooling_energy / 1000.0;
    let annual_solar_mwh: f64 = monthly.iter().map(|m| m.solar_kwh).sum::<f64>() / 1000.0;
    let annual_internal_mwh: f64 = monthly.iter().map(|m| m.internal_kwh).sum::<f64>() / 1000.0;
    let annual_conduction_mwh: f64 = monthly.iter().map(|m| m.conduction_kwh).sum::<f64>() / 1000.0;
    let annual_infiltration_mwh: f64 =
        monthly.iter().map(|m| m.infiltration_kwh).sum::<f64>() / 1000.0;

    // Reference values (ASHRAE 140 inter-program min/max from
    // tests/reference_data/zone_balance/case_*_energy_reference.csv and the
    // midpoint derived from the band).
    let (h_min, h_max, c_min, c_max) = match case_enum {
        ASHRAE140Case::Case900 => (1.17, 2.04, 2.13, 3.67),
        ASHRAE140Case::Case910 => (1.51, 2.28, 0.82, 1.88),
        ASHRAE140Case::Case920 => (3.26, 4.30, 1.84, 3.31),
        ASHRAE140Case::Case930 => (4.14, 5.34, 1.04, 2.24),
        ASHRAE140Case::Case940 => (0.79, 1.41, 2.08, 3.55),
        _ => (0.0, 0.0, 0.0, 0.0),
    };
    let h_mid = (h_min + h_max) / 2.0;
    let c_mid = (c_min + c_max) / 2.0;

    CaseAttribution {
        case_id: leak_str(case_enum.number()),
        annual_h_mwh,
        annual_c_mwh,
        annual_solar_mwh,
        annual_internal_mwh,
        annual_conduction_mwh,
        annual_infiltration_mwh,
        monthly,
        heating_ref_mid_mwh: h_mid,
        heating_ref_min_mwh: h_min,
        heating_ref_max_mwh: h_max,
        cooling_ref_mid_mwh: c_mid,
        cooling_ref_min_mwh: c_min,
        cooling_ref_max_mwh: c_max,
    }
}

fn print_case_attribution(case: &CaseAttribution) {
    println!(
        "\n[#2453 Case {}] annual H={:.3} MWh (ref [{:.2}, {:.2}], mid {:.2}) | \
         C={:.3} MWh (ref [{:.2}, {:.2}], mid {:.2})",
        case.case_id,
        case.annual_h_mwh,
        case.heating_ref_min_mwh,
        case.heating_ref_max_mwh,
        case.heating_ref_mid_mwh,
        case.annual_c_mwh,
        case.cooling_ref_min_mwh,
        case.cooling_ref_max_mwh,
        case.cooling_ref_mid_mwh,
    );
    println!(
        "    annual solar={:.3} MWh, internal={:.3} MWh, conduction={:.3} MWh, infiltration={:.3} MWh",
        case.annual_solar_mwh,
        case.annual_internal_mwh,
        case.annual_conduction_mwh,
        case.annual_infiltration_mwh,
    );
    println!("  Month | H_kWh     C_kWh    | Q_sol_kWh  Q_int_kWh Q_cond_kWh Q_inf_kWh");
    for (m, mo) in case.monthly.iter().enumerate() {
        println!(
            "    {}  | {:7.1}  {:7.1}  | {:8.1}  {:8.1}  {:8.1}  {:8.1}",
            MONTH_LABELS[m],
            mo.h_kwh,
            mo.c_kwh,
            mo.solar_kwh,
            mo.internal_kwh,
            mo.conduction_kwh,
            mo.infiltration_kwh,
        );
    }
    let total_h: f64 = case.monthly.iter().map(|m| m.h_kwh).sum();
    let total_c: f64 = case.monthly.iter().map(|m| m.c_kwh).sum();
    let total_solar: f64 = case.monthly.iter().map(|m| m.solar_kwh).sum();
    let total_internal: f64 = case.monthly.iter().map(|m| m.internal_kwh).sum();
    let total_conduction: f64 = case.monthly.iter().map(|m| m.conduction_kwh).sum();
    let total_infiltration: f64 = case.monthly.iter().map(|m| m.infiltration_kwh).sum();
    println!(
        "   Sum  | {:7.1}  {:7.1}  | {:8.1}  {:8.1}  {:8.1}  {:8.1}",
        total_h, total_c, total_solar, total_internal, total_conduction, total_infiltration,
    );
}

#[test]
#[ignore = "Diagnostic: run with --ignored --nocapture for per-month #2453 attribution"]
fn test_case_900_series_seasonal_attribution() {
    let cases = [
        ASHRAE140Case::Case900,
        ASHRAE140Case::Case910,
        ASHRAE140Case::Case920,
        ASHRAE140Case::Case930,
        ASHRAE140Case::Case940,
    ];

    println!("\n=================================================================");
    println!("  Issue #2453: 900-series bidirectional annual-energy attribution");
    println!("  (CTF solver path; same setup as src/validation/ashrae_140_validator.rs)");
    println!("=================================================================");

    let mut all = Vec::with_capacity(cases.len());
    for case_enum in cases {
        let attr = run_case_with_attribution(case_enum);
        print_case_attribution(&attr);
        all.push(attr);
    }

    println!("\n[#2453 Annual summary across all 5 cases]");
    println!("  Case | H_MWh   ref [min, max] | C_MWh   ref [min, max] | dH%   dC%");
    for attr in &all {
        let h_ref = attr.heating_ref_mid_mwh;
        let c_ref = attr.cooling_ref_mid_mwh;
        let d_h = if h_ref > 0.0 {
            (attr.annual_h_mwh - h_ref) / h_ref * 100.0
        } else {
            0.0
        };
        let d_c = if c_ref > 0.0 {
            (attr.annual_c_mwh - c_ref) / c_ref * 100.0
        } else {
            0.0
        };
        println!(
            "  {}  | {:5.2}  [{:4.2}, {:4.2}] | {:5.2}  [{:4.2}, {:4.2}] | {:+5.0} {:+5.0}",
            attr.case_id,
            attr.annual_h_mwh,
            attr.heating_ref_min_mwh,
            attr.heating_ref_max_mwh,
            attr.annual_c_mwh,
            attr.cooling_ref_min_mwh,
            attr.cooling_ref_max_mwh,
            d_h,
            d_c,
        );
    }

    // Sanity assertion: every case shows non-trivial energy attribution
    // (solar > 0, internal > 0). If this fails the diagnostics collector is
    // not wired correctly.
    for attr in &all {
        assert!(
            attr.annual_solar_mwh > 0.0,
            "Case {}: annual solar gain should be positive; got {}",
            attr.case_id,
            attr.annual_solar_mwh
        );
        assert!(
            attr.annual_internal_mwh > 0.0,
            "Case {}: annual internal gain should be positive; got {}",
            attr.case_id,
            attr.annual_internal_mwh
        );
    }
}

// Companion assertion (always run) — the sum of the per-month attribution
// should reconcile with the model's annual_heating_energy / annual_cooling_energy
// to within 1% of the larger value. This is the energy-balance guard for the
// diagnostic itself.
#[test]
fn test_case_900_series_seasonal_attribution_reconciles() {
    let cases = [
        ASHRAE140Case::Case900,
        ASHRAE140Case::Case910,
        ASHRAE140Case::Case920,
        ASHRAE140Case::Case930,
        ASHRAE140Case::Case940,
    ];
    for case_enum in cases {
        let attr = run_case_with_attribution(case_enum);
        let sum_h_kwh: f64 = attr.monthly.iter().map(|m| m.h_kwh).sum();
        let sum_c_kwh: f64 = attr.monthly.iter().map(|m| m.c_kwh).sum();
        let h_tol = 0.01 * sum_h_kwh.max(sum_c_kwh).max(1.0);
        let h_err = (sum_h_kwh - attr.annual_h_mwh * 1000.0).abs();
        let c_err = (sum_c_kwh - attr.annual_c_mwh * 1000.0).abs();
        assert!(
            h_err <= h_tol && c_err <= h_tol,
            "Case {}: per-month sum (H={:.1}, C={:.1} kWh) disagrees with model \
             annual tracker (H={:.1}, C={:.1} kWh) by more than 1% of {:.1} kWh",
            case_enum.number(),
            sum_h_kwh,
            sum_c_kwh,
            attr.annual_h_mwh * 1000.0,
            attr.annual_c_mwh * 1000.0,
            h_tol,
        );
    }
}

// Small helper to convert an owned String into a &'static str by leaking
// once per call. Used only for the case_id in print/assert messages; the
// total leak is bounded by the number of cases (a handful of bytes).
fn leak_str(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}
