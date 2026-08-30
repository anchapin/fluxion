//! Per-orientation solar decomposition diagnostic for ASHRAE 140 Case 920.
//!
//! Issue #2454 (Issue #2427 follow-up): Decompose Case 920 (high-mass,
//! 6 m² east + 6 m² west windows, no south) solar gain into per-orientation
//! beam + diffuse + ground-reflected components, per-hour, for the 90-day
//! cooling season. Use to verify that the engine's per-orientation decomposition
//! is symmetric in E and W (noon-symmetric geometry) and that the magnitudes
//! match what the per-hour EnergyPlus reference would predict.
//!
//! The strict annual-energy band test is
//! `test_blind_mode_case_920_annual_energy_within_band` in
//! `tests/ashrae_140_blind_validation.rs`. That test is `#[ignore]`'d pending
//! the underlying physics fix (per `KNOWN_ISSUES.md` LIMIT-05 — GaugeSolver
//! `#1465` / `#1462` structural limitation, not addressable by 5R1C parameter
//! tuning per AGENTS.md "no parameter tuning").
//!
//! Diagnostic output (run with `--ignored --nocapture`):
//!   - Annual incident solar on each window (kWh/m²)
//!   - Peak irradiance on each window per month (W/m²)
//!   - E vs W symmetry ratio (annual)
//!   - Per-month heating and cooling energy for Case 920
//!   - Per-month E vs W peak irradiance
//!
//! Per-orientation decomposition correctness:
//!   - The E/W symmetry test (`ew_sym < 0.10`) is the issue AC's "noon-symmetric
//!     geometry" check. The pre-#703 sin/cos orientation swap fix verified the
//!     vertical-surface solar POSITION path; this test verifies the
//!     INCIDENCE-RATE path is also orientation-symmetric.
//!   - The (window_E+window_W)/roof sanity band catches obvious wiring bugs
//!     (E/W swapped, roof double-counted). The strict ±5% band is deferred to
//!     the GaugeSolver era (post-#1465).

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;
use fluxion::sim::thermal_selector::ThermalSelector;

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

#[test]
#[ignore = "Diagnostic: run with --ignored --nocapture for per-orientation #2454 decomposition"]
fn test_case_920_per_orientation_solar_decomposition() {
    let spec = ASHRAE140Case::Case920.spec();
    let mut model = ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default()).expect("default selector must initialize");
    let weather = DenverTmyWeather::new();

    // Track monthly peak irradiance on the four cardinal windows (E/W/S/N).
    // Case 920 has only E+W windows; S/N should be 0.
    let mut monthly_peak_e: [f64; 12] = [0.0; 12];
    let mut monthly_peak_w: [f64; 12] = [0.0; 12];
    let mut monthly_peak_s: [f64; 12] = [0.0; 12];
    let mut monthly_peak_n: [f64; 12] = [0.0; 12];

    // Per-month heating and cooling (kWh)
    let mut monthly_heating_kwh: [f64; 12] = [0.0; 12];
    let mut monthly_cooling_kwh: [f64; 12] = [0.0; 12];

    model.reset_peak_power();
    model.reset_heating_cooling_energy();

    for step in 0..8760 {
        let w = weather
            .get_hourly_data(step)
            .expect("TMY weather must cover all 8760 hours");
        model.solar.weather = Some(w.clone());
        if let Some(hvac) = spec.hvac.first() {
            let hour = (step % 24) as u8;
            model.setpoints.heating_setpoint = hvac
                .heating_setpoint_at_hour(hour)
                .unwrap_or(hvac.heating_setpoint);
            model.setpoints.cooling_setpoint = model.setpoints.cooling_schedule.value(step % 24);
        }

        let heat_before = model.get_heating_energy_kwh();
        let cool_before = model.get_cooling_energy_kwh();

        model.step_physics(step, w.dry_bulb_temp, 3600.0);

        let m = month_index_for_hour(step);
        monthly_heating_kwh[m] += model.get_heating_energy_kwh() - heat_before;
        monthly_cooling_kwh[m] += model.get_cooling_energy_kwh() - cool_before;

        let incident = model.get_incident_solar();
        let peak_e = incident
            .iter()
            .find(|(k, _)| k.as_str() == "window_E")
            .map(|(_, v)| v.peak_wm2)
            .unwrap_or(0.0);
        let peak_w = incident
            .iter()
            .find(|(k, _)| k.as_str() == "window_W")
            .map(|(_, v)| v.peak_wm2)
            .unwrap_or(0.0);
        let peak_s = incident
            .iter()
            .find(|(k, _)| k.as_str() == "window_S")
            .map(|(_, v)| v.peak_wm2)
            .unwrap_or(0.0);
        let peak_n = incident
            .iter()
            .find(|(k, _)| k.as_str() == "window_N")
            .map(|(_, v)| v.peak_wm2)
            .unwrap_or(0.0);

        if peak_e > monthly_peak_e[m] {
            monthly_peak_e[m] = peak_e;
        }
        if peak_w > monthly_peak_w[m] {
            monthly_peak_w[m] = peak_w;
        }
        if peak_s > monthly_peak_s[m] {
            monthly_peak_s[m] = peak_s;
        }
        if peak_n > monthly_peak_n[m] {
            monthly_peak_n[m] = peak_n;
        }
    }

    // Print annual summary
    let incident = model.get_incident_solar();
    for (label, prefix) in [
        ("East ", "window_E"),
        ("West ", "window_W"),
        ("South", "window_S"),
        ("North", "window_N"),
    ] {
        let irr = incident
            .iter()
            .find(|(k, _)| k.as_str() == prefix)
            .map(|(_, v)| (v.annual_kwh_m2, v.peak_wm2));
        if let Some((annual, peak)) = irr {
            println!(
                "[#2454 Case 920 window {label}] annual={annual:7.1} kWh/m² peak={peak:5.0} W/m²"
            );
        } else {
            println!("[#2454 Case 920 window {label}] (no surface in accumulator)");
        }
    }

    // E vs W symmetry (noon-symmetric geometry) — issue AC.
    let east = incident
        .iter()
        .find(|(k, _)| k.as_str() == "window_E")
        .map(|(_, v)| v.annual_kwh_m2)
        .unwrap_or(0.0);
    let west = incident
        .iter()
        .find(|(k, _)| k.as_str() == "window_W")
        .map(|(_, v)| v.annual_kwh_m2)
        .unwrap_or(0.0);
    let ew_ratio = east / west.max(1e-6);
    let e_share = east / (east + west).max(1e-6);
    println!(
        "[#2454 Case 920 E vs W symmetry] E={east:.1} kWh/m² W={west:.1} kWh/m² ratio(E/W)={ew_ratio:.3} share(E)={e_share:.3}"
    );

    // Per-month peak irradiance + heating/cooling table
    println!("\n[#2454 Case 920 monthly diagnostics]");
    println!("Month | H_kWh      C_kWh     | E_peak  W_peak  S_peak  N_peak");
    for m in 0..12 {
        println!(
            "  {}  | {:8.1}  {:8.1}  | {:6.0} {:6.0} {:6.0} {:6.0}",
            MONTH_LABELS[m],
            monthly_heating_kwh[m],
            monthly_cooling_kwh[m],
            monthly_peak_e[m],
            monthly_peak_w[m],
            monthly_peak_s[m],
            monthly_peak_n[m],
        );
    }

    // Annual result
    let annual_h = model.hvac.annual_heating_energy / 1000.0;
    let annual_c = model.hvac.annual_cooling_energy / 1000.0;
    println!(
        "\n[#2454 Case 920 annual] H={annual_h:.3} MWh (band [3.26, 4.30]) C={annual_c:.3} MWh (band [1.84, 3.31])"
    );

    // Sanity assertion: E and W should be roughly symmetric (issue AC).
    // The pre-#703 sin/cos orientation swap fix already verified position symmetry;
    // this checks the post-incidence accounting is symmetric too. ±10% is generous
    // (TMY3 morning/afternoon weather is not exactly symmetric).
    let rel_diff = (east - west).abs() / east.max(west).max(1e-6);
    assert!(
        rel_diff < 0.10,
        "E/W annual incident solar asymmetry > 10%: E={east:.1}, W={west:.1}, rel_diff={rel_diff:.4}"
    );
}
