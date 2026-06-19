//! Diagnostic harness for Issue #1168: Free-float temperature over-damping.
//!
//! Prints the 5R1C network parameters, time constants, and per-case
//! free-floating min/max (air AND mass) so the root cause can be
//! distinguished between (a) heat-transfer-rate / envelope conductance
//! and (b) effective thermal capacitance / mass-air coupling.
//!
//! Run with:
//!   cargo test --test issue_1168_free_float_diagnostic -- --nocapture --ignored

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

fn first(v: &VectorField) -> f64 {
    v.as_slice().first().copied().unwrap_or(f64::NAN)
}

#[test]
#[ignore]
fn diagnose_free_float_network_and_temps() {
    let cases = [
        ("600FF", ASHRAE140Case::Case600FF),
        ("650FF", ASHRAE140Case::Case650FF),
        ("900FF", ASHRAE140Case::Case900FF),
        ("950FF", ASHRAE140Case::Case950FF),
    ];

    println!("\n========== ISSUE #1168 FREE-FLOAT OVER-DAMPING DIAGNOSTIC ==========");

    for (name, case) in cases {
        let spec = case.spec();
        let mut model = ThermalModel::<VectorField>::from_spec(&spec);
        let weather = DenverTmyWeather::new();

        // Ensure free-float (no HVAC)
        model.hvac_heating_capacity = 0.0;
        model.hvac_cooling_capacity = 0.0;
        model.hvac_enabled = VectorField::from_scalar(0.0, model.num_zones);

        // ---- Print network parameters (steady-state + dynamic) ----
        let h_ms = first(&model.h_tr_ms); // mass<->surface
        let h_is = first(&model.h_tr_is); // surface<->air
        let h_em = first(&model.h_tr_em); // exterior<->mass (opaque envelope)
        let h_w = first(&model.h_tr_w); // windows (air<->outdoor)
        let h_ve = first(&model.h_ve); // ventilation/infiltration (air<->outdoor)
        let cm = first(&model.thermal_capacitance); // mass capacitance [J/K]
        let h_ms_is_prod = first(&model.derived_h_ms_is_prod);
        let term_rest_1 = first(&model.derived_term_rest_1);
        let den = first(&model.derived_den);
        let h_tr_3 = first(&model.derived_h_tr_3);
        let h_ext = h_w + h_ve; // air<->outdoor total

        // Steady-state air<->mass series conductance (as used implicitly in t_i_free)
        let h_air_mass_ss = if (h_ms + h_is) > 0.0 {
            h_ms * h_is / (h_ms + h_is)
        } else {
            0.0
        };

        // Mass node time constant (tau = Cm / H_total_to_mass)
        // Mass couples to outdoor via h_em AND to air via h_tr_3
        let tau_mass_hours = cm / (h_em + h_tr_3).max(1e-9) / 3600.0;

        println!("\n--- Case {} ---", name);
        println!("  h_tr_ms (mass<->surface) : {:>10.2} W/K", h_ms);
        println!("  h_tr_is (surface<->air)  : {:>10.2} W/K", h_is);
        println!("  h_tr_em (ext<->mass)     : {:>10.2} W/K", h_em);
        println!("  h_tr_w  (windows)        : {:>10.2} W/K", h_w);
        println!("  h_ve    (ventilation)    : {:>10.2} W/K", h_ve);
        println!("  h_ext = h_w + h_ve       : {:>10.2} W/K", h_ext);
        println!("  Cm (mass capacitance)    : {:>10.2e} J/K", cm);
        println!("  ---- derived (used in t_i_free) ----");
        println!("  h_ms_is_prod (h_ms*h_is) : {:>10.2} W^2/K^2", h_ms_is_prod);
        println!("  term_rest_1 (h_ms+h_is)  : {:>10.2} W/K", term_rest_1);
        println!("  den (t_i_free denom)     : {:>10.2} W^2/K^2", den);
        println!("  h_tr_3 (dynamic mass)    : {:>10.2} W/K", h_tr_3);
        println!("  ---- derived diagnostics ----");
        println!("  H_air_mass (ss series)   : {:>10.2} W/K", h_air_mass_ss);
        println!(
            "  H_air_mass / h_ext ratio : {:>10.3}  (>1 => air tracks sluggish mass)",
            h_air_mass_ss / h_ext.max(1e-9)
        );
        println!("  tau_mass (Cm/(h_em+h3))  : {:>10.2} hours", tau_mass_hours);

        // ---- Run free-float for the year ----
        // Warm up (2 years) to reach periodic steady state for high-mass cases.
        let warmup = 2usize;
        let total_steps = 8760;
        for _ in 0..warmup {
            for step in 0..total_steps {
                let wd = weather.get_hourly_data(step).unwrap();
                model.weather = Some(wd.clone());
                model.step_physics(step, wd.dry_bulb_temp, 3600.0);
            }
        }

        let mut min_air = f64::INFINITY;
        let mut max_air = f64::NEG_INFINITY;
        let mut min_mass = f64::INFINITY;
        let mut max_mass = f64::NEG_INFINITY;
        // Track outdoor for swing reference
        let mut min_out = f64::INFINITY;
        let mut max_out = f64::NEG_INFINITY;

        for step in 0..total_steps {
            let wd = weather.get_hourly_data(step).unwrap();
            model.weather = Some(wd.clone());
            model.step_physics(step, wd.dry_bulb_temp, 3600.0);
            let air = model.temperatures.as_slice()[0];
            let mass = model.mass_temperatures.as_slice()[0];
            min_air = min_air.min(air);
            max_air = max_air.max(air);
            min_mass = min_mass.min(mass);
            max_mass = max_mass.max(mass);
            min_out = min_out.min(wd.dry_bulb_temp);
            max_out = max_out.max(wd.dry_bulb_temp);
        }

        println!("  ---- free-float annual results ----");
        println!("  Outdoor swing            : {:.1} to {:.1} C  (Δ{:.1})", min_out, max_out, max_out - min_out);
        println!("  AIR temp swing           : {:.1} to {:.1} C  (Δ{:.1})", min_air, max_air, max_air - min_air);
        println!("  MASS temp swing          : {:.1} to {:.1} C  (Δ{:.1})", min_mass, max_mass, max_mass - min_mass);
        println!(
            "  Mass swing / Air swing   : {:.3}  (<1 => mass damps air)",
            (max_mass - min_mass) / (max_air - min_air).max(1e-9)
        );
    }
    println!("\n==================================================================\n");
}
