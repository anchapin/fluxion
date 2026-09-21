//! LIMIT-21 Phase 6 Diagnostic: Case 640 Annual Cooling Gap Isolation
//!
//! Issue #3911 — Gauge solver annual cooling reports 1.314 MWh (22% of the
//! 5.95 MWh lower reference bound) vs the 5.95–8.10 MWh ASHRAE 140 band.
//!
//! This test instruments Case 640 (LowMass + thermostat setback) with the
//! gauge solver to isolate which of the three LIMIT-30 B1a hypotheses drives
//! the remaining annual cooling gap:
//!
//! **Hypothesis 1 — Per-surface distribution routing:**
//!   The gauge routes 100% of window solar through the window-conduction path.
//!   The 5R1C model routes `solar_distribution_to_air = 0.30` (30%) directly to
//!   the zone air node and 70% through the opaque surface path.
//!   Metric: `solar_to_air_ratio_gauge = Q_window_solar / Q_total_solar`
//!   Metric: `solar_to_air_ratio_5r1c = solar_distribution_to_air = 0.30`
//!   Δ reveals whether missing direct-to-air routing explains the gap.
//!
//! **Hypothesis 2 — Per-surface 5R1C conductance re-derivation:**
//!   The B1a audit found h_tr_is/h_tr_ms deviations of −86.8% / −78.0%
//!   vs hand-calculation. If the gauge's per-surface R_total (m²K/W) differs
//!   from the 5R1C h_tr_is by a similar factor, the conductance derivation
//!   is the cause.
//!   Metric: `R_gauge_vs_h_tr_is = R_total / (1/h_tr_is)`
//!
//! **Hypothesis 3 — Family-level 5R1C lumped-mass-node damping:**
//!   The gauge uses implicit Euler with N=3 sub-steps (dt_sub = 1200s) vs
//!   5R1C's single implicit Euler step (dt = 3600s). At comparable τ_air,
//!   the gauge's sub-stepping gives ~84% equilibration/hr vs 5R1C's ~78%/hr.
//!   This ~5% excess equilibration per hour drives faster overnight cooling,
//!   contributing to the annual cooling deficit.
//!   Metric: `dt_over_tau = dt_seconds / τ_air` where τ_air = C_air / h_total
//!   Note: before the τ_air formula fix, the gauge's τ_air was 7200s (7× too
//!   large), giving only ~40%/hr equilibration — severely under-damped.
//!
//! # References
//! - Issue #3911: https://github.com/anchapin/fluxion/issues/3911
//! - LIMIT-30 B1a: Issue #3797 / PR #3847
//! - LIMIT-21 Phase 5: PR #3910 (commit 91c7238)
//!
//! This test is DIAGNOSTIC ONLY — it does not modify production code or
//! assert on the gap. It emits structured output for offline analysis.

use fluxion::physics::cta::VectorField;
#[cfg(feature = "gauge-solver")]
use fluxion::physics::gauge_zone_solver::SurfaceType;
use fluxion::physics::units::ToF64;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::epw::EpwWeatherSource;
use fluxion::weather::WeatherSource;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

/// Reference values for Case 640 (ASHRAE 140-2023 Annex B).
const CASE640_COOLING_MIN_MWH: f64 = 5.95;
const CASE640_COOLING_MAX_MWH: f64 = 8.10;
const CASE640_HEATING_MIN_MWH: f64 = 2.75;
const CASE640_HEATING_MAX_MWH: f64 = 3.80;

/// J → MWh conversion.
const J_TO_MWH: f64 = 1.0 / 3.6e9;

// ============================================================================
// Hypothesis 1 — Per-surface distribution routing
// ============================================================================
//
// The 5R1C model (step_5r1c.rs) distributes solar gains as:
//
//   sol_to_air      = sol_W × solar_distribution_to_air   (direct to zone air)
//   remaining_sol   = sol_W × (1 − solar_distribution_to_air)  (to opaque surfaces)
//
// For Case 640 (solar_distribution_to_air = 0.30):
//   → 30% of zone solar goes DIRECTLY to zone air as convective gain
//   → 70% goes through opaque surfaces (conduction → zone air)
//
// The gauge model (gauge_zone_solver.rs step()):
//   → 100% of horizontal GHI goes through the WINDOW conduction path
//   → opaque surfaces receive 0% of solar (solar_fraction = 0.0 for Wall/Roof/Floor)
//
// Both models use the SAME `inputs.solar_gains.first()` (horizontal GHI) as the
// starting point. The architectural difference is:
//   5R1C: 30% goes straight to zone air as "instant" cooling demand
//   Gauge: 100% goes through window conduction before reaching zone air
//
// The window-conduction path introduces a time lag vs the direct-to-air path.
// This lag, compounded over the annual cycle, reduces total cooling demand.
//
// DIAGNOSTIC: At each timestep compute the gauge's effective "direct to air"
// fraction. Since all window solar eventually enters the zone air (conservation
// of energy), the question is HOW FAST it gets there. The gauge window path
// has an effective time constant of τ_window = R_window × C_window. For thin
// glazing (R_window ≈ 0.15 m²K/W, C_window ≈ 10 kJ/m²K), τ ≈ 1500 s = 0.4 h.
// The 5R1C direct-to-air path has τ ≈ 0 (instantaneous).
//
// If the annual cooling deficit correlates with the fraction of solar arriving
// "late" (τ_window timescale), hypothesis 1 is confirmed.

// ============================================================================
// Hypothesis 2 — Per-surface 5R1C conductance re-derivation
// ============================================================================
//
// The 5R1C model computes h_tr_is (interior surface-to-air conductance) as:
//
//   h_tr_is = h_c + h_r
//   where h_c = interior convection coefficient [W/m²K]
//   and   h_r = interior radiation coefficient [W/m²K]
//
// ASHRAE 140 / ISO 13790 specifies:
//   h_c = Nu × k_air / L_char  (natural convection, vertical plate)
//   h_r = 4 × ε × σ × (T_mean)³  (linearized long-wave)
//
// For the gauge model, each surface has its own R_total = 1/h_exterior + R_wall.
// The effective "surface to zone air" conductance is encoded in the gauge's
// step_with_boundary_conditions() via the effective exterior temperature formula:
//
//   t_ext = T_outdoor + G_solar/h_ext + (h_rad_sky/h_ext)×(T_sky − T_outdoor)
//   q = (t_ext − T_int) / R_total
//
// The zone air energy equation uses Σ(q_j × A_j) as the heat source.
//
// DIAGNOSTIC: Compare the gauge's per-surface R_total against the hand-computed
// h_tr_is from the 5R1C formula. For Case 640 low-mass construction:
//   R_wall ≈ 0.32 m²K/W (U ≈ 3.1 W/m²K)
//   h_ext = 25 W/m²K (exterior film)
//   h_tr_is ≈ 8.3 W/m²K (interior convective + radiative)
//
// If R_total_gauge ≠ 1/h_tr_is by a large factor, hypothesis 2 is confirmed.

// ============================================================================
// Hypothesis 3 — Family-level 5R1C lumped-mass-node damping
// ============================================================================
//
// The 5R1C air node ODE: C_air × dT_air/dt = Q_net + h_tr_is×(T_is − T_air) + h_ve×(T_ext − T_air)
//
// Discrete form (implicit Euler, dt = 3600 s, τ_air = C_air/h_total):
//   T_air_new = (C_air × T_air_old + dt × Q) / (C_air + dt × h_total)
//             = T_air_old × (1 − dt/τ) + (dt/C_air) × Q  [where h_ve >> h_tr_is]
//             ≈ T_air_old × exp(−dt/τ)  [for small dt/τ]
//
// At τ_air ≈ 1000 s and dt = 3600 s: dt/τ_air ≈ 3.6
//   → Analytical: T_air_new = T_air_old × exp(−3.6) ≈ 0.027 × T_air_old  (97.3% equilibration)
//   → Implicit Euler: T_air_new = (C×T_old + dt×Q) / (C + dt×h)
//                  = (τ×T_old + dt×Q) / (τ + dt) = (T_old + (dt/τ)×Q/C) / (1 + dt/τ)
//                  = (T_old + 3.6×Q/C) / 4.6 ≈ 0.22×T_old + 0.78×Q/C
//                  ≈ 0.78 × equilibration per hour (22% thermal memory retained)
//
// GAUGE MODEL: sub-stepping with N=3 and dt_sub = 1200 s each:
//   dt_sub/τ ≈ 1.2
//   After 3 sub-steps: T_air_new = (T_old + 1.2×Q/C) / 2.2  [sub-step 1]
//                      → T_1 = (T_old + 1.2×Q/C) / 2.2
//                      T_2 = (T_1 + 1.2×Q/C) / 2.2  [sub-step 2]
//                      T_3 = (T_2 + 1.2×Q/C) / 2.2  [sub-step 3]
//   After 3 sub-steps with constant Q: effective equilibration ≈ 0.84 per hour
//   (16% thermal memory retained) vs 5R1C's implicit Euler at dt/τ=3.6: ≈ 0.22
//   (78% memory retained)
//
// The gauge model with N=3 sub-steps UNDER-DAMPs relative to the true 5R1C
// implicit Euler solution. This means the gauge air node responds TOO SLOWLY
// to solar transients — T_air tracks the load with a lag that effectively
// reduces peak cooling demand.
//
// DIAGNOSTIC: Compute the gauge's effective dt/τ_air and compare to 5R1C's
// dt/τ_air. The gauge uses implicit Euler with N sub-steps, giving a different
// effective equilibration rate. Compute:
//   τ_air_gauge = C_air / h_total  (using gauge's h_total from infiltration ACH)
//   dt_over_tau_gauge = dt_seconds / τ_air_gauge
// If dt_over_tau_gauge < 3.6 (5R1C reference), the gauge under-damps.

/// Result of one diagnostic timestep.
#[derive(Debug, Clone)]
#[allow(non_snake_case)] // camelCase fields match CSV column headers
struct TimestepDiag {
    step: usize,
    hour: usize,
    day_of_year: usize,
    // Outdoor conditions
    outdoor_temp_C: f64,
    solar_horizontal_Wm2: f64,
    // Zone conditions
    t_zone_C: f64,
    t_air_C: f64,
    // Energy
    energy_kwh: f64,
    // Hypothesis 1: Solar routing
    window_solar_fraction: f64, // fraction of zone solar from windows (gauge)
    gauge_solar_to_air_ratio: f64, // effective "direct to air" fraction in gauge
    // Hypothesis 2: Conductance
    window_R_total: f64, // m²K/W
    opaque_R_total: f64, // m²K/W (average)
    // Hypothesis 3: Air-node damping
    dt_over_tau: f64,
    tau_air_s: f64,
    gauge_equilibration_per_hour: f64,
    five_r1c_equilibration_per_hour: f64,
}

impl TimestepDiag {
    fn header_csv() -> &'static str {
        "step,hour,DOY,T_outdoor_C,solar_Wm2,T_zone_C,T_air_C,energy_kWh,window_solar_fraction,gauge_solar_to_air_ratio,window_R_total_m2K_W,opaque_R_total_m2K_W,dt_over_tau,tau_air_s,gauge_equil_per_hour,5R1C_equil_per_hour"
    }

    fn to_csv(&self) -> String {
        format!(
            "{},{},{},{:.4},{:.4},{:.4},{:.4},{:.6},{:.4},{:.4},{:.6},{:.6},{:.6},{:.6},{:.4},{:.4}",
            self.step,
            self.hour,
            self.day_of_year,
            self.outdoor_temp_C,
            self.solar_horizontal_Wm2,
            self.t_zone_C,
            self.t_air_C,
            self.energy_kwh,
            self.window_solar_fraction,
            self.gauge_solar_to_air_ratio,
            self.window_R_total,
            self.opaque_R_total,
            self.dt_over_tau,
            self.tau_air_s,
            self.gauge_equilibration_per_hour,
            self.five_r1c_equilibration_per_hour
        )
    }
}

/// Run the Case 640 diagnostic and return per-timestep results.
#[cfg(feature = "gauge-solver")]
#[allow(non_snake_case)] // local vars intentionally match CSV column naming
fn run_case640_diagnostic() -> Vec<TimestepDiag> {
    let spec = ASHRAE140Case::Case640.spec();
    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");
    let weather = EpwWeatherSource::from_file("assets/weather/WD600.epw")
        .expect("Failed to load EPW weather data for Case 640");

    // Case 640 infiltration ACH (from the spec / 5R1C model)
    let infiltration_ach = 0.5;

    // Get gauge zone solver for telemetry (if available)
    let gauge_available = model
        .conduction
        .backend
        .gauge_zone_solver
        .is_some();

    let mut results = Vec::with_capacity(8760);

    let mut heating_total_j = 0.0;
    let mut cooling_total_j = 0.0;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(weather_data.clone());
        let outdoor_temp = weather_data.dry_bulb_temp;
        let solar_horizontal = weather_data.ghi;

        let energy_kwh = model.step_physics(step, outdoor_temp, 3600.0);
        let energy_j = energy_kwh * 3.6e6;

        if energy_kwh > 0.0 {
            heating_total_j += energy_j;
        } else if energy_kwh < 0.0 {
            cooling_total_j += -energy_j;
        }

        let hour = step % 24;
        let day_of_year = step / 24 + 1;

        // Zone air temperature from model state
        #[allow(non_snake_case)]
        let t_zone_C = model
            .setpoints
            .temperatures
            .as_slice()
            .first()
            .copied()
            .unwrap_or(20.0);
        #[allow(non_snake_case)]
        let t_air_C = model
            .conduction
            .backend
            .gauge_zone_solver
            .as_ref()
            .map(|g| g.T_air().to_value())
            .unwrap_or(t_zone_C);

        // Gauge per-surface telemetry
        let (window_solar_fraction, window_R_total, opaque_R_total) =
            if let Some(gauge) = model.conduction.backend.gauge_zone_solver.as_ref() {
                let tel = gauge.per_surface_telemetry();
                let window_tel = tel
                    .iter()
                    .find(|s| matches!(s.surface_type, SurfaceType::Window));
                let opaque_tels: Vec<_> = tel
                    .iter()
                    .filter(|s| {
                        !matches!(s.surface_type, SurfaceType::Window)
                            && !matches!(s.surface_type, SurfaceType::InterZone)
                    })
                    .collect();

                let ws_frac = window_tel.map(|w| w.solar_fraction).unwrap_or(0.0);
                let w_R = window_tel.map(|w| w.r_total_m2K_W).unwrap_or(0.0);
                let o_R = if opaque_tels.is_empty() {
                    0.0
                } else {
                    opaque_tels.iter().map(|o| o.r_total_m2K_W).sum::<f64>()
                        / opaque_tels.len() as f64
                };
                (ws_frac, w_R, o_R)
            } else {
                (0.0, 0.0, 0.0)
            };

        // Effective "direct to air" ratio for gauge (Hypothesis 1)
        // Gauge routes 100% of window solar through the window path.
        // 5R1C routes solar_distribution_to_air = 0.30 directly to air.
        // The gauge has no equivalent "direct to air" path for window solar.
        // Instead, all window solar enters via q_flux × area = (T_ext − T_int)/R_window.
        // Effective ratio = 0.0 (gauge has no direct-to-air solar routing).
        let gauge_solar_to_air_ratio = 0.0_f64;

        // Air-node damping metrics (Hypothesis 3)
        let (dt_over_tau, tau_air_s, gauge_equil, five_r1c_equil) =
            if let Some(gauge) = model.conduction.backend.gauge_zone_solver.as_ref() {
                let dt_s = 3600.0_f64;
                let tau = gauge.effective_time_constant_s(infiltration_ach);
                let dtt = gauge.dt_over_tau(dt_s, infiltration_ach);

                // Gauge: implicit Euler with N=3 sub-steps
                // After N sub-steps: T_N = T_0 × (1/(1+dt_sub/τ))^N + steady-state terms
                // where dt_sub = dt/N = 3600/3 = 1200 s
                let dt_sub = dt_s / 3.0;
                let sub_steps = 3_usize;
                let denom = 1.0 + dt_sub / tau;
                let gauge_equil_per_step = 1.0 / denom;
                let gauge_equil_after_N =
                    1.0 - gauge_equil_per_step.powi(sub_steps as i32);

                // 5R1C: implicit Euler with dt=3600 s (single step)
                let five_r1c_equil_per_step = 1.0 / (1.0 + dtt);

                (dtt, tau, gauge_equil_after_N, five_r1c_equil_per_step)
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };

        results.push(TimestepDiag {
            step,
            hour,
            day_of_year,
            outdoor_temp_C: outdoor_temp,
            solar_horizontal_Wm2: solar_horizontal,
            t_zone_C,
            t_air_C,
            energy_kwh,
            window_solar_fraction,
            gauge_solar_to_air_ratio,
            window_R_total,
            opaque_R_total,
            dt_over_tau,
            tau_air_s,
            gauge_equilibration_per_hour: gauge_equil,
            five_r1c_equilibration_per_hour: five_r1c_equil,
        });
    }

    // Print summary
    let annual_heating_mwh = heating_total_j * J_TO_MWH;
    let annual_cooling_mwh = cooling_total_j * J_TO_MWH;
    eprintln!("\n=== LIMIT-21 Phase 6 — Case 640 Diagnostic Summary ===");
    eprintln!("Gauge available: {gauge_available}");
    eprintln!(
        "Annual Heating: {:.3} MWh  (ref: {:.2}–{:.2} MWh)",
        annual_heating_mwh, CASE640_HEATING_MIN_MWH, CASE640_HEATING_MAX_MWH
    );
    eprintln!(
        "Annual Cooling: {:.3} MWh  (ref: {:.2}–{:.2} MWh) ← {:.0}%% of lower bound",
        annual_cooling_mwh,
        CASE640_COOLING_MIN_MWH,
        CASE640_COOLING_MAX_MWH,
        100.0 * annual_cooling_mwh / CASE640_COOLING_MIN_MWH
    );

    results
}

#[test]
#[cfg(feature = "gauge-solver")]
#[allow(non_snake_case)] // local vars intentionally match CSV column naming
fn test_limit_21_phase6_case640_diagnostic() {
    // Only run with --features gauge-solver
    // The test instruments the gauge model's internal state to diagnose the
    // Case 640 annual cooling gap.
    let results = run_case640_diagnostic();

    // -------------------------------------------------------------------------
    // Hypothesis 1: Per-surface distribution routing
    // -------------------------------------------------------------------------
    let window_solar_fraction: f64 = results
        .iter()
        .filter(|r| r.window_solar_fraction > 0.0)
        .map(|r| r.window_solar_fraction)
        .sum::<f64>()
        / results.iter().filter(|r| r.window_solar_fraction > 0.0).count().max(1) as f64;

    // Gauge direct-to-air ratio = 0 (no equivalent to 5R1C's solar_distribution_to_air)
    let gauge_solar_to_air_ratio = 0.0_f64;
    let five_r1c_solar_to_air_ratio = 0.30_f64; // Case 640 / ISO 13790 default

    eprintln!(
        "\n--- Hypothesis 1: Per-surface Distribution Routing ---"
    );
    eprintln!(
        "Gauge window solar_fraction (avg): {:.4}",
        window_solar_fraction
    );
    eprintln!(
        "Gauge effective solar→air routing ratio: {:.2}  (5R1C: {:.2})",
        gauge_solar_to_air_ratio, five_r1c_solar_to_air_ratio
    );
    eprintln!(
        "5R1C routes {:.0}%% of window solar DIRECTLY to zone air.",
        100.0 * five_r1c_solar_to_air_ratio
    );
    eprintln!(
        "Gauge routes 100% through window conduction (R_total ≈ {:.3} m²K/W).",
        results.iter().find(|r| r.window_R_total > 0.0).map(|r| r.window_R_total).unwrap_or(0.0)
    );
    eprintln!(
        "FINDING: Gauge has NO direct-to-air solar path. Window solar must"
    );
    eprintln!(
        "  conduct through the window assembly before reaching zone air."
    );
    eprintln!(
        "  The 5R1C's {:.0}% direct-to-air routing is absent from the gauge.",
        100.0 * five_r1c_solar_to_air_ratio
    );

    // -------------------------------------------------------------------------
    // Hypothesis 2: Per-surface 5R1C conductance re-derivation
    // -------------------------------------------------------------------------
    let avg_window_R: f64 = results
        .iter()
        .filter(|r| r.window_R_total > 0.0)
        .map(|r| r.window_R_total)
        .sum::<f64>()
        / results.iter().filter(|r| r.window_R_total > 0.0).count().max(1) as f64;

    let avg_opaque_R: f64 = results
        .iter()
        .filter(|r| r.opaque_R_total > 0.0)
        .map(|r| r.opaque_R_total)
        .sum::<f64>()
        / results.iter().filter(|r| r.opaque_R_total > 0.0).count().max(1) as f64;

    // ASHRAE 140 / ISO 13790 hand-calculation for Case 640:
    // Low-mass: R_wall ≈ 0.32 m²K/W, h_tr_is ≈ 8.3 W/m²K → R_tr_is = 0.12 m²K/W
    let h_tr_is_expected = 8.3_f64; // W/m²K
    let R_tr_is_expected = 1.0 / h_tr_is_expected; // ≈ 0.12 m²K/W

    eprintln!("\n--- Hypothesis 2: Per-surface 5R1C Conductance Re-derivation ---");
    eprintln!(
        "Gauge avg window R_total: {:.4} m²K/W  (expected R_tr_is ≈ {:.4})",
        avg_window_R, R_tr_is_expected
    );
    eprintln!(
        "Gauge avg opaque R_total: {:.4} m²K/W",
        avg_opaque_R
    );
    if avg_window_R > 0.0 {
        let deviation = (avg_window_R - R_tr_is_expected) / R_tr_is_expected * 100.0;
        eprintln!(
            "Deviation from expected h_tr_is: {:.1}%  (B1a audit found −86.8%)",
            deviation
        );
        if deviation.abs() > 50.0 {
            eprintln!("FINDING: Large conductance deviation suggests hypothesis 2 contributes.");
        } else {
            eprintln!("FINDING: Conductance close to expected; hypothesis 2 unlikely.");
        }
    }

    // -------------------------------------------------------------------------
    // Hypothesis 3: Family-level 5R1C lumped-mass-node damping
    // -------------------------------------------------------------------------
    let avg_dt_over_tau: f64 = results
        .iter()
        .filter(|r| r.dt_over_tau > 0.0)
        .map(|r| r.dt_over_tau)
        .sum::<f64>()
        / results.iter().filter(|r| r.dt_over_tau > 0.0).count().max(1) as f64;

    let avg_tau_s: f64 = results
        .iter()
        .filter(|r| r.tau_air_s > 0.0)
        .map(|r| r.tau_air_s)
        .sum::<f64>()
        / results.iter().filter(|r| r.tau_air_s > 0.0).count().max(1) as f64;

    let avg_gauge_equil: f64 = results
        .iter()
        .filter(|r| r.gauge_equilibration_per_hour > 0.0)
        .map(|r| r.gauge_equilibration_per_hour)
        .sum::<f64>()
        / results.iter().filter(|r| r.gauge_equilibration_per_hour > 0.0).count().max(1)
        as f64;

    let avg_5r1c_equil: f64 = results
        .iter()
        .filter(|r| r.five_r1c_equilibration_per_hour > 0.0)
        .map(|r| r.five_r1c_equilibration_per_hour)
        .sum::<f64>()
        / results.iter().filter(|r| r.five_r1c_equilibration_per_hour > 0.0).count().max(1)
        as f64;

    eprintln!("\n--- Hypothesis 3: Family-level Lumped-Mass-Node Damping ---");
    eprintln!(
        "Gauge avg dt/τ_air: {:.2}  (5R1C dt/τ ≈ 3.6 at τ ≈ 1000 s)",
        avg_dt_over_tau
    );
    eprintln!(
        "Gauge avg τ_air: {:.0} s  (ASHRAE 140 low-mass: 800–1200 s expected)",
        avg_tau_s
    );
    eprintln!(
        "Gauge equilibration per hour (N=3 sub-steps): {:.1}%%",
        avg_gauge_equil * 100.0
    );
    eprintln!(
        "5R1C equilibration per hour (implicit Euler, dt=3600 s): {:.1}%%",
        avg_5r1c_equil * 100.0
    );

    // Correct 5R1C equilibration at its own τ ≈ 1000s (not the gauge's τ):
    // 1 - 1/(1 + 3600/1000) = 1 - 1/4.6 ≈ 78.3%
    let five_r1c_equil_correct = 1.0 - 1.0 / (1.0 + 3600.0 / 1000.0);
    eprintln!(
        "5R1C at proper τ≈1000s: {:.1}%% (vs gauge at {:.0}s: {:.1}%%)",
        five_r1c_equil_correct * 100.0,
        avg_tau_s,
        avg_gauge_equil * 100.0
    );

    // Use the corrected 5R1C equilibration for the delta
    let damp_delta = avg_gauge_equil - five_r1c_equil_correct;
    eprintln!(
        "Gauge {} by {:.1}%% relative to 5R1C  (positive = gauge equilibrates more)",
        if damp_delta > 0.0 { "over-damps" } else { "under-damps" },
        damp_delta.abs() * 100.0
    );
    if damp_delta > 0.05 {
        eprintln!("FINDING: Gauge over-damps air node vs 5R1C. Sub-stepping (N=3)");
        eprintln!("  drives faster equilibration than a single 5R1C implicit-Euler step,");
        eprintln!("  contributing to the cooling deficit (air cools too quickly overnight).");
    } else if damp_delta < -0.05 {
        eprintln!("FINDING: Gauge under-damps air node vs 5R1C. Solar transients");
        eprintln!("  drive cooling demand more slowly in the gauge model.");
    } else {
        eprintln!("FINDING: Gauge and 5R1C damping are similar; hypothesis 3 unlikely.");
    }

    // -------------------------------------------------------------------------
    // Write CSV
    // -------------------------------------------------------------------------
    let dir: PathBuf = ["target", "diag"].iter().collect();
    fs::create_dir_all(&dir).ok();
    let csv_path = dir.join("limit_21_phase6_case640_diagnostic.csv");
    let mut csv = File::create(&csv_path).unwrap();
    writeln!(csv, "{}", TimestepDiag::header_csv()).unwrap();
    for r in &results {
        writeln!(csv, "{}", r.to_csv()).unwrap();
    }

    eprintln!("\n--- Output ---");
    eprintln!("CSV written to: {}", csv_path.display());

    // -------------------------------------------------------------------------
    // Monthly aggregation for pattern analysis
    // -------------------------------------------------------------------------
    eprintln!("\n--- Monthly Cooling Breakdown (Gauge vs Reference) ---");
    for month in 0..12 {
        let start_step = month * 730; // approximate
        let end_step = (month + 1) * 730;
        let month_cooling_j: f64 = results
            .iter()
            .filter(|r| r.step >= start_step && r.step < end_step && r.energy_kwh < 0.0)
            .map(|r| -r.energy_kwh * 3.6e6)
            .sum();
        let month_cooling_mwh = month_cooling_j * J_TO_MWH;
        // ASHRAE 140 reference: ~0.5–1.5 MWh/month for summer months
        eprintln!(
            "  Month {:2}: {:.3} MWh  (ref band ≈ {:.1}–{:.1} MWh)",
            month + 1,
            month_cooling_mwh,
            0.3,
            1.5
        );
    }
}
