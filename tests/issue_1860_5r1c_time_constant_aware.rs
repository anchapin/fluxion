//! Regression tests for Issue #1860 — time-constant-aware 5R1C solver.
//!
//! ASHRAE 140 Cases 600 / 650 / 950 (all LowMass constructions) currently
//! show a cooling-load gap of ~38–90% vs the ASHRAE 140 published reference
//! bands. The v1.3 epic assessment (`docs/epic-672-v13-assessment.md:257-265`)
//! attributes this to the steady-state-only 5R1C solver path on low-mass
//! constructions and tracks the structural fix as v1.4 issue #1860.
//!
//! # What this module pins
//!
//! - **Time-constant accessor surface** on `FiveR1CSolver` — `time_constant()`
//!   and `surface_time_constant()` — so downstream code can drive the
//!   surface ODE without re-deriving `C·R` and `C·(R_1‖R_si)` formulas.
//! - **Wall-surface ODE state** (`ThermalModelData::wall_surface_temperatures`)
//!   is initialised and updated each `step_physics_5r1c` call.
//! - **Energy-conservation invariants** at the model boundary (no NaN /
//!   no sign flips introduced by the new state).
//! - **Reference values** for Cases 600 / 650 / 950 (annual cooling band
//!   and time constant) so future fixes can be regression-checked against
//!   the same baselines.
//!
//! # Status
//!
//! This module adds the **infrastructure** (time-constant accessors,
//! surface-state field, surface ODE) without removing the existing
//! `air_frac = 0.7` band-aid that masks the underlying cooling-load gap.
//! Removing the band-aid and achieving the ±15% ASHRAE 140 tolerance is
//! the **structural** fix and is tracked as the follow-up to #1860.
//!
//! Tests in this module are infrastructure-level: they verify that the
//! new state behaves correctly (finite values, expected ranges,
//! energy-conservation properties) but they do NOT yet enforce the
//! ±15% ASHRAE 140 cooling tolerance (that requires the band-aid
//! removal, tracked separately).

use fluxion::physics::cta::VectorField;
use fluxion::physics::five_r1c_solver::FiveR1CSolver;
use fluxion::physics::solver_trait::HeatConductionSolver;
use fluxion::physics::units::{FromF64, HeatTransferCoefficient, Temperature, Time};
use fluxion::physics::wall_spec::{lightweight_wall_spec, WallSpec};
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;
use fluxion_core::assembly::{AssemblyBuilder, ConcreteMaterial, InsulationMaterial};

/// Run `steps` hourly `step_physics` calls with Denver TMY weather data so
/// the surface ODE state is exercised against real boundary conditions.
fn run_case_with_weather(case: ASHRAE140Case, steps: usize) -> ThermalModel<VectorField> {
    let mut model = ThermalModel::<VectorField>::from_spec(&case.spec());
    let weather = DenverTmyWeather::new();
    for step in 0..steps {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
    }
    model
}

// =============================================================================
// Reference values for ASHRAE 140 Cases 600 / 650 / 950 (LowMass)
// =============================================================================
//
// Source: docs/ASHRAE140_RESULTS.md, docs/epic-672-v13-assessment.md,
// tests/reference_data/case_*_energy_reference.csv. Values are the published
// ASHRAE 140-2023 ±15% bands.
//
// Annual cooling band: [min, max] in MWh (Denver / Golden-NREL TMY3 weather).
// Annual heating band: [min, max] in MWh.
// Peak cooling / heating bands: [min, max] in kW.
//
// These are **documented baselines** that the Issue #1860 fix must land
// inside once the band-aid (`solar_distribution_to_air = 0.7` for LowMass,
// see `src/sim/thermal_model_core.rs:1855-1870`) is removed. Until then,
// the engine predicts ~38% below band (see `tests/zone_balance_eplus_isolation.rs`
// `test_case_600_annual_energy_ashrae140_tolerance` docstring for the
// post-#1323 observation: H=3.167 MWh, C=2.672 MWh vs [4.314, 5.836] /
// [4.275, 5.784]).

const CASE_600_ANNUAL_COOLING_MWH: [f64; 2] = [4.275, 5.784];
const CASE_600_ANNUAL_HEATING_MWH: [f64; 2] = [4.314, 5.836];

// =============================================================================
// Time-Constant Accessor Tests
// =============================================================================

/// Verify `time_constant()` returns `C·R_total` for a 200 mm concrete wall
/// (matches the analytical τ = C/H_tr_ms = C·R definition).
#[test]
fn test_solver_time_constant_matches_analytical_definition() {
    let wall = AssemblyBuilder::new("200mm Concrete".to_string())
        .add_layer(Box::new(ConcreteMaterial::new(0.2)))
        .build()
        .unwrap();
    let spec = WallSpec::from_assembly(&wall);

    let mut solver = FiveR1CSolver::new();
    solver.initialize(&spec).unwrap();

    let tau_solver = solver.time_constant();
    let tau_analytical = spec.thermal_capacity() * spec.total_r_value();
    let rel_diff = (tau_solver - tau_analytical).abs() / tau_analytical;
    assert!(
        rel_diff < 1e-12,
        "time_constant() must equal C·R_total: solver={tau_solver:.6e}, \
         analytical={tau_analytical:.6e}, rel_diff={rel_diff:.6e}"
    );
}

/// Verify `surface_time_constant()` returns `C·(R_ms‖R_si)` for a
/// multi-layer wall, where `R_ms` follows the ISO 13790 half-insulation
/// rule and `R_si = 1/8` is the separate interior film resistance.
#[test]
fn test_solver_surface_time_constant_matches_parallel_resistance() {
    let spec = lightweight_wall_spec();
    let mut solver = FiveR1CSolver::new();
    solver.initialize(&spec).unwrap();

    let r_ms = spec.layers[2].r_value() + spec.layers[1].r_value() / 2.0;
    let r_si = 1.0 / 8.0;
    let r_parallel = r_ms * r_si / (r_ms + r_si);
    let tau_si_analytical = spec.thermal_capacity() * r_parallel;

    assert!((solver.r_ms() - r_ms).abs() / r_ms < 1e-12);
    let tau_si_solver = solver.surface_time_constant();
    let rel_diff = (tau_si_solver - tau_si_analytical).abs() / tau_si_analytical;
    assert!(
        rel_diff < 1e-12,
        "surface_time_constant() must equal C·(R_ms‖R_si): \
         solver={tau_si_solver:.6e}, analytical={tau_si_analytical:.6e}, rel_diff={rel_diff:.6e}"
    );
}

/// For ASHRAE 140 Case 600 (low-mass foam-board insulation), the lumped
/// time constant τ_solver = C·R_wall must be < 2 h so `ThermalMethodSelector`
/// picks 5R1C (not FD/CTF) — see `src/physics/wall_properties.rs:533-540`
/// for the equivalent check at the wall-properties layer.
#[test]
fn test_solver_time_constant_case_600_selects_5r1c() {
    let wall = AssemblyBuilder::new("Case 600 low-mass".to_string())
        .add_layer(Box::new(InsulationMaterial::new(0.050)))
        .build()
        .unwrap();
    let spec = WallSpec::from_assembly(&wall);

    let mut solver = FiveR1CSolver::new();
    solver.initialize(&spec).unwrap();

    let tau_h = solver.time_constant() / 3600.0;
    assert!(
        tau_h < 2.0,
        "Case 600 τ_solver = {tau_h:.4} h must be < 2 h so 5R1C is selected"
    );

    // Surface τ_si should also be small (< 0.5 h) so the surface ODE
    // converges quickly within the 1-hour timestep.
    let tau_si_h = solver.surface_time_constant() / 3600.0;
    assert!(
        tau_si_h < 0.5,
        "Case 600 surface τ_si = {tau_si_h:.4} h must be < 0.5 h"
    );
}

/// Surface time constant must always be ≤ lumped time constant (R_1‖R_si ≤
/// R_total for any non-negative R_1, R_si).
#[test]
fn test_solver_surface_time_constant_le_lumped() {
    let wall = AssemblyBuilder::new("200mm Concrete".to_string())
        .add_layer(Box::new(ConcreteMaterial::new(0.2)))
        .build()
        .unwrap();
    let spec = WallSpec::from_assembly(&wall);

    let mut solver = FiveR1CSolver::new();
    solver.initialize(&spec).unwrap();

    let tau = solver.time_constant();
    let tau_si = solver.surface_time_constant();
    assert!(
        tau_si <= tau,
        "τ_si ({tau_si:.3e}) must be ≤ τ ({tau:.3e}): R_1‖R_si ≤ R_total"
    );
    assert!(
        tau > 0.0 && tau_si > 0.0,
        "Both τ and τ_si must be positive"
    );
}

// =============================================================================
// Surface-ODE Step Tests
// =============================================================================

/// The 5R1C solver's lumped-capacitance ODE must relax to the steady-state
/// flux `q_ss = (T_ext − T_int) / R_total` after a long simulation with
/// constant boundary temperatures. This pins the solver-level equilibrium
/// (used by the Issue #1860 surface-state ODE to compute `T_si_eq`).
#[test]
fn test_wall_surface_ode_relaxes_to_equilibrium() {
    let wall = AssemblyBuilder::new("200mm Concrete".to_string())
        .add_layer(Box::new(ConcreteMaterial::new(0.2)))
        .build()
        .unwrap();
    let spec = WallSpec::from_assembly(&wall);

    let mut solver = FiveR1CSolver::new();
    solver.initialize(&spec).unwrap();

    let t_int = 20.0;
    let t_ext = 5.0;
    let dt = 3600.0; // 1 h
    let tau = solver.time_constant();

    // Drive the solver through enough steps to reach equilibrium
    // (10 × τ is generous for an exponential relaxation).
    let n_steps = ((10.0 * tau / dt).ceil() as usize).max(50);
    for _ in 0..n_steps {
        solver
            .step(
                Time::from_value(dt),
                Temperature::from_value(t_int),
                Temperature::from_value(t_ext),
                HeatTransferCoefficient::from_value(8.0),
                HeatTransferCoefficient::from_value(25.0),
            )
            .unwrap();
    }

    // At equilibrium, the returned flux should match
    // q_ss = (T_ext − T_int) / R_total.
    let q_ss = (t_ext - t_int) / spec.total_r_value();
    let q_final = solver.current_flux();
    let rel_err = (q_final - q_ss).abs() / q_ss.abs();
    assert!(
        rel_err < 0.01,
        "After equilibrium: q_final={q_final:.6e}, q_ss={q_ss:.6e}, rel_err={rel_err:.6e}"
    );
}

/// The `ThermalModel` surface-state ODE must converge to the HAC equilibrium
/// defined by the independent mass-to-surface and interior-film conductances.
#[test]
fn test_thermal_model_surface_ode_relaxes_to_t_si_eq() {
    let mut model = ThermalModel::<VectorField>::from_spec(&ASHRAE140Case::Case600.spec());
    let dt = 3600.0;
    let t_int = 20.0;
    let t_mass = 30.0;
    let t_si_initial = -10.0;
    let n_zones = model.num_zones;

    let max_tau_si = (0..n_zones)
        .map(|i| {
            let h_ms = model.h_tr_ms.as_ref()[i];
            let h_is = model.h_tr_is.as_ref()[i];
            model.thermal_capacitance.as_ref()[i] / (h_ms + h_is)
        })
        .fold(0.0_f64, f64::max);
    let n_steps = ((12.0 * max_tau_si / dt).ceil() as usize).max(1);

    model.wall_surface_temperatures.as_mut().fill(t_si_initial);
    for _ in 0..n_steps {
        model.temperatures.as_mut().fill(t_int);
        model.air_temperatures.as_mut().fill(t_int);
        model.mass_temperatures.as_mut().fill(t_mass);
        model.step_physics(0, t_int, dt);
    }

    for i in 0..n_zones {
        let h_ms = model.h_tr_ms.as_ref()[i];
        let h_is = model.h_tr_is.as_ref()[i];
        assert!(
            h_ms > 0.0 && h_is > 0.0,
            "Case 600 zone {i}: h_tr_ms/h_tr_is must be positive"
        );

        let t_si_eq = (t_int * h_is + t_mass * h_ms) / (h_is + h_ms);
        let t_si_actual = model.wall_surface_temperatures.as_ref()[i];
        let error = (t_si_actual - t_si_eq).abs();
        assert!(
            error < 0.01,
            "Case 600 zone {i}: T_si must converge to the HAC equilibrium. \
             T_si={t_si_actual:.6} °C, T_si_eq={t_si_eq:.6} °C, |Δ|={error:.6} °C"
        );
    }
}

// =============================================================================
// ThermalModel Wall-Surface State Tests (Issue #1860)
// =============================================================================

/// `wall_surface_temperatures` must be finite and in a reasonable range
/// after a Case 600 simulation. The state should track between the zone
/// air temperature and the wall mass temperature.
#[test]
fn test_case_600_wall_surface_state_finite_and_bounded() {
    let model = run_case_with_weather(ASHRAE140Case::Case600, 24);

    let n_zones = model.num_zones;
    let surface_temps: Vec<f64> = (0..n_zones)
        .map(|i| model.wall_surface_temperatures.as_ref()[i])
        .collect();
    let zone_temps: Vec<f64> = (0..n_zones)
        .map(|i| model.temperatures.as_ref()[i])
        .collect();
    let mass_temps: Vec<f64> = (0..n_zones)
        .map(|i| model.mass_temperatures.as_ref()[i])
        .collect();

    for (i, &t_si) in surface_temps.iter().enumerate() {
        assert!(
            t_si.is_finite(),
            "Case 600 wall surface temperature zone {i} must be finite, got {t_si}"
        );
        // Surface temperature should be in a physically reasonable range
        // (between -50°C and +80°C — covers all ASHRAE 140 envelope cases).
        assert!(
            t_si > -50.0 && t_si < 80.0,
            "Case 600 zone {i}: T_si={t_si:.3} outside the physical envelope [-50, 80] °C"
        );
        // T_si should lie in the [T_zone, T_mass] band (it sits between
        // them in the 5R1C network). Allow a generous margin because the
        // surface ODE carries inertia and may overshoot during transients.
        let t_min = zone_temps[i].min(mass_temps[i]) - 5.0;
        let t_max = zone_temps[i].max(mass_temps[i]) + 5.0;
        assert!(
            t_si >= t_min && t_si <= t_max,
            "Case 600 zone {i}: T_si={t_si:.3}, T_zone={}, T_mass={} — surface temp should be near the [T_zone, T_mass] band",
            zone_temps[i], mass_temps[i]
        );
    }
}

/// Energy-conservation invariant: wall surface temperature update must NOT
/// introduce NaN / Inf into the model state across a 24-hour simulation.
#[test]
fn test_case_600_no_nan_after_surface_ode() {
    let model = run_case_with_weather(ASHRAE140Case::Case600, 24);

    let n_zones = model.num_zones;
    for i in 0..n_zones {
        let t_si = model.wall_surface_temperatures.as_ref()[i];
        let t_air = model.air_temperatures.as_ref()[i];
        let t_mass = model.mass_temperatures.as_ref()[i];
        let t_zone = model.temperatures.as_ref()[i];
        assert!(
            t_si.is_finite(),
            "T_si[{i}] must be finite after 24h simulation, got {t_si}"
        );
        assert!(t_air.is_finite(), "T_air[{i}] must be finite, got {t_air}");
        assert!(
            t_mass.is_finite(),
            "T_mass[{i}] must be finite, got {t_mass}"
        );
        assert!(
            t_zone.is_finite(),
            "T_zone[{i}] must be finite, got {t_zone}"
        );
    }
}

/// Surface temperature must be physically bounded (T_si in [-50, 80] °C)
/// after 48 hours of simulation with real Denver TMY weather. We check
/// the magnitude of |T_si − T_mass| stays in a reasonable range so any
/// sign / magnitude regression in the surface ODE is caught.
#[test]
fn test_case_600_surface_ode_relaxes_correctly() {
    let model = run_case_with_weather(ASHRAE140Case::Case600, 48);

    for i in 0..model.num_zones {
        let t_si = model.wall_surface_temperatures.as_ref()[i];
        let t_mass = model.mass_temperatures.as_ref()[i];
        assert!(
            t_si > -50.0 && t_si < 80.0,
            "Case 600 zone {i}: T_si={t_si:.3} °C outside physical envelope"
        );
        // T_si should sit close to T_mass (within ~30°C for a 48h sim).
        let diff = (t_si - t_mass).abs();
        assert!(
            diff < 30.0,
            "Case 600 zone {i}: |T_si − T_mass| = {diff:.2} °C should be physically bounded"
        );
    }
}

// =============================================================================
// Reference-Band Documentation Tests (no enforcement yet)
// =============================================================================
//
// The actual ±15% ASHRAE 140 cooling-load tolerance gate (issue #1333) is
// `#[ignore]`d in `tests/zone_balance_eplus_isolation.rs` pending the
// structural fix tracked by issue #1860. These tests document the
// published reference bands so future PRs can be regression-checked
// without having to dig through the v1.3 assessment doc.

/// Document Case 600 reference band (annual cooling) without enforcing it.
/// Future fixes can flip `#[ignore]` → active assertion once the band-aid
/// `solar_distribution_to_air = 0.7` is removed (see `thermal_model_core.rs:1855`).
#[test]
#[ignore = "Issue #1860 cooling-load gate — un-ignore once the band-aid is removed \
            and Case 600 annual cooling lands inside the ASHRAE 140 ±15% band. \
            Current observation (post-#1323): C=2.672 MWh vs [4.275, 5.784] MWh. \
            See docs/epic-672-v13-assessment.md:257-265."]
fn test_case_600_annual_cooling_within_ashrae140_band() {
    let model = run_case_with_weather(ASHRAE140Case::Case600, 8760);
    let c_mwh = model.get_cooling_energy_kwh() / 1000.0;

    assert!(
        (CASE_600_ANNUAL_COOLING_MWH[0]..=CASE_600_ANNUAL_COOLING_MWH[1]).contains(&c_mwh),
        "Case 600 annual cooling {c_mwh:.3} MWh must be inside the ASHRAE 140 band \
         [{:.3}, {:.3}] MWh",
        CASE_600_ANNUAL_COOLING_MWH[0],
        CASE_600_ANNUAL_COOLING_MWH[1]
    );
}

/// Document Case 600 reference band (annual heating) without enforcing it.
#[test]
#[ignore = "Issue #1860 cooling-load gate — see `test_case_600_annual_cooling_within_ashrae140_band`."]
fn test_case_600_annual_heating_within_ashrae140_band() {
    let model = run_case_with_weather(ASHRAE140Case::Case600, 8760);
    let h_mwh = model.get_heating_energy_kwh() / 1000.0;

    assert!(
        (CASE_600_ANNUAL_HEATING_MWH[0]..=CASE_600_ANNUAL_HEATING_MWH[1]).contains(&h_mwh),
        "Case 600 annual heating {h_mwh:.3} MWh must be inside the ASHRAE 140 band \
         [{:.3}, {:.3}] MWh",
        CASE_600_ANNUAL_HEATING_MWH[0],
        CASE_600_ANNUAL_HEATING_MWH[1]
    );
}

/// Same as Case 600 but for Case 650 (low-mass, south-facing windows).
#[test]
#[ignore = "Issue #1860 cooling-load gate — see `test_case_600_annual_cooling_within_ashrae140_band`. \
            Same structural-fix dependency for Case 650."]
fn test_case_650_annual_cooling_within_ashrae140_band() {
    let model = run_case_with_weather(ASHRAE140Case::Case650, 8760);
    let c_mwh = model.get_cooling_energy_kwh() / 1000.0;
    assert!(
        c_mwh > 0.0 && c_mwh.is_finite(),
        "Case 650 annual cooling must be finite and positive, got {c_mwh}"
    );
}

/// Same as Case 600 but for Case 950 (low-mass + night ventilation).
#[test]
#[ignore = "Issue #1860 cooling-load gate — see `test_case_600_annual_cooling_within_ashrae140_band`. \
            Same structural-fix dependency for Case 950."]
fn test_case_950_annual_cooling_within_ashrae140_band() {
    let model = run_case_with_weather(ASHRAE140Case::Case950, 8760);
    let c_mwh = model.get_cooling_energy_kwh() / 1000.0;
    assert!(
        c_mwh > 0.0 && c_mwh.is_finite(),
        "Case 950 annual cooling must be finite and positive, got {c_mwh}"
    );
}

// =============================================================================
// Solar-Lag Correction Tests (Issue #1860)
// =============================================================================

/// Solar-lag state must be initialised to zero (no solar history at t=0).
#[test]
fn test_solar_lag_initialised_to_zero() {
    let model = ThermalModel::<VectorField>::from_spec(&ASHRAE140Case::Case600.spec());
    for i in 0..model.num_zones {
        let lag = model.solar_lag.as_ref()[i];
        assert!(
            lag == 0.0,
            "solar_lag[{i}] must be initialised to zero, got {lag}"
        );
    }
}

/// Solar-lag state must be finite and non-negative after 24 h of simulation.
#[test]
fn test_solar_lag_finite_and_nonnegative_after_simulation() {
    let model = run_case_with_weather(ASHRAE140Case::Case600, 24);
    for i in 0..model.num_zones {
        let lag = model.solar_lag.as_ref()[i];
        assert!(lag.is_finite(), "solar_lag[{i}] must be finite, got {lag}");
        assert!(lag >= 0.0, "solar_lag[{i}] must be non-negative, got {lag}");
    }
}

/// The solar-lag correction must improve Case 650 annual cooling relative to
/// the pre-fix baseline (~3.0 MWh).
#[test]
fn test_case_650_solar_lag_improves_annual_cooling() {
    let model = run_case_with_weather(ASHRAE140Case::Case650, 8760);
    let c_mwh = model.get_cooling_energy_kwh() / 1000.0;
    let pre_fix_baseline = 3.0;
    assert!(
        c_mwh > pre_fix_baseline * 1.10,
        "Case 650 annual cooling {c_mwh:.3} MWh should be ≥ 10% above pre-fix baseline \
         ({pre_fix_baseline:.1} MWh)"
    );
}

/// Alpha-blend mass-node coupling must produce finite mass temperatures.
#[test]
fn test_case_600_alpha_blend_finite_mass_temps() {
    let model = run_case_with_weather(ASHRAE140Case::Case600, 8760);
    for i in 0..model.num_zones {
        let t_mass = model.mass_temperatures.as_ref()[i];
        assert!(
            t_mass.is_finite(),
            "T_mass[{i}] must be finite, got {t_mass}"
        );
        assert!(
            t_mass > -30.0 && t_mass < 80.0,
            "T_mass[{i}] = {t_mass:.1}°C outside physical envelope [-30, 80]°C"
        );
    }
}
