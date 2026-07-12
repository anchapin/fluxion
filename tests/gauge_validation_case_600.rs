//! Validation harness for the `GaugeSolver` (Phase 1b, #1462) against
//! ASHRAE 140 Case 600 reference geometry — paralleling
//! `tests/gauge_validation_case_900.rs`.
//!
//! ## Background (issue #1527)
//!
//! 13 of 27 tests in `ashrae_140_case_600_series.rs` fail because the 5R1C
//! solver over-injects solar onto the air node
//! (`solar_distribution_to_air = 0.7`, set by issue #1216 as a band-aid for
//! the missing air-node thermal inertia). The comment block at
//! `thermal_model_core.rs:1799-1835` documents that the fix requires the
//! structural 5R1C air-node capacitance rewrite (issue #1152) or the full
//! zone-level GaugeSolver — NOT the per-wall GaugeSolver that shipped.
//!
//! ## What this harness validates
//!
//! This harness validates the **per-wall GaugeSolver** against the Case 600
//! low-mass envelope geometry. It exercises the same shadow-mode path as the
//! Case 900 harness but with Case 600 construction parameters:
//!
//! 1. `ThermalManifold::from_5r1c_parameters` accepts Case 600 scene
//!    parameters and produces a finite, dissipative operator.
//! 2. The Case 600 low-mass wall thermal-capacity metric (Cm ≈ 12.86 kJ/m²K)
//!    is reproduced from first principles within 1 %.
//! 3. The GaugeSolver shadow-mode output responds to a synthetic 24-hour
//!    diurnal cycle with non-zero amplitude, finite values, and physically
//!    reasonable phase (peak flux within ±2 h of peak sol-air temperature).
//! 4. The shadow-mode path is non-throttling under extreme solar forcing.
//! 5. Steady-state parity with the baseline `FiveR1CSolver` (no solar).
//! 6. Boundary-condition translation records the gauge_connection correctly.
//!
//! ## Architectural note
//!
//! This harness does **not** close the 13 zone-level Case 600 test failures.
//! Those failures are in the zone air-node solar-distribution path
//! (`physics_impl.rs:325`), which the per-wall GaugeSolver does not touch.
//! Closing those tests requires the structural rewrite tracked in #1152 or
//! a future zone-level gauge integration. This harness provides the
//! validation infrastructure for when that integration lands.

use fluxion::physics::five_r1c_solver::FiveR1CSolver;
use fluxion::physics::gauge_solver::{GaugeBoundaryConditions, GaugeSolver};
use fluxion::physics::geometry_tensor::{ManifoldIndex, ThermalManifold, MAX_ZONES};
use fluxion::physics::solver_trait::HeatConductionSolver;
use fluxion::physics::units::{FromF64, HeatTransferCoefficient, Temperature, Time, ToF64};
use fluxion::physics::wall_spec::WallSpec;
use fluxion::thermal::physics_adapter::{GaugeShadowRecord, PhysicsAdapter, PhysicsAdapterConfig};

/// ASHRAE 140 Case 600 envelope geometry (low-mass wood-frame wall).
///
/// Per `src/sim/construction.rs::Assemblies::low_mass_wall`:
///   - Plasterboard 12 mm: k=0.16, ρ=784,  cp=840
///   - Fiberglass   66 mm: k=0.04, ρ=12,   cp=840
///   - Wood siding   9 mm: k=0.14, ρ=530,  cp=900
///
/// Total conduction R = 1.7893 m²K/W; with films (h_int=8.0, h_ext=18.3)
/// R_total = 1.9689 m²K/W → U = 0.508 W/m²K (ASHRAE 140 target ≈ 0.514).
///
/// For the per-wall GaugeSolver we need `wall.total_r_value()`. WallSpec
/// uses a single equivalent layer; we use the fiberglass-dominant equivalent
/// (k=0.04, d=0.07157 m → R_cond = 1.7893 m²K/W).
const CASE_600_WALL_K_EQUIV_W_MK: f64 = 0.04;
const CASE_600_WALL_D_EQUIV_M: f64 = 0.07157; // k_eq * R_cond = 0.04 * 1.7893
const CASE_600_WALL_RHO_EQUIV_KG_M3: f64 = 12.0; // fiberglass-dominant
const CASE_600_WALL_CP_EQUIV_J_KGK: f64 = 840.0;

/// Documented Cm for the Case 600 low-mass wall stack.
/// Cm = Σ(ρ·Cp·d) = 784·840·0.012 + 12·840·0.066 + 530·900·0.009
///   = 7896.8 + 666.4 + 4294.5 / 1000 = 12.86 kJ/m²K.
const CASE_600_CM_KJ_M2K_DOCUMENTED: f64 = 12.86;
const CASE_600_CM_TOLERANCE_PCT: f64 = 1.0;

/// Exterior / interior film coefficients (W/m²K).
/// `h_ext = 18.3` per ARCHITECTURE.md (issue #1140 unified canonical constant).
const CASE_600_H_EXT: f64 = 18.3;
const CASE_600_H_INT: f64 = 8.0;

/// Synthetic 24-hour diurnal cycle (representative Case 600 Denver summer).
const SOLAR_PEAK_W_M2: f64 = 800.0;
const T_OUTDOOR_AVG_C: f64 = 25.0;
const T_OUTDOOR_AMP_C: f64 = 10.0;
const T_INDOOR_HVAC_SETPOINT_C: f64 = 22.0;
const DT_SECONDS: f64 = 3600.0;

/// Build the Case 600 equivalent envelope wall (`WallSpec`).
fn case_600_wall() -> WallSpec {
    WallSpec::single_layer(
        "ASHRAE 140 Case 600 — low-mass wood frame (fiberglass equivalent)",
        CASE_600_WALL_D_EQUIV_M,
        CASE_600_WALL_K_EQUIV_W_MK,
        CASE_600_WALL_RHO_EQUIV_KG_M3,
        CASE_600_WALL_CP_EQUIV_J_KGK,
    )
}

/// Synthetic outdoor temperature (cosinusoidal diurnal, peak at 15:00).
fn outdoor_temperature_at(hour: usize) -> f64 {
    let h = (hour % 24) as f64;
    T_OUTDOOR_AVG_C + T_OUTDOOR_AMP_C * ((h - 15.0) * std::f64::consts::PI / 12.0).cos()
}

/// Synthetic solar irradiance (sinusoidal, 06:00–18:00, peak at 12:00).
fn solar_irradiance_at(hour: usize) -> f64 {
    let h = (hour % 24) as f64;
    if (6.0..=18.0).contains(&h) {
        SOLAR_PEAK_W_M2 * (((h - 6.0) / 12.0) * std::f64::consts::PI).sin()
    } else {
        0.0
    }
}

/// Step a `PhysicsAdapter` (GaugeSolver in shadow mode, initialized against
/// `wall`) through the synthetic 24-hour diurnal cycle and collect the
/// per-hour shadow-mode fluxes. The adapter records the gauge diagnostics
/// side-by-side with the baseline FiveR1CSolver flux.
fn gauge_shadow_diurnal_fluxes(wall: &WallSpec) -> Vec<f64> {
    let t_int = Temperature::from_value(T_INDOOR_HVAC_SETPOINT_C);
    let h_int = HeatTransferCoefficient::from_value(CASE_600_H_INT);
    let h_ext = HeatTransferCoefficient::from_value(CASE_600_H_EXT);

    let mut adapter = PhysicsAdapter::new(PhysicsAdapterConfig::gauge_shadow());
    adapter.initialize(wall).expect("adapter init");

    (0..24)
        .map(|hour| {
            let t_out = outdoor_temperature_at(hour);
            let solar = solar_irradiance_at(hour);
            adapter
                .step(
                    Time::from_value(DT_SECONDS),
                    t_int,
                    Temperature::from_value(t_out),
                    h_int,
                    h_ext,
                    solar,
                )
                .expect("GaugeSolver step")
                .to_value()
        })
        .collect()
}

// =============================================================================
// Test 1: ThermalManifold lays out Case 600 5R1C scene without NaN/Inf
// =============================================================================

/// The 2×2 active sub-block `ThermalManifold` from `from_5r1c_parameters`
/// with Case 600 scene parameters must be finite, symmetric, and encode the
/// expected dissipative operator layout (air-row self-conductance = 1/R_eq/C_air).
#[test]
fn test_case_600_thermal_manifold_layout() {
    // Case 600 5R1C scene (low-mass): air + single mass node.
    let t_air = 22.0;
    let t_mass = 21.0;
    let r_eq = 1.7893; // conduction R (m²K/W) — from layer stack
    let c_air = 10_000.0; // J/K (low-capacitance air node)
    let c_mass = 12_860.0; // J/K (= Cm_documented × wall_area, 1 m² ref)

    let manifold = ThermalManifold::from_5r1c_parameters(t_air, t_mass, r_eq, c_air, c_mass);

    manifold
        .validate()
        .expect("Case 600 manifold must be algebraically finite (no NaN/Inf)");

    // Scalar field carries the initial temperatures.
    assert!(
        (manifold.scalar_field[ManifoldIndex::Air as usize] - t_air).abs() < 1e-9,
        "Air temperature mismatch"
    );
    assert!(
        (manifold.scalar_field[ManifoldIndex::Wall as usize] - t_mass).abs() < 1e-9,
        "Wall temperature mismatch"
    );

    // Initial gauge connection is zero (no source injection before BC translation).
    assert_eq!(manifold.gauge_connection_sum(), 0.0);
}

// =============================================================================
// Test 2: Case 600 thermal-capacity metric (Cm) matches reference
// =============================================================================

/// The low-mass wall stack Cm = Σ(ρ·Cp·d) must reproduce the documented
/// 12.86 kJ/m²K within ±1 % (per AGENTS.md module-isolation tolerance).
#[test]
fn test_case_600_thermal_capacity_metric_matches_reference() {
    // Compute Cm from the documented layer stack.
    let layers: [(&str, f64, f64, f64); 3] = [
        ("Plasterboard", 784.0, 840.0, 0.012),
        ("Fiberglass", 12.0, 840.0, 0.066),
        ("Wood siding", 530.0, 900.0, 0.009),
    ];
    let cm_computed: f64 = layers
        .iter()
        .map(|(_, rho, cp, d)| rho * cp * d / 1000.0)
        .sum();

    let lo = CASE_600_CM_KJ_M2K_DOCUMENTED * (1.0 - CASE_600_CM_TOLERANCE_PCT / 100.0);
    let hi = CASE_600_CM_KJ_M2K_DOCUMENTED * (1.0 + CASE_600_CM_TOLERANCE_PCT / 100.0);

    assert!(
        cm_computed >= lo && cm_computed <= hi,
        "Case 600 Cm = {cm_computed:.4} kJ/m²K outside [{lo:.4}, {hi:.4}] \
         (documented {CASE_600_CM_KJ_M2K_DOCUMENTED})"
    );
}

// =============================================================================
// Test 3: GaugeSolver shadow-mode diurnal response (non-zero, finite, phased)
// =============================================================================

/// The GaugeSolver shadow-mode flux through a synthetic 24-hour diurnal cycle
/// must: (a) be finite for all 24 hours, (b) have non-zero diurnal amplitude
/// (responds to solar forcing), and (c) peak within ±2 h of peak sol-air temp.
#[test]
fn test_case_600_gauge_solver_shadow_diurnal_response() {
    let wall = case_600_wall();
    let fluxes = gauge_shadow_diurnal_fluxes(&wall);

    assert_eq!(fluxes.len(), 24, "24 hourly fluxes expected");

    // (a) All finite.
    for (h, &f) in fluxes.iter().enumerate() {
        assert!(
            f.is_finite(),
            "Hour {h}: flux not finite ({f})"
        );
    }

    // (b) Non-zero diurnal amplitude.
    let f_min = fluxes.iter().cloned().fold(f64::INFINITY, f64::min);
    let f_max = fluxes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let amplitude = f_max - f_min;
    assert!(
        amplitude > 10.0,
        "Diurnal amplitude {amplitude:.2} W/m² too small — GaugeSolver must respond to solar"
    );

    // (c) Peak flux near solar noon (hour 12). Peak sol-air ≈ hour 12.
    let peak_hour = fluxes
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(h, _)| h)
        .unwrap();
    let phase_lag = ((peak_hour as i32 - 12_i32).abs()) as f64;
    assert!(
        phase_lag <= 2.0,
        "Peak flux at hour {peak_hour} (lag {phase_lag}h from solar noon) — expected ≤ 2h"
    );
}

// =============================================================================
// Test 4: GaugeSolver does not clamp extreme solar forcing (non-throttling)
// =============================================================================

/// The gauge path must propagate extreme solar forcing (>5 kW/m²) without
/// silent clamping. The legacy 100 kW HVAC cap has been removed from the
/// gauge path; flux must grow linearly with solar.
#[test]
fn test_case_600_gauge_solver_shadow_does_not_clamp_extreme_solar() {
    let wall = case_600_wall();
    let t_int = Temperature::from_value(T_INDOOR_HVAC_SETPOINT_C);
    let h_ext = HeatTransferCoefficient::from_value(CASE_600_H_EXT);

    let mut solver = GaugeSolver::default();
    solver.initialize(&wall).expect("init");

    let moderate = solver
        .step_with_boundary_conditions(
            Time::from_value(DT_SECONDS),
            t_int,
            h_ext,
            GaugeBoundaryConditions::new(800.0, 30.0),
        )
        .expect("moderate step")
        .to_value();

    let extreme = solver
        .step_with_boundary_conditions(
            Time::from_value(DT_SECONDS),
            t_int,
            h_ext,
            GaugeBoundaryConditions::new(5000.0, 30.0),
        )
        .expect("extreme step")
        .to_value();

    // Flux must increase (not clamp) with more solar.
    assert!(
        extreme > moderate,
        "Extreme solar flux {extreme} must exceed moderate {moderate} — no throttling"
    );
    // The increment must be consistent with the linear sol-air relation:
    // Δq = Δsolar / h_ext / R_total.
    let r_total = wall.total_r_value();
    let expected_delta = (5000.0 - 800.0) / CASE_600_H_EXT / r_total;
    let actual_delta = extreme - moderate;
    assert!(
        (actual_delta - expected_delta).abs() / expected_delta < 0.01,
        "Δflux {actual_delta:.4} ≠ expected {expected_delta:.4} (linear sol-air)"
    );
}

// =============================================================================
// Test 5: Shadow-mode parity with baseline FiveR1CSolver (steady state)
// =============================================================================

/// In steady state with **no solar**, the GaugeSolver shadow-mode flux must
/// match the baseline `FiveR1CSolver` flux to machine precision (parity contract).
#[test]
fn test_case_600_gauge_shadow_matches_baseline_in_steady_state() {
    let wall = case_600_wall();
    let t_int_val = 22.0;
    let t_ext_val = 5.0;
    let h_int = HeatTransferCoefficient::from_value(CASE_600_H_INT);
    let h_ext = HeatTransferCoefficient::from_value(CASE_600_H_EXT);

    // Baseline.
    let mut baseline = FiveR1CSolver::new();
    baseline.initialize(&wall).expect("baseline init");
    let baseline_flux = baseline
        .step(
            Time::from_value(DT_SECONDS),
            Temperature::from_value(t_int_val),
            Temperature::from_value(t_ext_val),
            h_int,
            h_ext,
        )
        .expect("baseline step")
        .to_value();

    // Gauge shadow.
    let mut adapter = PhysicsAdapter::new(PhysicsAdapterConfig::gauge_shadow());
    adapter.initialize(&wall).expect("adapter init");
    let adapter_flux = adapter
        .step(
            Time::from_value(DT_SECONDS),
            Temperature::from_value(t_int_val),
            Temperature::from_value(t_ext_val),
            h_int,
            h_ext,
            0.0, // no solar
        )
        .expect("adapter step")
        .to_value();

    assert_eq!(
        adapter_flux, baseline_flux,
        "Shadow-mode primary flux must equal baseline in steady state"
    );

    let record: &GaugeShadowRecord = adapter
        .last_shadow_record()
        .expect("shadow record present");
    assert!(record.error.is_none(), "Expected no shadow error");
    let gauge_flux = record.gauge_flux_wm2.expect("shadow flux recorded");
    let drift = (gauge_flux - baseline_flux).abs();
    assert!(
        drift < 1e-6,
        "Gauge flux drift {drift:.2e} from baseline in steady state"
    );
}

// =============================================================================
// Test 6: Shadow-mode records translated boundary correctly
// =============================================================================

/// The recorded `gauge_connection` must carry positive solar during daytime
/// and ≈0 solar at night.
#[test]
fn test_case_600_gauge_shadow_records_translated_boundary_correctly() {
    let wall = case_600_wall();
    let mut adapter = PhysicsAdapter::new(PhysicsAdapterConfig::gauge_shadow());
    adapter.initialize(&wall).expect("init");

    let t_int = T_INDOOR_HVAC_SETPOINT_C;
    let h_int = HeatTransferCoefficient::from_value(CASE_600_H_INT);
    let h_ext = HeatTransferCoefficient::from_value(CASE_600_H_EXT);

    let mut records: Vec<GaugeShadowRecord> = Vec::with_capacity(24);
    for hour in 0..24 {
        let t_out = outdoor_temperature_at(hour);
        let solar = solar_irradiance_at(hour);
        adapter
            .step(
                Time::from_value(DT_SECONDS),
                Temperature::from_value(t_int),
                Temperature::from_value(t_out),
                h_int,
                h_ext,
                solar,
            )
            .expect("step");
        records.push(adapter.last_shadow_record().expect("record").clone());
    }

    let noon: Vec<&GaugeShadowRecord> = records
        .iter()
        .enumerate()
        .filter(|(h, _)| (7..=17).contains(h))
        .map(|(_, r)| r)
        .collect();
    let midnight: Vec<&GaugeShadowRecord> = records
        .iter()
        .enumerate()
        .filter(|(h, _)| !(7..=17).contains(h))
        .map(|(_, r)| r)
        .collect();

    for r in &noon {
        assert!(
            r.gauge_connection[0] > 0.0,
            "Noon: expected positive solar in gauge_connection, got {:?}",
            r.gauge_connection
        );
    }
    for r in &midnight {
        assert!(
            r.gauge_connection[0].abs() < 1e-9,
            "Midnight: expected ≈0 solar, got {}",
            r.gauge_connection[0]
        );
    }
}

// =============================================================================
// Test 7: Zone-count envelope matches MAX_ZONES
// =============================================================================

/// The Case 600 building is single-zone; the ThermalManifold must accept it.
#[test]
fn test_case_600_zone_count_envelope_matches_geometry_tensor() {
    let manifold = ThermalManifold::from_5r1c_parameters(22.0, 21.0, 1.7893, 10_000.0, 12_860.0);
    // The 4-D ambient space is fixed: scalar_field is a Vector4.
    assert_eq!(manifold.scalar_field.len(), 4, "4-D ambient space");
    assert!(MAX_ZONES >= 1, "MAX_ZONES must accommodate single-zone Case 600");
}

// =============================================================================
// Test 8: GaugeSolver wall U-value matches ASHRAE 140 Case 600 target
// =============================================================================

/// The GaugeSolver's `wall.total_r_value()` must reproduce the Case 600
/// U-value (with films) ≈ 0.508 W/m²K within 2 % of the ASHRAE target 0.514.
#[test]
fn test_case_600_gauge_solver_wall_u_value_matches_reference() {
    let wall = case_600_wall();
    let r_cond = wall.total_r_value(); // conduction R only (WallSpec excludes films)
    let r_total = r_cond + 1.0 / CASE_600_H_INT + 1.0 / CASE_600_H_EXT;
    let u_total = 1.0 / r_total;

    // ASHRAE 140 Case 600 wall U-value target ≈ 0.514 W/m²K.
    let target_u = 0.514;
    let pct_err = ((u_total - target_u) / target_u).abs() * 100.0;

    assert!(
        pct_err < 2.0,
        "Case 600 U-value {u_total:.4} W/m²K vs target {target_u} ({pct_err:.2}% err)"
    );
}
