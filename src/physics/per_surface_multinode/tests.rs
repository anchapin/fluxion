//! Unit tests for the per-surface multi-node conduction formulation
//! (Issue #3983).
//!
//! **Option-B scope (Issue #3983).** The suite asserts only what the
//! formulation's own physics guarantees:
//!
//! 1. **Bounded, physical temperatures** — every node temperature stays
//!    inside the band the driving conditions allow (a passive RC network
//!    cannot leave the hull of its boundary temperatures), and the sol-air
//!    temperature follows ISO 13790 §C.3.2.
//! 2. **Energy balance** — the first-law residual of the ledger closes to
//!    round-off over a full synthetic run, in the free-float *and* in the
//!    conditioned (thermostat-clamped) protocol.
//! 3. **Formulation structure** — the two node states per surface are ODE
//!    states (not algebraic pins), they respond on distinct timescales,
//!    and the solar is distributed per surface by area × absorptance.
//!
//! Every assertion below is a property of the per-surface formulation
//! itself. **No test compares the per-surface model against the lumped
//! 5R1C reference**: the per-surface-vs-lumped peak-ordering claims (peak
//! magnitude order, peak-time order, and the "locked metrics" inequality
//! direction) were dropped under the option-B decision. The lumped network
//! carries a single thermal state per zone, so *no* ordering between the
//! two formulations' peaks follows from the physics — the two models
//! differ in structure, not in a guaranteed inequality — and the
//! comparison therefore carried no evidence, only a tuning target.

use super::*;
use fluxion_core::per_surface_conduction::SurfaceKind;

// ---------------------------------------------------------------------------
// Scenario helpers (synthetic single-zone case)
// ---------------------------------------------------------------------------

/// Zone geometry of the synthetic case: 8.0 m (E-W) × 6.0 m (N-S) × 2.7 m.
const ZONE_W: f64 = 8.0;
const ZONE_D: f64 = 6.0;
const ZONE_H: f64 = 2.7;

/// Construction: heavyweight concrete, U = 0.15 W/(m²·K), 200 kJ/(m²·K).
const U_VALUE: f64 = 0.15;
const C_AREAL: f64 = 2.0e5;
const H_EXTERIOR: f64 = 25.0;
const H_INTERIOR_VERTICAL: f64 = 8.29;
const H_INTERIOR_HORIZONTAL: f64 = 10.29;
const ALPHA_ROOF: f64 = 0.6;
const ALPHA_WALL: f64 = 0.5;

/// Zone air heat capacity [J/K] (2 × ρ × c × V) and ventilation conductance.
const C_AIR: f64 = 2.0 * 1.2 * 1005.0 * (ZONE_W * ZONE_D * ZONE_H);
const ACH: f64 = 2.0;
const H_VE: f64 = ACH * (ZONE_W * ZONE_D * ZONE_H) * 1200.0 / 3600.0;

/// Timestep [s] and simulation length [h].
const DT: f64 = 3600.0;
const HOURS: usize = 24 * 14;

/// Heating / cooling setpoints of the synthetic case [°C].
const T_HEAT: f64 = 20.0;
const T_COOL: f64 = 26.0;

/// Synthetic envelope: roof + four walls (no floor — the synthetic case
/// exercises the *air-side* envelope network, and a ground-coupled floor
/// is the §8 ground-slab work (#4270 area), which this module does not
/// touch).
fn synthetic_surfaces() -> Vec<SurfaceSpec> {
    let roof = SurfaceSpec::from_areal(
        0,
        SurfaceKind::Roof,
        ZONE_W * ZONE_D,
        U_VALUE,
        C_AREAL,
        H_EXTERIOR,
        H_INTERIOR_HORIZONTAL,
        ALPHA_ROOF,
    );
    let wall = |id: usize, width: f64| {
        SurfaceSpec::from_areal(
            id,
            SurfaceKind::Wall,
            width * ZONE_H,
            U_VALUE,
            C_AREAL,
            H_EXTERIOR,
            H_INTERIOR_VERTICAL,
            ALPHA_WALL,
        )
    };
    vec![
        roof,
        wall(1, ZONE_W), // south
        wall(2, ZONE_D), // east
        wall(3, ZONE_D), // west
        wall(4, ZONE_W), // north
    ]
}

/// Synthetic weather + solar for hour `t` of the run.
///
/// Winter day: outdoor air `10 ± 8 °C` (minimum at 03:00, maximum at
/// 15:00), horizontal irradiance peaking at 150 W/m² and south-vertical
/// irradiance peaking at 400 W/m² — the low-sun geometry of a winter day.
fn synthetic_weather(t: usize) -> (f64, f64, f64) {
    let hour = (t % 24) as f64;
    let t_out = 10.0 + 8.0 * (2.0 * std::f64::consts::PI * (hour - 15.0) / 24.0).cos();
    let i_roof = 150.0 * (std::f64::consts::PI * (hour - 6.0) / 12.0).sin().max(0.0);
    let i_south = 400.0 * (std::f64::consts::PI * (hour - 7.0) / 11.0).sin().max(0.0);
    (t_out, i_roof, i_south)
}

/// Incident irradiance on each synthetic surface at hour `t` [W/m²].
fn synthetic_irradiance(t: usize) -> Vec<f64> {
    let (_, i_roof, i_south) = synthetic_weather(t);
    vec![i_roof, i_south, 0.0, 0.0, 0.0]
}

/// One conditioned (thermostat-clamped) step of the synthetic case.
struct ConditionedStep {
    /// The HVAC load the thermostat must meet [W] (positive = the zone must
    /// be cooled).
    load: f64,
    /// The zone air temperature after the step [°C].
    t_air: f64,
    /// The interior surface-node temperatures after the step [°C].
    t_surface: Vec<f64>,
    /// The depth-node temperatures after the step [°C].
    t_mass: Vec<f64>,
    /// The energy ledger of the step.
    ledger: EnergyLedger,
}

/// Run the per-surface formulation over the synthetic case with the zone air
/// **held** at `t_set` by an ideal thermostat (the conditioned protocol);
/// returns the per-step record of the run.
fn run_per_surface_conditioned(specs: &[SurfaceSpec], t_set: f64) -> Vec<ConditionedStep> {
    let mut model =
        PerSurfaceMultiNodeConduction::new(specs.to_vec(), ZoneAirSpec::new(C_AIR, H_VE), t_set);
    let n = specs.len();
    let mut steps = Vec::with_capacity(HOURS);
    for t in 0..HOURS {
        let (t_out, _, _) = synthetic_weather(t);
        let irr = synthetic_irradiance(t);
        let exterior = vec![t_out; n];
        let absorbed: Vec<f64> = specs
            .iter()
            .zip(irr.iter())
            .map(|(s, i)| absorbed_flux(s, *i))
            .collect();
        let transmitted = vec![0.0; n];
        let inputs = StepInputs {
            exterior_temperatures: &exterior,
            t_outdoor: t_out,
            phi_ia: 0.0,
            absorbed_solar: &absorbed,
            transmitted_solar: &transmitted,
        };
        let load = model.step_clamped(DT, &inputs, t_set);
        steps.push(ConditionedStep {
            load,
            t_air: model.t_air(),
            t_surface: model.surface_temperatures(),
            t_mass: model.mass_temperatures(),
            ledger: model.energy_ledger(),
        });
    }
    steps
}

/// The band the driving conditions of the synthetic case allow: the lowest
/// and highest **boundary** temperature the network sees (the outdoor air
/// and the per-surface sol-air temperature).
///
/// A passive RC network advanced by backward Euler is a convex
/// combination of its boundary temperatures, so no node temperature can
/// leave that hull — this is the bound the "physical temperatures"
/// assertions below use.
fn synthetic_temperature_hull(specs: &[SurfaceSpec]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for t in 0..HOURS {
        let (t_out, _, _) = synthetic_weather(t);
        let irr = synthetic_irradiance(t);
        lo = lo.min(t_out);
        hi = hi.max(t_out);
        for (spec, i) in specs.iter().zip(irr.iter()) {
            let solair = spec.sol_air_temperature(t_out, absorbed_flux(spec, *i));
            lo = lo.min(solair);
            hi = hi.max(solair);
        }
    }
    (lo, hi)
}

// ---------------------------------------------------------------------------
// 1. Formulation structure
// ---------------------------------------------------------------------------

/// The interior surface node is a *state*: after a step it lags the
/// algebraic (massless) surface-node value implied by the current mass and
/// air temperatures, and it keeps evolving over successive steps.
#[test]
fn test_surface_node_is_an_ode_state_not_an_algebraic_pin() {
    let specs = vec![SurfaceSpec::from_areal(
        0,
        SurfaceKind::Wall,
        20.0,
        U_VALUE,
        C_AREAL,
        H_EXTERIOR,
        H_INTERIOR_VERTICAL,
        ALPHA_WALL,
    )];
    let mut model =
        PerSurfaceMultiNodeConduction::new(specs.clone(), ZoneAirSpec::new(C_AIR, H_VE), 20.0);

    let exterior = vec![30.0f64];
    let absorbed = vec![0.0f64];
    let transmitted = vec![0.0f64];
    let inputs = StepInputs {
        exterior_temperatures: &exterior,
        t_outdoor: 30.0,
        phi_ia: 0.0,
        absorbed_solar: &absorbed,
        transmitted_solar: &transmitted,
    };

    let t_before = model.surface(0).unwrap().t_surface;
    model.step(DT, &inputs);
    let state = model.surface(0).unwrap();
    let t_after = state.t_surface;
    let t_mass = state.t_mass;
    let t_air = model.t_air();
    let spec = &specs[0];

    // The algebraic (massless) surface-node value implied by the *new* mass
    // and air temperatures.
    let algebraic = (spec.h_tr_ms * t_mass + spec.h_tr_is * t_air) / (spec.h_tr_ms + spec.h_tr_is);

    assert!(
        (t_after - t_before).abs() > 1e-6,
        "the surface node must respond to the boundary change"
    );
    assert!(
        (t_after - algebraic).abs() > 1e-3,
        "the surface node must lag its algebraic value (state, not pin): \
         t_after={t_after:.6}, algebraic={algebraic:.6}"
    );
}

/// The depth node and the surface node respond on **distinct timescales**.
///
/// The protocol is the conditioned one: the zone air is held at the
/// setpoint by an ideal thermostat, and the *envelope* is then stepped by
/// the setpoint change (20 °C → 30 °C). The interior surface node tracks
/// the held air within hours (it is separated from the zone only by the
/// interior film), while the depth node — which is separated from the
/// zone by the construction resistance — is still essentially unchanged.
///
/// The 63.2 % crossing times are measured against the *converged* values
/// (the trace is run far past the slow node's time constant), so the
/// measurement does not depend on where the trace is truncated.
#[test]
fn test_step_response_exhibits_distinct_node_time_constants() {
    let specs = vec![SurfaceSpec::from_areal(
        0,
        SurfaceKind::Wall,
        20.0,
        U_VALUE,
        C_AREAL,
        H_EXTERIOR,
        H_INTERIOR_VERTICAL,
        ALPHA_WALL,
    )];
    let mut model =
        PerSurfaceMultiNodeConduction::new(specs.clone(), ZoneAirSpec::new(C_AIR, H_VE), 20.0);

    let exterior = vec![20.0f64];
    let absorbed = vec![0.0f64];
    let transmitted = vec![0.0f64];
    let inputs = StepInputs {
        exterior_temperatures: &exterior,
        t_outdoor: 20.0,
        phi_ia: 0.0,
        absorbed_solar: &absorbed,
        transmitted_solar: &transmitted,
    };

    // Settle the envelope at the initial setpoint (20 °C).
    for _ in 0..48 {
        model.step_clamped(DT, &inputs, 20.0);
    }
    let t_s0 = model.surface(0).unwrap().t_surface;
    let t_m0 = model.surface(0).unwrap().t_mass;

    // Step the held zone air by +10 K and record both node temperatures
    // until the slow node has converged (6000 steps ≫ its time constant).
    //
    // The 63.2 % crossings are measured against the *converged* values,
    // which are read off the end of the recorded trace — measuring them
    // against a running "final" value would make the first step trivially
    // satisfy the crossing condition.
    const STEPS: usize = 6000;
    let mut trace: Vec<(f64, f64)> = Vec::with_capacity(STEPS);
    for _ in 1..=STEPS {
        model.step_clamped(DT, &inputs, 30.0);
        trace.push((
            model.surface(0).unwrap().t_surface,
            model.surface(0).unwrap().t_mass,
        ));
    }
    let t_s_final = trace.last().unwrap().0;
    let t_m_final = trace.last().unwrap().1;

    let mut surface_63 = None;
    let mut mass_63 = None;
    for (i, (t_s, t_m)) in trace.iter().enumerate() {
        let step = i + 1;
        if surface_63.is_none() && *t_s >= t_s0 + 0.632 * (t_s_final - t_s0) {
            surface_63 = Some(step);
        }
        if mass_63.is_none() && *t_m >= t_m0 + 0.632 * (t_m_final - t_m0) {
            mass_63 = Some(step);
        }
    }

    let tau_surface = surface_63.expect("the surface node must reach 63.2 % of its response");
    let tau_mass = mass_63.expect("the depth node must reach 63.2 % of its response");
    assert!(
        tau_mass > 2 * tau_surface,
        "the depth node must respond on a clearly slower timescale than the \
         surface node: tau_surface={tau_surface} steps, tau_mass={tau_mass} steps"
    );
    assert!(
        tau_surface >= 1,
        "the surface node must respond within the simulated window"
    );

    // The step is part of the energy-conserving protocol: the ledger's
    // first-law residual stays at round-off while the nodes re-equilibrate.
    let residual = model.energy_residual().abs();
    let scale = model.energy_ledger().envelope_storage.abs().max(1.0);
    assert!(
        residual < 1e-6 * scale,
        "the clamped step must conserve energy: residual={residual} W (scale {scale} W)"
    );
}

/// The absorbed solar is apportioned across the surfaces by
/// `absorptance × area` — not by a fixed split, and not into a single
/// lumped node.
#[test]
fn test_absorbed_solar_is_distributed_by_area_times_absorptance() {
    let specs = synthetic_surfaces();
    let total = 10_000.0;
    let shares = distribute_by_area_absorptance(&specs, total);

    let weight_sum: f64 = specs.iter().map(|s| s.absorptance * s.area).sum();
    for (share, spec) in shares.iter().zip(specs.iter()) {
        let expected = total * (spec.absorptance * spec.area) / weight_sum;
        assert!(
            (share - expected).abs() < 1e-9,
            "share mismatch for surface {}: got {share}, expected {expected}",
            spec.id
        );
    }
    let sum: f64 = shares.iter().sum();
    assert!(
        (sum - total).abs() < 1e-9,
        "shares must sum to the total flux"
    );

    // The roof (larger area, higher absorptance) must take strictly more
    // than *every* wall — a fixed 50/50 split could not do that.
    //
    // (The earlier version of this assertion compared the south wall with
    // the north wall, which are both 8.0 m wide and therefore carry
    // identical `α · A` weights: it asserted a strict inequality between
    // two equal weights, which is a property of the *geometry*, not of the
    // distribution rule.)
    for (share, spec) in shares.iter().zip(specs.iter()).skip(1) {
        assert!(
            shares[0] > *share,
            "the roof must take strictly more than wall {} ({:.1} W vs {:.1} W)",
            spec.id,
            shares[0],
            share
        );
    }
}

/// Glazing-transmitted solar is distributed across the interior faces by
/// `interior absorptance × area`.
#[test]
fn test_transmitted_solar_is_distributed_by_interior_area_absorptance() {
    let specs: Vec<SurfaceSpec> = synthetic_surfaces()
        .into_iter()
        .enumerate()
        .map(|(i, s)| {
            if i == 1 {
                s.with_interior_absorptance(0.8)
            } else {
                s.with_interior_absorptance(0.4)
            }
        })
        .collect();
    let total = 1_000.0;
    let shares = distribute_transmitted(&specs, total);
    let weight_sum: f64 = specs.iter().map(|s| s.interior_absorptance * s.area).sum();
    for (share, spec) in shares.iter().zip(specs.iter()) {
        let expected = total * (spec.interior_absorptance * spec.area) / weight_sum;
        assert!(
            (share - expected).abs() < 1e-9,
            "transmitted share mismatch for surface {}",
            spec.id
        );
    }
    assert!((shares.iter().sum::<f64>() - total).abs() < 1e-9);
}

/// The absorbed solar reaches the surface it was incident on: irradiating
/// the roof alone warms the roof's depth node, while the shaded walls'
/// depth nodes stay (nearly) at their initial temperature — they can only
/// be warmed through the zone-air path, never by another surface's solar.
///
/// The earlier version of this test asserted that a shaded surface's depth
/// node stays within 0.05 K of its initial temperature *in absolute
/// terms*. That claim is not a physics guarantee: the shaded surfaces are
/// coupled to the (irradiated) zone air through the interior film, so the
/// zone-air path legitimately warms them a little. What the physics
/// guarantees is the *ordering* of the response — the irradiated surface
/// warms by orders of magnitude more than the shaded ones — plus the
/// boundedness of both.
#[test]
fn test_absorbed_solar_enters_only_the_irradiated_surface() {
    let specs = synthetic_surfaces();
    let mut model =
        PerSurfaceMultiNodeConduction::new(specs.clone(), ZoneAirSpec::new(C_AIR, H_VE), 20.0);
    let n = specs.len();
    let exterior = vec![20.0f64; n];
    let transmitted = vec![0.0f64; n];
    let irr = vec![900.0f64, 0.0, 0.0, 0.0, 0.0];
    let absorbed: Vec<f64> = specs
        .iter()
        .zip(irr.iter())
        .map(|(s, i)| absorbed_flux(s, *i))
        .collect();
    let inputs = StepInputs {
        exterior_temperatures: &exterior,
        t_outdoor: 20.0,
        phi_ia: 0.0,
        absorbed_solar: &absorbed,
        transmitted_solar: &transmitted,
    };
    for _ in 0..12 {
        model.step(DT, &inputs);
    }
    let masses = model.mass_temperatures();
    assert!(
        masses[0] > 20.05,
        "the irradiated roof depth node must warm: {}",
        masses[0]
    );
    // The shaded surfaces' depth nodes are warmed by the zone-air path
    // only: their rise must stay far below the irradiated surface's.
    let shaded_rise: f64 = masses.iter().skip(1).map(|m| m - 20.0).fold(0.0, f64::max);
    assert!(
        shaded_rise < 0.05,
        "a shaded surface must stay near its initial temperature (only the \
         zone-air path can warm it): {shaded_rise} K"
    );
    assert!(
        masses[0] - 20.0 > 10.0 * shaded_rise.max(1e-6),
        "the irradiated surface must warm by orders of magnitude more than \
         the shaded ones: roof={:.4} K, shaded={shaded_rise:.4} K",
        masses[0] - 20.0
    );
    // Bounded: no node temperature may leave the hull of the driving
    // conditions (the outdoor air and this scenario's sol-air
    // temperatures).
    let hi = specs
        .iter()
        .zip(absorbed.iter())
        .map(|(s, a)| s.sol_air_temperature(20.0, *a))
        .fold(f64::NEG_INFINITY, f64::max);
    let lo = 20.0;
    for (k, state) in model.surfaces().iter().enumerate() {
        for (label, t) in [("surface", state.t_surface), ("depth", state.t_mass)] {
            assert!(
                t.is_finite() && t >= lo - 1.0 && t <= hi + 1.0,
                "surface {k} {label} node temperature {t} °C is outside the \
                 physical band [{lo}, {hi}] °C"
            );
        }
    }
}

/// The implicit step conserves energy: the first-law residual of the ledger
/// is zero up to round-off over a full synthetic run.
#[test]
fn test_energy_ledger_residual_is_zero() {
    let specs = synthetic_surfaces();
    let mut model =
        PerSurfaceMultiNodeConduction::new(specs.clone(), ZoneAirSpec::new(C_AIR, H_VE), 20.0);
    let n = specs.len();
    for t in 0..HOURS {
        let (t_out, _, _) = synthetic_weather(t);
        let irr = synthetic_irradiance(t);
        let exterior = vec![t_out; n];
        let absorbed: Vec<f64> = specs
            .iter()
            .zip(irr.iter())
            .map(|(s, i)| absorbed_flux(s, *i))
            .collect();
        let transmitted = vec![0.0f64; n];
        let inputs = StepInputs {
            exterior_temperatures: &exterior,
            t_outdoor: t_out,
            phi_ia: 0.0,
            absorbed_solar: &absorbed,
            transmitted_solar: &transmitted,
        };
        model.step(DT, &inputs);
        let residual = model.energy_residual().abs();
        let scale = model
            .energy_ledger()
            .exterior_to_envelope
            .abs()
            .max(model.energy_ledger().envelope_storage.abs())
            .max(1.0);
        assert!(
            residual < 1e-6 * scale,
            "first-law residual {residual} W is not within tolerance (scale {scale} W) \
             at hour {t}"
        );
    }
}

/// The implicit solve converges to the analytic steady state of the linear
/// network (single surface + zone air node).
///
/// The analytic reference is the ISO 13790 §C network: the sol-air
/// temperature is `T_ext + α · I / h_se` (ISO 13790 §C.3.2), i.e. the
/// absorbed flux is divided by the exterior **film** conductance, not by
/// the combined exterior-to-depth-node conductance `1/(R_se + R_1)`.
#[test]
fn test_steady_state_matches_the_analytic_network_solution() {
    let specs = vec![SurfaceSpec::from_areal(
        0,
        SurfaceKind::Wall,
        20.0,
        U_VALUE,
        C_AREAL,
        H_EXTERIOR,
        H_INTERIOR_VERTICAL,
        ALPHA_WALL,
    )];
    let spec = specs[0].clone();
    let h_ve = 50.0;
    let mut model = PerSurfaceMultiNodeConduction::new(specs, ZoneAirSpec::new(C_AIR, h_ve), 20.0);

    let t_out = 5.0;
    let exterior = vec![t_out];
    let absorbed = vec![1_000.0];
    let transmitted = vec![0.0];
    let inputs = StepInputs {
        exterior_temperatures: &exterior,
        t_outdoor: t_out,
        phi_ia: 0.0,
        absorbed_solar: &absorbed,
        transmitted_solar: &transmitted,
    };
    for _ in 0..200_000 {
        model.step(DT, &inputs);
    }

    // Analytic steady state of the linear network (ISO 13790 §C sol-air).
    let solair = spec.sol_air_temperature(t_out, absorbed[0]);
    let k = h_ve / (spec.h_tr_is + h_ve) * spec.h_tr_is * (1.0 / spec.h_tr_ms + 1.0 / spec.h_tr_em);
    let t_s = (solair + k * t_out) / (1.0 + k);
    let t_air = (spec.h_tr_is * t_s + h_ve * t_out) / (spec.h_tr_is + h_ve);
    let t_m = t_s - (spec.h_tr_is / spec.h_tr_ms) * (t_air - t_s);

    let state = model.surface(0).unwrap();
    assert!(
        (state.t_surface - t_s).abs() < 1e-3,
        "surface node steady state mismatch: got {}, expected {t_s}",
        state.t_surface
    );
    assert!(
        (state.t_mass - t_m).abs() < 1e-3,
        "depth node steady state mismatch: got {}, expected {t_m}",
        state.t_mass
    );
    assert!(
        (model.t_air() - t_air).abs() < 1e-3,
        "zone air steady state mismatch: got {}, expected {t_air}",
        model.t_air()
    );
}

/// The sol-air temperature follows ISO 13790 §C.3.2: the absorbed flux is
/// divided by the exterior **film** conductance `h_se`, so the exterior
/// face is lifted above the ambient by `α · I / h_se`.
///
/// This is the property that keeps the construction's internal
/// temperatures physical: dividing by the combined
/// exterior-to-depth-node conductance `1/(R_se + R_1)` instead would push
/// the whole absorbed flux through the construction and inflate the
/// interior solar gain by `(R_se + R_1) / R_se` (≈ 80× here), driving the
/// depth node to hundreds of degrees.
#[test]
fn test_sol_air_temperature_follows_iso_13790() {
    let specs = synthetic_surfaces();
    for (spec, irr) in specs.iter().zip(synthetic_irradiance(11).iter()) {
        let phi_abs = absorbed_flux(spec, *irr);
        let expected = spec.sol_air_temperature(7.0, phi_abs);
        // ISO 13790 §C.3.2: T_solair = T_ext + α · I / h_se.
        let iso = 7.0 + spec.absorptance * irr / H_EXTERIOR;
        assert!(
            (expected - iso).abs() < 1e-9,
            "the sol-air temperature must be the ISO 13790 §C.3.2 value: \
             got {expected}, expected {iso}"
        );
        // The offset is bounded by the exterior film's rejection capacity,
        // not by the construction's resistance.
        if phi_abs > 0.0 {
            let amplified = 7.0 + phi_abs / spec.h_tr_em;
            assert!(
                expected < amplified,
                "the sol-air offset must not be amplified by the construction \
                 resistance: {expected} °C vs {amplified} °C"
            );
        }
    }
    // A surface without an exterior film conductance receives no offset.
    let mut bare = specs[0].clone();
    bare.h_ext_film = 0.0;
    assert!((bare.sol_air_temperature(7.0, 1_000.0) - 7.0).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// 2. Boundedness and energy balance over a full run
// ---------------------------------------------------------------------------

/// A free-float run of the synthetic case keeps every node temperature
/// inside the band the driving conditions allow, and its first-law
/// residual closes at every step.
///
/// The band is the hull of the boundary temperatures (the outdoor air and
/// the per-surface sol-air temperature): a passive network advanced by
/// backward Euler is a convex combination of its boundaries, so a node
/// temperature outside that hull would be a formulation defect, not a
/// modelling choice.
#[test]
fn test_free_float_run_keeps_temperatures_bounded_and_balances_energy() {
    let specs = synthetic_surfaces();
    let (lo, hi) = synthetic_temperature_hull(&specs);
    let mut model =
        PerSurfaceMultiNodeConduction::new(specs.clone(), ZoneAirSpec::new(C_AIR, H_VE), 20.0);
    let n = specs.len();
    for t in 0..HOURS {
        let (t_out, _, _) = synthetic_weather(t);
        let irr = synthetic_irradiance(t);
        let exterior = vec![t_out; n];
        let absorbed: Vec<f64> = specs
            .iter()
            .zip(irr.iter())
            .map(|(s, i)| absorbed_flux(s, *i))
            .collect();
        let transmitted = vec![0.0f64; n];
        let inputs = StepInputs {
            exterior_temperatures: &exterior,
            t_outdoor: t_out,
            phi_ia: 0.0,
            absorbed_solar: &absorbed,
            transmitted_solar: &transmitted,
        };
        model.step(DT, &inputs);

        let ledger = model.energy_ledger();
        let scale = ledger
            .exterior_to_envelope
            .abs()
            .max(ledger.envelope_storage.abs())
            .max(ledger.air_storage.abs())
            .max(1.0);
        assert!(
            model.energy_residual().abs() < 1e-6 * scale,
            "first-law residual must close at hour {t}: {} W (scale {scale} W)",
            model.energy_residual()
        );

        for (k, state) in model.surfaces().iter().enumerate() {
            for (label, temp) in [("surface", state.t_surface), ("depth", state.t_mass)] {
                assert!(
                    temp.is_finite(),
                    "surface {k} {label} node temperature must be finite, got {temp}"
                );
                assert!(
                    temp >= lo - 1.0 && temp <= hi + 1.0,
                    "surface {k} {label} node temperature {temp} °C is outside \
                     the physical band [{lo}, {hi}] °C at hour {t}"
                );
            }
        }
        let t_air = model.t_air();
        assert!(
            t_air.is_finite() && t_air >= lo - 1.0 && t_air <= hi + 1.0,
            "the zone air temperature {t_air} °C is outside the physical band \
             [{lo}, {hi}] °C at hour {t}"
        );
    }
}

/// The conditioned (thermostat-clamped) protocol holds the air node at the
/// setpoint, reports the load the thermostat has to meet, and closes its
/// first-law balance at every step.
#[test]
fn test_conditioned_run_holds_the_setpoint_and_balances_energy() {
    let specs = synthetic_surfaces();
    for t_set in [T_COOL, T_HEAT] {
        let (lo, hi) = synthetic_temperature_hull(&specs);
        // The held air node is a boundary of the network too.
        let lo = lo.min(t_set);
        let hi = hi.max(t_set);
        for (t, step) in run_per_surface_conditioned(&specs, t_set)
            .iter()
            .enumerate()
        {
            // The air node is prescribed: it must sit exactly on the
            // setpoint.
            assert!(
                (step.t_air - t_set).abs() < 1e-12,
                "the conditioned air node must stay at the setpoint: {} vs {t_set}",
                step.t_air
            );

            // The thermostat carries the air node's imbalance, so the
            // ledger's first-law residual still closes.
            let ledger = step.ledger;
            let scale = ledger
                .surface_to_air
                .abs()
                .max(ledger.ventilation.abs())
                .max(ledger.envelope_storage.abs())
                .max(1.0);
            let residual = ledger.residual();
            assert!(
                residual.abs() < 1e-6 * scale,
                "the conditioned step must conserve energy at hour {t}: {residual} W \
                 (scale {scale} W)"
            );

            // The reported load is the documented air-node imbalance:
            // positive when the envelope and the ventilation path deliver
            // more heat to the zone than the setpoint can absorb.
            let expected_load = ledger.surface_to_air + ledger.ventilation + ledger.internal;
            assert!(
                (step.load - expected_load).abs() < 1e-9 * scale,
                "the reported load must be the air node's imbalance: got \
                 {} W, expected {expected_load} W",
                step.load
            );
            assert!(step.load.is_finite(), "the reported load must be finite");

            // Bounded node temperatures (the held air node is a boundary
            // of the network too).
            for (k, (label, temp)) in step
                .t_surface
                .iter()
                .map(|t| ("surface", t))
                .chain(step.t_mass.iter().map(|t| ("depth", t)))
                .enumerate()
            {
                assert!(
                    temp.is_finite() && *temp >= lo - 1.0 && *temp <= hi + 1.0,
                    "node {k} ({label}) temperature {temp} °C is outside the \
                     physical band [{lo}, {hi}] °C at hour {t}"
                );
            }
        }
    }
}
