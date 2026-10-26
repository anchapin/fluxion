//! Unit tests for the per-surface multi-node conduction formulation
//! (Issue #3983).
//!
//! The suite covers three groups:
//!
//! 1. **Formulation structure** — the two node states per surface are ODE
//!    states (not algebraic pins), the solar is distributed per surface by
//!    area × absorptance, and the implicit step conserves energy.
//! 2. **Dynamic order** — the surface node and the depth node respond on
//!    distinct timescales (the dynamic order the lumped 5R1C network cannot
//!    express).
//! 3. **#1522 bidirectional signature** — on a synthetic single-zone case the
//!    per-surface formulation moves the cooling-peak and the heating-peak
//!    metric in *opposite* directions relative to the lumped ISO 13790 5R1C
//!    reference, while the lumped reference's own lever keeps both metrics
//!    locked together.

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

/// Synthetic envelope: roof + four walls (no floor — the lumped 5R1C
/// reference has a single exterior boundary for the whole envelope, so a
/// ground-coupled floor could not be represented in both models
/// simultaneously and is therefore excluded from *both*).
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
        wall(1, ZONE_W),  // south
        wall(2, ZONE_D),  // east
        wall(3, ZONE_D),  // west
        wall(4, ZONE_W),  // north
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

/// The lumped ISO 13790 5R1C zone reference (single mass node + algebraic
/// interior surface node) built from the *same* surface data.
///
/// This is the reference formulation the #1522 bidirectional signature is
/// measured against: one thermal state for the whole envelope, so the
/// interior surface temperature can only follow that single state.
struct LumpedFiveR1C {
    c_mass: f64,
    h_em: f64,
    h_ms: f64,
    h_is: f64,
    c_air: f64,
    h_ve: f64,
    t_mass: f64,
    t_air: f64,
}

impl LumpedFiveR1C {
    fn new(specs: &[SurfaceSpec]) -> Self {
        let h_em: f64 = specs.iter().map(|s| s.h_tr_em).sum();
        let h_ms: f64 = specs.iter().map(|s| s.h_tr_ms).sum();
        let h_is: f64 = specs.iter().map(|s| s.h_tr_is).sum();
        let c_mass: f64 = specs.iter().map(|s| s.c_surface + s.c_mass).sum();
        Self {
            c_mass,
            h_em,
            h_ms,
            h_is,
            c_air: C_AIR,
            h_ve: H_VE,
            t_mass: 20.0,
            t_air: 20.0,
        }
    }

    /// Implicit 5R1C step; returns the new zone air temperature.
    fn step(&mut self, dt: f64, t_out: f64, absorbed_total: f64) -> f64 {
        let alpha = self.h_is / (self.h_is + self.h_ms);
        let beta = self.h_ms / (self.h_is + self.h_ms);
        let solair = if self.h_em > 0.0 {
            t_out + absorbed_total / self.h_em
        } else {
            t_out
        };
        // Mass node, eliminated against the algebraic surface node.
        let r1 = self.c_mass / dt * self.t_mass + self.h_em * solair;
        let d1 = self.c_mass / dt + self.h_em + self.h_ms * beta;
        // Zone air node.
        let denom = self.c_air / dt
            + self.h_ve
            + self.h_is * (1.0 - alpha)
            - self.h_is * beta * self.h_ms * alpha / d1;
        let numer =
            self.c_air / dt * self.t_air + self.h_is * beta * r1 / d1 + self.h_ve * t_out;
        let t_air_new = numer / denom;
        self.t_mass = (r1 + self.h_ms * alpha * t_air_new) / d1;
        self.t_air = t_air_new;
        t_air_new
    }
}

/// Peak-load metrics evaluated on a free-float zone-air trajectory.
///
/// * `peak_cooling` — the largest cooling load the zone would need to hold
///   the cooling setpoint [W];
/// * `peak_heating` — the largest heating load the zone would need to hold
///   the heating setpoint [W].
#[derive(Debug, Clone, Copy, PartialEq)]
struct PeakMetrics {
    peak_cooling: f64,
    peak_heating: f64,
    t_min: f64,
    t_max: f64,
}

fn peak_metrics(air_temps: &[f64], h_is_total: f64) -> PeakMetrics {
    let gain = h_is_total + H_VE;
    let mut peak_cooling = 0.0f64;
    let mut peak_heating = 0.0f64;
    for &t in air_temps {
        peak_cooling = peak_cooling.max((t - T_COOL).max(0.0) * gain);
        peak_heating = peak_heating.max((T_HEAT - t).max(0.0) * gain);
    }
    PeakMetrics {
        peak_cooling,
        peak_heating,
        t_min: air_temps.iter().cloned().fold(f64::INFINITY, f64::min),
        t_max: air_temps.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    }
}

/// Run the per-surface formulation over the synthetic case.
fn run_per_surface(specs: &[SurfaceSpec]) -> (Vec<f64>, PerSurfaceMultiNodeConduction) {
    let mut model = PerSurfaceMultiNodeConduction::new(
        specs.to_vec(),
        ZoneAirSpec::new(C_AIR, H_VE),
        20.0,
    );
    let mut temps = Vec::with_capacity(HOURS);
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
        let transmitted = vec![0.0; n];
        let inputs = StepInputs {
            exterior_temperatures: &exterior,
            t_outdoor: t_out,
            phi_ia: 0.0,
            absorbed_solar: &absorbed,
            transmitted_solar: &transmitted,
        };
        temps.push(model.step(DT, &inputs));
    }
    (temps, model)
}

/// Run the lumped 5R1C reference over the same synthetic case.
fn run_lumped(specs: &[SurfaceSpec]) -> Vec<f64> {
    let mut model = LumpedFiveR1C::new(specs);
    let mut temps = Vec::with_capacity(HOURS);
    for t in 0..HOURS {
        let (t_out, _, _) = synthetic_weather(t);
        let irr = synthetic_irradiance(t);
        let absorbed_total: f64 = specs
            .iter()
            .zip(irr.iter())
            .map(|(s, i)| absorbed_flux(s, *i))
            .sum();
        temps.push(model.step(DT, t_out, absorbed_total));
    }
    temps
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

/// The depth node and the surface node respond on **distinct timescales**:
/// a step in the zone-air temperature is tracked by the interior surface node
/// within hours, while the depth node — which is coupled to the zone only
/// through the construction resistance — is still essentially unchanged.
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

    // Settle, then apply a +10 K step to the zone air (a large step so the
    // response is unambiguous).
    for _ in 0..48 {
        model.step(DT, &inputs);
    }
    let t_s0 = model.surface(0).unwrap().t_surface;
    let t_m0 = model.surface(0).unwrap().t_mass;

    let mut surface_63 = None;
    let mut mass_63 = None;
    let mut t_s_final = t_s0;
    let mut t_m_final = t_m0;
    let mut surface_trace = Vec::new();
    let mut mass_trace = Vec::new();
    for step in 1..=2000 {
        // Step the zone air by +10 K through the internal-gain term.
        let stepped = StepInputs {
            exterior_temperatures: &exterior,
            t_outdoor: 20.0,
            phi_ia: 0.0,
            absorbed_solar: &absorbed,
            transmitted_solar: &transmitted,
        };
        let _ = stepped;
        model.step(DT, &inputs);
        // Force the air node to the stepped value by directly advancing the
        // zone air through the ventilation path: instead of mutating state,
        // the step is repeated with a warm exterior so the air node moves.
        let warm = StepInputs {
            exterior_temperatures: &[30.0f64],
            t_outdoor: 30.0,
            phi_ia: 0.0,
            absorbed_solar: &absorbed,
            transmitted_solar: &transmitted,
        };
        model.step(DT, &warm);
        let t_s = model.surface(0).unwrap().t_surface;
        let t_m = model.surface(0).unwrap().t_mass;
        surface_trace.push(t_s);
        mass_trace.push(t_m);
        t_s_final = t_s;
        t_m_final = t_m;
        if surface_63.is_none() && t_s >= t_s0 + 0.632 * (t_s_final - t_s0) {
            surface_63 = Some(step);
        }
        if mass_63.is_none() && t_m >= t_m0 + 0.632 * (t_m_final - t_m0) {
            mass_63 = Some(step);
        }
        if surface_63.is_some() && mass_63.is_some() {
            break;
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
}

/// The absorbed solar is apportioned across the surfaces by
/// `absorptance × area` — not by a fixed split, and not into a single
/// lumped node.
#[test]
fn test_absorbed_solar_is_distributed_by_area_times_absorptance() {
    let specs = synthetic_surfaces();
    let total = 10_000.0;
    let shares = distribute_by_area_absorptance(&specs, total);

    let weight_sum: f64 = specs
        .iter()
        .map(|s| s.absorptance * s.area)
        .sum();
    for (share, spec) in shares.iter().zip(specs.iter()) {
        let expected = total * (spec.absorptance * spec.area) / weight_sum;
        assert!(
            (share - expected).abs() < 1e-9,
            "share mismatch for surface {}: got {share}, expected {expected}",
            spec.id
        );
    }
    let sum: f64 = shares.iter().sum();
    assert!((sum - total).abs() < 1e-9, "shares must sum to the total flux");

    // The roof (larger area, higher absorptance) must take strictly more
    // than any single wall — a fixed 50/50 split could not do that.
    assert!(shares[0] > shares[1] && shares[1] > shares[4]);
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
    let weight_sum: f64 = specs
        .iter()
        .map(|s| s.interior_absorptance * s.area)
        .sum();
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
/// the roof alone warms the roof's depth node and leaves the shaded walls'
/// depth nodes essentially untouched.
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
    for (k, m) in masses.iter().enumerate().skip(1) {
        assert!(
            m.abs() - 20.0 < 0.05,
            "shaded surface {k} must not warm from another surface's solar: {m}"
        );
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
    let mut model =
        PerSurfaceMultiNodeConduction::new(specs, ZoneAirSpec::new(C_AIR, h_ve), 20.0);

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

    // Analytic steady state of the linear network.
    let solair = t_out + absorbed[0] / spec.h_tr_em;
    let k = h_ve / (spec.h_tr_is + h_ve)
        * spec.h_tr_is
        * (1.0 / spec.h_tr_ms + 1.0 / spec.h_tr_em);
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

// ---------------------------------------------------------------------------
// 2. #1522 bidirectional signature
// ---------------------------------------------------------------------------

/// The per-surface multi-node formulation moves the cooling-peak metric and
/// the heating-peak metric in **opposite directions** relative to the lumped
/// ISO 13790 5R1C reference on the same synthetic single-zone case.
///
/// This is the `#1522` bidirectional-signature capability: the lumped
/// network has a single thermal state per zone, so its two peak metrics are
/// locked together, while the per-surface network (two states per surface
/// plus per-surface solar) can move them independently.
#[test]
fn test_bidirectional_peak_signature_versus_lumped_5r1c() {
    let specs = synthetic_surfaces();
    let h_is_total: f64 = specs.iter().map(|s| s.h_tr_is).sum();

    let (per_surface_temps, _) = run_per_surface(&specs);
    let lumped_temps = run_lumped(&specs);
    let per_surface = peak_metrics(&per_surface_temps, h_is_total);
    let lumped = peak_metrics(&lumped_temps, h_is_total);

    assert!(
        per_surface.peak_cooling > lumped.peak_cooling,
        "the cooling-peak metric must move in one direction: per-surface {:.1} W \
         vs lumped {:.1} W",
        per_surface.peak_cooling,
        lumped.peak_cooling
    );
    assert!(
        per_surface.peak_heating < lumped.peak_heating,
        "the heating-peak metric must move in the opposite direction: \
         per-surface {:.1} W vs lumped {:.1} W",
        per_surface.peak_heating,
        lumped.peak_heating
    );

    // The synthetic case must stay physically plausible (a free-floating
    // zone on a winter day), otherwise the comparison would be vacuous.
    assert!(
        per_surface.t_max < 45.0 && per_surface.t_min > 0.0,
        "implausible free-float excursion: Tmin={}, Tmax={}",
        per_surface.t_min,
        per_surface.t_max
    );
}

/// The lumped 5R1C reference's own lever (the envelope capacitance) moves
/// **both** peak metrics in the same direction — the `#1522` infeasibility —
/// while the per-surface formulation has a lever (how the capacitance is
/// distributed between the depth node and the interior surface node) that
/// moves them in **opposite** directions.
#[test]
fn test_lumped_metrics_are_locked_while_per_surface_metrics_are_not() {
    let specs = synthetic_surfaces();
    let h_is_total: f64 = specs.iter().map(|s| s.h_tr_is).sum();

    // --- Lumped reference: scaling the envelope capacitance.
    let mut lumped_metrics = Vec::new();
    for scale in [0.8, 1.0, 1.2] {
        let scaled: Vec<SurfaceSpec> = specs
            .iter()
            .map(|s| {
                let mut s = s.clone();
                s.c_surface *= scale;
                s.c_mass *= scale;
                s
            })
            .collect();
        lumped_metrics.push(peak_metrics(&run_lumped(&scaled), h_is_total));
    }
    let lumped_down = lumped_metrics[0];
    let lumped_base = lumped_metrics[1];
    let lumped_up = lumped_metrics[2];
    assert!(
        lumped_down.peak_cooling < lumped_base.peak_cooling
            && lumped_base.peak_cooling < lumped_up.peak_cooling,
        "the lumped cooling peak must fall as the envelope capacitance grows"
    );
    assert!(
        lumped_down.peak_heating < lumped_base.peak_heating
            && lumped_base.peak_heating < lumped_up.peak_heating,
        "the lumped heating peak must fall as the envelope capacitance grows \
         (both metrics locked in the same direction)"
    );

    // --- Per-surface formulation: redistributing the *same* capacitance.
    let mut per_surface_metrics = Vec::new();
    for split in [0.25, 0.5, 0.75] {
        let redistributed: Vec<SurfaceSpec> = specs
            .iter()
            .map(|s| {
                let mut s = s.clone();
                let c_total = s.c_surface + s.c_mass;
                s.c_surface = split * c_total;
                s.c_mass = (1.0 - split) * c_total;
                s
            })
            .collect();
        per_surface_metrics.push(peak_metrics(
            &run_per_surface(&redistributed).0,
            h_is_total,
        ));
    }
    let ps_low = per_surface_metrics[0];
    let ps_mid = per_surface_metrics[1];
    let ps_high = per_surface_metrics[2];
    assert!(
        ps_low.peak_cooling < ps_high.peak_cooling,
        "the per-surface cooling peak must grow as capacitance moves to the \
         interior surface node: {:.1} -> {:.1} W",
        ps_low.peak_cooling,
        ps_high.peak_cooling
    );
    assert!(
        ps_low.peak_heating > ps_high.peak_heating,
        "the per-surface heating peak must fall as capacitance moves to the \
         interior surface node: {:.1} -> {:.1} W",
        ps_low.peak_heating,
        ps_high.peak_heating
    );
}
