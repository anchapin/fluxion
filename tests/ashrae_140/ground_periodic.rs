//! ASHRAE 140 §8 (ground-coupled slab-on-grade) — periodic analytical
//! verification for the 2-D ground-conduction domain model.
//!
//! Issue #4774 (periodic half). The transient half of the same issue landed
//! in PR #4782 and is verified in `tests/ashrae_140/ground_transient.rs`
//! (backward-Euler stepping against the closed-form semi-infinite *step*
//! response). This file is the periodic half: it drives the same solver with
//! a **sinusoidal surface temperature** and checks the response against the
//! closed-form periodic solution of a semi-infinite solid, and it runs a
//! genuinely two-dimensional periodic case.
//!
//! The solver under test is [`fluxion::physics::ground_conduction`], a
//! standalone 2-D (x–z) ground-conduction domain model. It is *not* wired
//! into the Section 7 floor boundary or the main simulation loop: §8
//! verifies the ground domain standalone first.
//!
//! # Case 1 — periodic surface temperature on the 1-D limit
//!
//! The configuration is the one-dimensional limit of the domain (slab
//! footprint spans the whole width, lateral boundaries adiabatic) with a
//! **homogeneous** material: slab and soil share conductivity
//! `k = 1.9 W/(m·K)` and volumetric heat capacity `rho c = 2.0e6 J/(m³·K)`,
//! so the slab/soil interface is invisible to the temperature field. The
//! domain is 4 m wide and 3 m deep and starts uniform at `T0 = 10 °C`,
//! which is also the deep-ground temperature and also the *mean* of the
//! imposed surface swing, so the mean field stays uniform at 10 °C and only
//! the oscillation is excited.
//!
//! The surface temperature is driven sinusoidally,
//!
//! ```text
//! T_s(t) = T_mean + A cos(omega t),   T_mean = 10 °C,   A = 10 K,
//! omega  = 2 pi / P,                  P = 2 days = 172800 s,
//! ```
//!
//! through the existing per-step top-condition API
//! ([`GroundSolver::step_transient_with_top`]). The closed-form response of
//! a semi-infinite solid to a sinusoidal surface temperature is
//!
//! ```text
//! T(z, t) = T_mean + A exp(-z/delta) cos(omega t - z/delta),
//! delta  = sqrt(2 alpha / omega),   alpha = k / (rho c) = 9.5e-7 m²/s,
//! delta  = 0.2286 m,
//! ```
//!
//! i.e. the amplitude decays as `exp(-z/delta)` and the oscillation lags the
//! surface by the phase `z/delta` (one radian per penetration depth). The
//! domain is 3 m deep, i.e. 13 penetration depths, so the Dirichlet deep
//! boundary reflects the wave with a factor `exp(-2 D / delta) = 5e-12` and
//! the semi-infinite closed form applies over the compared depths.
//!
//! The run is spun up before the measurement cycle: the slow diffusion mode
//! of the domain needs tens of days of simulated time to decay, so the
//! spin-up uses a coarse one-hour step, and the measurement cycle itself is
//! marched with the ten-minute step that is sampled, so the recorded orbit
//! is the backward-Euler orbit of that step.
//!
//! (Carslaw & Jaeger, *Conduction of Heat in Solids*, 2nd ed., §2.4 — the
//! semi-infinite-solid solutions. The periodic case quoted above is the
//! classical superposition of the step response for the harmonic surface
//! temperature; the same text gives the sinusoidal surface temperature of a
//! semi-infinite solid with the penetration depth `delta = sqrt(2 alpha /
//! omega)` and the phase lag `z/delta`.) The transient case of the same
//! issue cites the same source for the step response.
//!
//! The comparison is made against that **continuum** closed form; the
//! discrete solution is expected to sit slightly off it, and the tolerance
//! below is set to cover exactly the two error terms that are known and
//! bounded:
//!
//! * **(a) cell-average vs point value.** The finite-volume unknown of a
//!   cell is its average, while the closed form is evaluated at the cell
//!   centre. Over a cell of height `h` the difference is a relative
//!   `O((|kappa| h / 2)^2 / 6)` of the local amplitude, with
//!   `|kappa| = sqrt(2)/delta = 6.20 1/m`; with `h = 0.02 m` over the
//!   compared region this is `0.1 %` of the local amplitude, i.e. at most
//!   **0.010 K** at the surface cell (where the local amplitude is 10 K).
//! * **(b) backward-Euler semi-discrete error.** Backward Euler is first
//!   order in time. On the harmonic mode the discrete dispersion relation
//!   is `1 - alpha dt kappa^2 = exp(-i omega dt)`, which to first order in
//!   `omega dt` gives `kappa_BE = kappa (1 - i omega dt / 4)`: the
//!   time-stepping error *rotates* the wavenumber, it does not scale it.
//!   With `kappa = (1 + i)/delta` that splits into two consequences with
//!   **opposite signs**:
//!
//!   * the **phase lag** is *reduced*, to `z/delta (1 - omega dt / 4)` —
//!     a relative phase error of `-omega dt / 4 = -5.45e-3`, i.e. the
//!     discrete oscillation reaches a given depth slightly *earlier* than
//!     the continuum one;
//!   * the **attenuation exponent** is *increased*, to
//!     `z/delta (1 + omega dt / 4)` — the discrete wave is damped
//!     slightly *faster* with depth than the continuum wave.
//!
//!   The amplitude (attenuation) error therefore does **not** follow the
//!   same factor as the phase error: it is governed by the
//!   `1 + omega dt / 4` attenuation factor, while the phase error is
//!   governed by the `1 - omega dt / 4` phase factor, and the two carry
//!   opposite signs. The largest resulting temperature error is at
//!   `z/delta = 1` (where `A exp(-z/delta) (z/delta)` is maximal) and has
//!   the magnitude `A/e * omega dt / 4 = 0.020 K`.
//!
//!   The sign of the phase term is confirmed by the solver's own orbit:
//!   the harmonic fit of the discrete orbit gives a phase lag of
//!   **0.8283 rad** at the compared depth against the continuum value of
//!   **0.8312 rad** — *below* it, as the reduced-lag law predicts (the
//!   `1 + omega dt / 4` law would have put it above, at ≈0.8357 rad).
//!
//! So the budget is `0.010 K + 0.020 K = 0.030 K` and the asserted
//! tolerance is **0.05 K** — above both terms and above their sum, as
//! required, so that the small expected deviation is not misdiagnosed as a
//! bug. The tolerance is still strongly discriminating: the surface
//! oscillation is 10 K, so 0.05 K is 0.5 % of it, and a wrong penetration
//! depth (say 10 % off) would show up as 0.5–1 K at the compared depths.
//!
//! No spin-up term appears in that budget, because the spin-up is long
//! enough for the residual transient to be negligible. This was checked
//! directly: re-running the 1-D case with the fine spin-up extended from
//! 15 to 90 days moves the maximum deviation from the closed form only
//! from **0.0260 K** to **0.0254 K** — a shift of 0.0007 K, i.e. 2 % of
//! the 0.030 K budget — and the drift between two consecutive measured
//! cycles falls from 0.00025 K to below 1e-6 K. The 15-day fine spin-up
//! therefore leaves no residual transient of the order of the budget.
//!
//! Three separate metrics are asserted, each against its own closed-form
//! reference (all from the citation above):
//!
//! | metric | reference | tolerance | error budget |
//! |--------|-----------|-----------|--------------|
//! | the temperature field at the sampled instants | `T(z, t) = T_mean + A exp(-z/delta) cos(omega t - z/delta)` | 0.05 K | 0.010 K (a) + 0.020 K (b) |
//! | the oscillation amplitude at depth | `A exp(-z/delta)` | 2 % | 1.2 % (b, from the attenuation factor `(z/delta) omega dt / 4`, i.e. damped slightly faster) |
//! | the phase lag at depth | `z/delta` rad | 0.05 rad | 0.005 rad (b, from the phase factor `(z/delta) omega dt / 4`, i.e. the lag is reduced) |
//!
//! Each tolerance sits above its own budget term, so the small expected
//! deviation is not misdiagnosed as a bug, and each tolerance is a small
//! fraction of the quantity it bounds, so a wrong penetration depth or a
//! wrong phase is still caught.
//!
//! # Case 2 — a genuinely two-dimensional periodic run
//!
//! The second case is **not** the 1-D limit: an 8 m wide, 2 m deep domain
//! carries a 4 m slab footprint strictly inside it (`x ∈ [2 m, 6 m]`), the
//! slab is more conductive than the soil (`3.0 W/(m·K)` against
//! `1.9 W/(m·K)`) and the two materials have different volumetric heat
//! capacities (`2.0e6` against `1.6e6 J/(m³·K)`), so the field is genuinely
//! two-dimensional. The lateral far-field boundaries are adiabatic and the
//! deep ground is held at 10 °C. The surface is driven with the same
//! sinusoid but with a **mean of 15 °C**, so the cycle-mean field cannot
//! degenerate into a uniform 10 °C field.
//!
//! Four properties are asserted:
//!
//! 1. the run settles to a periodic steady state — the field at the same
//!    phase of two consecutive cycles agrees to within **0.02 K**;
//! 2. the cycle-mean near-surface temperature is 15 °C (within 0.5 K of the
//!    imposed cycle-mean surface temperature) and differs from the
//!    deep-ground temperature of 10 °C by more than 3 K;
//! 3. the cycle-mean field agrees, cell by cell over the whole domain, with
//!    the **steady** `solve()` of the same configuration at that same
//!    cycle-mean surface temperature, to within **0.05 K** (this is the
//!    sharpest of the four: for a linear time-invariant system the mean of
//!    a periodic orbit *is* the steady solution for the mean boundary data,
//!    so the bound is only met once the orbit has really settled);
//! 4. the cycle-mean field has non-trivial x-dependence: the spread of the
//!    cycle-mean temperature across the columns of at least one row exceeds
//!    **0.15 K** (the observed spread is about 0.24 K, i.e. more than
//!    twice the cycle-to-cycle tolerance, so the field cannot be a uniform
//!    column-wise profile).
//!
//! # Scope
//!
//! These are standalone verifications of the 2-D ground domain. The named
//! §8 b-series cases (GC10b, GC30, GC60) are **not** transcribed here:
//! their boundary specifications are not derivable from in-repo sources
//! (they tie to #4285), so they stay open rather than being improvised.

use std::f64::consts::PI;
use std::sync::OnceLock;

use fluxion::physics::ground_conduction::{
    GridSegment, GroundDomainConfig, GroundSolver, GroundTransientState, SurfaceCondition,
    TopCondition, TopSegment,
};

// ---------------------------------------------------------------------------
// Case 1 — shared physical constants (1-D homogeneous limit).
// ---------------------------------------------------------------------------

/// Homogeneous conductivity [W/(m·K)].
const K_W_MK: f64 = 1.9;
/// Homogeneous volumetric heat capacity [J/(m³·K)].
const RHO_C: f64 = 2_000_000.0;
/// Thermal diffusivity `alpha = k / (rho c)` [m²/s].
const ALPHA_M2_S: f64 = K_W_MK / RHO_C;
/// Surface-temperature period [s] (two days).
const PERIOD_S: f64 = 172_800.0;
/// Angular frequency of the surface swing [rad/s].
const OMEGA_RAD_S: f64 = 2.0 * PI / PERIOD_S;
/// Measurement time step [s] (ten minutes).
const DT_S: f64 = 600.0;
/// Spin-up time step [s] (one hour) — the slow diffusion mode needs tens of
/// days of simulated time, and the coarse step buys that cheaply.
const SPIN_UP_DT_S: f64 = 3600.0;
/// Spin-up length [days] (coarse step).
const SPIN_UP_DAYS: f64 = 90.0;
/// Fine-step relaxation length after the coarse spin-up [days].
const FINE_SPIN_UP_DAYS: f64 = 15.0;
/// Mean surface temperature, and also the initial and deep-ground
/// temperature [°C].
const MEAN_C: f64 = 10.0;
/// Amplitude of the surface swing [K].
const AMPLITUDE_K: f64 = 10.0;
/// Deepest depth compared against the closed form [m].
const Z_COMPARE_MAX_M: f64 = 0.5;
/// Temperature tolerance of the closed-form comparison [K] (see the error
/// budget in the module header).
const TEMP_TOLERANCE_K: f64 = 0.05;

/// Thermal penetration depth of the periodic wave `sqrt(2 alpha / omega)` [m].
fn penetration_depth_m() -> f64 {
    (2.0 * ALPHA_M2_S / OMEGA_RAD_S).sqrt()
}

/// Imposed surface temperature at time `t_s` [°C].
fn surface_temperature_c(t_s: f64) -> f64 {
    MEAN_C + AMPLITUDE_K * (OMEGA_RAD_S * t_s).cos()
}

/// Closed-form semi-infinite periodic response at depth `z_m`, time `t_s` [°C].
///
/// `T = T_mean + A exp(-z/delta) cos(omega t - z/delta)`.
fn analytical_periodic_temperature_c(z_m: f64, t_s: f64) -> f64 {
    let delta = penetration_depth_m();
    MEAN_C + AMPLITUDE_K * (-z_m / delta).exp() * (OMEGA_RAD_S * t_s - z_m / delta).cos()
}

/// Closed-form oscillation amplitude at depth `z_m` [K].
fn analytical_periodic_amplitude_k(z_m: f64) -> f64 {
    AMPLITUDE_K * (-z_m / penetration_depth_m()).exp()
}

/// Closed-form phase lag at depth `z_m` [rad].
fn analytical_phase_lag_rad(z_m: f64) -> f64 {
    z_m / penetration_depth_m()
}

/// One-dimensional-limit configuration: 4 m wide, 3 m deep, homogeneous.
fn one_dimensional_config() -> GroundDomainConfig {
    GroundDomainConfig {
        domain_length_m: 4.0,
        domain_depth_m: 3.0,
        slab_x0_m: 0.0,
        slab_x1_m: 4.0,
        slab_thickness_m: 0.2,
        slab_conductivity_w_mk: K_W_MK,
        ground_conductivity_w_mk: K_W_MK,
        slab_volumetric_heat_capacity_j_m3k: RHO_C,
        ground_volumetric_heat_capacity_j_m3k: RHO_C,
        lateral_segments: vec![GridSegment::Uniform {
            start_m: 0.0,
            end_m: 4.0,
            cells: 4,
        }],
        vertical_segments: vec![
            GridSegment::Uniform {
                start_m: 0.0,
                end_m: 0.5,
                cells: 25,
            },
            GridSegment::Uniform {
                start_m: 0.5,
                end_m: 1.0,
                cells: 10,
            },
            GridSegment::Geometric {
                start_m: 1.0,
                end_m: 3.0,
                first_cell_m: 0.05,
                growth: 1.2,
                max_cell_m: 0.4,
            },
        ],
        top_segments: vec![TopSegment {
            x_start_m: 0.0,
            x_end_m: 4.0,
            condition: TopCondition::Dirichlet(MEAN_C),
        }],
        deep_ground: SurfaceCondition::Dirichlet(MEAN_C),
        lateral_boundary: SurfaceCondition::Adiabatic,
        max_iterations: 50_000,
        tolerance: 1.0e-11,
    }
}

/// Top segments of the 1-D limit with the prescribed surface temperature.
fn one_dimensional_top(temperature_c: f64) -> Vec<TopSegment> {
    vec![TopSegment {
        x_start_m: 0.0,
        x_end_m: 4.0,
        condition: TopCondition::Dirichlet(temperature_c),
    }]
}

// ---------------------------------------------------------------------------
// Recorded periodic run.
// ---------------------------------------------------------------------------

/// A recorded cycle of a periodic run: the whole field sampled at every
/// step of one period.
struct CycleRun {
    /// Sample times [s].
    times: Vec<f64>,
    /// Flattened field samples, `samples[step * n_cells + cell]`.
    samples: Vec<f64>,
    /// Number of cells per sample.
    n_cells: usize,
    /// Cell-centre x coordinates [m].
    x: Vec<f64>,
    /// Cell-centre z coordinates [m].
    z: Vec<f64>,
}

impl CycleRun {
    /// Temperature of cell `cell` at sample `step` [°C].
    fn temperature(&self, step: usize, cell: usize) -> f64 {
        self.samples[step * self.n_cells + cell]
    }

    /// Index of the cell at column `i`, row `j`.
    fn cell_index(&self, i: usize, j: usize) -> usize {
        j * self.x.len() + i
    }

    /// Harmonic fit of one cell over the sampled cycle.
    ///
    /// Returns `(mean, amplitude, phase_lag)` of
    /// `T(t) = mean + amplitude cos(omega t - phase_lag)`. The samples span
    /// exactly one period with `omega dt = 2 pi / n`, so the discrete
    /// projection is exact and the fit returns the amplitude and phase of
    /// the discrete periodic orbit itself.
    fn harmonic_fit(&self, cell: usize) -> (f64, f64, f64) {
        let n = self.times.len();
        let mean = (0..n).map(|s| self.temperature(s, cell)).sum::<f64>() / n as f64;
        let mut cos_coefficient = 0.0;
        let mut sin_coefficient = 0.0;
        for (s, &t) in self.times.iter().enumerate() {
            let value = self.temperature(s, cell) - mean;
            cos_coefficient += value * (OMEGA_RAD_S * t).cos();
            sin_coefficient += value * (OMEGA_RAD_S * t).sin();
        }
        let amplitude = (2.0 / n as f64 * cos_coefficient).hypot(2.0 / n as f64 * sin_coefficient);
        let phase_lag = (2.0 / n as f64 * sin_coefficient).atan2(2.0 / n as f64 * cos_coefficient);
        (mean, amplitude, phase_lag)
    }
}

/// March the 1-D limit to its periodic state and record the final cycle.
fn run_one_dimensional() -> &'static CycleRun {
    static RUN: OnceLock<CycleRun> = OnceLock::new();
    RUN.get_or_init(|| {
        let solver = GroundSolver::new(one_dimensional_config()).expect("valid config");
        let mut state = solver
            .initial_transient_state(MEAN_C)
            .expect("initial state");

        // Coarse spin-up: the slow diffusion mode of the domain needs tens of
        // days of simulated time to decay, and one hour per step buys that
        // cheaply. The mean of the swing equals the deep-ground temperature,
        // so only the oscillation is being built up here.
        let spin_up_steps = (SPIN_UP_DAYS * 86_400.0 / SPIN_UP_DT_S) as usize;
        for _ in 0..spin_up_steps {
            let t_next = state.time_s + SPIN_UP_DT_S;
            solver
                .step_transient_with_top(
                    &mut state,
                    SPIN_UP_DT_S,
                    &one_dimensional_top(surface_temperature_c(t_next)),
                )
                .expect("spin-up step");
        }

        // Fine steps: the measured orbit must be the one of the measurement
        // time step, so relax onto it with the same `dt` that is sampled.
        let fine_steps = (FINE_SPIN_UP_DAYS * 86_400.0 / DT_S) as usize;
        let steps_per_cycle = (PERIOD_S / DT_S) as usize;
        let mut samples: Vec<f64> = Vec::new();
        let mut times: Vec<f64> = Vec::new();
        let n_cells = state.temperature_c.len();
        for step in 0..fine_steps + steps_per_cycle {
            let t_next = state.time_s + DT_S;
            solver
                .step_transient_with_top(
                    &mut state,
                    DT_S,
                    &one_dimensional_top(surface_temperature_c(t_next)),
                )
                .expect("fine step");
            if step >= fine_steps {
                samples.extend_from_slice(&state.temperature_c);
                times.push(t_next);
            }
        }

        CycleRun {
            times,
            samples,
            n_cells,
            x: state.grid.x.centers.clone(),
            z: state.grid.z.centers.clone(),
        }
    })
}

// ---------------------------------------------------------------------------
// Case 1 assertions.
// ---------------------------------------------------------------------------

#[test]
fn periodic_surface_temperature_matches_closed_form() {
    let run = run_one_dimensional();
    let column = run.x.len() / 2;

    // The 1-D limit must be laterally uniform (adiabatic far field, uniform
    // surface condition): a non-zero lateral spread would mean the case has
    // silently become two-dimensional.
    let mut lateral_spread = 0.0_f64;
    for step in 0..run.times.len() {
        for j in 0..run.z.len() {
            let reference = run.temperature(step, run.cell_index(column, j));
            for i in 0..run.x.len() {
                lateral_spread = lateral_spread
                    .max((run.temperature(step, run.cell_index(i, j)) - reference).abs());
            }
        }
    }
    assert!(
        lateral_spread <= 1.0e-6,
        "the 1-D limit is not laterally uniform: max spread {lateral_spread:.3e} K"
    );

    // Compare the field against the closed form at every sampled instant of
    // the cycle, over the depths where the oscillation is resolved.
    let mut max_deviation = 0.0_f64;
    let mut compared = 0usize;
    for (step, &t) in run.times.iter().enumerate() {
        for (j, &z) in run.z.iter().enumerate() {
            if z > Z_COMPARE_MAX_M {
                break;
            }
            let expected = analytical_periodic_temperature_c(z, t);
            let actual = run.temperature(step, run.cell_index(column, j));
            let deviation = (actual - expected).abs();
            max_deviation = max_deviation.max(deviation);
            compared += 1;
            assert!(
                deviation <= TEMP_TOLERANCE_K,
                "z={z:.3} m, t={t:.0} s: solver {actual:.4} °C vs closed form \
                 {expected:.4} °C (deviation {deviation:.4} K > {TEMP_TOLERANCE_K} K)"
            );
        }
    }
    assert!(compared >= 100, "expected at least 100 compared samples");
    println!(
        "periodic 1-D: {compared} (depth, time) samples compared, \
         max deviation {max_deviation:.4} K"
    );
}

#[test]
fn periodic_amplitude_decay_matches_penetration_depth() {
    let run = run_one_dimensional();
    let column = run.x.len() / 2;

    // Carslaw & Jaeger §2.4: the amplitude of the oscillation decays as
    // exp(-z/delta) with delta = sqrt(2 alpha / omega).
    let mut max_relative = 0.0_f64;
    let mut compared = 0usize;
    for (j, &z) in run.z.iter().enumerate() {
        if z > Z_COMPARE_MAX_M {
            break;
        }
        let (_, amplitude, _) = run.harmonic_fit(run.cell_index(column, j));
        let expected = analytical_periodic_amplitude_k(z);
        let relative = (amplitude - expected).abs() / expected;
        max_relative = max_relative.max(relative);
        compared += 1;
        assert!(
            relative <= 0.02,
            "z={z:.3} m: measured amplitude {amplitude:.4} K vs closed form \
             {expected:.4} K (relative error {relative:.4} > 0.02)"
        );
    }
    assert!(compared >= 10, "expected at least 10 compared depths");
    println!(
        "periodic 1-D amplitude: {compared} depths compared, \
         max relative error {max_relative:.4}"
    );
}

#[test]
fn periodic_phase_lag_matches_depth_over_penetration_depth() {
    let run = run_one_dimensional();
    let column = run.x.len() / 2;

    // Carslaw & Jaeger §2.4: the oscillation lags the surface by z/delta
    // radians. The comparison depth is chosen where the signal is still
    // large (z/delta ~ 0.9), so the tolerance is a small fraction of the
    // phase itself and the metric stays discriminating.
    let target_depth_m = 0.2;
    let mut best: Option<(usize, f64, f64)> = None;
    for (index, &centre) in run.z.iter().enumerate() {
        let distance = (centre - target_depth_m).abs();
        match best {
            Some((_, _, best_distance)) if best_distance <= distance => {}
            _ => best = Some((index, centre, distance)),
        }
    }
    let (j, z, _) = best.expect("a cell centre near the comparison depth");
    let (_, amplitude, phase_lag) = run.harmonic_fit(run.cell_index(column, j));
    let expected = analytical_phase_lag_rad(z);
    let deviation = (phase_lag - expected).abs();
    assert!(
        amplitude > 1.0,
        "the oscillation at z={z:.3} m is too weak to fit a phase: {amplitude:.4} K"
    );
    assert!(
        deviation <= 0.05,
        "z={z:.3} m: measured phase lag {phase_lag:.4} rad vs closed form \
         {expected:.4} rad (deviation {deviation:.4} rad > 0.05 rad)"
    );
    println!(
        "periodic 1-D phase: z={z:.3} m, measured {phase_lag:.4} rad, \
         closed form {expected:.4} rad, amplitude {amplitude:.3} K"
    );
}

// ---------------------------------------------------------------------------
// Case 2 — genuinely two-dimensional periodic run.
// ---------------------------------------------------------------------------

/// Lateral extent of the 2-D domain [m].
const TWO_D_DOMAIN_LENGTH_M: f64 = 8.0;
/// Depth of the 2-D domain [m].
const TWO_D_DOMAIN_DEPTH_M: f64 = 2.0;
/// Left edge of the slab footprint [m].
const TWO_D_SLAB_X0_M: f64 = 2.0;
/// Right edge of the slab footprint [m].
const TWO_D_SLAB_X1_M: f64 = 6.0;
/// Slab conductivity [W/(m·K)] (more conductive than the soil).
const TWO_D_SLAB_K_W_MK: f64 = 3.0;
/// Soil conductivity [W/(m·K)].
const TWO_D_SOIL_K_W_MK: f64 = 1.9;
/// Slab volumetric heat capacity [J/(m³·K)].
const TWO_D_SLAB_RHO_C: f64 = 2_000_000.0;
/// Soil volumetric heat capacity [J/(m³·K)].
const TWO_D_SOIL_RHO_C: f64 = 1_600_000.0;
/// Cycle-mean surface temperature of the 2-D run [°C].
const TWO_D_MEAN_SURFACE_C: f64 = 15.0;
/// Deep-ground temperature of the 2-D run [°C].
const TWO_D_DEEP_C: f64 = 10.0;
/// Surface-swing amplitude of the 2-D run [K].
const TWO_D_AMPLITUDE_K: f64 = 5.0;
/// Time step of the 2-D run [s].
const TWO_D_DT_S: f64 = 3600.0;
/// Spin-up length of the 2-D run [days].
const TWO_D_SPIN_UP_DAYS: f64 = 35.0;
/// Tolerance of the cycle-to-cycle change [K].
const TWO_D_CYCLE_TOLERANCE_K: f64 = 0.02;
/// Tolerance of the cycle-mean against the steady solve [K].
const TWO_D_STEADY_TOLERANCE_K: f64 = 0.05;
/// Minimum spread of the cycle-mean across the columns of one row [K].
const TWO_D_MIN_X_SPREAD_K: f64 = 0.15;

/// The 2-D configuration with a given (constant) top-surface temperature.
fn two_dimensional_config(top_temperature_c: f64) -> GroundDomainConfig {
    GroundDomainConfig {
        domain_length_m: TWO_D_DOMAIN_LENGTH_M,
        domain_depth_m: TWO_D_DOMAIN_DEPTH_M,
        slab_x0_m: TWO_D_SLAB_X0_M,
        slab_x1_m: TWO_D_SLAB_X1_M,
        slab_thickness_m: 0.3,
        slab_conductivity_w_mk: TWO_D_SLAB_K_W_MK,
        ground_conductivity_w_mk: TWO_D_SOIL_K_W_MK,
        slab_volumetric_heat_capacity_j_m3k: TWO_D_SLAB_RHO_C,
        ground_volumetric_heat_capacity_j_m3k: TWO_D_SOIL_RHO_C,
        lateral_segments: vec![
            GridSegment::Uniform {
                start_m: 0.0,
                end_m: TWO_D_SLAB_X0_M,
                cells: 4,
            },
            GridSegment::Uniform {
                start_m: TWO_D_SLAB_X0_M,
                end_m: TWO_D_SLAB_X1_M,
                cells: 8,
            },
            GridSegment::Uniform {
                start_m: TWO_D_SLAB_X1_M,
                end_m: TWO_D_DOMAIN_LENGTH_M,
                cells: 4,
            },
        ],
        vertical_segments: vec![
            GridSegment::Uniform {
                start_m: 0.0,
                end_m: 0.5,
                cells: 10,
            },
            GridSegment::Uniform {
                start_m: 0.5,
                end_m: 1.0,
                cells: 5,
            },
            GridSegment::Geometric {
                start_m: 1.0,
                end_m: TWO_D_DOMAIN_DEPTH_M,
                first_cell_m: 0.05,
                growth: 1.3,
                max_cell_m: 0.4,
            },
        ],
        top_segments: vec![TopSegment {
            x_start_m: 0.0,
            x_end_m: TWO_D_DOMAIN_LENGTH_M,
            condition: TopCondition::Dirichlet(top_temperature_c),
        }],
        deep_ground: SurfaceCondition::Dirichlet(TWO_D_DEEP_C),
        lateral_boundary: SurfaceCondition::Adiabatic,
        max_iterations: 50_000,
        tolerance: 1.0e-9,
    }
}

/// Top segments of the 2-D run at the prescribed surface temperature.
fn two_dimensional_top(temperature_c: f64) -> Vec<TopSegment> {
    vec![TopSegment {
        x_start_m: 0.0,
        x_end_m: TWO_D_DOMAIN_LENGTH_M,
        condition: TopCondition::Dirichlet(temperature_c),
    }]
}

/// The 2-D surface temperature at time `t_s` [°C].
fn two_dimensional_surface_temperature_c(t_s: f64) -> f64 {
    TWO_D_MEAN_SURFACE_C + TWO_D_AMPLITUDE_K * (OMEGA_RAD_S * t_s).cos()
}

/// Result of the 2-D periodic run.
struct TwoDimensionalRun {
    /// Cycle-mean field, indexed as `j * nx + i` [°C].
    cycle_mean: Vec<f64>,
    /// Field at the end of the last cycle [°C].
    last_cycle: Vec<f64>,
    /// Field at the end of the previous cycle, same phase [°C].
    previous_cycle: Vec<f64>,
    /// Cell-centre x coordinates [m].
    x: Vec<f64>,
    /// Cell-centre z coordinates [m].
    z: Vec<f64>,
}

/// March the 2-D case to its periodic steady state.
fn run_two_dimensional() -> &'static TwoDimensionalRun {
    static RUN: OnceLock<TwoDimensionalRun> = OnceLock::new();
    RUN.get_or_init(|| {
        let solver =
            GroundSolver::new(two_dimensional_config(TWO_D_MEAN_SURFACE_C)).expect("valid config");
        let mut state: GroundTransientState = solver
            .initial_transient_state(TWO_D_DEEP_C)
            .expect("initial state");

        let steps_per_cycle = (PERIOD_S / TWO_D_DT_S) as usize;
        let spin_up_steps = (TWO_D_SPIN_UP_DAYS * 86_400.0 / TWO_D_DT_S) as usize;
        for _ in 0..spin_up_steps {
            let t_next = state.time_s + TWO_D_DT_S;
            solver
                .step_transient_with_top(
                    &mut state,
                    TWO_D_DT_S,
                    &two_dimensional_top(two_dimensional_surface_temperature_c(t_next)),
                )
                .expect("spin-up step");
        }

        // Two full cycles at the end: the first one is discarded, the second
        // one is the measured cycle, and the end of the first one is the
        // same-phase reference of the previous cycle.
        let mut previous_cycle: Vec<f64> = Vec::new();
        let mut last_cycle: Vec<f64> = Vec::new();
        let mut cycle_mean = vec![0.0_f64; state.temperature_c.len()];
        let mut samples = 0usize;
        for step in 0..2 * steps_per_cycle {
            let t_next = state.time_s + TWO_D_DT_S;
            solver
                .step_transient_with_top(
                    &mut state,
                    TWO_D_DT_S,
                    &two_dimensional_top(two_dimensional_surface_temperature_c(t_next)),
                )
                .expect("periodic step");
            if step == steps_per_cycle - 1 {
                previous_cycle = state.temperature_c.clone();
            }
            if step >= steps_per_cycle {
                for (mean, &value) in cycle_mean.iter_mut().zip(state.temperature_c.iter()) {
                    *mean += value;
                }
                samples += 1;
                last_cycle = state.temperature_c.clone();
            }
        }
        for mean in cycle_mean.iter_mut() {
            *mean /= samples as f64;
        }

        TwoDimensionalRun {
            cycle_mean,
            last_cycle,
            previous_cycle,
            x: state.grid.x.centers.clone(),
            z: state.grid.z.centers.clone(),
        }
    })
}

#[test]
fn two_dimensional_periodic_run_settles_and_keeps_its_mean_field() {
    let run = run_two_dimensional();
    let nx = run.x.len();
    let nz = run.z.len();

    // (i) The run settles to a periodic steady state: two consecutive
    // cycles agree at the same phase.
    let mut cycle_change = 0.0_f64;
    for (last, previous) in run.last_cycle.iter().zip(run.previous_cycle.iter()) {
        cycle_change = cycle_change.max((last - previous).abs());
    }
    assert!(
        cycle_change <= TWO_D_CYCLE_TOLERANCE_K,
        "the 2-D run has not settled to a periodic steady state: cycle-to-cycle \
         change {cycle_change:.6} K > {TWO_D_CYCLE_TOLERANCE_K} K"
    );

    // (ii) The cycle-mean near-surface field sits at the cycle-mean surface
    // temperature and differs from the deep-ground temperature, so the case
    // cannot degenerate into a uniform field.
    let mut surface_mean = 0.0_f64;
    for i in 0..nx {
        surface_mean += run.cycle_mean[i];
    }
    surface_mean /= nx as f64;
    assert!(
        (surface_mean - TWO_D_MEAN_SURFACE_C).abs() <= 0.5,
        "the cycle-mean near-surface temperature {surface_mean:.4} °C does not \
         track the imposed cycle-mean surface temperature {TWO_D_MEAN_SURFACE_C} °C"
    );
    assert!(
        surface_mean - TWO_D_DEEP_C > 3.0,
        "the cycle-mean near-surface temperature {surface_mean:.4} °C does not \
         differ from the deep-ground temperature {TWO_D_DEEP_C} °C"
    );

    // (iii) The cycle-mean field is the steady solution for the mean
    // surface temperature — cell by cell over the whole domain.
    let solver =
        GroundSolver::new(two_dimensional_config(TWO_D_MEAN_SURFACE_C)).expect("valid config");
    let steady = solver.solve().expect("steady solve");
    let mut max_deviation = 0.0_f64;
    for j in 0..nz {
        for i in 0..nx {
            let cell = j * nx + i;
            let deviation = (run.cycle_mean[cell] - steady.cell_temperature(i, j)).abs();
            max_deviation = max_deviation.max(deviation);
        }
    }
    assert!(
        max_deviation <= TWO_D_STEADY_TOLERANCE_K,
        "the cycle-mean field deviates from the steady solve by {max_deviation:.6} K \
         > {TWO_D_STEADY_TOLERANCE_K} K"
    );

    // (iv) The cycle-mean field has non-trivial x-dependence.
    let mut max_spread = 0.0_f64;
    for j in 0..nz {
        let row = &run.cycle_mean[j * nx..(j + 1) * nx];
        let spread = row.iter().copied().fold(f64::MIN, f64::max)
            - row.iter().copied().fold(f64::MAX, f64::min);
        max_spread = max_spread.max(spread);
    }
    assert!(
        max_spread >= TWO_D_MIN_X_SPREAD_K,
        "the cycle-mean field is laterally uniform: max spread across a row \
         {max_spread:.6} K < {TWO_D_MIN_X_SPREAD_K} K"
    );

    println!(
        "periodic 2-D: cycle-to-cycle change {cycle_change:.6} K, cycle-mean vs \
         steady max deviation {max_deviation:.6} K, max lateral spread {max_spread:.4} K, \
         cycle-mean surface {surface_mean:.4} °C"
    );
}
