//! ASHRAE 1052-RP analytical steady-periodic conduction regression (Issue #3981).
//!
//! Ground-truth module: every conduction implementation (FD multi-node with
//! BackwardEuler/Bdf2/CrankNicolson, CTF) is driven to steady-periodic state
//! with sinusoidal sol-air excitation and its interior heat flux is compared
//! against the closed-form complex transmission-matrix solution of the 1-D
//! conduction PDE.
//!
//! Physics (signs validated against an independent fine-grid Crank-Nicolson
//! PDE solve; see docs/validation/conduction_time_integration.md):
//!   * x runs interior -> exterior; layer/film matrices propagate [Theta; Q]
//!     with Q the complex flux amplitude in +x: Theta(x+dx) = Theta(x) - Q R.
//!   * q_zone_inward(t) = -Q_i(t): positive = heat flux INTO the zone.
//!   * Excitation T_solair(t) = T_EXT_MEAN + AMPL * sin(omega t);
//!     response q(t) = qbar + Amp * sin(omega t - phi).
//!   * The solvers integrate a STAIRCASE (zero-order-hold) sampling of the
//!     sinusoid at the timestep marks, so the primary reference is the
//!     ZOH-Fourier series of the exact PDE response (harmonics of the held
//!     waveform passed through the exact transfer function H(k w0)).
//!
//! CTF film note: `CTFSolverWrapper` bakes ASHRAE 140 films (R_SI = 0.125,
//! R_SE = 0.044 m2K/W) into its coefficients and ignores the h arguments of
//! `step`, so CTF comparisons use that film pair; FD comparisons use the
//! SurfaceBC films (h_int = 8.3, h_ext = 18.3).

use num_complex::Complex;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Boundary conditions (mirror tests/all_tests/fd_time_integration_accuracy.rs)
// ---------------------------------------------------------------------------

const T_ZONE: f64 = 21.5;
const T_EXT_MEAN: f64 = 29.5;
const AMPL: f64 = 12.5;
const PERIOD: f64 = 24.0 * 3600.0;
const OMEGA: f64 = 2.0 * PI / PERIOD;
const H_INT: f64 = 8.3;
const H_EXT: f64 = 18.3;
/// CTF-baked film resistances (ASHRAE 140): R_SI interior, R_SE exterior.
const R_SI_CTF: f64 = 0.125;
const R_SE_CTF: f64 = 0.044;

// ---------------------------------------------------------------------------
// Wall definitions (exterior -> interior, mirroring WallSpec constructors)
// ---------------------------------------------------------------------------

/// Homogeneous layer: (thickness [m], k [W/mK], rho [kg/m3], cp [J/kgK]).
#[derive(Clone, Copy)]
struct LayerDef {
    thickness: f64,
    k: f64,
    rho: f64,
    cp: f64,
}

const CONCRETE: LayerDef = LayerDef {
    thickness: 0.2,
    k: 1.73,
    rho: 2243.0,
    cp: 837.0,
};
const BRICK: LayerDef = LayerDef {
    thickness: 0.1,
    k: 0.81,
    rho: 1920.0,
    cp: 790.0,
};
const EPS: LayerDef = LayerDef {
    thickness: 0.08,
    k: 0.04,
    rho: 25.0,
    cp: 1400.0,
};
const GYPSUM: LayerDef = LayerDef {
    thickness: 0.013,
    k: 0.16,
    rho: 800.0,
    cp: 1090.0,
};

const HEAVY: &[LayerDef] = &[CONCRETE];
const MEDIUM: &[LayerDef] = &[BRICK];
const LIGHT: &[LayerDef] = &[EPS];
const MULTI: &[LayerDef] = &[BRICK, EPS, GYPSUM];

/// Film pair (R_interior, R_exterior) [m2K/W].
type Films = (f64, f64);

const FILMS_FD: Films = (1.0 / H_INT, 1.0 / H_EXT);
const FILMS_CTF: Films = (R_SI_CTF, R_SE_CTF);

// ---------------------------------------------------------------------------
// Analytical solution: complex transmission matrix
// ---------------------------------------------------------------------------

type Cx = Complex<f64>;
type Mat2 = [[Cx; 2]; 2];

fn layer_matrix(layer: &LayerDef, omega: f64) -> Mat2 {
    let alpha = layer.k / (layer.rho * layer.cp);
    let m = Complex::new(0.0, omega / alpha).sqrt();
    let m_l = m * layer.thickness;
    let sinh_ml = m_l.sinh();
    let cosh_ml = m_l.cosh();
    [
        [cosh_ml, -sinh_ml / (m * layer.k)],
        [-m * layer.k * sinh_ml, cosh_ml],
    ]
}

fn film_matrix(r: f64) -> Mat2 {
    [
        [Cx::new(1.0, 0.0), Cx::new(-r, 0.0)],
        [Cx::new(0.0, 0.0), Cx::new(1.0, 0.0)],
    ]
}

fn mat_mul(a: &Mat2, b: &Mat2) -> Mat2 {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

/// Chain interior-air -> exterior-air (interior film first, exterior film last).
fn total_matrix(layers: &[LayerDef], films: &Films, omega: f64) -> Mat2 {
    let mut m = film_matrix(films.0);
    for layer in layers.iter().rev() {
        m = mat_mul(&layer_matrix(layer, omega), &m);
    }
    mat_mul(&film_matrix(films.1), &m)
}

/// H(omega) = qhat_zone_inward / Ahat_solair for harmonic e^{i omega t}.
/// DC limit: H(0) = U_total.
fn transfer(layers: &[LayerDef], films: &Films, omega: f64) -> Cx {
    let m = total_matrix(layers, films, omega);
    -Cx::new(1.0, 0.0) / m[0][1]
}

fn u_total(layers: &[LayerDef], films: &Films) -> f64 {
    let r: f64 = films.0 + films.1 + layers.iter().map(|l| l.thickness / l.k).sum::<f64>();
    1.0 / r
}

// ---------------------------------------------------------------------------
// Golden constants (generated by tmp/golden_1052rp.py — closed form
// cross-validated against an independent fine-grid Crank-Nicolson PDE solve)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct GoldenCase {
    name: &'static str,
    /// U-value including films [W/m2K].
    u: f64,
    /// Steady-periodic mean into-zone flux [W/m2].
    qbar: f64,
    /// Continuous-sinusoid fundamental: amplitude [W/m2], phase lag [rad].
    cont_amp: f64,
    cont_phi: f64,
    /// ZOH fundamental at dt=3600/900: amplitude [W/m2], phase lag [rad].
    zoh3600_amp: f64,
    zoh3600_phi: f64,
    zoh900_amp: f64,
    zoh900_phi: f64,
}

const GOLDEN: &[GoldenCase] = &[
    GoldenCase {
        name: "heavy_fd",
        u: 3.4395740615e+00,
        qbar: 2.7516592492e+01,
        cont_amp: 2.4650831152e+01,
        cont_phi: 1.3411621382e+00,
        zoh3600_amp: 2.4582913597e+01,
        zoh3600_phi: 1.4721492690e+00,
        zoh900_amp: 2.4646431540e+01,
        zoh900_phi: 1.3738870617e+00,
    },
    GoldenCase {
        name: "medium_fd",
        u: 3.3491465904e+00,
        qbar: 2.6793172724e+01,
        cont_amp: 3.6281732865e+01,
        cont_phi: 6.8543030561e-01,
        zoh3600_amp: 3.6135011656e+01,
        zoh3600_phi: 8.1447317073e-01,
        zoh900_amp: 3.6275257402e+01,
        zoh900_phi: 7.1815522909e-01,
    },
    GoldenCase {
        name: "light_fd",
        u: 4.5974332587e-01,
        qbar: 3.6779466069e+00,
        cont_amp: 5.7394211968e+00,
        cont_phi: 7.9380537090e-02,
        zoh3600_amp: 5.7466996415e+00,
        zoh3600_phi: 2.6875703789e-01,
        zoh900_amp: 5.7383968409e+00,
        zoh900_phi: 1.1210546056e-01,
    },
    GoldenCase {
        name: "multi_fd",
        u: 4.2019745870e-01,
        qbar: 3.3615796696e+00,
        cont_amp: 3.5831331929e+00,
        cont_phi: 1.1938636601e+00,
        zoh3600_amp: 3.5737188290e+00,
        zoh3600_phi: 1.3249211271e+00,
        zoh900_amp: 3.5824936853e+00,
        zoh900_phi: 1.2265885836e+00,
    },
    GoldenCase {
        name: "heavy_ctf",
        u: 3.5136178077e+00,
        qbar: 2.8108942462e+01,
        cont_amp: 2.6338401646e+01,
        cont_phi: 1.2994401509e+00,
        zoh3600_amp: 2.6265755995e+01,
        zoh3600_phi: 1.4304361926e+00,
        zoh900_amp: 2.6333700842e+01,
        zoh900_phi: 1.3321650744e+00,
    },
    GoldenCase {
        name: "medium_ctf",
        u: 3.4193085398e+00,
        qbar: 2.7354468319e+01,
        cont_amp: 3.7612954639e+01,
        cont_phi: 6.5395460413e-01,
        zoh3600_amp: 3.7459781146e+01,
        zoh3600_phi: 7.8279243333e-01,
        zoh900_amp: 3.7606241584e+01,
        zoh900_phi: 6.8667952760e-01,
    },
    GoldenCase {
        name: "light_ctf",
        u: 4.6104195482e-01,
        qbar: 3.6883356385e+00,
        cont_amp: 5.7557176352e+00,
        cont_phi: 7.8928637181e-02,
        zoh3600_amp: 5.7629618107e+00,
        zoh3600_phi: 2.6867128345e-01,
        zoh900_amp: 5.7546903708e+00,
        zoh900_phi: 1.1165356066e-01,
    },
    GoldenCase {
        name: "multi_ctf",
        u: 4.2128202361e-01,
        qbar: 3.3702561889e+00,
        cont_amp: 3.7798430898e+00,
        cont_phi: 1.1404211463e+00,
        zoh3600_amp: 3.7699304178e+00,
        zoh3600_phi: 1.2715034751e+00,
        zoh900_amp: 3.7791684739e+00,
        zoh900_phi: 1.1731460698e+00,
    },
];

fn wall_and_films(name: &str) -> (&'static [LayerDef], Films) {
    match name {
        "heavy_fd" => (HEAVY, FILMS_FD),
        "medium_fd" => (MEDIUM, FILMS_FD),
        "light_fd" => (LIGHT, FILMS_FD),
        "multi_fd" => (MULTI, FILMS_FD),
        "heavy_ctf" => (HEAVY, FILMS_CTF),
        "medium_ctf" => (MEDIUM, FILMS_CTF),
        "light_ctf" => (LIGHT, FILMS_CTF),
        "multi_ctf" => (MULTI, FILMS_CTF),
        _ => panic!("unknown golden case {name}"),
    }
}

// ---------------------------------------------------------------------------
// Cycle 3: FD teacher path meets the ground truth (heavy wall first).
// ---------------------------------------------------------------------------

use fluxion::physics::fd_discretization::{MaterialLayer, WallDiscretization};
use fluxion::physics::fd_solver::{ImplicitFDSolver, SurfaceBC, TimeIntegrationScheme};

/// Steady-periodic FD run result: demodulated (amp, phi, mean) of the
/// into-zone flux plus the day-to-day periodicity drift.
struct FdRun {
    amp: f64,
    phi: f64,
    mean: f64,
    drift: f64,
}

/// Drive `ImplicitFDSolver` with the staircase sol-air excitation until
/// periodic steady state, then demodulate the final day. Samples sit at
/// end-of-step marks, so the demodulation offset is one step.
#[allow(clippy::too_many_arguments)]
fn run_fd_steady_periodic(
    layers: &[LayerDef],
    scheme: TimeIntegrationScheme,
    dt: f64,
    nodes_per_layer: usize,
    spinup_days: usize,
) -> FdRun {
    // `WallDiscretization::from_layers` expects layers interior-first; the
    // LayerDef lists here are exterior-first (mirroring WallSpec order), so
    // reverse. U is orientation-invariant, but the steady-periodic amplitude
    // and phase are not — the golden reference models gypsum at the interior.
    let mats: Vec<MaterialLayer> = layers
        .iter()
        .rev()
        .map(|l| MaterialLayer {
            name: "layer".to_string(),
            conductivity: l.k,
            density: l.rho,
            specific_heat: l.cp,
            thickness: l.thickness,
        })
        .collect();
    let disc = WallDiscretization::from_layers(&mats, nodes_per_layer);
    let mut solver = ImplicitFDSolver::with_scheme(disc, T_ZONE, scheme);
    let steps_per_day = (PERIOD / dt).round() as usize;
    let total_steps = spinup_days * steps_per_day + 2 * steps_per_day;
    let mut samples: Vec<f64> = Vec::with_capacity(total_steps);
    for step in 0..total_steps {
        let t_start = step as f64 * dt;
        let sol_air = T_EXT_MEAN + AMPL * (OMEGA * t_start).sin();
        let bc_int = SurfaceBC::new_interior(H_INT, T_ZONE);
        let bc_ext = SurfaceBC::new_exterior(H_EXT, sol_air, 0.0);
        solver.step(dt, &bc_int, &bc_ext);
        samples.push(-solver.interior_heat_flux(H_INT, T_ZONE));
    }
    // periodicity drift: max |last-day - previous-day| elementwise
    let n = steps_per_day;
    let drift: f64 = (0..n)
        .map(|i| (samples[total_steps - n + i] - samples[total_steps - 2 * n + i]).abs())
        .fold(0.0, f64::max);
    let (amp, phi, mean) = demodulate_offset(&samples[total_steps - n..], 1);
    FdRun {
        amp,
        phi,
        mean,
        drift,
    }
}

const FD_NODES_PER_LAYER: usize = 30;
const FD_SPINUP_DAYS: usize = 14;

fn golden(name: &str) -> GoldenCase {
    *GOLDEN
        .iter()
        .find(|&g| g.name == name)
        .unwrap_or_else(|| panic!("unknown golden case {name}"))
}

/// Tolerance budget for one FD run vs its ZOH golden fundamental, pinned
/// from the measured post-fix discretization error (conservative assembly,
/// Issue #3981) with ≥2.5x headroom on amp/mean and ≥1.8x on phi. Regression
/// = an exceeded budget. The full table is reproduced in
/// docs/validation/conduction_time_integration.md.
struct FdBudget {
    amp_rel: f64,
    phi_abs: f64,
    mean_rel: f64,
}

#[allow(clippy::too_many_lines)]
fn fd_budget_for(wall: &str, scheme: &str, dt: f64) -> FdBudget {
    let be = |amp: f64, phi: f64| FdBudget {
        amp_rel: amp,
        phi_abs: phi,
        mean_rel: 0.005,
    };
    match (wall, scheme, dt as i64) {
        // BDF2 (teacher scheme): measured amp |err| <= 1.7% @3600, <= 0.13% @900
        ("heavy", "bdf2", 3600) => be(0.040, 0.25),
        ("heavy", "bdf2", 900) => be(0.015, 0.10),
        ("medium", "bdf2", 3600) => be(0.040, 0.25),
        ("medium", "bdf2", 900) => be(0.015, 0.10),
        ("light", "bdf2", 3600) => be(0.040, 0.20),
        ("light", "bdf2", 900) => be(0.015, 0.10),
        ("multi", "bdf2", 3600) => be(0.040, 0.25),
        ("multi", "bdf2", 900) => be(0.015, 0.10),
        // Backward Euler: first-order amplitude damping up to ~10% @3600
        ("heavy", "backward_euler", 3600) => be(0.180, 0.15),
        ("heavy", "backward_euler", 900) => be(0.070, 0.10),
        ("medium", "backward_euler", 3600) => be(0.150, 0.20),
        ("medium", "backward_euler", 900) => be(0.050, 0.10),
        ("light", "backward_euler", 3600) => be(0.050, 0.20),
        ("light", "backward_euler", 900) => be(0.025, 0.10),
        ("multi", "backward_euler", 3600) => be(0.180, 0.15),
        ("multi", "backward_euler", 900) => be(0.070, 0.10),
        // Crank-Nicolson: 2nd order like BDF2, but not L-stable — light wall
        // at dt=3600 carries a bounded, non-decaying ringing component.
        ("heavy", "crank_nicolson", 3600) => be(0.040, 0.25),
        ("heavy", "crank_nicolson", 900) => be(0.015, 0.10),
        ("medium", "crank_nicolson", 3600) => be(0.040, 0.25),
        ("medium", "crank_nicolson", 900) => be(0.015, 0.10),
        ("light", "crank_nicolson", 3600) => be(0.040, 0.20),
        ("light", "crank_nicolson", 900) => be(0.015, 0.10),
        ("multi", "crank_nicolson", 3600) => be(0.040, 0.25),
        ("multi", "crank_nicolson", 900) => be(0.015, 0.10),
        _ => panic!("no budget for {wall}/{scheme}/dt={dt}"),
    }
}

/// Table-driven FD regression: every (wall, scheme, dt) cell must meet its
/// pinned analytical budget. The mean budget (0.5%) guards the conservative
/// network's exact steady-state U realization.
#[test]
fn fd_all_schemes_meet_analytical_budgets() {
    for (wall, layers) in [
        ("heavy", HEAVY),
        ("medium", MEDIUM),
        ("light", LIGHT),
        ("multi", MULTI),
    ] {
        for (scheme_name, scheme) in [
            ("bdf2", TimeIntegrationScheme::Bdf2),
            ("backward_euler", TimeIntegrationScheme::BackwardEuler),
            ("crank_nicolson", TimeIntegrationScheme::CrankNicolson),
        ] {
            for dt in [3600.0, 900.0] {
                let run =
                    run_fd_steady_periodic(layers, scheme, dt, FD_NODES_PER_LAYER, FD_SPINUP_DAYS);
                let g = golden(&format!("{wall}_fd"));
                let (g_amp, g_phi) = if dt == 3600.0 {
                    (g.zoh3600_amp, g.zoh3600_phi)
                } else {
                    (g.zoh900_amp, g.zoh900_phi)
                };
                // Periodicity: CN on the light wall at dt=3600 retains a
                // bounded ringing component (A-stable, not L-stable; z >> 2
                // for the finest modes) — allow 1% of amplitude there, exact
                // periodicity everywhere else.
                let drift_budget =
                    if wall == "light" && scheme_name == "crank_nicolson" && dt == 3600.0 {
                        0.01 * run.amp
                    } else {
                        1e-6 * run.amp
                    };
                assert!(
                    run.drift < drift_budget,
                    "{wall}/{scheme_name}/dt={dt}: periodicity drift {} exceeds budget",
                    run.drift
                );
                let b = fd_budget_for(wall, scheme_name, dt);
                let amp_rel = (run.amp - g_amp).abs() / g_amp;
                assert!(
                    amp_rel < b.amp_rel,
                    "{wall}/{scheme_name}/dt={dt}: amp err {:+.4}% (budget {:.2}%)",
                    100.0 * (run.amp - g_amp) / g_amp,
                    100.0 * b.amp_rel
                );
                assert!(
                    (run.phi - g_phi).abs() < b.phi_abs,
                    "{wall}/{scheme_name}/dt={dt}: phi err {:+.4} rad (budget {:.3})",
                    run.phi - g_phi,
                    b.phi_abs
                );
                let mean_rel = (run.mean - g.qbar).abs() / g.qbar;
                assert!(
                    mean_rel < b.mean_rel,
                    "{wall}/{scheme_name}/dt={dt}: mean err {:+.4}% (budget {:.2}%)",
                    100.0 * (run.mean - g.qbar) / g.qbar,
                    100.0 * b.mean_rel
                );
            }
        }
    }
}

/// The Issue #3981 gate: the teacher FD path (BDF2) must beat the
/// backward-Euler path on the amplitude metric at both timesteps for every
/// wall, and refine with dt. NOTE on phase: BE's smaller phase error at
/// dt=3600 is not scheme superiority — its extra amplitude damping cancels
/// part of the ZOH/sampling phase residual (which converges first-order,
/// ~0.147 rad @3600 -> ~0.034 rad @900 for the 2nd-order schemes). The
/// amplitude gate is the honest discriminator.
#[test]
fn fd_teacher_path_beats_backward_euler_gate() {
    for (wall, layers) in [
        ("heavy", HEAVY),
        ("medium", MEDIUM),
        ("light", LIGHT),
        ("multi", MULTI),
    ] {
        let g = golden(&format!("{wall}_fd"));
        for (dt, g_amp) in [(3600.0, g.zoh3600_amp), (900.0, g.zoh900_amp)] {
            let bdf2 = run_fd_steady_periodic(
                layers,
                TimeIntegrationScheme::Bdf2,
                dt,
                FD_NODES_PER_LAYER,
                FD_SPINUP_DAYS,
            );
            let be = run_fd_steady_periodic(
                layers,
                TimeIntegrationScheme::BackwardEuler,
                dt,
                FD_NODES_PER_LAYER,
                FD_SPINUP_DAYS,
            );
            let err_bdf2 = (bdf2.amp - g_amp).abs() / g_amp;
            let err_be = (be.amp - g_amp).abs() / g_amp;
            assert!(
                err_bdf2 < err_be,
                "{wall}/dt={dt}: teacher BDF2 amp err {:.4}% NOT below backward-Euler {:.4}%",
                100.0 * err_bdf2,
                100.0 * err_be
            );
        }
        // dt refinement: BDF2 amplitude error at dt=900 must not exceed its
        // dt=3600 error (2nd-order convergence, measured ratio >= 3x).
        let bdf2_3600 = run_fd_steady_periodic(
            layers,
            TimeIntegrationScheme::Bdf2,
            3600.0,
            FD_NODES_PER_LAYER,
            FD_SPINUP_DAYS,
        );
        let bdf2_900 = run_fd_steady_periodic(
            layers,
            TimeIntegrationScheme::Bdf2,
            900.0,
            FD_NODES_PER_LAYER,
            FD_SPINUP_DAYS,
        );
        assert!(
            ((bdf2_900.amp - g.zoh900_amp).abs() / g.zoh900_amp)
                < ((bdf2_3600.amp - g.zoh3600_amp).abs() / g.zoh3600_amp),
            "{wall}: BDF2 amp error did not refine from dt=3600 to dt=900"
        );
    }
}

/// RED test for the fd_solver multi-layer defect found by this module (Issue
/// #3981): the uniform-grid second difference `Fo_i*(T[i-1]-2T[i]+T[i+1])`
/// collapses at steady state to `T[i]=(T[i-1]+T[i+1])/2` — linear in node
/// INDEX — so interior resistance splits by node count, not layer R-value.
/// `WallDiscretization::interface_conductivities` were never consumed.
///
/// GREEN (conservative variable-spacing assembly) must realize the analytical
/// series resistance of every layer stack to truncation error, and conserve
/// energy (interior flux == exterior flux at steady state).
#[test]
fn fd_multilayer_wall_realizes_analytical_u_and_energy_balance() {
    let combos: &[(&str, &[LayerDef])] = &[
        ("brick+eps", &[BRICK, EPS]),
        ("eps+gyp", &[EPS, GYPSUM]),
        ("brick+gyp", &[BRICK, GYPSUM]),
        ("multi", MULTI),
    ];
    for (name, layers) in combos {
        let mats: Vec<MaterialLayer> = layers
            .iter()
            .map(|l| MaterialLayer {
                name: "l".to_string(),
                conductivity: l.k,
                density: l.rho,
                specific_heat: l.cp,
                thickness: l.thickness,
            })
            .collect();
        let disc = WallDiscretization::from_layers(&mats, 30);
        let mut solver = ImplicitFDSolver::with_scheme(disc, 20.0, TimeIntegrationScheme::Bdf2);
        for _ in 0..30 * 24 {
            let bc_int = SurfaceBC::new_interior(H_INT, 20.0);
            let bc_ext = SurfaceBC::new_exterior(H_EXT, 28.0, 0.0);
            solver.step(3600.0, &bc_int, &bc_ext);
        }
        let q_int_into_wall = solver.interior_heat_flux(H_INT, 20.0); // < 0 (wall hotter than zone)
        let q_ext_into_wall = solver.exterior_heat_flux(H_EXT, 28.0); // > 0
                                                                      // Through-flux magnitude; the two face evaluations must agree at
                                                                      // steady state (energy conservation: what enters outside leaves inside).
        let q_through = 0.5 * (q_ext_into_wall - q_int_into_wall);
        assert!(
            (q_ext_into_wall + q_int_into_wall).abs() < 0.01 * q_through.abs(),
            "{name}: steady flux imbalance: int={q_int_into_wall:.5} ext={q_ext_into_wall:.5} W/m2"
        );
        let u_real = q_through / 8.0;
        let u_true = u_total(layers, &FILMS_FD);
        assert!(
            (u_real / u_true - 1.0).abs() < 0.02,
            "{name}: realized U {u_real:.5} vs analytical {u_true:.5} (rel {:+.4}%)",
            100.0 * (u_real / u_true - 1.0)
        );
        eprintln!("[fix-check] {name:10} U_real={u_real:8.5} U_true={u_true:8.5} q_int={q_int_into_wall:8.4} q_ext={q_ext_into_wall:8.4}");
    }
}

/// Steady-periodic ZOH reference: q(t) = dc + sum_k 2*Re(coeff_k e^{i k w0 t}).
struct ZohReference {
    dc: f64,
    /// (harmonic index k, H(k w0) * c_k)
    harmonics: Vec<(usize, Cx)>,
    omega0: f64,
}

fn zoh_reference(
    layers: &[LayerDef],
    films: &Films,
    dt: f64,
    max_harmonics: usize,
) -> ZohReference {
    let n_steps = (PERIOD / dt).round() as usize;
    assert!(
        (n_steps as f64 * dt - PERIOD).abs() < 1e-9,
        "dt must divide the period"
    );
    let omega0 = 2.0 * PI / (n_steps as f64 * dt);
    let dc = u_total(layers, films) * (T_EXT_MEAN - T_ZONE);
    let i_cx = Cx::new(0.0, 1.0);
    let mut harmonics = Vec::new();
    for k in 1..=max_harmonics {
        let wk = k as f64 * omega0;
        if wk * dt > 2.0 * PI * 8.0 {
            break;
        }
        // Fourier coefficient c_k of the held sinusoid A sin(w0 n dt):
        // c_k = (1/P) sum_n A sin(w0 n dt) e^{-i k w0 n dt} (1-e^{-i k w0 dt})/(i k w0)
        let step_int = i_cx * (Cx::new(0.0, -wk * dt).exp() - Cx::new(1.0, 0.0)) / wk;
        let mut ck = Cx::new(0.0, 0.0);
        for n in 0..n_steps {
            let theta = omega0 * n as f64 * dt;
            ck += AMPL
                * theta.sin()
                * Cx::new(0.0, -(k as f64) * omega0 * n as f64 * dt).exp()
                * step_int;
        }
        ck /= n_steps as f64 * dt;
        if ck.norm() < 1e-16 * AMPL {
            continue;
        }
        harmonics.push((k, transfer(layers, films, wk) * ck));
    }
    ZohReference {
        dc,
        harmonics,
        omega0,
    }
}

impl ZohReference {
    /// Physical (real) series value at time t [s].
    fn at(&self, t: f64) -> f64 {
        let mut acc = 0.0;
        for &(k, c) in &self.harmonics {
            acc += (c * Cx::new(0.0, k as f64 * self.omega0 * t).exp()).re;
        }
        self.dc + 2.0 * acc
    }
}

/// Demodulate one full day of samples at marks t_n = n*dt_sample over the
/// fundamental: q ~ qbar + amp*sin(w t - phi). Basis is orthogonal over
/// complete periods. `offset_steps` shifts the time origin for samples that
/// sit at end-of-step marks ((i+1)*dt), as FD sampling does.
fn demodulate(samples: &[f64]) -> (f64, f64, f64) {
    demodulate_offset(samples, 0)
}

fn demodulate_offset(samples: &[f64], offset_steps: usize) -> (f64, f64, f64) {
    let n = samples.len();
    let dt_sample = PERIOD / n as f64;
    let mut a = 0.0;
    let mut b = 0.0;
    let mut mean = 0.0;
    for (n_idx, &q) in samples.iter().enumerate() {
        let t = (n_idx + offset_steps) as f64 * dt_sample;
        a += q * (OMEGA * t).sin();
        b += q * (OMEGA * t).cos();
        mean += q;
    }
    a *= 2.0 / n as f64;
    b *= 2.0 / n as f64;
    (a.hypot(b), -b.atan2(a), mean / n as f64)
}

const HEAVY_FD_ZOH3600_SERIES: [f64; 24] = [
    3.053193070,
    4.513390219,
    7.541217588,
    11.930333746,
    17.381627680,
    23.523602717,
    29.937693409,
    36.186790074,
    41.845027102,
    46.526804990,
    49.913068312,
    51.773048819,
    51.979991913,
    50.519794764,
    47.491967395,
    43.102851237,
    37.651557303,
    31.509582266,
    25.095491575,
    18.846394910,
    13.188157882,
    8.506379994,
    5.120116672,
    3.260136165,
];

#[test]
fn zoh_reconstruction_matches_golden_series_and_fundamentals() {
    // (a) full hourly series for heavy_fd at dt=3600
    let zoh = zoh_reference(HEAVY, &FILMS_FD, 3600.0, 64);
    for (n, &golden) in HEAVY_FD_ZOH3600_SERIES.iter().enumerate() {
        let v = zoh.at(n as f64 * 3600.0);
        assert!(
            (v - golden).abs() < 1e-8,
            "heavy_fd series[{n}]: {v:.12} vs golden {golden:.12}"
        );
    }
    // (b) demodulated fundamentals for every wall/film combo at dt=3600 and
    // dt=900, on the same sample grids the solver comparisons use
    for g in GOLDEN {
        let (layers, films) = wall_and_films(g.name);
        for (dt, g_a, g_p) in [
            (3600.0, g.zoh3600_amp, g.zoh3600_phi),
            (900.0, g.zoh900_amp, g.zoh900_phi),
        ] {
            let zoh = zoh_reference(layers, &films, dt, 64);
            let n_samples = (PERIOD / dt).round() as usize;
            let fine: Vec<f64> = (0..n_samples).map(|n| zoh.at(n as f64 * dt)).collect();
            let (amp, phi, mean) = demodulate(&fine);
            let g_qbar = u_total(layers, &films) * (T_EXT_MEAN - T_ZONE);
            assert!(
                (amp - g_a).abs() < 1e-8,
                "{} dt={dt}: zoh amp {amp:.12} vs golden {g_a:.12}",
                g.name
            );
            assert!(
                (phi - g_p).abs() < 1e-9,
                "{} dt={dt}: zoh phi {phi:.12} vs golden {g_p:.12}",
                g.name
            );
            assert!(
                (mean - g_qbar).abs() < 1e-8,
                "{} dt={dt}: zoh mean {mean:.12} vs {g_qbar:.12}",
                g.name
            );
        }
    }
}

#[test]
fn analytical_core_matches_golden_constants() {
    for g in GOLDEN {
        let (layers, films) = wall_and_films(g.name);
        let u = u_total(layers, &films);
        let qbar = u * (T_EXT_MEAN - T_ZONE);
        let h = transfer(layers, &films, OMEGA);
        let amp = h.norm() * AMPL;
        let phi = -h.arg();
        assert!(
            (u - g.u).abs() < 1e-9,
            "{}: U {u:.12} vs golden {:.12}",
            g.name,
            g.u
        );
        assert!(
            (qbar - g.qbar).abs() < 1e-8,
            "{}: qbar {qbar:.12} vs golden {:.12}",
            g.name,
            g.qbar
        );
        assert!(
            (amp - g.cont_amp).abs() < 1e-8,
            "{}: cont amp {amp:.12} vs golden {:.12}",
            g.name,
            g.cont_amp
        );
        assert!(
            (phi - g.cont_phi).abs() < 1e-9,
            "{}: cont phi {phi:.12} vs golden {:.12}",
            g.name,
            g.cont_phi
        );
    }
}

// ---------------------------------------------------------------------------
// Cycle 8: CTF within its linear envelope (ASHRAE 140 films baked into the
// coefficients: R_SI = 0.125, R_SE = 0.044 m2K/W — src/physics/ctf_coefficients.rs).
// ---------------------------------------------------------------------------

use fluxion::physics::ctf_solver_wrapper::CTFSolverWrapper;
use fluxion::physics::solver_trait::HeatConductionSolver;
use fluxion::physics::units::{FromF64, HeatTransferCoefficient, Temperature, Time, ToF64};
use fluxion::physics::wall_spec::{LayerSpec, WallSpec};

fn wall_spec_for(name: &str) -> WallSpec {
    match name {
        "heavy" => WallSpec::single_layer(
            "200mm Concrete",
            CONCRETE.thickness,
            CONCRETE.k,
            CONCRETE.rho,
            CONCRETE.cp,
        ),
        "medium" => {
            WallSpec::single_layer("100mm Brick", BRICK.thickness, BRICK.k, BRICK.rho, BRICK.cp)
        }
        "light" => WallSpec::single_layer("80mm EPS", EPS.thickness, EPS.k, EPS.rho, EPS.cp),
        "multi" => WallSpec::multi_layer(
            "Brick + Insulation + Gypsum",
            vec![
                LayerSpec::new("Clay Brick", BRICK.thickness, BRICK.k, BRICK.rho, BRICK.cp),
                LayerSpec::new("EPS Insulation", EPS.thickness, EPS.k, EPS.rho, EPS.cp),
                LayerSpec::new(
                    "Gypsum Board",
                    GYPSUM.thickness,
                    GYPSUM.k,
                    GYPSUM.rho,
                    GYPSUM.cp,
                ),
            ],
        ),
        _ => panic!("unknown wall {name}"),
    }
}

/// Drive the CTF wrapper with the same staircase sol-air excitation (start-
/// mark held values, hourly) until periodic steady state, demodulate the
/// final day. The wrapper's `initialize` runs its own 7-day warmup; the
/// staircase then needs additional spin-up for the CTF transient roots to
/// decay. Returned flux is positive INTO the zone (wrapper convention).
fn run_ctf_steady_periodic(wall: &str, spinup_days: usize) -> FdRun {
    let mut wrapper = CTFSolverWrapper::new();
    wrapper
        .initialize(&wall_spec_for(wall))
        .expect("CTF wrapper initialization");
    let dt = 3600.0;
    let steps_per_day = (PERIOD / dt) as usize;
    let total_steps = spinup_days * steps_per_day + 2 * steps_per_day;
    let mut samples: Vec<f64> = Vec::with_capacity(total_steps);
    for step in 0..total_steps {
        let t_start = step as f64 * dt;
        let sol_air = T_EXT_MEAN + AMPL * (OMEGA * t_start).sin();
        let q = wrapper
            .step(
                Time::from_value(dt),
                Temperature::from_value(T_ZONE),
                Temperature::from_value(sol_air),
                HeatTransferCoefficient::from_value(H_INT),
                HeatTransferCoefficient::from_value(H_EXT),
            )
            .expect("CTF step")
            .to_value();
        samples.push(q);
    }
    let n = steps_per_day;
    let drift: f64 = (0..n)
        .map(|i| (samples[total_steps - n + i] - samples[total_steps - 2 * n + i]).abs())
        .fold(0.0, f64::max);
    let (amp, phi, mean) = demodulate_offset(&samples[total_steps - n..], 1);
    FdRun {
        amp,
        phi,
        mean,
        drift,
    }
}

/// CTF linear envelope (Issue #3981). The pole-residue coefficient
/// approximation (src/physics/ctf_coefficients.rs) is normalized to the
/// exact U at DC for heavy/medium/multi walls, but its DYNAMIC response has
/// no tight envelope (measured: heavy amp +57%, medium +9.6%, multi +29%),
/// and the LIGHT wall's DC gain has the WRONG SIGN (Issue #4062: q_ss =
/// -5.53 vs +3.69 W/m2 under constant forcing — the y-coefficient
/// abs().max(0) clamping breaks the DC identity when all poles have
/// tau << dt). These envelopes characterize the CURRENT implementation;
/// they will fail (correctly) when CTF is reworked, prompting re-pinning.
#[test]
fn ctf_stays_within_linear_envelope() {
    // DC sanity probe first: constant forcing must converge to U*dT for
    // every wall (superposition demands the periodic mean equal this).
    // Light is the known-defective #4062 case.
    for wall in ["heavy", "medium", "light", "multi"] {
        let mut wrapper = CTFSolverWrapper::new();
        wrapper.initialize(&wall_spec_for(wall)).expect("init");
        let mut q_last = 0.0;
        for _ in 0..30 * 24 {
            q_last = wrapper
                .step(
                    Time::from_value(3600.0),
                    Temperature::from_value(20.0),
                    Temperature::from_value(28.0),
                    HeatTransferCoefficient::from_value(H_INT),
                    HeatTransferCoefficient::from_value(H_EXT),
                )
                .expect("step")
                .to_value();
        }
        let g = golden(&format!("{wall}_ctf"));
        let dc_target = g.u * 8.0;
        if wall == "light" {
            // KNOWN DEFECT #4062: wrong-sign DC gain. Assert the defective
            // magnitude band so the defect is regression-tracked, not hidden.
            assert!(
                q_last < 0.0 && q_last > -8.0,
                "light CTF DC flux {q_last} outside the known #4062 defect band (-8, 0); \
                 if CTF was fixed, tighten this to the physical envelope"
            );
        } else {
            assert!(
                (q_last - dc_target).abs() / dc_target < 0.005,
                "{wall}: CTF DC flux {q_last} vs U*dT {dc_target} exceeds 0.5%"
            );
        }
        eprintln!("[ctf-dc] {wall:6}: q_ss={q_last:9.5}  U*dT={dc_target:9.5}");
    }
    // Steady-periodic envelope vs the analytical ZOH reference. WallSpec
    // order [brick, eps, gypsum] is exterior-first (matches golden orientation).
    for wall in ["heavy", "medium", "light", "multi"] {
        let g = golden(&format!("{wall}_ctf"));
        let run = run_ctf_steady_periodic(wall, 14);
        eprintln!(
            "[ctf-ac] {wall:6}: amp={:9.5} ({:+8.3}%) phi={:9.5} ({:+8.4}) mean={:9.5} ({:+8.3}%) drift={:.1e}",
            run.amp,
            100.0 * (run.amp - g.zoh3600_amp) / g.zoh3600_amp,
            run.phi,
            run.phi - g.zoh3600_phi,
            run.mean,
            100.0 * (run.mean - g.qbar) / g.qbar,
            run.drift
        );
        assert!(
            run.drift < 1e-6 * run.amp,
            "{wall}: CTF periodicity drift {} exceeds budget",
            run.drift
        );
        let (amp_env, phi_env, mean_env) = if wall == "light" {
            (0.10, 0.60, 3.00) // #4062: mean includes the wrong-sign defect
        } else {
            (0.65, 0.45, 0.01)
        };
        let amp_rel = (run.amp - g.zoh3600_amp).abs() / g.zoh3600_amp;
        assert!(
            amp_rel < amp_env,
            "{wall}: CTF amp err {:+.4}% exceeds {:.0}% envelope",
            100.0 * (run.amp - g.zoh3600_amp) / g.zoh3600_amp,
            100.0 * amp_env
        );
        assert!(
            (run.phi - g.zoh3600_phi).abs() < phi_env,
            "{wall}: CTF phi err {:+.4} rad exceeds {:.2} envelope",
            run.phi - g.zoh3600_phi,
            phi_env
        );
        let mean_rel = (run.mean - g.qbar).abs() / g.qbar;
        assert!(
            mean_rel < mean_env,
            "{wall}: CTF mean err {:+.4}% exceeds {:.0}% envelope{}",
            100.0 * (run.mean - g.qbar) / g.qbar,
            100.0 * mean_env,
            if wall == "light" {
                " (known defect #4062)"
            } else {
                ""
            }
        );
    }
}
