//! ASHRAE 140 §8 (ground-coupled slab-on-grade) — steady-state analytical
//! verification cases for the 2-D ground-conduction domain model.
//!
//! Issue #4270 (foundation slice). The solver under test is
//! [`fluxion::physics::ground_conduction`], a standalone 2-D (x–z)
//! ground-conduction domain model. It is *not* wired into the Section 7
//! floor boundary or the main simulation loop: §8 verifies the ground domain
//! standalone first.
//!
//! # Cases
//!
//! | test | reference |
//! |------|-----------|
//! | `one_dimensional_limit` | closed-form 1-D series resistance |
//! | `two_dimensional_edge`   | closed-form Dirichlet-rectangle Fourier series |
//! | `temperature_field`      | same Fourier series, point-wise |
//! | `energy_balance`         | steady-state closure of the discrete system |
//!
//! Every reference value below is a **closed-form analytical solution
//! derived in this file** (or in the module documentation) — no tabulated or
//! published result numbers are used as references.
//!
//! # Case 1 — one-dimensional limit
//!
//! When the floor footprint spans the whole lateral extent of the domain and
//! the lateral far-field boundaries are adiabatic, the two-dimensional
//! problem degenerates to the one-dimensional steady conduction problem
//!
//! ```text
//! interior air (T_i, film h_i) -> slab layer (d_s, k_s) -> soil (D_g, k_g)
//! -> deep ground (T_g)
//! ```
//!
//! with the exact series-resistance solution
//!
//! ```text
//! R_total = 1/h_i + d_s/k_s + D_g/k_g          [m²·K/W]
//! q''     = (T_i - T_g) / R_total              [W/m²]
//! ```
//!
//! and the exact temperature profile
//!
//! ```text
//! T(z) = T_i - q'' (1/h_i + z/k_s)                    for 0 <= z <= d_s
//! T(z) = T_i - q'' (1/h_i + d_s/k_s + (z-d_s)/k_g)    for d_s <= z <= D
//! ```
//!
//! The finite-volume scheme reproduces a piecewise-linear 1-D profile
//! *exactly* at the cell centres (the face conductance uses the series of the
//! two half-cell resistances, which is exact for a linear profile), so the
//! tolerance below is round-off level rather than a discretisation
//! allowance.
//!
//! # Case 2 — two-dimensional slab edge
//!
//! The 2-D verification case is a 12 m wide floor footprint on a 12 m × 8 m
//! ground domain, with the ASHRAE 140 §8 GC10a boundary conditions (Addendum a
//! to Standard 140-2011, Table 5-29) applied to the cross-section:
//!
//! * slab/soil conductivity equal, `k = 1.9 W/(m·K)`;
//! * slab surface held at `T_i = 30 °C`;
//! * perimeter boundary of width `W = 0.24 m` with a linear temperature
//!   ramp from `T_i` down to the exterior ground surface temperature
//!   `T_o = 10 °C`;
//! * exterior ground surface at `T_o = 10 °C`;
//! * deep ground at `T_g = 10 °C` (bottom and lateral far-field
//!   boundaries).
//!
//! The reference is the classical separated-variable solution of Laplace's
//! equation in a rectangle with Dirichlet data on all four sides
//! (Carslaw & Jaeger, *Conduction of Heat in Solids*, 2nd ed., §7.4 — the
//! Fourier sine series for a rectangle with an arbitrary Dirichlet profile on
//! one side). Writing `theta = T - T_g` and `f(x)` for the prescribed
//! top-surface profile:
//!
//! ```text
//! theta(x, z) = sum_{n>=1} B_n sin(n pi x / L) sinh(n pi (D - z) / L)
//!                              / sinh(n pi D / L)
//! B_n        = (2/L) * integral_0^L f(x) sin(n pi x / L) dx
//! ```
//!
//! `f` is piecewise linear (flat `T_i` over the footprint, linear ramps of
//! width `W` on either side, flat `T_g` elsewhere), so `B_n` has the closed
//! form used by [`fourier_coefficients`] below (the integral of a linear
//! profile times a sine). The downward heat flux at the top surface and the
//! total heat entering the ground under the footprint follow from
//!
//! ```text
//! q(x)  = -k dT/dz|_{z=0} = k sum_n B_n (n pi / L) coth(n pi D / L)
//!                                 sin(n pi x / L)
//! Q     = integral_{x0}^{x1} q(x) dx
//!       = k sum_n B_n coth(n pi D / L)
//!               [cos(n pi x0 / L) - cos(n pi x1 / L)]
//! ```
//!
//! The series is truncated at `N` terms. The truncation error of `Q` is
//! `O(1/N)` (the top-surface profile has slope discontinuities at the ramp
//! ends, so the coefficients decay like `1/n^3` and the *flux* series carries
//! the usual Gibbs oscillations near those two points). At `N = 8000` the
//! truncated `Q` is within `4e-6` (relative) of the `N = 16000` value, which
//! is two orders of magnitude below the finite-volume discretisation error
//! that dominates the comparison.
//!
//! The point-wise temperature comparison deliberately avoids the two ramp
//! neighbourhoods, where the *truncated series* (not the finite-volume
//! solution) carries its Gibbs oscillations.
//!
//! # Tolerances
//!
//! The finite-volume error of the 2-D case was established by a grid
//! refinement study (lateral and vertical refinement, see the
//! `grid_refinement` test and `WORKER_REPORT.md`):
//!
//! | grid | relative error of `Q` |
//! |------|------------------------|
//! | medium (288 × 48 cells) | −0.43 % |
//! | fine   (576 × 48 cells) | −0.22 % |
//!
//! The tolerance is therefore set at **0.5 % for the medium grid** and
//! **0.3 % for the fine grid**, with the additional requirement that the
//! error must shrink under refinement (the refinement trend is asserted, not
//! just the bound). The point-wise temperature tolerance is 0.02 K (the
//! observed maximum deviation away from the ramp kinks is 0.0066 K).

use fluxion::physics::ground_conduction::{
    GridSegment, GroundDomainConfig, GroundSolver, SurfaceCondition, TopCondition, TopSegment,
};

/// ASHRAE 140 §8 GC10a cross-section parameters (Addendum a to Standard
/// 140-2011, Table 5-29).
mod gc10a {
    /// Floor footprint width [m].
    pub const SLAB_WIDTH_M: f64 = 12.0;
    /// Perimeter boundary width [m] (the linear temperature ramp).
    pub const PERIMETER_BOUNDARY_WIDTH_M: f64 = 0.24;
    /// Slab and soil conductivity [W/(m·K)] (equal in GC10a).
    pub const CONDUCTIVITY_W_MK: f64 = 1.9;
    /// Interior slab surface temperature [°C].
    pub const SLAB_SURFACE_TEMPERATURE_C: f64 = 30.0;
    /// Exterior ground surface temperature [°C].
    pub const GROUND_SURFACE_TEMPERATURE_C: f64 = 10.0;
    /// Deep-ground temperature [°C].
    pub const DEEP_GROUND_TEMPERATURE_C: f64 = 10.0;
}

// ---------------------------------------------------------------------------
// Closed-form analytical reference: Dirichlet-rectangle Fourier series.
// ---------------------------------------------------------------------------

/// One linear piece of the prescribed top-surface profile.
#[derive(Debug, Clone, Copy)]
struct ProfilePiece {
    /// Segment start [m].
    x0: f64,
    /// Segment end [m].
    x1: f64,
    /// Temperature at `x0` [°C].
    t0: f64,
    /// Temperature at `x1` [°C].
    t1: f64,
}

/// Closed-form integral of a linear profile times `sin(k x)` over `[x0, x1]`.
///
/// With `s = (t1 - t0) / (x1 - x0)`, the profile is `t0 + s (x - x0)` and
///
/// ```text
/// integral = t0 (cos(k x0) - cos(k x1)) / k
///          + s [ (-x1 cos(k x1)/k + sin(k x1)/k^2)
///              - (-x0 cos(k x0)/k + sin(k x0)/k^2)
///              - x0 (cos(k x0) - cos(k x1))/k ]
/// ```
fn linear_sine_integral(piece: &ProfilePiece, k: f64) -> f64 {
    let (x0, x1, t0, t1) = (piece.x0, piece.x1, piece.t0, piece.t1);
    let s = (t1 - t0) / (x1 - x0);
    let (c0, c1) = ((k * x0).cos(), (k * x1).cos());
    let (s0, s1) = ((k * x0).sin(), (k * x1).sin());
    t0 * (c0 - c1) / k
        + s * ((-x1 * c1 / k + s1 / (k * k)) - (-x0 * c0 / k + s0 / (k * k)) - x0 * (c0 - c1) / k)
}

/// Fourier sine coefficients `B_n` of the top-surface profile.
///
/// `B_n = (2/L) * integral_0^L (f(x) - t_ref) sin(n pi x / L) dx`, evaluated
/// with the closed-form segment integrals above.
fn fourier_coefficients(
    pieces: &[ProfilePiece],
    length_m: f64,
    terms: usize,
    t_ref: f64,
) -> Vec<f64> {
    let mut coefficients = Vec::with_capacity(terms);
    for n in 1..=terms {
        let k = std::f64::consts::PI * (n as f64) / length_m;
        let integral: f64 = pieces
            .iter()
            .map(|piece| {
                let shifted = ProfilePiece {
                    t0: piece.t0 - t_ref,
                    t1: piece.t1 - t_ref,
                    ..*piece
                };
                linear_sine_integral(&shifted, k)
            })
            .sum();
        coefficients.push(2.0 / length_m * integral);
    }
    coefficients
}

/// Temperature of the analytical solution at `(x, z)` [°C].
///
/// The `sinh(k (D - z)) / sinh(k D)` ratio is evaluated in its
/// numerically stable form `exp(-k z) (1 - exp(-2 k (D - z))) / (1 - exp(-2 k D))`.
fn analytical_temperature(
    x: f64,
    z: f64,
    coefficients: &[f64],
    length_m: f64,
    depth_m: f64,
    t_ref: f64,
) -> f64 {
    let mut total = t_ref;
    for (index, coefficient) in coefficients.iter().enumerate() {
        let n = index as f64 + 1.0;
        let k = std::f64::consts::PI * n / length_m;
        let ratio = (-k * z).exp() * (1.0 - (-2.0 * k * (depth_m - z)).exp())
            / (1.0 - (-2.0 * k * depth_m).exp());
        total += coefficient * (k * x).sin() * ratio;
    }
    total
}

/// Total heat entering the ground through the top surface over `[x0, x1]`
/// [W per m of out-of-plane length].
///
/// Closed-form integral of the analytical surface heat flux, as derived in
/// the module documentation:
///
/// ```text
/// Q = k sum_n B_n coth(n pi D / L) [cos(n pi x0 / L) - cos(n pi x1 / L)]
/// ```
fn analytical_total_heat(
    x0: f64,
    x1: f64,
    coefficients: &[f64],
    length_m: f64,
    depth_m: f64,
    conductivity_w_mk: f64,
) -> f64 {
    let mut total = 0.0;
    for (index, coefficient) in coefficients.iter().enumerate() {
        let n = index as f64 + 1.0;
        let k = std::f64::consts::PI * n / length_m;
        total += coefficient / (k * depth_m).tanh() * ((k * x0).cos() - (k * x1).cos());
    }
    conductivity_w_mk * total
}

// ---------------------------------------------------------------------------
// Case configurations
// ---------------------------------------------------------------------------

/// One-dimensional configuration: the footprint spans the whole domain, the
/// lateral boundaries are adiabatic.
fn one_dimensional_config() -> GroundDomainConfig {
    GroundDomainConfig {
        domain_length_m: 4.0,
        domain_depth_m: 4.0,
        slab_x0_m: 0.0,
        slab_x1_m: 4.0,
        slab_thickness_m: 0.2,
        slab_conductivity_w_mk: 1.4,
        ground_conductivity_w_mk: 1.9,
        lateral_segments: vec![GridSegment::Uniform {
            start_m: 0.0,
            end_m: 4.0,
            cells: 8,
        }],
        vertical_segments: vec![
            GridSegment::Uniform {
                start_m: 0.0,
                end_m: 0.2,
                cells: 2,
            },
            GridSegment::Geometric {
                start_m: 0.2,
                end_m: 4.0,
                first_cell_m: 0.05,
                growth: 1.15,
                max_cell_m: 0.4,
            },
        ],
        top_segments: vec![TopSegment {
            x_start_m: 0.0,
            x_end_m: 4.0,
            condition: TopCondition::Robin {
                film_coefficient_w_m2k: 8.29,
                ambient_temperature_c: 30.0,
            },
        }],
        deep_ground: SurfaceCondition::Dirichlet(10.0),
        lateral_boundary: SurfaceCondition::Adiabatic,
        max_iterations: 50_000,
        tolerance: 1.0e-11,
    }
}

/// Lateral segmentation of the 2-D case: fine cells in the ramp region and in
/// a band around the footprint edges, coarse cells in the middle.
fn lateral_segments(scale: usize) -> Vec<GridSegment> {
    let w = gc10a::PERIMETER_BOUNDARY_WIDTH_M;
    let x0 = 4.0;
    let x1 = x0 + gc10a::SLAB_WIDTH_M;
    let band = 0.5;
    vec![
        GridSegment::Uniform {
            start_m: 0.0,
            end_m: x0 - w,
            cells: 32 * scale,
        },
        GridSegment::Uniform {
            start_m: x0 - w,
            end_m: x0,
            cells: 32 * scale,
        },
        GridSegment::Uniform {
            start_m: x0,
            end_m: x0 + band,
            cells: 32 * scale,
        },
        GridSegment::Uniform {
            start_m: x0 + band,
            end_m: x1 - band,
            cells: 96 * scale,
        },
        GridSegment::Uniform {
            start_m: x1 - band,
            end_m: x1,
            cells: 32 * scale,
        },
        GridSegment::Uniform {
            start_m: x1,
            end_m: x1 + w,
            cells: 32 * scale,
        },
        GridSegment::Uniform {
            start_m: x1 + w,
            end_m: 20.0,
            cells: 32 * scale,
        },
    ]
}

/// The 2-D slab-edge configuration (GC10a boundary conditions on a
/// cross-section).
fn two_dimensional_config(scale: usize) -> GroundDomainConfig {
    let w = gc10a::PERIMETER_BOUNDARY_WIDTH_M;
    let x0 = 4.0;
    let x1 = x0 + gc10a::SLAB_WIDTH_M;
    let t_i = gc10a::SLAB_SURFACE_TEMPERATURE_C;
    let t_o = gc10a::GROUND_SURFACE_TEMPERATURE_C;
    GroundDomainConfig {
        domain_length_m: 20.0,
        domain_depth_m: 8.0,
        slab_x0_m: x0,
        slab_x1_m: x1,
        slab_thickness_m: 0.1,
        // GC10a: slab and soil conductivity are equal.
        slab_conductivity_w_mk: gc10a::CONDUCTIVITY_W_MK,
        ground_conductivity_w_mk: gc10a::CONDUCTIVITY_W_MK,
        lateral_segments: lateral_segments(scale),
        vertical_segments: vec![
            GridSegment::Uniform {
                start_m: 0.0,
                end_m: 0.1,
                cells: 10,
            },
            GridSegment::Geometric {
                start_m: 0.1,
                end_m: 8.0,
                first_cell_m: 0.05,
                growth: 1.15,
                max_cell_m: 0.25,
            },
        ],
        top_segments: vec![
            TopSegment {
                x_start_m: 0.0,
                x_end_m: x0 - w,
                condition: TopCondition::Dirichlet(t_o),
            },
            TopSegment {
                x_start_m: x0 - w,
                x_end_m: x0,
                condition: TopCondition::Ramp {
                    from_c: t_o,
                    to_c: t_i,
                },
            },
            TopSegment {
                x_start_m: x0,
                x_end_m: x1,
                condition: TopCondition::Dirichlet(t_i),
            },
            TopSegment {
                x_start_m: x1,
                x_end_m: x1 + w,
                condition: TopCondition::Ramp {
                    from_c: t_i,
                    to_c: t_o,
                },
            },
            TopSegment {
                x_start_m: x1 + w,
                x_end_m: 20.0,
                condition: TopCondition::Dirichlet(t_o),
            },
        ],
        deep_ground: SurfaceCondition::Dirichlet(gc10a::DEEP_GROUND_TEMPERATURE_C),
        lateral_boundary: SurfaceCondition::Dirichlet(gc10a::DEEP_GROUND_TEMPERATURE_C),
        max_iterations: 50_000,
        tolerance: 1.0e-10,
    }
}

/// The prescribed top-surface profile of the 2-D case, as linear pieces.
fn top_profile() -> Vec<ProfilePiece> {
    let w = gc10a::PERIMETER_BOUNDARY_WIDTH_M;
    let x0 = 4.0;
    let x1 = x0 + gc10a::SLAB_WIDTH_M;
    let t_i = gc10a::SLAB_SURFACE_TEMPERATURE_C;
    let t_o = gc10a::GROUND_SURFACE_TEMPERATURE_C;
    vec![
        ProfilePiece {
            x0: 0.0,
            x1: x0 - w,
            t0: t_o,
            t1: t_o,
        },
        ProfilePiece {
            x0: x0 - w,
            x1: x0,
            t0: t_o,
            t1: t_i,
        },
        ProfilePiece {
            x0: x0,
            x1: x1,
            t0: t_i,
            t1: t_i,
        },
        ProfilePiece {
            x0: x1,
            x1: x1 + w,
            t0: t_i,
            t1: t_o,
        },
        ProfilePiece {
            x0: x1 + w,
            x1: 20.0,
            t0: t_o,
            t1: t_o,
        },
    ]
}

/// Analytical reference for the 2-D case.
struct Analytical2d {
    coefficients: Vec<f64>,
    length_m: f64,
    depth_m: f64,
    t_ref: f64,
}

impl Analytical2d {
    fn new(terms: usize) -> Analytical2d {
        let length_m = 20.0;
        let depth_m = 8.0;
        let t_ref = gc10a::GROUND_SURFACE_TEMPERATURE_C;
        Analytical2d {
            coefficients: fourier_coefficients(&top_profile(), length_m, terms, t_ref),
            length_m,
            depth_m,
            t_ref,
        }
    }

    fn total_heat(&self, x0: f64, x1: f64) -> f64 {
        analytical_total_heat(
            x0,
            x1,
            &self.coefficients,
            self.length_m,
            self.depth_m,
            gc10a::CONDUCTIVITY_W_MK,
        )
    }

    fn temperature(&self, x: f64, z: f64) -> f64 {
        analytical_temperature(
            x,
            z,
            &self.coefficients,
            self.length_m,
            self.depth_m,
            self.t_ref,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Case 1 — the 2-D solver reproduces the closed-form 1-D steady conduction
/// solution in the wide/deep slab limit.
#[test]
fn ground_slab_steady_state_one_dimensional_limit_matches_closed_form() {
    let config = one_dimensional_config();
    let solver = GroundSolver::new(config.clone()).expect("valid configuration");
    let solution = solver.solve().expect("steady solve");

    // Closed form: q'' = (T_i - T_g) / (1/h_i + d_s/k_s + D_g/k_g).
    let h_i = 8.29;
    let t_i = 30.0;
    let t_g = 10.0;
    let d_s = 0.2;
    let k_s = 1.4;
    let k_g = 1.9;
    let d_g = config.domain_depth_m - d_s;
    let r_total = 1.0 / h_i + d_s / k_s + d_g / k_g;
    let q_flux = (t_i - t_g) / r_total;
    let expected_total = q_flux * config.domain_length_m;

    let flows = solution.heat_flows(&config);
    println!(
        "1-D limit: slab heat = {:.6} W/m (closed form {:.6} W/m), \
         profile check at {} depths",
        flows.slab_surface_w,
        expected_total,
        solution.grid.z.len()
    );

    // The finite-volume scheme is exact for a piecewise-linear 1-D profile,
    // so the tolerance is round-off level (1e-9 relative).
    assert!(
        (flows.slab_surface_w - expected_total).abs() / expected_total < 1.0e-9,
        "slab heat {} W/m vs closed form {} W/m",
        flows.slab_surface_w,
        expected_total
    );

    // The full temperature profile must match the closed form as well.
    let mut worst = 0.0_f64;
    for (j, z) in solution.grid.z.centers.iter().enumerate() {
        let resistance = if *z < d_s {
            1.0 / h_i + z / k_s
        } else {
            1.0 / h_i + d_s / k_s + (z - d_s) / k_g
        };
        let expected = t_i - q_flux * resistance;
        let actual = solution.cell_temperature(4, j);
        worst = worst.max((actual - expected).abs());
    }
    assert!(
        worst < 1.0e-8,
        "1-D temperature profile deviates by {worst} K from the closed form"
    );

    // Energy balance: the slab heat is the only boundary heat flow.
    assert!(
        flows.net().abs() / expected_total < 1.0e-9,
        "energy balance residual {} W/m",
        flows.net()
    );
}

/// Case 2 — the 2-D slab-edge total heat transfer matches the closed-form
/// Fourier-series reference, and the error shrinks under grid refinement.
#[test]
fn ground_slab_steady_state_two_dimensional_edge_matches_analytical_series() {
    let analytical = Analytical2d::new(8_000);
    let x0 = 4.0;
    let x1 = x0 + gc10a::SLAB_WIDTH_M;
    let reference = analytical.total_heat(x0, x1);

    // Series truncation check: the reference itself must be converged.
    let finer_series = Analytical2d::new(16_000);
    let series_truncation = (finer_series.total_heat(x0, x1) - reference).abs() / reference;
    assert!(
        series_truncation < 1.0e-5,
        "Fourier series truncation error {series_truncation} is not negligible"
    );

    // Medium grid.
    let medium_config = two_dimensional_config(1);
    let medium = GroundSolver::new(medium_config.clone())
        .expect("valid configuration")
        .solve()
        .expect("steady solve");
    let medium_flows = medium.heat_flows(&medium_config);
    let medium_error = (medium_flows.slab_surface_w - reference) / reference;

    // Fine grid (all lateral cell counts doubled).
    let fine_config = two_dimensional_config(2);
    let fine = GroundSolver::new(fine_config.clone())
        .expect("valid configuration")
        .solve()
        .expect("steady solve");
    let fine_flows = fine.heat_flows(&fine_config);
    let fine_error = (fine_flows.slab_surface_w - reference) / reference;

    // Effective 2-D shape factor of the strip (heat per unit conductivity
    // and unit temperature difference): the classical "shape factor"
    // reading of the same number.
    let shape_factor = reference
        / (gc10a::CONDUCTIVITY_W_MK
            * (gc10a::SLAB_SURFACE_TEMPERATURE_C - gc10a::GROUND_SURFACE_TEMPERATURE_C));

    println!(
        "2-D slab edge: analytical Q = {reference:.5} W/m (shape factor {shape_factor:.4} W/(m·K)); \
         medium grid {} cells: Q = {:.5} W/m (err {medium_error:+.4} %); \
         fine grid {} cells: Q = {:.5} W/m (err {fine_error:+.4} %)",
        medium.grid.x.len() * medium.grid.z.len(),
        medium_flows.slab_surface_w,
        fine.grid.x.len() * fine.grid.z.len(),
        fine_flows.slab_surface_w,
    );

    // Tolerances from the grid refinement study (see the module documentation
    // and WORKER_REPORT.md): 0.5 % on the medium grid, 0.3 % on the fine grid.
    assert!(
        medium_error.abs() < 5.0e-3,
        "medium-grid error {medium_error} exceeds the 0.5 % tolerance"
    );
    assert!(
        fine_error.abs() < 3.0e-3,
        "fine-grid error {fine_error} exceeds the 0.3 % tolerance"
    );
    // The refinement trend must hold: the error shrinks when the grid is
    // refined (this is the grid-resolution study the tolerance is based on).
    assert!(
        fine_error.abs() < medium_error.abs(),
        "grid refinement did not reduce the error: medium {medium_error}, fine {fine_error}"
    );
}

/// Case 3 — the analytical surface-temperature distribution is reproduced by
/// the 2-D solver.
#[test]
fn ground_slab_steady_state_surface_temperature_matches_analytical_series() {
    let analytical = Analytical2d::new(8_000);
    let config = two_dimensional_config(1);
    let solution = GroundSolver::new(config)
        .expect("valid configuration")
        .solve()
        .expect("steady solve");

    let w = gc10a::PERIMETER_BOUNDARY_WIDTH_M;
    let x0 = 4.0;
    let x1 = x0 + gc10a::SLAB_WIDTH_M;

    // Probe points on horizontal lines, skipping the ramp neighbourhoods
    // where the truncated Fourier series carries its Gibbs oscillations
    // (the finite-volume solution itself is smooth there).
    let mut worst = 0.0_f64;
    let mut worst_at = (0.0, 0.0);
    let mut probes = 0;
    for z in [0.5_f64, 1.0, 2.0, 4.0] {
        let mut x = 0.5;
        while x < 20.0 {
            let near_ramp = (x - (x0 - w)).abs() < 0.3
                || (x - x0).abs() < 0.3
                || (x - x1).abs() < 0.3
                || (x - (x1 + w)).abs() < 0.3;
            if !near_ramp {
                let deviation =
                    (solution.temperature_at(x, z) - analytical.temperature(x, z)).abs();
                if deviation > worst {
                    worst = deviation;
                    worst_at = (x, z);
                }
                probes += 1;
            }
            x += 0.25;
        }
    }
    println!(
        "2-D temperature field: max |dT| = {worst:.6} K over {probes} probes (worst at x={}, z={})",
        worst_at.0, worst_at.1
    );
    assert!(probes > 100, "too few probe points ({probes})");
    assert!(
        worst < 2.0e-2,
        "temperature field deviates by {worst} K at {worst_at:?} (tolerance 0.02 K)"
    );
}

/// Case 4 — energy-balance closure: at steady state the heat entering the
/// domain equals the heat leaving it.
#[test]
fn ground_slab_steady_state_energy_balance_closes() {
    for scale in [1_usize, 2] {
        let config = two_dimensional_config(scale);
        let solution = GroundSolver::new(config.clone())
            .expect("valid configuration")
            .solve()
            .expect("steady solve");
        let flows = solution.heat_flows(&config);
        let scale_reference = flows.slab_surface_w.abs() + flows.ground_surface_w.abs();
        println!(
            "energy balance (scale {scale}): slab {:+.6}, ground {:+.6}, deep {:+.6}, \
             lateral {:+.6} W/m -> net {:+.3e}",
            flows.slab_surface_w,
            flows.ground_surface_w,
            flows.deep_ground_w,
            flows.lateral_w,
            flows.net()
        );
        assert!(
            flows.net().abs() / scale_reference < 1.0e-8,
            "energy balance residual {} W/m (scale {scale_reference})",
            flows.net()
        );
    }
}
