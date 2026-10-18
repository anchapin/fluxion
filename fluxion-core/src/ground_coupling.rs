//! ASHRAE 140 §8 ground-coupled slab analytical verification — Case GC10a.
//!
//! Issue #4270. Case GC10a is the true analytical verification case of
//! Standard 140 §8 (introduced by Addendum a to Standard 140-2011,
//! §5.2.4.2.1 / Annex B11.2.1): steady-state 3-D conduction from a
//! 12 m × 12 m floor slab whose surface is held at Ti = 30 °C, across a
//! perimeter surface boundary of width W = 0.24 m with a linear surface
//! temperature ramp from Ti to the exterior ground surface temperature
//! To = 10 °C, into a semi-infinite solid at uniform conductivity
//! k = 1.9 W/(m·K), slab and soil equal, no radiation, deep ground at To.
//!
//! The mathematical truth standard is the closed-form Delsante, Stokes
//! and Walsh (1983) solution, Annex B11-1/B11-2:
//!
//! q = k (Ti − To) F(L, B, W) / π, with target 2433 W ± 0.5 %.
//!
//! Two independent implementations live here:
//!
//! 1. [`gc10a_closed_form_watts`] — the closed form itself, transcribed
//!    from the rendered B11-2 equation and verified against the published
//!    2433 W (0.017 % off).
//! 2. [`gc10a_fd_flux_watts`] — a cell-centered finite-volume measuring
//!    solver on the quarter domain (adiabatic symmetry faces at x = 0 and
//!    y = 0, exterior ground surface at To, deep ground at To), solved by
//!    Jacobi-preconditioned conjugate gradients. At d_fine = 0.1 m it
//!    measures 2429.8 W, 0.12 % below the closed form — inside the
//!    release-gate band of ± 0.5 %.
//!
//! # What this ground model is and is not validated for
//!
//! Validated: steady-state conduction under prescribed surface
//! temperatures for a semi-infinite, homogeneous ground with
//! k_slab = k_soil and no radiative exchange — exactly the GC10a
//! boundary conditions. The [`Gc10aConfig::default`] mesh parameters
//! were selected by refinement (0.25 m: −4 %, 0.15 m: −0.7 %,
//! 0.1 m: −0.12 %) and domain-sensitivity (far field 50 m / depth 25 m
//! vs 60 m / 30 m: 0.07 %) studies.
//!
//! Not validated: transient or periodic operation (the §8 b-series),
//! convectively coupled surfaces (GC30b and later), layered or
//! moisture-dependent soil, finite-thickness slabs, or any configuration
//! where the ramp-ring/singularity mesh sensitivity has not been
//! re-demonstrated. This module is the §8 measuring case only; the §11
//! QASv2 anchors live in [`crate::qas_airside`] (Issue #4274). The §8
//! b-series (GC10b transient, GC30/GC60 periodic and convectively-coupled
//! variants) remains unimplemented: its boundary-condition specification is
//! not derivable from the in-repo sources, so those cases stay open on
//! #4274 rather than improvised.

use crate::error::FluxionError;

/// GC10a input parameters (Table 5-29, Addendum a to Standard 140-2011)
/// plus the measuring-solver mesh parameters (§5.2.4.2.1.2 precision rules).
#[derive(Debug, Clone, PartialEq)]
pub struct Gc10aConfig {
    /// Slab length L (m).
    pub slab_length_m: f64,
    /// Slab width B (m).
    pub slab_width_m: f64,
    /// Perimeter surface boundary width W (m).
    pub perimeter_boundary_width_m: f64,
    /// Conductivity of slab and soil k (W/(m·K)).
    pub conductivity_w_mk: f64,
    /// Interior slab surface temperature Ti (°C).
    pub slab_surface_temperature_c: f64,
    /// Exterior ground surface temperature To (°C).
    pub ground_surface_temperature_c: f64,
    /// Fine cell size over the slab and ramp region (m).
    pub fine_cell_m: f64,
    /// Semi-infinite truncation: distance from the slab edge to the far-field
    /// Dirichlet boundary (m).
    pub far_field_m: f64,
    /// Semi-infinite truncation: depth of the deep-ground Dirichlet boundary (m).
    pub ground_depth_m: f64,
    /// Cell growth factor and cap outside the fine region.
    pub cell_growth: f64,
    pub max_cell_m: f64,
}

impl Default for Gc10aConfig {
    fn default() -> Self {
        Self {
            slab_length_m: 12.0,
            slab_width_m: 12.0,
            perimeter_boundary_width_m: 0.24,
            conductivity_w_mk: 1.9,
            slab_surface_temperature_c: 30.0,
            ground_surface_temperature_c: 10.0,
            fine_cell_m: 0.1,
            far_field_m: 50.0,
            ground_depth_m: 25.0,
            cell_growth: 1.5,
            max_cell_m: 8.0,
        }
    }
}

/// Result of one GC10a measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct Gc10aResult {
    /// Total steady-state floor conduction into the ground (W), full slab.
    pub floor_conduction_w: f64,
    /// Number of finite-volume unknowns solved.
    pub unknowns: usize,
    /// Conjugate-gradient iterations used.
    pub iterations: u64,
    /// Final relative residual ||A x − b|| / ||b||.
    pub relative_residual: f64,
}

/// Half-cell axis: cell centers and widths from 0 to `far`, fine spacing
/// `d_fine` until `fine_to`, then geometric growth capped at `cap`.
fn cell_axis(fine_to: f64, d_fine: f64, growth: f64, far: f64, cap: f64) -> (Vec<f64>, Vec<f64>) {
    let mut edges = vec![0.0];
    let mut x = 0.0_f64;
    let mut d = d_fine;
    while x < far - 1e-12 {
        if x >= fine_to {
            d = (d * growth).min(cap);
        }
        d = d.min(far - x);
        x += d;
        edges.push(x);
    }
    let n = edges.len() - 1;
    let widths = (0..n).map(|i| edges[i + 1] - edges[i]).collect::<Vec<_>>();
    let centers = (0..n).map(|i| (edges[i] + edges[i + 1]) / 2.0).collect();
    (centers, widths)
}

/// Harmonic mean of two adjacent cell widths.
#[inline]
fn harmonic(a: f64, b: f64) -> f64 {
    2.0 * a * b / (a + b)
}

/// CSR sparse matrix for the symmetric SPD quarter-domain system.
struct Csr {
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<f64>,
}

/// Case GC10a closed-form solution, Annex B11-1 with B11-2 (Delsante,
/// Stokes and Walsh 1983). Returns the total steady-state floor conduction
/// in W for the Table 5-29 parameters of `cfg`.
///
/// The transcription was verified against the published GC10a analytical
/// value of 2433 W (deviation 0.017 %); the standard states F to 7 terms
/// families whose G = L − B terms vanish identically for the square case.
pub fn gc10a_closed_form_watts(cfg: &Gc10aConfig) -> f64 {
    let l = cfg.slab_length_m;
    let b = cfg.slab_width_m;
    let w = cfg.perimeter_boundary_width_m;
    let g = l - b;
    let d = l + b;
    let sqrt2 = 2.0_f64.sqrt();
    let ln = f64::ln;
    let s = f64::sqrt;
    // (B11-2), terms in the printed order of the addendum.
    let mut f = (2.0 + (l + b) / w) * s((l + w) * (l + w) + (b + w) * (b + w));
    f += -sqrt2 * (1.0 + l / (2.0 * w)) * s(l * l + (l + 2.0 * w) * (l + 2.0 * w));
    f += -sqrt2 * (1.0 + b / (2.0 * w)) * s(b * b + (b + 2.0 * w) * (b + 2.0 * w));
    f += -((l + b) / w) * s(l * l + b * b);
    f += ((l * l + b * b) / w) * (1.0 + sqrt2 * ln(sqrt2 - 1.0));
    f += 2.0 * w * (sqrt2 + ln(sqrt2 - 1.0));
    f += -(sqrt2 * g * g / w)
        * ln((s(g * g + (d + 2.0 * w) * (d + 2.0 * w)) + d + 2.0 * w) / (s(g * g + d * d) + d));
    f += ((g * g - (b + w) * (b + w)) / w)
        * ln((s((l + w) * (l + w) + (b + w) * (b + w)) + b + w) / (l + w));
    f += ((g * g - (l + w) * (l + w)) / w)
        * ln((s((l + w) * (l + w) + (b + w) * (b + w)) + l + w) / (b + w));
    f += (l * (2.0 * b - l) / w) * ln((s(l * l + b * b) + b) / l);
    f += (b * (2.0 * l - b) / w) * ln((s(l * l + b * b) + l) / b);
    f += -((l * l - w * w) / w) * ln((s(w * w + (l + w) * (l + w)) + w) / (l + w));
    f += -((b * b - w * w) / w) * ln((s(w * w + (b + w) * (b + w)) + w) / (b + w));
    f += (2.0 * l + w) * ln((s(w * w + (l + w) * (l + w)) + l + w) / w);
    f += (2.0 * b + w) * ln((s(w * w + (b + w) * (b + w)) + b + w) / w);
    f += (sqrt2 * l * l / w) * ln((s(l * l + (l + 2.0 * w) * (l + 2.0 * w)) + l + 2.0 * w) / l);
    f += (sqrt2 * b * b / w) * ln((s(b * b + (b + 2.0 * w) * (b + 2.0 * w)) + b + 2.0 * w) / b);
    cfg.conductivity_w_mk * (cfg.slab_surface_temperature_c - cfg.ground_surface_temperature_c) * f
        / std::f64::consts::PI
}

/// Surface temperature field on the quarter domain (x ≥ 0, y ≥ 0): Ti on
/// the slab (x ≤ L/2 and y ≤ B/2 from the symmetry corner), linear ramp
/// Ti → To across the perimeter boundary of width W, then To everywhere.
fn surface_temperature(cfg: &Gc10aConfig, xc: &[f64], yc: &[f64]) -> Vec<f64> {
    let lq = cfg.slab_length_m / 2.0;
    let bq = cfg.slab_width_m / 2.0;
    let w = cfg.perimeter_boundary_width_m;
    let ti = cfg.slab_surface_temperature_c;
    let to = cfg.ground_surface_temperature_c;
    let nx = xc.len();
    let ny = yc.len();
    let mut t = vec![0.0; nx * ny];
    for iy in 0..ny {
        for ix in 0..nx {
            let dx = (xc[ix] - lq).max(0.0);
            let dy = (yc[iy] - bq).max(0.0);
            let d = dx.max(dy);
            let v = if d <= 0.0 {
                ti
            } else {
                ti + (d.min(w) / w) * (to - ti)
            };
            t[iy * nx + ix] = v;
        }
    }
    t
}

/// Case GC10a measuring solver: quarter-domain cell-centered finite
/// volume, Jacobi-preconditioned conjugate gradients. Returns the total
/// floor conduction for the full slab (quarter-domain integral × 4).
///
/// The discretization is second-order away from the Ti/To junction, where
/// the prescribed linear ramp has a slope discontinuity; mesh-refinement
/// studies under [`Gc10aConfig::default`] document its effect (module docs).
pub fn gc10a_fd_flux_watts(cfg: &Gc10aConfig) -> Result<Gc10aResult, FluxionError> {
    if cfg.perimeter_boundary_width_m <= 0.0
        || cfg.conductivity_w_mk <= 0.0
        || cfg.fine_cell_m <= 0.0
    {
        return Err(FluxionError::Validation(
            "GC10a: width, conductivity and fine cell must be positive".into(),
        ));
    }
    let growth_to = 13.0_f64.max(cfg.slab_length_m / 2.0 + 1.0);
    let (xc, wx) = cell_axis(
        growth_to,
        cfg.fine_cell_m,
        cfg.cell_growth,
        cfg.far_field_m,
        cfg.max_cell_m,
    );
    let (zc, wz) = cell_axis(
        2.0,
        cfg.fine_cell_m,
        cfg.cell_growth,
        cfg.ground_depth_m,
        cfg.max_cell_m,
    );
    let nx = xc.len();
    let nz = zc.len();
    let n = nx * nx * nz;
    let s_t = surface_temperature(cfg, &xc, &xc);
    let k = cfg.conductivity_w_mk;
    let to = cfg.ground_surface_temperature_c;

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    let mut rhs = vec![0.0_f64; n];

    #[inline]
    fn id(ix: usize, iy: usize, iz: usize, nx: usize) -> usize {
        ix + iy * nx + iz * nx * nx
    }

    for iz in 0..nz {
        for iy in 0..nx {
            for ix in 0..nx {
                let r = id(ix, iy, iz, nx);
                row_ptr.push(col_idx.len());
                let mut diag = 0.0;
                // x faces: mirror (adiabatic) at ix == 0, Dirichlet To at nx-1
                for dx in [0usize, 1usize] {
                    let jx = if dx == 0 { ix.wrapping_sub(1) } else { ix + 1 };
                    if dx == 0 && ix == 0 {
                        continue;
                    }
                    if jx > nx - 1 {
                        let a = 2.0 * k * wx[iy] * wz[iz] / wx[ix];
                        rhs[r] += a * to;
                        diag += a;
                    } else {
                        let a = k * wx[iy] * wz[iz] / harmonic(wx[ix], wx[jx]);
                        col_idx.push(id(jx, iy, iz, nx));
                        values.push(-a);
                        diag += a;
                    }
                }
                for dy in [0usize, 1usize] {
                    let jy = if dy == 0 { iy.wrapping_sub(1) } else { iy + 1 };
                    if dy == 0 && iy == 0 {
                        continue;
                    }
                    if jy > nx - 1 {
                        let a = 2.0 * k * wx[ix] * wz[iz] / wx[iy];
                        rhs[r] += a * to;
                        diag += a;
                    } else {
                        let a = k * wx[ix] * wz[iz] / harmonic(wx[iy], wx[jy]);
                        col_idx.push(id(ix, jy, iz, nx));
                        values.push(-a);
                        diag += a;
                    }
                }
                for dz in [0usize, 1usize] {
                    let jz = if dz == 0 { iz.wrapping_sub(1) } else { iz + 1 };
                    if dz == 0 && iz == 0 {
                        // ground surface Dirichlet via half cell
                        let a = 2.0 * k * wx[ix] * wx[iy] / wz[iz];
                        rhs[r] += a * s_t[iy * nx + ix];
                        diag += a;
                        continue;
                    }
                    if jz > nz - 1 {
                        let a = 2.0 * k * wx[ix] * wx[iy] / wz[iz];
                        rhs[r] += a * to;
                        diag += a;
                    } else {
                        let a = k * wx[ix] * wx[iy] / harmonic(wz[iz], wz[jz]);
                        col_idx.push(id(ix, iy, jz, nx));
                        values.push(-a);
                        diag += a;
                    }
                }
                col_idx.push(r);
                values.push(diag);
            }
        }
    }
    row_ptr.push(col_idx.len());
    let a = Csr {
        row_ptr,
        col_idx,
        values,
    };

    // Jacobi-preconditioned conjugate gradients.
    let mut x = vec![to; n];
    let mut r = rhs.clone();
    {
        // r = b − A x
        for i in 0..n {
            let s = spmv_row(&a, &x, i);
            r[i] = rhs[i] - s;
        }
    }
    let b_norm = dot(&rhs, &rhs).sqrt();
    if b_norm == 0.0 {
        return Err(FluxionError::Validation("GC10a: zero RHS".into()));
    }
    let mut p = r.clone();
    let mut z = vec![0.0; n];
    let mut rz = dot(&r, &r);
    let mut iterations = 0u64;
    let tol = 1e-9_f64;
    let mut rel = 1.0_f64;
    while rel > tol {
        for i in 0..n {
            let diag = a.values[a.row_ptr[i + 1] - 1];
            z[i] = r[i] / diag;
        }
        let zr = dot(&z, &r);
        if iterations > 0 {
            let beta = zr / rz;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
        } else {
            p.copy_from_slice(&z);
        }
        rz = zr;
        let mut ap = vec![0.0; n];
        for i in 0..n {
            ap[i] = spmv_row(&a, &p, i);
        }
        let pap = dot(&p, &ap);
        let alpha = rz / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        rz = dot(&r, &r);
        iterations += 1;
        rel = rz.sqrt() / b_norm;
        if iterations > 200_000 {
            return Err(FluxionError::Validation(format!(
                "GC10a: CG did not converge in 200000 iterations (rel {rel:.3e})"
            )));
        }
    }

    // Floor conduction: half-cell face gradient over the slab cells, ×4.
    let mut q_quarter = 0.0;
    for iy in 0..nx {
        for ix in 0..nx {
            if s_t[iy * nx + ix] == cfg.slab_surface_temperature_c {
                q_quarter += k * wx[ix] * wx[iy] * (s_t[iy * nx + ix] - x[id(ix, iy, 0, nx)])
                    / (wz[0] / 2.0);
            }
        }
    }
    Ok(Gc10aResult {
        floor_conduction_w: 4.0 * q_quarter,
        unknowns: n,
        iterations,
        relative_residual: rel,
    })
}

#[inline]
fn spmv_row(a: &Csr, x: &[f64], i: usize) -> f64 {
    let lo = a.row_ptr[i];
    let hi = a.row_ptr[i + 1];
    let mut s = 0.0;
    for j in lo..hi {
        s += a.values[j] * x[a.col_idx[j]];
    }
    s
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_form_matches_published_2433w() {
        let cfg = Gc10aConfig::default();
        let q = gc10a_closed_form_watts(&cfg);
        let dev = (q - 2433.0).abs() / 2433.0 * 100.0;
        assert!(
            dev < 0.1,
            "closed form {q:.2} W deviates {dev:.3}% from 2433 W"
        );
    }

    #[test]
    fn closed_form_is_deterministic() {
        let cfg = Gc10aConfig::default();
        assert_eq!(gc10a_closed_form_watts(&cfg), gc10a_closed_form_watts(&cfg));
    }

    /// Coarse-mesh sanity only (−4 % at 0.25 m); the gate rests on the
    /// fine measurement committed to analytical_results.json, produced by
    /// `cargo run --release --example section8_gc10a_report`.
    #[test]
    fn fd_coarse_mesh_sanity() {
        let cfg = Gc10aConfig {
            fine_cell_m: 0.25,
            ..Gc10aConfig::default()
        };
        let r = gc10a_fd_flux_watts(&cfg).expect("solve");
        let closed = gc10a_closed_form_watts(&Gc10aConfig::default());
        let dev = (r.floor_conduction_w - closed).abs() / closed * 100.0;
        assert!(
            dev < 6.0,
            "coarse FD {r:?} deviates {dev:.2}% from closed form"
        );
    }
}
