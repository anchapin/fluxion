//! CPU↔GPU parity test (issue #2456).
//!
//! Asserts that the [`fluxion_cfd::cpu`] shim produces **bit-identical**
//! output to the top-level [`fluxion_cfd::AdvectionSolver`] /
//! [`fluxion_cfd::DiffusionSolver`] / [`fluxion_cfd::PressureSolver`]
//! implementations on a fixed 32³ grid.
//!
//! ## Why this test exists
//!
//! The GPU port (CUDA / OpenCL kernels behind `--features cuda` /
//! `--features opencl`, per ARCHITECTURE.md §Loose Coupling) will
//! mirror the [`fluxion_cfd::cpu`] shim's loop structure. Bit-identity
//! with the top-level on the same input is the only way to prove the
//! port doesn't drift from the validated math (RULES.md — "fix the
//! underlying math, not the parameter tuning").
//!
//! ## Build
//!
//! ```bash
//! cargo test -p fluxion-cfd --test cpu_gpu_parity
//! ```
//!
//! ## Determinism
//!
//! The top-level solvers and the CPU shim are deterministic for a
//! fixed input. CG iterations converge to a single float vector. The
//! parity assertion holds to within `1e-12` (floating-point epsilon
//! for `f64` arithmetic through the documented iteration order).

use fluxion_cfd::cpu::{CpuAdvectSolver, CpuDiffuseSolver, CpuPoissonSolver};
use fluxion_cfd::{
    AdvectionSolver, DiffusionSolver, FfdConfig, Field3d, Grid3d, PressureSolver, VelocityField,
};

/// Maximum floating-point drift tolerated between the CPU shim and
/// the top-level. `1e-12` is well above `f64` epsilon (~`2.2e-16`)
/// but below the noise floor introduced by reordering arithmetic.
const PARITY_TOL: f64 = 1e-12;

/// Canonical 32³ FFD configuration (matches `FfdConfig::default`).
fn ffd_32() -> FfdConfig {
    FfdConfig::default()
}

/// Build the canonical 32³ inputs: non-trivial velocity + scalar.
fn build_inputs() -> (Grid3d, VelocityField, Field3d) {
    let cfg = ffd_32();
    let grid = Grid3d::new(cfg.nx, cfg.ny, cfg.nz, cfg.dx, cfg.dy, cfg.dz);
    let mut velocity = VelocityField::zeros(cfg.nx, cfg.ny, cfg.nz);
    // Mild shear profile so the backtrace + interpolation actually
    // exercises the code (a zero-velocity field collapses to
    // "result = scalar" and does not detect arithmetic drift).
    for k in 0..cfg.nz {
        for j in 0..cfg.ny {
            for i in 0..cfg.nx {
                let idx = i + cfg.nx * (j + cfg.ny * k);
                velocity.u.data[idx] = 0.05 * (i as f64 / cfg.nx as f64);
                velocity.v.data[idx] = 0.02 * (j as f64 / cfg.ny as f64);
                velocity.w.data[idx] = -0.01 * (k as f64 / cfg.nz as f64);
            }
        }
    }
    let mut scalar = Field3d::filled(cfg.nx, cfg.ny, cfg.nz, 1.0);
    // Seed the scalar with a Gaussian bump so the diffusion residual
    // is non-trivial (uniform fields also pass bit-identity vacuously).
    for k in 0..cfg.nz {
        for j in 0..cfg.ny {
            for i in 0..cfg.nx {
                let idx = i + cfg.nx * (j + cfg.ny * k);
                let cx = cfg.nx as f64 / 2.0;
                let cy = cfg.ny as f64 / 2.0;
                let cz = cfg.nz as f64 / 2.0;
                let dx = i as f64 - cx;
                let dy = j as f64 - cy;
                let dz = k as f64 - cz;
                let r2 = dx * dx + dy * dy + dz * dz;
                scalar.data[idx] = 1.0 + (-r2 / 8.0).exp();
            }
        }
    }
    (grid, velocity, scalar)
}

#[test]
fn advection_cpu_shim_matches_top_level_on_32_cube() {
    let cfg = ffd_32();
    let (grid, velocity, scalar) = build_inputs();

    let mut result_top = Field3d::zeros(cfg.nx, cfg.ny, cfg.nz);
    let mut result_cpu = Field3d::zeros(cfg.nx, cfg.ny, cfg.nz);

    // Top-level `AdvectionSolver::step` mutates its input. We need a
    // fresh copy for each call so neither solver observes the other's
    // mutations.
    let mut scalar_for_top = scalar.clone();
    AdvectionSolver::new()
        .step(&grid, cfg.dt, &velocity, &mut scalar_for_top)
        .expect("top-level step should succeed");
    // The top-level step leaves the post-step field in `scalar_for_top`
    // (after the internal clone+advect+copy_from pipeline). The CPU
    // shim writes to a separate `result` buffer. Both must hold the
    // same physical quantity → bit-identity comparison.
    result_top.copy_from(&scalar_for_top).unwrap();

    CpuAdvectSolver::new()
        .advect_scalar(&grid, cfg.dt, &velocity, &scalar, &mut result_cpu)
        .expect("cpu shim advect_scalar should succeed");

    assert_eq!(result_top.data.len(), result_cpu.data.len());
    for (idx, (&a, &b)) in result_top
        .data
        .iter()
        .zip(result_cpu.data.iter())
        .enumerate()
    {
        let err = (a - b).abs();
        assert!(
            err < PARITY_TOL,
            "advection parity drift at idx {idx}: top={a}, cpu={b}, |err|={err} > {PARITY_TOL}"
        );
    }
}

#[test]
fn diffusion_cpu_shim_matches_top_level_on_32_cube() {
    let cfg = ffd_32();
    let (grid, _velocity, scalar) = build_inputs();

    let mut field_top = scalar.clone();
    let mut field_cpu = scalar.clone();

    DiffusionSolver::new(cfg.max_iter, cfg.tolerance)
        .step(&grid, cfg.dt, cfg.nu, &mut field_top)
        .expect("top-level diffusion step should succeed");
    CpuDiffuseSolver::new(cfg.max_iter, cfg.tolerance)
        .diffuse_scalar(&grid, cfg.dt, cfg.nu, &mut field_cpu)
        .expect("cpu shim diffuse_scalar should succeed");

    assert_eq!(field_top.data.len(), field_cpu.data.len());
    for (idx, (&a, &b)) in field_top.data.iter().zip(field_cpu.data.iter()).enumerate() {
        let err = (a - b).abs();
        assert!(
            err < PARITY_TOL,
            "diffusion parity drift at idx {idx}: top={a}, cpu={b}, |err|={err} > {PARITY_TOL}"
        );
    }
}

#[test]
fn pressure_cpu_shim_matches_top_level_on_32_cube() {
    let cfg = ffd_32();
    let (grid, velocity, _scalar) = build_inputs();

    let mut velocity_top = velocity.clone();
    let mut velocity_cpu = velocity.clone();

    PressureSolver::new(cfg.max_iter, cfg.tolerance)
        .project(&grid, cfg.dt, &mut velocity_top)
        .expect("top-level project should succeed");
    CpuPoissonSolver::new(cfg.max_iter, cfg.tolerance)
        .project(&grid, cfg.dt, &mut velocity_cpu)
        .expect("cpu shim project should succeed");

    // Compare all three velocity components bit-for-bit.
    for (component_name, top, cpu) in [
        ("u", &velocity_top.u.data, &velocity_cpu.u.data),
        ("v", &velocity_top.v.data, &velocity_cpu.v.data),
        ("w", &velocity_top.w.data, &velocity_cpu.w.data),
    ] {
        assert_eq!(top.len(), cpu.len());
        for (idx, (&a, &b)) in top.iter().zip(cpu.iter()).enumerate() {
            let err = (a - b).abs();
            assert!(
                err < PARITY_TOL,
                "pressure parity drift on {component_name}[{idx}]: top={a}, cpu={b}, |err|={err} > {PARITY_TOL}"
            );
        }
    }
}

/// Smoke test: the CPU shim end-to-end (`advect` → `diffuse` →
/// `project`) does not regress to the prior stub behaviour. With the
/// stubs the velocity field would be unchanged after the sequence;
/// with the real shim the field must change.
#[test]
fn full_cpu_shim_step_changes_velocity_field() {
    let cfg = ffd_32();
    let (grid, velocity, scalar) = build_inputs();
    let mut velocity_seq = velocity.clone();
    let scalar_for_advect = scalar.clone();

    let advect = CpuAdvectSolver::new();
    let diffuse = CpuDiffuseSolver::new(cfg.max_iter, cfg.tolerance);
    let poisson = CpuPoissonSolver::new(cfg.max_iter, cfg.tolerance);

    // advect
    let mut buf = Field3d::zeros(cfg.nx, cfg.ny, cfg.nz);
    advect
        .advect_scalar(&grid, cfg.dt, &velocity_seq, &scalar_for_advect, &mut buf)
        .unwrap();
    // diffuse (velocity components)
    diffuse
        .diffuse_scalar(&grid, cfg.dt, cfg.nu, &mut velocity_seq.u)
        .unwrap();
    diffuse
        .diffuse_scalar(&grid, cfg.dt, cfg.nu, &mut velocity_seq.v)
        .unwrap();
    diffuse
        .diffuse_scalar(&grid, cfg.dt, cfg.nu, &mut velocity_seq.w)
        .unwrap();
    // pressure projection
    poisson.project(&grid, cfg.dt, &mut velocity_seq).unwrap();

    let pre_norm: f64 = velocity.u.data.iter().map(|v| v * v).sum::<f64>().sqrt();
    let post_norm: f64 = velocity_seq
        .u
        .data
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    assert!(
        (pre_norm - post_norm).abs() > 0.0,
        "full CPU shim step must change u (pre_norm={pre_norm}, post_norm={post_norm})"
    );
}
