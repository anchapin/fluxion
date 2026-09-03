//! FFD benchmark module — measures real CPU solver cost on a fixed
//! 32³ grid (issue #2456).
//!
//! Per the issue's "First Step", the prior criterion bodies returned
//! `black_box(32 * 32 * 32)` (integer multiplication cost), so the
//! Performance Dashboard received `0.0 ns/iter` for the FFD track.
//! This file replaces each body with a real solver call against the
//! [`fluxion_cfd::cpu`] shim so the criterion harness emits a real
//! ns/iter that the Performance Dashboard can collect.
//!
//! The 32³ grid is the canonical FFD benchmark size (`FfdConfig::default`).
//! Run with:
//!
//! ```bash
//! cargo bench -p fluxion-cfd --bench ffd_bench
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fluxion_cfd::cpu::{CpuAdvectSolver, CpuDiffuseSolver, CpuPoissonSolver};
use fluxion_cfd::{FfdConfig, Field3d, Grid3d, VelocityField};
use std::hint::black_box;

/// Build the canonical 32³ FFD inputs once. Setup cost (allocation,
/// initialization) is paid outside the `b.iter` loop so the criterion
/// harness measures only the solver call cost.
fn build_inputs() -> (Grid3d, VelocityField, Field3d, Field3d) {
    let cfg = FfdConfig::default();
    let grid = Grid3d::new(cfg.nx, cfg.ny, cfg.nz, cfg.dx, cfg.dy, cfg.dz);
    let velocity = VelocityField::zeros(cfg.nx, cfg.ny, cfg.nz);
    let scalar = Field3d::filled(cfg.nx, cfg.ny, cfg.nz, 1.0);
    let result = Field3d::zeros(cfg.nx, cfg.ny, cfg.nz);
    (grid, velocity, scalar, result)
}

fn bench_ffd_advection(c: &mut Criterion) {
    let cfg = FfdConfig::default();
    let (grid, velocity, scalar, mut result) = build_inputs();
    let solver = CpuAdvectSolver::new();
    c.bench_with_input(
        BenchmarkId::new("advection", format!("{}x{}x{}", cfg.nx, cfg.ny, cfg.nz)),
        &cfg,
        |b, _cfg| {
            b.iter(|| {
                solver
                    .advect_scalar(&grid, cfg.dt, &velocity, &scalar, &mut result)
                    .expect("advect_scalar should succeed");
                black_box(&result);
            });
        },
    );
}

fn bench_ffd_diffusion(c: &mut Criterion) {
    let cfg = FfdConfig::default();
    let (grid, _velocity, mut scalar, _result) = build_inputs();
    let solver = CpuDiffuseSolver::default();
    c.bench_with_input(
        BenchmarkId::new("diffusion", format!("{}x{}x{}", cfg.nx, cfg.ny, cfg.nz)),
        &cfg,
        |b, _cfg| {
            b.iter(|| {
                solver
                    .diffuse_scalar(&grid, cfg.dt, cfg.nu, &mut scalar)
                    .expect("diffuse_scalar should succeed");
                black_box(&scalar);
            });
        },
    );
}

fn bench_ffd_pressure(c: &mut Criterion) {
    let cfg = FfdConfig::default();
    let (grid, mut velocity, _scalar, _result) = build_inputs();
    let solver = CpuPoissonSolver::default();
    c.bench_with_input(
        BenchmarkId::new("pressure", format!("{}x{}x{}", cfg.nx, cfg.ny, cfg.nz)),
        &cfg,
        |b, _cfg| {
            b.iter(|| {
                solver
                    .project(&grid, cfg.dt, &mut velocity)
                    .expect("project should succeed");
                black_box(&velocity);
            });
        },
    );
}

criterion_group!(
    benches,
    bench_ffd_advection,
    bench_ffd_diffusion,
    bench_ffd_pressure
);
criterion_main!(benches);
