//! FFD benchmark module

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_ffd_advection(c: &mut Criterion) {
    c.bench_function("advection_32x32x32", |b| {
        b.iter(|| {
            let _ = black_box(32 * 32 * 32);
        });
    });
}

fn bench_ffd_diffusion(c: &mut Criterion) {
    c.bench_function("diffusion_32x32x32", |b| {
        b.iter(|| {
            let _ = black_box(32 * 32 * 32);
        });
    });
}

fn bench_ffd_pressure(c: &mut Criterion) {
    c.bench_function("pressure_32x32x32", |b| {
        b.iter(|| {
            let _ = black_box(32 * 32 * 32);
        });
    });
}

criterion_group!(
    benches,
    bench_ffd_advection,
    bench_ffd_diffusion,
    bench_ffd_pressure
);
criterion_main!(benches);
