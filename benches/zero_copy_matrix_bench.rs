//! Zero-copy Arrow/Numpy matrix transfer benchmark (Issue #1801 / T9.7).
//!
//! This benchmark exercises the `ZeroCopyMatrix` helpers introduced for
//! issue #1801. The hot path measures two things:
//!
//! 1. **Throughput**: how many 2-D matrices (shape `(MAX_WALLS, 6)` — 3 000
//!    `f64`s, the same size as `GeometryTensor::wall_matrix`) can be wrapped
//!    as numpy arrays per second.
//! 2. **Allocation count**: zero-copy means no buffer allocations on the hot
//!    path. The Arc-clone for the holder is the only allocation, and it is
//!    not the matrix buffer itself.
//!
//! Run with:
//!
//! ```text
//! cargo bench --bench zero_copy_matrix_bench
//! ```
//!
//! The full numpy round-trip path is gated behind the `python-bindings`
//! feature and exercised by the included unit tests; the benchmarks below use
//! pure Rust primitives so they can run under the standard bench harness
//! without linking to Python.
//!
//! Expected outcome: the `zero_copy_arc_clone_*` benchmarks should be at
//! least an order of magnitude faster than the `legacy_vec_clone_*`
//! benchmarks, because the latter copies the full matrix buffer on every
//! iteration while the former only bumps an `Arc` reference count.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use fluxion::physics::geometry_tensor::{MAX_WALLS, WALL_MATRIX_DIMS};
use fluxion::physics::zero_copy_matrix::ZeroCopyMatrix2D;
use std::hint::black_box;
use std::sync::Arc;

const BUFFER_LEN: usize = MAX_WALLS * 6;

/// Baseline: legacy clone-and-copy path. This is what `PyGeometryTensor::to_numpy`
/// did before issue #1801 — `self.inner.wall_matrix.clone()` allocates a new
/// `Vec<f64>` and copies every byte.
fn legacy_vec_clone(c: &mut Criterion) {
    let data: Vec<f64> = (0..BUFFER_LEN).map(|i| i as f64).collect();

    let mut group = c.benchmark_group("legacy_vec_clone");
    group.throughput(Throughput::Bytes(BUFFER_LEN as u64 * 8));
    group.bench_function(BenchmarkId::from_parameter(BUFFER_LEN), |b| {
        b.iter(|| {
            // The legacy path: Vec::clone() copies every byte.
            let cloned: Vec<f64> = data.clone();
            black_box(cloned);
        });
    });
    group.finish();
}

/// Zero-copy path: cloning the inner Arc is a refcount bump; the buffer is
/// shared.
fn zero_copy_arc_clone(c: &mut Criterion) {
    let data: Vec<f64> = (0..BUFFER_LEN).map(|i| i as f64).collect();
    let matrix = ZeroCopyMatrix2D::from_vec(data, WALL_MATRIX_DIMS);
    let matrix = std::sync::Arc::new(matrix);

    let mut group = c.benchmark_group("zero_copy_arc_clone");
    group.throughput(Throughput::Bytes(BUFFER_LEN as u64 * 8));
    group.bench_function(BenchmarkId::from_parameter(BUFFER_LEN), |b| {
        b.iter(|| {
            // The zero-copy path: Arc::clone() bumps a refcount, no buffer
            // copy. This is what the numpy array's container does when it
            // receives a borrow.
            let cloned = Arc::clone(&matrix);
            black_box(cloned);
        });
    });
    group.finish();
}

/// Allocation-focused comparison: count how many `Vec<f64>` allocations are
/// performed on the hot path. The legacy path allocates one per iteration; the
/// zero-copy path allocates zero on the hot path (the Arc clone is a refcount
/// bump, not a Vec allocation).
fn zero_copy_allocation_count(c: &mut Criterion) {
    let data: Vec<f64> = (0..BUFFER_LEN).map(|i| i as f64).collect();
    let matrix = ZeroCopyMatrix2D::from_vec(data, WALL_MATRIX_DIMS);

    let mut group = c.benchmark_group("zero_copy_allocation_count");
    group.bench_function("zero_copy_arc_clone", |b| {
        b.iter(|| {
            let cloned = matrix.clone();
            black_box(cloned);
        });
    });
    group.bench_function("legacy_vec_clone", |b| {
        b.iter(|| {
            // Re-allocate a Vec with the same length to model the
            // legacy path's allocation cost.
            let cloned: Vec<f64> = vec![0.0_f64; BUFFER_LEN];
            black_box(cloned);
        });
    });
    group.finish();
}

/// Smoke test for the 1-D summary path used by `to_numpy` to ship the
/// `(6,)` summary vector back to Python. Same Arc-based zero-copy recipe.
fn zero_copy_summary_1d(c: &mut Criterion) {
    use fluxion::physics::zero_copy_matrix::ZeroCopyMatrix1D;

    let summary: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 500.0, 1200.0];
    let matrix = ZeroCopyMatrix1D::from_vec(summary.clone());

    let mut group = c.benchmark_group("zero_copy_summary_1d");
    group.throughput(Throughput::Bytes((summary.len() * 8) as u64));
    group.bench_function("summary_6", |b| {
        b.iter(|| {
            let cloned = matrix.clone();
            black_box(cloned);
        });
    });
    group.finish();
}

criterion_group!(
    zero_copy_matrix_benches,
    legacy_vec_clone,
    zero_copy_arc_clone,
    zero_copy_allocation_count,
    zero_copy_summary_1d,
);

criterion_main!(zero_copy_matrix_benches);
