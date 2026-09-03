//! View Factor Performance Benchmark (Issue #2028)
//!
//! Measures Monte Carlo ray-tracing view factor computation across 100 random
//! surface pairs and reports mean / median / p95 latency per pair.
//!
//! Target: **< 50 ms per surface pair**.
//!
//! Run: `cargo bench -p fluxion-city --bench view_factor_perf`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fluxion_city::{MonteCarloViewFactor, Surface3D};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

// ────────────────── random surface pair generation ──────────────────

/// Generate `count` random surface pairs simulating urban building facades.
///
/// Uses a fixed seed so results are reproducible across runs.
fn generate_random_pairs(count: usize) -> Vec<(Surface3D, Surface3D)> {
    let mut rng = StdRng::seed_from_u64(2028);
    let mut pairs = Vec::with_capacity(count);

    for _ in 0..count {
        // Source surface: random position in a 100 m urban block.
        let cx = rng.random_range(0.0..100.0);
        let cy = rng.random_range(0.0..100.0);
        let cz = rng.random_range(3.0..30.0);
        let width = rng.random_range(3.0..20.0);
        let height = rng.random_range(3.0..15.0);

        // Random orientation: pick a horizontal facing direction.
        let azimuth = rng.random_range(0.0..std::f64::consts::TAU);
        // tangent_u is horizontal (in xy-plane), tangent_v is vertical (+z).
        let tu = [azimuth.cos(), azimuth.sin(), 0.0];
        let tv = [0.0, 0.0, 1.0];
        let surf_i = Surface3D::new([cx, cy, cz], tu, tv, width, height).unwrap();

        // Target surface: offset along the facing normal by a random distance.
        let normal_i = surf_i.normal();
        let distance = rng.random_range(2.0..50.0);
        let tx = cx + normal_i[0] * distance;
        let ty = cy + normal_i[1] * distance;
        let tz = cz + normal_i[2] * distance + rng.random_range(-5.0..5.0);
        let tw = rng.random_range(3.0..20.0);
        let th = rng.random_range(3.0..15.0);

        // Target faces back toward source (normal = -normal_i, roughly).
        let t_normal = [-normal_i[0], -normal_i[1], -normal_i[2]];
        // Build orthonormal basis from the facing normal.
        let (ttu, ttv) = orthonormal_basis_from_normal(&t_normal);
        let surf_j = Surface3D::new([tx, ty, tz], ttu, ttv, tw, th).unwrap();

        pairs.push((surf_i, surf_j));
    }
    pairs
}

/// Build an orthonormal (tangent_u, tangent_v) pair from a unit normal.
fn orthonormal_basis_from_normal(n: &[f64; 3]) -> ([f64; 3], [f64; 3]) {
    // Pick a helper vector not parallel to n.
    let helper = if n[2].abs() < 0.9 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    // u = normalize(cross(helper, n))
    let u = [
        helper[1] * n[2] - helper[2] * n[1],
        helper[2] * n[0] - helper[0] * n[2],
        helper[0] * n[1] - helper[1] * n[0],
    ];
    let u_len = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    let tu = [u[0] / u_len, u[1] / u_len, u[2] / u_len];
    // v = cross(n, u)
    let v = [
        n[1] * tu[2] - n[2] * tu[1],
        n[2] * tu[0] - n[0] * tu[2],
        n[0] * tu[1] - n[1] * tu[0],
    ];
    (tu, v)
}

// ───────────────────── criterion benchmarks ─────────────────────────

/// Benchmark a single typical pair (criterion reports mean / median natively).
fn bench_single_pair(c: &mut Criterion) {
    let pairs = generate_random_pairs(100);
    // Use the median-difficulty pair (index 50) as representative.
    let (ref si, ref sj) = pairs[50];
    let mc = MonteCarloViewFactor::default();

    c.bench_function("view_factor/single_pair_10k_rays", |b| {
        b.iter(|| {
            std::hint::black_box(
                mc.compute(std::hint::black_box(si), std::hint::black_box(sj))
                    .unwrap(),
            );
        });
    });
}

/// Benchmark different ray counts to show the cost / accuracy trade-off.
fn bench_ray_counts(c: &mut Criterion) {
    let pairs = generate_random_pairs(100);
    let (ref si, ref sj) = pairs[50];

    let mut group = c.benchmark_group("view_factor/ray_count");
    for &rays in &[1_000usize, 5_000, 10_000, 50_000, 100_000] {
        let mc = MonteCarloViewFactor::new(rays).with_adaptive(false);
        group.bench_with_input(BenchmarkId::from_parameter(rays), &rays, |b, &_| {
            b.iter(|| {
                std::hint::black_box(
                    mc.compute(std::hint::black_box(si), std::hint::black_box(sj))
                        .unwrap(),
                );
            });
        });
    }
    group.finish();
}

/// Benchmark all 100 pairs as a batch (total wall-clock).
fn bench_100_pairs_batch(c: &mut Criterion) {
    let pairs = generate_random_pairs(100);
    let mc = MonteCarloViewFactor::default();

    c.bench_function("view_factor/100_pairs_batch", |b| {
        b.iter(|| {
            for (si, sj) in std::hint::black_box(&pairs) {
                std::hint::black_box(
                    mc.compute(std::hint::black_box(si), std::hint::black_box(sj))
                        .unwrap(),
                );
            }
        });
    });
}

/// Benchmark with adaptive ray count disabled (fixed count) vs enabled.
fn bench_adaptive_vs_fixed(c: &mut Criterion) {
    let pairs = generate_random_pairs(100);
    let (ref si, ref sj) = pairs[75]; // pick a pair that benefits from adaptive

    let mc_fixed = MonteCarloViewFactor::new(10_000).with_adaptive(false);
    let mc_adaptive = MonteCarloViewFactor::new(10_000).with_adaptive(true);

    let mut group = c.benchmark_group("view_factor/adaptive_vs_fixed");
    group.bench_function("fixed_10k", |b| {
        b.iter(|| {
            std::hint::black_box(
                mc_fixed
                    .compute(std::hint::black_box(si), std::hint::black_box(sj))
                    .unwrap(),
            )
        });
    });
    group.bench_function("adaptive_10k", |b| {
        b.iter(|| {
            std::hint::black_box(
                mc_adaptive
                    .compute(std::hint::black_box(si), std::hint::black_box(sj))
                    .unwrap(),
            )
        });
    });
    group.finish();
}

/// Custom benchmark that explicitly computes mean / median / p95 per pair
/// across the 100 random pairs and prints a summary.
fn bench_pair_latency_stats(c: &mut Criterion) {
    let pairs = generate_random_pairs(100);
    let mc = MonteCarloViewFactor::default();

    let mut group = c.benchmark_group("view_factor/pair_latency_stats");
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("100_pairs_latency", |b| {
        b.iter_custom(|iters| {
            let mut all_times = Vec::new();
            for _ in 0..iters {
                for (si, sj) in &pairs {
                    let start = Instant::now();
                    let _ = mc.compute(si, sj);
                    all_times.push(start.elapsed());
                }
            }

            // Compute and print statistics (only on the first iteration set).
            if all_times.len() == 100 * iters as usize && iters == 1 {
                let mut ms: Vec<f64> = all_times.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
                ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let n = ms.len();
                let mean: f64 = ms.iter().sum::<f64>() / n as f64;
                let median = ms[n / 2];
                let p95 = ms[(n as f64 * 0.95) as usize];
                let max = ms[n - 1];
                eprintln!();
                eprintln!("╔══════════════════════════════════════════════════════╗");
                eprintln!("║  View Factor Pair Latency — 100 random pairs        ║");
                eprintln!("╠══════════════════════════════════════════════════════╣");
                eprintln!(
                    "║  Mean:   {mean:>8.3} ms                              ║",
                    mean = mean
                );
                eprintln!(
                    "║  Median: {median:>8.3} ms                              ║",
                    median = median
                );
                eprintln!(
                    "║  P95:    {p95:>8.3} ms                              ║",
                    p95 = p95
                );
                eprintln!(
                    "║  Max:    {max:>8.3} ms                              ║",
                    max = max
                );
                eprintln!("║  Target: < 50.000 ms                                 ║");
                eprintln!(
                    "║  Status: {}                                           ║",
                    if max < 50.0 { "✅ PASS" } else { "❌ FAIL" }
                );
                eprintln!("╚══════════════════════════════════════════════════════╝");
            }

            // Return total elapsed for criterion's timing model.
            all_times.iter().sum::<Duration>()
        });
    });

    group.finish();
}

// ──────────────────────── matrix benchmark ──────────────────────────

/// Benchmark computing the full N×N view factor matrix for a set of surfaces
/// (simulates an urban block with multiple buildings).
fn bench_view_factor_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("view_factor/matrix");

    for &n in &[10usize, 25, 50] {
        // Generate n surfaces in a line facing each other (urban canyon).
        let pairs = generate_random_pairs(n);
        let surfaces: Vec<Surface3D> = pairs.iter().map(|(s, _)| s.clone()).collect();

        let mc = MonteCarloViewFactor::new(5_000);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &_| {
            b.iter(|| {
                std::hint::black_box(mc.compute_matrix(std::hint::black_box(&surfaces)).unwrap());
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_pair,
    bench_ray_counts,
    bench_100_pairs_batch,
    bench_adaptive_vs_fixed,
    bench_pair_latency_stats,
    bench_view_factor_matrix,
);
criterion_main!(benches);
