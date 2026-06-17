//! Surrogate vs Physics Benchmark for Fluxion
//!
//! This benchmark compares ONNX surrogate inference speed against
//! physics-based thermal solver to verify the 10-100x speedup claim.
//!
//! Run with: cargo bench --release --bench surrogate_vs_physics
//!
//! Issue #720: Formal benchmarking of surrogate speedup

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use fluxion::ai::surrogate::SurrogateManager;
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;

const DUMMY_ONNX_MODEL: &str = "assets/dummy_surrogate.onnx";

fn bench_surrogate_onnx_single_inference(c: &mut Criterion) {
    let surrogate = match SurrogateManager::load_onnx(DUMMY_ONNX_MODEL) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Warning: Could not load ONNX model for benchmarking: {}", e);
            eprintln!("Surrogate benchmarks will be skipped.");
            return;
        }
    };

    let temps = vec![20.0, 21.0, 22.0, 23.0, 24.0, 25.0];

    let mut group = c.benchmark_group("surrogate_onnx");
    group.throughput(Throughput::Elements(temps.len() as u64));
    group.sample_size(1000);

    group.bench_function("single_inference_6zones", |b| {
        b.iter(|| {
            let _ = surrogate
                .predict_loads_onnx(black_box(&temps))
                .expect("ONNX inference failed");
        })
    });

    group.finish();
}

fn bench_surrogate_onnx_batched_inference(c: &mut Criterion) {
    let surrogate = match SurrogateManager::load_onnx(DUMMY_ONNX_MODEL) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Warning: Could not load ONNX model for benchmarking: {}", e);
            return;
        }
    };

    let batch_sizes = [1, 10, 100];

    for &batch_size in &batch_sizes {
        let mut group = c.benchmark_group(format!("surrogate_onnx_batch_{}", batch_size));
        group.throughput(Throughput::Elements(batch_size as u64 * 6));
        group.sample_size(100);

        let batch: Vec<Vec<f64>> = (0..batch_size)
            .map(|_| vec![20.0, 21.0, 22.0, 23.0, 24.0, 25.0])
            .collect();

        group.bench_function("batched_inference", |b| {
            b.iter(|| {
                let _ = surrogate.predict_loads_batched(black_box(&batch));
            })
        });

        group.finish();
    }
}

fn bench_physics_step_single_zone(c: &mut Criterion) {
    let mut model = ThermalModel::<VectorField>::new(1);

    c.bench_function("physics_step_single_zone", |b| {
        b.iter(|| {
            model.step_physics(black_box(0), black_box(20.0), black_box(3600.0));
        })
    });
}

fn bench_physics_step_multi_zone(c: &mut Criterion) {
    let zone_counts = [1, 5, 10];

    for &zones in &zone_counts {
        let mut model = ThermalModel::<VectorField>::new(zones);

        let mut group = c.benchmark_group(format!("physics_step_{}_zones", zones));
        group.throughput(Throughput::Elements(zones as u64));
        group.sample_size(1000);

        group.bench_function("step", |b| {
            b.iter(|| {
                model.step_physics(black_box(0), black_box(20.0), black_box(3600.0));
            })
        });

        group.finish();
    }
}

fn bench_comparison_8760_timesteps(c: &mut Criterion) {
    let zones = 10;
    let timesteps = 8760;
    let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

    let mut group = c.benchmark_group("comparison_8760_timesteps");
    group.sample_size(10);

    group.bench_function("physics_analytical_10zones_8760", |b| {
        b.iter(|| {
            let mut model = ThermalModel::<VectorField>::new(zones);
            model.solve_timesteps(
                black_box(timesteps),
                &surrogates,
                black_box(false),
                None,
                None,
                None,
            );
        })
    });

    group.bench_function("physics_with_surrogate_10zones_8760", |b| {
        b.iter(|| {
            let mut model = ThermalModel::<VectorField>::new(zones);
            model.solve_timesteps(
                black_box(timesteps),
                &surrogates,
                black_box(true),
                None,
                None,
                None,
            );
        })
    });

    group.finish();
}

fn bench_surrogate_inference_timing(c: &mut Criterion) {
    let surrogate = match SurrogateManager::load_onnx(DUMMY_ONNX_MODEL) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Warning: Could not load ONNX model for benchmarking: {}", e);
            return;
        }
    };

    let mut group = c.benchmark_group("surrogate_timing");
    group.sample_size(1000);

    group.bench_function("onnx_inference_6input", |b| {
        let temps = vec![20.0, 21.0, 22.0, 23.0, 24.0, 25.0];
        b.iter(|| {
            let _ = surrogate
                .predict_loads_onnx(black_box(&temps))
                .expect("ONNX inference failed");
        })
    });

    group.finish();
}

fn bench_analytical_loads_timing(c: &mut Criterion) {
    let surrogate = SurrogateManager::new().expect("Failed to create SurrogateManager");
    let temps = vec![20.0, 21.0, 22.0, 23.0, 24.0, 25.0];

    let mut group = c.benchmark_group("analytical_timing");
    group.sample_size(1000);

    group.bench_function("analytical_loads_6zones", |b| {
        b.iter(|| {
            let _ = surrogate
                .analytical_loads(black_box(&temps))
                .expect("Analytical loads failed");
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_surrogate_onnx_single_inference,
    bench_surrogate_onnx_batched_inference,
    bench_physics_step_single_zone,
    bench_physics_step_multi_zone,
    bench_comparison_8760_timesteps,
    bench_surrogate_inference_timing,
    bench_analytical_loads_timing
);
criterion_main!(benches);
