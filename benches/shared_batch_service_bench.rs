//! Benchmarks for [`SharedBatchInferenceService`] — issue #1438.
//!
//! Measures end-to-end configs/sec for two scheduler configurations:
//!
//! * `single_worker_fixed_wait` — the legacy `DynamicBatchConfig { wait_ms: 10 }`
//!   path (single worker, fixed wait).
//! * `multi_worker_adaptive_wait` — the new `SchedulerConfig` with adaptive
//!   `target_latency_ms` + N=`available_parallelism/4` workers.
//!
//! Both run the same workload (10 000 random `[T0..T4]` requests distributed
//! across 8 producer threads) against the mock `SurrogateManager`. The
//! comparison is the headline throughput number called out in the issue
//! body, and the per-publisher-CPU performance criterion in the project's
//! acceptance criteria ("≥18 000 configs/sec on 8-core CPU").
//!
//! Run with `cargo bench --bench shared_batch_service_bench`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use fluxion::ai::shared_batch_service::{
    DynamicBatchConfig, SchedulerConfig, SharedBatchInferenceService,
};
use fluxion::ai::surrogate::SurrogateManager;
use std::sync::Arc;
use std::thread;

/// Total number of inference requests per benchmark iteration.
const N_REQUESTS: usize = 10_000;

/// Number of producer threads that fan requests into the service.
const N_PRODUCERS: usize = 8;

/// Length of each temperature vector. Five floats is representative of the
/// real `BatchOracle::evaluate_population` payloads (T0..T4 zone temps).
const TEMPS_LEN: usize = 5;

/// Generate a deterministic synthetic workload of `N_REQUESTS` requests
/// with `TEMPS_LEN` temperatures each. Deterministic so benchmark numbers are
/// comparable across runs.
fn generate_workload() -> Vec<Vec<f64>> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(0x1438_C0DE);
    (0..N_REQUESTS)
        .map(|i| {
            (0..TEMPS_LEN)
                .map(|j| {
                    let lo = 10.0 + i as f64 * 0.001;
                    let hi = 30.0 + j as f64;
                    rng.gen_range(lo..hi)
                })
                .collect()
        })
        .collect()
}

/// Drive `workload` through `service` using `N_PRODUCERS` threads and report
/// the elapsed wall time.
///
/// Producers use the **pipeline pattern** from the issue body's
/// `BatchOracle::evaluate_population` (`src/lib.rs:1169`) hot path: each
/// producer submits its entire slice first, capturing all the
/// per-request `Receiver`s, then waits on them. This decouples producer
/// throughput from per-batch worker latency — the realistic shape of the
/// production workload where rayon threads fan-out `valid_configs.len()`
/// submits concurrently into the service.
fn drive_workload(service: &SharedBatchInferenceService, workload: &[Vec<f64>]) -> u64 {
    let chunk_size = workload.len().div_ceil(N_PRODUCERS);
    let workload = Arc::new(workload.to_vec());
    let mut handles = Vec::with_capacity(N_PRODUCERS);

    let start = std::time::Instant::now();
    for t in 0..N_PRODUCERS {
        let svc = service.clone();
        let workload = Arc::clone(&workload);
        let lo = t * chunk_size;
        let hi = ((t + 1) * chunk_size).min(workload.len());
        handles.push(thread::spawn(move || {
            let mut rxs = Vec::with_capacity(hi - lo);
            for i in lo..hi {
                rxs.push(svc.submit(workload[i].clone()));
            }
            for rx in rxs {
                let _ = rx.recv().expect("SharedBatchInferenceService response");
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    start.elapsed().as_micros() as u64
}

/// Legacy single-worker path with the original fixed `wait_ms: 10`. The
/// service is constructed ONCE outside the `b.iter` closure — so the per-iter
/// measurement captures only the workload-through-service wall time, not the
/// one-time OS thread-spawn overhead. This matches the production
/// `BatchOracle::evaluate_population` path (issue #1438), which constructs the
/// service once and reuses it across many submissions.
fn bench_single_worker_fixed_wait(c: &mut Criterion) {
    let surrogate = SurrogateManager::new().expect("mock SurrogateManager");
    let config = DynamicBatchConfig {
        max_batch_size: 512,
        wait_ms: 10,
    };
    let workload = generate_workload();

    let svc = SharedBatchInferenceService::new(surrogate, config, N_PRODUCERS * 4);

    let mut group = c.benchmark_group("shared_batch_throughput");
    group.throughput(Throughput::Elements(N_REQUESTS as u64));
    group.bench_function(
        BenchmarkId::from_parameter("legacy_single_worker_wait_10ms"),
        |b| {
            b.iter(|| drive_workload(&svc, &workload));
        },
    );
    group.finish();
}

/// Multi-worker fan-out with adaptive `wait_ms` (the post-#1438 scheduler).
fn bench_multi_worker_adaptive_wait(c: &mut Criterion) {
    let surrogate = SurrogateManager::new().expect("mock SurrogateManager");
    let sched = SchedulerConfig {
        max_batch_size: 512,
        target_latency_ms: 5,
        min_wait_ms: 1,
        max_wait_ms: 10,
        num_workers: 0, // auto: `available_parallelism() / 4` clamped to [1, 8]
        channel_capacity: 8192,
    };
    let workload = generate_workload();

    let svc = SharedBatchInferenceService::with_workers(surrogate, sched);

    let mut group = c.benchmark_group("shared_batch_throughput");
    group.throughput(Throughput::Elements(N_REQUESTS as u64));
    group.bench_function(
        BenchmarkId::from_parameter("multi_worker_adaptive_wait"),
        |b| b.iter(|| drive_workload(&svc, &workload)),
    );
    group.finish();
}

criterion_group!(
    shared_batch_benches,
    bench_single_worker_fixed_wait,
    bench_multi_worker_adaptive_wait
);
criterion_main!(shared_batch_benches);
