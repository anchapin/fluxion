//! Multi-zone throughput criterion benchmark (Issue #2522).
//!
//! `release_gates.yaml` (Issue #2362) gates multi-zone throughput at
//! `min_configs_per_sec: 10` for a 10-zone simulation, and
//! `tests/performance_ci_test.rs` asserts that floor on a population of 100.
//! The existing benches (`benches/performance.rs`, `benches/batch_oracle_bench.rs`,
//! `benches/benchmark_8760_timesteps.rs`) only measure single-config 10-zone
//! latency or sweep populations on a *single-zone* model. None sweeps population
//! sizes on the 10-zone model that the gate actually targets.
//!
//! This benchmark closes that gap: it measures
//! `BatchOracle::evaluate_population` throughput (configs/sec) on a 10-zone
//! `ThermalModel` across populations {10, 100, 1000, 10 000}, reporting the
//! result as a criterion throughput.
//!
//! The bench is gated behind `required-features = ["multi-zone"]` in Cargo.toml:
//! at ~28 configs/sec a full population_10k run is ~1 h, too long for the
//! Performance Dashboard's 15-min `cargo bench` sweep (which the Fluxion
//! Performance Gate #1618 observes). The gate-relevant compile check runs in
//! `.github/workflows/performance.yml` (`cargo bench --bench multi_zone_throughput
//! --features multi-zone --no-run`); the runtime floor (10 configs/sec on
//! population 100) stays enforced by `tests/performance_ci_test.rs`.
//!
//! Run: `cargo bench --bench multi_zone_throughput --features multi-zone`

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::BatchOracle;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Zone count enforced by the multi-zone throughput gate
/// (`release_gates.yaml`: `multi_zone.zones: 10`).
const ZONE_COUNT: usize = 10;

/// Population sizes swept by this benchmark.
///
/// {10, 100, 1000, 10 000} spans the single-config regime through the
/// full-scale batch regime, covering the {10, 100, 1000} acceptance set
/// from Issue #2522 plus a 10 000-config stress point that exercises the
/// rayon-parallel `evaluate_population` inner loop.
const POPULATION_SIZES: [usize; 4] = [10, 100, 1000, 10_000];

/// Synthetic population generator matching the fixture used by
/// `tests/performance_ci_test.rs::test_multi_zone_throughput` (the gate test),
/// so the benchmark measures the exact same configuration distribution the
/// release gate asserts against.
///
/// Parameters are within the valid bounds documented for
/// `BatchOracle::evaluate_population`:
/// - `[0]` U-value: 0.1-5.0 W/m²K
/// - `[1]` Heating setpoint: 15-25 °C
/// - `[2]` Cooling setpoint: 22-32 °C
fn generate_synthetic_population(size: usize) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut population = Vec::with_capacity(size);
    for _ in 0..size {
        let u_value = rng.random_range(0.1..5.0);
        let heating_setpoint = rng.random_range(15.0..25.0);
        let cooling_setpoint = rng.random_range(22.0..32.0);
        population.push(vec![u_value, heating_setpoint, cooling_setpoint]);
    }
    population
}

/// Benchmark `BatchOracle::evaluate_population` throughput (configs/sec) on a
/// 10-zone `ThermalModel` across population sizes {10, 100, 1000, 10 000}.
///
/// `Throughput::Elements(size)` makes criterion report configs/sec directly —
/// the same metric the multi-zone release gate (`min_configs_per_sec: 10`)
/// enforces. Analytical mode (`use_surrogates = false`) avoids the ONNX runtime
/// dependency, matching `test_multi_zone_throughput`.
fn bench_multi_zone_throughput(c: &mut Criterion) {
    let base_model = ThermalModel::<VectorField>::new(ZONE_COUNT);
    let oracle = BatchOracle::from_model(base_model);

    let mut group = c.benchmark_group("multi_zone_throughput_10_zones");

    for &size in &POPULATION_SIZES {
        let population = generate_synthetic_population(size);

        // Report configs/sec so the metric is directly comparable to the
        // `multi_zone.min_configs_per_sec: 10` release gate.
        group.throughput(Throughput::Elements(size as u64));
        // Larger populations are slow per iteration; cap at criterion's floor
        // to keep total bench wall-time bounded while staying statistically valid.
        group.sample_size(10);

        let name = format!("population_{}", size);
        group.bench_function(&name, |b| {
            b.iter(|| {
                let pop = black_box(population.clone());
                let _ = black_box(oracle.evaluate_population(pop, false));
            })
        });
    }

    group.finish();
}

criterion_group!(multi_zone_throughput_benches, bench_multi_zone_throughput);
criterion_main!(multi_zone_throughput_benches);
