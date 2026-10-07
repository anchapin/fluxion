//! Issue #4188 acceptance: BatchOracle population workloads must stay inside
//! a bounded memory budget and remain bit-identical to the pre-#4188 path.
//!
//! This file is a **separate test target** (its own process) on purpose: the
//! peak-RSS assertion reads `VmHWM` from `/proc/self/status`, which is
//! process-wide, so it is only meaningful when this binary runs nothing else.
//! Run it standalone:
//!
//! ```sh
//! cargo test --release --test batch-oracle-memory-budget -- --ignored --nocapture
//! ```
//!
//! The nightly memory-budget workflow (`.github/workflows/memory-budget.yml`)
//! runs exactly that command through `scripts/memory-budget-gate.sh`.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::BatchOracle;

/// 10-zone model — matches `benchmark.multi_zone` in `release_gates.yaml`.
const ZONES: usize = 10;
/// Issue #4188 acceptance criterion #3: 10-zone / 10,000-config population
/// peak HWM must stay under 64 MiB (down from the measured 316 MB).
const RSS_BUDGET_BYTES: u64 = 64 * 1024 * 1024;

fn peak_hwm_bytes() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: u64 = rest
                .trim()
                .trim_end_matches("kB")
                .trim()
                .parse()
                .expect("parse VmHWM");
            return kb * 1024;
        }
    }
    panic!("VmHWM not found in /proc/self/status");
}

/// Same seeded fixture shape as
/// `tests/all_tests/performance_regression_test.rs::generate_multi_zone_population`.
fn generate_population(size: usize) -> Vec<Vec<f64>> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

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

fn fnv1a_hash_euis(euis: &[f64]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for eui in euis {
        for byte in eui.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// Bit-identity guard: the streamed (#4188) population path must produce the
/// exact same f64 EUI bit patterns as the pre-#4188 single-pass path. The
/// golden digest below was captured from develop @ c9fa258d (pre-#4188) on
/// the same fixture.
#[test]
fn multi_zone_population_euis_bit_identical_to_pre_4188_path() {
    let oracle = BatchOracle::from_model(ThermalModel::<VectorField>::new(ZONES)).unwrap();
    let population = generate_population(128);

    let euis = oracle
        .evaluate_population(population, false)
        .expect("analytical evaluate_population must succeed");

    // Configs that fail physics stability legitimately yield NaN; every
    // non-NaN result must be finite and non-negative.
    assert!(
        euis.iter()
            .all(|e| e.is_nan() || (e.is_finite() && *e >= 0.0)),
        "every EUI must be NaN or finite and non-negative"
    );

    let digest = fnv1a_hash_euis(&euis);
    println!("EUI FNV-1a digest: {digest:016x}");
    assert_eq!(
        digest, 0x9591_b72d_6872_00d9,
        "EUI bit patterns diverged from the pre-#4188 golden digest"
    );
}

/// Issue #4188 acceptance criterion #3: a 10-zone / 10,000-config
/// `evaluate_population` run stays under the 64 MiB peak-HWM budget.
#[test]
#[ignore = "slow 10k x 8760-step population workload; the nightly memory-budget workflow runs it explicitly with --ignored"]
fn multi_zone_10k_population_peak_rss_under_budget() {
    let oracle = BatchOracle::from_model(ThermalModel::<VectorField>::new(ZONES)).unwrap();
    let population = generate_population(10_000);

    let start = std::time::Instant::now();
    let euis = oracle
        .evaluate_population(population, false)
        .expect("analytical evaluate_population must succeed");
    let elapsed = start.elapsed();

    // Configs that fail physics stability legitimately yield NaN; every
    // non-NaN result must be finite and non-negative.
    assert!(
        euis.iter()
            .all(|e| e.is_nan() || (e.is_finite() && *e >= 0.0)),
        "every EUI must be NaN or finite and non-negative"
    );

    let throughput = 10_000.0 / elapsed.as_secs_f64();
    let hwm = peak_hwm_bytes();
    println!("Population: 10,000 configs / {ZONES} zones");
    println!("Elapsed: {:.2} ms", elapsed.as_secs_f64() * 1e3);
    println!("Throughput: {throughput:.0} configs/sec");
    println!(
        "Peak RSS (VmHWM): {:.1} MiB",
        hwm as f64 / (1024.0 * 1024.0)
    );

    assert!(
        hwm < RSS_BUDGET_BYTES,
        "peak HWM {:.1} MiB exceeds the {} MiB population-workload budget \
         (Issue #4188)",
        hwm as f64 / (1024.0 * 1024.0),
        RSS_BUDGET_BYTES / (1024 * 1024)
    );
}
