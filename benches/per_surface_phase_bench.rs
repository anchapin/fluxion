//! Single-configuration multi-zone measurement for Fluxion issue #4697.
//!
//! This is intentionally **not** a `BatchOracle` population benchmark.  It
//! drives one `ThermalModel` so that the per-surface timestep phase can be
//! compared with the coupled zone solve without nested population-level
//! Rayon work.
//!
//! Usage (environment variables are zones, timesteps, repeats):
//!
//! ```text
//! FLUXION_BENCH_ZONES=50 FLUXION_BENCH_TIMESTEPS=720 \
//!   FLUXION_BENCH_REPEATS=5 \
//!   cargo test --release --bench per_surface_phase_bench -- --nocapture --test-threads=1
//! ```
//!
//! The default is a bounded 50-zone, 720-hour (30-day) run.  Model creation
//! and surrogate-manager creation are outside the timed interval; each repeat
//! uses a fresh model and runs the same bounded single configuration.

use std::time::{Duration, Instant};

use fluxion::ai::surrogate::SurrogateManager;
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;

const DEFAULT_ZONES: usize = 50;
const DEFAULT_TIMESTEPS: usize = 720;
const DEFAULT_REPEATS: usize = 5;

fn parse_env<T>(name: &str, default: T) -> T
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match std::env::var(name) {
        Err(_) => default,
        Ok(raw) => raw.parse::<T>().unwrap_or_else(|err| {
            panic!("invalid benchmark environment variable {name} ({raw:?}): {err}");
        }),
    }
}

fn median(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn median_u64(samples: &mut [u64]) -> u64 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

#[test]
fn measure_single_config_per_surface_phase() {
    let zones = parse_env("FLUXION_BENCH_ZONES", DEFAULT_ZONES);
    let timesteps = parse_env("FLUXION_BENCH_TIMESTEPS", DEFAULT_TIMESTEPS);
    let repeats = parse_env("FLUXION_BENCH_REPEATS", DEFAULT_REPEATS);
    assert!(zones > 1, "the audit requires a multi-zone model");
    assert!(timesteps > 0, "the audit requires at least one timestep");
    assert!(repeats > 0, "the audit requires at least one repeat");

    let surrogates = SurrogateManager::new().expect("failed to create SurrogateManager");
    let mut wall_samples = Vec::with_capacity(repeats);
    let mut preparation_samples = Vec::with_capacity(repeats);
    let mut boundary_samples = Vec::with_capacity(repeats);

    for repeat in 0..repeats {
        // Construction is deliberately excluded: this measures timestep work,
        // not allocation of the model graph.
        let mut model = ThermalModel::<VectorField>::new(zones);
        let started = Instant::now();
        let energy = model.solve_timesteps(timesteps, &surrogates, false, None, None, None);
        let elapsed = started.elapsed();
        assert!(energy.is_finite(), "non-finite energy in repeat {repeat}");
        let (preparation_ns, boundary_ns, preparation_steps, boundary_steps) =
            model.per_surface_phase_audit();
        eprintln!(
            "repeat={repeat} wall_ns={} preparation_ns={preparation_ns} \
             boundary_ns={boundary_ns} preparation_steps={preparation_steps} \
             boundary_steps={boundary_steps} energy={energy:.12}",
            elapsed.as_nanos()
        );
        wall_samples.push(elapsed);
        preparation_samples.push(preparation_ns);
        boundary_samples.push(boundary_ns);
    }

    let median_wall = median(&mut wall_samples);
    let median_preparation_ns = median_u64(&mut preparation_samples);
    let median_boundary_ns = median_u64(&mut boundary_samples);
    let median_phase_ns = median_preparation_ns + median_boundary_ns;
    let phase_share = median_phase_ns as f64 / median_wall.as_nanos() as f64;
    let amdahl_infinite = 1.0 / (1.0 - phase_share);
    let threads = rayon::current_num_threads();
    let amdahl_threads = 1.0 / ((1.0 - phase_share) + phase_share / threads as f64);
    println!(
        "PER_SURFACE_AUDIT zones={zones} timesteps={timesteps} repeats={repeats} \
         median_wall_ns={} median_wall_ms={:.6} median_preparation_ns={median_preparation_ns} \
         median_boundary_ns={median_boundary_ns} median_phase_ns={median_phase_ns} \
         phase_share_pct={:.6} amdahl_infinite_speedup={:.6} \
         amdahl_{threads}_thread_speedup={:.6}",
        median_wall.as_nanos(),
        median_wall.as_secs_f64() * 1_000.0,
        phase_share * 100.0,
        amdahl_infinite,
        amdahl_threads,
    );
}
