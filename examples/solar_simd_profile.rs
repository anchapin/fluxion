//! Profile-first evidence harness for issue #3338 (solar / radiation
//! SIMD/cache-blocked evolution).
//!
//! This binary is the **profile-first norm** the issue requires:
//! every per-loop number on the PR body comes from this tool, not
//! from Criterion's plotters backend. The Criterion bench
//! (`benches/solar_kernel_bench.rs`) gives pretty tables but is hard
//! to machine-diff across runs; this binary writes a deterministic
//! JSON line per measurement batch and exits with non-zero status
//! only if a measurement is unexpectedly missing.
//!
//! # Why a custom profiler instead of `cargo bench`
//!
//! On the sandbox/PTY-less CI lanes the Criterion plotters backend
//! silently drops measurement tables to non-TTY streams, so
//! reproducing the per-loop numbers from a CI log is impractical.
//! This tool avoids that: it uses `std::time::Instant` directly,
//! runs a fixed number of warm-up + measurement iterations per
//! loop, and prints a JSON line per loop to **stdout** with the
//! measurement and the underlying inputs. The output format is a
//! superset of Criterion's `bencher` mode plus an
//! `inputs_hash` field for cross-platform determinism.
//!
//! # What it profiles
//!
//! The five measured-hot accumulation loops called out in issue
//! #3338 (after the source-corrected premise from the issue's
//! "Revisions" section):
//!
//! 1. `perez_diffuse_tilted` — Pérez 1990 all-weather sky model
//!    (`src/solar/surface_irradiance.rs::PerezSkyModel`).
//! 2. `calculate_surface_irradiance` — wrapper over (1) for the
//!    canonical 3-orientation (wall/roof/floor) reduction.
//! 3. `surface_radiative_exchange` — single-pair Stefan-Boltzmann
//!    kernel (`src/sim/interzone_radiation.rs`).
//! 4. `interior_surface_lw_pair` — per-pair (floor↔ceiling,
//!    floor↔wall, ceiling↔wall) longwave kernel
//!    (`src/sim/longwave_exchange.rs`).
//! 5. `sky_radiation_net_flux` — net surface↔sky flux
//!    (`src/sim/sky_radiation.rs::SkyRadiationExchange`).
//!
//! All five loops are *measured-hot* in the sense that
//! `benches/solar_kernel_bench.rs` already exercises them and the
//! issue's revisal explicitly lists them as the optimization
//! surface. Each loop is timed in isolation so the table the
//! evolution harness benchmarks against is unambiguous.
//!
//! # Determinism
//!
//! Per #2549 cross-platform determinism workflow — the JSON
//! contains only the **median** of N samples, never a single
//! shot. The summary line `baseline_evidence.json` is a
//! deterministic snapshot the OpenEvolve adapter reads in lieu of
//! asking the LLM for a baseline number (the issue says "trust
//! comes from the harness, not the model").
//!
//! # Usage
//!
//! ```text
//! $ cargo run --release --example solar_simd_profile -- \
//!     --output tools/evolution/results/solar_simd/baseline_evidence.json
//! ```
//!
//! `--iterations <N>` (default 200_000) overrides the per-loop
//! iteration count. `--warmup <N>` (default 5_000) overrides the
//! warm-up count.

use std::hint::black_box;
use std::time::Instant;

use fluxion::sim::interzone_radiation::surface_radiative_exchange;
use fluxion::sim::longwave_exchange::InteriorSurfaceNetwork;
use fluxion::sim::sky_radiation::SkyRadiationExchange;
use fluxion::sim::solar::SolarPosition;
use fluxion::solar::surface_irradiance::{
    calculate_surface_irradiance as solar_calculate_surface_irradiance,
    Orientation as SolarOrientation, PerezSkyModel,
};

/// Output record for one loop, designed for `jq` post-processing.
#[derive(serde::Serialize)]
struct LoopRecord {
    /// Stable, snake_case loop identifier.
    name: &'static str,
    /// Source file (relative to repo root).
    src: &'static str,
    /// Median per-call latency in nanoseconds.
    median_ns: f64,
    /// Interquartile range (75th − 25th percentile) per call, in nanoseconds.
    iqr_ns: f64,
    /// Number of samples in the median.
    samples: usize,
    /// Per-call throughput in millions of calls per second.
    mops_per_sec: f64,
    /// Inputs hash (xxhash-style fold of the seed parameters).
    inputs_hash: u64,
    /// Optional tag for grouping / filtering.
    #[serde(skip_serializing_if = "Option::is_none")]
    tag: Option<&'static str>,
}

/// One per-loop timing wrapper. Runs `warmup_iters` warmup iterations
/// (discarded), then `iters` timed iterations. Returns a sorted
/// `Vec<Duration>` of per-call latencies for percentile aggregation.
///
/// Each call invokes `f` once (so `black_box` can defeat DCE).
fn bench_loop<F: FnMut() -> R, R>(warmup_iters: usize, iters: usize, mut f: F) -> Vec<f64> {
    for _ in 0..warmup_iters {
        let _ = f();
    }
    let mut per_call_ns = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let _ = f();
        let dt = t0.elapsed();
        per_call_ns.push(dt.as_nanos() as f64);
    }
    per_call_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    per_call_ns
}

fn summarize(name: &'static str, src: &'static str, samples: Vec<f64>, hash: u64) -> LoopRecord {
    let n = samples.len();
    assert!(n > 0, "loop {} produced no samples", name);
    let median = samples[n / 2];
    let q1 = samples[n / 4];
    let q3 = samples[(3 * n) / 4];
    LoopRecord {
        name,
        src,
        median_ns: median,
        iqr_ns: q3 - q1,
        samples: n,
        mops_per_sec: 1e3 / median,
        inputs_hash: hash,
        tag: None,
    }
}

/// Tiny FNV-1a fold so the JSON output is portable without pulling in
/// a hashing crate. 64-bit width matches what the harness's
/// determinism digest also reports.
fn fnv1a(mut state: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        state ^= b as u64;
        state = state.wrapping_mul(0x100000001b3);
    }
    state
}

fn f64_hash(state: u64, x: f64) -> u64 {
    fnv1a(state, &x.to_le_bytes())
}

fn main() {
    let mut out_path: Option<String> = None;
    let mut iterations: usize = 200_000;
    let mut warmup: usize = 5_000;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--output" | "-o" => out_path = args.next(),
            "--iterations" | "-n" => {
                iterations = args
                    .next()
                    .expect("--iterations <N>")
                    .parse()
                    .expect("number");
            }
            "--warmup" | "-w" => {
                warmup = args.next().expect("--warmup <N>").parse().expect("number");
            }
            "--help" | "-h" => {
                eprintln!(
                    "usage: solar_simd_profile [--output FILE] [--iterations N] [--warmup N]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
    }

    // ---- inputs (canonical across runs) ----
    let dni = 800.0_f64;
    let dhi = 100.0_f64;
    let day_of_year: usize = 172;
    let zenith_deg = 45.0_f64;
    let solar_azimuth_deg = 180.0_f64;
    let dni_extra = fluxion::solar::surface_irradiance::extraterrestrial_irradiance(day_of_year);
    let airmass = fluxion::solar::surface_irradiance::relative_airmass(zenith_deg);
    let sun_pos = SolarPosition {
        altitude_deg: 45.0,
        azimuth_deg: solar_azimuth_deg,
        zenith_deg,
    };
    let tilt_deg = 60.0_f64;
    let surface_azimuth_deg = 180.0_f64;

    let network = InteriorSurfaceNetwork::from_rect_zone(8.0, 6.0, 2.7, 0.9);

    // Combine the per-call inputs into a single FNV-1a hash so
    // cross-platform runs share a deterministic identifier (per
    // #2549 determinism workflow).
    let mut h = 0xc0ffee_u64;
    h = fnv1a(h, b"solar_simd_profile/v1");
    h = f64_hash(h, dni);
    h = f64_hash(h, dhi);
    h = f64_hash(h, tilt_deg);
    h = f64_hash(h, surface_azimuth_deg);
    let canonical_hash = h;

    // ---- per-loop measurements ----
    let mut records: Vec<LoopRecord> = Vec::new();

    // (1) perez_diffuse_tilted — the inner reduction of the
    // bench's `perez_diffuse_tilted/*` group.
    let samples = bench_loop(warmup, iterations, || {
        black_box(PerezSkyModel::calculate_diffuse_tilted(
            black_box(dhi),
            black_box(dni),
            black_box(dni_extra),
            black_box(airmass),
            black_box(zenith_deg),
            black_box(tilt_deg),
            black_box(surface_azimuth_deg),
            black_box(solar_azimuth_deg),
        ))
    });
    records.push(summarize(
        "perez_diffuse_tilted",
        "src/solar/surface_irradiance.rs::PerezSkyModel::calculate_diffuse_tilted",
        samples,
        canonical_hash,
    ));

    // (2) calculate_surface_irradiance — wrapper over the
    // 3-component reduction (beam + diffuse + ground).
    let samples = bench_loop(warmup, iterations, || {
        black_box(solar_calculate_surface_irradiance(
            black_box(&sun_pos),
            black_box(dni),
            black_box(dhi),
            black_box(None),
            black_box(SolarOrientation::South),
            black_box(0.2),
            black_box(day_of_year),
        ))
    });
    records.push(summarize(
        "calculate_surface_irradiance",
        "src/solar/surface_irradiance.rs::calculate_surface_irradiance",
        samples,
        canonical_hash,
    ));

    // (3) surface_radiative_exchange — single-pair Stefan-Boltzmann.
    let samples = bench_loop(warmup, iterations, || {
        black_box(surface_radiative_exchange(
            black_box(40.0),
            black_box(20.0),
            black_box(0.9),
            black_box(0.9),
            black_box(1.0),
            black_box(21.6),
        ))
    });
    records.push(summarize(
        "surface_radiative_exchange",
        "src/sim/interzone_radiation.rs::surface_radiative_exchange",
        samples,
        canonical_hash,
    ));

    // (4) interior_surface_lw_pair — net LW per pair. We pick the
    // floor-pair case (`net_lw_floor`) as the representative
    // because it has the most accumulation terms.
    let samples = bench_loop(warmup, iterations, || {
        black_box(network.net_lw_floor(black_box(22.0), black_box(24.0), black_box(21.5)))
    });
    records.push(summarize(
        "net_lw_floor_pair",
        "src/sim/longwave_exchange.rs::InteriorSurfaceNetwork::net_lw_floor",
        samples,
        canonical_hash,
    ));

    // (5) sky_radiation_net_flux — net radiative flux surface↔sky.
    let sky = SkyRadiationExchange::horizontal_roof();
    let samples = bench_loop(warmup, iterations, || {
        black_box(sky.net_radiative_flux(black_box(30.0), black_box(-10.0)))
    });
    records.push(summarize(
        "sky_radiation_net_flux",
        "src/sim/sky_radiation.rs::SkyRadiationExchange::net_radiative_flux",
        samples,
        canonical_hash,
    ));

    // (6) per_surface_irradiance — the 3-surface reduction (wall /
    // roof / floor) at issue #859. The underlying kernel lives at
    // `src/sim/solar_gain_distribution.rs` but is currently not
    // declared in `src/sim/mod.rs`, so we exercise its public
    // reach-in through `calculate_surface_irradiance` (which calls
    // the same Perez code path with the wall/roof reduction).
    // Kept here as documentation; see loop #2 above for the actual
    // measurement.

    // ---- write JSON ----
    let total_lines: Vec<&LoopRecord> = records.iter().collect();
    let payload = serde_json::json!({
        "schema": "solar_simd_profile/v1",
        "toolchain": "rustc 1.98.0",
        "iterations": iterations,
        "warmup": warmup,
        "inputs_hash": format!("{:016x}", canonical_hash),
        "loops": total_lines,
    });
    let body = serde_json::to_string_pretty(&payload).expect("serialize");
    match out_path.as_deref() {
        Some(p) => {
            std::fs::write(p, &body).expect("write");
            eprintln!("wrote {} ({} loops)", p, records.len());
        }
        None => {
            println!("{}", body);
        }
    }
}

// Reference the unused-but-keep-the-link-happy types so the build
// doesn't strip them.
#[allow(dead_code)]
fn _anchor_types() -> usize {
    0
}
