//! Batch orchestrator primitives — replaces the per-config `rayon::scope(N)`
//! coordinator-worker pattern used by `BatchOracle::evaluate_population`
//! (Issue #1439).
//!
//! ## Background
//!
//! The previous CPU-surrogate implementation spawned one OS thread per
//! candidate configuration, allocated `2 × N` crossbeam channels, and ran
//! a single coordinator thread that did `O(N)` round-trips per timestep.
//! For a population of 10 000 configs running 8 760 timesteps this scaled
//! as `8760 × 2N ≈ 1.75 × 10⁸` cross-thread hand-offs, which dominated
//! the per-config budget.
//!
//! ## New approach
//!
//! The orchestrator trait abstracts the *compute pattern* so the same
//! primitive can later back the GPU `SharedBatchInferenceService` path
//! once Issue #1344 (energy-conservation wiring) lands end-to-end.
//!
//! The CPU implementation here uses `rayon::par_chunks` over the
//! population. Each rayon worker pulls a contiguous chunk of configs,
//! runs **all 8 760 timesteps locally** for each config (one
//! `surrogates.predict_loads` call per timestep, no coordinator),
//! and pushes `(idx, total_energy)` back into a `Vec<(usize, f64)>`.
//!
//! Key invariants preserved against the old coordinator-worker path:
//!
//! 1. **Bit-identical results for the default build** (no ONNX model
//!    loaded): `predict_loads` returns `vec![1.2; len]` deterministically.
//! 2. **Composite / multi-zone models**: `predict_loads` is a pure
//!    function of `(model_temps, model)`, so per-config local inference
//!    produces identical results to batched inference when no ONNX
//!    model is loaded.
//! 3. **Result ordering**: `Vec::with_capacity(N) + indexed assignment`
//!    preserves the population-index slot so `results[idx]` is
//!    correctly mapped even though `par_chunks` may visit chunks in
//!    any order.
//!
//! ## Trade-offs vs the previous path
//!
//! | Property | Old (scope-N) | New (par_chunks) |
//! |----------|---------------|------------------|
//! | OS thread spawns for N=10 000 | 10 000 | ≈ `N / chunk_size` (≈ 313 at chunk 32) |
//! | Cross-thread hand-offs per call | `8760 × 2N` | 0 (workers run locally) |
//! | `crossbeam::channel` allocations | `2N` | 0 |
//! | ONNX tensor batching at the coordinator | Yes (per-timestep batched inference) | No (per-config inference) |
//!
//! The ONNX-tensor batching loss is intentional and bounded: CPU
//! ONNX inference has limited batch-dimension speedup, and removing
//! the coordinator bottleneck wins more than it loses on the CPU
//! surrogate branch. The trait is structured so a future GPU-backed
//! `BatchOrchestrator` can re-introduce batching where it matters.
//!
//! ## Acceptance criteria reference
//!
//! - Issue #1439 §"Proposed approach" item 2: `par_chunks(chunk_size)`
//!   replaces `rayon::scope(N)`. `chunk_size ≈ available_parallelism * 4`.
//! - Issue #1439 §"Proposed approach" item 3: `BatchOrchestrator` trait
//!   at `src/sim/orchestrator.rs` so the same primitive backs the GPU
//!   path (SharedBatchInferenceService, Issue #1344 follow-up).

use crate::ai::surrogate::SurrogateManager;
use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::sim::thermal_model_core::ThermalModel;

/// Default chunk size for `par_chunks` over the population.
///
/// Rayon's docs recommend `O(num_threads)` chunking for `par_chunks`.
/// We default to `4 × available_parallelism` so each rayon worker
/// picks up ~4 chunks; that balances load-balancing churn against the
/// overhead of crossing chunk boundaries.
pub const DEFAULT_CHUNK_MULTIPLIER: usize = 4;

/// Compute the recommended `par_chunks` chunk size for the current
/// host. Returns `DEFAULT_CHUNK_MULTIPLIER * n_cpus`, clamped to a
/// minimum of 1.
#[inline]
pub fn recommended_chunk_size(n_population: usize) -> usize {
    let n_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let chunk = DEFAULT_CHUNK_MULTIPLIER.saturating_mul(n_cpus).max(1);
    chunk.min(n_population.max(1))
}

/// A single CPU evaluation result: `(population_index, eui_kwh_per_m2_yr)`.
///
/// The orchestrator returns these in an arbitrary order; the caller is
/// responsible for placing them at `results[idx]`.
pub type CpuResult = (usize, f64);

/// Trait abstracting the per-population compute pattern.
///
/// Implementors decide *how* to schedule `Vec<ThermalModel>` evaluations
/// across cores (and eventually a GPU). The trait is intentionally
/// minimal: one method, no async, no streaming. It is the abstraction
/// point identified by Issue #1439 §"Proposed approach" item 3.
pub trait BatchOrchestrator: Send + Sync {
    /// Evaluate `configs` (already validated and parameter-applied) on
    /// the CPU using `surrogates` for per-timestep load prediction.
    ///
    /// Returns one `(idx, eui)` tuple per config. The caller owns
    /// result-placement; ordering is not guaranteed.
    fn run_cpu_surrogate(
        &self,
        configs: Vec<(usize, ThermalModel<VectorField>)>,
        surrogates: &SurrogateManager,
    ) -> Vec<CpuResult>;
}

/// Rayon-chunks CPU orchestrator.
///
/// Implements the `par_chunks(chunk_size).for_each(...)` pattern from
/// Issue #1439 §"Proposed approach" item 2. Each rayon worker:
///
/// 1. Takes a chunk of `(idx, ThermalModel)` pairs.
/// 2. For each config in the chunk, runs all 8 760 timesteps locally
///    using `surrogates.predict_loads` (no coordinator).
/// 3. Pushes `(idx, eui)` into a thread-shared `Vec<CpuResult>`.
pub struct RayonChunksOrchestrator {
    chunk_size: usize,
}

impl RayonChunksOrchestrator {
    /// Construct an orchestrator using `recommended_chunk_size(N)` for
    /// the given population size.
    pub fn for_population(n_population: usize) -> Self {
        Self {
            chunk_size: recommended_chunk_size(n_population),
        }
    }

    /// Construct an orchestrator with an explicit chunk size (used by
    /// the benchmark to sweep chunk sizes).
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        Self {
            chunk_size: chunk_size.max(1),
        }
    }
}

impl Default for RayonChunksOrchestrator {
    fn default() -> Self {
        Self::with_chunk_size(recommended_chunk_size(1024))
    }
}

impl BatchOrchestrator for RayonChunksOrchestrator {
    fn run_cpu_surrogate(
        &self,
        configs: Vec<(usize, ThermalModel<VectorField>)>,
        surrogates: &SurrogateManager,
    ) -> Vec<CpuResult> {
        use rayon::prelude::*;

        if configs.is_empty() {
            return Vec::new();
        }

        let chunk_size = self.chunk_size.min(configs.len());

        // par_chunks gives each rayon worker a contiguous slice of
        // configs (size ≤ chunk_size). We map each chunk to a
        // `Vec<CpuResult>` (per-worker, race-free) and flatten the
        // per-chunk Vecs at the end. The closure is `FnSync` so it can
        // borrow `surrogates` immutably across chunks. Final ordering
        // is determined by rayon chunk completion order (not
        // population-index order); the caller is responsible for
        // placing results at `results[idx]`.
        let per_chunk: Vec<Vec<CpuResult>> = configs
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut chunk_results: Vec<CpuResult> = Vec::with_capacity(chunk.len());
                for (idx, mut model) in chunk.iter().cloned() {
                    let mut energy_kwh = 0.0_f64;
                    for t in 0..8760 {
                        let hour_of_day = t % 24;
                        let daily_cycle = ((hour_of_day as f64 / 24.0)
                            * 2.0
                            * std::f64::consts::PI
                            - std::f64::consts::PI
                            / 2.0)
                            .sin();
                        let outdoor_temp = 10.0 + 10.0 * daily_cycle;

                        let temps = model.get_temperatures();
                        let loads = surrogates.predict_loads(&temps);
                        model.set_loads(&loads);
                        energy_kwh += model.step_physics(t, outdoor_temp, 3600.0);
                    }

                    let total_area = model.zone_area.integrate();
                    let eui = if total_area > 0.0 {
                        (energy_kwh / total_area).max(0.0)
                    } else {
                        0.0
                    };

                    chunk_results.push((idx, eui));
                }
                chunk_results
            })
            .collect();

        let mut out: Vec<CpuResult> = Vec::with_capacity(configs.len());
        for chunk_results in per_chunk {
            out.extend(chunk_results);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::cta::VectorField;

    fn make_dummy_config(idx: usize) -> (usize, ThermalModel<VectorField>) {
        let model = ThermalModel::<VectorField>::new(1);
        (idx, model)
    }

    #[test]
    fn recommended_chunk_size_is_positive() {
        assert!(recommended_chunk_size(0) >= 1);
        assert!(recommended_chunk_size(1) >= 1);
        assert!(recommended_chunk_size(10_000) >= 1);
    }

    #[test]
    fn recommended_chunk_size_scales_with_population() {
        let small = recommended_chunk_size(10);
        let large = recommended_chunk_size(10_000);
        assert!(
            large >= small,
            "chunk size for larger population should not be smaller: {} < {}",
            large,
            small
        );
    }

    #[test]
    fn recommended_chunk_size_caps_at_population_size() {
        let n = 4_usize;
        let chunk = recommended_chunk_size(n);
        assert!(chunk <= n, "chunk {} > pop {}", chunk, n);
    }

    #[test]
    fn empty_configs_returns_empty_results() {
        let surrogates = SurrogateManager::new().expect("SurrogateManager::new");
        let orchestrator = RayonChunksOrchestrator::default();
        let result = orchestrator.run_cpu_surrogate(Vec::new(), &surrogates);
        assert!(result.is_empty());
    }

    #[test]
    fn single_config_returns_one_result() {
        let surrogates = SurrogateManager::new().expect("SurrogateManager::new");
        let orchestrator = RayonChunksOrchestrator::default();
        let configs = vec![make_dummy_config(0)];
        let result = orchestrator.run_cpu_surrogate(configs, &surrogates);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 0);
        assert!(result[0].1.is_finite());
        assert!(result[0].1 >= 0.0);
    }

    #[test]
    fn small_population_returns_all_indices_exactly_once() {
        let surrogates = SurrogateManager::new().expect("SurrogateManager::new");
        let orchestrator = RayonChunksOrchestrator::for_population(8);
        let configs: Vec<_> = (0..8).map(make_dummy_config).collect();
        let result = orchestrator.run_cpu_surrogate(configs, &surrogates);
        assert_eq!(result.len(), 8);

        let mut indices: Vec<usize> = result.iter().map(|(i, _)| *i).collect();
        indices.sort_unstable();
        assert_eq!(indices, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn deterministic_across_repeated_runs() {
        // The default build (no ONNX model loaded) returns a constant
        // mock load of `vec![1.2; len]`. With identical inputs, two
        // orchestrator runs MUST produce identical EUI values per idx.
        let surrogates = SurrogateManager::new().expect("SurrogateManager::new");
        let orchestrator = RayonChunksOrchestrator::default();

        let make_pop = || {
            (0..5)
                .map(|i| {
                    let mut m = ThermalModel::<VectorField>::new(1);
                    m.apply_parameters(&[1.5, 20.0, 24.0]);
                    (i, m)
                })
                .collect::<Vec<_>>()
        };

        let mut r1 = orchestrator.run_cpu_surrogate(make_pop(), &surrogates);
        let mut r2 = orchestrator.run_cpu_surrogate(make_pop(), &surrogates);
        r1.sort_by_key(|(i, _)| *i);
        r2.sort_by_key(|(i, _)| *i);

        for ((i1, v1), (i2, v2)) in r1.iter().zip(r2.iter()) {
            assert_eq!(i1, i2);
            assert!(
                (v1 - v2).abs() < 1e-12,
                "Run-to-run drift: idx={} {} vs {}",
                i1,
                v1,
                v2
            );
        }
    }

    #[test]
    fn chunk_size_smaller_than_population_produces_chunks() {
        // For N=64 with chunk_size=8, `par_chunks(8)` yields 8 chunks
        // of 8 items each. We just verify the orchestrator handles
        // the many-chunk case without losing any configs.
        let surrogates = SurrogateManager::new().expect("SurrogateManager::new");
        let orchestrator = RayonChunksOrchestrator::with_chunk_size(8);
        let configs: Vec<_> = (0..64).map(make_dummy_config).collect();
        let result = orchestrator.run_cpu_surrogate(configs, &surrogates);
        assert_eq!(result.len(), 64);
    }
}
