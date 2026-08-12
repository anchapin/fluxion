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
//! The ONNX-tensor batching loss is intentional and bounded for the
//! **mock / analytical** path: CPU ONNX inference has limited
//! batch-dimension speedup there, and removing the coordinator
//! bottleneck wins more than it loses. The trait is structured so a
//! future GPU-backed `BatchOrchestrator` can re-introduce batching
//! where it matters.
//!
//! ## Issue #2520 — per-timestep ONNX batching restored
//!
//! [`RayonChunksOrchestrator::run_cpu_surrogate_batched`] adds a
//! crossbeam-channel rendezvous that re-batches ONNX inference across
//! the whole population per timestep when a real model is loaded
//! (`SurrogateManager::model_loaded`). The mock path keeps using the
//! zero-coordinator `par_chunks` path; only the ONNX path pays the
//! rendezvous cost, which the 1024× call-count reduction
//! (`8.97 M → 8 760` calls) dwarfs.
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

    /// Per-timestep **batched** CPU surrogate path (Issue #2520).
    ///
    /// Restores the ONNX tensor batching that Issue #1439 deliberately
    /// traded away when it moved to `par_chunks`. The unbatched
    /// [`Self::run_cpu_surrogate`] runs every timestep locally inside
    /// each `par_chunks` worker and calls `predict_loads` (batch size 1)
    /// once per config per timestep — `1024 × 8760 = 8.97 M` inference
    /// calls for a 1024-config population. ONNX matrix-batch inference
    /// is much cheaper per sample, so this path re-batches across the
    /// whole population per timestep, collapsing the call count to
    /// `8760`.
    ///
    /// The default implementation delegates to [`Self::run_cpu_surrogate`]
    /// (no batching) so the trait stays implementable by a single-method
    /// orchestrator. [`RayonChunksOrchestrator`] overrides it with the
    /// crossbeam-rendezvous implementation described below.
    ///
    /// Callers should select this path only when a real ONNX model is
    /// loaded (`SurrogateManager::model_loaded`); for the mock /
    /// analytical fallback there is no batch-dimension speedup and the
    /// unbatched `par_chunks` path is strictly faster (zero coordinator
    /// round-trips).
    fn run_cpu_surrogate_batched(
        &self,
        configs: Vec<(usize, ThermalModel<VectorField>)>,
        surrogates: &SurrogateManager,
    ) -> Vec<CpuResult> {
        self.run_cpu_surrogate(configs, surrogates)
    }
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
                    // Issue #2687: hoist the per-timestep surrogate I/O
                    // buffers out of the 8 760-step inner loop and reuse them
                    // each timestep. Previously this body allocated three Vecs
                    // per timestep (`get_temperatures`, `predict_loads`,
                    // `set_loads`'s `to_vec`) — ~26 M heap allocations for a
                    // 1 000-config × 8 760-timestep run. The `_into` /
                    // `_from_slice` variants reuse capacity, so after warm-up
                    // the inner loop performs no heap allocation here. The
                    // bytes flowing into `step_physics` are bit-identical.
                    let mut temps_buf: Vec<f64> = Vec::new();
                    let mut loads_buf: Vec<f64> = Vec::new();
                    for t in 0..8760 {
                        let hour_of_day = t % 24;
                        let daily_cycle =
                            ((hour_of_day as f64 / 24.0) * 2.0 * std::f64::consts::PI
                                - std::f64::consts::PI / 2.0)
                                .sin();
                        let outdoor_temp = 10.0 + 10.0 * daily_cycle;

                        model.get_temperatures_into(&mut temps_buf);
                        surrogates.predict_loads_into(&temps_buf, &mut loads_buf);
                        model.set_loads(&loads_buf);
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

    /// Crossbeam-rendezvous implementation of
    /// [`BatchOrchestrator::run_cpu_surrogate_batched`] (Issue #2520).
    ///
    /// # Design — per-timestep batched inference with NO nested `par_iter`
    ///
    /// The population is split into `n_workers` contiguous slices, where
    /// `n_workers = min(ceil(N / chunk_size), available_parallelism)` so we
    /// never oversubscribe the host. Each slice is owned by one persistent
    /// worker thread spawned with `std::thread::scope`. Per timestep:
    ///
    /// 1. Every worker gathers its slice's temperature vectors and hands
    ///    them to the coordinator over a bounded `crossbeam::channel`
    ///    (the rendezvous).
    /// 2. The coordinator concatenates all slices (preserving worker-id
    ///    order), runs **one** `SurrogateManager::predict_loads_batched`
    ///    call for the whole population, and scatters each worker's load
    ///    vectors back over per-worker reply channels.
    /// 3. Workers apply the loads and step the physics sequentially.
    ///
    /// This brings inference calls from `N × 8760` down to `8760`.
    ///
    /// No `rayon` is used inside this method — `std::thread::scope`
    /// spawns the workers and the calling thread runs the coordinator —
    /// so there is zero possibility of nested `par_iter` (rayon
    /// thread-pool exhaustion, #1065/#2524). The `par_iter` in
    /// `BatchOracle::evaluate_population` (population validation) has
    /// already terminated before this method is invoked.
    fn run_cpu_surrogate_batched(
        &self,
        configs: Vec<(usize, ThermalModel<VectorField>)>,
        surrogates: &SurrogateManager,
    ) -> Vec<CpuResult> {
        use crossbeam::channel;
        use std::thread;

        let n = configs.len();
        if n == 0 {
            return Vec::new();
        }

        let n_cpus = thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        // One persistent worker per `par_chunks`-equivalent unit, capped at
        // the host's parallelism to avoid thread oversubscription.
        let nominal = n.div_ceil(self.chunk_size.max(1));
        let n_workers = nominal.min(n_cpus).max(1);

        // Split into contiguous worker-owned slices (each worker processes a
        // strided-but-contiguous block; ordering within a worker is preserved).
        let slice_cap = n.div_ceil(n_workers);
        let slices: Vec<Vec<(usize, ThermalModel<VectorField>)>> = configs
            .chunks(slice_cap.max(1))
            .map(|c| c.to_vec())
            .collect();
        let n_workers = slices.len();
        // Per-worker config count is constant across timesteps (workers own a
        // fixed slice), so we precompute it once for the scatter slicing.
        let worker_counts: Vec<usize> = slices.iter().map(|s| s.len()).collect();

        // Work channel: worker -> coordinator, carrying (worker_id, temps).
        // Capacity = n_workers so the first per-timestep burst of sends never
        // blocks before the coordinator begins draining (tight hand-off).
        let (work_tx, work_rx) = channel::bounded::<(usize, Vec<Vec<f64>>)>(n_workers);
        // One reply channel per worker (coordinator -> worker).
        let mut reply_senders: Vec<channel::Sender<Vec<Vec<f64>>>> = Vec::with_capacity(n_workers);
        let mut reply_receivers: Vec<channel::Receiver<Vec<Vec<f64>>>> =
            Vec::with_capacity(n_workers);
        for _ in 0..n_workers {
            let (tx, rx) = channel::bounded::<Vec<Vec<f64>>>(1);
            reply_senders.push(tx);
            reply_receivers.push(rx);
        }
        // Result channel: workers -> coordinator, carrying finished (idx, eui).
        let (result_tx, result_rx) = channel::unbounded::<Vec<CpuResult>>();

        let n_timesteps: usize = 8760;

        thread::scope(|s| {
            // ---------------- workers (one OS thread per slice) ----------------
            // Each worker gets its own clone of the work/result senders and
            // borrows its dedicated reply receiver (which outlives this scope).
            for (wid, mut models) in slices.into_iter().enumerate() {
                let work_tx = work_tx.clone();
                let result_tx = result_tx.clone();
                let reply_rx = &reply_receivers[wid];
                s.spawn(move || {
                    // Deterministic daily cycle — identical to `run_cpu_surrogate`,
                    // computed once per worker (cheap: 24 sin evaluations) so the
                    // mock build produces bit-identical EUI between the two paths.
                    let cycle: [f64; 24] = {
                        let mut arr = [0.0_f64; 24];
                        for (h, v) in arr.iter_mut().enumerate() {
                            *v = ((h as f64 / 24.0) * 2.0 * std::f64::consts::PI
                                - std::f64::consts::PI / 2.0)
                                .sin();
                        }
                        arr
                    };

                    let mut energy: Vec<f64> = vec![0.0; models.len()];

                    // Issue #2771: hoist the per-timestep temps buffer out of
                    // the 8 760-step loop. The rendezvous hands ownership of
                    // `msg` to the coordinator each timestep and gets the
                    // loads back; we recycle the *received* loads Vec as the
                    // next `msg` (refilling each inner via
                    // `get_temperatures_into`), so after the first timestep
                    // the worker performs zero heap allocation. The bytes
                    // sent each step are bit-identical to the prior
                    // `models.iter().map(|m| m.get_temperatures()).collect()`
                    // gather — only the buffer ownership differs.
                    let mut msg: Vec<Vec<f64>> = models
                        .iter()
                        .map(|(_, m)| {
                            let mut v = Vec::new();
                            m.get_temperatures_into(&mut v);
                            v
                        })
                        .collect();

                    for t in 0..n_timesteps {
                        let outdoor_temp = 10.0 + 10.0 * cycle[t % 24];

                        // Hand this slice's temps (msg) to the coordinator.
                        if work_tx.send((wid, msg)).is_err() {
                            return;
                        }
                        // Block until the coordinator returns this slice's loads.
                        let loads = match reply_rx.recv() {
                            Ok(l) => l,
                            Err(_) => return,
                        };

                        // Apply loads + step physics. Sequential within the
                        // worker — NO nested par_iter (would exhaust the rayon
                        // pool, #1065/#2524). Borrow `loads` immutably so the
                        // buffer can be recycled below.
                        for (((_, model), load), e) in
                            models.iter_mut().zip(loads.iter()).zip(energy.iter_mut())
                        {
                            model.set_loads(load);
                            *e += model.step_physics(t, outdoor_temp, 3600.0);
                        }

                        // Recycle the received loads Vec as the next `msg`.
                        // The reassignment is unconditional so the borrow
                        // checker sees `msg` reinitialized every iteration
                        // before the next `send` moves it; the (cheap)
                        // temperature refill only runs when a further step
                        // follows, since the final step's gather is wasted.
                        msg = loads;
                        if t + 1 < n_timesteps {
                            for ((_, m), inner) in models.iter().zip(msg.iter_mut()) {
                                m.get_temperatures_into(inner);
                            }
                        }
                    }

                    // Reduce to per-config EUI and ship to the coordinator.
                    let results: Vec<CpuResult> = models
                        .into_iter()
                        .zip(energy.into_iter())
                        .map(|((idx, model), e)| {
                            let total_area = model.zone_area.integrate();
                            let eui = if total_area > 0.0 {
                                (e / total_area).max(0.0)
                            } else {
                                0.0
                            };
                            (idx, eui)
                        })
                        .collect();
                    let _ = result_tx.send(results);
                });
            }

            // ---------------- coordinator (runs on this scope thread) ----------------
            // Borrows `work_rx`, `surrogates`, `reply_senders`, and
            // `worker_counts` from the enclosing frame (all outlive the scope).
            //
            // Issue #2771: hoist the per-timestep coordinator buffers out of
            // the 8 760-step loop. `slices_by_wid`, `batch`, and the three
            // surrogate scratch buffers are cleared/refilled in place each
            // timestep. The scatter uses `drain` (ownership transfer of the
            // inner Vecs) instead of `.to_vec()` (clone), eliminating the N
            // per-timestep clone allocations of the prior code. Together with
            // `predict_loads_batched_into` this removes the per-timestep
            // flattened-input / results / N-chunk allocations the prior
            // `predict_loads_batched` call made.
            let mut slices_by_wid: Vec<Option<Vec<Vec<f64>>>> =
                (0..n_workers).map(|_| None).collect();
            let mut batch: Vec<Vec<f64>> = Vec::with_capacity(n);
            let mut flat_in: Vec<f32> = Vec::new();
            let mut flat_out: Vec<f64> = Vec::new();
            let mut loads: Vec<Vec<f64>> = Vec::with_capacity(n);

            for _t in 0..n_timesteps {
                // Gather every worker's slice for this timestep. Workers may
                // arrive in any order; we index by the worker id they tag onto
                // each message so the scatter stays deterministic.
                let mut received = 0usize;
                while received < n_workers {
                    match work_rx.recv() {
                        Ok((wid, temps)) => {
                            slices_by_wid[wid] = Some(temps);
                            received += 1;
                        }
                        Err(_) => {
                            // All workers dropped their senders (exited). Aborting
                            // here is safe — no worker is left blocked on a reply.
                            return;
                        }
                    }
                }

                // Flatten in worker-id order so the scatter slicing is stable.
                batch.clear();
                for slice in slices_by_wid.iter_mut() {
                    if let Some(temps) = slice.take() {
                        batch.extend(temps);
                    }
                }

                // ONE batched inference call for the whole population this
                // timestep — the whole point of Issue #2520. The `_into`
                // variant reuses `flat_in` / `flat_out` / `loads` across
                // timesteps (Issue #2771).
                surrogates.predict_loads_batched_into(
                    &batch,
                    &mut flat_in,
                    &mut flat_out,
                    &mut loads,
                );

                // Scatter each worker's load slice back via `drain`, which
                // moves the inner Vecs out by ownership rather than cloning.
                // The loads the workers receive are recycled by them into the
                // next timestep's temps (see the worker loop above).
                for wid in 0..n_workers {
                    let count = worker_counts[wid];
                    let slice_loads: Vec<Vec<f64>> = loads.drain(..count).collect();
                    // A send error means the worker exited; keep going so the
                    // remaining workers can finish this timestep.
                    let _ = reply_senders[wid].send(slice_loads);
                }
            }
        });

        // All worker result-sender clones dropped when their threads exited;
        // drop the last original so `result_rx` hits EOF and the drain loop
        // terminates.
        drop(result_tx);

        // Collect worker results in arrival order; the caller maps by `idx`.
        let mut out: Vec<CpuResult> = Vec::with_capacity(n);
        while let Ok(chunk) = result_rx.recv() {
            out.extend(chunk);
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

    // -----------------------------------------------------------------------
    // Issue #2520 — per-timestep batched path (crossbeam rendezvous).
    // -----------------------------------------------------------------------

    #[test]
    fn batched_empty_configs_returns_empty_results() {
        let surrogates = SurrogateManager::new().expect("SurrogateManager::new");
        let orchestrator = RayonChunksOrchestrator::default();
        let result = orchestrator.run_cpu_surrogate_batched(Vec::new(), &surrogates);
        assert!(result.is_empty());
    }

    #[test]
    fn batched_returns_all_indices_exactly_once() {
        let surrogates = SurrogateManager::new().expect("SurrogateManager::new");
        let orchestrator = RayonChunksOrchestrator::for_population(16);
        let configs: Vec<_> = (0..16).map(make_dummy_config).collect();
        let result = orchestrator.run_cpu_surrogate_batched(configs, &surrogates);
        assert_eq!(result.len(), 16);
        let mut indices: Vec<usize> = result.iter().map(|(i, _)| *i).collect();
        indices.sort_unstable();
        assert_eq!(indices, (0..16).collect::<Vec<_>>());
    }

    #[test]
    fn batched_matches_unbatched_for_mock_surrogate() {
        // The mock fallback returns a constant 1.2 load per zone, and both
        // paths drive identical physics with the same deterministic daily
        // cycle. So the batched (crossbeam rendezvous) path MUST produce
        // bit-identical EUI per population index — this is the correctness
        // invariant that lets `evaluate_population` swap paths freely.
        let surrogates = SurrogateManager::new().expect("SurrogateManager::new");
        let orchestrator = RayonChunksOrchestrator::with_chunk_size(4);

        let make_pop = || {
            (0..12)
                .map(|i| {
                    let mut m = ThermalModel::<VectorField>::new(1);
                    m.apply_parameters(&[1.5, 20.0, 24.0]);
                    (i, m)
                })
                .collect::<Vec<_>>()
        };

        let mut unbatched = orchestrator.run_cpu_surrogate(make_pop(), &surrogates);
        let mut batched = orchestrator.run_cpu_surrogate_batched(make_pop(), &surrogates);
        unbatched.sort_by_key(|(i, _)| *i);
        batched.sort_by_key(|(i, _)| *i);

        assert_eq!(unbatched.len(), batched.len());
        for ((i1, v1), (i2, v2)) in unbatched.iter().zip(batched.iter()) {
            assert_eq!(i1, i2);
            assert!(
                (v1 - v2).abs() < 1e-12,
                "batched vs unbatched drift: idx={} {} vs {}",
                i1,
                v1,
                v2
            );
        }
    }

    #[test]
    fn batched_deterministic_across_repeated_runs() {
        let surrogates = SurrogateManager::new().expect("SurrogateManager::new");
        let orchestrator = RayonChunksOrchestrator::for_population(8);

        let make_pop = || {
            (0..8)
                .map(|i| {
                    let mut m = ThermalModel::<VectorField>::new(1);
                    m.apply_parameters(&[1.5, 20.0, 24.0]);
                    (i, m)
                })
                .collect::<Vec<_>>()
        };

        let mut r1 = orchestrator.run_cpu_surrogate_batched(make_pop(), &surrogates);
        let mut r2 = orchestrator.run_cpu_surrogate_batched(make_pop(), &surrogates);
        r1.sort_by_key(|(i, _)| *i);
        r2.sort_by_key(|(i, _)| *i);
        for ((i1, v1), (i2, v2)) in r1.iter().zip(r2.iter()) {
            assert_eq!(i1, i2);
            assert!(
                (v1 - v2).abs() < 1e-12,
                "batched run-to-run drift: idx={} {} vs {}",
                i1,
                v1,
                v2
            );
        }
    }
}
