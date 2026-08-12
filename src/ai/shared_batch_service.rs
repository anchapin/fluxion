//! Shared batch inference service for concurrent surrogate requests.
//!
//! Aggregates inference requests from multiple workers into dynamic batches
//! and dispatches them across one or more worker threads that call
//! `SurrogateManager::predict_loads_batched`. Results are returned to the
//! original requesters via per-request `Sender<Vec<f64>>` channels.
//!
//! The service ships two tunables (issue #1438):
//!
//! * **Adaptive `wait_ms`** — each worker maintains an EWMA of its own
//!   per-batch inference latency and sets the next batch-fill window to
//!   `clamp(target_latency_ms - ema_inference_ms, min_wait_ms, max_wait_ms)`.
//!   Fast surrogates get a longer fill window (more batching, fewer kernel
//!   launches); slow surrogates spend less time idle.
//! * **Multi-worker fan-out** — requests are pushed into a single shared
//!   `crossbeam::channel::bounded(capacity)` from which N independent workers
//!   `recv()`. No serialized coordinator overhead; per-request dispatch cost
//!   is one channel send.
//!
//! The original [`DynamicBatchConfig`] + [`SharedBatchInferenceService::new`]
//! API is preserved verbatim (backward compatible); the new
//! [`SchedulerConfig`] + [`SharedBatchInferenceService::with_workers`] entry
//! point opts into multi-worker + the adaptive-wait machinery. Existing
//! callers continue to work unchanged and additionally gain adaptive-wait
//! automatically because the single-worker path is now a strict specialization
//! of the multi-worker scheduler.

use crate::ai::surrogate::SurrogateManager;
use crossbeam::channel::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Configuration for dynamic batch sizing — preserved for backward compatibility.
///
/// New code should prefer [`SchedulerConfig`] which also exposes the adaptive
/// wait bounds and the multi-worker fan-out factor.
#[derive(Clone, Debug)]
pub struct DynamicBatchConfig {
    /// Maximum number of requests to include in a single batch.
    pub max_batch_size: usize,
    /// Maximum time to wait (milliseconds) for a batch to fill before
    /// processing whatever has been collected. Reused as
    /// `SchedulerConfig::target_latency_ms` in the back-compat translation.
    pub wait_ms: u64,
}

impl Default for DynamicBatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 512,
            wait_ms: 10,
        }
    }
}

/// Full configuration for the adaptive + multi-worker scheduler (issue #1438).
///
/// Use [`SchedulerConfig::default`] for a sensible production default, or
/// construct it explicitly to tune for a specific inference backend.
#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    /// Maximum number of requests per batch. Caps the work each worker pulls
    /// out of the channel before it stops waiting and processes the batch.
    pub max_batch_size: usize,
    /// Target end-to-end batch latency in milliseconds. The adaptive-wait
    /// state targets `target_latency_ms - ema_inference_ms` as the next fill
    /// window — i.e. on a fast surrogate the worker waits longer for the
    /// batch to fill, on a slow surrogate it bails out earlier.
    pub target_latency_ms: u64,
    /// Lower clamp for the adaptive wait window.
    pub min_wait_ms: u64,
    /// Upper clamp for the adaptive wait window.
    pub max_wait_ms: u64,
    /// Number of inference workers. `0` (or any value below 1) selects
    /// `available_parallelism / 4` at construction time, clamped to the
    /// inclusive range `[1, 8]`. Set to `1` to recover the old single-worker
    /// semantics with adaptive wait.
    pub num_workers: usize,
    /// Bounded channel capacity for the request queue. Should be at least
    /// the expected number of concurrent producers.
    pub channel_capacity: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        let cpus = num_cpus::get().max(2);
        Self {
            max_batch_size: 512,
            target_latency_ms: 20,
            min_wait_ms: 1,
            max_wait_ms: 50,
            num_workers: (cpus / 4).clamp(1, 8),
            channel_capacity: 4096,
        }
    }
}

impl SchedulerConfig {
    /// Resolve `num_workers == 0` to a positive concrete value derived from
    /// `num_cpus::get()`. Keeps construction cheap to call repeatedly.
    pub fn resolve_num_workers(&self) -> usize {
        if self.num_workers == 0 {
            let cpus = num_cpus::get().max(2);
            (cpus / 4).clamp(1, 8)
        } else {
            self.num_workers
        }
    }
}

/// Back-compat translation from the legacy 2-field config.
///
/// The translation preserves the legacy `wait_ms` as both the adaptive-wait
/// `target_latency_ms` AND the upper clamp, so an existing caller that sets
/// `wait_ms = 100` continues to see up-to-100ms fill windows (just with the
/// adaptive shrinks below that bound when the surrogate turns out to be
/// slow). It pins `num_workers = 1` for source-level back-compat with the
/// single-threaded hot path.
impl From<DynamicBatchConfig> for SchedulerConfig {
    fn from(c: DynamicBatchConfig) -> Self {
        Self {
            max_batch_size: c.max_batch_size,
            target_latency_ms: c.wait_ms.max(1),
            min_wait_ms: 1,
            max_wait_ms: c.wait_ms.max(1),
            num_workers: 1,
            channel_capacity: 4096,
        }
    }
}

/// Atomically-readable batch service metrics (issue #1438, item 3 in the
/// proposed approach). All counters are monotonic and the EMA is an advisory
/// snapshot — they are intended for ops dashboards and CI assertions, not as
/// ground-truth per-request timing.
#[derive(Debug, Default)]
pub struct BatchMetrics {
    /// Cumulative number of batches dispatched across all workers.
    pub batches_processed: AtomicU64,
    /// Cumulative number of individual requests that have been processed
    /// (a batch of size N contributes N to this counter).
    pub requests_processed: AtomicU64,
    /// Last-published adaptive-wait EMA of per-batch inference latency, in
    /// microseconds. Loaded by [`Self::ema_inference_ms`] and surfaced via
    /// [`SharedBatchInferenceService::metrics`].
    pub ema_inference_us: AtomicU64,
}

impl BatchMetrics {
    /// EMA inference latency in milliseconds (floating point).
    pub fn ema_inference_ms(&self) -> f64 {
        self.ema_inference_us.load(Ordering::Relaxed) as f64 / 1000.0
    }

    /// Take a consistent-enough snapshot of all three counters at once.
    /// The three reads are not transactional but the gaps between them are
    /// small enough that dashboard consumers can treat the resulting
    /// `BatchMetricsSnapshot` as a point-in-time view.
    pub fn snapshot(&self) -> BatchMetricsSnapshot {
        BatchMetricsSnapshot {
            batches_processed: self.batches_processed.load(Ordering::Relaxed),
            requests_processed: self.requests_processed.load(Ordering::Relaxed),
            ema_inference_ms: self.ema_inference_ms(),
        }
    }
}

/// Immutable snapshot of [`BatchMetrics`], suitable for logging / serialization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BatchMetricsSnapshot {
    pub batches_processed: u64,
    pub requests_processed: u64,
    pub ema_inference_ms: f64,
}

/// EWMA smoothing factor for the adaptive-wait EMA. `alpha = 0.2` reacts in
/// ~5 batches while staying stable against single-batch noise.
const EWMA_ALPHA: f64 = 0.2;

/// Incoming inference request: temperature vector and channel to send response.
struct InferenceRequest {
    temps: Vec<f64>,
    response_tx: Sender<Vec<f64>>,
}

/// Shared batch inference service.
///
/// Workers call `submit()` to send their temperature vectors and receive a
/// `Receiver<Vec<f64>>` for the corresponding loads. The service aggregates
/// submissions into batches and runs `predict_loads_batched` on one of N
/// worker threads (default = single worker, opt into multi via
/// [`Self::with_workers`]).
#[derive(Clone)]
pub struct SharedBatchInferenceService {
    inner: Arc<Inner>,
}

struct Inner {
    sender: Option<Sender<InferenceRequest>>,
    workers: Vec<JoinHandle<()>>,
    metrics: Arc<BatchMetrics>,
}

impl SharedBatchInferenceService {
    /// Creates a new service with the given surrogate manager, dynamic batch
    /// configuration, and channel capacity (legacy 3-argument form).
    ///
    /// **Backward compatible with the pre-#1438 single-worker API.** The
    /// service is internally a single-worker instance of the new
    /// adaptive-wait scheduler, so it also benefits from EMA-driven wait
    /// shrinkage when the surrogate is slower than the caller's `wait_ms`
    /// assumed. Use [`Self::with_workers`] for the multi-worker fan-out.
    pub fn new(
        surrogate: SurrogateManager,
        config: DynamicBatchConfig,
        channel_capacity: usize,
    ) -> Self {
        let mut sched: SchedulerConfig = config.into();
        // Legacy callers sized the channel to `valid_configs.len()` at the
        // call site (see BatchOracle::evaluate_population, src/lib.rs:1162).
        // Honour their value rather than our default of 4096.
        if channel_capacity > 0 {
            sched.channel_capacity = channel_capacity;
        }
        Self::with_workers(surrogate, sched)
    }

    /// Creates a new service backed by `sched.num_workers` inference worker
    /// threads sharing a single bounded request channel.
    ///
    /// Each worker independently fills its batch up to `max_batch_size`
    /// within an EWMA-adaptive wait window, then runs `predict_loads_batched`
    /// on the accumulated inputs. No coordinator overhead — request dispatch
    /// is a single `channel::send`.
    pub fn with_workers(surrogate: SurrogateManager, sched: SchedulerConfig) -> Self {
        let num_workers = sched.resolve_num_workers();
        let channel_capacity = sched.channel_capacity.max(1);
        let metrics = Arc::new(BatchMetrics::default());

        let (tx, rx) = channel::bounded::<InferenceRequest>(channel_capacity);
        // Keep one sender inside `Inner`; the workers only need the receiver.
        let sender = tx.clone();
        drop(tx); // close the locals-only handle so workers see Disconnected
                  // only when all clones (incl. the `Inner` one) are dropped.

        let mut workers = Vec::with_capacity(num_workers);
        for worker_idx in 0..num_workers {
            let rx = rx.clone();
            let surrogate = surrogate.clone();
            let sched = sched.clone();
            let metrics = Arc::clone(&metrics);
            let handle = thread::Builder::new()
                .name(format!("fluxion-shared-batch-worker-{worker_idx}"))
                .spawn(move || Self::run_worker(rx, surrogate, sched, metrics))
                .expect("failed to spawn SharedBatchInferenceService worker thread");
            workers.push(handle);
        }
        drop(rx); // workers each hold their own clone now

        Self {
            inner: Arc::new(Inner {
                sender: Some(sender),
                workers,
                metrics,
            }),
        }
    }

    /// Submits a temperature vector for inference.
    ///
    /// Returns a `Receiver` that will receive the predicted loads (`Vec<f64>`).
    /// The call is non-blocking; the returned receiver should be used to wait
    /// for the result.
    ///
    /// **Allocation note:** this convenience method allocates a fresh
    /// `crossbeam::channel::unbounded()` per call — one heap block for the
    /// channel structure plus its internal queue. For one-shot requests that
    /// is negligible, but callers that submit once per timestep inside a hot
    /// loop (e.g. [`BatchOracle::evaluate_population`]'s GPU path) should use
    /// [`Self::submit_with_sender`] instead, which reuses a caller-owned
    /// channel and eliminates the per-timestep channel allocation (Issue
    /// #2751).
    pub fn submit(&self, temps: Vec<f64>) -> Receiver<Vec<f64>> {
        let (resp_tx, resp_rx) = channel::unbounded();
        let request = InferenceRequest {
            temps,
            response_tx: resp_tx,
        };
        // Send the request to the service. If the service has been dropped,
        // `send` returns an Err — surface that to the caller via the receiver.
        match self.inner.sender.as_ref() {
            Some(sender) => {
                if let Err(e) = sender.send(request) {
                    // Channel closed before we could push. Forward the empty
                    // vector to wake any caller waiting on `recv()`; the empty
                    // result is the documented "service dropped" signal.
                    let _ = e.0.response_tx.send(Vec::new());
                }
            }
            None => {
                let _ = request.response_tx.send(Vec::new());
            }
        }
        resp_rx
    }

    /// Submits a temperature vector using a caller-provided response sender,
    /// avoiding the per-call channel allocation of [`Self::submit`] (Issue
    /// #2751).
    ///
    /// The caller owns a `(Sender<Vec<f64>>, Receiver<Vec<f64>>)` pair created
    /// **once** outside the hot loop and reuses it across timesteps:
    ///
    /// 1. `submit_with_sender(temps_buf, resp_tx.clone())` — moves the temps
    ///    into the service and hands it a **clone** of the sender (a cheap
    ///    `Arc` bump; crossbeam `Sender` is `Arc`-backed internally, so
    ///    `clone()` performs no heap allocation).
    /// 2. `resp_rx.recv()` — blocks until the service ships the loads back via
    ///    the cloned sender.
    /// 3. The loads `Vec<f64>` received from `resp_rx` is recycled as the next
    ///    timestep's `temps_buf` (ping-pong buffer pattern — same approach the
    ///    CPU batched orchestrator uses, `orchestrator.rs:486`).
    ///
    /// Compared to [`Self::submit`], this eliminates the `channel::unbounded()`
    /// heap allocation that `submit` performs per call — one heap block per
    /// channel structure, per timestep, per worker. For a 1 024-config × 8 760
    /// -timestep run that is ~8.97 M channel allocations removed. The bytes
    /// flowing into and out of the service are bit-identical to `submit`; only
    /// the buffer / channel ownership differs.
    ///
    /// # Error handling
    /// If the service has been dropped (all worker threads exited), the
    /// caller's receiver will receive an empty `Vec::new()` — the same
    /// "service dropped" signal that `submit` uses. The caller should treat an
    /// empty loads vector as a fatal service-disconnect.
    pub fn submit_with_sender(&self, temps: Vec<f64>, response_tx: Sender<Vec<f64>>) {
        let request = InferenceRequest { temps, response_tx };
        match self.inner.sender.as_ref() {
            Some(sender) => {
                if let Err(e) = sender.send(request) {
                    let _ = e.0.response_tx.send(Vec::new());
                }
            }
            None => {
                let _ = request.response_tx.send(Vec::new());
            }
        }
    }

    /// Atomically-readable panel of dispatcher metrics.
    pub fn metrics(&self) -> BatchMetricsSnapshot {
        self.inner.metrics.snapshot()
    }

    /// Handle to the shared atomic metrics panel — useful for embedding in an
    /// ops dashboard that wants incremental reads rather than snapshots.
    pub fn metrics_handle(&self) -> Arc<BatchMetrics> {
        Arc::clone(&self.inner.metrics)
    }

    /// Inference worker main loop with adaptive-wait fill window.
    ///
    /// Lifecycle:
    /// 1. Blocking `recv()` to obtain the first request (returns Err when all
    ///    senders are dropped — exit cleanly).
    /// 2. Up to `max_batch_size - 1` additional requests collected under
    ///    `recv_timeout(Duration::from_millis(adaptive_wait_ms))`.
    /// 3. Time the batch, update the per-worker EMA, publish to the shared
    ///    atomic metrics, and ship results back through the per-request
    ///    `response_tx` channels.
    fn run_worker(
        req_rx: Receiver<InferenceRequest>,
        surrogate: SurrogateManager,
        sched: SchedulerConfig,
        metrics: Arc<BatchMetrics>,
    ) {
        // EWMA smoothing factor: alpha=0.2 reacts in ~5 batches while staying
        // stable against single-batch noise.
        let mut ema_ms: f64 = sched.target_latency_ms as f64;
        let mut wait_ms: u64 = sched.target_latency_ms;

        // Issue #2751: hoist the per-batch scratch buffers above the batch
        // loop. `predict_loads_batched_into` recycles `flat_in` (flattened f32
        // ONNX input), `flat_out` (flattened f64 raw output), and `outputs`
        // (per-config `Vec<f64>` load vectors) via `clear`/`resize_with` in
        // place, so after warm-up these contribute zero heap allocation per
        // batch. `inputs` and `senders` similarly reuse their outer capacity
        // via `clear` each batch; the inner temps `Vec`s are moved out of
        // `InferenceRequest` and dropped on the next `clear`, which is the
        // inherent per-request floor (the temps `Vec` crosses the thread
        // boundary by ownership — see `dhat_batched_surrogate_zero_growth.rs`
        // module docs).
        let mut inputs: Vec<Vec<f64>> = Vec::new();
        let mut senders: Vec<Sender<Vec<f64>>> = Vec::new();
        let mut flat_in: Vec<f32> = Vec::new();
        let mut flat_out: Vec<f64> = Vec::new();
        let mut outputs: Vec<Vec<f64>> = Vec::new();
        // Issue #2751: hoist the per-batch `batch` Vec above the loop so its
        // capacity is reused (clear + refill) instead of reallocated each
        // batch-cycle. Previously this was `Vec::with_capacity(...)` inside the
        // loop — one allocation per batch.
        let mut batch: Vec<InferenceRequest> = Vec::with_capacity(sched.max_batch_size.min(64));

        loop {
            // 1) Block until the first request arrives.
            let first_req = match req_rx.recv() {
                Ok(req) => req,
                Err(_) => break, // All senders gone — exit cleanly.
            };
            batch.clear();
            batch.push(first_req);

            // 2) Try to fill the batch within the adaptive wait window. The
            //    loop bails out on timeout, channel disconnect, or reaching
            //    `max_batch_size`.
            while batch.len() < sched.max_batch_size {
                let timeout = Duration::from_millis(wait_ms);
                match req_rx.recv_timeout(timeout) {
                    Ok(req) => batch.push(req),
                    Err(RecvTimeoutError::Timeout) => break,
                    Err(RecvTimeoutError::Disconnected) => {
                        // Channel closed. Process what we have and exit.
                        Self::process_batch(
                            &mut batch,
                            &surrogate,
                            &metrics,
                            &mut ema_ms,
                            &mut wait_ms,
                            &sched,
                            &mut inputs,
                            &mut senders,
                            &mut flat_in,
                            &mut flat_out,
                            &mut outputs,
                        );
                        return;
                    }
                }
            }

            Self::process_batch(
                &mut batch,
                &surrogate,
                &metrics,
                &mut ema_ms,
                &mut wait_ms,
                &sched,
                &mut inputs,
                &mut senders,
                &mut flat_in,
                &mut flat_out,
                &mut outputs,
            );
        }
    }

    /// Run one batch through the surrogate, update the per-worker adaptive
    /// state, publish metrics, and ship results back to requesters.
    ///
    /// # Hoisted scratch buffers (Issue #2751)
    ///
    /// `inputs`, `senders`, `flat_in`, `flat_out`, and `outputs` are borrowed
    /// from the caller (`run_worker`) and `clear`'d in place each batch so
    /// their capacity is reused across batches rather than reallocated. The
    /// inner `Vec<f64>` load vectors in `outputs` are moved out to requesters
    /// via `drain` (the scatter step), so each batch does allocate `batch.len()`
    /// fresh inner `Vec`s via `predict_loads_batched_into`'s `resize_with` —
    /// that is the inherent per-request floor documented in
    /// `tests/dhat_batched_surrogate_zero_growth.rs` (the load `Vec` must cross
    /// the thread boundary by ownership). Everything else — the flattened f32
    /// ONNX input, the flattened raw output, the outer `Vec` capacities — is
    /// fully reused after warm-up.
    #[allow(clippy::too_many_arguments)]
    fn process_batch(
        batch: &mut Vec<InferenceRequest>,
        surrogate: &SurrogateManager,
        metrics: &Arc<BatchMetrics>,
        ema_ms: &mut f64,
        wait_ms: &mut u64,
        sched: &SchedulerConfig,
        inputs: &mut Vec<Vec<f64>>,
        senders: &mut Vec<Sender<Vec<f64>>>,
        flat_in: &mut Vec<f32>,
        flat_out: &mut Vec<f64>,
        outputs: &mut Vec<Vec<f64>>,
    ) {
        // Unzip the batch into the hoisted `inputs` and `senders` buffers,
        // reusing their outer capacity. The inner temps Vecs are moved out of
        // each InferenceRequest via `drain(..)` (retaining `batch`'s capacity
        // for the next batch-cycle); they are dropped when `inputs.clear()`
        // runs next batch (the borrowing `predict_loads_batched_into` has
        // already returned by then).
        inputs.clear();
        senders.clear();
        inputs.reserve(batch.len());
        senders.reserve(batch.len());
        for req in batch.drain(..) {
            inputs.push(req.temps);
            senders.push(req.response_tx);
        }

        let start = Instant::now();
        // Issue #2751: use the `_into` variant so the flattened f32 ONNX
        // input, the flattened raw output, and the per-config load vectors
        // are recycled via the hoisted scratch buffers rather than allocated
        // fresh each batch. The bytes produced are bit-identical to the prior
        // `predict_loads_batched(&inputs)` call — only buffer ownership
        // differs (verified by `dhat_batched_surrogate_zero_growth`).
        surrogate.predict_loads_batched_into(inputs, flat_in, flat_out, outputs);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Update EMA + adaptive wait.
        *ema_ms = EWMA_ALPHA * elapsed_ms + (1.0 - EWMA_ALPHA) * *ema_ms;
        let target_f = sched.target_latency_ms as f64;
        let raw = (target_f - *ema_ms).max(sched.min_wait_ms as f64);
        let bounded = raw.min(sched.max_wait_ms as f64);
        *wait_ms = bounded.max(sched.min_wait_ms as f64) as u64;

        // Publish to shared atomics. Relaxed ordering is fine — these are
        // advisory metrics consumed by `metrics()` readers and CI benches.
        metrics.batches_processed.fetch_add(1, Ordering::Relaxed);
        metrics
            .requests_processed
            .fetch_add(outputs.len() as u64, Ordering::Relaxed);
        metrics
            .ema_inference_us
            .store((*ema_ms * 1000.0) as u64, Ordering::Relaxed);

        // Ship results back. If a requester dropped, ignore the send error.
        // The inner load Vecs are drained out of `outputs` by ownership —
        // callers that use `submit_with_sender` recycle them as their next
        // timestep's temps buffer (ping-pong pattern, Issue #2751).
        for (tx, out) in senders.drain(..).zip(outputs.drain(..)) {
            let _ = tx.send(out);
        }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        // Dropping the sender makes worker `recv()` calls return Err
        // (closed channel), which is the documented shutdown signal.
        drop(self.sender.take());

        // Join all workers so the OS thread handles are released cleanly.
        // Each worker exits after processing its current batch.
        while let Some(handle) = self.workers.pop() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::surrogate::SurrogateManager;

    /// Construct a `SurrogateManager` for unit tests. Always returns the
    /// mock surrogate (the unit tests do not depend on any ONNX model file).
    fn create_test_surrogate() -> SurrogateManager {
        SurrogateManager::new().expect("SurrogateManager::new never fails")
    }

    #[test]
    fn test_shared_batch_service_single() {
        let surrogate = create_test_surrogate();
        let config = DynamicBatchConfig {
            max_batch_size: 4,
            wait_ms: 10,
        };
        let service = SharedBatchInferenceService::new(surrogate, config, 4);
        let temps = vec![20.0, 21.0];
        let rx = service.submit(temps);
        let result = rx.recv().expect("No result received");
        assert_eq!(result.len(), 2);
        assert!(result[0] > 0.0);
        assert!(result[1] > 0.0);
    }

    #[test]
    fn test_shared_batch_service_concurrent() {
        let surrogate = create_test_surrogate();
        let config = DynamicBatchConfig {
            max_batch_size: 10,
            wait_ms: 100,
        };
        let n_workers = 20;
        let service = SharedBatchInferenceService::new(surrogate, config, n_workers);

        let mut handles = Vec::new();

        for i in 0..n_workers {
            let service = service.clone();
            let handle = thread::spawn(move || {
                let input = vec![20.0 + i as f64, 21.0 + i as f64];
                let rx = service.submit(input.clone());
                rx.recv().expect("Failed to receive output")
            });
            handles.push(handle);
        }

        for h in handles {
            let out = h.join().expect("Thread panicked");
            assert_eq!(out.len(), 2);
            assert!(out[0] > 0.0);
            assert!(out[1] > 0.0);
        }
    }

    #[test]
    fn test_shared_batch_service_batching() {
        let surrogate = create_test_surrogate();
        let config = DynamicBatchConfig {
            max_batch_size: 5,
            wait_ms: 50,
        };
        let service = SharedBatchInferenceService::new(surrogate, config, 10);

        let mut handles = Vec::new();
        let n_requests = 10;

        for i in 0..n_requests {
            let service = service.clone();
            let handle = thread::spawn(move || {
                let rx = service.submit(vec![i as f64, (i + 1) as f64]);
                rx.recv().unwrap()
            });
            handles.push(handle);
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_dynamic_batch_config_default() {
        let config = DynamicBatchConfig::default();
        assert_eq!(config.max_batch_size, 512);
        assert_eq!(config.wait_ms, 10);
    }

    #[test]
    fn test_dynamic_batch_config_custom() {
        let config = DynamicBatchConfig {
            max_batch_size: 64,
            wait_ms: 100,
        };
        assert_eq!(config.max_batch_size, 64);
        assert_eq!(config.wait_ms, 100);
    }

    #[test]
    fn test_dynamic_batch_config_clone() {
        let config = DynamicBatchConfig {
            max_batch_size: 32,
            wait_ms: 50,
        };
        let cloned = config.clone();
        assert_eq!(cloned.max_batch_size, 32);
        assert_eq!(cloned.wait_ms, 50);
    }

    #[test]
    fn test_dynamic_batch_config_debug() {
        let config = DynamicBatchConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("max_batch_size"));
        assert!(debug_str.contains("wait_ms"));
    }

    // ======== Issue #1438: new tests for the multi-worker + adaptive-wait path. ========

    #[test]
    fn test_scheduler_config_default_resolves_workers() {
        let cfg = SchedulerConfig::default();
        let resolved = cfg.resolve_num_workers();
        assert!(
            resolved >= 1,
            "resolved workers must be >= 1, got {resolved}"
        );
        assert!(
            resolved <= 8,
            "resolved workers must be <= 8, got {resolved}"
        );
        assert!(cfg.target_latency_ms >= cfg.min_wait_ms);
        assert!(cfg.target_latency_ms <= cfg.max_wait_ms);
    }

    #[test]
    fn test_dynamic_to_scheduler_translation_preserves_wait_ms() {
        let dyn_cfg = DynamicBatchConfig {
            max_batch_size: 64,
            wait_ms: 100,
        };
        let sched: SchedulerConfig = dyn_cfg.clone().into();
        assert_eq!(sched.max_batch_size, dyn_cfg.max_batch_size);
        assert_eq!(sched.target_latency_ms, dyn_cfg.wait_ms);
        assert_eq!(sched.max_wait_ms, dyn_cfg.wait_ms);
        assert_eq!(
            sched.num_workers, 1,
            "back-compat translation pins single worker"
        );
    }

    #[test]
    fn test_metrics_panel_updates_after_processing() {
        let surrogate = create_test_surrogate();
        let sched = SchedulerConfig {
            max_batch_size: 8,
            target_latency_ms: 5,
            min_wait_ms: 1,
            max_wait_ms: 50,
            num_workers: 2,
            channel_capacity: 16,
        };
        let service = SharedBatchInferenceService::with_workers(surrogate, sched);

        let initial = service.metrics();
        assert_eq!(initial.batches_processed, 0);
        assert_eq!(initial.requests_processed, 0);

        // Submit a small burst from multiple producer threads.
        let n = 32;
        let mut handles = Vec::new();
        for i in 0..n {
            let service = service.clone();
            handles.push(thread::spawn(move || {
                let rx = service.submit(vec![i as f64]);
                let out = rx.recv().expect("response");
                assert_eq!(out.len(), 1);
            }));
        }
        for h in handles {
            h.join().expect("worker thread panicked");
        }

        let snap = service.metrics();
        assert!(
            snap.batches_processed >= 1,
            "expected at least 1 batch, got {}",
            snap.batches_processed
        );
        assert_eq!(snap.requests_processed, n as u64);
        assert!(snap.ema_inference_ms >= 0.0, "EMA must be non-negative");
    }

    #[test]
    fn test_multi_worker_processes_all_requests() {
        // Smoke test: multi-worker scheduler processes every submitted
        // request without losing any. This is the correctness invariant that
        // the throughput benchmark builds on.
        let surrogate = create_test_surrogate();
        let sched = SchedulerConfig {
            max_batch_size: 4,
            target_latency_ms: 10,
            min_wait_ms: 1,
            max_wait_ms: 50,
            num_workers: 4,
            channel_capacity: 64,
        };
        let service = SharedBatchInferenceService::with_workers(surrogate, sched);

        let producers = 16;
        let requests_per_producer = 8;
        let total = producers * requests_per_producer;

        let mut handles = Vec::new();
        for p in 0..producers {
            let service = service.clone();
            handles.push(thread::spawn(move || {
                for q in 0..requests_per_producer {
                    let temps = vec![p as f64 + q as f64 * 0.1, (p + q) as f64];
                    let rx = service.submit(temps);
                    let out = rx.recv().expect("response");
                    assert_eq!(out.len(), 2);
                    // mock SurrogateManager always returns the constant mock load
                    assert!(out[0] > 0.0);
                }
            }));
        }
        for h in handles {
            h.join().expect("producer thread panicked");
        }

        let snap = service.metrics();
        assert_eq!(snap.requests_processed, total as u64);
    }
}
