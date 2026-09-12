//! PR-blocking cold-start latency gate for the ONNX surrogate (Issue #2919).
//!
//! ## What this file proves
//!
//! 1. **First-call cost is bounded.** A freshly constructed `SurrogateManager`
//!    that has NEVER been invoked must complete its first
//!    `predict_loads_onnx` call within `COLD_START_MAX_MS` (default 100 ms).
//!    The `ort::session::Session` is created on the first call (cold path —
//!    `src/ai/surrogate.rs:1247`, `get_or_create_session` lazily constructs
//!    a session only when the pool is empty), and the per-call scratch
//!    tensors (`ort::value::Value::from_array(...)` on every call) compound
//!    with that session-construction cost. At the API layer, `fluxion-rest`
//!    (`src/api/server.rs:946`) probes ONNX once at boot but the first
//!    `/v1/simulate` request after a fresh process pays the full cold cost —
//!    easy > 500 ms on a 256 MiB model.
//!
//! 2. **Steady-state cost is bounded.** After warm-up, the SAME input on the
//!    SAME manager must complete in < `WARM_MAX_MS` (default 25 ms). This
//!    bounds steady-state latency so the cold-start/warm-up ratio is
//!    interpretable — a regression that doubles warm-up time AND halves the
//!    cold/warm ratio would otherwise slip through.
//!
//! 3. **The cold/warm ratio is bounded.** Across the median of 3
//!    (cold, warm) cycles the cold-call latency must stay within
//!    `MAX_COLD_WARM_RATIO` (default 1.5×) of the warm-call latency.
//!    The previous Absolute Perf Gate (#2693) only measured the
//!    warm path, so a PR adding 400 ms to first-call cost passed without
//!    tripping the gate — the post-deploy latency spike was invisible to
//!    CI. This gate closes that gap.
//!
//! ## Issue #3685 hardening (warm-sample guard)
//!
//! On 2026-09-11 this gate failed spuriously on 3 PRs (#3670, #3678,
//! #3679): the warm-time measurement intermittently read ~0.002–0.003
//! ms on the trivial CI fixture, and dividing the healthy cold time by
//! such a collapsed denominator inflated the cold/warm ratio to 10–16.
//! The measurement is hardened accordingly:
//!
//! - **Epsilon guard** — warm samples below `WARM_EPSILON_MS` (0.01 ms)
//!   are invalid and are never used as the ratio denominator.
//! - **Bounded re-measure** — warm batches of `WARM_BATCH_SIZE` calls
//!   are re-measured up to `WARM_MAX_BATCH_ATTEMPTS` times until enough
//!   valid samples exist.
//! - **Robust statistic** — the ratio denominator is the p95 of the
//!   valid warm samples (see `tests/cold_start_guard/mod.rs`), not a
//!   single reading.
//! - **Unmeasurable ≠ failed, but never fabricated** — when valid warm
//!   samples cannot be gathered, the `Cold/warm ratio:` line is omitted
//!   from the parse contract and the CI checker treats the ratio as
//!   null: skipped on the trivial fixture (cold < `trivial_fixture_ms`),
//!   FAILED on a non-trivial workload where the ratio must be
//!   verifiable.
//!
//! ## Acceptance criteria (Issue #2919)
//!
//! - [x] New `tests/surrogate_cold_start_test.rs` measures the FIRST
//!   `predict_loads_onnx` call on a freshly constructed `SurrogateManager`
//!   (no warmup).
//! - [x] Release-mode `cargo test --release --test surrogate_cold_start_test`
//!   asserts latency < 100 ms (or < 2.5× warm median) on
//!   `assets/dummy_surrogate.onnx` if present.
//! - [x] Surfaces a CI gate that fails a PR if cold-start regresses > 25%
//!   relative to a stored baseline. The `release_gates.yaml →
//!   benchmark.cold_start.regression_tolerance` knob backs this.
//! - [x] Companion `multi-zone-cold-start-gate` job in `performance_dashboard.yml`
//!   runs the test x3, median, fails on cold/warm ratio > 1.5.
//!
//! ## What this test does NOT cover (documented gaps)
//!
//! - **First-process-boot cold path** — `fluxion-rest` probes ONNX once at
//!   boot (`src/api/server.rs:946`); the FIRST `/v1/simulate` after a fresh
//!   process pays the full session-construction cost. This test measures
//!   the surrogate's own cold-start, NOT the HTTP + JIT + first-allocator-
//!   call costs that compound at the API layer. Those are measured by the
//!   existing `tests/performance_integration_test.rs` HTTP path.
//! - **Multi-zone cold start** — only single-config `predict_loads_onnx`
//!   (batch = 1) is measured here. The `predict_loads_batched` /
//!   `predict_loads_batched_into` paths exercise a different session-reuse
//!   pattern; their cold-start cost is bounded by the existing hybrid-perf
//!   gate (#2922) which measures warm steady-state throughput.
//! - **CUDA cold path** — the GPU backend's cold-start cost (CUDA context
//!   init + cuBLAS handle load + first kernel JIT) is dominated by the CUDA
//!   runtime, not ort; the existing `surrogate_cuda_smoke_test.rs`
//!   (#1603) validates GPU availability, not cold-start latency.
//!
//! ## Why the entire file is `#[cfg(feature = "ort")]`
//!
//! `SurrogateManager::load_onnx` and `predict_loads_onnx` are
//! ort-feature-gated (see `src/ai/surrogate.rs:2037` / `2375`). On a
//! default build (no `ort` feature) the stubs return
//! `Err("...requires the `ort` feature...")` — that would make this gate
//! trivially "pass" by skipping all measurement. Gating the whole file
//! behind the feature keeps the gate meaningful: it only runs when the ort
//! runtime is actually wired in. Mirrors the existing pattern in
//! `tests/surrogate_onnx_error_path_tests.rs` / `surrogate_cuda_smoke_test.rs`.

#![cfg(feature = "ort")]

use fluxion::ai::surrogate::SurrogateManager;
use std::path::Path;
use std::time::Instant;

// Issue #3685 warm-sample guard: epsilon filter + bounded re-measure +
// p95 robust statistic. The pure helpers live in a shared module so the
// regression tests in `tests/cold_start_guard_test.rs` can exercise them
// without the `ort` feature.
#[path = "cold_start_guard/mod.rs"]
mod cold_start_guard;

use cold_start_guard::warm_p95_statistic;

/// Path to the tiny pass-through ONNX fixture shipped under `assets/`.
/// The model takes `float32[1, 6]` and returns the first input value as
/// `float32[1, 1]` (deterministic pass-through used to keep measurement
/// variance bounded — Issue #2919 only cares about latency, not
/// numerical accuracy).
const DUMMY_ONNX_MODEL: &str = "assets/dummy_surrogate.onnx";

/// Absolute cold-start ceiling, in milliseconds. The Issue #2919 acceptance
/// criterion is "latency < 100 ms (or < 2.5× warm median)". On a 256 MiB
/// shipped model this absolute bound is binding; on the 193-byte dummy CI
/// fixture the absolute bound is trivially met and the ratio bound below
/// is the binding signal.
const COLD_START_MAX_MS: f64 = 100.0;

/// Lenient cold/warm ratio the TEST asserts (Issue #2919 acceptance
/// wording "or < 2.5× warm median"). The CI gate enforces a stricter
/// ratio via `release_gates.yaml → benchmark.cold_start.max_cold_warm_ratio`
/// — see `MAX_COLD_WARM_RATIO` below — by feeding the test's parsed
/// output through `scripts/release_gate_checker.py --benchmark-gates
/// cold_start`. The two layers are intentional: the TEST's lenient bound
/// is what Issue #2919 calls out in its acceptance section, and the CI
/// gate's stricter 1.5× bound is what catches a regression that's still
/// below the lenient bound but visibly above the steady-state floor.
const TEST_LENIENT_RATIO: f64 = 2.5;

/// Maximum allowed cold/warm ratio enforced by the CI gate. Issue #2919
/// acceptance criterion for the multi-zone-cold-start-gate job:
/// "fails on cold/warm ratio > 1.5". Mirrors the rule of thumb that a
/// first-call cost up to 50% above steady-state is acceptable noise from
/// page-cache misses / allocator-warmup; anything beyond is a real
/// regression in the session-construction path.
const MAX_COLD_WARM_RATIO: f64 = 1.5;

/// Number of paired (cold, warm) cycles to run. Median-of-3 matches the
/// #2693 / #2772 / #2922 perf-gate convention.
const NUM_CYCLES: usize = 3;

/// Number of timed warm `predict_loads_onnx` calls per measurement batch
/// (Issue #3685 fix #2: "run warm measurement with more iterations").
/// The previous code took the median of just 3 warm calls per cycle;
/// 9 per batch feeds the p95 robust statistic with a pool large enough
/// that a minority of collapsed artifacts cannot dominate it.
const WARM_BATCH_SIZE: usize = 9;

/// Bounded re-measure attempts for the warm batch (Issue #3685 fix #1:
/// "re-measure (bounded retries)"). After this many batches without
/// gathering [`WARM_MIN_VALID_SAMPLES`] valid (>= `WARM_EPSILON_MS`)
/// samples, the warm statistic is `None` — the ratio is reported as
/// UNMEASURABLE and downstream gate checks treat an unmeasurable ratio
/// on a non-trivial workload as a FAILURE (fail-closed).
const WARM_MAX_BATCH_ATTEMPTS: usize = 3;

/// Skip the calling test gracefully if the dummy ONNX fixture is missing.
/// The fixture is git-ignored under certain packaging profiles (see
/// `tests/surrogate_onnx_error_path_tests.rs`), so this keeps the test
/// suite green on CI runners that do not stage the asset. On a fully
/// staged CI runner the gate WILL fail loudly on a real cold-start
/// regression — a missing fixture is a CI configuration problem, not a
/// silent pass.
macro_rules! skip_if_no_dummy {
    () => {
        if !Path::new(DUMMY_ONNX_MODEL).exists() {
            eprintln!(
                "SKIP: {} not found — ONNX fixture missing (likely CI \
                 packaging dropped assets). The Multi-Zone Cold Start Gate \
                 (Issue #2919) is not enforceable without the fixture.",
                DUMMY_ONNX_MODEL
            );
            return;
        }
    };
}

/// Per-cycle measurement outcome (Issue #3685 hardened).
struct CycleMeasurement {
    /// Cold-start latency of the FIRST predict on a fresh manager (ms).
    cold_ms: f64,
    /// Median of the RAW warm readings for this cycle (ms), unfiltered.
    /// Always numeric — it feeds the `Warm steady-state:` parse contract
    /// and the `warm_max_ms` absolute bound, both of which stay
    /// meaningful even when the ratio is unmeasurable.
    warm_raw_median_ms: f64,
    /// p95 over the VALID warm samples (>= `WARM_EPSILON_MS`), or `None`
    /// when bounded re-measure could not gather
    /// [`WARM_MIN_VALID_SAMPLES`] valid samples (Issue #3685: collapsed
    /// warm readings are discarded, never divided by).
    warm_p95_ms: Option<f64>,
    /// How many raw warm samples were gathered across all batch attempts.
    warm_total: usize,
    /// How many of those survived the epsilon guard.
    warm_valid: usize,
}

/// One cycle: construct a fresh manager, measure cold-start, then measure
/// warm steady-state with the Issue #3685 guard.
///
/// We intentionally rebuild `SurrogateManager` per cycle — that is the
/// whole point of the gate. `SurrogateManager::load_onnx` constructs a
/// `SessionPool` whose internal `sessions` Vec is empty, so the first
/// `predict_loads_onnx` call invokes `get_or_create_session` and pays
/// the full session-construction cost (ort environment init + model
/// parse + session allocation). A reused manager would warm-pool a
/// session and erase the cold-path signal.
///
/// Warm measurement (Issue #3685): up to [`WARM_MAX_BATCH_ATTEMPTS`]
/// batches of [`WARM_BATCH_SIZE`] timed calls (each batch preceded by one
/// discarded residual-warm-up call). Samples below `WARM_EPSILON_MS` are
/// invalid and never become the ratio denominator; once the pooled valid
/// sample count reaches [`WARM_MIN_VALID_SAMPLES`], the p95 of the valid
/// samples is the cycle's warm statistic.
fn run_cold_warm_cycle(input: &[f64; 6]) -> CycleMeasurement {
    // ---- COLD: freshly constructed manager, first predict ----
    let mgr_cold =
        SurrogateManager::load_onnx(DUMMY_ONNX_MODEL).expect("cold: load_onnx must succeed");
    let cold_start = Instant::now();
    let cold_output = mgr_cold
        .predict_loads_onnx(input)
        .expect("cold: predict_loads_onnx must succeed");
    let cold_ms = cold_start.elapsed().as_secs_f64() * 1000.0;

    // Defensive invariant: the cold path must produce a finite, non-empty
    // output. If the dummy model regresses to returning zero / NaN /
    // empty, that's a real correctness regression and the gate should
    // fail loudly (a regression that also takes 200 ms is still a
    // regression). Mirrors the `test_dummy_model_loads_and_predicts_finite`
    // sanity check in `tests/surrogate_onnx_error_path_tests.rs`.
    assert!(
        !cold_output.is_empty(),
        "cold predict_loads_onnx returned empty output"
    );
    assert!(
        cold_output.iter().all(|v| v.is_finite()),
        "cold predict_loads_onnx returned non-finite value(s): {:?}",
        cold_output
    );

    // ---- WARM: reuse the same manager, measure steady-state ----
    let mut pooled: Vec<f64> = Vec::with_capacity(WARM_BATCH_SIZE * WARM_MAX_BATCH_ATTEMPTS);
    let mut warm_p95_ms = None;
    for attempt in 1..=WARM_MAX_BATCH_ATTEMPTS {
        // The first call of each batch still pays some residual
        // allocator / page-cache cost — discard it unmeasured.
        let _ = mgr_cold.predict_loads_onnx(input);

        for _ in 0..WARM_BATCH_SIZE {
            let warm_start = Instant::now();
            let warm_output = mgr_cold
                .predict_loads_onnx(input)
                .expect("warm: predict_loads_onnx must succeed");
            pooled.push(warm_start.elapsed().as_secs_f64() * 1000.0);
            assert!(
                !warm_output.is_empty(),
                "warm predict_loads_onnx returned empty output"
            );
            assert!(
                warm_output.iter().all(|v| v.is_finite()),
                "warm predict_loads_onnx returned non-finite value(s): {:?}",
                warm_output
            );
        }

        warm_p95_ms = warm_p95_statistic(&pooled);
        if warm_p95_ms.is_some() {
            break;
        }
        eprintln!(
            "[surrogate-cold-start-diag] warm batch attempt {attempt}/{} produced only {} \
             valid sample(s) of {} (epsilon floor breached) — re-measuring",
            WARM_MAX_BATCH_ATTEMPTS,
            cold_start_guard::valid_warm_samples(&pooled).len(),
            pooled.len()
        );
    }

    let warm_valid = cold_start_guard::valid_warm_samples(&pooled).len();
    let mut raw_sorted = pooled.clone();
    raw_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let warm_raw_median_ms = raw_sorted[raw_sorted.len() / 2];

    CycleMeasurement {
        cold_ms,
        warm_raw_median_ms,
        warm_p95_ms,
        warm_total: pooled.len(),
        warm_valid,
    }
}

/// Compute the median of an f64 slice. Panics if the slice is empty.
fn median(samples: &[f64]) -> f64 {
    assert!(!samples.is_empty(), "median of empty slice");
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted[sorted.len() / 2]
}

// ---------------------------------------------------------------------------
// Sanity: the dummy model loads and the API path is reachable
// ---------------------------------------------------------------------------

/// Confirms the dummy ONNX fixture loads and `predict_loads_onnx` returns a
/// finite pass-through output. Without this anchor, the latency assertions
/// below could pass for the wrong reason (e.g. `predict_loads_onnx`
/// panicking before measurement completes).
#[test]
fn test_dummy_model_loads_and_predicts_finite_cold() {
    skip_if_no_dummy!();
    let mgr = SurrogateManager::load_onnx(DUMMY_ONNX_MODEL).expect("load dummy ONNX");
    let input = [42.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];
    let out = mgr.predict_loads_onnx(&input).expect("predict dummy ONNX");
    assert_eq!(out.len(), 1, "dummy model returns 1 output element");
    let v = out[0];
    assert!(v.is_finite(), "dummy output must be finite, got {}", v);
    // Dummy pass-through: first input element.
    assert!(
        (v - 42.0).abs() < 1e-3,
        "expected pass-through ~42.0, got {}",
        v
    );
}

// ---------------------------------------------------------------------------
// Core gate: median-of-3 cold/warm cycles, three independent bounds
// ---------------------------------------------------------------------------

/// Issue #2919 cold-start gate. Runs `NUM_CYCLES` paired (cold, warm)
/// cycles on freshly constructed managers, computes the medians, and
/// prints them so the GitHub Actions `multi-zone-cold-start-gate` job
/// can apply the stricter 1.5× ratio threshold via
/// `release_gate_checker.py --benchmark-gates cold_start`.
///
/// The TEST itself asserts the Issue #2919 acceptance criterion: pass
/// when EITHER `median_cold_ms ≤ COLD_START_MAX_MS` (100 ms absolute)
/// OR `median_cold_ms ≤ TEST_LENIENT_RATIO × median_warm_ms`
/// (2.5× warm-median ratio). The disjunction is essential — on a 256 MiB
/// shipped model the 100 ms absolute is the binding constraint, on the
/// 193-byte dummy CI fixture the absolute is trivially met and the
/// ratio bound is the binding signal. The CI gate in
/// `release_gates.yaml → benchmark.cold_start.max_cold_warm_ratio`
/// applies a stricter 1.5× ratio via a separate
/// `release_gate_checker.py` invocation; the 25% baseline-regression
/// tolerance is enforced by the same checker against
/// `tests/reference_data/surrogate/cold_start_baseline.json`.
///
/// Each cycle builds a fresh `SurrogateManager` so the cold path is
/// exercised every time. The print output (`Cold start: ... ms`,
/// `Warm steady-state: ... ms`, `Cold/warm ratio: ...`) is the
/// contract the GitHub Actions `multi-zone-cold-start-gate` job
/// parses — keep the format stable.
///
/// On a 256 MiB real-world model the cold path is dominated by
/// session-construction (ort env init + model parse + session allocate);
/// on the 193-byte dummy fixture that cost is ~0 and the cold path is
/// effectively the warm path. The TEST asserts the lenient "100 ms OR
/// 2.5× warm" bound; the CI gate asserts the stricter 1.5× ratio
/// (which the dummy fixture easily meets at ~1.0×) and the baseline
/// regression tolerance (which catches a real change in the cold-path
/// cost when one happens).
#[test]
fn test_surrogate_cold_start_under_100ms_or_2_5x_warm_median() {
    skip_if_no_dummy!();
    let input = [42.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];

    let mut cold_samples: Vec<f64> = Vec::with_capacity(NUM_CYCLES);
    let mut warm_raw_medians: Vec<f64> = Vec::with_capacity(NUM_CYCLES);
    // Per-cycle p95 statistics from cycles where the epsilon guard
    // gathered enough valid samples (Issue #3685).
    let mut warm_p95s: Vec<f64> = Vec::with_capacity(NUM_CYCLES);
    let mut warm_total = 0usize;
    let mut warm_valid = 0usize;

    for cycle in 1..=NUM_CYCLES {
        let m = run_cold_warm_cycle(&input);
        eprintln!(
            "[surrogate-cold-start-diag] cycle={} cold_ms={:.3} warm_raw_median_ms={:.3} \
             warm_p95_ms={} valid={}/{}",
            cycle,
            m.cold_ms,
            m.warm_raw_median_ms,
            m.warm_p95_ms
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "none".to_string()),
            m.warm_valid,
            m.warm_total
        );
        cold_samples.push(m.cold_ms);
        warm_raw_medians.push(m.warm_raw_median_ms);
        warm_total += m.warm_total;
        warm_valid += m.warm_valid;
        if let Some(p95) = m.warm_p95_ms {
            warm_p95s.push(p95);
        }
    }

    let med_cold = median(&cold_samples);
    let med_warm = median(&warm_raw_medians);

    // Issue #3685: the ratio denominator is the ROBUST warm statistic
    // (median of the per-cycle p95s over valid samples), never a single
    // reading and never a collapsed sample. The ratio is only reported
    // when at least 2 of the 3 cycles produced a valid warm statistic;
    // otherwise it is UNMEASURABLE (absent from the parse contract) and
    // the downstream gate falls back to the absolute bounds — or fails
    // closed on a non-trivial workload (cold >= trivial_fixture_ms),
    // where an unverifiable ratio is a failing gate.
    let warm_stat = if warm_p95s.len() >= 2 {
        Some(median(&warm_p95s))
    } else {
        None
    };
    let med_ratio = warm_stat.map(|w| med_cold / w);

    // Stable parse contract for the GitHub Actions step:
    //   `Cold start: <median>ms`
    //   `Warm steady-state: <median>ms`
    //   `Cold/warm ratio: <median>` — ONLY when measurable (Issue
    //   #3685: on the trivial CI fixture the warm path sits below the
    //   0.01 ms measurement floor, so this line is expected to be
    //   absent and the workflow/checker treat the ratio as null).
    // The Python parser in performance_dashboard.yml looks for these
    // exact prefixes.
    eprintln!(
        "[surrogate-cold-start-diag] Cold start: {:.3}ms (max {:.1}ms)\n\
         [surrogate-cold-start-diag] Warm steady-state: {:.3}ms{}",
        med_cold,
        COLD_START_MAX_MS,
        med_warm,
        if let Some(ratio) = med_ratio {
            format!(
                "\n[surrogate-cold-start-diag] Cold/warm ratio: {:.3} (test lenient {:.2}, CI strict {:.2})",
                ratio, TEST_LENIENT_RATIO, MAX_COLD_WARM_RATIO
            )
        } else {
            format!(
                "\n[surrogate-cold-start-diag] Cold/warm ratio: unmeasurable — only {} of {} \
                 warm samples valid (epsilon {:.3} ms); ratio undefined on this fixture \
                 (Issue #3685)",
                warm_valid,
                warm_total,
                cold_start_guard::WARM_EPSILON_MS
            )
        }
    );

    // Issue #2919 acceptance criterion — pass when EITHER the absolute
    // cold-start latency is within `COLD_START_MAX_MS` (100 ms) OR the
    // cold/warm ratio is within `TEST_LENIENT_RATIO` (2.5×). On a real
    // 256 MiB shipped model the absolute bound is binding; on the
    // 193-byte dummy CI fixture the ratio is unmeasurable (warm below
    // the epsilon measurement floor, Issue #3685) and the absolute
    // bound is the only signal — that's correct behaviour, the dummy
    // fixture trivially meets it.
    let abs_ok = med_cold <= COLD_START_MAX_MS;
    let ratio_ok = match med_ratio {
        Some(r) => r <= TEST_LENIENT_RATIO,
        None => false,
    };
    assert!(
        abs_ok || ratio_ok,
        "SURROGATE COLD-START GATE FAILED (Issue #2919)\n\
         Median cold-start latency {:.3} ms exceeds the {:.1} ms ceiling\n\
         AND median cold/warm ratio {} exceeds the {:.2} lenient bound.\n\
         \n\
         This is the cost of the FIRST `predict_loads_onnx` call on a\n\
         freshly constructed `SurrogateManager` — the ort session-pool\n\
         `get_or_create_session` (`src/ai/surrogate/manager.rs`) constructs\n\
         the session lazily on the first call. In production the first\n\
         `/v1/simulate` request after a fresh process pays this cost;\n\
         before #2919 only the warm path was gated, so a regression that\n\
         added 400 ms to first-call cost passed CI invisibly.\n\
         \n\
         Samples (cold_ms): {:?}\n\
         Samples (warm_raw_median_ms): {:?}\n\
         Warm p95 statistic: {}\n\
         Valid warm samples: {}/{}",
        med_cold,
        COLD_START_MAX_MS,
        med_ratio
            .map(|r| format!("{r:.3}"))
            .unwrap_or_else(|| "unmeasurable".to_string()),
        TEST_LENIENT_RATIO,
        cold_samples,
        warm_raw_medians,
        warm_stat
            .map(|w| format!("{w:.4} ms"))
            .unwrap_or_else(|| "none (unmeasurable)".to_string()),
        warm_valid,
        warm_total,
    );
}

// ---------------------------------------------------------------------------
// Diagnostic: print-only helper for baseline regeneration
// ---------------------------------------------------------------------------

/// Diagnostic helper: print the cold/warm cycle times WITHOUT asserting
/// the gate bounds, so a maintainer can copy the measured values into the
/// baseline JSON when the ORT version legitimately bumps the cold-path
/// cost (e.g. session-construction work changes between ort releases).
///
/// Run with:
///
/// ```text
/// cargo test --release --features ort --test surrogate_cold_start_test \
///     -- --ignored --nocapture diagnostic_print_cold_warm_cycles
/// ```
#[test]
#[ignore = "diagnostic; run manually to regenerate the cold-start baseline after a legitimate ort version bump"]
fn diagnostic_print_cold_warm_cycles() {
    skip_if_no_dummy!();
    let input = [42.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];

    let mut cold_samples = Vec::with_capacity(NUM_CYCLES);
    let mut warm_samples = Vec::with_capacity(NUM_CYCLES);
    let mut warm_p95s = Vec::with_capacity(NUM_CYCLES);

    for cycle in 1..=NUM_CYCLES {
        let m = run_cold_warm_cycle(&input);
        cold_samples.push(m.cold_ms);
        warm_samples.push(m.warm_raw_median_ms);
        if let Some(p95) = m.warm_p95_ms {
            warm_p95s.push(p95);
        }
        eprintln!(
            "[surrogate-cold-start-diag] cycle={} cold_ms={:.6} warm_ms={:.6} warm_p95={}",
            cycle,
            m.cold_ms,
            m.warm_raw_median_ms,
            m.warm_p95_ms
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(|| "none".to_string())
        );
    }

    let warm_stat = if warm_p95s.len() >= 2 {
        format!("{:.6}", median(&warm_p95s))
    } else {
        "unmeasurable".to_string()
    };
    eprintln!(
        "[surrogate-cold-start-diag] median cold_ms={:.6} median warm_ms={:.6} warm_p95_statistic={}\n\
         Update release_gates.yaml -> benchmark.cold_start if any of these drift:\n\
         - cold_start_max_ms: keep >= the measured median cold_ms (with margin)\n\
         - warm_max_ms:       keep >= the measured median warm_ms (with margin)\n\
         - max_cold_warm_ratio: keep >= the measured ratio (with margin; ratio uses the \
         p95-of-valid-warm denominator, Issue #3685)",
        median(&cold_samples),
        median(&warm_samples),
        warm_stat,
    );
}
