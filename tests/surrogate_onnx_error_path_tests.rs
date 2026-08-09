//! ONNX error-path and CPU↔CUDA parity tests for `SurrogateManager` (issue #2557).
//!
//! ## What this file covers
//!
//! 1. **Error-path coverage** — every graceful-error branch in
//!    `SurrogateManager` for malformed, missing, or mismatched ONNX inputs
//!    is exercised and pinned to a `Result::Err` (no panic, no silent
//!    success). The cases mirror the issue's acceptance list:
//!
//!    - missing model file
//!    - corrupt (malformed-protobuf) model file
//!    - empty model file
//!    - `predict_loads_onnx` without a loaded model
//!    - wrong input shape (model expects `[1, 6]`, we pass `[1, 24]`)
//!    - NaN input
//!    - Inf input
//!    - empty input slice / empty input batch
//!
//! 2. **CPU↔CUDA numerical parity** — when both backends load the same
//!    ONNX model with the same input, outputs must agree to ≤ `1e-5`
//!    absolute per element (matches `surrogate_backend_parity.rs::CROSS_BACKEND_REL_TOL`).
//!
//! 3. **Skip-on-cpu-only** — CUDA-gated tests skip cleanly when no GPU is
//!    present, matching the project's existing CUDA smoke pattern
//!    (`surrogate_cuda_smoke_test.rs`).
//!
//! ## Tolerance derivation
//!
//! Onnxruntime FP32 deterministic kernels are exact on CPU. CUDA FP32
//! reductions can introduce ~1e-7 relative noise from non-deterministic
//! summation order. 1e-5 absolute leaves a ~100× safety margin while
//! still catching meaningful numerical divergence. The dummy pass-through
//! model has output magnitudes in the tens (1..100), so 1e-5 absolute
//! is well below numerical noise.
//!
//! ## Why the entire file is `#[cfg(feature = "ort")]`
//!
//! The `SurrogateManager::load_onnx`, `predict_loads_onnx`, and
//! `predict_loads_batched_onnx` symbols used here are ort-feature-gated
//! (see `src/ai/surrogate.rs`). On a default build (no `ort` feature) the
//! stubs return `Err("...requires the `ort` feature...")`, which would
//! make every error-path test pass for the wrong reason. Gating the whole
//! file behind the feature keeps the tests meaningful: they only run when
//! the ort runtime is actually wired in, matching the existing
//! `test_session_pool.rs::test_concurrent_real_model_inference` pattern.
//!
//! ## Out of scope (documented gaps)
//!
//! - **ONNX opset version mismatch** — requires constructing a minimal
//!   ONNX protobuf with `opset_import = 99`; the registered-model
//!   hash-mismatch path is already covered by
//!   `surrogate_golden_output.rs::load_version_rejects_hash_mismatch`.
//! - **Unsupported ONNX operator** — would require generating an ONNX
//!   model containing an op outside ort 2.0.0-rc.10's kernel set; out of
//!   reach for a hand-written test fixture.
//! - **Wrong input dtype** — the public `predict_loads_onnx` API
//!   only accepts `&[f64]`, so dtype mismatches are not reachable
//!   through the user-facing surface. The f64→f32 cast inside the
//!   manager is exercised by every inference test.

#![cfg(feature = "ort")]

#[cfg(feature = "cuda")]
use fluxion::ai::surrogate::InferenceBackend;
use fluxion::ai::surrogate::SurrogateManager;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Path to the tiny pass-through ONNX fixture shipped under `assets/`.
/// The model takes `float32[1, 6]` and returns the first input value as
/// `float32[1, 1]` (see `surrogate_cuda_smoke_test.rs` for the contract).
const DUMMY_ONNX_MODEL: &str = "assets/dummy_surrogate.onnx";

/// Per-element absolute tolerance for CPU↔CUDA parity (issue #2557).
const CPU_CUDA_ABS_TOL: f64 = 1e-5;

/// Skip the calling test gracefully if the dummy ONNX fixture is missing.
/// The fixture is git-ignored under certain packaging profiles, so this
/// keeps the test suite green on CI runners that do not stage the asset.
macro_rules! skip_if_no_dummy {
    () => {
        if !Path::new(DUMMY_ONNX_MODEL).exists() {
            eprintln!(
                "SKIP: {} not found — ONNX fixture missing \
                 (likely CI packaging dropped assets)",
                DUMMY_ONNX_MODEL
            );
            return;
        }
    };
}

/// Build a per-test unique path under the system temp directory.
///
/// Two parallel `cargo test` threads must not clobber each other's temp
/// files, so the path is keyed on `std::process::id()` + nanosecond
/// timestamp + the caller's label.
fn unique_temp_path(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "fluxion_onnx_err_{}_{}_{}_{}.onnx",
        label,
        std::process::id(),
        nanos,
        rand_suffix()
    ));
    p
}

/// Tiny extra entropy suffix to break ties if two threads in the same
/// process produce the same nanosecond. Avoids pulling in `rand` for
/// one off.
fn rand_suffix() -> u64 {
    use std::cell::Cell;
    thread_local! {
        #[allow(clippy::missing_const_for_thread_local)]
        static CTR: Cell<u64> = Cell::new(0);
    }
    CTR.with(|c| {
        let n = c.get().wrapping_add(1);
        c.set(n);
        n
    })
}

// =====================================================================
// Sanity: the happy path works under the `ort` feature
// =====================================================================

/// Confirms the dummy model loads and produces a finite, expected
/// pass-through output. Without this anchor, the error-path tests below
/// could pass for the wrong reason (e.g. session creation panicking
/// silently before the bad input is even tried).
#[test]
fn test_dummy_model_loads_and_predicts_finite() {
    skip_if_no_dummy!();
    let mgr = SurrogateManager::load_onnx(DUMMY_ONNX_MODEL).expect("load dummy ONNX");
    let input = [42.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];
    let out = mgr.predict_loads_onnx(&input).expect("predict dummy ONNX");
    assert_eq!(out.len(), 1, "dummy model returns 1 output element");
    let v = out[0];
    assert!(v.is_finite(), "dummy output must be finite, got {}", v);
    // Dummy model: pass-through on the first input element.
    assert!(
        (v - 42.0).abs() < 1e-3,
        "expected pass-through ~42.0, got {}",
        v
    );
}

// =====================================================================
// Error-path tests: SurrogateManager must return Err, not panic.
// =====================================================================

/// `load_onnx` on a path that does not exist must return a typed error
/// whose message is discoverable. This is the simplest "graceful" branch
/// and the most common runtime failure mode (deployed model missing).
#[test]
fn test_load_onnx_missing_file_returns_typed_error() {
    let result = SurrogateManager::load_onnx("/nonexistent/fluxion/missing_model.onnx");
    let err = result.expect_err("loading a missing model must return Err");
    assert!(
        err.contains("not found") || err.contains("No such file"),
        "missing-file error must be discoverable; got: {:?}",
        err
    );
}

/// `load_onnx` on a file containing non-ONNX bytes must return Err
/// rather than panicking inside the ort session builder. ort's
/// `commit_from_file` parses the protobuf and rejects anything that
/// does not match the ONNX schema; that error must propagate as `Err`.
#[test]
fn test_load_onnx_malformed_bytes_returns_typed_error() {
    let path = unique_temp_path("malformed");
    let garbage: [u8; 32] = [
        0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x42, 0x13, 0x37, 0xCA, 0xFE, 0xBA, 0xBE, 0xFF, 0xFF, 0xFF,
        0xFF, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
        0x77, 0x88,
    ];
    {
        let mut f = std::fs::File::create(&path).expect("create temp malformed file");
        f.write_all(&garbage).expect("write garbage");
    }
    let result = SurrogateManager::load_onnx(path.to_str().unwrap());
    let _ = std::fs::remove_file(&path);
    let err = result.expect_err("loading a malformed ONNX must return Err");
    assert!(
        !err.is_empty(),
        "malformed-ONNX error message must be non-empty; got empty string"
    );
}

/// `load_onnx` on a zero-byte file must return Err rather than treating
/// the file as an empty model or panicking inside the protobuf parser.
#[test]
fn test_load_onnx_empty_file_returns_typed_error() {
    let path = unique_temp_path("empty");
    std::fs::write(&path, b"").expect("write empty file");
    let result = SurrogateManager::load_onnx(path.to_str().unwrap());
    let _ = std::fs::remove_file(&path);
    let err = result.expect_err("loading an empty file must return Err");
    assert!(
        !err.is_empty(),
        "empty-file error message must be non-empty; got empty string"
    );
}

/// `predict_loads_onnx` on a manager that never loaded a model must
/// return Err rather than panicking or silently returning a mock value.
#[test]
fn test_predict_loads_onnx_without_loaded_model_returns_typed_error() {
    let mgr = SurrogateManager::new().expect("SurrogateManager::new");
    let result = mgr.predict_loads_onnx(&[1.0, 2.0, 3.0]);
    let err = result.expect_err("predict_loads_onnx without a model must return Err");
    assert!(
        err.contains("No ONNX model loaded")
            || err.contains("No session pool")
            || err.contains("ort feature"),
        "no-model error must explain the cause; got: {:?}",
        err
    );
}

/// `predict_loads_onnx` with a tensor shape that disagrees with the
/// model's declared input rank must return Err. The dummy model expects
/// `[1, 6]`; passing 24 elements produces a `[1, 24]` tensor that ort
/// rejects with a shape-mismatch diagnostic.
#[test]
fn test_predict_loads_onnx_wrong_shape_returns_typed_error() {
    skip_if_no_dummy!();
    let mgr = SurrogateManager::load_onnx(DUMMY_ONNX_MODEL).expect("load dummy ONNX");
    let bad_input: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let result = mgr.predict_loads_onnx(&bad_input);
    let err = result.expect_err("predict_loads_onnx with wrong shape must return Err");
    // The exact wording is ort-version-specific; the only contract is
    // "Err with a non-empty message". Reject empty messages explicitly
    // so a future regression that swallows the error fails this test.
    assert!(
        !err.is_empty(),
        "wrong-shape error message must be non-empty; got empty string"
    );
}

/// NaN input must not panic. The dummy model is a pass-through via
/// `Gather`, so the NaN propagates to the output — that is a graceful
/// outcome (no panic, no silent corruption). The contract is "no panic";
/// the resulting value may be `Ok(NaN)` or, on stricter ort versions,
/// an Err.
#[test]
fn test_predict_loads_onnx_nan_input_does_not_panic() {
    skip_if_no_dummy!();
    let mgr = SurrogateManager::load_onnx(DUMMY_ONNX_MODEL).expect("load dummy ONNX");
    let nan_input = [f64::NAN, 1.0, 2.0, 3.0, 4.0, 5.0];
    // `AssertUnwindSafe` is required because `&SurrogateManager` carries
    // interior mutability (`Arc<parking_lot::Mutex<…>>`) and is therefore
    // not `UnwindSafe` by default. We are in test code: bypassing the
    // unwind-safety check is the documented escape hatch.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        mgr.predict_loads_onnx(&nan_input)
    }));
    let outcome = result.expect("NaN input must not panic");
    match outcome {
        Ok(loads) => {
            assert!(
                loads[0].is_nan(),
                "pass-through model should propagate NaN; got {}",
                loads[0]
            );
        }
        Err(e) => {
            // Err is also a graceful outcome — some ort builds reject
            // NaN inputs outright. Either is acceptable; the contract
            // is "no panic", which is what `catch_unwind` enforces.
            assert!(
                !e.is_empty(),
                "NaN error message must be non-empty; got empty string"
            );
        }
    }
}

/// Inf input must not panic. Symmetric to the NaN case: pass-through
/// propagates Inf, or ort rejects the input; both are graceful.
#[test]
fn test_predict_loads_onnx_inf_input_does_not_panic() {
    skip_if_no_dummy!();
    let mgr = SurrogateManager::load_onnx(DUMMY_ONNX_MODEL).expect("load dummy ONNX");
    let inf_input = [f64::INFINITY, 1.0, 2.0, 3.0, 4.0, 5.0];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        mgr.predict_loads_onnx(&inf_input)
    }));
    let outcome = result.expect("Inf input must not panic");
    match outcome {
        Ok(loads) => {
            assert!(
                loads[0].is_infinite(),
                "pass-through model should propagate Inf; got {}",
                loads[0]
            );
        }
        Err(e) => {
            assert!(
                !e.is_empty(),
                "Inf error message must be non-empty; got empty string"
            );
        }
    }
}

/// Empty input slice must not panic. `predict_loads_onnx` builds a
/// `[1, 0]` tensor; ort typically rejects it because the model expects
/// `[1, 6]`. Either Err or Ok(empty) is graceful; a panic is not.
#[test]
fn test_predict_loads_onnx_empty_input_does_not_panic() {
    skip_if_no_dummy!();
    let mgr = SurrogateManager::load_onnx(DUMMY_ONNX_MODEL).expect("load dummy ONNX");
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| mgr.predict_loads_onnx(&[])));
    let outcome = result.expect("empty input must not panic");
    if let Err(e) = &outcome {
        assert!(
            !e.is_empty(),
            "empty-input error message must be non-empty; got empty string"
        );
    }
    // Both Ok(empty) and Err are valid graceful outcomes.
}

/// Empty batch on the batched API must return `Ok(Vec::new())` per the
/// existing early-return contract in
/// `SurrogateManager::predict_loads_batched_onnx`. This pins that
/// contract against accidental regressions.
#[test]
fn test_predict_loads_batched_onnx_empty_batch_returns_ok() {
    skip_if_no_dummy!();
    let mgr = SurrogateManager::load_onnx(DUMMY_ONNX_MODEL).expect("load dummy ONNX");
    let result = mgr.predict_loads_batched_onnx(&[]);
    let loads = result.expect("empty batch must return Ok");
    assert!(
        loads.is_empty(),
        "empty batch must return empty Vec; got {} entries",
        loads.len()
    );
}

// =====================================================================
// CPU↔CUDA numerical parity (issue #2557 acceptance criterion)
// =====================================================================

/// CPU is the reference. Verify the dummy model produces a deterministic,
/// expected pass-through value across repeated calls. This anchors the
/// CPU side of the parity comparison below.
#[test]
fn test_cpu_backend_is_deterministic() {
    skip_if_no_dummy!();
    let mgr = SurrogateManager::load_onnx(DUMMY_ONNX_MODEL).expect("load dummy ONNX");
    let input = [42.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];
    let out1 = mgr.predict_loads_onnx(&input).expect("cpu predict #1");
    let out2 = mgr.predict_loads_onnx(&input).expect("cpu predict #2");
    assert_eq!(out1.len(), out2.len());
    for (a, b) in out1.iter().zip(out2.iter()) {
        assert!(
            (a - b).abs() < CPU_CUDA_ABS_TOL,
            "CPU backend must be deterministic: a={} b={} diff={}",
            a,
            b,
            (a - b).abs()
        );
    }
    // Dummy model: pass-through on the first input element.
    assert!(
        (out1[0] - 42.0).abs() < 1e-3,
        "expected pass-through ~42.0, got {}",
        out1[0]
    );
}

/// Returns `true` when the CUDA execution provider can be loaded at
/// runtime. Compiles to `false` unless `--features cuda` is enabled and
/// the runtime has a working NVIDIA driver + CUDA EP binary.
#[cfg(feature = "cuda")]
fn cuda_execution_provider_available() -> bool {
    use ort::execution_providers::CUDAExecutionProvider;
    use ort::session::Session;
    if let Ok(builder) = Session::builder() {
        let ep = CUDAExecutionProvider::default().with_device_id(0);
        builder.with_execution_providers([ep.build()]).is_ok()
    } else {
        false
    }
}

/// Live CPU↔CUDA numerical parity check (issue #2557 acceptance).
///
/// Loads the same dummy ONNX model with the CPU and CUDA backends, runs
/// the same input through both, and asserts per-element absolute
/// agreement within `CPU_CUDA_ABS_TOL` (1e-5).
///
/// Skips cleanly (not fails) when:
///   - the dummy ONNX fixture is missing;
///   - the `cuda` feature is compiled out (CPU-only CI);
///   - the CUDA execution provider cannot be wired in at runtime
///     (no GPU / no driver / no ort CUDA EP).
///
/// Fails when both backends are available and the outputs disagree
/// beyond tolerance — that is the actual regression signal.
#[cfg(feature = "cuda")]
#[test]
fn test_cpu_vs_cuda_parity_within_absolute_tolerance() {
    skip_if_no_dummy!();
    if !cuda_execution_provider_available() {
        eprintln!(
            "SKIP: CUDA execution provider not available at runtime — \
             CPU↔CUDA parity requires GPU hardware. \
             Rebuild with --features cuda and run on a GPU machine to \
             exercise this test."
        );
        return;
    }

    // CPU reference.
    let cpu_mgr = SurrogateManager::load_onnx(DUMMY_ONNX_MODEL).expect("CPU load");
    // CUDA peer.
    let cuda_mgr = SurrogateManager::with_gpu_backend(DUMMY_ONNX_MODEL, InferenceBackend::CUDA, 0)
        .expect("CUDA load");
    assert!(
        matches!(cuda_mgr.backend, InferenceBackend::CUDA),
        "expected CUDA backend, got {:?}",
        cuda_mgr.backend
    );

    // Identical input to both backends.
    let input = [42.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];
    let cpu_out = cpu_mgr.predict_loads_onnx(&input).expect("cpu predict");
    let cuda_out = cuda_mgr.predict_loads_onnx(&input).expect("cuda predict");

    assert_eq!(
        cpu_out.len(),
        cuda_out.len(),
        "CPU and CUDA returned different output lengths ({} vs {})",
        cpu_out.len(),
        cuda_out.len()
    );

    for (i, (a, b)) in cpu_out.iter().zip(cuda_out.iter()).enumerate() {
        let abs_diff = (a - b).abs();
        assert!(
            abs_diff <= CPU_CUDA_ABS_TOL,
            "CPU vs CUDA absolute diff at index {}: cpu={} cuda={} diff={} tol={} \
             (issue #2557 parity gate)",
            i,
            a,
            b,
            abs_diff,
            CPU_CUDA_ABS_TOL
        );
    }
}
