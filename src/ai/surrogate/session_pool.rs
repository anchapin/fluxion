//! ONNX Runtime session pools and execution-provider probing.
//!
//! [`SessionPool`] is the thread-safe pool of `ort` sessions behind
//! [`SurrogateManager`](crate::ai::surrogate::SurrogateManager); the
//! multi-device pool, backend/config types, and the Issue #3313 runtime
//! execution-provider probe live here too.

// Issue #3313: CoreML/DirectML EP types only exist when the matching `ort`
// feature is enabled (`ort/coreml` / `ort/directml`), so the target-OS gate
// alone is not sufficient to reference them. Each is wired through a
// dedicated fluxion feature (see Cargo.toml); `create_session` degrades
// gracefully with an explicit rebuild hint when the feature is missing.
#[allow(unused_imports)]
use log::warn;
#[cfg(all(feature = "ort", feature = "coreml", target_os = "macos"))]
use ort::ep::CoreML;
#[cfg(all(feature = "ort", feature = "directml", target_os = "windows"))]
use ort::ep::DirectML;
#[cfg(feature = "ort")]
#[cfg(feature = "cuda")]
use ort::ep::CUDA;
#[allow(unused_imports)]
use parking_lot::Mutex;
#[allow(unused_imports)]
use std::path::Path;
use std::sync::Arc;

#[allow(unused_imports)]
#[cfg(feature = "ort")]
use super::integrity::open_and_verify_onnx;

#[derive(Clone, Debug, Copy, Default, PartialEq, Eq)]
/// Inference backend for ONNX runtime execution.
pub enum InferenceBackend {
    #[default]
    CPU,
    CUDA,
    CoreML,
    DirectML,
    OpenVINO,
}

impl InferenceBackend {
    /// Lowercase Prometheus label value for this backend (Issue #2498). Used
    /// as the `backend` label on `fluxion_onnx_*` metrics so dashboards can
    /// split CPU vs CUDA vs CoreML vs DirectML vs OpenVINO throughput.
    pub fn as_str(self) -> &'static str {
        match self {
            InferenceBackend::CPU => "cpu",
            InferenceBackend::CUDA => "cuda",
            InferenceBackend::CoreML => "coreml",
            InferenceBackend::DirectML => "directml",
            InferenceBackend::OpenVINO => "openvino",
        }
    }
}

/// Coarse cardinality-bounded label for the batch size of an ONNX inference
/// call (Issue #2498). Keeps the `batch_bucket` label of
/// `fluxion_onnx_inference_duration_seconds` to 5 values regardless of how
/// many distinct batch sizes callers feed in.
#[cfg(feature = "ort")]
pub(crate) fn batch_bucket_label(batch_size: usize) -> &'static str {
    match batch_size {
        0 => "0",
        1 => "1",
        2..=8 => "2-8",
        9..=64 => "9-64",
        _ => "65+",
    }
}

#[derive(Clone, Debug, Copy, Default, PartialEq, Eq)]
pub enum QuantizationType {
    #[default]
    FP32,
    FP16,
    INT8,
}

#[derive(Clone, Debug, Default)]
pub struct QuantizationConfig {
    pub quantization_type: QuantizationType,
    pub auto_quantize: bool,
}

impl QuantizationConfig {
    pub fn fp32() -> Self {
        QuantizationConfig {
            quantization_type: QuantizationType::FP32,
            auto_quantize: false,
        }
    }
    pub fn fp16() -> Self {
        QuantizationConfig {
            quantization_type: QuantizationType::FP16,
            auto_quantize: true,
        }
    }
    pub fn int8() -> Self {
        QuantizationConfig {
            quantization_type: QuantizationType::INT8,
            auto_quantize: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MultiDeviceConfig {
    pub device_ids: Vec<usize>,
    pub sessions_per_device: usize,
    pub auto_select: bool,
    pub enable_affinity: bool,
    pub fallback_to_cpu: bool,
    pub max_retries: usize,
}

impl MultiDeviceConfig {
    pub fn single_gpu(device_id: usize) -> Self {
        MultiDeviceConfig {
            device_ids: vec![device_id],
            sessions_per_device: 4,
            auto_select: false,
            enable_affinity: true,
            fallback_to_cpu: true,
            max_retries: 3,
        }
    }
    pub fn multi_gpu(device_ids: Vec<usize>) -> Self {
        MultiDeviceConfig {
            device_ids,
            sessions_per_device: 2,
            auto_select: false,
            enable_affinity: true,
            fallback_to_cpu: true,
            max_retries: 3,
        }
    }
    pub fn auto() -> Self {
        MultiDeviceConfig {
            device_ids: vec![],
            sessions_per_device: 4,
            auto_select: true,
            enable_affinity: false,
            fallback_to_cpu: true,
            max_retries: 3,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CudaDeviceInfo {
    pub device_id: usize,
    pub name: String,
    pub compute_capability: Option<(u32, u32)>,
}

#[derive(Clone, Debug, Default)]
pub enum LoadBalancingStrategy {
    #[default]
    RoundRobin,
    LeastLoaded,
    Random,
}

/// Thread-safe pool of ONNX Runtime sessions for concurrent inference.
///
/// Issue #1294: When the `ort` feature is disabled, `SessionPool` is an inert
/// stub. The public surface (`Option<Arc<SessionPool>>` field on
/// [`crate::ai::surrogate::SurrogateManager`], `SessionPool::new` callers in tests) still compiles —
/// only `create_session` actually attempts to load an ONNX model, and it
/// returns an error when `ort` is disabled.
#[cfg(feature = "ort")]
#[derive(Debug)]
#[allow(dead_code)]
pub struct SessionPool {
    sessions: Mutex<Vec<ort::session::Session>>,
    model_path: String,
    /// SHA-256-verified ONNX model bytes (Issue #3573).
    ///
    /// Held as an `Arc` so [`MultiDeviceSessionPool`] can share a single
    /// verified buffer across per-device sub-pools without a second copy
    /// of the model in memory. Every session produced by this pool —
    /// including the lazy pool-reuse path in `get_or_create_session` —
    /// is built from these bytes via `commit_from_memory` rather than
    /// re-resolving `model_path` on the filesystem.
    model_bytes: Arc<Vec<u8>>,
    backend: InferenceBackend,
    device_id: usize,
}

/// Inert stub of [`SessionPool`] used when the `ort` feature is disabled
/// (issue #1294). Carries no ONNX state. Construction succeeds; any attempt
/// to actually create an ONNX session via the corresponding methods returns
/// an error (those methods only exist under `#[cfg(feature = "ort")]`).
///
/// `model_bytes` mirrors the `Arc<Vec<u8>>` field on the ort-feature
/// counterpart (Issue #3573) but is inert: we never call
/// `commit_from_memory` here because the `ort` crate is disabled.
#[cfg(not(feature = "ort"))]
#[derive(Debug)]
#[allow(dead_code)]
pub struct SessionPool {
    model_path: String,
    model_bytes: Arc<Vec<u8>>,
    backend: InferenceBackend,
    device_id: usize,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct MultiDeviceSessionPool {
    #[allow(dead_code)]
    // `pub(crate)`: pre-split this was a same-module field accessed from the
    // manager code that now lives in `surrogate/manager.rs` (Issue #3669
    // decomposition). Visibility widened one step so the ort-gated
    // `from_bytes` consumer compiles exactly as before the split.
    pub(crate) device_pools: Vec<Arc<SessionPool>>,
    _config: MultiDeviceConfig,
    _model_path: String,
}

#[cfg(feature = "ort")]
impl MultiDeviceSessionPool {
    /// Public constructor preserved for source-compat — opens the model
    /// with `O_NOFOLLOW` (Issue #3573), hashes it, and forwards the
    /// verified bytes to [`Self::from_bytes`]. The original `path` is
    /// retained only for diagnostics.
    pub fn new(model_path: String, config: &MultiDeviceConfig) -> Result<Self, String> {
        let bytes = open_and_verify_onnx(Path::new(&model_path))?;
        Self::from_bytes(Arc::new(bytes), model_path, config)
    }

    /// Preferred constructor (Issue #3573). Builds one `SessionPool` per
    /// CUDA device, each pointing at the same SHA-256-verified byte
    /// buffer. Because every session is materialised from the shared
    /// `Arc<Vec<u8>>` via `commit_from_memory`, no per-device pool ever
    /// re-resolves the original path — closing the TOCTOU window that
    /// previously existed between `verify_onnx_signature` and
    /// `SessionPool::create_session(path, …)`.
    pub fn from_bytes(
        model_bytes: Arc<Vec<u8>>,
        model_path: String,
        config: &MultiDeviceConfig,
    ) -> Result<Self, String> {
        let mut device_pools = Vec::new();
        let device_ids = if config.auto_select {
            Self::detect_cuda_devices().unwrap_or_else(|| vec![0])
        } else if config.device_ids.is_empty() {
            vec![0]
        } else {
            config.device_ids.clone()
        };

        for device_id in &device_ids {
            match SessionPool::create_session_from_bytes(
                &model_bytes,
                InferenceBackend::CUDA,
                *device_id,
            ) {
                Ok(session) => {
                    let pool = SessionPool::new(
                        model_path.clone(),
                        Arc::clone(&model_bytes),
                        InferenceBackend::CUDA,
                        *device_id,
                        session,
                    );
                    device_pools.push(Arc::new(pool));
                }
                Err(e) => eprintln!(
                    "Warning: Failed to create session for device {}: {}",
                    device_id, e
                ),
            }
        }

        if device_pools.is_empty() {
            return Err("Failed to create any device pools".to_string());
        }

        Ok(MultiDeviceSessionPool {
            device_pools,
            _config: config.clone(),
            _model_path: model_path,
        })
    }

    pub(crate) fn detect_cuda_devices() -> Option<Vec<usize>> {
        #[cfg(feature = "cuda")]
        {
            use ort::session::Session;
            // Issue #3313: attaching a CUDA EP to a session builder succeeds
            // even on machines with no GPU (provider initialization is
            // deferred to session creation, where failure silently falls
            // back to CPU). Gate discovery on ORT's own EP-device
            // enumeration first — `Environment::devices()` only reports a
            // `CUDAExecutionProvider` device when ORT actually found CUDA
            // hardware at environment creation. The registration probe
            // below is retained as a fallback for backends whose EP-ABI
            // device enumeration is unavailable (it returns an empty
            // device list there).
            let cuda_device_enumerated = ort::environment::Environment::current()
                .ok()
                .map(|env| {
                    env.devices()
                        .any(|d| d.ep().map(|name| name == ep_names::CUDA).unwrap_or(false))
                })
                .unwrap_or(false);
            let mut available_devices = Vec::new();
            if cuda_device_enumerated {
                for device_id in 0..8 {
                    let builder = match Session::builder() {
                        Ok(b) => b,
                        Err(_) => continue,
                    };
                    let cuda_ep = CUDA::default().with_device_id(device_id as i32);
                    match builder.with_execution_providers([cuda_ep.build()]) {
                        Ok(_) => available_devices.push(device_id),
                        Err(_) => continue,
                    }
                }
            }
            if available_devices.is_empty() {
                None
            } else {
                Some(available_devices)
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    pub fn get_cuda_device_info() -> Option<Vec<CudaDeviceInfo>> {
        #[cfg(feature = "cuda")]
        {
            let devices = Self::detect_cuda_devices()?;
            Some(
                devices
                    .into_iter()
                    .map(|id| CudaDeviceInfo {
                        device_id: id,
                        name: format!("GPU {}", id),
                        compute_capability: None,
                    })
                    .collect(),
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    pub fn get_session(&self) -> Result<MultiDeviceSessionGuard, String> {
        for pool in &self.device_pools {
            if let Ok(_session) = pool.get_or_create_session() {
                return Ok(MultiDeviceSessionGuard {
                    pool: Arc::clone(pool),
                });
            }
        }
        Err("No available sessions in multi-device pool".to_string())
    }

    pub fn num_devices(&self) -> usize {
        self.device_pools.len()
    }
}

/// Stub implementation when the `ort` feature is disabled (issue #1294).
/// `MultiDeviceSessionPool::new` always fails; `num_devices` reports zero.
#[cfg(not(feature = "ort"))]
impl MultiDeviceSessionPool {
    pub fn new(_model_path: String, _config: &MultiDeviceConfig) -> Result<Self, String> {
        Err("Multi-device ONNX inference requires the `ort` feature".to_string())
    }

    pub fn num_devices(&self) -> usize {
        0
    }
}

/// RAII guard returned by [`MultiDeviceSessionPool::get_session`].
///
/// Issue #1294: only available with the `ort` feature — the `run_inference`
/// signature depends on `ort::value::Value`.
#[cfg(feature = "ort")]
pub struct MultiDeviceSessionGuard {
    pool: Arc<SessionPool>,
}

#[cfg(feature = "ort")]
impl MultiDeviceSessionGuard {
    pub fn run_inference(&self, input_tensor: ort::value::Value) -> Result<Vec<f64>, String> {
        let mut guard = self.pool.get_or_create_session()?;
        let outputs = guard
            .run(ort::inputs![input_tensor])
            .map_err(|e| e.to_string())?;
        if outputs.len() > 0 {
            let array = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| e.to_string())?;
            Ok(array.iter().copied().map(|x| x as f64).collect())
        } else {
            Err("No outputs from inference".to_string())
        }
    }
}

#[cfg(feature = "ort")]
impl SessionPool {
    pub(crate) fn new(
        model_path: String,
        model_bytes: Arc<Vec<u8>>,
        backend: InferenceBackend,
        device_id: usize,
        initial_session: ort::session::Session,
    ) -> Self {
        SessionPool {
            sessions: Mutex::new(vec![initial_session]),
            model_path,
            model_bytes,
            backend,
            device_id,
        }
    }

    /// Build a fresh ONNX session from this pool's verified byte buffer.
    ///
    /// Used by the [`crate::ai::surrogate::SurrogateManager`] constructors
    /// (`with_gpu_backend`, `with_multi_device`) so the model leaves the
    /// filesystem exactly once — at `open_and_verify_onnx` time. Lazier
    /// session creation (when the per-slot cache is empty) goes through
    /// [`Self::get_or_create_session`] and reuses the same cached bytes,
    /// never re-resolving the path.
    pub(crate) fn create_session_from_bytes(
        model_bytes: &[u8],
        backend: InferenceBackend,
        _device_id: usize,
    ) -> Result<ort::session::Session, String> {
        use ort::session::Session;
        #[allow(unused_mut)]
        let mut builder =
            Session::builder().map_err(|e| format!("Failed to create session builder: {}", e))?;
        match backend {
            InferenceBackend::CUDA => {
                #[cfg(feature = "cuda")]
                {
                    let ep = CUDA::default().with_device_id(_device_id as i32);
                    builder = builder
                        .with_execution_providers([ep.build()])
                        .map_err(|e| format!("Failed to add CUDA execution provider: {}", e))?;
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(
                        "CUDA backend requested but fluxion was built without the `cuda` feature; \
                         rebuild with `cargo build --features cuda` or set FLUXION_ONNX_BACKEND=cpu"
                            .to_string(),
                    );
                }
            }
            InferenceBackend::CoreML => {
                #[cfg(all(feature = "coreml", target_os = "macos"))]
                {
                    let ep = CoreML::default();
                    builder = builder
                        .with_execution_providers([ep.build()])
                        .map_err(|e| format!("Failed to add CoreML execution provider: {}", e))?;
                }
                #[cfg(not(all(feature = "coreml", target_os = "macos")))]
                {
                    if cfg!(target_os = "macos") {
                        return Err(
                            "CoreML backend requested but fluxion was built without the `coreml` feature; \
                             rebuild with `cargo build --features coreml` or set FLUXION_ONNX_BACKEND=cpu"
                                .to_string(),
                        );
                    }
                    return Err("CoreML backend requested but is only available on macOS; \
                         set FLUXION_ONNX_BACKEND=cpu or FLUXION_ONNX_BACKEND=cuda"
                        .to_string());
                }
            }
            InferenceBackend::DirectML => {
                #[cfg(all(feature = "directml", target_os = "windows"))]
                {
                    let ep = DirectML::default().with_device_id(_device_id as i32);
                    builder = builder
                        .with_execution_providers([ep.build()])
                        .map_err(|e| format!("Failed to add DirectML execution provider: {}", e))?;
                }
                #[cfg(not(all(feature = "directml", target_os = "windows")))]
                {
                    if cfg!(target_os = "windows") {
                        return Err(
                            "DirectML backend requested but fluxion was built without the `directml` feature; \
                             rebuild with `cargo build --features directml` or set FLUXION_ONNX_BACKEND=cpu"
                                .to_string(),
                        );
                    }
                    return Err(
                        "DirectML backend requested but is only available on Windows; \
                         set FLUXION_ONNX_BACKEND=cpu or FLUXION_ONNX_BACKEND=cuda"
                            .to_string(),
                    );
                }
            }
            InferenceBackend::OpenVINO => {
                return Err(
                    "OpenVINO execution provider is not available in the pre-built \
                     ort v2.0.0-rc.13 binaries for any platform. \
                     OpenVINO requires building ONNX Runtime from source with Intel OpenVINO toolkit. \
                     Use FLUXION_ONNX_BACKEND=cpu or FLUXION_ONNX_BACKEND=cuda instead."
                        .to_string(),
                );
            }
            InferenceBackend::CPU => {}
        }
        builder
            .commit_from_memory(model_bytes)
            .map_err(|e| format!("Failed to load ONNX model: {}", e))
    }

    /// Back-compat shim that retains the old `path` signature for any
    /// downstream test or inert stub that happens to call it directly.
    /// Goes through `open_and_verify_onnx` so the path is opened with
    /// `O_NOFOLLOW` exactly once and the verified bytes are handed to
    /// `commit_from_memory` (no second filesystem read).
    #[allow(dead_code)]
    fn create_session(
        path: &str,
        backend: InferenceBackend,
        device_id: usize,
    ) -> Result<ort::session::Session, String> {
        let bytes = open_and_verify_onnx(Path::new(path))?;
        Self::create_session_from_bytes(&bytes, backend, device_id)
    }

    pub(crate) fn get_or_create_session(&self) -> Result<SessionGuard<'_>, String> {
        {
            let mut sessions = self.sessions.lock();
            if let Some(session) = sessions.pop() {
                return Ok(SessionGuard {
                    pool: self,
                    session: Some(session),
                });
            }
        }
        // Build the new session from the cached verified bytes — we
        // never re-open `self.model_path` here, so an attacker who
        // swaps the file AFTER the manager was constructed cannot
        // influence these later session loads (Issue #3573).
        Self::create_session_from_bytes(&self.model_bytes, self.backend, self.device_id).map(
            |session| SessionGuard {
                pool: self,
                session: Some(session),
            },
        )
    }

    fn return_session(&self, session: ort::session::Session) {
        let mut sessions = self.sessions.lock();
        sessions.push(session);
    }
}

/// Stub [`SessionPool`] methods when the `ort` feature is disabled
/// (issue #1294). `SessionPool::new` accepts a model path and verified
/// bytes (Issue #3573) but never loads a session; `get_or_create_session`
/// returns an error explaining that ONNX inference is unavailable.
#[cfg(not(feature = "ort"))]
impl SessionPool {
    #[allow(dead_code)]
    pub(crate) fn new(
        model_path: String,
        model_bytes: Arc<Vec<u8>>,
        backend: InferenceBackend,
        device_id: usize,
    ) -> Self {
        SessionPool {
            model_path,
            model_bytes,
            backend,
            device_id,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn get_or_create_session(&self) -> Result<SessionGuard<'_>, String> {
        Err("ONNX inference requires the `ort` feature (build with --features ort)".to_string())
    }
}

#[cfg(feature = "ort")]
pub(crate) struct SessionGuard<'a> {
    pool: &'a SessionPool,
    session: Option<ort::session::Session>,
}

/// Stub [`SessionGuard`] when the `ort` feature is disabled (issue #1294).
/// Constructed by [`SessionPool::get_or_create_session`] in its error path,
/// but never actually used (the function returns `Err` before producing it).
#[cfg(not(feature = "ort"))]
#[allow(dead_code)]
pub(crate) struct SessionGuard<'a> {
    _pool: &'a SessionPool,
}

#[cfg(feature = "ort")]
impl<'a> Drop for SessionGuard<'a> {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            self.pool.return_session(session);
        }
    }
}

#[cfg(feature = "ort")]
impl<'a> std::ops::Deref for SessionGuard<'a> {
    type Target = ort::session::Session;
    fn deref(&self) -> &Self::Target {
        self.session.as_ref().unwrap()
    }
}

#[cfg(feature = "ort")]
impl<'a> std::ops::DerefMut for SessionGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.session.as_mut().unwrap()
    }
}

// ===== Issue #3313 — runtime execution-provider probe =======================
//
// The ort rc.13 migration (#3296) was verified compile-only: `cargo check`
// proves the EP API surface exists, but says nothing about whether a
// GPU/NPU execution provider actually activates at runtime. ORT's classic
// failure mode is the *silent fallback*: `with_execution_providers`
// succeeds, session creation succeeds, and every node quietly runs on the
// CPU EP.
//
// [`ExecutionProviderReport::capture`] makes EP status *observable* by
// combining three independent signals per provider:
//
// 1. `compiled_in`      — can this binary construct the EP at all
//                         (fluxion feature + target-OS gate)?
// 2. `environment_device_present` — did ORT enumerate a real hardware
//                         device for this EP at environment creation
//                         (`Environment::devices()`)? Absent hardware ⇒
//                         absent device, no guessing.
// 3. `registration`     — does attaching the EP (with
//                         `error_on_failure`) to a session builder succeed?
//                         Catches missing provider shared libraries.
//
// On any machine — GPU or not — `capture()` never panics and degrades to an
// explicit per-provider verdict, so the CPU-only contract (EP absent ⇒
// reported unavailable ⇒ CPU path used) is directly assertable in tests.

/// ONNX Runtime's canonical execution-provider names, as reported by
/// [`ort::device::Device::ep`] and [`ort::ep::ExecutionProvider::name`].
///
/// Note the DirectML spelling: ORT names it `DmlExecutionProvider`.
#[cfg(feature = "ort")]
mod ep_names {
    pub const CPU: &str = "CPUExecutionProvider";
    pub const CUDA: &str = "CUDAExecutionProvider";
    pub const COREML: &str = "CoreMLExecutionProvider";
    pub const DIRECTML: &str = "DmlExecutionProvider";
}

/// One hardware device enumerated by ORT for an execution provider
/// (`Environment::devices()` → `OrtEpDevice`).
#[cfg(feature = "ort")]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EpDeviceSummary {
    /// EP name, e.g. `"CUDAExecutionProvider"` / `"CPUExecutionProvider"`.
    pub ep_name: String,
    /// EP vendor, e.g. `"Microsoft"` for DirectML devices.
    pub ep_vendor: Option<String>,
    /// `"cpu"`, `"gpu"`, or `"npu"`.
    pub hardware_type: &'static str,
    /// Hardware manufacturer, when ORT reports one.
    pub hardware_vendor: Option<String>,
    /// Device id as reported by ORT (may differ from CUDA device ordinals).
    pub device_id: Option<u32>,
}

/// Probe verdict for a single [`InferenceBackend`] execution provider.
#[cfg(feature = "ort")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpProbeOutcome {
    pub backend: InferenceBackend,
    /// ORT's canonical name for this EP (see the `ep_names` constants).
    pub ep_name: &'static str,
    /// `true` when this binary can construct the EP (feature + target gate).
    pub compiled_in: bool,
    /// `Some(reason)` when the current target OS cannot ever use this EP
    /// (e.g. CoreML outside macOS). Independent of feature gates.
    pub unsupported_on_target: Option<&'static str>,
    /// `true` when ORT enumerated a real hardware device for this EP in the
    /// current environment.
    pub environment_device_present: bool,
    /// Result of attaching this EP (with `error_on_failure`) to a session
    /// builder. `None` when the probe was skipped because the EP type is
    /// not compiled into this binary.
    pub registration: Option<Result<(), String>>,
    /// Best-effort activation verdict: compiled in AND a device was
    /// enumerated AND registration succeeded. For definitive per-node
    /// assignment proof see `docs/ORT_EP_VALIDATION.md` (ORT EP-assignment
    /// log lines, and the hardware-gated `#[ignore]` tests in this module).
    pub activated: bool,
}

#[cfg(feature = "ort")]
impl EpProbeOutcome {
    /// Human-readable one-line status, e.g. for `--list-backends` style
    /// diagnostics or the validation runbook.
    pub fn status_line(&self) -> String {
        if self.activated {
            format!("{}: ACTIVE ({})", self.ep_name, self.backend.as_str())
        } else if let Some(reason) = self.unsupported_on_target {
            format!("{}: unavailable on this target — {}", self.ep_name, reason)
        } else if !self.compiled_in {
            format!(
                "{}: not compiled into this binary (backend `{}` needs its fluxion feature)",
                self.ep_name,
                self.backend.as_str()
            )
        } else if !self.environment_device_present {
            format!(
                "{}: compiled in, but ORT enumerated no hardware device — silent CPU fallback would occur",
                self.ep_name
            )
        } else {
            format!(
                "{}: device present but registration failed — {:?}",
                self.ep_name, self.registration
            )
        }
    }
}

/// Snapshot of execution-provider availability and activation status for
/// the running process (issue #3313).
///
/// Obtain via [`ExecutionProviderReport::capture`]. Probes cover the CPU
/// baseline plus every target-appropriate GPU/NPU EP (CUDA, CoreML,
/// DirectML); EPs that are not applicable on the current target are still
/// reported, with `unsupported_on_target` explaining why.
#[cfg(feature = "ort")]
#[derive(Clone, Debug)]
pub struct ExecutionProviderReport {
    /// ONNX Runtime API version of the linked backend (`ort::MINOR_VERSION`).
    pub ort_api_version: u32,
    /// All hardware devices ORT enumerated, across all EPs.
    pub devices: Vec<EpDeviceSummary>,
    /// One probe per backend, always including the CPU baseline.
    pub probes: Vec<EpProbeOutcome>,
}

#[cfg(feature = "ort")]
impl ExecutionProviderReport {
    /// Probe every relevant execution provider in the current process.
    ///
    /// Never panics and never returns `Err`: EP absence is a *report*, not
    /// a failure. Safe to call repeatedly; the ORT environment is a
    /// process-global singleton.
    pub fn capture() -> Self {
        // Enumerate real EP devices. `Environment::devices()` returns an
        // empty iterator when the linked backend lacks EP-ABI device
        // enumeration support, in which case `device_enumeration` degrades
        // to `false` and activation verdicts fall back to compile-time +
        // registration signals only.
        let env = ort::environment::Environment::current()
            .map_err(|e| {
                warn!("EP probe: could not obtain ORT environment: {}", e);
                e
            })
            .ok();
        let devices: Vec<EpDeviceSummary> = env
            .as_ref()
            .map(|env| {
                env.devices()
                    .map(|d| {
                        let hw = d.hardware_device();
                        EpDeviceSummary {
                            ep_name: d.ep().unwrap_or("<unknown>").to_string(),
                            ep_vendor: d.ep_vendor().ok().map(str::to_string),
                            hardware_type: match hw.ty() {
                                ort::memory::DeviceType::CPU => "cpu",
                                ort::memory::DeviceType::GPU => "gpu",
                                ort::memory::DeviceType::NPU => "npu",
                            },
                            hardware_vendor: hw.vendor().ok().map(str::to_string),
                            device_id: Some(hw.id()),
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        let device_enumeration = !devices.is_empty();

        let mut probes = Vec::with_capacity(4);

        // ---- CPU baseline: always compiled in, always target-appropriate.
        let cpu_registration = {
            use ort::ep::ExecutionProvider as _;
            ort::ep::CPU::default()
                .is_available()
                .map(|_| ())
                .map_err(|e| e.to_string())
        };
        probes.push(EpProbeOutcome {
            backend: InferenceBackend::CPU,
            ep_name: ep_names::CPU,
            compiled_in: true,
            unsupported_on_target: None,
            environment_device_present: !device_enumeration
                || devices.iter().any(|d| d.ep_name == ep_names::CPU),
            registration: Some(cpu_registration),
            // CPU is the fallback of last resort: report it active whenever
            // the runtime is usable at all, even if device enumeration is
            // unsupported (then the CPU EP is the only meaningful answer).
            activated: true,
        });

        // ---- CUDA (Linux/Windows + NVIDIA GPU, `--features cuda`).
        probes.push(Self::probe_ep(
            InferenceBackend::CUDA,
            ep_names::CUDA,
            cfg!(feature = "cuda"),
            if cfg!(target_os = "macos") {
                Some("CUDA execution providers are not shipped for macOS ORT builds")
            } else {
                None
            },
            &devices,
            device_enumeration,
            || {
                #[cfg(feature = "cuda")]
                {
                    (|| -> Result<(), String> {
                        // `Session::builder` and `with_execution_providers`
                        // carry different recoverable-error payloads, so
                        // chain via `?` inside a `String`-error closure.
                        let builder =
                            ort::session::Session::builder().map_err(|e| e.to_string())?;
                        builder
                            .with_execution_providers([ort::ep::CUDA::default()
                                .with_device_id(0)
                                .build()
                                .error_on_failure()])
                            .map(|_| ())
                            .map_err(|e| e.to_string())
                    })()
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err("not compiled in".to_string())
                }
            },
        ));

        // ---- CoreML (Apple Silicon macOS, `--features coreml`).
        probes.push(Self::probe_ep(
            InferenceBackend::CoreML,
            ep_names::COREML,
            cfg!(all(feature = "coreml", target_os = "macos")),
            if !cfg!(target_os = "macos") {
                Some("CoreML is only available on macOS")
            } else {
                None
            },
            &devices,
            device_enumeration,
            || {
                #[cfg(all(feature = "coreml", target_os = "macos"))]
                {
                    (|| -> Result<(), String> {
                        let builder =
                            ort::session::Session::builder().map_err(|e| e.to_string())?;
                        builder
                            .with_execution_providers([ort::ep::CoreML::default()
                                .build()
                                .error_on_failure()])
                            .map(|_| ())
                            .map_err(|e| e.to_string())
                    })()
                }
                #[cfg(not(all(feature = "coreml", target_os = "macos")))]
                {
                    Err("not compiled in".to_string())
                }
            },
        ));

        // ---- DirectML (Windows + DirectX 12 GPU, `--features directml`).
        probes.push(Self::probe_ep(
            InferenceBackend::DirectML,
            ep_names::DIRECTML,
            cfg!(all(feature = "directml", target_os = "windows")),
            if !cfg!(target_os = "windows") {
                Some("DirectML is only available on Windows")
            } else {
                None
            },
            &devices,
            device_enumeration,
            || {
                #[cfg(all(feature = "directml", target_os = "windows"))]
                {
                    (|| -> Result<(), String> {
                        let builder =
                            ort::session::Session::builder().map_err(|e| e.to_string())?;
                        builder
                            .with_execution_providers([ort::ep::DirectML::default()
                                .with_device_id(0)
                                .build()
                                .error_on_failure()])
                            .map(|_| ())
                            .map_err(|e| e.to_string())
                    })()
                }
                #[cfg(not(all(feature = "directml", target_os = "windows")))]
                {
                    Err("not compiled in".to_string())
                }
            },
        ));

        ExecutionProviderReport {
            ort_api_version: ort::MINOR_VERSION,
            devices,
            probes,
        }
    }

    /// Assemble one [`EpProbeOutcome`] from the shared device list plus a
    /// lazily-run registration probe (skipped entirely when `compiled_in`
    /// is `false`).
    fn probe_ep(
        backend: InferenceBackend,
        ep_name: &'static str,
        compiled_in: bool,
        unsupported_on_target: Option<&'static str>,
        devices: &[EpDeviceSummary],
        device_enumeration: bool,
        registration: impl FnOnce() -> Result<(), String>,
    ) -> EpProbeOutcome {
        let environment_device_present = devices.iter().any(|d| d.ep_name == ep_name);
        let registration = if compiled_in {
            Some(registration())
        } else {
            None
        };
        let activated = compiled_in
            && unsupported_on_target.is_none()
            && (!device_enumeration || environment_device_present)
            && registration.as_ref().is_some_and(|r| r.is_ok());
        EpProbeOutcome {
            backend,
            ep_name,
            compiled_in,
            unsupported_on_target,
            environment_device_present,
            registration,
            activated,
        }
    }

    /// Probe result for `backend`, if it was probed.
    pub fn probe(&self, backend: InferenceBackend) -> Option<&EpProbeOutcome> {
        self.probes.iter().find(|p| p.backend == backend)
    }

    /// `true` when no GPU/NPU EP activated, i.e. inference runs (or would
    /// run) on the CPU execution provider.
    pub fn cpu_only(&self) -> bool {
        !self
            .probes
            .iter()
            .any(|p| p.activated && p.backend != InferenceBackend::CPU)
    }

    /// Backends whose probe concluded `activated`.
    pub fn activated_backends(&self) -> Vec<InferenceBackend> {
        self.probes
            .iter()
            .filter(|p| p.activated)
            .map(|p| p.backend)
            .collect()
    }

    /// Human-readable status lines, one per probe, for logs and the
    /// validation runbook (`ExecutionProviderReport::capture()` output).
    pub fn status_lines(&self) -> Vec<String> {
        let mut lines = vec![format!(
            "ORT api version: {} (device enumeration: {})",
            self.ort_api_version,
            if self.devices.is_empty() {
                "unavailable"
            } else {
                "available"
            }
        )];
        for d in &self.devices {
            lines.push(format!(
                "device: {} [{}, vendor {:?}, id {:?}] (ep vendor {:?})",
                d.ep_name, d.hardware_type, d.hardware_vendor, d.device_id, d.ep_vendor
            ));
        }
        lines.extend(self.probes.iter().map(EpProbeOutcome::status_line));
        lines
    }
}
