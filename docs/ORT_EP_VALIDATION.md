# ORT Execution Provider Runtime Validation (issue #3313)

Runtime validation runbook for CUDA / CoreML / DirectML execution providers after the ort 2.0.0-rc.13 migration (#3296).
Audience: maintainers and operators with access to matching hardware (NVIDIA GPU / Apple Silicon / Windows + DirectX 12).
Covers: the `ExecutionProviderReport::capture()` probe in `src/ai/surrogate.rs`, hardware-gated `#[ignore]` tests, exact commands, and the log lines that prove EP activation.
Related: `docs/KNOWN_ISSUES.md` (validation gaps), `docs/ASHRAE140_RESULTS.md` (status snapshot), `src/ai/surrogate.rs` (`ort`-gated probe code), Cargo.toml (`ort`, `cuda`, `coreml`, `directml` features).
Scope: observability and validation plumbing only — no physics, no energy balance, no numerical behavior is affected by the probe or this runbook.

## Background

The rc.13 migration (#3296) was verified **compile-only** for GPU paths: `cargo check`
proves the API surface (`ort::ep::{CUDA, CoreML, DirectML}`) exists, but says nothing
about whether an execution provider actually **activates** at runtime. ORT's failure
mode is the *silent fallback*: attaching an EP succeeds, session creation succeeds, and
every node quietly runs on the CPU EP at CPU speed.

Issue #3313 therefore requires runtime validation on real hardware. This machine's
constraints (no NVIDIA GPU, no macOS, no Windows) cannot produce that proof directly —
so the deliverable is to make runtime validation **possible, automatic, and honest**:

1. **Observable**: `ExecutionProviderReport::capture()` probes every target EP and
   reports per-provider verdicts (compiled in / device enumerated / registration /
   activated) without panicking when hardware is absent.
2. **Automatic**: CPU-only machines run the graceful-fallback contract tests on every
   `--features ort` build (CI already does this).
3. **Honest**: true-activation assertions are `#[ignore]`-gated tests that require the
   named hardware; they never fake a pass on machines that cannot execute them.
   Ignore census from this issue: **3 entries** (one per EP), documented in the test
   header at `src/ai/surrogate.rs` (`surrogate::tests`, `hardware_*` tests).

## The probe API

```rust
use fluxion::ai::surrogate::{ExecutionProviderReport, InferenceBackend};

let report = ExecutionProviderReport::capture();
for line in report.status_lines() {
    println!("{line}"); // ORT api version, enumerated devices, per-EP verdicts
}
assert!(report.cpu_only()); // no GPU/NPU EP activated → CPU path in effect
```

Each `EpProbeOutcome` combines three independent signals:

| Signal | Source | Catches |
|---|---|---|
| `compiled_in` | fluxion feature + target-OS gate | "built without the EP" (e.g. missing `--features cuda`) |
| `environment_device_present` | `Environment::devices()` (ORT EP-ABI hardware enumeration) | GPU/driver absent — the classic silent-fallback precondition |
| `registration` | attaching the EP (`error_on_failure`) to a throwaway session builder | missing provider shared libraries |

`activated = compiled_in ∧ device_present ∧ registration_ok`. For *definitive*
per-node assignment proof, run the hardware-gated tests below on the real hardware and
check ORT's log lines (next section).

## Hardware validation matrix

| EP | Target | Fluxion feature | ORT EP name | Probe test (`#[ignore]`) |
|---|---|---|---|---|
| CUDA | Linux/Windows + NVIDIA GPU + CUDA 12.x runtime | `--features ort,cuda` | `CUDAExecutionProvider` | `hardware_cuda_ep_probe_reports_activation_and_runs_inference` |
| CoreML | Apple Silicon macOS | `--features ort,coreml` | `CoreMLExecutionProvider` | `hardware_coreml_ep_probe_reports_activation_and_runs_inference` |
| DirectML | Windows + DirectX 12 GPU | `--features ort,directml` | `DmlExecutionProvider` | `hardware_directml_ep_probe_reports_activation_and_runs_inference` |
| CPU | everywhere (baseline) | `--features ort` | `CPUExecutionProvider` | always-green contract tests (not ignored) |

> **Feature-gate note (fixed in this issue):** `ort::ep::CoreML` / `ort::ep::DirectML`
> only exist when the matching `ort` crate feature is enabled, so a bare
> `--features ort` build on macOS/Windows previously could not compile the EP import
> paths at all. Fluxion now exposes dedicated `coreml` / `directml` features (mirroring
> `cuda`), and `SessionPool::create_session` returns an explicit, actionable error when
> a backend is requested without its feature — never a panic, never a silent downgrade.

## Exact commands

### 0. CPU-only machine (no hardware needed — must always pass)

```bash
cargo check -p fluxion --features ort
cargo test  -p fluxion --features ort --lib surrogate::tests::ep_report_
cargo test  -p fluxion --features ort --lib surrogate::tests::coreml_session_request_degrades_gracefully_on_linux
cargo test  -p fluxion --features ort --lib surrogate::tests::directml_session_request_degrades_gracefully_on_linux
cargo test  -p fluxion --features ort --lib surrogate::tests::detect_cuda_devices_returns_none_without_cuda_feature
```

Expected: all green. The report must show CUDA not compiled in, CoreML/DirectML
target-inappropriate, CPU active, and `cpu_only() == true`. This is the
graceful-fallback contract of the issue's last checkbox.

### 1. CUDA (Linux/Windows + NVIDIA GPU)

```bash
nvidia-smi                          # driver + GPU present?
cargo test -p fluxion --features ort,cuda --lib surrogate::tests::hardware_cuda_ -- --ignored --nocapture
```

Then exercise the ONNX surrogate inference path with the GPU backend:

```bash
FLUXION_ONNX_BACKEND=cuda RUST_LOG=info cargo run -p fluxion --bin fluxion -- \
    benchmark --model models/surrogate_zone_thermal.onnx --runs 10 2>&1 \
    | grep -Ei 'cuda|fall.?back|backend'
# or run any ONNX-gated integration test with the backend env var set:
FLUXION_ONNX_BACKEND=cuda RUST_LOG=info cargo test --features ort,cuda --test surrogate_cold_start_test -- --nocapture
```

Expected: the ignored test passes (probe verdict `activated`, real inference through
the production `with_gpu_backend` path), and the logs show CUDA EP registration
without a CPU fallback line.

### 2. CoreML (Apple Silicon macOS)

```bash
cargo test -p fluxion --features ort,coreml --lib surrogate::tests::hardware_coreml_ -- --ignored --nocapture
FLUXION_ONNX_BACKEND=coreml RUST_LOG=info cargo test --features ort,coreml --lib surrogate:: -- --nocapture
```

### 3. DirectML (Windows)

```powershell
cargo test -p fluxion --features ort,directml --lib surrogate::tests::hardware_directml_ -- --ignored --nocapture
$env:FLUXION_ONNX_BACKEND="directml"; RUST_LOG=info cargo test --features ort,directml --lib surrogate:: -- --nocapture
```

## What log lines prove EP activation

`ort` rc.13 routes ORT's internal logger into `tracing` (feature on by default), so
`RUST_LOG=info` (or `debug`) surfaces ORT's EP-assignment output:

**Failure / silent-fallback signatures (must be ABSENT):**

- `Falling back to CPUExecutionProvider` — ORT could not initialize the requested EP
  and downgraded the session.
- `Failed to create CUDAExecutionProvider` (resp. `CoreMLExecutionProvider` /
  `DmlExecutionProvider`) — provider construction failed (missing DLLs/dylibs, driver
  mismatch).
- `No execution providers from session options registered successfully; may fall back
  to CPU.` — `ort`'s own warning when every registration failed.
- `An error occurred when attempting to register` — per-EP registration failure
  (only printed in the default `fail_silently` mode).

**Success signatures (must be PRESENT):**

- `Successfully registered \`CUDAExecutionProvider\`` (resp. `CoreMLExecutionProvider`
  / `DmlExecutionProvider`) — emitted by `ort` when the EP attached to session options.
- With `RUST_LOG=debug`, ORT's session-initialization output names the EPs actually
  computing nodes; the requested EP must appear **without** any fallback line.
- On the fluxion side, `SessionPool` init logs `backend: CUDA|CoreML|DirectML`, and
  issue #2920's one-shot `tracing::warn!` (target
  `fluxion::ai::surrogate::backend`, message `downgraded to CPU`) must **not** fire on
  a properly activated GPU path.

Rule of thumb: **no fallback signature + success signature + probe `activated == true`
+ numerical pass-through output = EP validated.**

## Session-pool device enumeration

`MultiDeviceSessionPool::detect_cuda_devices()` (the `available_devices` enumeration)
previously accepted any device id whose EP *registration* succeeded — which is every
id, even with no GPU. It now gates discovery on ORT's own EP-device enumeration
(`Environment::devices()` must list a `CUDAExecutionProvider` device) before probing
ordinals, so the pool degrades gracefully (`None` → CPU fallback) when the EP is
absent. Covered by
`surrogate::tests::detect_cuda_devices_returns_none_without_cuda_feature` (CPU-only
builds) and by the hardware-gated CUDA test (GPU machines).

## Acceptance criteria (mirrors issue #3313)

- [ ] **CUDA** (Linux/Windows + NVIDIA GPU): `--features ort,cuda`, run the ONNX
      surrogate inference path, confirm activation (probe + log lines above), no
      silent CPU fallback. → run §1 on hardware.
- [ ] **CoreML** (Apple Silicon macOS): `--features ort,coreml` on macOS target, same
      proof. → run §2 on hardware.
- [ ] **DirectML** (Windows): `--features ort,directml`, same proof, including the
      `MultiDeviceSessionPool` device-probing paths. → run §3 on hardware.
- [x] Session-pool device enumeration (`available_devices`) reflects real EP
      availability and degrades gracefully when an EP is absent (probe API + contract
      tests; CPU-only portion runs in CI on every `--features ort` build).

> These boxes stay **unchecked in the issue** until someone with the matching hardware
> runs the corresponding section and records the result (comment or commit). The
> `#[ignore]` tests exist precisely so that check is a single command.

## Known interactions

- **#3311** — `assets/dummy_surrogate.onnx.sha256` is missing, so fixture-loading
  tests fail closed (8 pre-existing lib failures). The hardware-gated tests here use
  the documented `FLUXION_ONNX_MODEL_SIGNATURE` env override (issue #2906) at runtime
  and do **not** pre-empt #3311; do not add the manifest file in this workstream.
- **#2920** — the silent CUDA→CPU downgrade warn fires when `FLUXION_ONNX_BACKEND=cuda`
  is requested without the `cuda` feature. That warning is the runtime companion of
  this probe's `compiled_in: false` verdict.
- **#1603** — `tests/surrogate_cuda_smoke_test.rs` (GPU-vs-CPU parity smoke) remains
  the numerical end-to-end check on CUDA hardware; it was migrated to the rc.13
  `ort::ep::CUDA` import path as part of this issue.
