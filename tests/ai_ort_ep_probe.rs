//! Runtime execution-provider probe harness (issue #3313).
//!
//! ## Why this file exists
//!
//! The ort 2.0.0-rc.13 migration (#3296) moved the execution-provider API
//! from `ort::execution_providers::{CUDA,CoreML,DirectML}ExecutionProvider`
//! to `ort::ep::{CUDA,CoreML,DirectML}`, and was verified **compile-only**:
//! no GPU / Apple Silicon / Windows hardware was available, so nobody ever
//! observed whether an EP *activates* at runtime or silently falls back to
//! the CPU execution provider.
//!
//! This test binary is the portable half of that validation. It runs
//! everywhere — including the GPU-less Linux CI runner — and **records** EP
//! state as structured `EP_RUNTIME_PROBE:` lines instead of demanding
//! hardware. On a machine that *does* have the hardware, the very same test
//! tightens into an assertion that the EP really registered.
//!
//! ## Contract
//!
//! - EP absent  ⇒ emit `EP_RUNTIME_PROBE: <backend>=absent, reason=…` and PASS.
//! - EP present ⇒ emit `EP_RUNTIME_PROBE: <backend>=active, device=…` and
//!   assert the provider is compiled in, target-appropriate, and that
//!   registration succeeded (no silent CPU fallback).
//!
//! Absence is a *report*, not a failure: the harness must be available
//! everywhere so the hardware-dependent half can be executed by whoever
//! owns the hardware. See `docs/agents/runtime-ort-ep-probe-runbook.md`.
//!
//! ## Scope guard
//!
//! Read-only observability. This file does not modify — and must never be
//! used to modify — the production EP activation path in
//! `src/ai/surrogate.rs`. It only reads `ExecutionProviderReport::capture()`.
//!
//! ## Running
//!
//! ```bash
//! cargo test -p fluxion --features ort --test ai_ort_ep_probe -- --nocapture
//! cargo test -p fluxion --features ort,cuda --test ai_ort_ep_probe -- --nocapture
//! ```
//!
//! Without `--nocapture` the probe lines are captured by libtest and only
//! shown for failing tests; the assertions still run.

#![cfg(feature = "ort")]

use fluxion::ai::surrogate::{EpProbeOutcome, ExecutionProviderReport, InferenceBackend};

/// Prefix every consumer (CI log scraper, runbook reader, operator) greps
/// for. Kept as a single constant so the format cannot drift between tests.
const PROBE_PREFIX: &str = "EP_RUNTIME_PROBE";

/// Emit one structured probe line for `outcome`.
///
/// Format (stable — parsed by humans and `grep` alike):
///
/// ```text
/// EP_RUNTIME_PROBE: cuda=active, ep=CUDAExecutionProvider, device=NVIDIA gpu#0, registration=ok
/// EP_RUNTIME_PROBE: cuda=absent, ep=CUDAExecutionProvider, device=none, registration=skipped, reason=…
/// ```
fn emit(report: &ExecutionProviderReport, outcome: &EpProbeOutcome) {
    let state = if outcome.activated {
        "active"
    } else {
        "absent"
    };
    let device = report
        .devices
        .iter()
        .find(|d| d.ep_name == outcome.ep_name)
        .map(|d| {
            format!(
                "{} {}#{}",
                d.hardware_vendor.as_deref().unwrap_or("<unknown-vendor>"),
                d.hardware_type,
                d.device_id
                    .map(|id| id.to_string())
                    .unwrap_or_else(|| "?".to_string())
            )
        })
        .unwrap_or_else(|| "none".to_string());
    let registration = match &outcome.registration {
        None => "skipped".to_string(),
        Some(Ok(())) => "ok".to_string(),
        Some(Err(e)) => format!("error({e})"),
    };

    let line = format!(
        "{PROBE_PREFIX}: {}={}, ep={}, compiled_in={}, device={}, registration={}, reason={}",
        outcome.backend.as_str(),
        state,
        outcome.ep_name,
        outcome.compiled_in,
        device,
        registration,
        outcome.status_line(),
    );

    // `println!` keeps the harness dependency-free and visible under
    // `--nocapture`; `tracing::info!` carries the same line into any
    // subscriber a caller has installed (e.g. a CI JSON log collector).
    println!("{line}");
    tracing::info!(target: "fluxion::ai::ep_probe", "{line}");
}

/// Shared probe body: emit the line, then assert only what is provable on
/// the *current* machine.
///
/// - `activated == false` — hardware/feature absent. Nothing to assert
///   beyond internal consistency; the recorded line is the deliverable.
/// - `activated == true` — the EP is really there, so hold it to the
///   no-silent-fallback contract.
fn probe_backend(backend: InferenceBackend) {
    let report = ExecutionProviderReport::capture();
    let Some(outcome) = report.probe(backend) else {
        panic!(
            "{PROBE_PREFIX}: {}=unprobed — ExecutionProviderReport::capture() \
             must emit one probe per backend (issue #3313)",
            backend.as_str()
        );
    };
    emit(&report, outcome);

    if outcome.activated {
        // Hardware path. Everything below is a fact about a machine that
        // actually has the EP, so it is safe to assert.
        assert!(
            outcome.compiled_in,
            "{PROBE_PREFIX}: {} reported active but is not compiled into this binary",
            backend.as_str()
        );
        assert!(
            outcome.unsupported_on_target.is_none(),
            "{PROBE_PREFIX}: {} reported active on a target that cannot support it: {:?}",
            backend.as_str(),
            outcome.unsupported_on_target
        );
        assert!(
            matches!(outcome.registration, Some(Ok(()))),
            "{PROBE_PREFIX}: {} reported active but EP registration did not succeed: {:?} \
             — this is the silent-CPU-fallback failure mode (#3313)",
            backend.as_str(),
            outcome.registration
        );
        assert!(
            !report.cpu_only(),
            "{PROBE_PREFIX}: {} reported active yet the report claims cpu_only",
            backend.as_str()
        );
    } else {
        // Hardware-less path (the Linux CI runner). Record and pass.
        assert!(
            outcome.unsupported_on_target.is_some()
                || !outcome.compiled_in
                || !outcome.environment_device_present
                || matches!(outcome.registration, Some(Err(_))),
            "{PROBE_PREFIX}: {} reported inactive without any reason — the probe \
             must always explain absence (unsupported target, not compiled in, \
             no device enumerated, or failed registration)",
            backend.as_str()
        );
    }
}

/// CUDA (Linux/Windows + NVIDIA, `--features ort,cuda`).
///
/// Passes on CPU-only machines by recording `cuda=absent`.
#[test]
fn ep_runtime_probe_cuda() {
    probe_backend(InferenceBackend::CUDA);
}

/// CoreML (Apple Silicon macOS, `--features ort,coreml`).
///
/// Off macOS this always records `coreml=absent, …unsupported target…`.
#[test]
fn ep_runtime_probe_coreml() {
    probe_backend(InferenceBackend::CoreML);
}

/// DirectML (Windows + DirectX 12 GPU, `--features ort,directml`).
///
/// Off Windows this always records `directml=absent, …unsupported target…`.
#[test]
fn ep_runtime_probe_directml() {
    probe_backend(InferenceBackend::DirectML);
}

/// The CPU execution provider is the fallback of last resort: if it is not
/// usable, no inference path works at all, so this one *is* assertable
/// everywhere.
#[test]
fn ep_runtime_probe_cpu_baseline_always_available() {
    let report = ExecutionProviderReport::capture();
    let cpu = report
        .probe(InferenceBackend::CPU)
        .expect("CPU baseline probe must always be present");
    emit(&report, cpu);

    assert!(
        cpu.compiled_in && cpu.activated,
        "{PROBE_PREFIX}: cpu baseline unavailable — ORT is unusable in this build: {}",
        cpu.status_line()
    );
}

/// Full-report snapshot: emits every status line plus a machine-readable
/// summary, so a single `--nocapture` run of this binary is enough to fill
/// in the hardware matrix in the runbook.
#[test]
fn ep_runtime_probe_emits_full_report() {
    let report = ExecutionProviderReport::capture();

    for line in report.status_lines() {
        println!("{PROBE_PREFIX}: {line}");
    }

    let activated = report
        .activated_backends()
        .iter()
        .map(|b| b.as_str())
        .collect::<Vec<_>>()
        .join("+");
    println!(
        "{PROBE_PREFIX}: summary ort_api_version={}, devices={}, activated={}, cpu_only={}",
        report.ort_api_version,
        report.devices.len(),
        if activated.is_empty() {
            "none"
        } else {
            &activated
        },
        report.cpu_only(),
    );

    // The report must always describe every backend the migration touched,
    // otherwise the runbook's hardware matrix has a blind spot.
    for backend in [
        InferenceBackend::CPU,
        InferenceBackend::CUDA,
        InferenceBackend::CoreML,
        InferenceBackend::DirectML,
    ] {
        assert!(
            report.probe(backend).is_some(),
            "{PROBE_PREFIX}: missing probe for {} — every EP touched by #3296 must be reported",
            backend.as_str()
        );
    }

    // `cpu_only()` is the operator-facing verdict; it must agree with the
    // per-probe activation flags or the runbook conclusions are unsound.
    let gpu_active = report
        .activated_backends()
        .iter()
        .any(|b| *b != InferenceBackend::CPU);
    assert_eq!(
        report.cpu_only(),
        !gpu_active,
        "{PROBE_PREFIX}: cpu_only() disagrees with the per-probe activation flags"
    );
}
