# Runtime ORT execution-provider probe runbook (issue #3313)

`tests/ai_ort_ep_probe.rs` records CUDA / CoreML / DirectML execution-provider state at runtime as `EP_RUNTIME_PROBE:` log lines, closing the compile-only verification gap left by the ort rc.13 migration (#3296).
Audience: agents and operators running the EP validation, plus anyone with NVIDIA / Apple Silicon / Windows hardware who can tick a box in the hardware matrix below.
Covers: what the probe asserts, how to read its output, the runner/OS/feature matrix, and the failure modes with their remediation.
Related: `docs/ORT_EP_VALIDATION.md` (the operator-facing validation runbook and hardware-gated `#[ignore]` tests), `src/ai/surrogate.rs` (`ExecutionProviderReport::capture`), `Cargo.toml` (`ort`, `cuda`, `coreml`, `directml` features).
Scope: observability only — the probe reads EP state and never modifies the production activation path, physics, or energy balance.

## Purpose

The rc.13 migration moved `ort::execution_providers::{CUDA,CoreML,DirectML}ExecutionProvider`
to `ort::ep::{CUDA,CoreML,DirectML}` and was verified **compile-only**. Compilation proves the
API surface is correct; it proves nothing about whether an EP *activates*. ORT's classic
failure mode is the silent fallback: `with_execution_providers` returns `Ok`, the session
builds, and every node quietly runs on the CPU EP.

The probe makes that state observable and, critically, **runs everywhere**:

- On hardware-less machines (the Linux CI runner) it records absence and **passes**. The
  harness must be available on every runner so the hardware half can be executed by whoever
  owns the hardware.
- On machines that do have the EP it tightens into assertions: compiled in,
  target-appropriate, registration succeeded, and `cpu_only()` is false.

Absence is a report, not a failure.

## Running the probe

```bash
# Baseline (CPU-only, works anywhere)
cargo test -p fluxion --features ort --test ai_ort_ep_probe -- --nocapture

# With a GPU/NPU EP compiled in
cargo test -p fluxion --features ort,cuda     --test ai_ort_ep_probe -- --nocapture  # Linux/Windows + NVIDIA
cargo test -p fluxion --features ort,coreml   --test ai_ort_ep_probe -- --nocapture  # macOS
cargo test -p fluxion --features ort,directml --test ai_ort_ep_probe -- --nocapture  # Windows
```

`--nocapture` is required to see the lines on a passing run; without it libtest only prints
them for failing tests. The assertions run either way.

## Reading `EP_RUNTIME_PROBE:` lines

Per-backend lines use a stable `key=value` format:

```text
EP_RUNTIME_PROBE: cuda=active, ep=CUDAExecutionProvider, compiled_in=true, device=NVIDIA gpu#0, registration=ok, reason=CUDAExecutionProvider: ACTIVE (cuda)
EP_RUNTIME_PROBE: cuda=absent, ep=CUDAExecutionProvider, compiled_in=false, device=none, registration=skipped, reason=CUDAExecutionProvider: not compiled into this binary (backend `cuda` needs its fluxion feature)
```

| Field | Meaning |
|---|---|
| `<backend>=active\|absent` | Overall verdict. `active` ⇒ EP will really be used; `absent` ⇒ inference falls back to CPU. |
| `ep=` | ORT's canonical EP name. Note DirectML is spelled `DmlExecutionProvider`. |
| `compiled_in=` | Whether this binary can construct the EP at all (fluxion feature + target-OS gate). |
| `device=` | Hardware device ORT enumerated for this EP (`vendor type#id`), or `none`. |
| `registration=ok\|error(..)\|skipped` | Result of attaching the EP with `error_on_failure`. `skipped` ⇒ not compiled in. |
| `reason=` | Human-readable status line explaining the verdict. |

The `ep_runtime_probe_emits_full_report` test additionally prints every device ORT
enumerated plus a one-line summary that is enough to fill in the matrix below:

```text
EP_RUNTIME_PROBE: summary ort_api_version=27, devices=1, activated=cpu, cpu_only=true
```

`activated=cpu` with `cpu_only=true` is the expected result on a GPU-less runner.

## Hardware matrix

Tick a row by running the probe on that hardware and pasting the `summary` line into the
issue. `cpu_only=false` plus `<backend>=active` is the proof of activation.

| Runner / OS | Feature flags | EP under test | Expected on matching hardware | Expected without it |
|---|---|---|---|---|
| Linux CI (no GPU) | `ort` | — | n/a | `cpu=active`, `cpu_only=true`; CUDA `absent, compiled_in=false` |
| Linux + NVIDIA | `ort,cuda` | CUDA | `cuda=active, device=NVIDIA gpu#0` | `cuda=absent, device=none` (driver/toolkit missing) |
| Windows + NVIDIA | `ort,cuda` | CUDA | `cuda=active` | `cuda=absent` |
| Windows + DX12 GPU | `ort,directml` | DirectML | `directml=active, ep=DmlExecutionProvider` | `directml=absent` |
| macOS (Apple Silicon) | `ort,coreml` | CoreML | `coreml=active` | `coreml=absent` |
| macOS | `ort,cuda` | CUDA | n/a — unsupported target | `cuda=absent, …not shipped for macOS…` |
| Linux / Windows | `ort` (no EP flag) | CoreML / DirectML | n/a — unsupported target | `absent, …only available on macOS/Windows` |

For end-to-end proof that a real inference runs on the EP (not just that it registered), use
the hardware-gated `#[ignore]` tests documented in `docs/ORT_EP_VALIDATION.md`.

## Failure modes

| Symptom | Meaning | What to do |
|---|---|---|
| `absent, compiled_in=false` on hardware you have | The fluxion feature was not enabled. | Rebuild with `--features ort,cuda` (or `coreml` / `directml`). Not a bug. |
| `absent, …only available on macOS/Windows` | Target-OS gate. | Expected off-target; nothing to do. |
| `absent, compiled_in=true, device=none` | EP compiled in but ORT enumerated no hardware device — **this is the silent-CPU-fallback case**. | Check driver/runtime install (NVIDIA driver + CUDA runtime, DirectX 12, macOS version). Inference is running on CPU. |
| `absent, registration=error(...)` | The provider shared library is missing or incompatible. | Read the embedded error; usually a missing `onnxruntime_providers_*` library alongside the binary. |
| Test *fails* with `reported active but EP registration did not succeed` | Report is internally inconsistent — a genuine regression in `ExecutionProviderReport::capture()`. | Investigate `src/ai/surrogate.rs`; do not relax the assertion. |
| Test *fails* with `cpu baseline unavailable` | ORT itself is unusable in this build. | The whole `ort` feature is broken; fix before interpreting any GPU result. |
| Test *fails* with `reported inactive without any reason` | The probe returned an unexplained verdict. | Bug in `probe_ep`; absence must always carry a reason. |

A GPU-less machine reporting everything `absent` is **not** a failure — that is the harness
working as designed.

## Scope guard

This probe is read-only. Do not "fix" a red probe by changing the EP activation path to make
the line say `active`; the line is a measurement, and a wrong measurement is worse than an
absent one.
