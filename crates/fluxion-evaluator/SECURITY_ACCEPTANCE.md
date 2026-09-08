# `fluxion-evaluator` dynamic-loading security acceptance

> **Status (Issue #3554):** Acceptance-criteria contract. The
> `libloading` follow-up PR is **out of scope** for this issue. This
> document is the security gate that follow-up must satisfy before
> `load_candidate` can return any variant other than
> `DynamicLoadError::NotImplementedInThisBuild`.

This file is the single source of truth for the security contract that
must hold for `crates/fluxion-evaluator/src/dynamic.rs` before the
feature is shipped. The drift gate
`scripts/check_evaluator_dynamic_secure.py` enforces it
mechanically, and the unit test
`crates/fluxion-evaluator/tests/security_acceptance.rs` pins the
contract in CI.

## Threat model (binding)

A candidate cdylib loaded by `dynamic.rs` is **fully untrusted code**
that runs **in-process** inside the evaluator harness. The current
harness only ever loads candidates in an *isolated subprocess*; the
dynamic-load path closes that gap and opens a new attack surface. The
candidate can:

1. Call back into the host via the resolved symbol table
   (`fluxion_kernel_evaluate`'s `output: *mut u8` argument has no
   provenance check, no length pre-validation, and no destination
   isolation).
2. Trigger `SIGSEGV` / `SIGILL` / `SIGABRT` and bring the harness
   process down — there is no panic-or-signal boundary around the FFI
   call today.
3. Read host process memory via the resolved `Library` handle (any
   global symbol the host links in is visible from the candidate if
   `RTLD_GLOBAL` semantics leak through).
4. Misbehave in lazy-binding order: if loaded with `RTLD_LAZY`, the
   harness never sees a symbol-resolution fault until the kernel is
   actually dispatched, which defers failure into the timing-critical
   hot path.

Each numbered mitigation below corresponds to a numbered threat above.

## Acceptance criteria

### (a) Signature verification — SHA-256 manifest

The follow-up MUST mirror `verify_onnx_signature` at
`src/ai/surrogate.rs:3348`. The contract:

- A SHA-256 manifest file MUST live next to the candidate cdylib, at
  `<candidate>.sha256` (consistent with the ONNX path).
- The manifest format MUST be a `sha256sum`-compatible text manifest:
  ```text
  # comments start with '#'; blank lines are ignored
  <64-hex-digest>  <relative-path>
  ```
- An optional explicit-digest override MUST be honored via an
  environment variable (parallels `FLUXION_ONNX_MODEL_SIGNATURE`).
  For the dynamic loader this variable is
  `FLUXION_EVAL_CANDIDATE_SIGNATURE`.
- A mismatch (or missing manifest, or invalid hex) MUST be a hard
  error returning `DynamicLoadError::CandidateError` (or a new
  dedicated `SignatureMismatch` variant — see implementation note).
  Failure mode is **fail-closed**: no manifest = no load.

> Code-shape sketch (no implementation):
> ```rust,ignore
> let expected = read_manifest_hash_or_env(path, "FLUXION_EVAL_CANDIDATE_SIGNATURE")?;
> let actual = compute_file_sha256(path)?;
> validate_hash(&expected, &actual).map_err(...)?;
> ```

### (b) Sandbox cross-reference — `crates/fluxion-evaluator/src/sandbox.rs`

The current `sandbox.rs` defends the **recompilation** path (subprocess
isolation, wall-clock cap, env-isolation, best-effort memory cap).
The dynamic-load path is in-process and inherits none of those
guarantees. The follow-up MUST identify and close every gap. Today,
the surface of `sandbox.rs` provides:

- `SandboxConfig` — wall-clock cap, network isolation flag, candidate
  dir, best-effort memory cap.
- `SandboxEnforcer` — `enforce_for_command` mutates a
  `std::process::Command`; **`enforce_for_command` is useless against
  in-process FFI** because the candidate never becomes a subprocess.
- `determinism_digest` — the only existing SHA-256 in the crate;
  useful as a template for (a).

The follow-up MUST add a new function (suggested:
`SandboxEnforcer::enforce_for_dynamic_load(&self, candidate: &Path)
-> Result<DynamicSandboxReceipt, EvaluatorError>`) that returns a
receipt object holding the things the loader needs to verify before
each call (memory-cap advisory, wall-clock advisory, candidate dir,
process-leak detector), and the loader MUST consult it before every
dispatch — not only at load time.

> Implementation note: the in-process path cannot enforce `RLIMIT_AS`
> on the candidate's stack/heap without `libc` (which the project is
> at zero headroom on). Document the gap honestly as
> `SandboxViolation::InProcessCandidate` rather than over-promising.

### (c) Panic and signal capture

The follow-up MUST wrap every FFI call (the `Library::get` resolution
+ the `fluxion_kernel_evaluate` dispatch) in:

1. `std::panic::catch_unwind` — Rust panics that propagate across
   `extern "C"` boundaries are undefined behavior. The wrapper MUST
   map the recovered payload to
   `DynamicLoadError::CandidateError` (or a new `PanicInCandidate`
   variant) and MUST NOT re-panic.
2. Signal handler trampoline for `SIGILL` / `SIGSEGV` / `SIGABRT` —
   `panic::catch_unwind` does **not** catch these. The follow-up MUST
   install a thread-local handler that converts the signal into a
   normal `Err` return and restores the previous handler before the
   FFI returns. Suggested crate: a tiny `signal-hook` adapter, gated
   behind the `dynamic` feature so the rest of the crate is
   unaffected.

   **Edge case:** `SIGILL`/`SIGSEGV` can also indicate a real bug in
   the harness rather than candidate malice. The handler MUST log the
   candidate path + dispatch index and then re-raise after the
   handler returns so post-mortem tooling can still see the fault.

> Code-shape sketch (no implementation):
> ```rust,ignore
> let prev = signal::swap_handler(Signal::SIGSEGV, candidate_handler)?;
> let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| unsafe {
>     let f: extern "C" fn(...) -> i32 = std::mem::transmute(symbol.as_ptr());
>     f(input_ptr, input_len, output_ptr, &mut output_cap)
> }));
> signal::restore_handler(Signal::SIGSEGV, prev)?;
> match outcome {
>     Ok(0) => Ok(()),
>     Ok(status) => Err(DynamicLoadError::CandidateError(status)),
>     Err(payload) => Err(DynamicLoadError::CandidateError(-1)), // panic sentinel
> }
> ```

### (d) `RTLD` flags — reject lazy / global bindings

The follow-up MUST only accept candidates loaded with the equivalent
of `RTLD_NOW | RTLD_LOCAL`:

- **`RTLD_NOW`** — every symbol is resolved at `Library::new` time,
  not lazily at dispatch time. `libloading::Library::new` defaults to
  `RTLD_LAZY`; the follow-up MUST override this on every platform
  that supports it (Linux/macOS: pass `RTLD_NOW`; Windows: the
  analogue is `LOAD_WITH_ALTERED_SEARCH_PATH` plus
  `DONT_RESOLVE_DLL_REFERENCES = 0`).
- **`RTLD_LOCAL`** — the candidate's symbols MUST NOT be added to
  the host's global symbol table. This prevents the candidate from
  hijacking symbols the harness later `dlsym`s (e.g. `malloc`,
  `fopen`). On Windows the analogue is load-library-private.

The follow-up MUST add a dedicated
`DynamicLoadError::InsecureLoadFlags` variant (or reuse
`CandidateError` with a documented sentinel) and the gate below MUST
be enforced at load time, not deferred to dispatch.

> Code-shape sketch (no implementation):
> ```rust,ignore
> let flags = RTLD_NOW | RTLD_LOCAL;
> let lib = unsafe { libloading::os::unix::Library::open(Some(path), flags)? };
> if (lib.load_flags() & RTLD_LAZY) != 0 {
>     return Err(DynamicLoadError::InsecureLoadFlags);
> }
> if (lib.load_flags() & RTLD_GLOBAL) != 0 {
>     return Err(DynamicLoadError::InsecureLoadFlags);
> }
> ```

### (e) CI drift gate

Analogous to `scripts/check_audit_config_unique.py`. The follow-up
MUST add `scripts/check_evaluator_dynamic_secure.py` (landed in this
issue) and wire it into `.github/workflows/security.yml` (also
landed in this issue). The gate fails if:

- `crates/fluxion-evaluator/src/dynamic.rs` produces any error
  variant other than `DynamicLoadError::NotImplementedInThisBuild`
  in a return path (i.e. the loader has been implemented), **or**
- The `dynamic` feature in `crates/fluxion-evaluator/Cargo.toml` has
  any non-empty `dependencies` line (i.e. `libloading` has been
  pulled in),

  AND one or more of the `// SECURITY-ACCEPTANCE: satisfied`
  markers is missing from
  `crates/fluxion-evaluator/src/dynamic.rs` (one per criterion (a)
  through (d)).

The marker format MUST be exactly:

```rust
// SECURITY-ACCEPTANCE: satisfied — (a) signature verification mirrors verify_onnx_signature
```

with one such marker per criterion. The gate's parsing is strict
about the prefix `// SECURITY-ACCEPTANCE: satisfied` followed by the
criterion letter in `(a)`/`(b)`/`(c)`/`(d)`.

## Summary

| Criterion | Where it lives | How it's verified |
|-----------|---------------|-------------------|
| (a) Signature | `dynamic.rs::load_candidate` | `<candidate>.sha256` read + SHA-256 compare |
| (b) Sandbox | new function on `SandboxEnforcer` | consulted on every dispatch, not only at load |
| (c) Panic/signal | `dynamic.rs::evaluate_dynamic` | `catch_unwind` + signal trampoline |
| (d) RTLD flags | `dynamic.rs::load_candidate` | rejects `RTLD_LAZY` / `RTLD_GLOBAL` |
| (e) Drift gate | `scripts/check_evaluator_dynamic_secure.py` | CI job in `.github/workflows/security.yml` |

A follow-up PR that lands without all four markers in `dynamic.rs`
and the matching implementations in `sandbox.rs` will fail the
`evaluator-dynamic-secure` CI job and cannot merge.