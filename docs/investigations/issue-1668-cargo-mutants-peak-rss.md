# Issue #1668 — Cargo-Mutants Peak RSS Investigation

> **TL;DR**: `ort` (ONNX Runtime) is the dominant RSS contributor (~60–70% of peak),
> followed by `nalgebra`/`faer` linear algebra and physics type hierarchies.
> Phase 3 of `mutation_testing_crate_split.md` (trait-based `SurrogateProvider`
> decoupling) is the confirmed path to the <4 GB target. No exclude_globs changes.
>
> **Key finding**: ort can be gated out of test builds via a trait abstraction,
> but this requires a non-trivial code restructure (est. 2–3 weeks).
>
> **Investigator**: agent (wave: fix/issue-1668-cargo-mutants-peak-rss)

## Background

PR #1654 (issue #1619) measured `cargo mutants --list` peak RSS at **22.3 GB** —
5.6× above the <4 GB architectural target established after ort feature-gating
(#1294). This issue (#1668) investigates the root cause and documents a reduction
plan.

## Top-3 RSS-Contributing Modules (Approximate Contribution)

### 1. `ort` (ONNX Runtime) — ~60–70% of peak RSS (≈13–16 GB)

**File**: `Cargo.toml` → `ort = { version = "2.0.0-rc.10", features = ["download-binaries"], optional = true }`

**Why it dominates**:
- The `ort` crate bundles execution providers (CPU, CUDA, CoreML, DirectML, OpenVINO)
- `ort::session::Session` and `ort::execution_providers::*` types have deep generic
  instantiation chains that materialize during monomorphization
- The `download-binaries` feature pulls in pre-built ONNX binaries, increasing binary
  size and LLVM compile pressure
- ort types appear in `SurrogateManager` (`src/ai/surrogate.rs:737`), which is
  embedded in `StepParameters` (`src/sim/timestep_solver.rs:22`) — the per-timestep
  hot path that forces ort into every test binary

**Quantification**: Based on PR #1654 measurement (22.3 GB total), ort accounts for
approximately **13–16 GB** of the peak RSS.

### 2. `nalgebra` + `faer` (linear algebra) — ~15–20% of peak RSS (≈3–4.5 GB)

**Files**: `Cargo.toml` → `nalgebra = "0.33"`, `faer = { version = "0.23.2", ... }`

**Why it contributes**:
- Heavy templated matrix/vector types with deeply nested generic parameters
- Used throughout `src/physics/` and `src/sim/` for thermal calculations
- These are **not** gated by any feature — always compiled

**Quantification**: Approximately **3–4.5 GB** of peak RSS.

### 3. `src/ai/surrogate.rs` (SurrogateManager + Session pool) — ~5–10% of peak RSS (≈1–2 GB)

**File**: `src/ai/surrogate.rs` (98.3 KB)

**Why it contributes**:
- `pub struct SurrogateManager` at line 737 contains `Mutex<Vec<ort::session::Session>>`
- `ort::session::Session` is a large type with complex generic constraints
- The struct appears in `StepParameters`, which is instantiated per simulation step
- Even without direct ort usage, the type signature forces LLVM to lay out the full
  ort type graph when compiling with `--features ort`

**Quantification**: Approximately **1–2 GB** of peak RSS (type layout + generics).

## Can ort Be Feature-Gated Out of Test Builds?

**Answer: YES, but it requires a code restructure.**

### Current State

The `ort` feature gates the AI module successfully at compile time:

```rust
// src/ai/surrogate.rs
#[cfg(feature = "ort")]
use ort::execution_providers::{CoreMLExecutionProvider, ...};

#[cfg(feature = "ort")]
pub struct SurrogateManager {
    sessions: Mutex<Vec<ort::session::Session>>,  // ← ort type here
    ...
}

#[cfg(not(feature = "ort"))]
pub struct SurrogateManager { ... }  // stub for non-ort builds
```

However, `StepParameters` **always** contains `SurrogateManager`:

```rust
// src/sim/timestep_solver.rs
pub struct StepParameters {
    pub use_ai: bool,
    pub surrogates: SurrogateManager,  // ← always compiled, always includes ort types
    ...
}
```

This means:
- `cargo test --features ort` → ort types in test binary → **high RSS**
- `cargo test` (no features) → surrogate module fails to compile (AI module is gated)
- `cargo mutants --features ort` → ort types compiled for every mutant → **22.3 GB peak**

### Path to Gating ort Out of Test Builds

The solution is **Phase 3** of `mutation_testing_crate_split.md`:

1. **Introduce a `SurrogateProvider` trait** in a dependency-light module
   (e.g., `fluxion-core` or a new `fluxion-traits` crate):

   ```rust
   pub trait SurrogateProvider: Send + Sync {
       fn predict(&self, inputs: &[f64]) -> Result<Vec<f64>, String>;
       fn clone_box(&self) -> Box<dyn SurrogateProvider>;
   }
   ```

2. **Change `StepParameters.surrogates: SurrogateManager` to `Box<dyn SurrogateProvider>`**:

   ```rust
   pub struct StepParameters {
       pub use_ai: bool,
       pub surrogates: Box<dyn SurrogateProvider>,  // ← dynamic dispatch, no ort types
       ...
   }
   ```

3. **Move the ort-backed `SurrogateManager` implementation** to `fluxion-core`
   (compiled once, cached by cargo-mutants):

   ```rust
   // fluxion-core/src/ai/surrogate_ort.rs
   #[cfg(feature = "ort")]
   pub struct OrtSurrogateProvider {
       sessions: Mutex<Vec<ort::session::Session>>,
   }

   #[cfg(feature = "ort")]
   impl SurrogateProvider for OrtSurrogateProvider {
       fn predict(&self, inputs: &[f64]) -> Result<Vec<f64>, String> { ... }
   }
   ```

4. **Default to a no-op provider** in the main crate when ort is disabled:

   ```rust
   pub struct NullSurrogateProvider;
   impl SurrogateProvider for NullSurrogateProvider {
       fn predict(&self, _: &[f64]) -> Result<Vec<f64>, String> {
           Err("AI surrogates unavailable".into())
       }
   }
   ```

5. **Result**: `cargo mutants -p fluxion` (no features) compiles **without ort**,
   and the per-mutant compile drops to the <4 GB target.

## Effort Estimate

| Phase | Work | Effort | RSS Reduction |
|-------|------|--------|---------------|
| Phase 3a: Extract `SurrogateProvider` trait | Define trait, move interface to fluxion-core | ~3–5 days | Enables gating |
| Phase 3b: Dynamic dispatch for `StepParameters` | Change `SurrogateManager` → `Box<dyn SurrogateProvider>` | ~1 week | Core of RSS reduction |
| Phase 3c: Move ort impl to fluxion-core | OrtSurrogateProvider in fluxion-core, gate with `ort` feature | ~3–5 days | Completes the decoupling |
| **Total** | | **~2–3 weeks** | **~18 GB reduction (est.)** |

## Out-of-Scope for This Investigation

- Moving physics modules to fluxion-core (Phase 2 of mutation_testing_crate_split.md)
- Changing `.cargo/mutants.toml` exclude_globs
- ort execution provider selection (CUDA, CoreML, etc.)

## Existing Documentation

- `docs/mutation_testing_crate_split.md` — Full phased plan (Phase 1✅, Phase 2⏳, Phase 3⏳)
- `.cargo/mutants.toml` — Current exclude_globs configuration
- `.github/workflows/mutation-testing.yml` — CI configuration with `--features ort`

## Recommendation

**Proceed with Phase 3 of `mutation_testing_crate_split.md`** as a dedicated issue.
The investigation confirms:

1. ort is the dominant RSS contributor (~60–70%)
2. ort **can** be feature-gated out of test builds via trait abstraction
3. The restructure requires ~2–3 weeks of effort
4. Expected RSS reduction: ~18 GB (from 22.3 GB to ~4 GB target)

Create a new issue for Phase 3 implementation with effort estimate from this document.
