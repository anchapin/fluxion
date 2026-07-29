# fluxion-fluid WASM Compatibility Status

**Created:** 2026-07-28  
**Issue:** #1998

## Executive Summary

`fluxion-fluid` **can be compiled** to `wasm32-unknown-unknown` with the current dependency set. Full WASM support is feasible for the port trait and graph modules (Issues 1.2/1.3). The solver and ECS modules face constraints due to `faer-rs` and `ort` (Burn) dependencies (see sections below).

## Dependency WASM Compatibility Matrix

| Dependency | Purpose | WASM Status | Notes |
|------------|---------|-------------|-------|
| `thiserror` | Error handling | **WASM Compatible** | `no_std` compatible, pure Rust |
| `num-traits` | Numeric trait bounds | **WASM Compatible** | `no_std` compatible, no platform code |
| `uom` | Units of measurement | **WASM Compatible** | `no_std` compatible via `f32` feature |
| `serde` | Serialization | **WASM Compatible** | Fully WASM-compatible, widely used in WASM projects |
| `petgraph` | Graph data structure | **WASM Compatible** | `no_std` compatible, but ships `std` and `alloc` features |
| `faer-rs` | Linear algebra (DAE solver) | **NOT Compatible** | Uses `rayon` internally for parallel operations; no WASM threads |
| `rayon` | Parallelism | **NOT Compatible** | Does NOT work on WASM targets (no thread support) |
| `ort` (Burn) | ONNX inference | **NOT Compatible** | Requires `burn` DNN framework; GPU/CUDA backend not WASM-compatible |

## Current fluxion-fluid Dependencies (v1.0.0)

```toml
[dependencies]
thiserror = "1.0"      # ✅ WASM Compatible
num-traits = "0.2"     # ✅ WASM Compatible
uom = { version = "0.35", features = ["f32"] }  # ✅ WASM Compatible
```

**Result:** `cargo check --target wasm32-unknown-unknown -p fluxion-fluid` **SUCCEEDS**

## Planned Dependencies (Phase 4/5)

### 1. `petgraph` — Graph Data Structure
- **Status:** `no_std` compatible via feature flags
- **Required for:** Port connection graphs, system topology
- **WASM Path:** Use `default = []` and enable only `alloc` feature for WASM
- **Action Required:** Ensure no `std` feature is enabled when targeting WASM

### 2. `faer-rs` — Linear Algebra (CTF/DAE Solver)
- **Status:** **NOT WASM Compatible**
- **Problem:** Internally uses `rayon` for parallel matrix operations
- **Required for:** State-space CTF solver, coupled DAE solver
- **Mitigation Options:**
  1. Sequential fallback using serial matrix operations
  2. WebAssembly SIMD (`wasm32-unknown-unknown` + SIMD target)
  3. Pure physics fallback without ML surrogate acceleration
- **Recommendation:** Document as "requires sequential fallback" for Phase 5 WASM target

### 3. `rayon` — Parallelism
- **Status:** **NOT Compatible with Standard WASM**
- **Problem:** No thread support in `wasm32-unknown-unknown`
- **Required for:** Population-level parallelism in `BatchOracle`, ASHRAE validation sweeps
- **WASM Path:** `wasm-bindgen-threads` with `SharedArrayBuffer` (requires special headers)
- **Current Usage in fluxion:** At population level only (verified, see `.githooks/batch-oracle-check.sh`)

### 4. `ort` / `burn` — ONNX Inference
- **Status:** **NOT WASM Compatible**
- **Problem:** Burn DNN framework has CUDA/GPU backend dependencies
- **Required for:** ONNX surrogate inference in `fluxion-ai`
- **Mitigation Options:**
  1. WebAssembly-native physics fallback when ONNX unavailable
  2. Co-processing architecture (WASM handles physics, external service handles ML)
  3. Lightweight WASM-native inference (e.g., `wasmnn`, `onxruntime-wasm`)
- **Recommendation:** WASM target should be "pure physics, no ML" for Phase 5

## Code Path Analysis: `rayon` Usage

### In `fluxion-fluid` (direct)
**None.** Current `fluxion-fluid` does not use `rayon`.

### In `fluxion` main crate (upstream consumers)
The pre-commit hook `.githooks/batch-oracle-check.sh` enforces that `rayon` is **only used at the population level** in `BatchOracle::evaluate_population`. This means:
- WASM-safe: individual zone simulation does not use rayon
- WASM-unsafe: batch/oracle evaluation at population level

**Verified:** `rayon` usage is isolated to:
- `src/ai/surrogate.rs` (population-level only)
- `src/validation/performance/parallel_executor.rs`
- `src/analysis/monte_carlo.rs`

## Recommendations for Phase 5 WASM Target

### Architecture Decision Required

**Option A: Pure Physics WASM (Recommended for Phase 5)**
- `fluxion-fluid` + `fluxion-core` compile to WASM
- Physics simulation only (no ML surrogates)
- ONNX inference handled by co-processing or external service
- Fallback: sequential iteration instead of `rayon`

**Option B: Full WASM with Threading**
- Requires `wasm32-wasip2` or `wasm-bindgen-threads` with `SharedArrayBuffer`
- Needs special server headers (COOP/COEP)
- More complex deployment
- Better performance for parallel sweeps

### Minimum Viable WASM (Phase 5.1)
1. `fluxion-fluid` (port traits, graph) — **WASM Compatible NOW**
2. `fluxion-core` (weather, assembly, multi_node) — **WASM Compatible**
3. `fluxion-behavior` (occupancy, comfort) — **Needs Review**

### Out of Scope for Phase 5
- `fluxion-ai` (ONNX surrogates) — requires co-processing architecture
- `fluxion-city` (urban radiation) — computationally heavy, better server-side
- `fluxion-grid` (electrical network) — future phase

## Verification Commands

```bash
# Verify current fluxion-fluid WASM compatibility
cargo check --target wasm32-unknown-unknown -p fluxion-fluid

# Full workspace WASM check (will fail on ort/faer/rayon dependencies)
cargo check --target wasm32-unknown-unknown -p fluxion

# Check rayon usage is isolated to population level
./.githooks/batch-oracle-check.sh
```

## References

- Issue #1980: fluxion-fluid crate creation
- Issue #1981: Core modules for testing
- Issue #1982: Additional core modules
- Issue #1991: ECS/rayon analysis
- WASM Threading: <https://github.com/rustwasm/wasm-bindgen/blob/main/API.md#cfgtarget_feature>
