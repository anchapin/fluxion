# Fluxion Troubleshooting Guide

Developer-facing build, CI, and physics-constant pitfalls. Use this
when a CI gate fails, a build won't compile, or a physics constant
shows up in the wrong place. For user-facing caveats (peak loads,
EUI semantics, simple_config.json, surrogate fallback, etc.) see
[docs/FAQ.md](FAQ.md) instead. Each entry links to the authoritative
source (script, test, or doc).

## Build & Toolchain

### `cargo build` fails to link `fluxion-rest` / `fluxion-mcp`

These crates pull in extra features:

```bash
cargo build -p fluxion-mcp                          # forces multi-zone + fluid + toon
cargo build --bin fluxion-rest                       # needs `wiring-tracing` off for fast builds
```

`fluxion-mcp` depends on `fluxion` with `default-features = false` and
enables `multi-zone` via its own default feature (#2540); it
unconditionally pulls `fluxion-fluid` + `fluxion-toon`. If you
see unresolved-import errors for `MultiZoneThermalModel` or
`ThermalPort`, you forgot the `multi-zone` / `fluid` feature.

### PyO3 linker SIGSEGV when building Python bindings

Set the two env vars the bindings CI uses:

```bash
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
export RUST_MIN_STACK=33554432
```

Python 3.10+ is required for `maturin develop`.

### `cargo fmt --check` fails on `?` or `async` syntax

`.rustfmt.toml` sets `edition = "2021"`. Without it rustfmt falls back
to 2015 and breaks on `?` / `async`. Confirm `rust-toolchain.toml`
pins **stable** + rustfmt + clippy. Stable rustfmt has **no `exclude`**:
auto-generated fixture data must use `#[rustfmt::skip]` per-item (see
`tests/per_tilt_per_azimuth_fixture_data.rs`).

### Clippy CI runs out of memory

`CARGO_BUILD_JOBS=1` is set by Clippy CI to keep peak RSS low. The
canonical invocation is exactly:

```bash
cargo clippy --lib -- -D warnings
```

Mutation testing (`cargo mutants`) is heavier: the full suite needs
**32 GB+ RAM**; `.cargo/mutants.toml` excludes combinatorial physics
files and all of `src/validation/**`. The suite skips gracefully
below 16 GB (#2130).

## CI Gates

### `Energy Conservation` check fails

Greps test output for the literal string `"violated energy conservation"`.
If you see it, the bug is real physics — not a CI hiccup. Reproduce
with `cargo test --test zone_balance_eplus_isolation` and follow
`RULES.md` (energy balance must close; no parameter tuning).

### `Known Issues Stale Check` (#1723) fails

`docs/KNOWN_ISSUES.md` has a top-level `*Last Updated: YYYY-MM-DD*`
line at line 10. The gate fails if it is older than 60 days. Update
the line — and the prose around it — when you refresh the file. The
check skips (passes) if the file is absent.

### `Ashrae Cases Cycle Check` (#1441) / `Physics-Sim-Cycle-Check` (#2463) fail

These guard the crate-split cycle rules:

- `fluxion-core/src/**/*.rs` must **not** import `crate::sim_*` /
  `crate::physics_*` / `crate::ai_*` / `crate::validation_*`.
- No new `use crate::sim::*` under `src/physics/**` or
  `use crate::physics::*` under `src/sim/construction.rs` /
  `src/sim/per_surface_conduction.rs`.

Fix the import, don't silence the guard.

### `Architecture Drift Detection` fails

`ARCHITECTURE.md` and `CODEBASE_MAP.md` are source-of-truth for module
boundaries. Run `python3 scripts/check_architecture_drift.py` locally;
either update the doc to reflect reality **or** fix the code. Do **not**
modify physics code without checking `ARCHITECTURE.md` first.

### `Code Coverage Gate` (#1932) fails

The gate is a ratchet; baseline lives in
`validation/coverage_baseline.json`. A `0.0` threshold means that path
is unenforced. Run `python3 scripts/coverage_critical_paths.py` for the
per-path list. Do not regress covered paths.

## Physics Constants

### The `29.3 W/m²K` exterior film coefficient appears in a computation path

This is a **regression**. The only correct value is
**`EXTERIOR_FILM_COEFF = 18.3 W/m²K`** (ASHRAE 140 v2023, vertical
surfaces, ~3.4 m/s wind), defined in
`src/physics/constants/thermal/ashrae_140/v2023.rs`. The legacy `29.3
W/m²K` (6.7 m/s) must **not** appear in any computation path. Guard:
`tests/regression_exterior_film_unification.rs`.

### ASHRAE 140 material constants (`HW_CONCRETE_K`, `FOAM_BOARD_K`, `GYPSUM_K`, …)

These are **inlined at `fluxion_core::assembly` call sites** — the old
`crate::physics::constants::thermal::ashrae_140::materials` import path
is gone. If you see an unresolved import for it, inline the constant
at the call site.

## ONNX / AI Surrogate

### `cargo build` errors with unresolved `ort` symbols

The ONNX runtime is opt-in. Build with `--features ort` (alias `onnx`)
for any AI-surrogate path. Default builds skip the runtime and the
`SurrogateManager` falls back to a deterministic analytical mock
(`deterministic_analytical_loads`).

### GPU inference silently falls back to CPU

`FLUXION_ONNX_BACKEND=cuda` is a no-op when the `cuda` cargo feature
is not built; the manager auto-downgrades to `cpu` at runtime. Set
`FLUXION_GPU=0` or `FLUXION_GPU=false` to force CPU. CUDA smoke test:
`cargo test --features cuda --test surrogate_cuda_smoke` (skips on
CPU-only hosts).

## Performance

### `BatchOracle::evaluate_population` is slow or hangs

The population-level rayon `par_iter()` is the only allowed
parallelism layer. **Nested parallelism in the inner loop causes
thread-pool exhaustion.** Pre-commit hook
`.githooks/batch-oracle-check.sh` enforces this on `lib.rs`. If you
added `par_iter()` inside `evaluate_population`, remove it.

### Heavy Linux CI jobs fail with credential-lock / git-ref-lock errors

Run `./scripts/disk-space-check.sh` first. Minimum **10 GB free**
(50 GB recommended) — exhaustion has caused credential-lock,
PR-creation, and git-ref-lock failures.

## Debugging Physics

### Enable `eprintln!` in physics hot loops

Build with `--features debug-physics` (#1967). The feature gates every
`eprintln!` in physics hot loops so default builds stay quiet.

### Heap profiling

Build with `--features dhat` (#2384) to enable the `dhat` profiler.

### Verify determinism locally

```bash
RUSTFLAGS="-C opt-level=3 -C debug-assertions=no" \
  cargo test --test case_900_determinism --release -- --nocapture
```

The expected canonical hash for the three-OS matrix is published in
the `Determinism Check (ubuntu-latest)` step summary on `main`. See
[FAQ §8](FAQ.md) for the full gate description.

## Environment Variables Quick Reference

| Variable | Effect |
|----------|--------|
| `FLUXION_REST_BIND` / `FLUXION_REST_PORT` | `fluxion-rest` bind (default `0.0.0.0:8080`) |
| `FLUXION_ONNX_MODEL` | Explicit ONNX model path (default `models/surrogate_zone_thermal.onnx`) |
| `FLUXION_ONNX_BACKEND` | `cpu` \| `cuda` \| `coreml` \| `directml` \| `openvino` |
| `FLUXION_GPU` | `0`/`false` to force CPU inference |
| `DWAVE_API_TOKEN` | Required at runtime for the `dwave` feature |
| `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` + `RUST_MIN_STACK=33554432` | Avoid PyO3 linker SIGSEGV |
| `CARGO_BUILD_JOBS=1` | Keep Clippy CI peak RSS low |
| `LOOM=1` | Enable `loom` concurrency tests (~32 GB) |

See [`AGENTS.md` §Environment Variables](../AGENTS.md) for the
canonical list.

## Getting Help

- GitHub Issues: <https://github.com/anchapin/fluxion/issues>
- Documentation: <https://fluxion.readthedocs.io>
