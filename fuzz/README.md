# fluxion fuzz targets

libFuzzer fuzz targets for the fluxion building-energy-modeling engine, standing
up the coverage called for in [issue #2537]. These complement the
`loom::fuzz` concurrency tests (`tests/concurrency/loom_concurrency_tests.rs`)
with randomised *input* coverage of the FFI surface and the zone-balance
physics solver.

[issue #2537]: https://github.com/anchapin/fluxion/issues/2537

## Layout

```
fuzz/
  Cargo.toml              # standalone cargo workspace (NOT a member of the root workspace)
  README.md               # this file
  fuzz_targets/
    ffi_batch_oracle.rs   # BatchOracle::evaluate_population  (PyO3 / NAPI FFI surface)
    zone_balance_solver.rs# ThermalModel::step_physics + solve_timesteps  (zone-balance solver)
    epw_parser.rs         # EpwWeatherSource::from_file  (EPW weather-file parser)
```

The fuzz crate depends on the production `fluxion` crate via a `path` dependency
and is declared as its **own** workspace root (`[workspace]` in `fuzz/Cargo.toml`)
so that a bare `cargo build` / `cargo check` at the repo root never pulls in
the nightly-only `libfuzzer-sys` dependency.

## Targets

| Target | Public API exercised | Invariant asserted |
|--------|---------------------|--------------------|
| `ffi_batch_oracle` | `fluxion::BatchOracle::evaluate_population` (backs the PyO3 `BatchOracle.evaluate_population_py` and the NAPI `BatchOracle.evaluate_population`) | Never panics on NaN/Inf/extreme U-values/out-of-range/swapped setpoints; returns one finite-or-NaN EUI per candidate. |
| `zone_balance_solver` | `ThermalModel::step_physics`, `ThermalModel::solve_timesteps` (the 5R1C/6R2C/8R3C/9R4C thermal network behind `Model.simulate`) | Never panics; per-step energy is finite &ge; 0; zone temperatures never become NaN/Inf. |
| `epw_parser` | `EpwWeatherSource::from_file` | Never panics on arbitrary bytes; malformed input returns `Err(WeatherError)`. |

## Requirements

Fuzzing requires the **nightly** toolchain and `cargo-fuzz` (which wraps
`libfuzzer-sys` + `-Csanitizer=address`):

```bash
rustup toolchain install nightly
cargo +nightly install cargo-fuzz --version "^0.11"
```

The stable toolchain is enough to **compile-check** the targets (see below) but
cannot link the libFuzzer runtime.

## Running the targets

```bash
# Run a single target (defaults to ~indefinite; Ctrl-C to stop). libFuzzer
# writes any crash-triggering inputs to fuzz/artifacts/<target>/.
cargo +nightly fuzz run ffi_batch_oracle
cargo +nightly fuzz run zone_balance_solver
cargo +nightly fuzz run epw_parser

# Run for a bounded number of iterations (useful in CI):
cargo +nightly fuzz run ffi_batch_oracle -- -max_total_time=60

# Resume from a known corpus / replay a crash:
cargo +nightly fuzz run ffi_batch_oracle -- fuzz/corpus/ffi_batch_oracle/
cargo +nightly fuzz run ffi_batch_oracle -- fuzz/artifacts/ffi_batch_oracle/crash-*
```

## Building without nightly (type-check only)

The targets are written so they **type-check** under stable Rust without the
libFuzzer runtime, which lets CI verify they stay compilable even on runners
that lack nightly:

```bash
cargo check --manifest-path fuzz/Cargo.toml
```

(For a full link you still need nightly; pass `cargo +nightly fuzz build`.)

## Adding a new target

1. Create `fuzz/fuzz_targets/<name>.rs` with a `libfuzzer_sys::fuzz_target!`
   body.
2. Register a new `[[bin]]` stanza in `fuzz/Cargo.toml`.
3. Run `cargo +nightly fuzz run <name>` to confirm it launches.

## See also

- `tests/concurrency/loom_concurrency_tests.rs` &mdash; `loom`-based
  concurrency-state-space fuzzing of the same `BatchOracle` parallel paths.
- `AGENTS.md` &mdash; cycle-breaking and feature-flag conventions that the fuzz
  crate must respect (it depends only on the default `fluxion` feature set).
