# fluxion fuzz targets

libFuzzer fuzz targets for the fluxion building-energy-modeling engine, standing
up the coverage called for in [issue #2537] and wired into nightly CI in
[issue #3461]. These complement the `loom::fuzz` concurrency tests
(`tests/concurrency/loom_concurrency_tests.rs`) with randomised *input* coverage
of the FFI surface and the zone-balance physics solver.

[issue #2537]: https://github.com/anchapin/fluxion/issues/2537
[issue #3461]: https://github.com/anchapin/fluxion/issues/3461

## Layout

```
fuzz/
  Cargo.toml              # standalone cargo workspace (NOT a member of the root workspace)
  README.md               # this file
  fuzz_targets/
    ffi_batch_oracle.rs   # BatchOracle::evaluate_population  (PyO3 / NAPI FFI surface)
    zone_balance_solver.rs# ThermalModel::step_physics + solve_timesteps  (zone-balance solver)
    epw_parser.rs         # EpwWeatherSource::from_file  (EPW weather-file parser)
  seed_corpus/            # committed seed corpus (Issue #3461 AC #2)
    epw_parser/           #   header_only.epw, single_hour.epw, summer_peak.epw
    ffi_batch_oracle/     #   empty.bin, single_candidate.bin, multi_candidate.bin
    zone_balance_solver/  #   empty.bin, one_step.bin, four_step_day.bin
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
cargo install cargo-fuzz --locked --version "^0.13"
```

> **Note:** `cargo-fuzz 0.13.x` is the floor that builds clean against the
> current `pulp 0.22.3` / `faer 0.24.4` / nightly combination. Earlier
> `0.11.x` releases trip a transitive `pulp` compile-time size assertion
> under ASAN. The nightly CI workflow (`.github/workflows/fuzz.yml`) installs
> `^0.13` automatically.

The stable toolchain is enough to **compile-check** the targets (see below) but
cannot link the libFuzzer runtime.

## Running the targets

```bash
# Run a single target (defaults to ~indefinite; Ctrl-C to stop). libFuzzer
# writes any crash-triggering inputs to fuzz/artifacts/<target>/.
cargo +nightly fuzz run ffi_batch_oracle
cargo +nightly fuzz run zone_balance_solver
cargo +nightly fuzz run epw_parser

# Seed the runtime corpus from the committed seed corpus (recommended
# for first runs and CI; see Issue #3461):
mkdir -p fuzz/corpus/ffi_batch_oracle
cp -n fuzz/seed_corpus/ffi_batch_oracle/* fuzz/corpus/ffi_batch_oracle/
cargo +nightly fuzz run ffi_batch_oracle

# Run for a bounded number of iterations (used by CI):
cargo +nightly fuzz run ffi_batch_oracle -- -max_total_time=300

# Resume from a known corpus / replay a crash:
cargo +nightly fuzz run ffi_batch_oracle -- fuzz/corpus/ffi_batch_oracle/
cargo +nightly fuzz run ffi_batch_oracle -- fuzz/artifacts/ffi_batch_oracle/crash-*
```

## Nightly CI

The targets are exercised nightly by
[`.github/workflows/fuzz.yml`](../.github/workflows/fuzz.yml). The workflow:

1. Schedules at `0 4 * * *` (04:00 UTC nightly) and accepts `workflow_dispatch`
   for on-demand re-runs (configurable wall-clock budget per target).
2. Runs on the ephemeral `ubuntu-latest` pool (per the
   [Issue #3445](#3445) runner-isolation contract; the self-hosted Hetzner
   pool is reserved for `push:main`-gated jobs).
3. Installs `cargo-fuzz ^0.13` and the latest nightly.
4. Builds all targets in a single `cargo +nightly fuzz build`, then runs each
   for `-max_total_time=300` (5 minutes) by default.
5. Seeds the runtime corpus from the committed `fuzz/seed_corpus/<target>/`
   so subsequent runs start from prior coverage (Issue #3461 acceptance
   criteria #2).
6. Uploads `fuzz/summaries/` (machine- and human-readable run summaries),
   plus `fuzz/artifacts/` and `fuzz/corpus/` (crash files and incremental
   corpus) as workflow artefacts. Any crash fails the job — the fuzz
   targets assert no-panic, so a crash is always a real regression.

The job is bounded to a 30-minute wall-clock timeout. Total monthly compute
is roughly 3 targets × 5 min × 30 days ≈ 7.5 hours, well under any of
the heavy-supplement suites (full Loom / mutation suites need ~32 GB RAM
and are not launched casually; see `AGENTS.md`).

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
3. Add 1-2 seed files under `fuzz/seed_corpus/<name>/` so the nightly CI
   starts from known coverage rather than from an empty corpus.
4. Run `cargo +nightly fuzz run <name>` to confirm it launches.

## See also

- `tests/concurrency/loom_concurrency_tests.rs` &mdash; `loom`-based
  concurrency-state-space fuzzing of the same `BatchOracle` parallel paths.
- `AGENTS.md` &mdash; cycle-breaking and feature-flag conventions that the fuzz
  crate must respect (it depends only on the default `fluxion` feature set).
- `.github/workflows/fuzz.yml` &mdash; the nightly CI wiring (Issue #3461).
