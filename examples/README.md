# Examples Overview

This folder contains small, self-contained examples that demonstrate
the primary user-facing APIs in Fluxion. After PR #1411 the on-ramp
matches the actual PyO3 and axum surfaces — no more `Model("config.json")`
or `BatchOracle.load_surrogate(...)` references that would fail at
runtime.

| File / directory                       | What it shows                                                              |
|----------------------------------------|----------------------------------------------------------------------------|
| `run_model.py`                          | `fluxion.Model(num_zones=N).simulate(...)` (analytical + ONNX surrogate)  |
| `run_oracle.py`                         | `fluxion.BatchOracle().evaluate_population(...)` (parallel population eval) |
| `run_rest.sh`                           | `curl` against `fluxion-rest` on port 8080 (`/v1/healthz`, `/v1/simulate`, `/v1/schema/{id}`) |
| `quick_start.sh`                        | One-shot: `maturin develop --release` + run `run_oracle.py`                |
| `tests/fixtures/single_zone.json` (sibling of this folder) | Canonical `SimulationSchemaV1` for `POST /v1/simulate` |
| `dummy_surrogate.onnx`                  | 1.2-constant ONNX model so `Model.load_surrogate` is exercisable          |
| `multi_zone_demo.rs`                    | Rust-only `MultiZoneThermalModel` reference                                |
| `tutorial_custom_model.rs`              | Custom thermal model demo (Rust)                                           |
| `validate_surrogate.py`                 | ONNX validation helper (used by surrogate work)                           |
| `risk_aware_optimization.py`            | Risk-aware `BatchOracle` workflow (Python)                                 |
| `performance_example.rs`                | Throughput measurement (Rust)                                              |
| `construction_example.rs`               | Multi-layer construction assembly (Rust)                                   |
| `legacy/`                               | Stale `simple_config.json` + `simulation_schema_v1.json` kept for historical reference only (moved in #2544) |
| `packs/`                                | Curated example model packs (Rust reference data)                         |
| `*.rs` under the inner `examples/Cargo.toml` | Standalone Rust binaries (built with `cargo run --manifest-path examples/Cargo.toml`) |

> **Canonical REST fixture.** The only JSON document the `fluxion-rest`
> `POST /v1/simulate` endpoint is validated against is
> [`../tests/fixtures/single_zone.json`](../tests/fixtures/single_zone.json),
> round-tripped by `tests/examples_smoke.rs` on every CI run. The
> historical stubs under `legacy/` are **not** consumed by `Model`,
> `BatchOracle`, or `fluxion-rest`.

## Purpose

- Provide reproducible examples for new users to run locally after
  building the Python bindings with `maturin develop --release`.
- Demonstrate the expected input formats (population vectors, REST
  request bodies) and show simple output interpretation.
- Cover both the **in-process Python** surface (`fluxion.Model`,
  `fluxion.BatchOracle`) and the **out-of-process REST** surface
  (`fluxion-rest` on port 8080).

## Running the examples

From the repository root, after building/installing the Python
bindings locally:

```bash
# Optional: create and activate a venv
python3 -m venv .venv
source .venv/bin/activate

# Build & install local Python bindings
pip install --upgrade pip maturin
maturin develop --release

# Python: in-process evaluation
python examples/run_model.py
python examples/run_oracle.py

# REST: out-of-process evaluation
cargo run --bin fluxion-rest &
sleep 1
bash examples/run_rest.sh
```

## Quick-start script

`examples/quick_start.sh` automates the `maturin develop` install
(it leaves the source `.venv` alone — invoke it after activating
yours) and runs `examples/run_oracle.py`. Inspect it before running
on CI environments.

## Modifying examples

- `run_oracle.py` uses a small default population size (20). To
  stress-test throughput, increase the number passed to
  `make_population(n)` and observe `Elapsed` time.
- `run_model.py` runs a 1-year simulation by calling
  `Model.simulate(1, use_surrogates)`. Adjust the `years` argument
  to run multi-year checks.
- The parameter vector in `run_oracle.py` is
  `[window_u_value, heating_setpoint, cooling_setpoint]` (three
  elements). The previous `[u, setpoint]` two-element form would
  fail `BatchOracle.validate_parameters` with
  "Cooling setpoint (index 2) … out of range".

## Notes on determinism

- The repository's `SurrogateManager` currently returns deterministic
  mock loads when no ONNX model is loaded.
- `run_model.py` only calls `model.load_surrogate` if
  `examples/dummy_surrogate.onnx` is present; if the file is missing
  it falls through to the analytical path so the script still
  succeeds.
- `BatchOracle` does **not** expose `load_surrogate` — the oracle
  always uses its internal `SurrogateManager`. The legacy
  `oracle.load_surrogate(dummy_path)` call in this example was
  removed in #1411.

## Where to go next

- See [`../docs/EXAMPLES.md`](../docs/EXAMPLES.md) for detailed
  input/output semantics and example calculations.
- See [`../docs/REST_API.md`](../docs/REST_API.md) for the REST API
  reference (curl examples for every endpoint, OpenAPI 3.1 contract).
- See [`../docs/QUICKSTART.md`](../docs/QUICKSTART.md) for the
  five-minute on-ramp.
- See [`../docs/ARCHITECTURE.md`](../ARCHITECTURE.md) for the
  module-boundary overview and `src/lib.rs:1727` for the
  PyO3 module entry point.
