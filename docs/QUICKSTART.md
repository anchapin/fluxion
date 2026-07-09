# Fluxion Quickstart Guide

Get started with Fluxion in minutes.

> **⚠️ Pre-release note**
>
> The `eui` returned by `model.simulate(...)` in the examples below is
> currently a **raw cumulative temperature-departure metric**, not a
> calibrated `kWh/m²/year` value. The label `kWh/m²/year` in the
> examples is a placeholder that will become accurate after the
> ASHRAE 140 physics calibration work in
> [#749-G2](https://github.com/anchapin/fluxion/issues/749) lands. Do
> not benchmark the raw metric against ASHRAE 90.1 / RESNET HERS until
> then ([#767](https://github.com/anchapin/fluxion/issues/767)).

## What Fluxion actually exposes today (Issue #1411)

The `fluxion` Python module is built with PyO3 and ships (see `src/lib.rs:1727`):

- `fluxion.Model` — single-building detailed analysis (`Model(num_zones=1)`,
  `model.simulate(years, use_surrogates)`, `model.load_surrogate(path)`, …)
- `fluxion.BatchOracle` — high-throughput parallel population evaluation
  (`oracle.evaluate_population(population, use_surrogates)`, …)
- `fluxion.MultiZoneThermalModel` — multi-zone simulation
  (`MultiZoneThermalModel(num_zones=2)`, `simulate_multi_zone(years, use_surrogates)`,
  `get_zone_energies()`)
- `fluxion.BuildingParameters` / `fluxion.ParameterBounds` — typed
  building-design vectors for the oracle
- `fluxion.{VectorField, Construction, ConstructionLayer, WallSurface,
  GeometryTensor, SurfaceType, MassClass}` — geometric/material data types

There is no `fluxion.Model(config_path)`, no `fluxion.cli`, and no
`fluxion.serve` subcommand. The `Model` constructor only takes a
zone count. To drive Fluxion with a JSON config, use the REST server
described below.

The CLI binary is `fluxion` (the `fluxion` console script in
`pyproject.toml` would only exist if you pip-installed the wheel; the
real CLI is built from `src/bin/fluxion.rs` as `cargo run --bin fluxion
…`). The REST server is the separate `fluxion-rest` binary built from
`src/bin/fluxion_rest.rs` (port 8080, env vars `FLUXION_REST_BIND` /
`FLUXION_REST_PORT`).

## Installation

### From source (recommended)

```bash
git clone https://github.com/anchapin/fluxion.git
cd fluxion
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip maturin
maturin develop --release
```

`maturin develop --release` builds the Rust extension in
`--release` mode and installs the resulting `fluxion` Python module
into the active virtualenv. After this, `python -c "import fluxion"`
succeeds.

### From PyPI (future — not yet published)

The `fluxion` package is not yet on PyPI
([#766](https://github.com/anchapin/fluxion/issues/766)).
Once published, installation will be:

```bash
pip install fluxion
```

### With Docker

The published Docker image is built from the multi-stage `Dockerfile`
in this repo and runs the `fluxion-rest` binary on port **8080**:

```bash
docker build -t fluxion-rest .
docker run --rm -p 8080:8080 fluxion-rest
# In another terminal:
curl -s http://localhost:8080/v1/healthz
# => {"status":"ok","version":"..."}
```

> The earlier "fluxion-api" image (port 8000, `python -m api.main`,
> healthcheck on `/health`) documented in older revisions of this
> file no longer exists — it predates the Rust REST scaffold in
> PR #1371 and was never published. If you find a reference to it
> elsewhere in the repo, treat it as a stale doc and link here.

## Quick Examples

### 1. Basic simulation with `Model`

```python
from fluxion import Model

# Create a 1-zone thermal model with default setpoints.
# The constructor takes only `num_zones`; there is no config-file
# path. To drive a full geometry/construction/control setup use the
# REST API (see example 4) or the multi-zone path (example 5).
model = Model(num_zones=1)

# Run physics-based simulation.
# NOTE: `eui` is currently a raw cumulative temperature-departure
# metric, not calibrated kWh/m²/year. See the pre-release notes at the
# top of this document and issue #767 for context.
eui = model.simulate(years=1, use_surrogates=False)
print(f"EUI (uncalibrated, see #767): {eui:.2f}")
```

### 2. Using a pre-trained ONNX surrogate

```python
from fluxion import Model

model = Model(num_zones=1)

# Load a pre-trained ONNX surrogate for ~100x faster inference.
# `tools/generate_dummy_surrogate.py` produces a 1.2-constant dummy
# model so the wiring can be exercised end-to-end without a real
# training run:
#
#   python tools/generate_dummy_surrogate.py --zones 1 \
#       --out examples/dummy_surrogate.onnx
model.load_surrogate("examples/dummy_surrogate.onnx")

# Same uncalibrated-metric caveat applies (see #767).
eui = model.simulate(years=1, use_surrogates=True)
print(f"EUI with surrogate (uncalibrated): {eui:.2f}")
```

### 3. Population optimisation with `BatchOracle`

```python
from fluxion import BatchOracle

oracle = BatchOracle()

# Each candidate is [window_u_value, heating_setpoint, cooling_setpoint].
# `BatchOracle` validates the vector internally (see
# `BatchOracle.validate_parameters` and `BatchOracle::MIN_*` / `MAX_*`
# constants in `src/lib.rs`):
#   window_u_value    : 0.1 – 5.0 W/m²K
#   heating_setpoint  : 15.0 – 25.0 °C  (must be < cooling_setpoint)
#   cooling_setpoint  : 22.0 – 32.0 °C
population = [[1.5, 20.0, 24.0]] * 1000

# Returns a list of EUI values (kWh/m²/year, currently uncalibrated —
# see #767), one per candidate, evaluated in parallel via rayon.
results = oracle.evaluate_population(population, use_surrogates=True)

print(f"Evaluated {len(results)} designs")
print(f"Best EUI: {min(results):.2f}")
```

### 4. Driving Fluxion over the REST API

The Rust binary `fluxion-rest` exposes the canonical
`SimulationSchemaV1` (see `src/api/schema.rs` and
`src/api/openapi.yaml`) over HTTP on port 8080. JSON fixtures live
under `tests/fixtures/`. A complete request body is shipped at
[`tests/fixtures/single_zone.json`](../tests/fixtures/single_zone.json).

```bash
# 1. Start the server (in one terminal)
cargo run --bin fluxion-rest
# fluxion-rest listening on 0.0.0.0:8080 ...

# 2. Run a simulation (in another terminal)
curl -s -X POST http://localhost:8080/v1/simulate \
  -H 'content-type: application/json' \
  -d @tests/fixtures/single_zone.json | python3 -m json.tool
```

For a complete tour of every endpoint (health, openapi, simulate,
schema retrieval, import), see [`docs/REST_API.md`](REST_API.md) and
[`examples/run_rest.sh`](../examples/run_rest.sh).

### 5. Multi-zone Python

```python
from fluxion import MultiZoneThermalModel

model = MultiZoneThermalModel(num_zones=3)
model.set_zone_setpoints(0, heating=20.0, cooling=24.0)
total_energy_kwh = model.simulate_multi_zone(years=1, use_surrogates=False)
print(f"3-zone annual energy (heating+cooling): {total_energy_kwh:.2f} kWh")
```

### 6. The `fluxion` CLI (OpenStudio-compatible workflow)

The `fluxion` binary understands OpenStudio-style `.fwf` workflow
files. JSON config files (e.g. `simple_config.json`) are **not**
accepted by `fluxion run`; that path is reserved for the
`SimulationSchemaV1` JSON consumed by `fluxion-rest`.

```bash
# Build the CLI
cargo build --bin fluxion

# Run an OpenStudio-style workflow
fluxion run -w examples/workflow.fwf
```

## Your first configuration

If you want a `SimulationSchemaV1` you can hand to `fluxion-rest`, the
canonical example is [`tests/fixtures/single_zone.json`](../tests/fixtures/single_zone.json).
It matches `fluxion::api::schema::SimulationSchemaV1` byte-for-byte
(`tests/examples_smoke.rs` round-trips it on every CI run).

## Next Steps

- [`docs/REST_API.md`](REST_API.md) — every endpoint with curl examples
- [`docs/API_REFERENCE.md`](API_REFERENCE.md) — full API documentation
- [`docs/EXAMPLES.md`](EXAMPLES.md) — more usage examples
- [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md) — module boundaries
- [`examples/`](../examples/) — runnable scripts

## Getting Help

- GitHub Issues: https://github.com/anchapin/fluxion/issues
- Documentation: https://fluxion.readthedocs.io
