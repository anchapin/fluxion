# Examples: Inputs, Outputs, and Expected Behaviour

This document explains the inputs and outputs used by the scripts in
`examples/`, how to interpret the printed results, and shows small
recipes for normalising the toy-metric currently produced by the
physics engine. The examples are kept in lock-step with the live
PyO3 / axum surface — see `tests/examples_smoke.rs` (Issue #1411) for
the CI guard that fails the build if the fixtures and example
scripts drift away from the public types.

## 1. Inventory of `examples/`

| File                       | Path / surface                                  | Status                          |
|----------------------------|-------------------------------------------------|---------------------------------|
| `run_model.py`             | `fluxion.Model(num_zones=…)` (Python)           | Works as written                |
| `run_oracle.py`            | `fluxion.BatchOracle.evaluate_population` (Python) | Works as written              |
| `quick_start.sh`           | Helper: `maturin develop` + run `run_oracle.py` | Works as written                |
| `run_rest.sh`              | `curl` against `fluxion-rest` on port 8080      | Works as written (added #1411)  |
| `tests/fixtures/single_zone.json` | Canonical `SimulationSchemaV1` for `POST /v1/simulate` | Round-tripped by `tests/examples_smoke.rs` on every CI run |
| `dummy_surrogate.onnx`     | Pre-generated dummy ONNX for `Model.load_surrogate` | Works (1.2-constant)        |
| `multi_zone_demo.rs`       | `MultiZoneThermalModel` from Rust               | Reference only                  |
| `tutorial_custom_model.rs` | Custom thermal model demo (Rust)                | Reference only                  |
| `validate_surrogate.py`    | ONNX validation helper                          | Reference only                  |
| `risk_aware_optimization.py` | Risk-aware oracle workflow (Python)            | Reference only                  |
| `packs/`                   | Curated example model packs                     | Reference only                  |
| `legacy/`                  | Stale `simple_config.json` + `simulation_schema_v1.json` | Historical reference only — not consumed by any live surface (moved in #2544) |
| `*.rs` (construction_example, performance_example, …) | Rust binary examples under a separate `examples/Cargo.toml` | Out of scope for the Python/REST on-ramp |

## 2. Input formats

### 2.1 `run_model.py` — single `Model`

The current `Model` constructor (see `src/lib.rs:189`) takes a
single positional argument: the number of zones. There is **no**
config-file path. The constructor returns a `ThermalModel<VectorField>`
with default setpoints (heating 20 °C, cooling 24 °C, default
construction) — to drive a more detailed setup, use the REST API
(section 2.4) or the `MultiZoneThermalModel` path (section 2.5).

```python
from fluxion import Model

model = Model(num_zones=2)        # 2 zones, default setpoints
model.set_ground_temp(10.0)       # °C, optional
eui = model.simulate(years=1, use_surrogates=False)
```

`model.simulate(years, use_surrogates)` runs the physics for
`years * 8760` timesteps and returns a single `f64` EUI value
(uncalibrated — see the "Production scope" section in
[`docs/QUICKSTART.md`](QUICKSTART.md); calibration of the absolute
energy rate is tracked in issues #749 / #767).

If you want to use a pre-trained ONNX surrogate, call
`model.load_surrogate("path/to/model.onnx")` before
`model.simulate(...)`. The Python bindings only call
`SurrogateManager::load_onnx`; ONNX format compatibility is
inherited from `ort` 2.x (see `Cargo.toml`).

### 2.2 `run_oracle.py` — `BatchOracle` population evaluation

`BatchOracle.evaluate_population(population, use_surrogates)` expects
a `List[List[float]]`. Each inner list is one design candidate and
**must** have at least three elements (see
`src/lib.rs:1015` and `BatchOracle::validate_parameters`):

| Index | Meaning                       | Valid range      |
|-------|-------------------------------|------------------|
| 0     | Window U-value (W/m²K)        | 0.1 – 5.0        |
| 1     | Heating setpoint (°C)         | 15.0 – 25.0      |
| 2     | Cooling setpoint (°C)         | 22.0 – 32.0, **must be > heating** |

`BatchOracle` does **not** expose a `load_surrogate` method (only
`Model` does) — the oracle always uses its internal
`SurrogateManager`, which falls back to a deterministic mock when no
ONNX model is loaded (see `src/ai/surrogate`).

```python
from fluxion import BatchOracle

oracle = BatchOracle()
population = [
    [1.5, 20.0, 24.0],   # OK
    [0.8, 21.0, 25.0],   # OK
    # [99.0, 20.0, 24.0] # would raise ValidationError
]
results = oracle.evaluate_population(population, use_surrogates=False)
```

For type safety, prefer `oracle.evaluate_population_typed(...)` with
`fluxion.BuildingParameters(window_u_value=…, heating_setpoint=…,
cooling_setpoint=…)` objects (see `src/api/parameters.rs`).

### 2.3 Historical config stubs (`examples/legacy/`)

`examples/legacy/simple_config.json` and
`examples/legacy/simulation_schema_v1.json` pre-date the PyO3 / axum
surface and are **not** consumed by either `Model` or `BatchOracle`,
nor accepted by `fluxion-rest`. They were moved out of the top of
`examples/` in #2544 and are kept under `examples/legacy/` for
historical reference only. The canonical REST request body — and the
only JSON document the `POST /v1/simulate` endpoint is validated
against — is `tests/fixtures/single_zone.json`, which is round-tripped
by `tests/examples_smoke.rs` on every CI run.

### 2.4 `run_rest.sh` — REST API curl examples

The REST server (`cargo run --bin fluxion-rest`, port 8080) is the
only path that accepts a `SimulationSchemaV1` JSON document. The
OpenAPI 3.1 spec is at `src/api/openapi.yaml`; the canonical fixture
is `tests/fixtures/single_zone.json`. See `examples/run_rest.sh` for
ready-to-paste curl invocations of:

- `GET  /v1/healthz`
- `GET  /v1/openapi.yaml`
- `POST /v1/simulate` with `tests/fixtures/single_zone.json`
- `GET  /v1/schema/{id}` (using the `schema_id` returned above)
- `POST /v1/import/{osm|gbxml|idf}` (the `idf` variant returns 501)

## 3. Output explained

The Python examples print three things:

- `Elapsed` — wall-clock time to evaluate the population or run the
  model. Useful for measuring throughput.
- `Best candidate index` and its `EUI` — index in the population with
  the lowest (best) EUI and the numeric EUI value.
- `Sample results` — per-candidate printout showing `U`, setpoints,
  and `EUI`.

The REST `simulate` handler returns:

```json
{
  "schema_id": "sch-0",
  "output": {
    "eui": 12.34,
    "total_energy": 592.32,
    "peak_heating_load": 0.0,
    "peak_cooling_load": 0.0,
    "heating_energy": 350.0,
    "cooling_energy": 242.32,
    "zone_temperatures": [20.1],
    "hourly_zone_temperatures": null
  }
}
```

`peak_heating_load` and `peak_cooling_load` are always `0.0` in the
current release — the per-hour peak tracking is not wired into the
REST handler yet. Track via `tests/KNOWN_ISSUES.md` once it is.

## 4. Why are EUI values large?

The current `ThermalModel::solve_timesteps` accumulates a simple
metric: at each timestep it sums |temperature - setpoint| across
all zones and adds this to a running total. With `num_zones = 10`
and `timesteps = 8760`, you get `87,600` contributions — hence the
large numeric outputs. These are intentionally uncalibrated and
intended for algorithm correctness and performance testing.

## 5. Normalising the toy metric

To create a more human-friendly, average metric, normalise like this
(performed in Python after `results` are returned):

```python
num_zones = 1   # matches `Model(num_zones=…)` you constructed
timesteps_per_year = 8760

def normalize(raw_eui):
    return raw_eui / (num_zones * timesteps_per_year)

# Example usage
# normalized = normalize(results[best_idx])
```

This yields an average hourly temperature-gap per zone, which is
useful for relative comparisons between candidates.

## 6. Converting to physical units

To convert the normalised metric to physical energy (kWh/m²/year) you
need additional data:

- Thermal capacity / heat capacity (J/K) of zones or mass
- Area (m²) that the metric should be expressed per
- Time-step duration (hours) — here `1 hour` per step

Rough pipeline:

1. Convert average temperature-gap (°C) to energy using heat
   capacity (J/°C).
2. Convert Joules to kWh (1 kWh = 3.6 × 10⁶ J).
3. Divide by area (m²) and by simulation years to get kWh/m²/year.

## 7. Example: compute normalised EUI and print

```python
raw = results[best_idx]
normalized = raw / (num_zones * 8760)
print(f"Raw: {raw:.1f}, normalized avg temp-gap per zone (°C-hr): {normalized:.6f}")
```

## 8. Tips for using examples in tests or CI

- Use small populations (20–100) for CI to keep run-time small.
- Pin your Python interpreter (venv) in CI to match the
  maturin-built wheel platform.
- The REST `single_zone.json` fixture is round-tripped by
  `tests/examples_smoke.rs` on every CI run — keep that test green
  and your docs/examples stay in sync with the live API.
- To avoid flakiness, set the random seed in `run_oracle.py` (or
  use NumPy RNG) and/or mock the `SurrogateManager`.
