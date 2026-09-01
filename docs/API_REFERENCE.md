# Fluxion API Reference

## Python API

### Model

The main model class for single-building energy simulation.

```python
from fluxion import Model

# Create a model
model = Model(num_zones=1)

# Run simulation
eui = model.simulate(years=1, use_surrogates=False)
print(f"Energy Use Intensity: {eui:.2f} kWh/m²/year")

# Load a surrogate model
model.load_surrogate("loads_predictor.onnx")

# Run with surrogates
eui_surrogate = model.simulate(years=1, use_surrogates=True)
```

#### Constructor

```python
Model(num_zones: int = 1) -> Model
```

Create a new Model instance.

**Parameters:**
- `num_zones` (int): Number of thermal zones (default: 1)

**Returns:** A new Model instance

#### Methods

##### simulate()

```python
model.simulate(years: int, use_surrogates: bool) -> float
```

Simulate building energy consumption over specified years.

**Parameters:**
- `years` (int): Number of years to simulate (1-5 typical)
- `use_surrogates` (bool): If true, use AI surrogates; if false, use physics

**Returns:** Total energy use intensity (EUI) in kWh/m²/year

**Simulation Modes:**

**Analytical Mode (use_surrogates=False):**
- Uses ISO 13790 5R1C thermal network physics
- Calculates loads analytically from outdoor temperature
- Baseline for validation and accuracy testing
- Slower but physically exact

**Surrogate Mode (use_surrogates=True):**
- Uses neural network surrogates for load predictions
- Fast inference via ONNX Runtime
- Enables high-throughput optimization (>10,000 configs/sec)
- Requires pre-trained ONNX model loaded via `load_surrogate()`

##### load_surrogate()

```python
model.load_surrogate(model_path: str) -> None
```

Register an ONNX surrogate model.

**Parameters:**
- `model_path` (str): Path to ONNX model file

**Model Requirements:**
- Format: ONNX (Open Neural Network Exchange)
- Input: Temperature vector (8760 hourly values per zone)
- Output: Load predictions (8760 hourly values per zone)
- Supported backends: CPU, CUDA, CoreML, DirectML, OpenVINO

**Raises:** `SurrogateError` if model loading fails

##### get_parameter_bounds()

```python
model.get_parameter_bounds() -> ParameterBounds
```

Get parameter bounds for building design variables.

**Returns:** `ParameterBounds` struct with fields:
- `min_u_value` (float): Minimum window U-value (0.1 W/m²K)
- `max_u_value` (float): Maximum window U-value (5.0 W/m²K)
- `min_heating_setpoint` (float): Minimum heating setpoint (15.0°C)
- `max_heating_setpoint` (float): Maximum heating setpoint (25.0°C)
- `min_cooling_setpoint` (float): Minimum cooling setpoint (22.0°C)
- `max_cooling_setpoint` (float): Maximum cooling setpoint (32.0°C)

**Use Case:** Generate valid parameter vectors for optimization libraries.

```python
bounds = model.get_parameter_bounds()
print(f"U-value range: [{bounds.min_u_value}, {bounds.max_u_value}] W/m²K")
print(f"Heating setpoint range: [{bounds.min_heating_setpoint}, {bounds.max_heating_setpoint}]°C")
```

##### validate_parameters()

```python
model.validate_parameters(params: List[float]) -> None
```

Validate a parameter vector against physical constraints.

**Parameters:**
- `params` (List[float]): Parameter vector to validate:
  - `[0]`: Window U-value (W/m²K, range: 0.1-5.0)
  - `[1]`: Heating setpoint (°C, range: 15.0-25.0)
  - `[2]`: Cooling setpoint (°C, range: 22.0-32.0)

**Validation Checks:**
- All values must be finite (not NaN or infinite)
- Window U-value must be in [0.1, 5.0] W/m²K
- Heating setpoint must be in [15.0, 25.0]°C
- Cooling setpoint must be in [22.0, 32.0]°C
- Heating setpoint must be less than cooling setpoint

**Raises:** `ValidationError` with detailed message including:
- Parameter index
- Invalid value
- Valid range
- Type of error (NaN, infinite, or out of range)

**Example:**
```python
import fluxion

oracle = fluxion.BatchOracle()

# Valid parameters
oracle.validate_parameters([1.5, 20.0, 27.0])  # OK

# Invalid U-value (raises ValidationError)
try:
    oracle.validate_parameters([-1.0, 20.0, 27.0])
except fluxion.ValidationError as e:
    print(f"Validation failed: {e}")
    # Output: Window U-value (index 0, -1.00 W/m²K) out of range [0.1, 5.0] W/m²K

# NaN value (raises ValidationError)
try:
    oracle.validate_parameters([float('nan'), 20.0, 27.0])
except fluxion.ValidationError as e:
    print(f"Validation failed: {e}")
    # Output: Window U-value (index 0) is NaN (value: nan W/m²K). Cannot use in simulation.
```

---

### BatchOracle

High-throughput parallel oracle for optimization workflows.

```python
from fluxion import BatchOracle

# Create oracle
oracle = BatchOracle()

# Define population
population = [
    [1.5, 20.0, 27.0],  # [u_value, heating, cooling]
    [2.0, 21.0, 26.0],
    [1.0, 19.0, 28.0],
]

# Evaluate population
results = oracle.evaluate_population(population, use_surrogates=True)
print(f"Results: {results}")
```

#### Constructor

```python
BatchOracle() -> BatchOracle
```

Create a new BatchOracle instance.

#### Methods

##### evaluate_population()

```python
oracle.evaluate_population(
    population: List[List[float]],
    use_surrogates: bool
) -> List[float]
```

Evaluate a population of building designs in parallel.

**Parameters:**
- `population` (List[List[float]]): List of parameter vectors
  - Each vector: [window_u_value, heating_setpoint, cooling_setpoint]
  - u_value: W/m²K, range 0.1-5.0
  - heating_setpoint: °C, range 15-25
  - cooling_setpoint: °C, range 22-32
- `use_surrogates` (bool): Use AI surrogates for fast inference

**Returns:** List of EUI values (kWh/m²/year) for each parameter vector.

**Performance:**
- Analytical mode (use_surrogates=False): ~900 configs/sec on 8-core CPU (release build)
- Surrogate mode (use_surrogates=True): GPU acceleration enables higher throughput
- Target latency: <100ms for 1000 configs

**Architecture:**

**Without Surrogates (Config-First Loop):**
- Each configuration runs independently through all 8760 timesteps
- Parallelized with Rayon at population level
- Good for validation and accuracy testing

**With Surrogates (Time-First Loop):**
- Time loop (0..8760) runs sequentially on main thread
- Collect all temperatures from all configurations → single batched inference
- Distribute loads via Rayon, run physics in parallel
- Maximizes GPU tensor core utilization for batched inference

**Raises:** `FluxionError` if population validation fails.

##### load_surrogate()

```python
oracle.load_surrogate(model_path: str) -> None
```

Load an ONNX surrogate model.

**Parameters:**
- `model_path` (str): Path to ONNX model file

**Supported Backends:**
- CPU (default)
- CUDA (NVIDIA GPUs)
- CoreML (Apple Silicon)
- DirectML (Windows)
- OpenVINO (Intel)

**Raises:** `SurrogateError` if model loading fails

**GPU Backend Selection:**

To use GPU acceleration, use the Rust API directly:
```rust
use fluxion::ai::surrogate::{SurrogateManager, InferenceBackend};

let manager = SurrogateManager::with_gpu_backend(
    "model.onnx",
    InferenceBackend::CUDA,
    device_id
)?;
```

##### get_parameter_bounds()

```python
oracle.get_parameter_bounds() -> ParameterBounds
```

Get parameter bounds for building design variables.

**Returns:** `ParameterBounds` struct with fields:
- `min_u_value` (float): Minimum window U-value (0.1 W/m²K)
- `max_u_value` (float): Maximum window U-value (5.0 W/m²K)
- `min_heating_setpoint` (float): Minimum heating setpoint (15.0°C)
- `max_heating_setpoint` (float): Maximum heating setpoint (25.0°C)
- `min_cooling_setpoint` (float): Minimum cooling setpoint (22.0°C)
- `max_cooling_setpoint` (float): Maximum cooling setpoint (32.0°C)

**Use Case:** Generate valid parameter vectors for optimization libraries.

```python
bounds = oracle.get_parameter_bounds()
print(f"U-value range: [{bounds.min_u_value}, {bounds.max_u_value}] W/m²K")
print(f"Heating setpoint range: [{bounds.min_heating_setpoint}, {bounds.max_heating_setpoint}]°C")
```

##### validate_parameters()

```python
oracle.validate_parameters(params: List[float]) -> None
```

Validate a parameter vector against physical constraints.

**Parameters:**
- `params` (List[float]): Parameter vector to validate:
  - `[0]`: Window U-value (W/m²K, range: 0.1-5.0)
  - `[1]`: Heating setpoint (°C, range: 15.0-25.0)
  - `[2]`: Cooling setpoint (°C, range: 22.0-32.0)

**Validation Checks:**
- All values must be finite (not NaN or infinite)
- Window U-value must be in [0.1, 5.0] W/m²K
- Heating setpoint must be in [15.0, 25.0]°C
- Cooling setpoint must be in [22.0, 32.0]°C
- Heating setpoint must be less than cooling setpoint

**Raises:** `ValidationError` with detailed message including:
- Parameter index
- Invalid value
- Valid range
- Type of error (NaN, infinite, or out of range)

**Example:**
```python
import fluxion

oracle = fluxion.BatchOracle()

# Valid parameters
oracle.validate_parameters([1.5, 20.0, 27.0])  # OK

# Invalid U-value (raises ValidationError)
try:
    oracle.validate_parameters([-1.0, 20.0, 27.0])
except fluxion.ValidationError as e:
    print(f"Validation failed: {e}")
    # Output: Window U-value (index 0, -1.00 W/m²K) out of range [0.1, 5.0] W/m²K

# NaN value (raises ValidationError)
try:
    oracle.validate_parameters([float('nan'), 20.0, 27.0])
except fluxion.ValidationError as e:
    print(f"Validation failed: {e}")
    # Output: Window U-value (index 0) is NaN (value: nan W/m²K). Cannot use in simulation.
```

---

## Quick Start Examples

### Example 1: Basic Population Evaluation

```python
from fluxion import BatchOracle

# Create oracle
oracle = BatchOracle()

# Define test population
population = [
    [1.5, 20.0, 27.0],  # Low U-value, standard setpoints
    [2.5, 18.0, 26.0],  # High U-value, cooler setpoints
    [1.0, 22.0, 28.0],  # Very low U-value, warmer setpoints
]

# Evaluate analytically (physics-only)
results = oracle.evaluate_population(population, use_surrogates=False)
print(f"Analytical EUI values: {results}")
```

### Example 2: GPU Batching with Custom Backend

```python
from fluxion import BatchOracle
import time

# Create oracle (CPU backend by default)
oracle = BatchOracle()

# Load surrogate model for GPU acceleration
oracle.load_surrogate("loads_predictor.onnx")

# Generate large population for benchmarking
import random
population = [
    [
        random.uniform(0.1, 5.0),  # U-value
        random.uniform(15.0, 25.0),  # Heating setpoint
        random.uniform(22.0, 32.0),  # Cooling setpoint
    ]
    for _ in range(10000)
]

# Benchmark surrogate mode (GPU-accelerated)
start = time.time()
results = oracle.evaluate_population(population, use_surrogates=True)
duration = time.time() - start
throughput = len(population) / duration

print(f"Evaluated {len(population)} configs in {duration:.2f}s")
print(f"Throughput: {throughput:.0f} configs/sec")
print(f"Sample results: {results[:5]}")
```

### Example 3: Parameter Validation with Bounds Checking

```python
from fluxion import BatchOracle

# Create oracle
oracle = BatchOracle()

# Get parameter bounds
bounds = oracle.get_parameter_bounds()

print("Parameter Bounds:")
print(f"  U-value: [{bounds.min_u_value:.1f}, {bounds.max_u_value:.1f}] W/m²K")
print(f"  Heating: [{bounds.min_heating_setpoint:.1f}, {bounds.max_heating_setpoint:.1f}]°C")
print(f"  Cooling: [{bounds.min_cooling_setpoint:.1f}, {bounds.max_cooling_setpoint:.1f}]°C")

# Test various parameter vectors
test_cases = [
    [1.5, 20.0, 27.0],  # Valid
    [0.05, 20.0, 27.0],  # U-value too low
    [6.0, 20.0, 27.0],  # U-value too high
    [1.5, 14.0, 27.0],  # Heating too low
    [1.5, 26.0, 27.0],  # Heating too high
    [1.5, 20.0, 21.0],  # Cooling too low
    [1.5, 20.0, 33.0],  # Cooling too high
    [1.5, 25.0, 20.0],  # Heating >= cooling (invalid)
    [1.5, float('nan'), 27.0],  # NaN value
]

print("\nValidation Results:")
for params in test_cases:
    try:
        oracle.validate_parameters(params)
        print(f"  {params} -> VALID")
    except Exception as e:
        print(f"  {params} -> INVALID: {e}")
```

### Example 4: Error Recovery (Fallback to Analytical)

```python
from fluxion import BatchOracle, SurrogateError

# Create oracle
oracle = BatchOracle()

# Try to load surrogate model
try:
    oracle.load_surrogate("loads_predictor.onnx")
    print("Surrogate model loaded successfully")
    use_surrogates = True
except SurrogateError as e:
    print(f"Failed to load surrogate: {e}")
    print("Falling back to analytical mode")
    use_surrogates = False

# Evaluate population (using available mode)
population = [[1.5, 20.0, 27.0], [2.0, 21.0, 26.0]]
results = oracle.evaluate_population(population, use_surrogates=use_surrogates)
print(f"Results: {results}")
```

### Example 5: Performance Benchmarking

```python
from fluxion import BatchOracle
import time

# Create oracle
oracle = BatchOracle()

# Generate test populations of different sizes
population_sizes = [100, 1000, 5000, 10000]

print("Performance Benchmark:")
print("Size\tAnalytical\tSurrogate\tSpeedup")
print("-" * 60)

for size in population_sizes:
    # Generate random population
    import random
    population = [
        [random.uniform(0.1, 5.0), random.uniform(15.0, 25.0), random.uniform(22.0, 32.0)]
        for _ in range(size)
    ]

    # Benchmark analytical mode
    start = time.time()
    results_analytical = oracle.evaluate_population(population, use_surrogates=False)
    time_analytical = time.time() - start

    # Load surrogate model
    oracle.load_surrogate("loads_predictor.onnx")

    # Benchmark surrogate mode
    start = time.time()
    results_surrogate = oracle.evaluate_population(population, use_surrogates=True)
    time_surrogate = time.time() - start

    # Calculate speedup
    speedup = time_analytical / time_surrogate

    print(f"{size}\t{time_analytical:.3f}s\t\t{time_surrogate:.3f}s\t\t{speedup:.1f}x")
```

---

## Error Cases

### Invalid Parameters

**Out of Range:**
```python
import fluxion

oracle = fluxion.BatchOracle()

try:
    oracle.validate_parameters([6.0, 20.0, 27.0])  # U-value too high
except fluxion.ValidationError as e:
    print(f"Error: {e}")
    # Output: Window U-value (index 0, 6.00 W/m²K) out of range [0.1, 5.0] W/m²K
```

**NaN or Infinite Values:**
```python
try:
    oracle.validate_parameters([1.5, float('inf'), 27.0])
except fluxion.ValidationError as e:
    print(f"Error: {e}")
    # Output: Heating setpoint (index 1) is infinite (value: inf°C). Cannot use in simulation.
```

**Heating/Cooling Conflict:**
```python
try:
    oracle.validate_parameters([1.5, 25.0, 20.0])  # Heating >= cooling
except fluxion.ValidationError as e:
    print(f"Error: {e}")
    # Output: Heating setpoint (25.00°C, index 1) must be less than cooling setpoint (20.00°C, index 2)
```

### ONNX Runtime Failures

**Model File Not Found:**
```python
try:
    oracle.load_surrogate("nonexistent.onnx")
except fluxion.SurrogateError as e:
    print(f"Error: {e}")
    # Output: Failed to load ONNX surrogate model 'nonexistent.onnx': No such file or directory
```

**Invalid Model Format:**
```python
try:
    oracle.load_surrogate("invalid_model.txt")
except fluxion.SurrogateError as e:
    print(f"Error: {e}")
    # Output: Failed to load ONNX surrogate model 'invalid_model.txt': Failed to load model
```

### Population Format Errors

**Empty Population:**
```python
results = oracle.evaluate_population([], use_surrogates=False)
print(f"Results: {results}")  # Output: []
```

**Invalid Vector Length:**
```python
# Note: This will not raise an error but will use default values for missing parameters
results = oracle.evaluate_population([[1.5, 20.0]], use_surrogates=False)
print(f"Results: {results}")
```

---

## Cross-References

- **Parameter Vector Semantics:** See `CLAUDE.md` for detailed parameter semantics and design variable definitions
- **Architecture Details:** See `docs/ARCHITECTURE.md` for BatchOracle pattern and thermal network structure
- **Testing Guidelines:** See `docs/CONTRIBUTING.md` for testing strategies and validation approaches
- **Known Limitations:** See `docs/KNOWN_ISSUES.md` for 5R1C model limitations and accuracy constraints

---

## Rust API

### ThermalModel

```rust
use fluxion::sim::engine::ThermalModel;
use fluxion::physics::cta::VectorField;

// Create model
let mut model = ThermalModel::<VectorField>::new(10);

// Apply parameters
model.apply_parameters(&[1.5, 20.0, 27.0]);

// Run simulation
let energy = model.solve_timesteps(8760, &surrogates, false);
```

### SurrogateManager

```rust
use fluxion::ai::surrogate::SurrogateManager;

// Load ONNX model
let surrogates = SurrogateManager::load_onnx("model.onnx")?;

// Get predictions
let loads = surrogates.predict_loads(&temperatures);
```

---

## REST API

Issue **#1342** added an axum-based REST server (mounted via the `fluxion-rest`
bin target) as a complement to the Python and Rust surfaces above. PR **#1468**
extended it with `x-request-id` propagation, structured tracing, and a
`/v1/metrics` Prometheus endpoint (Issue **#1447**).

The canonical machine-readable contract is `src/api/openapi.yaml` (also served
at `GET /v1/openapi.yaml`). A drift test in `src/api/server.rs`
(`openapi_yaml_paths_match_router`) fails CI if a route is added to either
side without the other, so the spec cannot silently orphan itself.

For installation, environment variables, error semantics, and full design notes
see [`docs/REST_API.md`](REST_API.md). This section enumerates the endpoint
surface and request/response shapes.

### Starting the server

```bash
cargo run --bin fluxion-rest --release
# Listens on 0.0.0.0:8080 by default; override with
#   FLUXION_REST_BIND=127.0.0.1 FLUXION_REST_PORT=9090 cargo run --bin fluxion-rest
```

Verify with:

```bash
curl -sf http://localhost:8080/v1/healthz
# => {"status":"ok","version":"<CARGO_PKG_VERSION>"}
```

Every response carries an `x-request-id` header (UUIDv4, generated by
`SetRequestIdLayer`/`MakeRequestUuid`). Inbound `x-request-id` headers are
preserved; outbound responses always include the id so operators can correlate
a 5xx with the structured log line emitted by `TraceLayer`.

### Endpoint index

| Method | Path                  | Purpose                                        |
|--------|-----------------------|------------------------------------------------|
| GET    | `/v1/healthz`         | Liveness probe (`200 OK`, static JSON body)    |
| GET    | `/v1/metrics`         | Prometheus text exposition (Issue #1447)       |
| GET    | `/v1/openapi.json`    | OpenAPI 3.1 spec as JSON envelope              |
| GET    | `/v1/openapi.yaml`    | OpenAPI 3.1 spec as raw YAML                   |
| POST   | `/v1/simulate`        | Run a simulation against a schema              |
| GET    | `/v1/schema/{id}`     | Fetch a previously stored schema               |
| POST   | `/v1/import/{fmt}`    | Convert OSM/gbXML/IDF to `SimulationSchemaV1`  |

The 7 routes are pinned in two places that must stay in sync:

- `Router::new()` in `src/api/server.rs:476` (axum-style `:id`/`:fmt`)
- `paths:` in `src/api/openapi.yaml` (OpenAPI-style `{id}`/`{fmt}`)

### `GET /v1/healthz`

Liveness probe. Always 200; does not touch downstream services.

```bash
curl -s http://localhost:8080/v1/healthz
```

Response `200 OK` (`application/json`):

```json
{ "status": "ok", "version": "1.0.0" }
```

### `GET /v1/metrics`

Prometheus text exposition. Scraped counters/histograms are populated by the
`metrics::record` middleware (`src/api/metrics.rs`).

```bash
curl -s http://localhost:8080/v1/metrics | head -5
# => # HELP fluxion_rest_requests_total Total number of HTTP requests
# => # TYPE fluxion_rest_requests_total counter
# => fluxion_rest_requests_total{method="GET",route="/v1/healthz",status="200"} 3
# => ...
```

| Metric                                  | Type      | Labels                  |
|-----------------------------------------|-----------|-------------------------|
| `fluxion_rest_requests_total`           | counter   | `route,method,status`   |
| `fluxion_rest_errors_total`             | counter   | `route,method,status`   |
| `fluxion_rest_request_duration_seconds` | histogram | `route,method`          |

The `route` label is the **matched pattern** (e.g. `/v1/schema/:id`) so a
unique id per request does not fragment cardinality.

### `GET /v1/openapi.{json,yaml}`

Two views of the same embedded `src/api/openapi.yaml` (compiled in via
`include_str!` at `src/api/server.rs:247`, so the served spec cannot drift
from the on-disk spec).

```bash
curl -s http://localhost:8080/v1/openapi.yaml | head -2
# => openapi: 3.1.0
# => info:
```

`/v1/openapi.json` wraps the YAML in `{ "openapi": "3.1.0", "spec": "<yaml>" }`
for clients that prefer JSON.

### `POST /v1/simulate`

Run a simulation synchronously against a `SimulationSchemaV1`. The wire shape
is the same `SimulationSchema` reused by the Python `Model` API
(`src/api/schema.rs`); no schema-specific alternates.

Request body (`application/json`):

```json
{
  "version": "V1",
  "metadata":    { "name": "single-zone", "description": null, "author": null, "created_at": null },
  "geometry":    { "zones": [{ "name": "Zone1", "floor_area": 48.0, "volume": 129.6, "height": 2.7 }],
                   "total_floor_area": 48.0, "total_volume": 129.6, "number_of_floors": 1, "floor_height": 2.7 },
  "constructions": { "wall": ..., "roof": ..., "floor": ... },
  "schedules":   { ... },
  "weather":     { ... },
  "controls":    { "zone_control": { "heating_setpoint": 20.0, "cooling_setpoint": 27.0, ... } },
  "output":      {},
  "options":     { "years": 1, "use_surrogates": false, "store_as": null }
}
```

The top-level `options` field is optional. `store_as` lets the caller pin a
schema id; otherwise the server auto-stores and returns the id so the schema
can be retrieved via `GET /v1/schema/{id}`.

Solver selection (Issue #3281): `options` also accepts

| Field               | Values                              | Default    |
|---------------------|-------------------------------------|------------|
| `zone_solver`       | `"gauge"` \| `"5r1c"` \| `"9r4c"`   | *(omitted)*|
| `conduction_solver` | `"default"` \| `"ctf"` \| `"fd"`    | `"default"`|

Unknown values are rejected with `400 invalid_request`. The experimental
`"6r2c"` / `"8r3c"` zone solvers are also rejected — with a message naming
the `FLUXION_EXPERIMENTAL_ZONE_SOLVERS=1` env var — unless that env var is
set on the server; even then they stay unavailable until the
`fluxion-experimental-zone-solvers` cargo feature ships (issue #3291).

An *explicit* `zone_solver: "gauge"` is rejected with `400 invalid_request`
(fail-closed, issue #3305): the REST schema does not carry per-surface
construction detail (`wall_spec`), so the gauge solver can never initialise
on this path and the request would silently fall through to 5R1C. Omitting
the field keeps the legacy default-selector behaviour (the β-phase 5R1C
fall-through) unchanged; `"5r1c"` / `"9r4c"` dispatch strictly. Each
successful simulation increments the `fluxion_simulation_solver_kind`
counter with a `solver="{zone}+{conduction}"` label (e.g.
`solver="gauge+default"`) that reports the *requested* stack.

Response `200 OK`:

```json
{
  "schema_id": "sch-0",
  "output": {
    "eui": 42.7,
    "total_energy": 2050.5,
    "peak_heating_load": 1850.0,
    "peak_cooling_load": 2240.0,
    "heating_energy": 800.0,
    "cooling_energy": 1250.5,
    "zone_temperatures": [21.4, 21.6],
    "hourly_zone_temperatures": [[21.4, 21.1, ... 8760 values ...], ...],
    "effective_solver": "5r1c"
  }
}
```

`output.effective_solver` (issue #3305) reports the zone solver that
ACTUALLY executed — derived from the dispatcher's per-step outcome, not
from the request. On the REST path today this is `"5r1c"` for the omitted
default and for explicit `"5r1c"`, and `"9r4c"` for explicit `"9r4c"`; it
can read `"gauge"` only once the gauge path is wired behind REST (post-#3291
PR4). The field is omitted on non-REST uses of the schema.

Errors are returned via the `ApiError` envelope (`src/api/server.rs:189`):

| Status | `error.kind`        | Trigger                                            |
|--------|---------------------|----------------------------------------------------|
| 400    | `invalid_schema`    | setpoints inverted or `geometry.zones` empty       |
| 400    | `invalid_request`   | unknown / experimental `zone_solver` or `conduction_solver`; explicit `zone_solver: "gauge"` over REST (issue #3305) |
| 422    | `import_failed`     | foreign file decode failed                         |
| 500    | `simulation_failed` | physics or surrogate manager error                 |
| 501    | `not_implemented`   | `POST /v1/import/idf` (no reader yet — see #1341)  |

```bash
curl -s -X POST http://localhost:8080/v1/simulate \
  -H 'content-type: application/json' \
  -d @tests/fixtures/single_zone.json | jq '.cooling_energy'
```

### `GET /v1/schema/{id}`

Retrieve a previously stored schema by id. 404 if the id is unknown (storage
is process-local; restarting the server clears the store — a persistent
backend is explicitly out of scope per #1342).

```bash
curl -s http://localhost:8080/v1/schema/sch-0 | jq '.geometry.zones | length'
```

Response is the `SimulationSchemaV1` JSON itself (200) or the standard
error envelope (404 with `error.kind = "schema_not_found"`).

### `POST /v1/import/{fmt}`

Convert an external model file into a `SimulationSchemaV1` and store it. The
body is the **raw file bytes** (no multipart envelope).

| `{fmt}`   | Reader                          | Status                  |
|-----------|---------------------------------|-------------------------|
| `osm`     | `crate::interop::osm::import_osm`       | supported         |
| `gbxml`   | `crate::interop::gbxml::import_gbxml`   | supported         |
| `idf`     | _(no reader in `src/interop/*` yet)_    | **501**            |

```bash
curl -s -X POST http://localhost:8080/v1/import/osm \
  --data-binary @model.osm | jq '.schema_id'
```

Response `200 OK`:

```json
{
  "schema_id": "sch-7",
  "schema": { /* SimulationSchemaV1 */ }
}
```

### See also

- [`docs/REST_API.md`](REST_API.md) — installation, environment variables, error
  semantics, design notes.
- `src/api/server.rs` — router definition (lines 476-486) and handlers.
- `src/api/openapi.yaml` — canonical OpenAPI 3.1 contract.
- `tests/api_integration_tests.rs`, `tests/api_observability_tests.rs` —
  end-to-end HTTP tests for the surface documented above.
- PR #1468 (Issue #1447) — provenance of `/v1/metrics` and `x-request-id`.

---

## Configuration

### Building Configuration (JSON)

```json
{
  "zone_area": 48.0,
  "zone_volume": 129.6,
  "window_u_value": 1.5,
  "window_area": 12.0,
  "heating_setpoint": 20.0,
  "cooling_setpoint": 27.0,
  "infiltration_rate": 0.5,
  "internal_gains": 100.0
}
```

### Surrogate Model Requirements

- Format: ONNX (Open Neural Network Exchange)
- Input: Temperature vector (8760 timesteps)
- Output: Load predictions (8760 values)
- Runtime: ONNX Runtime

---

## Output Data

### Hourly Zone Temperature Profiles (Issue #763)

After running a simulation with `solve_timesteps()` or `solve_timesteps_with_dt()`, the full hourly temperature profiles are available:

```python
# Get hourly temperatures after simulation
hourly_temps = model.get_hourly_temperatures()
if hourly_temps is not None:
    # hourly_temps[zone_idx][timestep] -> temperature in °C
    for zone_idx, zone_temps in enumerate(hourly_temps):
        print(f"Zone {zone_idx}: {len(zone_temps)} timesteps, "
              f"min={min(zone_temps):.1f}°C, max={max(zone_temps):.1f}°C")
```

**Data Format:** `Option<Vec<Vec<f64>>>` — `[num_zones][8760]` values in °C.

| Field | Type | Description |
|-------|------|-------------|
| `hourly_zone_temperatures` | `Option<Vec<Vec<f64>>>` | `[zone][timestep]` zone temperatures (°C), indexed by zone then timestep |

**Python Example:**
```python
model = Model(num_zones=2)
model.simulate(years=1, use_surrogates=False)
hourly = model.get_hourly_temperatures()
if hourly:
    zone0_temps = hourly[0]  # 8760 hourly values for zone 0
    zone1_temps = hourly[1]  # 8760 hourly values for zone 1
```

**Rust Example:**
```rust
let hourly = model.get_hourly_temperatures();
if let Some(temps) = hourly {
    for (zone_idx, zone_temps) in temps.iter().enumerate() {
        println!("Zone {}: {} timesteps", zone_idx, zone_temps.len());
    }
}
```


### ValidationResult (Issue #761)

The `ValidationResult` struct contains a single validation result for a specific case and metric, including peak load timestamp information.

**Rust Struct Definition:**
```rust
pub struct ValidationResult {
    pub case_id: String,           // Case identifier (e.g., "600", "900", "600FF")
    pub metric: MetricType,        // Metric type
    pub fluxion_value: f64,        // Fluxion simulation value
    pub ref_min: f64,              // Reference minimum value
    pub ref_max: f64,              // Reference maximum value
    pub percent_error: f64,        // Percent error from reference midpoint
    pub status: ValidationStatus,  // Validation status
    pub per_program: Option<HashMap<String, ValidationStatus>>,  // Per-program statuses
    pub peak_date: Option<String>, // Date of peak value occurrence (e.g., "Jan 15")
    pub peak_hour: Option<u32>,    // Hour of peak value occurrence (0-23)
    // Issue #761: ASHRAE 140-2023 Section 8.2.2 - peak timestamp (month, day, hour)
    pub peak_timestamp: Option<(u32, u32, u32)>,
}
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `case_id` | `String` | Case identifier (e.g., "600", "900", "600FF") |
| `metric` | `MetricType` | Metric type (heating, cooling, peak, etc.) |
| `fluxion_value` | `f64` | Fluxion simulation value |
| `ref_min` | `f64` | Reference minimum value |
| `ref_max` | `f64` | Reference maximum value |
| `percent_error` | `f64` | Percent error from reference midpoint |
| `status` | `ValidationStatus` | Validation status (Pass, Warning, Fail) |
| `per_program` | `Option<HashMap<String, ValidationStatus>>` | Per-program validation statuses |
| `peak_date` | `Option<String>` | Date of peak value occurrence (e.g., "Jan 15") for peak metrics |
| `peak_hour` | `Option<u32>` | Hour of peak value occurrence (0-23) for peak metrics |
| `peak_timestamp` | `Option<(u32, u32, u32)>` | Peak timestamp (month, day, hour) for peak metrics, per ASHRAE 140-2023 Section 8.2.2 |

**ASHRAE 140 Section 8 Compliance:**

The `peak_date`, `peak_hour`, and `peak_timestamp` fields capture when peak heating or cooling loads occur, supporting ASHRAE 140 Section 8.2.2 gap analysis for peak load timestamp validation. The `peak_timestamp` field provides the month, day, and hour as a tuple `(month, day, hour)`.

### Incident Solar Radiation per Surface (Issue #762)

Fluxion reports incident solar radiation on a per-surface basis using the `IncidentSolar` metric type. This metric captures the total annual solar radiation incident on each building surface orientation.

```python
# Get incident solar radiation results
results = model.get_validation_results()
for result in results:
    if hasattr(result, 'metric') and result.metric.startswith('IncidentSolar'):
        print(f"{result.metric}: {result.value} kWh/m²")
```

**Data Format:** `IncidentSolar` metric type with fields:
- `surface_id`: Surface identifier (e.g., "roof", "N", "S", "E", "W")
- `orientation`: Surface orientation (North, South, East, West, Roof)
- `value`: Annual incident solar radiation in kWh/m²

| Field | Type | Description |
|-------|------|-------------|
| `surface_id` | `String` | Surface identifier |
| `orientation` | `String` | Surface orientation (N/S/E/W/Roof) |
| `annual_incident_solar` | `f64` | Annual incident solar radiation (kWh/m²) |

**Rust Example:**
```rust
use fluxion::validation::report::{MetricType, ValidationResult};

// IncidentSolar variant carries surface_id and orientation
let metric = MetricType::IncidentSolar {
    surface_id: "roof".to_string(),
    orientation: crate::validation::ashrae_140_cases::Orientation::Roof,
};
let result = ValidationResult::new("600", metric, 180.5, 0.0, 0.0);
println!("{}", result.metric.display_name()); // "Incident Solar Radiation (kWh/m²)"
```

**Metric Type:** `MetricType::IncidentSolar { surface_id, orientation }`

---

## Error Handling

All methods may raise exceptions:

```python
try:
    model = Model("config.json")
except Exception as e:
    print(f"Error: {e}")
```

Common exceptions:
- `FileNotFoundError`: Configuration or model file not found
- `ValueError`: Invalid parameter values
- `RuntimeError`: Surrogate model loading/execution failed
