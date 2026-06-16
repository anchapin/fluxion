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
- **Known Limitations:** See `docs/KNOWN_LIMITATIONS.md` for 5R1C model limitations and accuracy constraints

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

**ASHRAE 140 Section 8 Compliance:**

The `peak_date` and `peak_hour` fields capture when peak heating or cooling loads occur, supporting ASHRAE 140 Section 8 gap analysis for peak load timestamp validation.

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
