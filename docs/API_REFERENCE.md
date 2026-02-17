# Fluxion API Reference

## Python API

### Model

The main model class for single-building energy simulation.

```python
from fluxion import Model

# Create a model
model = Model(config_path="config.json")

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
Model(config_path: str) -> Model
```

Create a new Model instance.

**Parameters:**
- `config_path` (str): Path to building configuration JSON file

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

##### load_surrogate()

```python
model.load_surrogate(model_path: str) -> None
```

Register an ONNX surrogate model.

**Parameters:**
- `model_path` (str): Path to ONNX model file

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

**Returns:** List of EUI values (kWh/m²/year)

##### load_surrogate()

```python
oracle.load_surrogate(model_path: str) -> None
```

Load an ONNX surrogate model.

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
