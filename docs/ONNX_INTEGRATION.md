# ONNX Model Integration Guide

This document explains how ONNX Runtime integration works in Fluxion, how to use surrogate models for fast inference, and how to validate surrogate predictions against analytical physics.

## Architecture Overview

### Why ONNX?

Fluxion uses **ONNX Runtime** to load and execute pre-trained neural network models for fast thermal load prediction. This enables:

- **100x speedup**: Neural networks replace expensive physics calculations (CFD, ray-tracing)
- **Hardware acceleration**: ONNX Runtime can offload to GPU/TPU if available
- **Portability**: ONNX is a standard format; models trained in PyTorch, TensorFlow, or other frameworks can be used
- **Production-ready**: Mature, stable inference engine used by enterprises

### Surrogate Manager Architecture

```
┌─────────────────────────────────────────────┐
│        SurrogateManager                     │
│  ┌───────────────────────────────────────┐ │
│  │ model_loaded: bool                    │ │
│  │ model_path: Option<String>            │ │
│  │ session: Option<Arc<Mutex<Session>>>  │ │
│  └───────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
         │
         ├─→ new() → Create empty manager (returns mock loads 1.2 W/m²)
         ├─→ load_onnx(path) → Load pre-trained model
         └─→ predict_loads(&temps) → Inference
              ├─ If model loaded: Call ONNX Runtime
              ├─ On inference error: Fallback to mock
              └─ Return Vec<f64> loads (one per zone)
```

**Key Design Decisions:**

- **Arc<Mutex<_>>**: Session is wrapped for thread-safe sharing across rayon parallel workers
- **Graceful degradation**: If ONNX inference fails, returns mock loads (1.2 W/m²) instead of panicking
- **Zero-copy input**: Uses ndarray for efficient tensor conversion

## Using Surrogate Models

### 1. Generating a Dummy Model (for Testing)

A simple dummy model is included for testing and CI:

```bash
# Generate a model that returns constant loads (1.2 W/m² per zone)
python3 tools/generate_dummy_surrogate.py --zones 10 --out assets/loads_predictor.onnx
```

This creates a minimal ONNX model with:
- **Input**: `[zones]` float32 array (current temperatures)
- **Output**: `[zones]` float32 array (constant 1.2 repeated)
- **Size**: ~168 bytes (trivial for demonstration)

### 2. Loading a Model (Rust)

```rust
use fluxion::ai::surrogate::SurrogateManager;

// Load model
let manager = SurrogateManager::load_onnx("path/to/model.onnx")?;

// Use in ThermalModel
let mut model = ThermalModel::new(10);
let energy = model.solve_timesteps(8760, &manager, use_ai=true);
```

If the model file doesn't exist or libonnxruntime is not installed, inference gracefully degrades to mock loads.

### 3. Loading a Model (Python/BatchOracle)

```python
import fluxion

oracle = fluxion.BatchOracle()
oracle.load_surrogate("assets/loads_predictor.onnx")

# Now evaluate_population uses ONNX inference
results = oracle.evaluate_population(population, use_surrogates=True)
```

## Training & Exporting Custom Models

### Expected Input/Output Specification

For a Fluxion model, the ONNX graph should have:

| Aspect | Specification |
|--------|---------------|
| **Input name** | `"input"` (or any name; Fluxion queries from model metadata) |
| **Input shape** | `[num_zones]` or `[1, num_zones]` |
| **Input dtype** | `float32` |
| **Output name** | `"output"` (or any name; Fluxion uses first output) |
| **Output shape** | `[num_zones]` (same as input) |
| **Output dtype** | `float32` |
| **Semantics** | Thermal loads in W/m² for each zone |

### Example: PyTorch to ONNX

```python
import torch
import torch.nn as nn

class LoadsPredictor(nn.Module):
    def __init__(self, num_zones: int):
        super().__init__()
        self.num_zones = num_zones
        self.fc1 = nn.Linear(num_zones, 64)
        self.fc2 = nn.Linear(64, num_zones)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model
model = LoadsPredictor(10)

# Export to ONNX
dummy_input = torch.randn(10)
torch.onnx.export(
    model,
    dummy_input,
    "loads_predictor.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=12
)
```

### Example: TensorFlow/Keras to ONNX

```python
import tensorflow as tf
import tf2onnx

# Build Keras model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Convert to ONNX
spec = (tf.TensorSpec((None, 10), tf.float32, name="input"),)
output_path = "loads_predictor.onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
```

## Validation Framework

### Comparing Surrogate vs Analytical

Fluxion provides infrastructure to validate surrogate predictions:

```rust
#[test]
fn test_surrogate_vs_analytical_consistency() {
    let mut model_analytical = ThermalModel::new(10);
    let mut model_surrogate = ThermalModel::new(10);
    let surrogates = SurrogateManager::load_onnx("path/to/model.onnx")?;

    let params = vec![1.5, 21.0];
    model_analytical.apply_parameters(&params);
    model_surrogate.apply_parameters(&params);

    let energy_analytical = model_analytical.solve_timesteps(8760, &surrogates, false);
    let energy_surrogate = model_surrogate.solve_timesteps(8760, &surrogates, true);

    // Compare results and compute error metrics
    let mae = (energy_analytical - energy_surrogate).abs();
    println!("MAE: {}", mae);
}
```

### Python Validation Script

See `examples/validate_surrogate.py` for a complete example that:
1. Loads a surrogate model
2. Evaluates a population analytically
3. Evaluates the same population using surrogates
4. Computes error metrics (MAE, RMSE, max error)
5. Prints side-by-side comparisons

Run with:
```bash
maturin develop
python3 examples/validate_surrogate.py
```

## Performance Considerations

### Throughput Targets

- **Analytical**: ~1 config/ms (slow; full physics)
- **Surrogate**: ~100 configs/ms (100x speedup)
- **Target population**: 10,000+ configs in <100ms

### Optimization Tips

1. **Batch multiple zones/candidates** if possible (vectorization)
2. **Use release builds**: `cargo build --release`
3. **Profile with**: `cargo flamegraph --bin fluxion`
4. **Disable copy-dylibs if libonnxruntime.so is system-installed**

## Troubleshooting

### libonnxruntime not found

**Error:**
```
dlopen(libonnxruntime.dylib, 0x0005): (no such file)
```

**Solution:**
1. Install ONNX Runtime:
   ```bash
   # macOS
   brew install onnxruntime

   # Linux
   sudo apt install libonnxruntime-dev
   ```

2. Or disable ONNX for testing (Fluxion gracefully degrades):
   - Tests will use mock loads
   - Add `#[ignore]` to ONNX-specific tests if needed

### Model shape mismatch

**Error:**
```
ONNX inference error: Input 0 has shape mismatch
```

**Solution:**
- Verify model input shape matches `num_zones`
- Check if model expects batch dimension: `[1, num_zones]` vs `[num_zones]`
- Update `predict_loads` input conversion if needed:
  ```rust
  let input_arr = input_arr.insert_axis(ndarray::Axis(0)); // Add batch dim
  ```

### Output dtype mismatch

**Error:**
```
Failed to extract tensor: output is float64, expected float32
```

**Solution:**
- Ensure ONNX model exports float32, not float64
- In PyTorch: add `.float()` before export
- In TensorFlow: use `dtype=tf.float32` in model

## Future Directions

### Phase 4: Real Models

1. **Train surrogate on simulation data**
   - Use analytical engine to generate synthetic training data
   - Train neural network to predict loads from temperatures
   - Export to ONNX

2. **Calibrate against field data**
   - Compare predictions to actual building energy data
   - Adjust model weights or add physics constraints

3. **Hardware acceleration**
   - Enable GPU inference (CUDA/Metal)
   - Benchmark on inference accelerators

### Phase 5: Advanced Inference

1. **Uncertainty quantification**
   - Output confidence bounds with predictions
   - Ensemble methods for robustness

2. **Online learning**
   - Adapt model weights during optimization
   - Feedback loops for continuous improvement

## References

- **ONNX Specification**: https://github.com/onnx/onnx/blob/main/docs/IR.md
- **ONNX Runtime**: https://onnxruntime.ai/
- **ORT Rust Bindings**: https://crates.io/crates/ort (v2.0.0-rc.10)
- **Example Models**: https://github.com/onnx/models
