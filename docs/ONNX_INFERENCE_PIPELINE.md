# ONNX Inference Pipeline

## Overview

The surrogate module provides ONNX Runtime-based inference for thermal load predictions. When an ONNX model is loaded, the `SurrogateManager` uses the `ort` crate (ONNX Runtime Rust bindings) for efficient GPU/CPU inference.

## ONNX Model Format Requirements

### Input/Output Specification

The ONNX model must conform to the following I/O specification for use with the SurrogateManager:

#### Input Tensor
- **Name**: `X`
- **Shape**: `[batch_size, num_features]` or `[1, num_features]` for single inference
- **Data Type**: Float32

#### Output Tensor
- **Name**: `Y`
- **Shape**: Same as input shape (model produces load predictions per input point)
- **Data Type**: Float32

### Example Model Creation (Python)

```python
import onnx
from onnx import helper, TensorProto

# Input: 2 features (exterior_temp, zone_temp) per sample
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [None, 2])

# Output: 2 load predictions per sample
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [None, 2])

# Weight matrix (identity-like transformation)
W = helper.make_tensor('W', TensorProto.FLOAT, [2, 2], [1.0, 0.0, 0.0, 1.0])
B = helper.make_tensor('B', TensorProto.FLOAT, [2], [0.0, 0.0])

# Simple pass-through model
matmul = helper.make_node('MatMul', ['X', 'W'], ['XM'])
add = helper.make_node('Add', ['XM', 'B'], ['Y'])

graph = helper.make_model([matmul, add], 'thermal_surrogate', [X], [Y], [W, B])
onnx.save(graph, 'thermal_surrogate.onnx')
```

### Compatible Model Architectures

- **Linear models**: Simple matrix multiplication + bias
- **MLP networks**: Multi-layer perceptrons with ReLU/sigmoid activations
- **Custom trained models**: Any ONNX-compliant model with correct I/O names

## Loading Models

```rust
use fluxion::ai::surrogate::{SurrogateManager, InferenceBackend};

// Load with CPU backend
let manager = SurrogateManager::load_onnx("path/to/model.onnx")?;

// Load with GPU backend (CUDA)
let manager = SurrogateManager::with_gpu_backend(
    "path/to/model.onnx",
    InferenceBackend::CUDA,
    0, // device_id
)?;

// Load with multi-device support
let config = MultiDeviceConfig::single_gpu(0);
let manager = SurrogateManager::with_multi_device("path/to/model.onnx", config)?;
```

## Inference Methods

### Single Inference
```rust
let temps = vec![20.0, 22.0];  // [exterior_temp, zone_temp]
let loads = manager.predict_loads(&temps);
```

### Batch Inference
```rust
let batch = vec![
    vec![20.0, 22.0],  // sample 1
    vec![21.0, 23.0],  // sample 2
];
let loads = manager.predict_loads_batched(&batch);
```

### With Fallback (Recommended)
```rust
let loads = manager.predict_loads_with_fallback(&temps)?;
```

### Governed Inference (Domain-Aware)
```rust
let domain = SurrogateDomain::default_residential();
let loads = manager.predict_loads_governed(&temps, &domain, SurrogateMode::NeuralWithFallback)?;
```

## Inference Backends

| Backend | Feature Flag | Description |
|---------|--------------|-------------|
| CPU | Default | Standard CPU inference |
| CUDA | `cuda` | NVIDIA GPU acceleration |
| CoreML | N/A | Apple Silicon acceleration (macOS) |
| DirectML | N/A | DirectX GPU acceleration (Windows) |
| OpenVINO | N/A | Intel GPU/CPU acceleration |

## Session Pooling

The `SessionPool` provides thread-safe session management for concurrent inference:

- Sessions are pooled and reused across calls
- Thread-safe acquisition/release via mutex
- Automatic session creation when pool is empty

## Fallback Behavior

When no ONNX model is loaded, the surrogate uses an analytical model based on:

- Time of day (daily solar cycle)
- Exterior temperature
- Zone temperature

This ensures predictions are always available, even without a trained model.

## Testing

Run surrogate tests:
```bash
cargo test ai::surrogate --lib
```

Run all AI module tests:
```bash
cargo test ai:: --lib
```

## Dependencies

```toml
ort = { version = "2.0.0-rc.10", features = ["download-binaries"] }
```

Optional features:
- `cuda` - NVIDIA GPU support
- `tensorrt` - NVIDIA TensorRT acceleration
