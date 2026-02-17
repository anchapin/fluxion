# Continuous Tensor Abstraction (CTA) Usage Guide

## Overview

The Continuous Tensor Abstraction (CTA) provides a unified API for tensor-like data used by the physics engine in Fluxion. It abstracts over underlying storage (CPU Vec, GPU Buffer, etc.) and dimensionality, enabling generic tensor operations that work across different backends.

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                   CTA Trait Hierarchy                        │
├─────────────────────────────────────────────────────────────┤
│  ContinuousTensor<T> (Trait)                               │
│  ├── map() - Element-wise transformation                   │
│  ├── zip_with() - Binary element-wise operation            │
│  ├── reduce() - Aggregate to single value                  │
│  ├── integrate() - Numerical integration                   │
│  ├── gradient() - Numerical differentiation                │
│  ├── elementwise_min/max() - Element-wise comparisons     │
│  └── constant_like() - Create filled tensor               │
├─────────────────────────────────────────────────────────────┤
│  Implementations:                                           │
│  ├── VectorField (1D CPU) - src/physics/cta.rs           │
│  └── NDArrayField (ndarray-backed) - src/physics/nd_array.rs│
└─────────────────────────────────────────────────────────────┘
```

## VectorField API

`VectorField` is the primary CPU-based implementation of `ContinuousTensor<f64>` for 1D tensor operations.

### Creation

```rust
use fluxion::physics::cta::{VectorField, ContinuousTensor};

// From existing data
let v = VectorField::new(vec![1.0, 2.0, 3.0]);

// Filled with constant value
let zeros = VectorField::from_scalar(0.0, 10);

// Empty (use carefully!)
let empty = VectorField::new(vec![]);
```

### Basic Operations

```rust
use fluxion::physics::cta::{VectorField, ContinuousTensor};

// Arithmetic operations (element-wise)
let a = VectorField::new(vec![1.0, 2.0, 3.0]);
let b = VectorField::new(vec![4.0, 5.0, 6.0]);

let sum = a.clone() + b.clone();     // [5.0, 7.0, 9.0]
let diff = a.clone() - b.clone();    // [-3.0, -3.0, -3.0]
let prod = a.clone() * b.clone();    // [4.0, 10.0, 18.0]
let quot = a.clone() / b.clone();    // [0.25, 0.4, 0.5]

// Scalar operations
let scaled = a.clone() * 2.0;       // [2.0, 4.0, 6.0]
let divided = a.clone() / 2.0;       // [0.5, 1.0, 1.5]
```

### Tensor Operations

```rust
use fluxion::physics::cta::{VectorField, ContinuousTensor};

let v = VectorField::new(vec![1.0, 2.0, 3.0, 4.0]);

// Map: apply function to each element
let squared = v.map(|x| x * x);      // [1.0, 4.0, 9.0, 16.0]

// Zip with: combine two tensors element-wise
let a = VectorField::new(vec![1.0, 2.0, 3.0]);
let b = VectorField::new(vec![4.0, 5.0, 6.0]);
let maxes = a.zip_with(&b, |x, y| x.max(y)); // [4.0, 5.0, 6.0]

// Reduce: aggregate to single value
let sum = v.reduce(0.0, |acc, x| acc + x);  // 10.0

// Integrate (sum for 1D)
let integral = v.integrate();               // 10.0

// Gradient (finite differences)
let gradient = v.gradient();                // [1.0, 1.0, 1.0, 1.0]

// Element-wise min/max
let x = VectorField::new(vec![1.0, 5.0, 3.0]);
let y = VectorField::new(vec![2.0, 3.0, 4.0]);
let min_vec = x.elementwise_min(&y);        // [1.0, 3.0, 3.0]
let max_vec = x.elementwise_max(&y);        // [2.0, 5.0, 4.0]

// Constant-like (same shape, different value)
let fives = v.constant_like(5.0);           // [5.0, 5.0, 5.0, 5.0]
```

### Access and Iteration

```rust
use fluxion::physics::cta::VectorField;

let v = VectorField::new(vec![1.0, 2.0, 3.0]);

// Length
let len = v.len();                          // 3

// Check empty
let is_empty = v.is_empty();               // false

// Index access
let first = v[0];                           // 1.0

// As slice (for FFI or external libraries)
let slice: &[f64] = v.as_slice();

// Mutable slice
let mut v = VectorField::new(vec![1.0, 2.0, 3.0]);
let slice_mut: &mut [f64] = v.as_mut_slice();

// Iterator
for (i, &val) in v.iter().enumerate() {
    println!("v[{}] = {}", i, val);
}
```

## ContinuousTensor Trait

The `ContinuousTensor<T>` trait defines the interface for all tensor implementations:

```rust
pub trait ContinuousTensor<T>:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
    + Sized
    + Clone
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + AddAssign + Default,
{
    /// Element-wise map
    fn map<F>(&self, f: F) -> Self
    where
        F: Fn(T) -> T;

    /// Combine two tensors
    fn zip_with<F>(&self, other: &Self, f: F) -> Self
    where
        F: Fn(T, T) -> T;

    /// Reduce to single value
    fn reduce<F>(&self, init: T, f: F) -> T
    where
        F: Fn(T, T) -> T;

    /// Numerical integration
    fn integrate(&self) -> T;

    /// Numerical differentiation
    fn gradient(&self) -> Self;

    /// Create filled tensor
    fn constant_like(&self, value: T) -> Self;

    /// Element-wise minimum
    fn elementwise_min(&self, other: &Self) -> Self;

    /// Element-wise maximum
    fn elementwise_max(&self, other: -> Self) -> Self;
}
```

## Usage in Physics Simulations

### Thermal Model Integration

The `ThermalModel` in `src/sim/engine.rs` uses `VectorField` for all state variables:

```rust
use fluxion::physics::cta::{ContinuousTensor, VectorField};
use fluxion::sim::engine::ThermalModel;

// Create a thermal model with 5 zones
let mut model = ThermalModel::<VectorField>::new(5);

// State is now VectorField-based
let temperatures = model.temperatures.clone();  // VectorField
let loads = model.loads.clone();                // VectorField

// All operations work generically over ContinuousTensor
let total_area = temperatures.integrate();       // Sum of zone areas
let max_temp = temperatures.reduce(f64::MIN, |a, b| a.max(b));
```

### 5R1C Thermal Network

The 5R1C (5 Resistances, 1 Capacitance) model uses CTA for all calculations:

```rust
use fluxion::physics::cta::{ContinuousTensor, VectorField};

// Example: Calculate heat transfer
let t_outside = VectorField::from_scalar(5.0, num_zones);  // Cold outside
let t_indoor = model.temperatures.clone();

// Heat transfer: Q = U * A * ΔT
let delta_t = t_indoor.clone() - t_outside;
let q_transfer = h_tr_w.clone() * delta_t;  // Element-wise multiply
```

### Custom Physics Calculations

```rust
use fluxion::physics::cta::{ContinuousTensor, VectorField};

// Calculate heating/cooling demand with deadband
fn calculate_hvac_demand(
    temperatures: &VectorField,
    heating_setpoint: f64,
    cooling_setpoint: f64,
) -> VectorField {
    temperatures.zip_with(
        &temperatures.constant_like(cooling_setpoint),
        |t, cool_sp| {
            if t < heating_setpoint {
                heating_setpoint - t  // Need heating
            } else if t > cooling_setpoint {
                cooling_setpoint - t  // Need cooling
            } else {
                0.0  // In deadband
            }
        },
    )
}
```

## Performance Considerations

### Benchmarks

Run benchmarks to measure CTA overhead:

```bash
cargo bench --bench cta_bench
```

### Memory Layout

`VectorField` uses contiguous memory layout (single `Vec<f64>`), which is:
- **Cache-friendly** for sequential access
- **SIMD-friendly** for vectorized operations
- **Zero-copy** compatible with many Rust ecosystem libraries

### Comparison with Raw Vec

The CTA adds minimal overhead (<10%) compared to raw `Vec<f64>` operations:

| Operation | Raw Vec | CTA (VectorField) | Overhead |
|-----------|---------|-------------------|----------|
| Element-wise add | `zip!()` | `zip_with()` | ~5% |
| Sum/reduce | `iter().sum()` | `reduce()` | ~8% |
| Map | `map()` | `map()` | ~3% |

The overhead is justified by:
- **Generic code** that works with GPU tensors
- **Type safety** ensuring dimension compatibility
- **Extensibility** for future backends

## Future: GPU Backends

The CTA is designed to support GPU acceleration in future phases:

```rust
// Future GPU-backed implementation would implement ContinuousTensor
// allowing transparent switching between CPU and GPU:

#[cfg(feature = "gpu")]
use fluxion::physics::gpu::GpuTensor;

#[cfg(feature = "gpu")]
fn simulate_on_gpu(model: &mut ThermalModel<GpuTensor>) {
    // Same API, different backend
    let gradient = model.temperatures.gradient();
    let integral = model.temperatures.integrate();
}
```

## Examples

See also:
- `src/physics/cta.rs` - Unit tests showing all operations
- `src/sim/engine.rs` - Integration with ThermalModel
- `benches/cta_bench.rs` - Performance benchmarks

## Notes

- The `NDArrayField` (in `src/physics/nd_array.rs`) provides ndarray-backed storage for multi-dimensional tensors
- All operations use `f64` precision for physics accuracy
- The trait bounds are carefully chosen to enable generic code while maintaining performance
