CTA Usage

Overview

The Continuous Tensor Abstraction (CTA) provides a unified API for tensor-like data used by the physics engine.

Examples

- VectorField (1D CPU) - src/physics/cta.rs
- NDArrayField (ndarray-backed) - src/physics/nd_array.rs

Basic usage:

```rust
use fluxion::physics::cta::VectorField;
let v = VectorField::new(vec![1.0, 2.0, 3.0]);
let g = v.gradient();
let integral = v.integrate();
```

Benchmarks

Run benches with:

cargo bench --bench cta_bench

Notes

The NDArrayField is suitable for future N-dimensional backends and GPU-backed implementations.
