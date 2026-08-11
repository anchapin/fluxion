# fluxion-grid

Grid-edge electrical network components for [Fluxion](https://github.com/anchapin/fluxion), the Rust-based building energy modeling engine.

## What it does

`fluxion-grid` models the electrical side of a building energy simulation: battery storage, bus nodes, power-flow solvers, and the joint thermal–electrical convergence that couples the grid back to the thermal model. It is an always-built sibling of the root `fluxion` crate.

The optional `fluxion-integration` feature wires `ThermalElectricalCoupler` to the main crate's `ThermalModelTrait` so the grid and the thermal solver can converge on a single solution instead of running decoupled.

## Build / test

```bash
cargo build -p fluxion-grid
cargo test  -p fluxion-grid
```

## See also

- [Top-level README](../README.md) — project overview and quickstart
- [ARCHITECTURE.md](../ARCHITECTURE.md) — module boundaries and data flow
- [AGENTS.md](../AGENTS.md) — workspace structure

## License

Apache-2.0
