# fluxion-grid

Grid-edge electrical network components for [Fluxion](https://github.com/anchapin/fluxion), the Rust-based building energy modeling engine.

## What it does

`fluxion-grid` models the electrical side of a building energy simulation: battery storage, bus nodes, power-flow solvers, and the joint thermal–electrical convergence that couples the grid back to the thermal model. It is an always-built sibling of the root `fluxion` crate.

The optional `fluxion-integration` feature wires `ThermalElectricalCoupler` to a caller-provided thermal model via the grid-side `ThermalModelQuery` trait, so the grid and the thermal solver can converge on a single solution instead of running decoupled.

## Consumers

- **Main `fluxion` crate** (Issue #4005, default-off `grid` feature): `crate::sim::grid_adapter::GridAdapter` wraps `fluxion_grid::{ThermalElectricalCoupler, PvSystem, BatteryStorage}` in a per-timestep `step()` API — thermal state in, electrical state out — with the batch `post_process` built on top of it. Phase 1 is post-processing only (an additive `ElectricalResults` block); the `step()` shape keeps phase-2 in-loop co-simulation (demand-response / pre-cooling controllers) a wiring change, not a rewrite. Example: `cargo run --example grid_coupling_demo --features grid`.
- **Standalone**: pure electrical-network work (batteries, bus nodes, power flow) needs no thermal solver stack at all.

**Dependency direction**: strictly `fluxion` → `fluxion-grid`, never the reverse (Cargo rejects optional-optional package cycles, so the crate has no dependency edge back into the main crate).

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
