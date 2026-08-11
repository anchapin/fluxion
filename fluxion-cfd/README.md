# fluxion-cfd

GPU-accelerated Fast Fluid Dynamics (FFD) airflow solver for [Fluxion](https://github.com/anchapin/fluxion), the Rust-based building energy modeling engine.

## What it does

`fluxion-cfd` provides the indoor airflow / CFD co-simulation path for Fluxion. It solves the fast-fluid-dynamics approximation of the Navier–Stokes equations to produce room-scale air-temperature and velocity fields that feed back into the zone thermal model. Backend selection is controlled by the crate's own `cpu` / `cuda` / `opencl` features (default `cpu`).

This is a **feature-gated** workspace sibling: it is pulled into the main `fluxion` engine only when the `fluxion-cfd` feature is enabled.

## Build / test

```bash
# Build this crate directly
cargo build -p fluxion-cfd
cargo test  -p fluxion-cfd

# Or enable it in the main fluxion engine
cargo build --features fluxion-cfd
cargo test  --features fluxion-cfd
```

## See also

- [Top-level README](../README.md) — project overview and quickstart
- [ARCHITECTURE.md](../ARCHITECTURE.md) — module boundaries and co-simulation data flow
- [AGENTS.md](../AGENTS.md) — feature-flag reference (Feature flags section)

## License

Apache-2.0
