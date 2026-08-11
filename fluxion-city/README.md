# fluxion-city

Urban-scale radiation modeling for [Fluxion](https://github.com/anchapin/fluxion), the Rust-based building energy modeling engine.

## What it does

`fluxion-city` computes inter-building radiative exchange using a Nusselt-analog view-factor formulation, so a building's thermal model can account for longwave and shortwave radiation from neighboring buildings rather than treating it as isolated. It is designed to plug into the urban context layer of a Fluxion simulation.

This is a **feature-gated** workspace sibling: it is pulled into the main `fluxion` engine only when the `fluxion-city` feature is enabled (issue #2344).

## Build / test

```bash
# Build this crate directly
cargo build -p fluxion-city
cargo test  -p fluxion-city

# Or enable it in the main fluxion engine
cargo build --features fluxion-city
cargo test  --features fluxion-city
```

## See also

- [Top-level README](../README.md) — project overview and quickstart
- [ARCHITECTURE.md](../ARCHITECTURE.md) — module boundaries and data flow
- [AGENTS.md](../AGENTS.md) — feature-flag reference (Feature flags section)

## License

Apache-2.0
