# fluxion-fluid

Compile-time strongly typed fluid port traits for [Fluxion](https://github.com/anchapin/fluxion), the Rust-based building energy modeling engine.

## What it does

`fluxion-fluid` provides the acausal HVAC / fluid-system port traits used to assemble plant loops, air loops, and other DAE (differential-algebraic equation) systems in Fluxion (ADR-005, issues #1980). It exposes strongly typed ports (`ports/`), a graph layer for connecting them (`graph/`), and solvers (`solvers/`) — including a WASM-compatible sequential fallback that does not require `rayon`.

> **Not to be confused:** this crate is *not* the same as `fluxion-core/src/fluid/`. This `fluxion-fluid` crate is the feature-gated, acausal-HVAC port-trait layer; `fluxion-core/src/fluid/` is a different, lighter in-core module.

This is a **feature-gated** workspace sibling: it is pulled into the main `fluxion` engine only when the `fluid` feature is enabled. `fluxion-wasm` and `fluxion-mcp` both depend on it unconditionally.

## Build / test

```bash
# Build this crate directly
cargo build -p fluxion-fluid
cargo test  -p fluxion-fluid

# Or enable it in the main fluxion engine
cargo build --features fluid
cargo test  --features fluid
```

## WASM compatibility

See [WASM_STATUS.md](./WASM_STATUS.md) for the dependency compatibility matrix; the `ports/`, `graph/`, and `solvers/` layers are WASM-compatible.

## See also

- [Top-level README](../README.md) — project overview and quickstart
- [ARCHITECTURE.md](../ARCHITECTURE.md) — module boundaries and ADR-005
- [AGENTS.md](../AGENTS.md) — workspace structure and feature-flag reference

## License

Apache-2.0
