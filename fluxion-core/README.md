# fluxion-core

Dependency-light *leaf* modules for [Fluxion](https://github.com/anchapin/fluxion), the Rust-based building energy modeling engine.

## What it does

`fluxion-core` holds the foundational, allocation-light modules shared across the Fluxion workspace: weather/TMY parsing (`weather/`), envelope assembly and material definitions (`assembly.rs`, `construction.rs`), multi-node and per-surface conduction primitives (`multi_node.rs`, `per_surface_conduction.rs`), ASHRAE 140 reference cases (`ashrae_cases.rs`), and physics constants (`physics_constants.rs`).

It is intentionally kept free of dependencies on `sim/`, `physics/`, `ai/`, and `validation/` so it can be built once and cached by `cargo-mutants` in CI (issue #1255) without pulling in the heavier engine. The cycle-breaking rule is enforced by `scripts/check_ashrae_cases_cycle.py` (#1441).

## Build / test

```bash
cargo build -p fluxion-core
cargo test  -p fluxion-core
```

## See also

- [Top-level README](../README.md) — project overview and quickstart
- [ARCHITECTURE.md](../ARCHITECTURE.md) — module boundaries and trait contracts
- [AGENTS.md](../AGENTS.md) — workspace structure and cycle-breaking rules

## License

Apache-2.0
