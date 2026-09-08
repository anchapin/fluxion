# fluxion-core

Dependency-light *leaf* modules for [Fluxion](https://github.com/anchapin/fluxion), the Rust-based building energy modeling engine.

## What it does

`fluxion-core` holds the foundational, allocation-light modules shared across the Fluxion workspace: weather/TMY parsing (`weather/`), envelope assembly and material definitions (`assembly.rs`, `construction.rs`), multi-node and per-surface conduction primitives (`multi_node.rs`, `per_surface_conduction.rs`), ASHRAE 140 reference cases (`ashrae_cases.rs`), and physics constants (`physics_constants.rs`).

It is intentionally kept free of dependencies on `sim/`, `physics/`, `ai/`, and `validation/` so it can be built once and cached by `cargo-mutants` in CI (issue #1255) without pulling in the heavier engine. The cycle-breaking rule is enforced by `scripts/check_ashrae_cases_cycle.py` (#1441) and `fluxion-core/tests/boundary_enforcement.rs` (#3168).

## Dependency budget (Issue #3467)

The default `cargo build -p fluxion-core` pulls in only:

| Crate        | Why it's allowed                                    |
|--------------|-----------------------------------------------------|
| `num-traits` | `Float` arithmetic traits for psychrometrics        |
| `serde`      | `Serialize`/`Deserialize` derives on data structs   |
| `serde_json` | Weather-record JSON, hash digests, etc.             |
| `serde_yaml` | ASHRAE 140 assembly / materials YAML inputs         |
| `thiserror`  | `WeatherError` / `CarbonError` / assembly errors    |
| `log`        | Logging in `method_selector.rs` and weather modules |

The TMY3 network-download / on-disk-cache path (NREL weather, SHA-256
verification) lives behind the opt-in `tmy3-download` cargo feature
and pulls in `reqwest` + `directories` + `sha2`. The regression gate
`scripts/check_fluxion_core_dep_budget.py` fails CI if a heavyweight
crate is reintroduced to the default build.

## Build / test

```bash
# Default — leaf is built without reqwest / hyper / tokio / rustls
cargo build -p fluxion-core
cargo test  -p fluxion-core

# Opt in to the TMY3 download/cache module + its network deps
cargo build -p fluxion-core --features tmy3-download
cargo test  -p fluxion-core --features tmy3-download
```

## See also

- [Top-level README](../README.md) — project overview and quickstart
- [ARCHITECTURE.md](../ARCHITECTURE.md) — module boundaries, trait contracts, dependency budget
- [CODEBASE_MAP.md](../CODEBASE_MAP.md) — `fluxion-core` crate navigation
- [AGENTS.md](../AGENTS.md) — workspace structure and cycle-breaking rules
- [`scripts/check_fluxion_core_dep_budget.py`](../scripts/check_fluxion_core_dep_budget.py) — Issue #3467 regression gate

## License

Apache-2.0
