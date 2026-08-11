# fluxion-behavior

Behavioral and thermal-comfort models for [Fluxion](https://github.com/anchapin/fluxion), the Rust-based building energy modeling engine.

## What it does

`fluxion-behavior` implements occupant-side thermal comfort and behavior: Fanger PMV/PPD, adaptive comfort models, and the stochastic occupant triggers (window opening, shading, setpoint adjustments) that feed back into the zone energy balance. It is an always-built sibling of the root `fluxion` crate.

Comfort and behavior outputs are consumed by the zone solver to close the loop between indoor conditions and the actions real occupants take.

## Build / test

```bash
cargo build -p fluxion-behavior
cargo test  -p fluxion-behavior
```

## See also

- [Top-level README](../README.md) — project overview and quickstart
- [ARCHITECTURE.md](../ARCHITECTURE.md) — module boundaries and data flow
- [AGENTS.md](../AGENTS.md) — workspace structure

## License

Apache-2.0
