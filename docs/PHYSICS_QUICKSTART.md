# Physics Contributor Quickstart

A human building scientist's path from fresh checkout to a running ASHRAE 140 case, using only Rust tooling. No AI agents, no Python, no `ctx_execute` — just `cargo` and the `fluxion` CLI.
Covers: Rust prerequisites, building the CLI, running Case 600, reading the validation table, where the physics code lives, and which of the 14 required CI checks actually apply to a physics-only PR.
Status: Active.
Action: Run `cargo run --bin fluxion -- validate --case 600` to see the engine validate itself in ~6 seconds.

## Who This Is For

You know building physics — conduction, solar gains, infiltration, setpoints — and you want to change Fluxion's thermal model and see what happens to the ASHRAE 140 results. This guide assumes comfort with a terminal and Rust basics, and nothing else. (For the Python `BatchOracle` API or the REST server, see [QUICKSTART.md](QUICKSTART.md). For AI-agent workflows, see [CONTRIBUTING.md](../CONTRIBUTING.md).)

## Prerequisites

- **Rust** (stable) via [rustup](https://rustup.rs). The repo pins its minimum version in `Cargo.toml` (`rust-version`); rustup's latest stable is fine.
- ~10 GB free disk (debug builds of the workspace are large).
- No Python, no Node, no API keys needed for anything below.

## Checkout and Build

```bash
git clone https://github.com/anchapin/fluxion.git && cd fluxion
git checkout develop          # all work targets develop; main is release-only
cargo build --bin fluxion     # builds just the CLI (~2-4 min first time)
```

## Run One ASHRAE Case

Case 600 is the low-mass baseline — the simplest BESTEST envelope. This runs the validation harness and prints a results table:

```bash
cargo run --bin fluxion -- validate --case 600
```

Takes about 6 seconds. You will see a markdown table with one row per case:

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 600 | 4604.57 kWh (Ref: 4360.00-5790.00) | … | … | … | ❌ FAIL |

### Reading the Table

- Each metric shows Fluxion's value against the ASHRAE 140 reference **range** in parentheses. Inside the range = pass.
- **FAIL does not mean you broke something.** The engine is mid-development: as of this writing only a fraction of the 84 metric results pass. Failing metrics are tracked as known structural gaps in [KNOWN_ISSUES.md](KNOWN_ISSUES.md) (e.g. `SOLAR-02`, `LIMIT-05`) — check there before assuming a regression.
- To confirm your change didn't regress anything, run the same command before and after and diff the table. The numbers should be identical unless you intended to move them.
- Full-suite results and methodology live in [ASHRAE140_RESULTS.md](ASHRAE140_RESULTS.md).

## Where the Physics Lives

- `src/physics/` — conduction (CTF/FD), solar gains, longwave exchange, infiltration. Start here.
- `src/sim/` — the zone solver that wires physics modules together (`thermal_model.rs`).
- `ARCHITECTURE.md` (repo root) — the source of truth for module boundaries and swap-point traits. Read the relevant section before changing an interface.
- `assets/weather/` — the EPW weather files the cases run against (`WD600.epw` for Case 600).

A typical physics change: edit a correlation in `src/physics/`, rebuild, re-run `validate --case 600`, compare the table.

## CI: Which Checks Apply to You

A physics-only PR must satisfy the required checks in `release_gates.yaml` → `ci.required_checks` (14 as of 2026-09-25). In practice, only these can fail on a pure physics change:

| Check | What it means for you | Local equivalent |
|-------|----------------------|------------------|
| Workspace Check (GH) | your code compiles across the whole workspace | `cargo check --workspace` |
| Rustfmt (GH) | formatting | `cargo fmt -- --check` |
| Clippy (GH) | lints, deny warnings | `cargo clippy --lib -- -D warnings` |
| Energy Conservation (GH) | the zone energy-balance invariant still holds | runs in the test suite |
| Physics-Sim-Cycle-Check (GH) | no new `use` cycles between `src/physics/` and `src/sim/` | — |
| Ashrae Cases Cycle Check (GH) | no new `sim` ↔ validation cycles | — |
| Cycle Downward Trend Guard | the above cycle counts don't grow | — |
| Architecture Drift Detection | if you add/change a swap-point trait, document it in `ARCHITECTURE.md` | `python3 scripts/check_architecture_drift.py` |

You can safely ignore the rest unless you touch their area:

- **Surrogate Drift Tolerance Gate** — only if you modify surrogate code paths.
- **Docs Hygiene Gate** — only if you edit `.md` files (requires the 7-line summary convention at the top of new docs).
- **Module Size / Crate Size Gate** — only for large new modules.
- **Cargo Deny / MSRV Check** — only when changing dependencies or using new Rust features.

The heavy gates (cross-platform determinism matrix, performance gates, coverage) run nightly on `develop`, not on your PR — a red nightly freezes merges until fixed, but it won't block your PR's checks list.

## Quick Checklist Before Opening Your PR

```bash
cargo fmt -- --check
cargo clippy --lib -- -D warnings
cargo check --workspace
cargo run --bin fluxion -- validate --case 600   # eyeball the table vs. develop
```

Then `gh pr create --base develop`. CI handles the rest.
