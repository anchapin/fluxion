# fluxion-grid Integration Design — Issue #4005

Design scoping pass for integrating `fluxion-grid` into a real consumer, per Alex's 2026-09-26 decision: **integrate; do not extract**. This doc proposes how. No integration code is written here — Alex picks the design first.

## 1. What `fluxion-grid` actually is

`fluxion-grid` is a standalone, dependency-light Rust library for grid-edge electrical modeling of buildings. It provides: battery storage models (`battery.rs`, `battery_storage.rs`, `battery_storage_node.rs` — SoC tracking, C-rate discharge, internal-resistance terminal voltage, DoD-weighted degradation per #2037, plus an Arrhenius/calendar-aging model), PV panels and simple inverters (`pv.rs`, EnergyPlus `Generator:Photovoltaic` / `Inverter:Simple` mappings, NREL SAM-aligned), AC power-flow (Newton-Raphson solver, buses, transmission lines, `GridConvergenceReport`), and a `ThermalElectricalCoupler` that converts thermal/HVAC loads to electrical loads via COP — including optional `fluid`-feature coupling to `fluxion_fluid::hvac::{HvacState, HvacMode}` and an optional `fluxion-integration` feature holding `Arc<dyn ThermalModelTrait>` for joint thermal-electrical convergence. Test coverage is solid for a library nobody calls: ~116 inline unit tests plus `tests/grid_solver_integration.rs` (Issue #2908 — 3-node NR round-trip, 10-heat-pump case, IEEE 33-bus feeder co-simulation), all run in CI via the `fluxion-grid-integration-gh/hz` jobs in `rust-tests.yml` (demoted to advisory 2026-09-25). In practice it is both a component library by design and a standalone simulator by accident: it has **zero workspace consumers** — nothing outside its own tests calls any `fluxion_grid` type.

## 2. Git archaeology and stated integration intent

The crate predates the 2026-09-10 monorepo import (commit `3f89f48` squashed the whole tree, so fine-grained history is unavailable, but the crate already carried issue references). Integration was *anticipated* but never *completed*:

- **Issue #2275** — the `fluxion-integration` feature: `fluxion_bridge::ThermalModelTraitBridge` holds `Arc<dyn ThermalModelTrait>` for joint convergence with the full thermal solver. The bridge exists; nothing constructs it.
- **Issue #2561** — the `fluid` feature: decoupled the default build from `fluxion-fluid` while keeping `hvac_state_to_electrical` / `thermal_to_electrical` available behind the feature. Nothing enables it.
- **Issue #2908** — integration tests proving the public API from outside the crate.
- **ARCHITECTURE.md §"Module N+1"** documents the intended wiring, including a joint-convergence pattern snippet.
- The main crate's `Cargo.toml` carries a stub `fluxion = []` feature "to allow fluxion-grid to depend on this crate" — evidence the dependency direction was designed for, then never used.

## 3. Candidate consumers (ranked)

1. **Main `fluxion` crate — new optional `grid` feature + `src/sim/grid_adapter.rs`** (best fit). The engine (`src/sim/engine.rs`) already accumulates per-timestep thermal energy (`total_energy_kwh`); a post-processing adapter converts thermal loads → electrical demand via `ThermalElectricalCoupler` (COP-based), adds on-site PV generation and a battery dispatch simulation (self-consumption / peak-shaving), and extends the results block. Direct precedent: `fluxion-cfd = ["dep:fluxion-cfd"]` + `FfdCfdAdapter` (Issue #2460, ARCHITECTURE.md §N+2). Also resolves the duplication in §4: the dead `src/solar/pv.rs` can be deleted and re-exported from `fluxion-grid`.
2. **`fluxion-rest` binary (`src/bin/fluxion_rest.rs`)** — the axum REST server's `/v1/simulate` route. Once the engine exposes an electrical block (via candidate 1), the REST layer can serve it (a `/v1/grid` dispatch endpoint or an extended simulate response). Real consumer, but downstream of candidate 1 — it consumes the adapter's output, not the crate directly.
3. **`fluxion-examples` crate (`examples/`)** — a `grid_coupling_demo.rs` example wiring PV + battery + thermal-electrical coupling end-to-end. Cheapest real consumer and good documentation, but an example alone is the weakest reading of "wire it into a real consumer."
4. **`fluxion-twin` (MQTT/REST digital twin)** — battery dispatch responding to grid signals is a classic twin use case, and the twin has MQTT telemetry surfaces. Ranked last: the twin is a UKF thermal state estimator; adding electrical state to the state vector is a genuine design change to twin semantics, not a thin adapter.

## 4. Non-goals check: duplication

- **`src/solar/pv.rs` (main crate) is a near-verbatim duplicate of `fluxion-grid/src/pv.rs`** — same EnergyPlus mappings (`Generator:Photovoltaic` → `PvPanel`, `Inverter:Simple` → `SimpleInverter`), same struct fields, same NREL SAM alignment notes. It is **completely unconsumed**: the only reference is the re-export in `src/solar/mod.rs`; `src/sim/solar.rs` uses the irradiance parts of `crate::solar`, not the PV types. Integration must delete this file and re-export from `fluxion-grid` (or from the new adapter module).
- **No battery model exists in the main crate** (the one "battery" hit in `src/physics/mod.rs` is a comment about "evolution fitness battery" test fixtures). No power-flow anywhere else. Integration is not redundant.

## 5. Integration options

### Option A — `grid` feature + engine adapter + example (RECOMMENDED)

- **What gets built:**
  1. `grid = ["dep:fluxion-grid"]` feature in the main crate's `Cargo.toml` (mirrors the `fluxion-cfd` precedent; default off so the default build and physics-core semantics are untouched).
  2. `src/sim/grid_adapter.rs` (feature-gated): `GridAdapter` takes the engine's per-timestep thermal results, converts thermal→electrical via `fluxion_grid::ThermalElectricalCoupler`, runs `PvPanel`/`SimpleInverter` generation and a `BatteryStorage` dispatch policy, and returns an `ElectricalResults` block appended to the engine's output struct (new fields only — no existing field changes).
  3. Delete `src/solar/pv.rs`; re-export `fluxion_grid::{PvPanel, PvSystem, SimpleInverter}` from `src/solar/mod.rs` so the one existing import path keeps working.
  4. `examples/grid_coupling_demo.rs` in `fluxion-examples`: end-to-end PV + battery + coupling demo with `--features grid`.
  5. Feature-matrix CI coverage for `--features grid` in the existing `rust-tests.yml` matrix (no new workflow, no new required-check name).
- **Touch surface:** main crate `Cargo.toml` (1 feature line + 1 optional dep), 1 new adapter module (~300 lines), 1 deleted file, 1 re-export edit, 1 example, CI matrix row, docs (§6 below).
- **Test plan:** unit tests for the adapter's thermal→electrical conversions against hand-computed values; an integration test running a short annual simulation with/without the feature asserting the physics results are bit-identical and the electrical block matches a recorded baseline; the existing `fluxion-grid` self-tests and the IEEE 33-bus suite unchanged; determinism gate unaffected (feature off in the determinism job).
- **"Documented" concretely:** ARCHITECTURE.md Module N+1 rewritten from "standalone crate" to the wired design with a data-flow diagram; `fluxion-grid/README.md` gains a "Consumers" section; `docs/FEATURES.md` gains the `grid` feature row; the feature flag gets a doc comment citing #4005; the example is the executable documentation.

### Option B — `fluxion grid` CLI subcommand

A `grid` subcommand on `src/bin/fluxion.rs` that runs a standalone PV + battery dispatch scenario from a config file (no engine coupling). Cheaper (~200 lines, no engine touch) but it leaves `fluxion-grid` as a standalone simulator with a CLI wrapper — the weakest integration that still counts. Rejected as the primary option; could be a follow-up demo on top of Option A.

### Option C — wire into `fluxion-twin`

Add electrical state (battery SoC, bus voltages) to the twin's UKF state vector with MQTT telemetry for dispatch signals. Genuinely useful for the digital-twin roadmap (#4023), but it changes twin semantics, needs new measurement/transition functions, and is the largest scope. Defer to the twin roadmap; not the #4005 answer.

## 6. Risks

- **Physics-core semantics:** Option A is strictly post-processing — the adapter reads engine results after the thermal solver finishes. No timestep, tolerance, constant, or ASHRAE-path changes. The determinism contract is preserved because the feature is off by default and the determinism job doesn't enable it.
- **Dependency cycle:** `fluxion-grid` already optionally depends on the main `fluxion` crate (`fluxion-integration` feature). Adding `fluxion` → `fluxion-grid` (`grid` feature) creates an optional-optional cycle. Cargo permits this (the `fluxion`/`fluxion-cfd` and stub-feature precedent suggests the workspace already lives with this shape), but the implementer must verify `cargo build --features grid,fluxion-integration` resolves — a cycle that only materializes when *both* features are on is the failure mode to test.
- **CI/build surface (the original complaint):** a default-off optional dependency does not expand the default workspace build. It adds one feature combination to the feature matrix — acceptable, and strictly less surface than the status quo ante if the dead `src/solar/pv.rs` is deleted. No new required-check names (the `check_required_checks_sync` gate stays green).
- **Scope creep:** the power-flow solver (IEEE 33-bus) is distribution-grid scale and has no building-simulation consumer; Option A deliberately does not wire it into the engine — it stays available as library surface with its existing tests. Wiring feeder-level power flow into per-building simulation is a separate design (and a separate issue).

## 7. Recommended sequence (after Alex picks Option A)

1. Add `grid` feature + optional dep; verify the both-features-on cycle resolves.
2. Write `src/sim/grid_adapter.rs` with unit tests; delete `src/solar/pv.rs`, re-export.
3. Extend engine output struct with the electrical block (additive only); integration test with recorded baseline.
4. Add the `fluxion-examples` demo; CI matrix row for `--features grid`.
5. Update ARCHITECTURE.md N+1, `fluxion-grid/README.md`, FEATURES.md; regenerate inventories; run the workflow/docs gates.
6. Open the PR against `develop`; the PR closes #4005 with the decision + wiring documented.
