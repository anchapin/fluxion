# T4.4: Solver Abstraction Architecture Review

**Status**: COMPLETE
**Issue**: #728
**Date**: 2026-05-16
**Reviewer**: architecture-reviewer

---

## Executive Summary

The Fluxion solver abstraction is **well-structured at the physics layer** but exhibits a **dual-dispatch problem** between `src/physics/` and `src/sim/` that creates coupling risks. The zone model hierarchy (5R1C through 9R4C) is sound but lives in a separate dispatch path from the heat conduction solver hierarchy (5R1C/CTF/FD), with no unified trait connecting them. The architecture review identifies 5 findings with actionable recommendations.

---

## CHARTER_CHECK

```
- Clarification level: LOW
- Task domain: architecture
- Must NOT do: modify code, create new files in src/, change .agents/ files
- Success criteria: architecture review produced; zone model hierarchy documented; coupling issues identified
- Assumptions: Issue #726 (FD for high-mass) is merged or near-merge; T4.2 is the driver
```

---

## 1. Architecture Overview

### 1.1 Current Module Layout

```
src/
├── physics/                          # Heat conduction solver layer
│   ├── solver_trait.rs               # HeatConductionSolver trait (5 methods)
│   ├── solver_registry.rs            # HashMap<usize, Box<dyn HeatConductionSolver>>
│   ├── solver_manager.rs             # SolverManager facade (542 lines)
│   ├── method_selector.rs            # ThermalMethodSelector (1082 lines)
│   ├── five_r1c_solver.rs            # FiveR1CSolver (implements trait directly)
│   ├── ctf_solver_wrapper.rs         # CTFSolverWrapper → adapts CTFSolver
│   ├── fd_solver_wrapper.rs          # FDSolverWrapper → adapts ImplicitFDSolver
│   ├── multi_node_solver.rs          # MultiNodeSolver (9R4C, SEPARATE from trait)
│   ├── ctf_solver.rs                 # Core CTF implementation
│   ├── fd_solver.rs                  # Core FD implementation
│   └── ...
├── sim/                              # Building simulation layer
│   ├── thermal_model_core.rs         # ThermalModel<T> with ThermalModelType enum
│   ├── thermal_model_solvers.rs      # ISO 13790 solver integration (649 lines)
│   ├── thermal_model_physics.rs      # Physics dispatch (2577 lines!)
│   ├── multi_node_thermal.rs         # MultiNodeThermalMass, ThermalMassNode
│   ├── timestep_solver.rs            # Timestep loop
│   ├── thermal_integration.rs        # Backward Euler / Crank-Nicolson integration
│   └── engine.rs                     # SimulationEngine (969 lines)
└── orchestration/                    # Decision tracing (TDQS harness)
    ├── decision_types.rs             # OrchestrationDecisionKind enum
    └── mod.rs                        # Re-exports
```

### 1.2 Dispatch Architecture (Current)

There are **two independent dispatch paths**:

**Path A: Heat Conduction Solver Dispatch** (physics layer, per-wall)
```
SolverManager.get_or_create_solver(wall)
  → ThermalMethodSelector.select_method(wall)
    → ThermalMethod::{FiveR1C | CTF | FiniteDifference}
      → Box<dyn HeatConductionSolver> stored in SolverRegistry
```

**Path B: Zone Model Dispatch** (sim layer, per-zone)
```
ThermalModel<T>.step_physics(...)
  → match self.model_type {
      ThermalModelType::FiveROneC    → 5R1C integrated thermal balance
      ThermalModelType::SixRTwoC     → 6R2C two-node balance
      ThermalModelType::EightRThreeC → 8R3C three-node balance
      ThermalModelType::NineRFourC   → 9R4C four-node (MultiNodeSolver)
    }
```

---

## 2. Zone Model Hierarchy Documentation

### 2.1 Complete Hierarchy

| Model | Nodes | Resistances | Capacitances | Use Case | Location |
|-------|-------|-------------|-------------|----------|----------|
| **5R1C** | 1 mass | 5 (h_tr_w, h_ve, h_tr_em, h_tr_ms, h_tr_is) | Cm (combined) | Low-mass, general | `thermal_model_solvers.rs` + `five_r1c_solver.rs` |
| **6R2C** | 2 mass | 6 (adds h_tr_me) | Cm_envelope, Cm_internal | High-mass lag | `thermal_model_solvers.rs` |
| **8R3C** | 3 mass | 8 (adds h_tr_ceiling, h_tr_floor, h_tr_partition) | Cm_ceiling, Cm_floor, Cm_partition | Phase 20 eval | `thermal_model_core.rs` enum only |
| **9R4C** | 4 mass | 9 (h_tr_em/ms per surface) | Cm_wall, Cm_roof, Cm_floor, Cm_internal | Heavy mass (900+) | `multi_node_solver.rs` + `multi_node_thermal.rs` |

### 2.2 Zone Model ↔ Solver Mapping

| Zone Model | Heat Conduction Method | Selection Logic |
|-----------|----------------------|-----------------|
| 5R1C | 5R1C solver (default) | `ThermalMethod::FiveR1C` when τ < threshold |
| 5R1C | CTF (override) | `ThermalMethod::CTF` when manual override |
| 5R1C | FD (override) | `ThermalMethod::FiniteDifference` when manual override or high-mass |
| 6R2C | Built-in (no separate solver) | Integrated in `thermal_model_solvers.rs` |
| 8R3C | Not implemented (evaluation) | Stub in `ThermalModelType` enum |
| 9R4C | `MultiNodeSolver` (separate path) | Does NOT use `HeatConductionSolver` trait |

---

## 3. Findings

### Finding 1: Dual Dispatch Problem (MEDIUM severity)

**Problem**: The zone model hierarchy (`ThermalModelType` in `sim/`) and the heat conduction solver hierarchy (`ThermalMethod` in `physics/`) are dispatched independently. A 5R1C zone model can use any of 5R1C/CTF/FD for wall conduction, but a 9R4C zone model bypasses `HeatConductionSolver` entirely and uses `MultiNodeSolver` directly.

**Impact**: Adding a new solver (e.g., surrogate) requires changes in two disconnected places. The `MultiNodeSolver` doesn't benefit from the `SolverManager` lifecycle (pre-warming, stats, fallback).

**Evidence**:
- `MultiNodeSolver` in `physics/multi_node_solver.rs` does NOT implement `HeatConductionSolver`
- `MultiNodeThermalMass` in `sim/multi_node_thermal.rs` is a pure data struct
- `thermal_model_physics.rs` (2577 lines!) contains separate code paths for each model type

### Finding 2: thermal_model_physics.rs God File (HIGH severity)

**Problem**: `src/sim/thermal_model_physics.rs` at 2577 lines contains the dispatch logic for ALL zone models in a single file. This is the single largest coupling risk — any change to any model type touches this file.

**Impact**: High merge conflict rate. Difficult to test individual model implementations in isolation. Violates SRP.

**Recommendation**: Split into `thermal_model_physics/` module with one file per model type.

### Finding 3: SolverManager is Opt-In, Not Default (LOW severity)

**Problem**: `ThermalModel` stores `solver_manager: Option<SolverManager>` and must be explicitly enabled via `enable_solver_manager()`. The default code path in `thermal_model_solvers.rs` uses the built-in 5R1C calculations directly, not through the trait.

**Impact**: Two code paths compute the same physics — one through the trait (when enabled) and one inline (default). Risk of behavioral divergence. Tests may pass for one path but not the other.

**Evidence**: `thermal_model_solvers.rs` has 14 references to `solver_manager`, all behind `if solver_manager.is_some()` guards.

### Finding 4: ThermalMethodSelector Over-Complexity (LOW severity)

**Problem**: `method_selector.rs` is 1082 lines for what is essentially a 3-branch dispatch (τ < threshold → 5R1C, else → FD, CTF only as override). The file includes config structs, tracing integration, per-surface selection, fallback chains, and statistics.

**Impact**: Cognitive overhead for maintainers. The tracing integration (Decision Types) is well-designed but adds ~30% of the file's complexity.

### Finding 5: No Surrogate Integration Path (INFO)

**Problem**: The orchestration layer defines `OrchestrationDecisionKind::SurrogateRouting` as a stub pointing to `src/ai/surrogate.rs` (not yet created). The `distributed_inference.rs` in `sim/` handles parallel building variants but does not route to a surrogate model.

**Impact**: When surrogate models arrive, they will need a clear integration point. The current architecture has the slot (`ThermalMethod` could gain a `Surrogate` variant), but the trait's `initialize(&BuildingAssembly)` signature may not fit surrogate models that don't use wall assemblies.

---

## 4. Trait Design Assessment

### 4.1 `HeatConductionSolver` Trait — Grade: A-

```rust
pub trait HeatConductionSolver: Send + Sync {
    fn name(&self) -> &str;
    fn initialize(&mut self, wall: &BuildingAssembly) -> Result<(), SolverError>;
    fn step(&mut self, timestep: f64, T_interior: f64, T_exterior: f64,
            h_interior: f64, h_exterior: f64) -> Result<f64, SolverError>;
    fn energy_storage_rate(&self) -> f64;
    fn is_valid(&self) -> bool;
}
```

**Strengths**:
- Clean 5-method interface, easy to implement
- `Send + Sync` bounds enable concurrent use
- `SolverError` enum covers all failure modes
- Consistent units (W/m², °C, seconds)

**Weaknesses**:
- `initialize(&BuildingAssembly)` couples trait to `sim::assembly` — physics layer depends on sim layer (circular dependency risk)
- No `reset()` or `invalidate()` method for timestep rollback scenarios
- No method to query internal state (e.g., node temperatures for FD) — abstraction prevents diagnostics
- `step()` returns only flux, not surface temperatures — some callers need both

### 4.2 Wrapper Pattern — Grade: B+

The three wrappers (`FiveR1CSolver`, `CTFSolverWrapper`, `FDSolverWrapper`) correctly adapt their underlying solvers to the trait. The wrapper pattern is clean.

**Observation**: `FiveR1CSolver` implements the trait directly (no wrapper needed), while CTF and FD use explicit wrapper structs. This asymmetry is minor but worth noting for documentation consistency.

### 4.3 `SolverRegistry` — Grade: A

```rust
solvers: HashMap<usize, Box<dyn HeatConductionSolver>>
```

Clean, minimal registry with typed access. The separation of `SolverRegistry` (internal state) from `SolverManager` (public facade) is a good pattern — allows testing the registry independently.

---

## 5. Coupling Analysis

### 5.1 Dependency Flow (Current)

```
orchestration/decision_types.rs
  ← physics/method_selector.rs (tracing integration)

physics/solver_trait.rs
  ← sim/assembly.rs (BuildingAssembly parameter)  ⚠ CROSS-LAYER

physics/solver_manager.rs
  ← physics/solver_registry.rs
  ← physics/method_selector.rs
  ← physics/{ctf,fd,five_r1c}_solver{,_wrapper}.rs
  ← sim/assembly.rs  ⚠ CROSS-LAYER

sim/thermal_model_core.rs
  ← physics/constants (thermal coefficients)
  ← sim/thermal_model_solvers.rs
  ← sim/thermal_integration.rs

sim/thermal_model_solvers.rs
  ← physics/solver_manager.rs (optional)
  ← sim/multi_node_thermal.rs
```

### 5.2 Coupling Concerns

1. **physics → sim dependency** (Finding 3 from trait design): `solver_trait.rs` imports `sim::assembly::BuildingAssembly`. This creates a reverse dependency where the physics layer depends on the simulation layer. In a clean architecture, physics should be lower-level than sim.

2. **thermal_model_physics.rs as a coupling nexus**: At 2577 lines, this file imports from physics (constants, solvers), sim (models, HVAC, integration), and orchestration (decision types). It is the system's widest coupling point.

3. **MultiNodeSolver is standalone**: It lives in `physics/` but does not participate in the solver trait system. This is correct (it's a zone-level solver, not a wall-level solver), but the co-location in `physics/` may confuse future contributors.

---

## 6. Recommendations

### R1: Unify Dispatch (Priority: MEDIUM, Effort: MEDIUM)

Create a `ZoneSolver` trait that encompasses both the zone model dispatch and the wall conduction dispatch:

```rust
trait ZoneSolver {
    fn zone_model_type(&self) -> ThermalModelType;
    fn step_zone(&mut self, params: &ZoneStepParams) -> ZoneResult;
    fn wall_solver(&self, wall_index: usize) -> Option<&dyn HeatConductionSolver>;
}
```

This would allow `MultiNodeSolver` to participate in the same lifecycle as 5R1C/6R2C models.

### R2: Split thermal_model_physics.rs (Priority: HIGH, Effort: LOW)

Move to module structure:
```
src/sim/thermal_model_physics/
├── mod.rs              # dispatch + shared types
├── five_r1c.rs         # 5R1C step
├── six_r2c.rs          # 6R2C step
├── nine_r4c.rs         # 9R4C step
└── common.rs           # shared physics calculations
```

### R3: Make SolverManager Default (Priority: LOW, Effort: MEDIUM)

Remove the `Option<SolverManager>` pattern and always route through the trait. This eliminates the dual-codepath risk. The 5R1C built-in calculations become the default `FiveR1CSolver` trait implementation.

### R4: Prepare Surrogate Slot (Priority: LOW, Effort: LOW)

Add `ThermalMethod::Surrogate` to the enum now (behind a feature flag or `#[cfg(test)]`). Define the integration contract so surrogate work can proceed independently:

```rust
pub enum ThermalMethod {
    FiveR1C,
    CTF,
    FiniteDifference,
    #[cfg(feature = "surrogate")]
    Surrogate,
}
```

### R5: Invert physics → sim Dependency (Priority: LOW, Effort: MEDIUM)

Extract `BuildingAssembly` (or a `WallConstruction` trait) into a shared types module that both `physics` and `sim` depend on, breaking the reverse dependency.

---

## 7. Architecture Diagrams

### 7.1 Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         orchestration/                          │
│   OrchestrationDecisionKind { SolverSelection, AdaptiveTS,      │
│     SurrogateRouting, ConstraintWarning, HvacHorizon }          │
└───────────────────────────┬─────────────────────────────────────┘
                            │ tracing spans
┌───────────────────────────▼─────────────────────────────────────┐
│                            sim/                                  │
│  ┌─────────────────┐    ┌──────────────────────┐                │
│  │ ThermalModel<T> │    │ ThermalModelType      │                │
│  │  .model_type ───┼───►│  FiveROneC (default) │                │
│  │                 │    │  SixRTwoC             │                │
│  │  .solver_manager│    │  EightRThreeC (stub)  │                │
│  │   (Option<SM>)──┼─┐  │  NineRFourC           │                │
│  └─────────────────┘ │  └──────────────────────┘                │
│                       │                                           │
│  ┌────────────────────▼─────────────────────────┐               │
│  │ thermal_model_physics.rs (2577 lines)        │               │
│  │   match model_type → per-model step logic    │               │
│  └──────────────────────────────────────────────┘               │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                          physics/                                │
│  ┌──────────────────────────────────────────────────┐           │
│  │ SolverManager (facade)                           │           │
│  │   ├── ThermalMethodSelector.select_method()      │           │
│  │   │     τ < threshold → FiveR1C                  │           │
│  │   │     τ ≥ threshold → FiniteDifference          │           │
│  │   │     override → CTF                           │           │
│  │   └── SolverRegistry                             │           │
│  │         HashMap<usize, Box<dyn HeatConductionSolver>>        │
│  └──────────────────────────────────────────────────┘           │
│                                                                  │
│  impl HeatConductionSolver:                                      │
│    FiveR1CSolver    (direct)                                     │
│    CTFSolverWrapper (adapts CTFSolver)                           │
│    FDSolverWrapper  (adapts ImplicitFDSolver)                    │
│                                                                  │
│  NOT implementing HeatConductionSolver:                          │
│    MultiNodeSolver (9R4C, separate path)                        │
└──────────────────────────────────────────────────────────────────┘
```

### 7.2 Zone Model Hierarchy

```
                    ThermalModelType (enum)
                         │
          ┌──────────────┼──────────────┬──────────────┐
          │              │              │              │
       FiveROneC     SixRTwoC    EightRThreeC    NineRFourC
     (5R, 1C)      (6R, 2C)     (8R, 3C)       (9R, 4C)
          │              │              │              │
     Combined Cm   Envelope/Int  Ceil/Flr/Part  Wall/Roof/Flr/Int
          │              │         (stub)          │
     ┌────┴────┐    Built-in                  MultiNodeSolver
     │         │    (thermal_model_           (multi_node_solver.rs)
  Default   SolverManager  solvers.rs)              │
  (built-in) (opt-in)                         MultiNodeThermalMass
     │         │                              (multi_node_thermal.rs)
  thermal_  Box<dyn HC>
  model_    Solver>
  solvers.rs
```

---

## 8. Validation Steps

1. **Verify no regression**: Run `cargo test` after any architectural change
2. **Check ASHRAE 140**: Cases 600/900 should pass with both 5R1C and FD paths
3. **Trace dispatch**: Enable `RUST_LOG=trace` and verify `OrchestrationDecisionKind::SolverSelection` spans fire correctly
4. **Check coupling**: `cargo depgraph` should show no circular physics ↔ sim dependencies (currently violated)

---

## 9. Artifacts Created

| File | Description |
|------|-------------|
| `.agents/results/result-t4-4-architecture.md` | This architecture review document |

---

## 10. Conclusion

The Fluxion solver abstraction is **architecturally sound in isolation** — the `HeatConductionSolver` trait, wrapper pattern, and `SolverManager` facade form a clean Strategy pattern with registry ownership. The primary concern is that the **zone model hierarchy and wall conduction hierarchy are parallel dispatch systems** that don't share a common abstraction. This is not a bug today, but it will become a maintenance burden as new model types (8R3C, surrogate) are added.

The highest-priority action item is **R2 (split thermal_model_physics.rs)** — this is a low-effort, high-impact change that reduces coupling risk immediately. R1 (unified dispatch) can wait until the surrogate model design is clearer.
