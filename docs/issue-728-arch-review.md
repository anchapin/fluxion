# Architecture Review: Issue #728

## Solver Abstraction, Zone Model Hierarchy, and Physics Component Interfaces

**Date**: 2026-06-16
**Issue**: #728
**Status**: Review Complete
**Branch**: `fix/issue-728-arch-review`

---

## Executive Summary

This review identifies gaps in the solver abstraction layer, zone model hierarchy clarity, and physics component interface documentation. The codebase has a solid trait foundation (`HeatConductionSolver`, `ThermalModelTrait`) but suffers from unclear selection policies, parallel hierarchies that are not well documented, and implicit routing logic.

---

## 1. Current State of Solver Abstraction

### 1.1 Trait Hierarchy (as documented in ARCHITECTURE.md)

```text
HeatConductionSolver (physics/solver_trait.rs)
├── FiveR1CSolver     (5R1C thermal network - fast, low-mass)
├── CTFSolverWrapper   (Conduction Transfer Functions - high-mass)
└── FDSolverWrapper    (Finite Difference - robust fallback)

ThermalModelTrait (sim/thermal_model.rs)
├── PhysicsThermalModel      (analytical 5R1C thermal network)
├── SurrogateThermalModel    (neural network inference)
├── UnifiedThermalModel      (runtime switching)
└── MockThermalModel         (testing)
```

### 1.2 Actual Implementation State

**HeatConductionSolver trait** (`src/physics/solver_trait.rs`):
- Well-defined interface with `initialize()`, `step()`, `energy_storage_rate()`, `is_valid()`
- Uses strongly-typed units (`Time`, `Temperature`, `HeatFlux`, `HeatTransferCoefficient`)
- No information hiding issues - clean abstraction

**ThermalMethodSelector** (`src/physics/method_selector.rs`):
- Selection based on thermal mass time constant τ
- Threshold default: 2.0 hours (but config default is 24.0 hours - inconsistency)
- Selection logic (line 350-354):
  ```rust
  let method = if tau < self.threshold_hours {
      ThermalMethod::FiveR1C // Low mass: use fast 5R1C
  } else {
      ThermalMethod::FiniteDifference // High mass: use FD (Issue #726)
  };
  ```
- **CTF is excluded from automatic selection** after Issue #726 fix

**SolverManager** (`src/physics/solver_manager.rs`):
- Manages per-wall solver instances via `SolverRegistry`
- Uses `get_or_create_solver()` pattern with lazy initialization
- CTF→FD fallback built into creation (lines 131-150)

### 1.3 Zone Model Hierarchy

The `ThermalModelTrait` hierarchy in `thermal_model.rs` has three concrete implementations that all wrap the same underlying `ThermalModel<VectorField>`:

| Model | Mode Flag | Use Case |
|-------|-----------|----------|
| `PhysicsThermalModel` | `Physics` | Analytical calculations |
| `SurrogateThermalModel` | `Surrogate` | Neural network inference |
| `UnifiedThermalModel` | `Physics/Surrogate/Hybrid` | Runtime switching |

**Problem**: All three wrap identical inner types (`ThermalModel<VectorField>`). The mode flag is the only differentiator, suggesting this may be over-engineered or the actual differentiation lives deeper in the code.

---

## 2. Gaps and Problems Identified

### 2.1 Solver Selection is Implicit

**Problem**: `ThermalMethodSelector::select_method()` uses ad-hoc threshold logic with no explicit policy documentation.

**Evidence**:
- Threshold defaults are inconsistent: `ThermalMethodSelector` default is 2.0h, but `ThermalMethodSelectorConfig` default is 24.0h
- The selection comment mentions ISO 13790 guidance but provides no citation
- Selection rules are embedded in code, not in a documented policy

**Impact**: Users cannot reason about solver selection without reading code.

### 2.2 Zone Model Hierarchy Unclear

**Problem**: ARCHITECTURE.md documents the hierarchy but provides no guidance on when to use each model type.

**Evidence**:
- `PhysicsThermalModel` vs `SurrogateThermalModel` - no documented accuracy/complexity tradeoffs
- `UnifiedThermalModel` mode switching methods (`use_physics()`, `use_surrogates()`) exist but use cases are unclear
- No mention of 6R2C or 8R3C models mentioned in the issue - where are they?

### 2.3 CTF in Limbo

**Problem**: `ThermalMethod::CTF` remains in the enum but is excluded from automatic selection after Issue #726.

**Evidence**:
- `select_method()` routes high-mass → FD, not CTF
- CTF solver wrapper still exists (`CTFSolverWrapper`)
- Tests still check for CTF in method counts
- Comment at line 347: "Issue #726: CTF is architecturally wrong for high-mass constructions"

**Impact**: Technical debt - CTF cannot be selected automatically but still exists as an option.

### 2.4 No Explicit Solver Policy

**Problem**: The architecture document mentions "SolverManager auto-selects based on thermal mass" but doesn't define what "appropriate" means.

**Missing**:
- Accuracy bounds for each solver method
- Computational cost comparison
- Validation status against E+ reference data per method
- Recommended use cases per solver type

### 2.5 Hard-coded Heat Transfer Coefficients

**Problem**: `step_all()` in `SolverManager` (lines 346-347) uses hard-coded h values:
```rust
let h_int = 8.0;
let h_ext = 25.0;
```
But `ThermalMethodSelector` uses ASHRAE 140 values (8.29 and 29.3).

**Impact**: Inconsistent boundary conditions between selection and execution.

---

## 3. Recommended Improvements

### 3.1 Document Solver Selection Policy

**Recommendation**: Create a `SOLVER_SELECTION_POLICY.md` that defines:

1. **Decision criteria for each solver**:
   - 5R1C: τ < 2h, lightweight constructions, first-order accuracy
   - FD: τ ≥ 2h, high-mass constructions, robust for extreme geometries
   - CTF: Deprecated - remove from enum or mark as experimental

2. **Validation targets**:
   - 5R1C: Heat flux within 1% of E+ for low-mass walls
   - FD: Heat flux within 1% of E+ for high-mass walls

3. **Computational cost**:
   - 5R1C: O(1) per surface per timestep
   - FD: O(n_layers × n_timesteps)

### 3.2 Clarify Zone Model Hierarchy

**Recommendation**: Document in ARCHITECTURE.md:

1. When to use `PhysicsThermalModel` vs `SurrogateThermalModel`
2. The actual difference between the three implementations (currently they all wrap the same inner type)
3. Hybrid mode intended use case

### 3.3 Resolve CTF Status

**Recommendation**: Either:
- **Option A**: Remove CTF from `ThermalMethod` enum and delete `CTFSolverWrapper` (if truly deprecated)
- **Option B**: Properly validate CTF and include it in automatic selection if it provides accuracy benefits

Given Issue #726's finding that "CTF is architecturally wrong for high-mass," Option A is recommended.

### 3.4 Fix Hard-coded Boundary Conditions

**Recommendation**: In `SolverManager::step_all()`, use the same h values that `ThermalMethodSelector` uses for consistency:

```rust
// Use ASHRAE 140 standard values
let h_int = 8.29;  // W/m²K per ASHRAE 140 Section 5.2
let h_ext = 29.3;  // W/m²K at 6.7 m/s wind speed
```

### 3.5 Add Selection Rationale to SolverSelectionResult

**Recommendation**: Enhance `SolverSelectionResult` to include:
- Accuracy classification (low/medium/high mass)
- Computational estimate
- E+ validation status

---

## 4. Priority Actions

| Priority | Action | Complexity |
|----------|--------|------------|
| High | Document solver selection policy | Low |
| High | Resolve CTF status (remove or validate) | Medium |
| Medium | Fix hard-coded h values inconsistency | Low |
| Medium | Clarify zone model hierarchy in docs | Low |
| Low | Refactor ThermalModelTrait hierarchy if needed | High |

---

## 5. Files Requiring Changes

| File | Change |
|------|--------|
| `ARCHITECTURE.md` | Add solver selection policy section |
| `src/physics/method_selector.rs` | Remove CTF or document why it remains |
| `src/physics/solver_manager.rs` | Use consistent h values |
| `SOLVER_SELECTION_POLICY.md` | Create new document |

---

## 6. Questions for Further Investigation

1. **6R2C and 8R3C models**: The issue mentions these but they don't appear in the current codebase. Are they planned features or legacy references?

2. **CTF validation**: Has CTF ever been validated against E+ reference data? If not, why was it included?

3. **Surrogate models**: What is the expected accuracy of `SurrogateThermalModel` compared to `PhysicsThermalModel`? The architecture mentions 2% tolerance but this may be for future ML surrogates.

---

## 7. Conclusion

The trait abstraction layer is fundamentally sound, but the **policies and documentation** around solver selection need improvement. The immediate actions should be:

1. **Document the solver selection policy** with explicit thresholds and rationale
2. **Resolve CTF status** - either remove it or properly validate it
3. **Fix boundary condition inconsistency** between selection and execution

The zone model hierarchy appears to be over-engineered with three implementations that all wrap the same inner type. This warrants a closer investigation to determine if simplification is possible.