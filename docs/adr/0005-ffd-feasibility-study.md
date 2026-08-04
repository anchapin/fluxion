# ADR-XXX: Feasibility Study — Fast Fluid Dynamics (FFD) for Airflow Simulation

- **Status:** Draft (for review)
- **Date:** 2026-08-04
- **Deciders:** TBD
- **Supersedes:** None
- **Depends on:** None (this is a pre-decision study)

---

## Executive Summary

**Recommendation: PROCEED with Phase 1 (FFD baseline + GPU acceleration) as a new
workspace crate, but DEFER Phases 2-4 (FMI/FMU co-simulation infrastructure) until
Phase 1 demonstrates < 100ms/FFD-step performance on representative geometry.**

The proposed FFD (Fast Fluid Dynamics) solver for whole-building airflow co-simulation
is technically feasible but represents a significant new physics domain outside the
current Fluxion core competency (lumped-capacitance thermal networks). A staged
approach is recommended to manage risk:

| Phase | Deliverable | Risk | Recommendation |
|-------|-------------|------|----------------|
| 1 | FFD baseline + GPU acceleration | Medium | **Proceed** |
| 2 | FMI/FMU encapsulation | Medium-High | **Defer** (validate Phase 1 first) |
| 3 | Shared memory + loose coupling | High | **Defer** |
| 4 | End-to-end BES+FFD validation | High | **Defer** |

This ADR covers the feasibility analysis; Issues #2385–#2392 track the implementation.

---

## Context

### Why airflow simulation matters for Fluxion

The current Fluxion engine solves **whole-building energy simulation (BES)** using
lumped-capacitance thermal network models (5R1C, CTF, 9R4C). These models:

- Adequately predict **zonal average air temperatures** and total HVAC loads
- **Cannot** resolve:
  - Localized air stratification (e.g., vertical temperature gradients in atria)
  - Personalized comfort (draft risk, thermal plume behavior)
  - Smoke and contaminant transport paths
  - Detailed ventilation effectiveness at sub-zonal scales
  - Transient buoyancy-driven airflow (natural ventilation sizing)

ASHRAE 140 validation confirms that for cases involving natural ventilation,
infiltration, or mixed-mode operation, Fluxion relies on empirical correlations
rather than physics-based airflow resolution.

### What is FFD?

**Fast Fluid Dynamics (FFD)** is a reduced-order Computational Fluid Dynamics (CFD)
method that trades accuracy for speed. Unlike full CFD (which may take hours per
building), FFD can achieve **real-time or faster-than-real-time** performance by:

1. Using a **coarse Eulerian grid** (10–100x coarser than traditional CFD)
2. Employing **semi-Lagrangian advection** (unconditionally stable, large time steps)
3. Applying **algebraic turbulence models** instead of RANS/LES
4. Exploiting **GPU parallelism** for the pressure projection step

FFD is not intended to replace CFD for detailed design studies. It is intended to
enable:

- **Parametric sweeps**: 1000+ configurations for optimization
- **Real-time visualization**: Interactive design exploration
- **Co-simulation with BES**: Dynamic boundary condition exchange
- **Monte Carlo uncertainty propagation**: Statistical ensembles

### Relationship to existing Fluxion architecture

Fluxion currently models heat conduction and surface thermal radiation.
Airflow is modeled implicitly via:
- Fixed ACH (air changes per hour) infiltration schedules
- Zone mixing effectiveness factors
- Empirical convection correlation coefficients (CHTC)

FFD would **add a new physics module** (airflow) that interfaces with the existing
thermal solver via:
- **Inputs to FFD**: Wall/floor surface temperatures, HVAC flow rates, wind pressure
- **Outputs from FFD**: Localized CHTCs, zone air temperatures (stratified),
  surface heat flux distributions

---

## Problem Statement

**Core Question**: Can we build a GPU-accelerated FFD solver and integrate it with
the existing BES engine to enable high-fidelity airflow simulation at speeds
compatible with whole-building energy analysis?

### Specific challenges

1. **FFD is a different physics domain**: Fluxion's team has deep expertise in
   thermal network solvers (5R1C, CTF, 9R4C). FFD requires Navier-Stokes
   numerical methods expertise that is currently outside the core team.

2. **GPU complexity**: Achieving 500–1500x speedup requires careful CUDA/OpenCL
   implementation. Memory coalescing, register pressure, and warp divergence
   are non-trivial to debug.

3. **Co-simulation coupling stability**: Loose coupling (data exchange at BES
   macro-steps) can introduce instability if the FFD time constants and BES
   time constants are mismatched.

4. **Validation burden**: FFD accuracy must be validated against benchmark cases
   (NIST, ASHRAE, experimental data). This is a significant commitment.

5. **Architectural disruption**: Adding FFD as a new workspace crate (`fluxion-cfd`)
   avoids disrupting the existing validated thermal solver, but requires careful
   interface design.

---

## Technical Analysis

### FFD vs. Alternatives

| Approach | Speed | Accuracy | Implementation Effort | Notes |
|----------|-------|----------|----------------------|-------|
| **Full RANS CFD** | ~1 hr/building | High | Very High | Not real-time capable |
| **Large Eddy Simulation (LES)** | ~10 hr/building | Very High | Extreme | Research only |
| **FFD (this proposal)** | ~1–60 sec/building | Moderate | High | Balances speed/accuracy |
| **PEM (Porous Enclosure Model)** | ~1 sec/building | Low-Moderate | Low | Currently in Fluxion |
| **zonal models** | ~1 sec/building | Low | Low | Already in Fluxion |

**FFD fills the gap** between fast empirical zonal models (current Fluxion) and
slow research-grade CFD.

### FFD Algorithm Overview

The fractional-step (time-splitting) method solves Navier-Stokes in three steps:

```
1. ADVECTION (semi-Lagrangian):
   - Backtrace particles to find origin
   - Interpolate velocity from grid
   - Update velocity field

2. DIFFUSION (implicit):
   - Solve (I - ν∇²)u = u*  [implicit, unconditionally stable]
   - Typically CG or Gauss-Seidel iteration

3. PRESSURE PROJECTION:
   - Solve ∇²p = ∇·u**  [Poisson equation]
   - Subtract ∇p from velocity to enforce divergence-free
   - Requires efficient linear solver (PCG, multigrid)
```

**GPU acceleration targets**:
- Advection: Highly parallel, simple arithmetic → **~1000x speedup on GPU**
- Diffusion: Sparse matrix-vector products → **~100x speedup on GPU**
- Pressure Poisson: Most expensive step; benefits most from GPU → **~200-500x speedup**

### Co-simulation Coupling Modes

| Mode | Description | Complexity | Stability Risk |
|------|-------------|------------|----------------|
| **Loose (quasi-dynamic)** | Exchange at BES macro-steps only | Low | Medium |
| **Serial (staggered)** | BES → FFD → BES in sequence | Medium | Low |
| **Parallel (iterative)** | Iterate within each macro-step | High | High (convergence) |

**Recommendation**: Loose coupling (Issue #2390) with shared memory (Issue #2389)
to minimize coupling overhead and avoid intra-step iterations.

---

## Feasibility Assessment

### Phase 1: FFD Baseline + GPU (Issues #2385, #2386)

**Feasibility: HIGH**

- The fractional-step algorithm is well-documented and mature
- Semi-Lagrangian advection is straightforward to implement
- GPU toolchains (CUDA, OpenCL) are stable
- Existing FFD reference implementations available (Stanford, NIST)
- Unit tests can validate against analytical benchmarks (lid-driven cavity, etc.)

**Key risk**: Achieving the 100x+ speedup target requires careful profiling and
optimization. Starting with CPU baseline (Issue #2385) before GPU port
(Issue #2386) reduces risk.

**Estimated effort**: 3–6 months for a team with CFD experience

### Phase 2: FMI/FMU Encapsulation (Issue #2388)

**Feasibility: MEDIUM**

- FMI standard is well-documented and tooling exists
- PyFMI provides validation harness
- FFD FMU would be a first for the Fluxion project
- Requires careful interface design (input/output variable definitions)

**Key risk**: FFI complexity between Rust FFD and C FMI API. Consider implementing
FMI wrapper in a separate C or C++ thin layer.

**Estimated effort**: 1–2 months

### Phase 3: Shared Memory + Loose Coupling (Issues #2389, #2390)

**Feasibility: MEDIUM-HIGH**

- POSIX shared memory is portable and well-understood
- Double-buffering prevents read/write conflicts
- Loose coupling avoids convergence complexity

**Key risk**: Cross-platform compatibility (Windows vs. POSIX). Consider abstracting
the shared memory interface behind a trait.

**Estimated effort**: 2–3 months

### Phase 4: End-to-End Validation (Issue #2392)

**Feasibility: MEDIUM**

- Benchmark cases exist (ASHRAE 140, NIST BESTEST)
- Performance metrics are well-defined

**Key risk**: Achieving < 10% error on coupled simulation may require tuning
FFD parameters, which is forbidden by AGENTS.md ("fix the math"). FFD physics
must be sound from the start.

**Estimated effort**: 2–4 months

---

## Integration with Existing Architecture

### Workspace Structure

FFD should be implemented as a **new workspace crate** to avoid disrupting the
validated thermal solver:

```
fluxion/                          # main engine (unchanged)
fluxion-core/                     # leaf modules (unchanged)
fluxion-cfd/                     # NEW: FFD solver crate
  src/
    advection.rs                 # Semi-Lagrangian advection
    diffusion.rs                 # Implicit diffusion solver
    pressure.rs                  # Poisson equation solver
    ffd_solver.rs                # Main FFD orchestrator
    gpu/                         # CUDA/OpenCL kernels
      advect_kernel.rs
      diffuse_kernel.rs
      poisson_kernel.rs
  tests/
    lid_driven_cavity.rs
    buoyancy_driven.rs
  benches/
    cpu_vs_gpu.rs
```

### Module Boundaries (to be documented in ARCHITECTURE.md)

| Module | Responsibility | Interface |
|--------|----------------|-----------|
| `fluxion::sim::thermal_model` | Zone energy balance | Outputs wall temps, HVAC flows to FFD |
| `fluxion_cfd::FFD` | Airflow simulation | Accepts BCs, outputs CHTCs, temperatures |
| `fluxion_cfd::FMI` | Co-simulation wrapper | Exposes FFD as FMU |

**Critical constraint**: The `fluxion-core` ↔ `fluxion::sim_*` ↔
`fluxion::physics_*` cycle-breaking rule (Issue #1441) must **not** be violated.
FFD must not be imported by `fluxion-core` or `fluxion::physics`.

### Data Exchange Contract

```
BES (Zone Balance)                          FFD (Airflow)
      │                                            │
      │  wall_surface_temps[K] (N surfaces)        │
      │  hvac_supply_flows[m³/s] (N zones)        │
      │  wind_pressure[Pa] (N facades)             │
      ├────────────────────────────────────────►  │
      │                                            │
      │                           (FFD runs micro-steps)
      │                                            │
      │  chtc[W/m²K] (N surfaces)                 │
      │  zone_air_temps[K] (N zones, stratified)   │
      │  surface_heat_flux[W/m²]                   │
      ├◄─────────────────────────────────────────  │
      │                                            │
```

---

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GPU implementation fails to achieve speedup target | Medium | High | Start with CPU baseline; profile before GPU port |
| FFD physics validation fails against benchmarks | Medium | High | Engage CFD expert for first 6 months; iterate |
| FMI wrapper introduces FFI complexity | Medium | Medium | Use C++ thin layer for FMI; Rust FFD behind trait |
| Shared memory causes deadlock/stability issues | Low | High | Double-buffering; extensive integration testing |
| FFD module becomes unmaintained after initial development | Medium | High | Document clearly; consider OSS community engagement |
| Scope creep: FFD grows to full CFD | High | Very High | Strict phase gates; reject features outside real-time scope |
| Coupling instability in BES+FFD co-simulation | Medium | High | Loose coupling only; validate with 24-hr runs before Phase 4 |

---

## Decision

### Recommended Action

**Proceed with Phase 1 (Issues #2385, #2386) as a new workspace crate
`fluxion-cfd`, with explicit phase gates before committing to Phases 2–4.**

### Phase Gate Criteria

Before opening Issues #2388–#2392, the following must be true:

| Gate | Criteria | Evidence |
|------|----------|----------|
| G1 | FFD CPU baseline passes analytical benchmarks | Test results in `tests/cfd/` |
| G2 | GPU version achieves ≥ 100x speedup over CPU | Benchmarks in `benches/cpu_vs_gpu/` |
| G3 | FFD numerical stability confirmed for 1000+ time steps | Stability test results |
| G4 | Code review approval from at least one CFD-experienced reviewer | PR review comments |

### Why not defer entirely?

- Current zonal models cannot capture sub-zonal airflow effects
- No existing open-source GPU-accelerated FFD suitable for BES integration
- The proposed implementation plan (fractional-step + GPU) is well-understood
- Starting Phase 1 now positions Fluxion for **real-time building airflow
  simulation** within 12–18 months

### Why not proceed with all phases now?

- Phase 1 validates feasibility at lowest risk
- Phases 2–4 require FFD to be stable and fast first
- FMI/shared-memory infrastructure can be redesigned once FFD is validated
- Resource constraints: FFD development should not block existing BES roadmap

---

## Consequences

### Positive

- Enables sub-zonal airflow resolution in whole-building energy simulation
- Opens door to real-time airflow visualization for design tools
- Positions Fluxion for next-generation BES+CF D co-simulation
- Attracts CFD expertise to the project

### Negative

- Significant new physics domain outside current team expertise
- Initial development will require 6–12 months before Phase 1 is validated
- Risk of scope creep toward full CFD complexity
- Additional crate increases CI/build complexity

### Neutral

- Existing thermal solver (5R1C, CTF, 9R4C) is unaffected
- Phase 1 can proceed without disturbing current ASHRAE 140 validation
- Fluxion remains a BES tool; FFD is an optional add-on for advanced users

---

## Alternatives Considered

### Alternative A: Purchase commercial CFD co-simulation tool

- **Examples**: Siemens STAR-CCM+, Autodesk CFD, ANSYS Fluent
- **Pros**: Validated physics, professional support
- **Cons**: Expensive licenses; proprietary; not real-time capable; integration
  complexity similar to building FFD
- **Verdict**: Rejected — does not meet real-time speed requirement

### Alternative B: Use existing open-source CFD (OpenFOAM)

- **Pros**: Mature, validated, free
- **Cons**: Not real-time capable; complex interface; poor GPU support;
  co-simulation with BES still requires custom adapter
- **Verdict**: Rejected — does not meet real-time speed requirement

### Alternative C: Enhanced zonal model (no FFD)

- **Approach**: Extend existing zonal models with better correlations for
  stratification, natural ventilation
- **Pros**: Low development effort; stays within current expertise
- **Cons**: Still empirical; cannot capture complex airflow paths; limited
  improvement over current approach
- **Verdict**: Rejected — insufficient accuracy improvement

### Alternative D: Do nothing

- **Pros**: No development cost; no risk
- **Cons**: Fluxion remains limited to zonal-average airflow modeling
- **Verdict**: Rejected — missed opportunity; FFD is the industry direction

---

## References

- #2385 — Implement FFD baseline (Issue #1 from plan)
- #2386 — Port FFD to GPU (Issue #2 from plan)
- #2388 — FMI/FMU encapsulation (Issue #3 from plan)
- #2389 — Shared memory buffer (Issue #4 from plan)
- #2390 — Loose coupling (Issue #5 from plan)
- #2391 — Master simulation integration (Issue #6 from plan)
- #2392 — End-to-end validation (Issue #7 from plan)
- Staniforth & Côté (1991). "Semi-Lagrangian Integration Schemes." *Monthly Weather Review*
- Bell, Colella, Glaz (1989). "A Second-Order Projection Method for the Navier-Stokes Equations." *JCP*
- Zhai & Clarke (2005). "Simulation-based configuration of building energy systems." *Energy and Buildings*
- Zuo et al. (2016). "On coupling BES and CFD simulations at different timescales." *Building and Environment*
- FMI 2.0.3 Specification: https://fmi-standard.org/
