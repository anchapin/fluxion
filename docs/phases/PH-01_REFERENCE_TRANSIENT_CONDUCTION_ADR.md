# ADR-001: Reference Transient Conduction Path for Hard Cases

**Date:** 2026-04-17
**Status:** Accepted
**Deciders:** Core Physics Team
**Wave:** Wave 1
**Estimate:** 1 week
**Owner:** Core Physics
**Repository focus:** `src/physics/*`, `docs/literature/*`
**Depends on:** None

---

## Context

High thermal mass buildings (τ ≥ 2 hours) present challenges for standard conduction solvers. The current CTF (Conduction Transfer Function) approach becomes unstable or produces large errors for:
- Extreme construction types (very thick concrete > 300mm)
- Multi-layer assemblies with discontinuous properties
- Phase change materials (PCMs)
- Cases requiring fine temporal resolution (< 1 hour)

The team needs to commit to a **reference-grade** conduction path that serves as the authoritative solver for these hard cases.

---

## Decision

We select **Implicit Finite Difference (BTCS)** as the reference transient conduction path for hard cases.

### Solver Selection: `ImplicitFDSolver` (BTCS Scheme)

**Location:** `src/physics/fd_solver.rs`

### Why Not Crank-Nicolson?

The literature review (`docs/literature/FINITE_DIFFERENCE_STATE_OF_THE_ART.md`) recommends Crank-Nicolson for best accuracy. However:

| Criterion | BTCS (Chosen) | Crank-Nicolson |
|-----------|---------------|----------------|
| Temporal Accuracy | O(Δt) first-order | O(Δt²) second-order |
| Stability | Unconditional | Unconditional |
| Robustness | High (simpler scheme) | Moderate (oscillation risk) |
| Implementation Complexity | Lower | Higher |
| Debugging Ease | Easier | Harder |
| Current Codebase State | Existing, tested | Would require new implementation |

**Rationale for BTCS:**
1. **Robustness over precision**: Hard cases already stress the solver; simpler is better
2. **Existing implementation**: `ImplicitFDSolver` is already in the codebase and tested
3. **Unconditional stability**: Allows large timesteps without numerical issues
4. **O(n) efficiency**: Thomas algorithm (TDMA) scales linearly with nodes
5. **ASHRAE 140 compliance**: BTCS is acceptable for standard validation cases

---

## Target Use Cases

The reference FD path is used when:

1. **CTF coefficient failure**: CTF coefficients are NaN or Inf (checked via `ThermalMethodSelector::validate_ctf_coefficients`)
2. **Extreme thermal mass**: Wall time constant τ > 24 hours
3. **Multi-layer discontinuity**: Layer conductivity ratio > 100:1
4. **PCM materials**: When phase change detection is enabled
5. **Manual override**: User explicitly requests FD via `ThermalMethodSelector::with_override(ThermalMethod::FiniteDifference)`

### Use Case Matrix

| Use Case | Primary Solver | Reference FD? | Reason |
|----------|----------------|--------------|--------|
| Standard lightweight wall (τ < 2h) | 5R1C | No | Fast, sufficient accuracy |
| Standard high-mass wall (2h ≤ τ ≤ 24h) | CTF | No | Optimal accuracy/speed |
| Thick concrete (> 300mm) | CTF → FD fallback | Yes | CTF may produce NaN |
| Phase change materials | FD | Yes | CTF cannot handle nonlinearity |
| Multi-layer with insulation gap | CTF → FD fallback | Yes | Property discontinuity |
| User-specified accuracy test | FD | Yes | Reference-grade comparison |

---

## Numerical Scheme

### Scheme: Backward Time, Central Space (BTCS)

**Governing Equation:**
```
ρ·cₚ·∂T/∂t = ∂/∂x(k·∂T/∂x)
```

**Discretization:**
```
-(Fo)·T_{i-1}^{n+1} + (1+2·Fo)·T_i^{n+1} - (Fo)·T_{i+1}^{n+1} = T_i^n
```

Where: Fo = α·Δt/Δx² (Fourier number)

**Boundary Conditions:**
- Interior: Robin BC with combined convection/radiation coefficient
- Exterior: Robin BC with sol-air temperature
- Interface: Flux continuity between layers

**Solution Method:**
- Thomas Algorithm (Tridiagonal Matrix Algorithm, TDMA)
- Complexity: O(n) operations per timestep
- Stability: Unconditional for all Fo > 0

**Spatial Discretization:**
- Non-uniform grid available (geometric progression, r = 1.2)
- Default: 20 nodes for standard walls, 40 nodes for high-mass walls
- Minimum Δx: 5mm (to resolve boundary layer)

**Temporal Parameters:**
- Default timestep: 15 minutes (900s)
- Acceptable range: 5 minutes to 1 hour
- For annual simulations: 1 hour default (CTF replacement)

---

## Comparison Plan

### Phase 1: Validation Against Analytical Solutions

**Test 1: Periodic Boundary Condition (Kusuda Test)**
- Configuration: 200mm concrete, 24-hour sinusoidal temperature variation
- Analytical solution available (Carslaw & Jaeger)
- Acceptance criteria: RMS error < 2.0°C

**Test 2: Step Change Response**
- Configuration: 100mm concrete, sudden 10°C exterior change
- Acceptance criteria: Flux error < 5% at 1h, 6h, 24h

**Test 3: Multi-Layer Interface**
- Configuration: Concrete (150mm) + Insulation (50mm) + Concrete (50mm)
- Acceptance criteria: Temperature continuity at interface < 0.1°C discontinuity

### Phase 2: ASHRAE 140 Comparison

| Case | Description | 5R1C Error | CTF Error | FD Target | FD Actual |
|------|-------------|------------|-----------|-----------|-----------|
| 600 | Light mass, simple | < 5% | < 3% | < 3% | TBD |
| 900 | High mass, simple | 229% | < 5% | < 3.5% | TBD |
| 920 | High mass, complex | 285% | < 5% | < 4.0% | TBD |
| 960 | High mass, sunspace | 322% | < 5% | < 4.0% | TBD |

**Acceptance criteria:** FD error within 50% of CTF error for all cases.

### Phase 3: Performance Benchmarks

| Metric | Target | Measurement |
|--------|--------|-------------|
| Annual simulation (8760h) | < 50ms per zone | Benchmark with `cargo bench` |
| Memory per zone | < 1KB | Instrumented allocation |
| Node scaling | O(n) | Timing vs node count |

---

## Consequences

### Positive
- **Robust fallback**: FD provides reliable solution where CTF fails
- **Existing implementation**: No new solver code required
- **Unconditional stability**: No CFL restriction on timestep
- **Clear use cases**: Decision document clarifies when FD is used

### Negative
- **First-order temporal accuracy**: Less precise than Crank-Nicolson
- **Slower than CTF**: ~5-20× computational overhead
- **Not optimal for all cases**: overkill for standard high-mass walls

### Neutral
- **Parallelization limited**: TDMA is inherently sequential (acceptable for reference path)

---

## Review History

| Date | Reviewer | Decision |
|------|----------|----------|
| 2026-04-17 | Core Physics | Initial ADR creation |

---

## References

- `docs/literature/FINITE_DIFFERENCE_STATE_OF_THE_ART.md` - FD literature review
- `src/physics/fd_solver.rs` - Implementation
- `src/physics/method_selector.rs` - Solver selection logic
- ASHRAE Standard 140-2017 - Test validation standard
- Carslaw, H.S. & Jaeger, J.C. (1959) - Analytical solutions
