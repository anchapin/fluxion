# Issue: LIMIT-21 Phase 7 Follow-up - GaugeSolver Air-Trajectory Fidelity

**Created**: 2026-09-21
**Status**: Open - Awaiting Implementation
**Parent**: Issue #3911 (solar splitting completed), LIMIT-21 cohort
**GitHub Issue**: #3916
**Blocked By**: None (diagnostic complete)
**Blocks**: §LIMIT-21 β-soak closure, gauge-solver production default flip

## Executive Summary

The GaugeSolver air trajectory fails on Cases 600/640/650 series because:

1. **Missing Solar Lag Correction**: 5R1C has a `solar_lag` state variable filtering high-frequency solar transients; gauge has no equivalent
2. **Different Effective Damping**: Gauge N=3 sub-stepping gives ~84%/hr equilibration vs 5R1C's ~78%/hr
3. **τ_air Formula Difference**: 5R1C includes internal surface coupling terms; gauge's τ_air doesn't

## Root Cause Analysis

### Air Node Time Constant Issue

The air-node time constant is τ_air = C_air / h ≈ 0.28 h, giving dt/τ ≈ 3.6 at the 1-hour timestep.

**GaugeSolver** (`src/physics/gauge_zone_solver.rs:802-808`):
```rust
// Implicit Euler - no exponential solution
T_air_current = (self.C_air * T_air_current
    + dt_sub * (net_power_watts + h_total * T_ext_val))
    / (self.C_air + h_total * dt_sub);
```

**5R1C** (`src/sim/thermal_model_physics/physics_impl/step_5r1c.rs:936`):
```rust
// Exact exponential solution
let exponent = -dt_sub / tau_air;
t_i_free_i = steady + (t_air_old_i - steady) * exponent.exp()
```

### Solar Lag Correction

**5R1C has** (`step_5r1c.rs:944-1002`):
```rust
// Issue #1860: Solar-lag correction
let tau_lag = (tau_air_i * tau_mass_i).sqrt();
let decay = (-dt_sub / tau_lag).exp();
let new_solar_lag = old_solar_lag * decay + lag_input * (1.0 - decay);
```

**GaugeSolver does NOT have** - all solar enters zone air instantly through the direct-to-air fraction.

### Effective Damping Comparison

| Aspect | Gauge Solver | 5R1C Solver |
|--------|-------------|---------------|
| Air ODE solution | Implicit Euler (lines 802-808) | Exact exponential (line 936) |
| Solar lag correction | **MISSING** | Present (lines 944-1002) |
| Sub-step equilibration | ~84%/hr (16% memory) | ~78%/hr (22% memory) |
| τ_air formula | C_air / (h_inf + h_surface) | C_air * term_rest_1 / den |

## Impact

- Case 640 annual cooling: 1.62 MWh vs 5.95-8.10 MWh expected (~75% under)
- Cases 600/610/620/630/650 fail nightly β-soak criteria 2/4/5
- Blocks gauge-solver production default flip (Issue #3291 Phase A8)

## Implementation Options

### Option A: Add Solar Lag State to GaugeSolver

Add a `solar_lag` state variable similar to 5R1C that filters high-frequency solar transients before injection to the air node.

**Files to modify**: `src/physics/gauge_zone_solver.rs`

**Complexity**: Medium - requires understanding the lag state machine

### Option B: Improve Air ODE Solution

Replace implicit Euler with exact exponential solution in the gauge air node update.

**Files to modify**: `src/physics/gauge_zone_solver.rs:802-808`

**Complexity**: Low - straightforward math change

### Option C: Combined Fix

Implement both solar lag correction AND exact exponential solution for air node.

**Complexity**: Medium-High - more comprehensive but addresses both damping issues

## Recommended Approach

**Option B (Air ODE fix) is the simplest starting point** since the math is straightforward. However, the solar lag correction (Option A) is what actually differentiates the 5R1C behavior for Cases 600 series.

The 5R1C solar lag correction filters solar gain through a geometric mean time constant (τ_lag = √(τ_air × τ_mass)), which better represents how thermal mass delays the zone's response to solar radiation.

## References

- KNOWN_ISSUES.md §LIMIT-21
- Issue #3911 (solar splitting - completed)
- Issue #1860 (5R1C solar lag correction)
- beta-soak-gate-escape-hatch.md
- beta-soak-criterion-2-tracker.md
