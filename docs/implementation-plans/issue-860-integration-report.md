# Issue #860: Integration of Multi-Node HVAC Runner into step_physics Pipeline

**Status**: COMPLETED
**Parent Epic**: #856 — Full Multi-Node HVAC Energy Simulation

## Summary

The multi-node HVAC energy calculation has been integrated into the `step_physics` pipeline. The integration is **inline** within `step_physics_9r4c` in `src/sim/thermal_model_physics/physics_impl.rs`, not via the standalone `MultiNodeHvacRunner` (which is deprecated).

## Architecture Decision

**Decision**: Inline HVAC integration rather than `MultiNodeHvacRunner` wrapper

**Rationale**:
1. The `MultiNodeHvacRunner` was a test harness that duplicated thermal solver state
2. Inline integration avoids synchronization between runner and thermal model
3. Direct use of `multi_node_solvers` in `ThermalModel` provides proper energy accounting
4. The HVAC demand formula uses the multi-node air temperature for physically accurate setpoint control

## Integration Points in step_physics_9r4c

### 1. Multi-node t_air for HVAC Demand (Lines 2294-2323)

```rust
// Issue #860: Prefer multi-node t_air over 5R1C t_free for HVAC demand
let t_free_val = if i < self.0.multi_node_solvers.len() {
    _t_i_free_mn.as_ref().get(i).copied().unwrap_or_else(|| {
        t_i_free_5r1c.as_ref().get(i).copied().unwrap_or(20.0)
    })
} else {
    t_i_free_5r1c.as_ref()[i]
};
```

The multi-node solver's air temperature (`_t_i_free_mn`) is used when available, providing a physically accurate free-floating temperature from the 9R4C thermal balance.

### 2. Multi-node Mass Temperatures for Cooling (Lines 2343-2360)

```rust
let t_mass_mn = if i < self.0.multi_node_solvers.len() {
    let solver = &self.0.multi_node_solvers[i];
    let h_ms_total = solver.mass.wall.h_tr_ms + solver.mass.roof.h_tr_ms + solver.mass.floor.h_tr_ms;
    if h_ms_total > 1e-6 {
        (h_ms_w * solver.mass.wall.temperature + ...)
            / h_ms_total
    } else { ... }
} else { ... };
```

The conductance-weighted envelope temperature is used for the cooling demand formula.

### 3. Corrected HVAC Formula (Lines 2421-2434)

```rust
let q = if t_free_val < self.0.heating_setpoint {
    h_coeff * (self.0.heating_setpoint - t_free_val)
} else if t_mass_mn > self.0.cooling_setpoint {
    -h_coeff * (t_mass_mn - self.0.cooling_setpoint)
} else {
    0.0
};
```

## Case Routing

| Case Type | Model | HVAC Approach |
|-----------|-------|---------------|
| Case 600/610 (low-mass) | 5R1C/6R2C | Ventilation-based ideal loads |
| Case 900 (high-mass) | 9R4C | Per-surface multi-node HVAC |

The routing is automatic via `ThermalModel::step_physics` dispatcher:
- `is_nine_r4c_model()` → `step_physics_9r4c` (multi-node HVAC)
- Otherwise → `step_physics_5r1c`/6R2C (ventilation-based)

## Per-Surface Solar Gain Distribution (Issue #859)

Solar gains are distributed per surface in `step_physics_9r4c` (lines 1767-1800):
- `st_sol_frac` and `m_sol_frac` control beam-to-mass fraction
- `st_int_frac` and `m_air_frac` control internal gain distribution

This feeds into `step_per_surface()` for accurate per-surface temperature tracking.

## Deprecated MultiNodeHvacRunner

The `MultiNodeHvacRunner` in `src/sim/multi_node_hvac_runner.rs` is deprecated (since 0.9.0) because:

> "Use multi-node thermal model with inline HVAC control instead. The `MultiNodeSolver` now supports HVAC integration directly, providing better energy accounting and Crank-Nicolson time integration."

It remains as a reference implementation and for existing tests.

## Validation

The integration is validated through:
- ASHRAE 140 Case 900 tests in `ashrae_140_validator.rs`
- Per-surface conduction isolation tests in `tests/per_surface_conduction_isolation.rs`
- Multi-node solver unit tests in `src/physics/multi_node_solver.rs`

## References

- Issue #856: Full Multi-Node HVAC Energy Simulation (Epic)
- Issue #857: Per-surface conduction solver
- Issue #858: Multi-node solver with HVAC control logic
- Issue #859: Per-surface solar gain distribution
- Issue #860: This integration