## Issue Description

The `FiveR1CSolver::step()` method in `src/physics/five_r1c_solver.rs` computes only steady-state heat flux (`Q = ΔT / R_total`) and **never updates the mass temperature** between timesteps.

**Current code (buggy):**
```rust
let Q = (T_ext - T_mass[i]) / R_total;  // T_mass never changes
// energy_storage_rate() returns 0.0
```

**Required (ISO 13790 5R1C transient):**
```rust
let dT_mass = (Q_ext + Q_int + Q_solar - Q_to_air) / C_mass;
T_mass[i] += dT_mass * dt;
let Q_to_air = (T_mass[i] - T_air) / R_1;
```

## Impact

- 6 transient tests are `#[ignore]` in `tests/conduction_5r1c_isolation.rs` (lines 459, 540, 598, 674, 744, 809)
- Zone cooling underestimates by ~90% (root cause of Case 900 failure)
- Night setback has no thermal effect
- No thermal lag — peak loads hit instantly

## Files Affected

- `src/physics/five_r1c_solver.rs` — `step()` method
- `tests/conduction_5r1c_isolation.rs` — 6 `#[ignore]` tests

## Acceptance Criteria

- [ ] Implement ISO 13790 5R1C transient mass temperature update
- [ ] Re-enable 6 transient tests, verify they pass
- [ ] Case 900 cooling energy reaches 8.00-10.50 MWh (currently 6.13 MWh)

## References

- ADR-002: `docs/adr/0002-promote-9r4c-high-mass-default.md`
- ARCHITECTURE.md lines 218-226 (documents the two 5R1C code paths)