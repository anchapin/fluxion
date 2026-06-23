## Issue Description

ARCHITECTURE.md line 212 implies `HeatConductionSolver` is the zone-level 5R1C solver. **ADR-002 reveals this is false**:

| Path | Location | Purpose | Dynamic? |
|------|----------|---------|----------|
| `FiveR1CSolver` | `physics/five_r1c_solver.rs` | Per-surface conduction | No — steady-state only |
| Zone-level network | `thermal_model_core.rs` | **Actual** zone solver | Yes |

The Module 3 `FiveR1CSolver::step()` ignores timestep and film coefficients — it computes only `Q = ΔT / R_total`. The Module 5 thermal network is what drives the zone balance.

## Documentation Fix Required

Update ARCHITECTURE.md to explicitly state:
- Module 3 = per-surface, steady-state conduction solver
- Module 5 = zone-level thermal network (5R1C/9R4C)

## Related Issue

This also relates to the 5R1C transient bug — users may assume Module 3 handles zone-level dynamics when it doesn't.

## Files Affected

- `ARCHITECTURE.md:218-226` (ADR-002 note already exists but core docs don't reflect it)
- `src/physics/solver_trait.rs`

## Acceptance Criteria

- [ ] ARCHITECTURE.md clearly distinguishes Module 3 (per-surface) from Module 5 (zone-level)
- [ ] Module contracts explicitly state which solver handles zone dynamics
- [ ] No ambiguity about which 5R1C implementation to use for what purpose