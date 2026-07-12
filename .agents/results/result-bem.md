# Issue #1527 — GaugeSolver Case 600 Integration

## Status: PARTIAL — architectural blocker identified

## Summary

Wired the existing GaugeSolver into the Case 600 validation path via two
non-breaking changes:

1. **Option (a)**: Registered `"gauge"` key in `SolverRegistry::construct` —
   GaugeSolver is now constructible through the standard solver-dispatch
   mechanism alongside `"5r1c"` and `"multinode_9r4c"`.
2. **Option (b)**: Created `tests/gauge_validation_case_600.rs` — an 8-test
   harness paralleling `gauge_validation_case_900.rs`, validating the
   GaugeSolver per-wall behavior against Case 600 low-mass geometry.

## Architectural Finding (Critical)

**The 13 failing Case 600 zone-level tests cannot be closed by the per-wall
GaugeSolver as currently implemented.** Root cause analysis:

| Layer | What it does | Can fix the 13 tests? |
|-------|-------------|----------------------|
| GaugeSolver (`gauge_solver.rs`) | Per-wall steady-state flux via sol-air temp (`energy_storage_rate = 0`) | **No** — per-wall, not zone-level |
| PhysicsAdapter (`physics_adapter.rs`) | Shadow mode — records diagnostics, returns baseline flux (line 110) | **No** — doesn't perturb primary path |
| Zone balance (`physics_impl.rs:325`) | `sol_to_air = sol_w * 0.7` — window solar → air node | This IS the failure site |

The 13 failures are **zone-level window-solar-distribution** problems, not
per-wall conduction problems. The code comment at `thermal_model_core.rs:1799-1835`
explicitly states the fix requires issue #1152 (structural 5R1C air-node
capacitance rewrite) — setting `air_frac = 0.0` trades peak-cooling-over for
annual-cooling-under (the 5R1C cannot simultaneously match both).

## Files Changed

| File | Lines | Description |
|------|-------|-------------|
| `src/physics/solver_registry.rs` | +42 | GAUGE key + construct dispatch + test |
| `tests/gauge_validation_case_600.rs` | +475 (new) | 8-test Case 600 gauge validation harness |

## Acceptance Criteria Checklist

- [x] GaugeSolver wired into Case 600 validation path (option a + b)
- [x] Case 600 gauge validation harness passes (8/8)
- [x] Case 900 validation: no regression
- [x] Lib tests: no regression
- [x] SolverRegistry GAUGE key constructible + tested
- [ ] 13 zone-level Case 600 tests closed — **BLOCKED by architectural gap**
      (per-wall GaugeSolver cannot reach zone solar-distribution path;
      requires #1152 structural rewrite or full zone-level gauge integration)
