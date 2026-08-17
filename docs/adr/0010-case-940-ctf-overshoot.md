# ADR-0010: Case 940 CTF Setback-Recovery Overshoot Tracking (Issue #3062)
> **Summary 1/7:** Case 940 annual heating is 1,289.9 kWh on the blind diagnostic path but 7,487.81 kWh on the CTF validator path.
> **Summary 2/7:** PR #3042 delivered sub-hour setback interpolation and corrected the per-zone heating-setpoint vector refresh.
> **Summary 3/7:** The remaining roughly 6× overshoot is structural CTF timestep aggregation across the 23:00–07:00 setback boundary.
> **Summary 4/7:** This ADR is Proposed and records no solver implementation or production-physics change.
> **Summary 5/7:** This PR adds LIMIT-12 plus ignored ratio and recovery-window diagnostics only.
> **Summary 6/7:** A future implementation must choose CTF sub-stepping, blind-path integration parity, or GaugeSolver adoption.
> **Summary 7/7:** CTF-path closure requires Case 940 annual heating in the 790–1,410 kWh reference band without parameter tuning.

- **Status:** Proposed (tracking stub only — no implementation recorded)
- **Date:** 2026-08-17
- **Deciders:** Fluxion maintainers (TBD)
- **Supersedes:** None
- **Depends on:** Issue #3059 and clarification of GaugeSolver #1465/#1462 production scope
- **Issue:** [#3062](https://github.com/anchapin/fluxion/issues/3062)
- **Related:** #2870, PR #3042, #3059, #1465, #1462, ADR-0008, ADR-0009

---

## Context

PR #3042 followed Issue #2870 by adding
`HvacSchedule::heating_setpoint_at_fractional_hour`, which interpolates the
Case 940 morning recovery instead of applying the entire occupied-setpoint
change as one discrete hourly jump. The same PR fixed a latent wiring bug:
the per-zone `heating_setpoints` vector had been refreshed only when
`spec.hvac.len() > 1`. After that gate was removed, the Case 940 ramped
setpoint reached the CTF physics path for the single-zone validator.

The blind diagnostic path now reports **1,289.9 kWh** annual heating, within
the ASHRAE 140 Case 940 reference range of **790–1,410 kWh**. The CTF
validator path reports **7,487.81 kWh**, a CTF/blind ratio of approximately
**5.80×** in Issue #3062's annual-heating snapshot and consistent with the
historical **6–8×** annual/peak path divergence documented by Issue #2870.
The CTF history integrates the per-zone vector over a one-hour timestep while
the thermostat crosses the 23:00 setback and 07:00 recovery boundaries. The
resulting transfer-function correction overshoots the blind-path recovery
load rather than smoothing the boundary response.

`src/sim/hvac_controller.rs:89-121` consumes the active setpoints when it
selects heating/cooling mode and calculates capacity-limited power. That file
is read-only for this work. The unresolved divergence belongs to solver
integration and cannot be closed by changing controller thresholds,
coefficients, reference bands, or case-specific constants.

## Decision

**No implementation is made in this PR.** This ADR remains **Proposed** and
records only the structural gap, its dependencies, and the diagnostic surface
needed to evaluate a later solver change. This PR adds:

1. `docs/KNOWN_ISSUES.md` §LIMIT-12 as the canonical tracking entry.
2. `test_case_940_blind_vs_ctf_ratio_pinned`, ignored by default, to preserve
   the current CTF/blind divergence as an explicit close-out signal.
3. `test_case_940_setback_recovery_window_diagnostic`, ignored by default, to
   expose zone, mass, setpoint, and heating-energy evolution across the
   23:00–07:00 transition.

No production physics, validation implementation, HVAC controller, strict
energy baseline, `ARCHITECTURE.md`, or `RULES.md` change is part of this
decision.

## Plan

Once the production GaugeSolver scope is clear, maintainers must choose one of
the three implementation directions from Issue #3062:

1. **Option (a): smooth CTF timestep aggregation.** Advance the CTF solver
   through multiple sub-hour steps, with the issue proposing four solver steps
   per weather hour, so the setback boundary is not integrated as one full-hour
   change.
2. **Option (b): match the blind-path integration methodology in CTF mode.**
   Preserve CTF envelope dynamics while applying the same setpoint/recovery
   integration semantics that produce the 1,289.9 kWh blind diagnostic result.
3. **Option (c): wait for GaugeSolver #1465/#1462.** Replace or bypass the
   structurally limited CTF/5R1C coupling when the GaugeSolver production
   switchover coordinated by Issue #3059 is ready.

Any implementation PR must satisfy all of the following:

- Case 940 annual heating is **790–1,410 kWh** on the **CTF validator path**.
- The blind path remains within its current reference band.
- Case 940 cooling remains within band or remains explicitly tracked under
  LIMIT-05 until the GaugeSolver work lands.
- Energy balance and the existing bottom-up module-isolation gates remain
  green.
- No case-specific parameter, hardcoded output, or relaxed reference assertion
  is used to obtain the result.

## Consequences

### Positive

- The post-PR-#3042 state is recorded without misrepresenting the partial fix as
  a complete CTF correction.
- Future solver work has two focused, ignored diagnostics for annual path ratio
  and setback-boundary state evolution.
- The acceptance band is attached specifically to the CTF validator path, so a
  blind-path-only success cannot close Issue #3062.

### Negative

- Case 940 remains outside the heating reference band on the CTF validator path.
- The ignored diagnostics are informational unless explicitly run with
  `--ignored --nocapture`.
- A real fix requires a substantial solver-integration change under option (a)
  or (b), or the wider GaugeSolver production switchover under option (c).

### Neutral

- Issue #3059 remains the architectural unblocker coordinating GaugeSolver
  #1465/#1462 scope. This ADR does not pre-select GaugeSolver over a CTF-local
  implementation.
- ADR-0008 and ADR-0009 establish the companion tracking-stub convention:
  status remains Proposed until a measured implementation and its validation
  evidence land together.

## References

- Issue #2870 — origin of the Case 940 setback-recovery investigation
- PR #3042 — sub-hour HVAC interpolation and per-zone vector refresh partial fix
- Issue #3062 — CTF-path overshoot follow-up and three implementation options
- Issue #3059 — architectural unblocker for the aggressive-baseline cohort
- Issue #1465 — GaugeSolver ASHRAE 140 validation work
- Issue #1462 — GaugeSolver shadow-mode implementation/rework
- ADR-0008 — ThermalModelData TDD-refactor tracking stub
- ADR-0009 — wind-dependent `h_tr_em` tracking stub
- `docs/KNOWN_ISSUES.md` §LIMIT-05 — wider discrete-node structural limitation
- `docs/KNOWN_ISSUES.md` §LIMIT-12 — canonical Case 940 CTF overshoot entry
- `tests/diagnostics/case_940_setback_diagnostic.rs` — diagnostic implementation
- `RULES.md` — no parameter tuning and no hardcoded physics results
- `ARCHITECTURE.md` — CTF/GaugeSolver module contracts and validation strategy
