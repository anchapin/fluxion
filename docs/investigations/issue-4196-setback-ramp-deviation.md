# Issue #4196: Two-Hour Setback Ramp — Specified-Boundary Deviation for Cases 640 and 940

**Status:** Documented Deviation  
**Date:** 2026-09-28  
**Severity:** Specification Conformance  
**Affected Cases:** 640 (low-mass), 940 (high-mass)  
**Reference:** ASHRAE 140-2023 §5.5 (Thermostat Setback)

---

## 1. Summary

`HvacSchedule::heating_setpoint_at_fractional_hour` (fluxion-core/src/ashrae_cases.rs:646-713)
does not return the scheduled setpoint for wraparound overnight setback windows. When the schedule
carries a setback with `setback_end ∈ [1, 12]` and a non-zero setback delta, it blends linearly
from the setback value to the occupied value over **RAMP_HOURS = 2.0** starting at `setback_end`.

This is a **specified-boundary deviation** — the model solves a modified boundary condition with
no conformance record. It is distinct from issues #3062 and #2452 (solver overshoot on a correct
step input); this is a schedule deviation upstream of the solver, present regardless of which
conduction backend runs.

---

## 2. Affected Cases

| Case | Construction | Setback Window | Setback Temp | Occupied Temp | Ramp Window |
|------|-------------|----------------|--------------|---------------|-------------|
| 640  | Low-mass    | 23→7           | 10°C         | 20°C          | 07:00–09:00  |
| 940  | High-mass   | 23→7           | 10°C         | 20°C          | 07:00–09:00  |

### Ramp Profile (Cases 640/940, hours 07:00–09:00)

| Hour | Discrete Spec | Implemented | Delta |
|------|-------------|-------------|-------|
| 7.00 | 20.00°C     | 10.00°C     | -10.00°C |
| 7.25 | 20.00°C     | 11.25°C     | -8.75°C  |
| 7.50 | 20.00°C     | 12.50°C     | -7.50°C  |
| 7.75 | 20.00°C     | 13.75°C     | -6.25°C  |
| 8.00 | 20.00°C     | 15.00°C     | -5.00°C  |
| 8.25 | 20.00°C     | 16.25°C     | -3.75°C  |
| 8.50 | 20.00°C     | 17.50°C     | -2.50°C  |
| 8.75 | 20.00°C     | 18.75°C     | -1.25°C  |
| 9.00 | 20.00°C     | 20.00°C     | 0.00°C   |

The ramp is a linear interpolation: `setback + t × (occupied - setback)` where
`t = (fh - ramp_start) / RAMP_HOURS`.

---

## 3. Technical Details

### Ramp Eligibility Conditions

The ramp applies only when ALL of the following are true:

1. **Wraparound setback window**: `setback_start > setback_end` (e.g., 23 > 7)
2. **Morning setback end**: `setback_end ∈ [1, 12]`
3. **Non-zero delta**: `|setback_setpoint - heating_setpoint| > 1e-9`

```rust
// fluxion-core/src/ashrae_cases.rs:676-677
let ramp_eligible =
    sb_start > sb_end && sb_end >= 1 && sb_end <= 12 && (setback - occupied).abs() > 1e-9;
```

### Ramp Geometry

```rust
// fluxion-core/src/ashrae_cases.rs:678-682
const RAMP_HOURS: f64 = 2.0;
let ramp_start = f64::from(sb_end);           // e.g., 7.0 for setback_end=7
let ramp_end = (f64::from(sb_end) + RAMP_HOURS).min(24.0);  // 9.0
if fh >= ramp_start && fh < ramp_end {
    let t = (fh - ramp_start) / (ramp_end - ramp_start);
    let ramped = setback + t * (occupied - setback);
    // ...
}
```

---

## 4. Observed Metric Impact

The ramp biases morning peak heating downward because the zone temperature is held lower
during the ramp window. The zone must "catch up" after 09:00, reducing the instantaneous
heating demand in the morning hours.

### Case 640 (Low-Mass Setback)

| Metric | Reference Range | Observed | Direction |
|--------|----------------|----------|-----------|
| Peak Heating | 4.30 – 5.70 kW | 4.04 kW | **Under** |
| Annual Heating | 2750 – 3800 kWh | 2385 kWh | **Under** |

### Case 940 (High-Mass Setback)

The Case 940 schedule carries the same ramp-eligible configuration as Case 640
(wraparound setback 23→7, 10°C setback), and the conformance test confirms the
same 07:00–09:00 deviation pattern. Metric impact for Case 940 was **not
measured in this work**; the direction is expected to match Case 640 (morning
peak heating suppressed).

---

## 5. Governance

### Decision Record

**This deviation is NOT a bug to fix — it is a documented specification deviation.**

Related prior context: Issue #2870 ("Case 940 setback recovery: morning ramp needs
sub-hour HVAC mode interpolation") discusses the morning-ramp behavior for Case 940.
The ramp smooths the discrete step from setback (10°C) to occupied (20°C) at 07:00,
reducing the solver's tendency to overshoot the setpoint.

However, the ramp changes the **boundary condition** that the physics solver operates against.
The ASHRAE 140 spec defines a step change at 07:00; the implemented boundary is a 2-hour
linear ramp. This is a **model parameter change**, not a solver bug fix.

### Rules for Future Modification

1. **Keeping the ramp**: Any decision to retain the ramp behavior must cite this record
   and explain why the modified boundary condition is acceptable for ASHRAE 140 validation.

2. **Removing the ramp**: Removal is a physics change and must be:
   - Approved via an Architecture Decision Record (ADR)
   - Evaluated under RULES.md
   - NOT implemented as a band adjustment to make results "fit" the reference range

3. **Adjusting ramp width**: Changing RAMP_HOURS from 2.0 is a physics change requiring
   the same governance as removal.

4. **Adjusting validation bands**: Validation bands must NOT be widened to accommodate
   ramp-biased results. Bands are the specification; the implementation must match them.

---

## 6. Verification

The conformance test `schedule_conformance` (fluxion-core/tests/schedule_conformance.rs)
asserts that the implemented setpoint profile equals the discrete (no-ramp) profile at
every hour. When run, the four ramp-deviation tests (Cases 640/940, integer and
sub-hour) **fail** — the failure output above (§2 ramp profile) is the committed
record of that failure. To keep CI green, those four tests are ignored pending the
keep/remove decision (Issue #4226); all 12 remaining conformance tests pass.

Run with:
```bash
cargo test --package fluxion-core --test schedule_conformance
```

To see the documented deviation fail live (expected):
```bash
cargo test --package fluxion-core --test schedule_conformance -- --ignored
```

---

## 7. References

- ASHRAE 140-2023 §5.5 (Thermostat Setback and Setup)
- Issue #2870: Case 940 morning overshoot (original ramp introduction)
- Issue #3062: Solver overshoot on correct step input (separate issue)
- Issue #2452: Peak load solver behavior (separate issue)
- fluxion-core/src/ashrae_cases.rs:646-713 (`heating_setpoint_at_fractional_hour`)
- fluxion-core/tests/schedule_conformance.rs (conformance test)
- `RULES.md` §Physics Changes (governance for modifications)

---

## 8. Changelog

| Date | Change |
|------|--------|
| 2026-09-28 | Initial documented-deviation record (Issue #4196) |
| 2026-09-28 | 4 ramp-deviation conformance tests quarantined pending the keep/remove decision (#4226); Case 940 metric figures corrected to unmeasured |
