# ADR-0007: GaugeSolver Structural Work — Aggressive-Baseline Cohort Unblocker (Issue #3072)

- **Status:** Proposed (tracking stub only — no architectural decision recorded)
- **Date:** 2026-08-16
- **Deciders:** Fluxion maintainers (TBD)
- **Supersedes:** None
- **Depends on:** None (this ADR records the gap; the actual structural fix is tracked by the underlying issues)
- **Issue:** [#3072](https://github.com/anchapin/fluxion/issues/3072) (meta-issue)
- **Related:** #1465 (Phase 3 GaugeSolver validation), #1462 (Phase 1b shadow-mode implementation), #3058 / #3059 / #3061 / #3062 / #3063 / #3060 / #3070 (cohort follow-ups)

---

## Executive Summary

This ADR is a **tracking stub** only. It records the fact that the
**aggressive-baseline cohort** (ASHRAE 140 Cases 195 / 600 / 620 / 940 / 960)
cannot be closed without the **GaugeSolver structural rework** tracked by
issues #1465 / #1462, and acknowledges the documentation gap that previously
left the cohort untracked at the cross-issue level. **It does not propose
architecture, does not record a decision, and does not modify physics code.**

Per **RULES.md** ("no parameter tuning", "must-never hardcode results"),
**AGENTS.md** ("fix the underlying math"), and **ADR-0001** (No-Parameter-
Tuning Rule), the only legitimate closure path for the cohort is the
GaugeSolver rework — a structural change that treats solar as geometric
curvature rather than per-timestep energy injection.

## Context

After the 24-wave orchestration (69 of 71 issues resolved), the strict ±15%
ASHRAE 140 pass rate remains low (post-#3044 / post-#2868: 12.5 % headline,
MAE 51.93 %). Five cases consistently fail outside the strict ±15% band and
share the same root cause — `step_physics_5r1c` / `step_physics_9r4c` use a
single lumped thermal-mass node that cannot capture multi-mode thermal
coupling accurately enough for the ASHRAE 140 reference ranges. Issue #3072
is the **meta-issue** that coordinates this cohort and explicitly states:
"This is a meta-issue coordinating the GaugeSolver structural work (#1465/
#1462). Track dependencies."

This ADR exists to give the cross-issue meta-tracking a single canonical
entry, complementing (not replacing) the per-issue entries in
`docs/KNOWN_ISSUES.md` §"Aggressive-baseline cohort tracking (Issue #3072)"
and the per-case diagnostics throughout §LIMIT-05, §LIMIT-08, and §SOLAR-02.

## Status of the underlying work

- **#1462 ([Physics] Phase 1b: Implement `GaugeSolver` in Shadow Mode inside `physics_adapter.rs`)** — ✅ **Closed**. Shadow-mode `GaugeSolver` is implemented and runs in parallel with the baseline solvers when a shadow-mode config flag is passed. Boundary-condition translation from raw solar irradiance and outside air temperature to `gauge_connection` vectors is shipped; the 100 kW HVAC clamp and matrix output bounds-clamp are removed.
- **#1465 ([Validation] Phase 3: Validate `GaugeSolver` against ASHRAE 140 Case 900)** — ✅ **Closed**. ASHRAE 140 Case 900 (High-Mass) validation harness ships via `tests/gauge_validation_case_900.rs`; diurnal temperature swings and phase lag are asserted against the ASHRAE analytical baseline.
- **Production-path switchover** — ❌ **NOT YET LANDED**. Both #1462 and #1465 ship the validation and shadow-mode paths but the production `step_physics_5r1c` / `step_physics_9r4c` paths have NOT been replaced by the `GaugeSolver`. This is the structural block that keeps the cohort pinned outside the ±15% band.

## What this ADR does NOT do

1. **It does not propose an architectural decision.** The GaugeSolver
   rework is already in motion via #1462 / #1465; this stub only
   acknowledges the cross-issue coordination gap surfaced by #3072.
2. **It does not modify physics code.** Per AGENTS.md and RULES.md, this
   meta-issue is documentation/tracking only.
3. **It does not modify `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`.**
   Per AGENTS.md, the strict-energy-gate baseline must NEVER be raised to
   hide a regression.
4. **It does not modify ARCHITECTURE.md or RULES.md.** Those are
   source-of-truth documents; this stub references them.
5. **It does not mark any case as passing.** It documents the structural
   blocker status only.

## Decision

**None recorded.** This is a tracking stub. The actual architectural decision
(production-path switchover from 5R1C/9R4C to `GaugeSolver`) is deferred to
a future ADR that will be co-authored with the structural PR that lands the
switchover. When that PR lands, this stub will be either:
- Superseded by a full ADR that records the switchover decision, the
  per-case metrics (Cases 195 / 600 / 620 / 940 / 960 must all pass the
  ±15% band), and the migration plan; OR
- Marked `Accepted` if the GaugeSolver path is adopted without further
  architectural debate.

## Consequences

### Positive

- The cross-issue coordination gap surfaced by #3072 has a single canonical
  ADR entry. Future contributors can trace the cohort to the GaugeSolver
  unblocker without having to reconstruct the dependency chain from the
  per-issue entries in `docs/KNOWN_ISSUES.md`.
- The stub explicitly states what it does NOT do, preventing misreading as
  a tuning escape hatch.

### Negative

- None. This is a tracking stub; it does not change any architecture,
  test, or pass-rate claim.

### Neutral

- The `Status: Proposed` marker is intentional and remains until the
  underlying structural PR lands. If the GaugeSolver rework is cancelled or
  re-routed, this stub will be superseded or marked `Rejected`.

## References

- Issue #3072 — meta-issue coordinating the GaugeSolver structural work
- Issue #1465 — Phase 3 GaugeSolver validation against ASHRAE 140 Case 900
- Issue #1462 — Phase 1b shadow-mode GaugeSolver in `physics_adapter.rs`
- `docs/KNOWN_ISSUES.md` §"Aggressive-baseline cohort tracking (Issue #3072)" — per-case status, dependent issues table
- `docs/KNOWN_ISSUES.md` §LIMIT-05 — discrete-node solar-injection pathology
- `docs/KNOWN_ISSUES.md` §LIMIT-05 UPDATE (#1522) — structurally infeasible at `dt/τ ≈ 3.6`
- `docs/KNOWN_ISSUES.md` §LIMIT-05 UPDATE (#2453) — 900-series bidirectional annual-energy over-prediction
- `docs/KNOWN_ISSUES.md` §LIMIT-05 UPDATE (#2452) — Case 940 CTF-vs-blind 6–8× ratio
- `docs/KNOWN_ISSUES.md` §LIMIT-08 — Case 195 weather-file peak-heating gap
- `docs/KNOWN_ISSUES.md` §SOLAR-02 UPDATE (#2239) — Case 900 residual deviation routed to GaugeSolver #1465
- `docs/gauge_solver_scalability.md` — `MultiZoneGaugeSolver` scalability characterisation (Issue #1771)
- `docs/ASHRAE140_RESULTS.md` §"Structural Blockers (Issue #3072)" — current pass-rate snapshot
- `RULES.md` — "no parameter tuning" + "must-never hardcode results"
- `AGENTS.md` — "fix the underlying math"; strict-energy-gate baseline must NEVER be raised
- ADR-0001 — No-Parameter-Tuning Rule
- ADR-0003 — ISO 13790 5R1C High-Mass Free-Float Temperature Limitations