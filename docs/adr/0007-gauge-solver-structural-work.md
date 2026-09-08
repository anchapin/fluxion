# ADR-0007: GaugeSolver Structural Work — Aggressive-Baseline Cohort Unblocker (Issue #3072)

- **Status:** ✅ Accepted (production-path switchover **shipped** via Phase A8, Issue #3291 / PR #3482, 2026-09-07; the `gauge-solver` cargo feature remains the production-path gate pending §LIMIT-21 / #3297 closure)
- **Date:** 2026-08-16
- **Accepted:** 2026-08-24 (per Issue #3172 — implementation plan now defined)
- **Phase A8 shipped:** 2026-09-07 (Issue #3291 / PR #3482, commit `e811df6`)
- **Deciders:** Fluxion maintainers
- **Supersedes:** None
- **Depends on:** None (this ADR records the gap; the actual structural fix is tracked by the underlying issues)
- **Issue:** [#3072](https://github.com/anchapin/fluxion/issues/3072) (meta-issue)
- **Related:** #1465 (Phase 3 GaugeSolver validation), #1462 (Phase 1b shadow-mode implementation), #3058 / #3059 / #3061 / #3062 / #3063 / #3060 / #3070 (cohort follow-ups)
- **Phase A8 cross-references:** `AGENTS.md:12` (Phase A8 note), `ARCHITECTURE.md:716-735` (Thermal selector Phase A8 note), `ARCHITECTURE.md:1267-1269` (Phase A8 status note), `docs/KNOWN_ISSUES.md` §LIMIT-21 (Issue #3297) + §LIMIT-22, `docs/agents/beta-soak-criterion-2-tracker.md`, `src/sim/thermal_selector.rs` (module docs)
- **Issue #3511:** refreshed the post-#3291 status. See the "Phase A8 implementation status" section below.

---

## Executive Summary

This ADR records the **production-path switchover scope** for the GaugeSolver
structural rework that unblocks the **aggressive-baseline cohort** (ASHRAE 140
Cases 195 / 600 / 620 / 940 / 960). The shadow-mode GaugeSolver is
implemented and validated (#1462, #1465 — both **Closed**). This ADR
commits the production-path switchover: `step_physics_5r1c` / `step_physics_9r4c`
routes to `GaugeSolver` as the default thermal solver.

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
- **#3291 (Phase A8: production-path switchover)** — ✅ **Closed** (PR #3482, 2026-09-07). The `Gauge` selector is now the unconditional default of `ThermalSelector::default()` (`src/sim/thermal_selector.rs:30-36`); with `--features gauge-solver` the dispatcher's gauge arm runs without fall-through to legacy 5R1C/9R4C and panics on missing gauge backend (`src/sim/thermal_model_physics/step_dispatcher.rs:126-133`). The `gauge-solver` cargo feature remains the production-path gate pending §LIMIT-21 (Issue #3297) closure — the β-soak gate (#3286) is at 0/30 nights green. The `Gauge` selector itself is the production default reached via `from_spec_with_selector` regardless of the cargo feature. The bare low-level constructor `ThermalModel::new` (Issue #3508) installs `ThermalSelector::legacy()` (`FiveROneC`) explicitly so it dispatches to the legacy 5R1C path even under `--features gauge-solver`.

## Phase A8 implementation status

What #3291 / PR #3482 actually shipped (the on-the-ground reality this ADR now reflects):

- **Selector-driven dispatch** — `ThermalSelector::default()` resolves to `ZoneSolverKind::Gauge` (`src/sim/thermal_selector.rs:30-36`, mirrored in `AGENTS.md:12`). The dispatcher at `src/sim/thermal_model_physics/step_dispatcher.rs:99-160` reads `self.0.hvac.thermal_selector.zone_solver` and routes accordingly.
- **`from_spec_with_selector` initialisation** — the production caller-facing entry point that initialises exactly one gauge backend per spec (single-zone for `num_zones == 1`, multi-zone for `num_zones >= 2`). The bare low-level `ThermalModel::new` does NOT call this and is now reconciled to `ThermalSelector::legacy()` (Issue #3508).
- **Unconditional-panic dispatcher** — under `--features gauge-solver` the cfg-gated gauge block is the only `Gauge` dispatch path. Missing or uninitialised gauge backend is a programming error that PANICS rather than silently switching to legacy 5R1C/9R4C. This is the §LIMIT-21 "fail-loud" design (Issue #3297).
- **`gauge-solver` cargo feature** — retained as the production-path gate. The cfg gate at `step_dispatcher.rs:99` and `step_dispatcher.rs:160` (and the corresponding `ThermalModelData` field declarations) is intentionally NOT removed in Phase A8; removal is the work tracked by #3290 (PR4 of the design tree, gated on the β-soak #3286).
- **Default build (no `gauge-solver` feature)** — the cfg-gated gauge block is absent and a `Gauge` selector falls through to legacy 5R1C/9R4C at the bottom of `step_physics`. Observable behaviour is the same as the pre-#3291 fall-through.

The architectural rationale (solar as geometric curvature, not per-timestep energy injection) is unchanged; only the production-path mechanism shifted from "fall-through on missing backend" to "panic on missing backend, behind a `gauge-solver` cfg gate".

## Production-Path Switchover Scope

### What changes

- **`src/sim/thermal_model_physics/physics_impl.rs`** — `step_physics_5r1c` and `step_physics_9r4c` route thermal coupling through `GaugeSolver` instead of the lumped thermal-mass node. The `GaugeSolver` path treats solar as geometric curvature (integrated surface-temperature response to irradiance history) rather than per-timestep energy injection into a single node.
- **`src/physics/physics_adapter.rs`** — the production thermal-model selection path calls `GaugeSolver::step` as the default instead of calling the legacy 5R1C/9R4C `step_physics_*` functions. The shadow-mode config flag is removed (GaugeSolver is now the primary path, not the secondary).
- **`src/sim/thermal_model.rs`** — `ThermalModelTrait::step` default implementation routes to `GaugeSolver`. The `HeatConductionSolver` swap point is unchanged; `GaugeSolver` implements `HeatConductionSolver`.
- **No changes to `src/physics/multi_node_solver.rs`** — the 9R4C multi-node path remains available as a fallback. No changes to `src/weather/`, `src/validation/`, or `tests/reference_data/`.

### Which LIMIT-* issues close when GaugeSolver ships

| LIMIT-* | Issue | Case / Metric | Structural signature |
|---------|-------|---------------|---------------------|
| **LIMIT-05** | #1280, #1457, #2453 | 900/910/920/930/940/950 peak cooling and annual energy | Discrete-node solar-injection pathology; bidirectional OVER+UNDER; `dt/τ ≈ 3.6` |
| **LIMIT-12** | #3062 | 940 annual heating | CTF-vs-blind 6–8× ratio; setback recovery overshoot |
| **LIMIT-13** | #2891, #3024 | 195 / 195FF surface balance | `h_tr_em` time-invariance in 5R1C path; wind-dependent correction |
| **LIMIT-14** | #3061 | 960 sunspace annual cooling | 5R1C air-mass distribution cannot accumulate back-zone cooling demand |
| **LIMIT-16** | #3059 | 610/630/650 peak cooling | 5/5 OVER signature; `dt/τ ≈ 3.6`; same root cause as LIMIT-05 |
| **LIMIT-17** | #3058 | 950FF min free-float | Night-vent `h_ve_night` overwhelms F_sky correction by ~8× |
| **LIMIT-18** | #3104 | 960 Blind `heating_max` | Case 960 Blind `heating_max = 2.45 MWh > 1.0 MWh AC4` upper bound |
| **LIMIT-20** | #3218 | Case 195 high-mass walls | Threshold updated `pass_rate >= 75.0` per Issue #3218 |

All seven LIMIT-* issues share the same root cause: a single lumped thermal-mass node at `dt/τ ≈ 3.6` cannot resolve multi-mode thermal coupling. GaugeSolver eliminates this by treating solar as geometric curvature rather than per-timestep energy injection.

## Test Acceptance Criteria

The production-path switchover is accepted when **all** of the following are true:

### v1.3 ASHRAE 140 validation gate

| Criterion | Value | Source |
|-----------|-------|--------|
| **ASHRAE 140 pass rate** | ≥ 60 % | `release_gates.yaml → validation.min_pass_rate` |
| **Mean Absolute Error (MAE)** | ≤ 50 % | `release_gates.yaml → validation.max_mae` |
| **Strict ±15% annual-energy band** | ≥ 60 % of all case/metric pairs in band | `ashrae_140_validator.rs` |

### Per-case cohort acceptance (all must flip from FAIL to PASS)

| Case | Metric | Pre-GaugeSolver | Post-GaugeSolver target |
|------|--------|-----------------|------------------------|
| 195 | Annual heating | 3238 kWh vs [0,0] ref; weather artefact | Within band or weather-file issue resolved per #3060 |
| 600 | Peak cooling | +48 % OVER band | Within ±15 % |
| 620 | Peak cooling | +11 % OVER band | Within ±15 % |
| 940 | Annual heating | 6.97 MWh vs [0.79, 1.41] MWh | Within ±15 % |
| 960 | Annual cooling | 8.85 MWh vs [1.55, 2.78] MWh | Within ±15 % |

### Required CI gates green

1. `cargo test --test ashrae_140_validation` — strict ±15% band gate
2. `cargo test --test zone_balance_eplus_isolation` — energy conservation gate
3. `cargo test --test integration-cli` — CLI behavior / stub guards
4. `cargo test -p fluxion --test ashrae_140_case_600_series` — Case 600 series (was 13/14 pass pre-GaugeSolver; target 14/14 or 15/15)
5. `tests/known_issues_regression.rs::issue_1457_case_600_series_tracking` — the 14 quarantined metrics flip green
6. `tests/gauge_validation_case_900.rs` — Case 900 high-mass validation harness passes
7. No `#[ignore]` removals that were not explicitly enabled by this PR

### What this ADR does NOT approve

- No relaxation of the strict ±15% band to absorb pre-existing 5R1C deviations.
- No raising of `tests/reference_data/zone_balance/strict_energy_gate_baseline.json` to hide regressions.
- No modification of `RULES.md`, `ARCHITECTURE.md`, or `AGENTS.md`.

## Milestone

- **Target:** v1.3 production release — GaugeSolver production-path switchover lands on `develop` (✅ SHIPPED via Phase A8 #3291 / PR #3482 on 2026-09-07; remaining work is §LIMIT-21 / β-soak #3286 + #3290 cfg-gate removal)
- **Verification:** ASHRAE 140 validation pass rate ≥ 60 %, MAE ≤ 50 % on `develop` CI (still pending; cohort entries in `docs/KNOWN_ISSUES.md` §LIMIT-21 / §LIMIT-22 track the residual work)
- **Issue for tracking:** #3172 (this ADR update); #3291 (Phase A8, shipped); #3286 (β-soak); #3297 (§LIMIT-21); #3290 (PR4 cfg-gate removal, gated on #3286)

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

**Accepted.** The production path for `step_physics_5r1c` / `step_physics_9r4c`
switches to `GaugeSolver` as the default thermal solver. The shadow-mode
implementation (#1462) and Case 900 validation harness (#1465) are both
**Closed**; the production-path switchover **shipped** via Phase A8
(#3291, PR #3482, 2026-09-07) — `ThermalSelector::default()` resolves to
`Gauge` and `from_spec_with_selector` initialises the matching gauge
backend. The switchover scope, acceptance criteria, and milestone are
defined above. Remaining work is the §LIMIT-21 / #3286 β-soak gate and
the #3290 cfg-gate removal (PR4).

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

- The `Status: Accepted` marker is set per Issue #3172. If the GaugeSolver
  rework is cancelled or re-routed, this ADR will be marked `Rejected` and
  the cohort will remain under the partial-fix wave approach.

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