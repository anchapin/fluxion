# ADR-002: Promote 9R4C Multi-Node Solver to Default for High-Mass Constructions

- **Status:** Proposed (awaiting sign-off)
- **Date:** 2026-06-20
- **Deciders:** @anchapin (Core Physics)
- **Resolves:** #1166
- **Supersedes / closes:** #1152, #1168
- **Depends on:** None (9R4C model already implemented and conditionally wired)

---

## Context

Issue #1166 asked which conduction/thermal solver should be the project default.
This decision was forced by four converging pieces of evidence gathered during
the #1152 investigation (the "spike"):

### 1. The steady-state 5R1C solver cannot resolve transient heat flow
`FiveR1CSolver` (`src/physics/five_r1c_solver.rs`, Module 3) is steady-state-only.
The conduction isolation test documents this verbatim:

> **CRITICAL:** `FiveR1CSolver::step()` ignores `timestep`, `h_interior`, and
> `h_exterior`. It computes only the steady-state flux `Q = ΔT / R_total`. The
> mass node `T_mass` is never updated, and `energy_storage_rate()` returns 0.0.
> Transient/time-constant tests are `#[ignore]` — require solver upgrade.

Per `BLIND_VALIDATION_RESULTS.md`, this is the dominant cause of ~40 of 48 blind
validation failures (cooling ratio ≈ 0.42, heating ratio ≈ 0.61). Peak loads are
averaged away and diurnal thermal storage is lost.

### 2. The "5R1C mass coupling" that #1152 targeted is NOT in Module 3
The ISO 13790 zone-level thermal network with the `h_ms_coeff` / `h_tr_ms` /
`h_tr_w` coupling lives in **`src/sim/thermal_model_core.rs` (Module 5, Zone
Balance)**, not in `FiveR1CSolver`. There are effectively *two code paths both
called "5R1C"*:

| Path | Location | Dynamic? |
|------|----------|----------|
| Per-wall steady-state solver | `physics/five_r1c_solver.rs` (Module 3) | No |
| Zone-level ISO 13790 network | `sim/thermal_model_core.rs` (Module 5) | Partially (coefficient-tuned) |

This is **architecture drift**: `ARCHITECTURE.md` line 212 implies the "5R1C
model" is a Module 3 `HeatConductionSolver` implementation, but the thermal
network that drives free-float and HVAC loads is the Module 5 zone solver.

### 3. The Module 5 coupling is coefficient-tuned, which AGENTS.md forbids
`thermal_model_core.rs:1034` sets `h_ms_coeff` per construction type
(`LowMass => 2.0`, `HighMass => 13.4`). PR #1151 reduced HighMass from 13.4 → 2.0
as a stability fix; that change over-damps the diurnal swing. `AGENTS.md` is
explicit: *"No parameter tuning to make system tests pass — fix the underlying
math."* The #1152 spike concluded the zone-level 5R1C topology's window↔mass
coupling cannot be restructured without either (a) touching Module 5 in violation
of module-isolation discipline, or (b) coefficient tuning. Neither is acceptable.

### 4. Free-float baseline confirms over-damping (issue #1168)
Measured on `main` (PR #1151's `h_ms_coeff`):

| Case | Metric | Simulated | ASHRAE 140 ref | Status |
|------|--------|-----------|----------------|--------|
| 900FF | T_max | 35.45 °C | [41.8, 46.4] | **FAIL** (over-damped) |
| 600FF | T_max | 54.59 °C | [43.6, 52.2] | **FAIL** (overshoots) |
| 900FF vs 600FF | swing reduction | 37 % | ≥ 10 % | PASS |

7 of 8 free-float metrics fail on the wrong side of the reference midpoint.

### 5. A validated dynamic alternative already exists
The **9R4C multi-node thermal model** (`src/sim/multi_node_thermal.rs`,
`src/physics/multi_node_solver.rs`) separates thermal mass into 4 nodes (wall,
roof, floor, internal) for heavy-mass buildings (Case 900+ series, #715). It is
**already conditionally wired** in `thermal_model_core.rs` (`is_9r4c_model`
branch, lines 1335–1523) and **already validated for ASHRAE 140 Case 900
multi-node HVAC**. The free-float over-damping persists because the legacy
`h_ms_coeff` coupling (line 1034) is still computed for all cases and appears to
drive the free-float path ahead of the 9R4C branch.

---

## Decision

**Promote the 9R4C multi-node thermal model to the default solver for high-mass
constructions** (Case 900+ series and high-mass free-floating cases). Low-mass
constructions retain the existing 5R1C path, which is adequate and fastest for
lightweight buildings.

In the issue's taxonomy this selects **Option B (9R4C promotion)**, gated by
construction type — functionally combining B with Option D's construction-type
selection, because (i) low-mass cases do not need a dynamic solver, (ii) the 9R4C
path is already conditionally wired for high-mass, and (iii) keeping 5R1C for
low-mass preserves its speed advantage for parametric studies.

Concrete implementation scope (tracked in a new issue, not #1152):

1. Make the 9R4C path the **sole** thermal solver for high-mass free-float and
   HVAC — bypass the legacy `h_ms_coeff`-based 5R1C coupling for high-mass so it
   no longer drives the free-float result.
2. Remove/retire the coefficient-tuned `h_ms_coeff` (13.4 / 2.0) for high-mass;
   rely on the 9R4C network's physics-based `h_tr_ms` (computed from `k·A/d`).
3. Keep CTF available as a secondary dynamic path; do **not** disable it, but do
   not make it the default (numerical instability in CTF↔5R1C coupling for 900FF,
   per #1152).
4. Re-run ASHRAE 140 blind validation and report the pass-rate delta.
5. Fix the architecture drift: update `ARCHITECTURE.md` to document the two
   solver paths accurately (per-wall 5R1C in Module 3 vs. zone-level 9R4C
   network in Module 5) and record this ADR's selection rule.

---

## Consequences

### Positive
- Removes the coefficient-tuning that violates `AGENTS.md` ("fix the math").
- 9R4C is a genuine dynamic (transient-capable) model → captures diurnal thermal
  storage and peak loads that steady-state 5R1C averages away.
- Reuses an already-implemented, already-validated solver — no new physics to
  write or vet.
- Directly addresses 7 of 8 failing free-float metrics (#1168) and the ~40
  system-level validation failures attributed to the steady-state limitation.
- Unblocks the v1.3 epic (#672) Phase D (80 % pass-rate target).

### Negative
- 9R4C is slower than 5R1C for high-mass cases (4 mass nodes vs. 1). Acceptable
  for validation; may need the ML surrogate (#1139) for high-throughput
  optimization sweeps over high-mass buildings.
- Some risk that the existing 9R4C HVAC validation does not automatically extend
  to free-float — the implementation must verify free-float specifically.
- CTF↔5R1C coupling instability for 900FF remains unresolved; CTF stays
  non-default. (Acceptable — 9R4C is the chosen dynamic path.)
- Closing #1152 marks its "restructure the 5R1C mass coupling" approach as not
  pursued; the rationale is preserved in this ADR.

### Neutral
- Low-mass behavior is unchanged (still 5R1C), so Case 600-series regressions
  are not expected, but must be verified.
- `MultiNodeModelType::Default` becomes `NineR4C`-for-high-mass rather than
  `FiveR1C` globally; the enum default stays `FiveR1C` for low-mass callers.

---

## Alternatives considered

- **Option A — Promote CTF to default.** Rejected: CTF is currently unstable for
  high-mass (900FF) due to CTF↔5R1C coupling issues (#1152 acceptance criteria
  to "enable CTF stably for high-mass" is itself unresolved). 9R4C is the
  lower-risk dynamic path that is already validated.

- **Option C — Keep 5R1C as default, accept the limitation.** Rejected: Phase D
  (80 % pass rate) is unreachable — the steady-state solver averages away peak
  loads and diurnal storage. The v1.3 epic (#672) would stay frozen indefinitely.

- **Option D (pure hybrid) — already absorbed.** This decision *is* a hybrid by
  construction type (9R4C high-mass, 5R1C low-mass). Listed separately in the
  issue only because it was framed as a distinct option; in practice it is the
  chosen scope.

- **Restructure the Module 5 5R1C topology (#1152 as written).** Rejected: the
  spike showed the target coupling is coefficient-tuned and lives in the wrong
  module per the current architecture; restructuring it would require either
  module-isolation violations or new coefficient tuning. 9R4C sidesteps both.

- **Move mass coupling down into Module 3 (make `FiveR1CSolver` dynamic).**
  Rejected for now: this requires a breaking `HeatConductionSolver` trait change
  and is the large "transient solver upgrade" the isolation test flags as
  blocked. May be revisited if 9R4C proves insufficient, but not on the critical
  path to Phase D.

---

## References

- #1166 — this decision (architectural decision: promote dynamic solver)
- #1152 — superseded by this ADR (5R1C mass-coupling restructure, not pursued)
- #1168 — superseded by this ADR (free-float over-damping, resolved via 9R4C)
- #715 — original 9R4C multi-node model implementation (Phase 6)
- #672 — v1.3 epic this unblocks (Phase D: 80 % blind-validation pass rate)
- `ARCHITECTURE.md` — Module 3 (Conduction) & Module 5 (Zone Balance) contracts
- `src/sim/multi_node_thermal.rs`, `src/physics/multi_node_solver.rs` — 9R4C
- `src/sim/thermal_model_core.rs:1034,1335-1523` — current coupling & 9R4C wiring
- `tests/conduction_5r1c_isolation.rs` — documents steady-state-only limitation
- ISO 13790:2008 — 5R1C / multi-node thermal network reference
