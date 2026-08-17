# ADR-0011: Case 950FF Night-Vent Mass Split — Implementation-Option Analysis (Issue #3058)

> **Summary 1/7:** Case 950FF min free-floating temperature is −23.92 °C against the ASHRAE 140 reference band −20.20 to −17.80 °C — 3.72 °C outside the band after PR #3040's per-surface F_sky view-factor correction.
> **Summary 2/7:** Root cause is `h_ve_night ≈ 570.8 W/K` overwhelming `h_tr_em_wall ≈ 71.6 W/K` by ~8× in `src/physics/multi_node_solver.rs::step_with_gains`'s mass-node update.
> **Summary 3/7:** The F_sky-weighted longwave correction is mathematically correct but effectively invisible against the dominant raw-outdoor forcing.
> **Summary 4/7:** Three proposed directions are (a) split air-node / surface-node mass coupling, (b) reduce `h_ve_night` by F_sky, (c) route `h_ve_night` only through the air node — all require solver code changes.
> **Summary 5/7:** This ADR is Proposed and records no implementation; per AGENTS.md / RULES.md / ADR-0001, no parameter tuning is permitted on `h_ve_night` to close the gap.
> **Summary 6/7:** The Case 950 (HVAC mode) annual cooling band (390–920 kWh) is the regression-avoidance clause for any future solver change.
> **Summary 7/7:** The structural fix is routed to the GaugeSolver production-path work (#1465 / #1462) coordinated by Issue #3059.

- **Status:** Proposed (tracking stub only — no implementation recorded)
- **Date:** 2026-08-17
- **Deciders:** Fluxion maintainers (TBD)
- **Supersedes:** None
- **Depends on:** Issue #3059 (GaugeSolver unblocker coordination); Issue #1465 / #1462 (GaugeSolver production-path switchover)
- **Issue:** [#3058](https://github.com/anchapin/fluxion/issues/3058)
- **Related:** #2872 (origin), PR #3040 (partial fix), #1898 (original `h_ve_night` introduction), #1422 (sister 5R1C night-vent override tracking), #3059, #1465, #1462, ADR-0007, ADR-0008, ADR-0009, ADR-0010

---

## Context

PR #3040 (issue #2872) introduced per-surface F_sky view factors for the
longwave sky-radiation correction. The fix moved Case 950FF min
free-floating temperature from −23.94 °C to −23.92 °C — an improvement of
0.02 °C, but the value is still **3.72 °C outside** the ASHRAE 140
reference band (−20.20 °C … −17.80 °C). The per-surface F_sky correction is
mathematically correct (applied to `t_ext_wall` via the longwave radiative
exchange on the exterior surface), but it is effectively invisible against
the dominant night-vent coupling.

In `src/physics/multi_node_solver.rs::step_with_gains` (lines 1069–1156),
the night-ventilation term is applied via `step_backward_euler_with_gains`
(lines 1164–1289) to each envelope mass node (wall, roof, floor) using the
raw outdoor air temperature as the driving temperature:

```text
// Update wall node — with gains and night ventilation (Issue #1898)
let denom = node.capacitance / dt + h_em + h_ms + h_ve_night;
let numer = node.capacitance / dt * node.temperature
    + h_em * t_ext_wall
    + h_ms * self.surface_temperature
    + h_ve_night * outdoor_temp          // <-- raw outdoor air
    + gains_wall;
```

For Case 950FF (post-#3040 measured by the validator path):

- `h_ve_night ≈ fan_capacity · ρ · cp / 3600 ≈ 570.8 W/K` (fan = 1703.16 m³/h,
  ACH ≈ 13.14 during 18:00–07:00, per
  `tests/ashrae_140_blind_validation.rs:2171`)
- `h_tr_em_wall ≈ 71.6 W/K` (the wall exterior-film / envelope-to-mass
  conductance)
- `h_ve_night / h_tr_em_wall ≈ 8.0` — the night-vent coupling to raw outdoor
  air overwhelms the wall exterior-film correction by ~8×.

`h_ve_night` was originally added by Issue #1898 to make Case 950 (HVAC
mode) night-vent *mass pre-cooling* work (the fan supply conductance
pre-cools the lumped mass node overnight so the morning cooling demand is
reduced). Removing it outright would:

1. Break Case 950 (HVAC mode) — the night flush would no longer pre-cool
   the mass, and the existing
   `test_case_950_mass_temperature_precooled_issue_1422` diagnostic (the
   only passing 5-day-July-overnight-ΔT > 2 °C test) would trip.
2. Mask the structural fix — the gap on Case 950FF is the same
   discrete-node pathology that the §LIMIT-05 cohort tracks and that the
   GaugeSolver rework (#1465 / #1462) is the architectural unblocker for.

The current Case 950 (HVAC) annual cooling is 33.08 kWh vs the reference
band 390–920 kWh — the band is far away and the architecture is **not**
this PR's scope. The Case 950 (HVAC) annual cooling band is the
regression-avoidance clause for any future solver change: any modification
that fixes Case 950FF but regresses Case 950 (HVAC) annual cooling is not
a valid closure.

## Decision

**No implementation is made in this PR.** This ADR remains **Proposed**
and records only the structural gap, the three proposed directions, their
risk / cost / benefit analysis, and the regression-avoidance clause. The
actual architectural decision between options (a) / (b) / (c) is deferred
to a future physics PR that:

1. Runs the Case 950 (HVAC) annual cooling regression-avoidance check
   (must remain in 390–920 kWh).
2. Runs the Case 950FF min free-floating temperature acceptance check
   (must be within −20.20 to −17.80 °C).
3. Runs the Case 900FF min free-floating temperature regression-avoidance
   check (no-night-vent Case 900FF must stay in −6.40 to −1.60 °C).
4. Verifies the F_sky correction double-check (the original PR #3040
   contribution is not undone).
5. Verifies energy balance, cross-case ASHRAE 140, architecture-drift,
   and cycle guards remain green without changing
   `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`.

This PR adds:

1. `docs/KNOWN_ISSUES.md` §LIMIT-17 as the canonical tracking entry.
2. `docs/ASHRAE140_RESULTS.md` §"Structural Blockers" entry for Case 950FF.
3. This ADR (`docs/adr/0011-case-950ff-night-vent-split.md`).
4. Regenerated `docs/doc-inventory.md` to include this ADR.

No production physics, validation implementation, solver code, strict
energy baseline, `ARCHITECTURE.md`, or `RULES.md` change is part of this
decision.

## Plan

Once the production GaugeSolver scope is clear (Issue #3059 coordinating
#1465 / #1462), maintainers must choose one of the three implementation
directions from Issue #3058:

### Option (a): Split `h_ve_night` into air-node mass (HVAC) and surface-node mass (FF) paths

**Mechanism:** Route `h_ve_night` to the air node only on the FF case
(Case 950FF), but keep the current mass-node direct coupling on the HVAC
case (Case 950). The HVAC pre-cooling works because the air node drives
the mass via the existing `h_tr_is` / `h_tr_ms` surface path; the FF case
loses the direct mass-node cooling but the longwave sky-radiation on
`t_ext_wall` (now F_sky-corrected per PR #3040) dominates.

**Risk:** Any drop in the multi-node coupling on Case 950 (HVAC) annual
cooling would regress the §LIMIT-05 / #1422 acceptance band regression
check. The split must be conditional on the case spec (HVAC vs FF), which
is a case-specific partial override — borderline parameter tuning per
RULES.md unless the case-spec conditional is documented as a first-
principles boundary (the FF case has no HVAC to drive the air-mass
coupling, so the air-resident fan mass becomes the only ventilation
effect).

**Cost:** Medium — requires new code path in `step_with_gains` plus
careful case-spec discriminator in `from_spec`.

**Benefit:** Preserves Case 950 (HVAC) pre-cooling; fixes Case 950FF
without globally reducing `h_ve_night`.

### Option (b): Reduce `h_ve_night` by F_sky on the mass coupling

**Mechanism:** Scale the night-vent forcing on the mass node by `F_sky`
(the same view factor PR #3040 introduced for the longwave correction) so
that the night-sky radiative exchange path is the dominant cooling pathway
on the FF case. The F_sky reduction only matters when the night fan is
active (18:00–07:00) — a case-specific partial override.

**Risk:** Per AGENTS.md / RULES.md risk classification, this is
**parameter adjustment** unless the F_sky reduction is derived from first
principles. The physically defensible motivation is the longwave
radiative exchange on the wall exterior surface: the night-sky view is
the dominant cooling pathway at night, and the F_sky-corrected
`t_ext_wall` is the right driving temperature for the mass coupling. If
the derivation is documented as first-principles (longwave radiative
exchange on the exterior surface), the reduction is justified; if it is
treated as a tuning constant, it is forbidden by RULES.md.

**Cost:** Low — single-line change in `step_backward_euler_with_gains`
plus documentation of the F_sky derivation.

**Benefit:** Closes the gap with the fewest solver-code changes; preserves
Case 950 (HVAC) pre-cooling (the air-node path is unchanged).

### Option (c): Route `h_ve_night` only through the air node

**Mechanism:** Remove the mass-node forcing entirely and rely on the
air-mass coupling via `h_tr_is` / `h_tr_ms` to drive the mass node via
the surface temperature. The air node's effective `h_ve_total = h_ve +
h_ve_night` already drives the mass node via the surface temperature
update at the end of `step_backward_euler_with_gains`.

**Risk:** Case 950 (HVAC) annual cooling may regress because the air node's
effective `h_ve_total` is used by the `t_i_free_mn` driving signal that the
HVAC controller sees — the air-node mass coupling is indirect and may not
pre-cool the mass fast enough over the 13-hour overnight window. The
`test_case_950_mass_temperature_precooled_issue_1422` test (the only passing
5-day-July-overnight-ΔT > 2 °C test) is the regression target.

**Cost:** High — requires a full rebalance of the air-mass coupling plus
extensive regression testing against the Case 950 (HVAC) annual cooling
band.

**Benefit:** Most architecturally clean — `h_ve_night` is treated as a
single fan-supply conductance, and the mass node update is purely
heat-transfer-driven.

Any implementation PR must satisfy all of the following:

- Case 950FF min free-floating temperature is **−20.20 to −17.80 °C** on
  the post-#3040 validator path.
- Case 950 (HVAC mode) annual cooling remains in the **390–920 kWh** band
  on the same validator path (regression-avoidance clause).
- Case 900FF min free-floating temperature stays in the current
  pass-band (−6.40 to −1.60 °C).
- Case 950FF annual cooling remains below 601 kWh (per the existing
  validator snapshot).
- Energy balance, cross-case ASHRAE 140, architecture-drift, and cycle
  guards remain green without changing
  `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`.
- No case-specific parameter, hardcoded output, or relaxed reference
  assertion is used to obtain the result.

## Consequences

### Positive

- The post-PR-#3040 state is recorded without misrepresenting the partial
  fix as a complete Case 950FF closure.
- The three implementation directions are documented with risk / cost /
  benefit analysis, giving the future solver implementer a clear
  decision matrix.
- The regression-avoidance clause for Case 950 (HVAC) annual cooling is
  attached to every option, so the future PR cannot close LIMIT-17 by
  regressing Case 950 (HVAC) annual cooling below 390 kWh.
- The architectural unblocker (GaugeSolver #1465 / #1462 per #3059) is
  explicitly listed, so the future implementer knows the long-term route
  even if a local fix is chosen.

### Negative

- Case 950FF remains outside the −20.20 to −17.80 °C free-floating min
  band on the validator path.
- The architectural decision between options (a) / (b) / (c) is deferred
  to a future physics PR that requires deep solver-code expertise and
  baseline-snapshot discipline.
- A real fix requires the GaugeSolver production-path switchover or a
  solver-code change under option (a) / (b) / (c) — both out of scope
  for this documentation PR.

### Neutral

- `docs/KNOWN_ISSUES.md` §LIMIT-09 (Case 950 5R1C free-float night-vent
  override, Issue #3071) remains the test-side tracking entry; §LIMIT-17
  is the structural-side tracking entry. The two are coupled through the
  same night-vent coupling-block.
- `docs/adr/0007-gauge-solver-structural-work.md` remains the cohort-level
  tracking stub; this ADR is the per-case (Case 950FF) decision matrix.
- The `Status: Proposed` marker is intentional and remains until the
  underlying structural PR lands. If the GaugeSolver rework is cancelled
  or re-routed, this ADR will be superseded or marked `Rejected`.

## References

- Issue #3058 — origin — Case 950FF night-vent mass coupling structural gap.
- Issue #2872 — origin of the original Case 950FF free-floating min
  over-prediction investigation.
- PR #3040 — `fix(physics): per-surface F_sky view factors for longwave
  sky-radiation correction` — the partial fix that #3058 follows up on.
- Issue #1898 — the original PR that introduced `h_ve_night` for Case 950
  (HVAC mode) mass pre-cooling.
- Issue #1422 — Case 950 5R1C night-vent override tracking (the
  structural-reduction sister issue, currently §LIMIT-09).
- Issue #2871 — sister issue — Case 950 / 950FF night-vent effective
  cooling tracking (closed by PR #3041 partial fix).
- Issue #3059 — 5R1C/9R4C architectural rework — the GaugeSolver
  unblocker.
- Issue #1465 — GaugeSolver ASHRAE 140 Case 900 validation work.
- Issue #1462 — GaugeSolver shadow-mode implementation/rework.
- ADR-0007 — GaugeSolver structural work stub (cohort-level tracking).
- ADR-0008 — ThermalModelData TDD-refactor tracking stub (controlled-
  delta baseline pattern).
- ADR-0009 — wind-dependent `h_tr_em` tracking stub.
- ADR-0010 — Case 940 CTF setback-recovery overshoot tracking stub.
- `docs/KNOWN_ISSUES.md` §LIMIT-09 — Case 950 5R1C free-float night-vent
  override (Issue #3071) — the test-side tracking entry.
- `docs/KNOWN_ISSUES.md` §LIMIT-17 — Case 950FF night-vent mass coupling
  structural gap (the canonical entry for this ADR).
- `docs/KNOWN_ISSUES.md` §LIMIT-05 — wider discrete-node structural
  limitation that the GaugeSolver work addresses.
- `src/physics/multi_node_solver.rs::step_with_gains` (lines 1069–1156)
  and `step_backward_euler_with_gains` (lines 1164–1289) — the mass-update
  path that `h_ve_night` enters.
- `tests/ashrae_140_blind_validation.rs::test_case_950_mass_temperature_precooled_issue_1422`
  (line 2189) — the passing Case 950 (HVAC mode) regression test that
  guards the pre-cooling path.
- `tests/ashrae_140_blind_validation.rs::test_case_950_5r1c_free_float_uses_night_vent_overrides_issue_1422`
  (line 2291) — the `#[ignore]`-quarantined Case 950FF integration test
  that pins the structural fix in step_physics_9r4c.
- `src/validation/benchmark.rs` Case 950 entries — the ASHRAE 140
  reference bands (annual cooling 390–920 kWh; peak heating 0.70–0.90 kW).
- `docs/ASHRAE140_RESULTS.md` §"Structural Blockers" — Case 950FF row
  added in this PR.
- `fluxion-core/src/weather/denver.rs` — `DenverTmyWeather`, the repo's
  synthetic weather source (annual min −12.47 °C).
- `RULES.md` — "no parameter tuning" + "must-never hardcode results".
- `AGENTS.md` — "do NOT modify physics code without checking
  `ARCHITECTURE.md` first"; "fix the underlying math"; strict-energy-gate
  baseline must NEVER be raised.
- `ADR-0001` — No-Parameter-Tuning Rule.
- `ARCHITECTURE.md` — module contracts and the 5R1C / 9R4C / GaugeSolver
  boundary that the §LIMIT-05 / §LIMIT-17 structural fix must respect.
