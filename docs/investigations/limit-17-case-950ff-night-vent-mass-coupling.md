# LIMIT-17 — investigation history

Narrative history for **LIMIT-17**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-17` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### LIMIT-17: Case 950FF night-vent mass coupling overwhelms F_sky correction — tracking stub (Issue #3058)

- **Description:** Issue #3058 is the direct follow-up to PR #3040 / Issue #2872.
  PR #3040 introduced per-surface F_sky view factors for the longwave
  sky-radiation correction on the wall exterior-film path. The fix moved the
  Case 950FF **annual min free-floating temperature** from −23.94 °C to
  −23.92 °C — an improvement of 0.02 °C, but the result is still **3.72 °C
  outside** the ASHRAE 140 reference band (−20.20 °C … −17.80 °C). The
  per-surface F_sky correction is mathematically correct (applied to
  `t_ext_wall` via the longwave radiative exchange on the exterior surface),
  but it is effectively invisible against the dominant night-vent coupling
  described below.

- **Root cause (re-investigated by Issue #3058):** In
  `src/physics/multi_node_solver.rs::step_with_gains` (lines 1069–1156) the
  night-ventilation term is applied via `step_backward_euler_with_gains`
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

  - `h_ve_night ≈ fan_capacity · ρ · cp / 3600 ≈ 570.8 W/K`
    (fan = 1703.16 m³/h, ACH ≈ 13.14 during 18:00–07:00, per
    `tests/ashrae_140_blind_validation.rs:2171`)
  - `h_tr_em_wall ≈ 71.6 W/K` (the wall exterior-film / envelope-to-mass
    conductance)
  - `h_ve_night / h_tr_em_wall ≈ 8.0` — the night-vent coupling to raw outdoor
    air overwhelms the wall exterior-film correction by ~8×.

  The F_sky-weighted longwave correction on `t_ext_wall` only enters the
  mass update via the `h_em · t_ext_wall` term (weight ≈ 71.6 W/K), which
  is ~8× smaller than the `h_ve_night · outdoor_temp` term (weight
  ≈ 570.8 W/K). The F_sky correction is therefore *physically correct but
  mathematically dominated* by the raw outdoor coupling.

- **Why previous fix was incomplete (PR #3040 / Issue #2872):** `h_ve_night`
  was originally added by Issue #1898 to make Case 950 (HVAC mode) night-vent
  *mass pre-cooling* work (the fan supply conductance pre-cools the lumped
  mass node overnight so the morning cooling demand is reduced). Removing
  `h_ve_night` outright would:
  1. *Break Case 950 (HVAC mode)* — the night flush would no longer pre-cool
     the mass, and the existing `test_case_950_mass_temperature_precooled_issue_1422`
     diagnostic (the only passing 5-day-July-overnight-ΔT > 2 °C test) would
     trip. Current Case 950 (HVAC) annual cooling is 33.08 kWh vs the reference
     band 390–920 kWh — the band is far away and the architecture is **not**
     this PR's scope.
  2. *Mask the structural fix* — the gap on Case 950FF is the same
     discrete-node pathology that the §LIMIT-05 cohort tracks and that the
     GaugeSolver rework (#1465 / #1462) is the architectural unblocker for.
     Removing `h_ve_night` would be a **parameter tuning in band space** and
     is explicitly forbidden by AGENTS.md / RULES.md / ADR-0001.

- **Three proposed directions (per Issue #3058 body), all requiring solver
  code changes — none auto-implementable in this PR:**

  - **(a) Split `h_ve_night` into air-node mass (HVAC) and surface-node
    mass (FF) paths.** The air-node-mass coupling keeps the Case 950
    pre-cooling working (the lumped mass sees the cool air via the
    `h_tr_is`/`h_tr_ms` surface path); the surface-node-mass coupling is
    removed for the FF case so the `h_ve_night · outdoor_temp` forcing
    stops dominating `h_em · t_ext_wall`. This is a **solver-code change**
    that requires deep physics expertise and a controlled-delta baseline.
    Risk: any drop in the multi-node coupling on Case 950 (HVAC) annual
    cooling would regress the §LIMIT-05 / #1422 acceptance band regression
    check.

  - **(b) Reduce `h_ve_night` by F_sky on the mass coupling.** Scale the
    night-vent forcing on the mass node by `F_sky` (the same view factor
    PR #3040 introduced for the longwave correction) so that the night-sky
    radiative exchange path is the dominant cooling pathway on the FF case.
    This is **parameter adjustment** per AGENTS.md / RULES.md risk
    classification, and the F_sky reduction only matters when the night fan
    is active (18:00–07:00) — a case-specific partial override. Risk:
    forbidden by RULES.md "no parameter tuning" / "must-never hardcode
    results" unless the F_sky reduction is derived from first principles
    (the longwave radiative exchange on the wall exterior surface is the
    physically defensible motivation; the engineering case is documented
    in `docs/adr/0011-case-950ff-night-vent-split.md`).

  - **(c) Route `h_ve_night` only through the air node.** Remove the
    mass-node forcing entirely and rely on the air-mass coupling via
    `h_tr_is` / `h_tr_ms` to drive the mass node via the surface temperature.
    This is a **solver-code change** (delete the `h_ve_night · outdoor_temp`
    term from `step_backward_euler_with_gains` and rebalance the air
    node's `h_ve_total = h_ve + h_ve_night` term in
    `compute_zone_air_temperature`). Risk: Case 950 (HVAC) annual cooling
    may regress because the air node's effective `h_ve_total` is used by
    the `t_i_free_mn` driving signal that the HVAC controller sees — the
    air-node mass coupling is indirect and may not pre-cool the mass fast
    enough over the 13-hour overnight window.

- **Why this is NOT a fixable tuning change (per AGENTS.md / RULES.md / ADR-0001):**
  1. Closing the 3.72 °C gap by adjusting `h_ve_night`, `h_tr_em`, or any
     5R1C / 9R4C constant would be **parameter tuning to pass a system
     test** — explicitly forbidden by AGENTS.md ("fix the underlying math")
     and RULES.md ("no parameter tuning", "must-never hardcode results").
  2. The three proposed directions above each require solver-code changes
     that must be evaluated against the Case 950 (HVAC mode) annual cooling
     acceptance band (390–920 kWh per
     `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`
     — Case 950 is currently 33.08 kWh, i.e. **below band by ~91 %**, so
     the HVAC mode is the *more* sensitive regression target, not the FF
     mode). Any change that fixes the FF mode but breaks the HVAC mode is
     not a valid closure.
  3. The structural fix is the **GaugeSolver rework (#1465 / #1462)**,
     which treats solar and envelope heat transfer as geometric curvature
     rather than per-timestep energy injection. #1462 (Phase 1b shadow-mode
     implementation) and #1465 (Phase 3 ASHRAE 140 Case 900 validation
     harness) are both **closed** individually, but the **production-path
     switchover** is NOT yet landed — see
     `docs/adr/0007-gauge-solver-structural-work.md` §"Status of the
     underlying work".

- **Affected Tests:**
  - `tests/ashrae_140_case_950ff_free_floating.rs` (Case 950FF free-float
    annual min / max envelope — the directly-failing band).
  - `tests/ashrae_140_blind_validation.rs::test_case_950_5r1c_free_float_uses_night_vent_overrides_issue_1422`
    (the integration test, currently `#[ignore]`-quarantined per #3071 with
    the reason *"Pre-existing failure tracked in #3071; blocked by #1422
    + GaugeSolver #1465/#1462; once structural fix lands, re-test"* — the
    same limitation as LIMIT-09 under a different framing).
  - `tests/ashrae_140_blind_validation.rs::test_case_950_mass_temperature_precooled_issue_1422`
    (companion case 950 HVAC-mode test that **PASSES** — the per-day
    overnight ΔT > 2 °C assertion). This is the regression target for any
    future `h_ve_night` split per option (a) / (c).

- **Affected Metrics:** Case 950FF min free-floating temperature (°C) — the
  diagnostic / reference-band metric that the ASHRAE 140 reference band
  closes. The Case 950 (HVAC mode) annual cooling / peak heating / peak
  cooling metrics are the **regression-target** criteria for any future
  solver change (must remain in the 390–920 kWh / 0.70–0.90 kW / 0.70–0.90 kW
  bands respectively per `validation/benchmark.rs`).

- **Severity:** High (closes the Case 950FF annual-min band — a free-floating
  diagnostic metric that the validator path measures on every CI run). The
  Case 950 (HVAC mode) annual cooling band is currently far below 390 kWh
  (33.08 kWh per the 2026-08-16 snapshot in `docs/ASHRAE140_RESULTS.md`),
  so any regression on Case 950 (HVAC) annual cooling is **not** the
  limiting factor for this LIMIT-17 entry — the regression is guarded
  separately by the §LIMIT-05 / #1422 chain.

- **GitHub Issue:** [#3058](https://github.com/anchapin/fluxion/issues/3058)
  (this entry), with related issues **#2872** (origin — Case 950FF
  free-floating min over-prediction; PR #3040 partial fix), **#3040** (the
  PR that introduced the per-surface F_sky view-factor correction),
  **#1898** (the PR that originally introduced `h_ve_night` for Case 950
  HVAC-mode mass pre-cooling), **#1422** (Case 950 5R1C night-vent
  override tracking — the structural-reduction sister issue),
**#3059** (5R1C/9R4C architectural rework — the GaugeSolver unblocker),
  **#1465 / #1462** (GaugeSolver implementation and validation harness —
   both closed individually; production-path switchover staged via
   **#3291 / PR #3482** (Phase A8 default flip) — `ThermalSelector::default() =
   ZoneSolverKind::Gauge` is wired but the `gauge-solver` cargo feature
   remains the production-path gate pending §LIMIT-21 closure).
  Long-term fix routed to GaugeSolver rework **#1465 / #1462**, which
  treats solar / envelope heat transfer as geometric curvature rather than
  per-timestep energy injection (per AGENTS.md / RULES.md "fix the
  underlying math"; per-case parameter tuning to close this gap is
  explicitly out of scope).

- **Status:** 🟡 **Documentation/tracking only — no solver-code change in
  this PR.** The three proposed directions above are routed to the GaugeSolver
  architectural fix and to a future physics PR that satisfies the
  regression-avoidance clause for Case 950 (HVAC mode) annual cooling.
  The architectural decision between options (a) / (b) / (c) is recorded
  in **`docs/adr/0011-case-950ff-night-vent-split.md`** (Proposed; tracking
  stub — no implementation recorded). The existing
  `tests/ashrae_140_blind_validation.rs::test_case_950_5r1c_free_float_uses_night_vent_overrides_issue_1422`
  `#[ignore]` quarantine (per §LIMIT-09 / #3071) is sufficient for the
  test surface until the structural fix lands; no new test
  addition / modification is required by this PR.

- **What this PR ships (documentation/tracking scaffolding):**
  1. **This LIMIT-17 entry** — categorises the gap, links to #2872 / #3040
     / #1898 / #1422 / #3059 / #1465 / #1462, and lays out the three
     implementation options with their risk / cost / benefit analysis.
  2. **`docs/adr/0011-case-950ff-night-vent-split.md`** — the architectural
     decision record (Status: Proposed; tracking stub), with the same
     "no architectural decision recorded" status as ADR-0007 / ADR-0008 /
     ADR-0009 / ADR-0010. The ADR documents the implementation plan,
     the regression-avoidance clause for Case 950 (HVAC mode), and the
     dependencies on **#1465 / #1462** (the GaugeSolver unblocker).
  3. **§"Structural Blockers" entry in `docs/ASHRAE140_RESULTS.md`** — the
     Case 950FF row is added to the cohort table with the LIMIT-17 + #3058
     + #2872 + #1465 / #1462 reference chain, mirroring the Case 195 /
     600 / 620 / 940 / 960 entries already in the table.
  4. **Regenerated `docs/doc-inventory.md`** — the auto-generated inventory
     gains the new `docs/adr/0011-case-950ff-night-vent-split.md` entry.
  5. **Top-of-file `*Last Updated*` header** — updated to
     *"2026-08-17 (LIMIT-17 #3058 added — Case 950FF night-vent mass
     coupling; LIMIT-14 #3061 merged with LIMIT-15 #3060 from #3096)"*,
     keeping the existing LIMIT-14 / LIMIT-15 merge-note intact.

- **What this PR does NOT do (and why):**
  1. **It does NOT modify `src/physics/multi_node_solver.rs`** — per
     AGENTS.md ("do NOT modify physics code without checking
     `ARCHITECTURE.md` first"), the actual `step_with_gains` /
     `step_backward_euler_with_gains` changes are deferred to a future
     PR that runs the regression-avoidance check (Case 950 HVAC annual
     cooling stays in 390–920 kWh) and the F_sky-correction double-check
     simultaneously.
  2. **It does NOT modify `src/sim/`, `src/physics/`, or `src/validation/`.**
     Per the CRITICAL SCOPE CONSTRAINT in the Issue #3058 PR template.
  3. **It does NOT modify `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`.**
     Per AGENTS.md, the strict-energy-gate baseline must NEVER be raised
     to hide a regression.
  4. **It does NOT modify ARCHITECTURE.md or RULES.md.** Those are
     source-of-truth documents; this entry references them.
  5. **It does NOT record an architectural decision.** The actual
     air-node / surface-node / F_sky split choice is deferred to the
     future PR that submits both the solver-code change and the
     regression-avoidance evidence.
  6. **It does NOT mark Case 950FF as passing.** The Case 950FF min
     target (−20.20 to −17.80 °C) is the acceptance criterion for the
     future PR, not this one.
  7. **It does NOT remove the `#[ignore]` on
     `test_case_950_5r1c_free_float_uses_night_vent_overrides_issue_1422`**
     — that quarantine is governed by §LIMIT-09 / #3071 and the
     GaugeSolver production-path switchover, not by this documentation
     PR.

- **Acceptance for the future structural PR:**
  1. Case 950FF min free-floating temperature is within −20.20 to −17.80 °C
     on the post-#3040 validator path.
  2. Case 950 (HVAC mode) annual cooling remains in the 390–920 kWh band
     (regression-avoidance clause — any drop below 390 kWh indicates
     the `h_ve_night` modification has broken the night-vent
     pre-cooling).
  3. Case 900FF min free-floating temperature stays in the current
     pass-band (−6.40 to −1.60 °C) — the analogous free-floating winter
     min metric on the no-night-vent Case 900FF.
  4. Case 950FF annual cooling remains below 601 kWh (per the existing
     `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`
     — Case 950FF is NOT in the strict gate baseline per `release_gates.yaml`
     known structural failures, but is in the validator snapshot).
  5. Energy balance, cross-case ASHRAE 140, architecture-drift, and
     cycle guards remain green without changing
     `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`.

- **Per-step `h_ve_night` semantics (documentation for the future implementer):**
  - The current production path (post-#1898) consumes `h_ve_night` as a
    constant per Case 950 spec (≈ 570.6 W/K for the 1703.16 m³/h fan).
  - The `h_ve_night` is sourced from the Case 950 night-ventilation
    schedule (active 18:00–07:00 per the ASHRAE 140 §7.3 spec) and is
    passed to `step_with_gains` as an argument — its value is NOT
    recomputed per-step from the air-mass distribution.
  - The F_sky correction (PR #3040) is applied to `t_ext_wall` via the
    longwave radiative exchange on the exterior surface; the F_sky
    view factor is stored per-surface (north / south / east / west /
    roof / floor) in `ThermalModelData::exterior_surface_view_factors`.
  - The gap is the *mass coupling topology* (the `h_ve_night · outdoor_temp`
    term in the mass-node denominator / numerator), not the value of
    `h_ve_night` itself. Any reduction of `h_ve_night` would be a
    parameter tuning unless derived from first principles (the
    physically defensible motivation is the longwave radiative exchange
    on the exterior surface at night, which is the same path the F_sky
    correction addresses).

- **Why this PR is documentation-only (per AGENTS.md / RULES.md):**
  1. The 3.72 °C gap is the **structural** signature of the same
     discrete-node solar-injection pathology that the §LIMIT-05 cohort
     tracks and that the GaugeSolver rework (#1465 / #1462) is the
     architectural unblocker for. Closing the gap by adjusting
     `h_ve_night` or `h_tr_em` would be **parameter tuning to pass a
     system test** — explicitly forbidden by AGENTS.md ("fix the
     underlying math").
  2. The three proposed directions above each require solver-code
     changes that, per AGENTS.md, cannot be done by a single sub-agent
     without (a) deep physics expertise, (b) bit-identical or
     controlled-delta baseline snapshots (per **ADR-0008**), and (c)
     coordination with the GaugeSolver rework (#1465 / #1462 per
     **#3059**).
  3. The Case 950 (HVAC mode) annual cooling band is the
     regression-avoidance clause that any future solver change must
     satisfy — the band is currently 390–920 kWh and the engine output
     is 33.08 kWh, so the regression-avoidance check is *more*
     sensitive than the LIMIT-17 acceptance check (Case 950FF min within
     −20.20 to −17.80 °C). A solver change that fixes Case 950FF but
     regresses Case 950 (HVAC) is not a valid closure.

- **Related sections in this document:**
  - §LIMIT-09 — Case 950 5R1C free-float night-vent override (Issue #3071)
    — the pre-existing test failure on the same night-vent path, also
    quarantined pending the GaugeSolver unblocker.
  - §LIMIT-05 — discrete-node solar-injection pathology (the wider
    structural limitation that the GaugeSolver rework addresses).
  - §"Aggressive-baseline cohort tracking (Issue #3072)" — Case 950FF
    is implicitly part of the cohort via Issue #3058 (the open-issue
    row at line 1131).
  - §SOLAR-04 — Night Ventilation Cooling Ineffective (legacy §issue #276) —
    the original Case 650 / Case 950 night-vent tracking entry, now
    re-routed to #1422 per the Issue Table at the bottom of this document.

- **External references:**
  - Issue #3058 (origin) — Case 950FF night-vent mass coupling structural gap.
  - Issue #2872 — origin of the original Case 950FF free-floating min
    over-prediction investigation.
  - PR #3040 — `fix(physics): per-surface F_sky view factors for
    longwave sky-radiation correction` — the partial fix that
    #3058 follows up on.
  - Issue #1898 — the original PR that introduced `h_ve_night` for
    Case 950 (HVAC mode) mass pre-cooling.
  - Issue #1422 — Case 950 5R1C night-vent override tracking (the
    structural-reduction sister issue).
  - Issue #2871 — sister issue — Case 950 / 950FF night-vent effective
    cooling tracking (closed by PR #3041 partial fix).
  - Issue #3059 — 5R1C/9R4C architectural rework — the GaugeSolver
    unblocker.
  - Issue #1465 / #1462 — GaugeSolver validation and `GaugeSolver`
    implementation; the Phase A8 production-path switchover is staged
    via Issue #3291 / PR #3482 — `ThermalSelector::default() =
    ZoneSolverKind::Gauge` is wired but the `gauge-solver` cargo
    feature remains the production-path gate pending §LIMIT-21 closure.
  - `src/physics/multi_node_solver.rs::step_with_gains` (lines 1069–1156)
    and `step_backward_euler_with_gains` (lines 1164–1289) — the
    mass-update path that `h_ve_night` enters.
  - `tests/ashrae_140_blind_validation.rs::test_case_950_mass_temperature_precooled_issue_1422`
    (line 2189) — the passing Case 950 (HVAC mode) regression test that
    guards the pre-cooling path.
  - `tests/ashrae_140_blind_validation.rs::test_case_950_5r1c_free_float_uses_night_vent_overrides_issue_1422`
    (line 2291) — the `#[ignore]`-quarantined Case 950FF integration
    test that pins the structural fix in step_physics_9r4c.
  - `src/validation/benchmark.rs` Case 950 entries — the ASHRAE 140
    reference bands (annual cooling 390–920 kWh; peak heating 0.70–0.90 kW).
  - `docs/adr/0007-gauge-solver-structural-work.md` — the architectural
    unblocker (#1465 / #1462 production-path switchover).
  - `docs/adr/0011-case-950ff-night-vent-split.md` — the
    implementation-option analysis (this PR's contribution).
  - `RULES.md` — "no parameter tuning" + "must-never hardcode results".
  - `AGENTS.md` — "do NOT modify physics code without checking
    `ARCHITECTURE.md` first"; strict-energy-gate baseline must NEVER
    be raised.
  - `ADR-0001` — No-Parameter-Tuning Rule.
  - `ADR-0007` — GaugeSolver structural work stub.
  - `ADR-0008` — ThermalModelData TDD-refactor tracking stub
    (controlled-delta baseline pattern).
  - `ADR-0009` — wind-dependent `h_tr_em` tracking stub.
  - `ADR-0010` — Case 940 CTF setback-recovery overshoot tracking stub.
