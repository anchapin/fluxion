# LIMIT-13 — investigation history

Narrative history for **LIMIT-13**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-13` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### LIMIT-13: `h_tr_em` (envelope-to-mass conductance) remains time-invariant in 5R1C path — tracking stub (Issue #3063)

- **Description:** Issue #3063 is the direct follow-up to PR #3024 / issue #2891.
  PR #3024 introduced wind-velocity-dependent `h_se` (the exterior film
  coefficient used for the sol-air longwave correction) per ASHRAE 140 §5.2.6
  via `physics::exterior_convection::h_c_ext_wind_dependent`. The sub-agent
  report explicitly noted: *"h_tr_em (envelope-to-mass conductance) remains
  time-invariant"*. That is the issue body: the build-time reciprocal
  `1/EXTERIOR_FILM_COEFF_DEFAULT` is still consumed at every timestep by the
  5R1C envelope-to-mass path without being recomputed from the per-step
  wind speed. The two conductances (exterior film `h_se` and
  envelope-to-mass `h_tr_em`) share the same `EXTERIOR_FILM_COEFF` source
  but live in different code paths; PR #3024 fixed the sol-air side, not
  the envelope-to-mass side.

- **Why this matters (per the issue acceptance criteria):**
  - Case 195 annual heating dropped from 7.42 MWh to 6.25 MWh post-#3024
    (≤ 6.30 MWh acceptance — **already met**).
  - Case 195 annual cooling went UP from 220 kWh to 758 kWh (target ≤ 50 kWh
    per issue acceptance — **open**, or ≤ 1500 kWh scoped-down full).
  - Case 195 peak heating ≤ 1.05 kW (already met, per Issue #2868 weather-file
    band; see LIMIT-08).
  - At low wind speeds (V ≈ 1–2 m/s, where `h_c` drops to 4–6 W/m²K) the
    wind-dependent `h_se` amplifies the sol-air longwave correction, shifting
    more hours into the cooling deadband (T_zone > 27 °C). The only way to
    close the 50 kWh Case 195 cooling target is to make `h_tr_em` wind-
    dependent at every timestep so the wall path aligns with the FD solver
    and the surface-balance paths.

- **Affected Cases:** 195 (primary), 600 / 620 (secondary — share the same
  5R1C envelope-to-mass path); any 5R1C construction whose
  `h_tr_em_zone` is constant at build time rather than recomputed per step.

- **Affected Metrics:** Case 195 annual cooling (kWh), Case 195 peak cooling
  (kW), Case 195 annual heating (kWh, marginal), annual cooling for the
  600 / 620 low-mass siblings.

- **Severity:** High (closes a Phase 16 partial fix; structurally infeasible
  to close the 50 kWh Case 195 cooling target without per-step recomputation).

- **GitHub Issue:** [#3063](https://github.com/anchapin/fluxion/issues/3063)
  (origin), with related issues **#2891** (original wind-dependent
  `h_se` request), **#3024** (the partial fix PR that closed the
  annual-heating half and exposed the cooling half), **#3059** (5R1C/9R4C
  architectural rework — the GaugeSolver unblocker), **#1465** / **#1462**
  (GaugeSolver shadow-mode and validation), **#2868** (sister issue — Case 195
  surface-balance initialisation fix, coupled through the same path).

- **Status:** 🔄 **Tracking stub — no physics-code change in this PR.**
  The closure path (per the issue's "Recommended Direction") is:

  1. Extend `HvacState` or `MassState` with `h_tr_em_zone: Vec<f64>`
     (per-zone, per-timestep), computed via
     `physics::exterior_convection::h_c_ext_wind_dependent` at the same
     cadence as the existing wind-dependent `h_se` (per step, sourced from
     `ThermalModelData::weather.wind_speed` via
     `wind_at_building_height_from_10m`).
  2. Recompute `h_tr_em_zone` at every timestep in `step_physics_5r1c`
     (the helper is already in `physics_impl.rs:155`).
  3. Update the `EnergyPlus-equivalent baseline` invariant check
     (`validation/ashrae_140_cases.rs` or whichever field tracks the
     `h_tr_em` reference) to read the per-step value rather than the
     build-time constant.

  Each of these is a structural solver-code change that, per
  **AGENTS.md** ("do NOT modify physics code without checking
  `ARCHITECTURE.md` first"), **RULES.md** ("no parameter tuning",
  "must-never hardcode results"), and **ADR-0001** (No-Parameter-Tuning
  Rule), cannot be done by a single sub-agent without (a) deep
  physics expertise, (b) bit-identical or controlled-delta baseline
  snapshots (per **ADR-0008**), and (c) coordination with the
  GaugeSolver rework (#1465/#1462 per **#3059**). This entry is
  **documentation/tracking only** — it does not propose, suggest, or
  hint at a tuning fix.

- **What this PR ships (documentation/tracking scaffolding):**
  1. **This LIMIT-13 entry** — categorises the gap, links to #2891 /
     #3024 / #3059 / #1465 / #1462 / #2868, and gives the implementer
     a concrete acceptance criterion (Case 195 cooling ≤ 50 kWh full
     or ≤ 1500 kWh scoped).
  2. **ADR-0009 (`docs/adr/0009-h-tr-em-wind-dependent.md`)** — the
     closing architectural decision record, with the same "Proposed
     tracking stub" status as ADR-0007 / ADR-0008 (no architectural
     decision recorded). The ADR documents the implementation plan
     and the dependencies on **ADR-0008** (snapshot diff verifier for
     bit-identical baselines) and **#3059** (GaugeSolver unblocker).
3. **`scripts/verify_h_tr_em_regression.py`** — *removed 2026-08-19 as orphan (see `.agents/results/result-pm.md`); was a snapshot diff verifier mirroring the `scripts/verify_gauge_solver_regression.py` pattern from #3070. Exit codes follow the same `EXIT_OK=0 / EXIT_REGRESSION=1 / EXIT_PLACEHOLDER=2 / EXIT_USAGE=3` contract. Fail-closed by default: a placeholder snapshot set (no `captured_at`) trips exit 2 so a future implementer cannot silently compare against an empty baseline. The `--strict` flag adds a SHA-256 fingerprint check. A future PR that submits the actual per-step recompute must re-derive this verifier from the contract documented here and in ADR-0009 §2.*
   4. **`scripts/ci/test_verify_h_tr_em_regression.py`** — *removed 2026-08-19 as orphan; was a pytest harness covering placeholder detection, no-drift, regression, tolerance-override, schema-drift, missing-manifest, JSON output, `--strict` SHA-256 mismatch, and CLI tolerance-override scenarios (mirror of `test_verify_gauge_solver_regression.py`).*
  5. **§"Aggressive-baseline cohort tracking (Issue #3072)"** row
     already lists #3063 as a dependent issue (line 1127) — no
     change to that table required by this PR.

- **What this PR does NOT do (and why):**
  1. **It does NOT modify physics code.** Per AGENTS.md, the actual
     `h_tr_em_zone: Vec<f64>` extension and the per-step
     recomputation are deferred to a future PR that runs the
     verifier end-to-end against a bit-identical baseline (captured
     via **ADR-0008**'s pattern).
  2. **It does NOT modify
     `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`.**
     Per AGENTS.md, the strict-energy-gate baseline must NEVER be
     raised to hide a regression. The bit-identical baseline for the
     TDD approach lives in a separate snapshot set
     (`tests/reference_data/h_tr_em_baseline/`, to be created by the
     future implementer; not created by this PR — the verifier
     rejects placeholder snapshots until real measurements are
     captured).
  3. **It does NOT modify ARCHITECTURE.md or RULES.md.** Those are
     source-of-truth documents; this stub references them.
  4. **It does NOT record an architectural decision.** The actual
     extend-HvacState-or-MassState choice is deferred to the future
     PR that submits both the recomputation and the verifier output.
  5. **It does NOT mark any case as passing.** The Case 195 cooling
     target (≤ 50 kWh full or ≤ 1500 kWh scoped) is the acceptance
     criterion for the future PR, not this one.

- **Why this is NOT a fixable tuning change (per AGENTS.md / RULES.md /
  ADR-0001):**
  1. The 220 kWh → 758 kWh cooling shift is the **bidirectional
     signature** of an incomplete wind-dependent correction: the sol-air
     side (`h_se`) is wind-dependent, but the envelope-to-mass side
     (`h_tr_em`) is not. Per ISO 13790 §C.3 / ASHRAE 140 §5.2.6, both
     conductances must be wind-dependent for the cooling-deadband hours
     to align with the FD solver. Forcing the cooling back into band
     by adjusting `h_se`, `h_tr_em`, or any 5R1C constant would be
     **parameter tuning to pass a system test** — explicitly forbidden
     by AGENTS.md ("fix the underlying math").
  2. The structural fix is the `h_tr_em_zone: Vec<f64>` per-step
     recomputation, which is the same class of solver-code change as
     #3059 / #3061 / #3062 — out of scope for a single sub-agent
     without physics deep-dive, baseline-snapshot discipline, and
     coordination with the GaugeSolver rework.
  3. The genuinely architectural fix is the GaugeSolver rework
     (#1465 / #1462), which treats solar as geometric curvature rather
     than per-timestep energy injection. #1462 (Phase 1b `GaugeSolver`
     implementation) and #1465 (Phase 3 ASHRAE 140 Case 900 validation
     harness) are both **closed** individually; the
     production-path switchover is **staged** — Phase A8 (#3291, PR
     #3482) wires `ThermalSelector::default() = ZoneSolverKind::Gauge`
     and the dispatcher's unconditional gauge default, but the
     `gauge-solver` cargo feature remains the production-path gate
     pending §LIMIT-21 closure (the β-soak program, Issue #3286). The
     default build (feature OFF) routes the `Gauge` selector to
     5R1C/9R4C via the dispatcher's `match` arm — see `AGENTS.md` and
     `ARCHITECTURE.md` Module 5 for the Phase A8 notes and
     `docs/adr/0007-gauge-solver-structural-work.md` §"Status of the
     underlying work" for the historical tracking.

- **Per-step `h_tr_em` semantics (documentation for the future
  implementer):**
  - The current production path consumes `h_tr_em` as a constant per zone
    (a build-time reciprocal of `EXTERIOR_FILM_COEFF_DEFAULT`).
  - The fault-tolerant recomputation is `h_tr_em_step[i] = 1.0 /
    h_c_ext_wind_dependent(ExteriorSurfaceDirection::VerticalWallWindward,
    v_building_at_step[i])` where `v_building_at_step[i]` is the
    per-step wind speed at building mid-height (sourced from the
    per-step weather buffer via `wind_at_building_height_from_10m`).
  - At V = 3.4 m/s, `h_c_ext_wind_dependent(VerticalWallWindward, 3.4)`
    returns 17.6 W/m²K, which yields `h_tr_em_step = 0.0568 m²·K/W`
    — within the 5 % band of the legacy `1/EXTERIOR_FILM_COEFF =
    0.0546 m²·K/W` (the residual 0.7 W/m²K is the longwave radiative
    portion that is added on the sol-air side, per
    `src/physics/exterior_convection.rs:128-135`).
  - The production-path side already does this for the sol-air
    longwave correction (see `physics_impl.rs:339-362`); the
    `h_tr_em_zone` extension generalises the same per-step recompute
    to the envelope-to-mass conductance and lets the 5R1C wall path
    align with the FD solver / surface-balance paths.

- **Related sections in this document:**
  - §LIMIT-08 (Case 195 weather-file peak-heating gap — sister issue,
    closed via Issue #2868 / PR #3044).
  - §"Aggressive-baseline cohort tracking (Issue #3072)" — Case 195
    is in the 5-case GaugeSolver-blocked cohort; #3063 is listed as
    a dependent issue (line 1127).
  - §SOLAR-02 UPDATE (Issue #2239) — Case 900 deviation routed to
    GaugeSolver #1465 (same structural pattern).
  - §LIMIT-05 UPDATE (Issue #2300) — sub-hour air-node sub-stepping
    also **blocked by GaugeSolver** (same route).

- **External references:**
  - Issue #3063 (origin) — also PR #3024 (partial fix) and the
    sub-agent report that flagged the gap directly.
  - Issue #2891 — original wind-dependent `h_se` request that PR #3024
    closed for the sol-air side only.
  - PR #3024 — `h_se` wind-dependent closure (annual heating
    7.42 → 6.25 MWh; exposes the cooling shift documented here).
  - Issue #2868 (Case 195 surface-balance initialisation fix — coupled
    to the same envelope-to-mass path; closed via PR #3044).
  - `src/physics/exterior_convection.rs` — `h_c_ext_wind_dependent`
    and `wind_at_building_height_from_10m` (the helpers the future
    implementer must call from `step_physics_5r1c`).
  - `src/sim/thermal_model_physics/physics_impl.rs:155` —
    `step_physics_5r1c` (the per-timestep loop where the future
    implementer must inject the per-step `h_tr_em_zone` recompute).
  - `docs/adr/0008-thermal-model-data-tdd-refactor.md` — the
    snapshot-diff verifier pattern (#3070) that the now-removed
    `verify_h_tr_em_regression.py` mirrored.
  - `docs/adr/0007-gauge-solver-structural-work.md` — the
    architectural unblocker (Phase A8 production-path switchover landed
    via Issue #3291 / PR #3482, gated on the `gauge-solver` cargo
    feature / §LIMIT-21 β-soak closure).
  - `RULES.md` — "no parameter tuning" + "must-never hardcode results".
  - `AGENTS.md` — "do NOT modify physics code without checking
    ARCHITECTURE.md first"; strict-energy-gate baseline must NEVER be
    raised.
  - `ADR-0001` — No-Parameter-Tuning Rule.
  - `Wave 14–22 partial-fix PRs #3040, #3041, #3042, #3044, #3052`
    (each closes a subset of the cohort; none closes the structural
    block that #3063 belongs to).
