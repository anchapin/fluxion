# LIMIT-28 — investigation history

Narrative history for **LIMIT-28**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-28` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### LIMIT-28: ASHRAE 140 Free-Floating cohort (600FF/650FF/900FF/950FF) — 5R1C + 9R4C + multi-node-night-vent structural gap, Phase B3b PHYSICS-03 (Issue #3802)

- **Description:** Phase B3b PHYSICS-03 (Issue #3802) — corrects
  free-floating temperature prediction per B3a's ranking so FF cases
  land within ASHRAE 140 ranges. Per the issue acceptance criteria
  ("FF cases within ASHRAE ranges OR LIMIT entries with named
  mechanisms") and the AGENTS.md / RULES.md / ADR-0001
  no-parameter-tuning rule, the post-#1323 measurement for the four
  FF cases is documented as a structural LIMIT entry rather than
  patched. The 2026-09-16 validator snapshot reports the cohort:

  | Case   | Min (engine) | Min (ref band)        | Verdict (min)       | Max (engine) | Max (ref band)   | Verdict (max)       |
  |--------|--------------|-----------------------|---------------------|--------------|------------------|---------------------|
  | 600FF  | −17.13 °C    | [−18.80, −15.60]      | below by 1.33 °C    | 55.22 °C     | [64.90, 75.10]   | below by 9.68 °C    |
  | 650FF  | −23.71 °C    | [−23.00, −21.00]      | below by 0.71 °C    | 52.43 °C     | [63.20, 73.50]   | below by 10.77 °C   |
  | 900FF  | −6.65 °C     | [−6.40, −1.60]        | below by 0.25 °C    | 39.83 °C     | [41.80, 46.40]   | below by 1.97 °C    |
  | 950FF  | −23.95 °C    | [−20.20, −17.80]      | below by 3.72 °C    | 31.30 °C     | [35.50, 38.50]   | below by 4.20 °C    |

  All four cases fail on both metrics; the cohort signature is
  bidirectional (winter min too warm AND summer max too cool) — i.e.
  the engine damps the diurnal swing more aggressively than the ASHRAE
  140 reference programs. The cohort is *not* in `release_gates.yaml
  -> validation.individual.known_failures` (per the AGENTS.md /
  RULES.md principle "structural failures are documented, not
  patched"), and the strict ±15% annual-energy gate cannot lift
  without closing it.

- **Named mechanisms (per case):**

  1. **Case 950FF** (high-mass + night vent): the
     `h_ve_night ≈ 570.8 W/K` fan supply during 18:00–07:00
     overwhelms the wall exterior-film correction
     `h_tr_em_wall ≈ 71.6 W/K` by ~8× in
     `src/physics/multi_node_solver.rs::step_with_gains`, locking the
     winter min at −23.95 °C vs the [−20.2, −17.8] °C band. The
     PR #3040 F_sky view-factor correction is mathematically correct
     but invisible against the dominant raw-outdoor forcing (3.72 °C
     outside band is the §LIMIT-17 / #3058 / ADR-0011 documented gap).
     **Mechanism: discrete-night-vent mass coupling on 5R1C/9R4C;
     architectural fix routed to GaugeSolver.**

  2. **Case 900FF** (high-mass, no night vent): the winter min is
     0.25 °C outside the [−6.4, −1.6] °C band — the smallest margin
     in the cohort but still outside. The summer max (39.83 °C vs
     [41.8, 46.4] °C) is ~2 °C below the band. **Mechanism: same
     residual as Case 950FF but without the night-vent amplification;
     the 5R1C/9R4C single lumped-mass-node damping under-predicts
     the high-mass diurnal swing.**

  3. **Case 650FF** (low-mass + night vent): winter min −23.71 °C is
     0.71 °C below the [−23.0, −21.0] °C band. The low-mass envelope
     cannot accumulate the thermal inertia needed to damp the cold
     snap, but the high night-vent ACH (13.14) drives the air node
     below the mass node's natural cooling rate. **Mechanism:
     low-mass + high-ACH night-vent air-node undershoot; the F_sky
     correction on `t_ext_wall` does not transfer to the air node
     through `h_tr_is` / `h_tr_ms` at the rates ASHRAE 140 expects.**

  4. **Case 600FF** (low-mass, no night vent): winter min −17.13 °C
     is 1.33 °C below the [−18.8, −15.6] °C band. The 5R1C
     lumped-mass under-predicts the diurnal damping the ASHRAE 140
     reference expects for a low-mass envelope. **Mechanism: 5R1C
     single-lumped-mass-node low-mass under-damping; the air-mass
     distribution alone cannot capture the ASHRAE 140 reference
     thermal response.**

- **Cross-references:**

  - §LIMIT-17 / Issue #3058 / ADR-0011 — Case 950FF night-vent mass
    coupling structural gap (Status: Proposed; the Option (a) / (b) /
    (c) decision matrix is documented in ADR-0011 §Plan).
  - §LIMIT-24 / Issue #3551 — Case 950 HVAC-mode annual cooling
    companion (the §LIMIT-17 regression-avoidance clause; the HVAC
    mode and FF mode are bidirectionally coupled — HVAC cooling UNDER
    + FF min UNDER — both routed to GaugeSolver; the §LIMIT-24 entry
    documents the HVAC-mode end of the bidirectional signature).
  - §LIMIT-16 / Issue #3059 — Cases 610 / 630 / 650 peak cooling OVER
    (the `MAX_CONVECTIVE_TO_AIR_MULTIPLIER = 2.0×` cap closed Case 650
    cooling OVER; the Case 650 FF cohort gap is structurally distinct
    from the cooling-mode OVER signature — Case 650 HVAC is in-band,
    Case 650FF is outside the band).
  - §LIMIT-05 UPDATE (#2453) — 900-series bidirectional annual-energy
    over-prediction (the air-mass distribution pathology is the same
    as the FF cohort's diurnal-swing damping).
  - §FREE-01 / §FREE-02 / §FREE-03 — pre-existing FREE-\* family
    documenting the same cohort gaps from the pre-#1323 baseline; the
    2026-09-16 snapshot numbers above are the post-#1323 ground truth.
  - ADR-0001 (No-Parameter-Tuning Rule) — per the rule, no coefficient
    adjustment to `h_ve_night`, `MAX_CONVECTIVE_TO_AIR_MULTIPLIER`,
    `h_tr_em_wall`, `solar_distribution_to_air`, or any other 5R1C /
    9R4C parameter is permitted to close the FF cohort gap; closing
    it requires a structural solver-code change routed to the
    GaugeSolver production-path work.
  - ADR-0011 (Case 950FF Night-Vent Mass Split, Status: Proposed) —
    the per-case decision matrix for the Case 950FF gap closure.
  - ADR-0007 (GaugeSolver Structural Work, Status: Accepted —
    production-path switchover staged via #3291 / PR #3482, gated on
    §LIMIT-21 β-soak closure).
  - `src/sim/ventilation.rs` — the `VentilationSchedule` swap point
    documented by this entry's §"Per-case swap-point awareness" note
    (the new `test_case_950ff_night_vent_h_ve_signature_issue_3802_limit28`
    unit test pins the input-side signature that drives the cohort).

- **Unblockers:**

  - **GaugeSolver production-path switchover** (Issues **#1465 /
    #1462**, both closed individually; production-path staged via
    **#3291 / PR #3482** for Phase A8 default flip — **gated on
    §LIMIT-21 β-soak closure**). The β-soak streak (Issue #3286,
    `#3286 β-soak` convention in CI comment threads) is currently at
    0/30 nights green; once the streak trips, the `gauge-solver`
    cargo feature flips to unconditional default. The post-flip
    §LIMIT-28 closure requires the FF cohort to land within ASHRAE
    140 ranges on the gauge path *without parameter tuning*.

  - **ADR-0011 implementation-options review** (Issue #3058
    follow-up): any of the three options — split `h_ve_night` into
    air-node mass (HVAC) and surface-node mass (FF) paths; reduce
    `h_ve_night` by F_sky on the mass coupling; route `h_ve_night`
    only through the air node — must preserve Case 950 (HVAC mode)
    annual cooling in the 390–920 kWh band and energy balance /
    cross-case ASHRAE 140 / architecture-drift / cycle guards remain
    green.

- **Affected Cases / Metrics (post-#1323 baseline):**

  - Cases 600FF / 650FF / 900FF / 950FF — all 8 metrics (min × 4,
    max × 4) are FAIL-rows on the 84-metric scorecard
    (`docs/ASHRAE140_RESULTS.md` §"Detailed Results / Free-Floating
    Cases"). The table above is the canonical reference; the
    per-month attribution diagnostic
    `tests/all_tests/ashrae_140_free_floating.rs` is wired into CI
    (live, no `#[ignore]` quarantine).

- **Severity:** **High** — the FF cohort is one of two fundamental
  ASHRAE 140 validation axes (the other being the 600/900-series
  peak-load and annual-energy gates tracked by §LIMIT-05 UPDATE and
  §LIMIT-16). The cohort gap blocks the FF-specific row of the strict
  ±15% annual-energy gate per `release_gates.yaml ->
  validation.individual.known_failures` (Cases 600 / 900 are excluded
  on the annual-energy axis because the FF cohort is the structural
  primary; the FF cases themselves are *not* in `known_failures` and
  must close via the §LIMIT-28 cohort logic, not by exclusion).

- **GitHub Issue:** [#3802](https://github.com/anchapin/fluxion/issues/3802)
  (Phase B3b PHYSICS-03 free-floating temperature correction,
  ADR-0011 night-vent context); depends on B3a (the FF cohort
  ranking).

- **Status:** 🟡 **Docs + cohort tracking shipped; structural fix
  routed to GaugeSolver #1465 / #1462 + ADR-0011 options (a) / (b) /
  (c).** No physics-code change; no `MAX_CONVECTIVE_TO_AIR_MULTIPLIER`
  / `h_ve_night` / `h_tr_em_wall` / `solar_distribution_to_air` /
  `derived_h_tr_3` change; no `tests/reference_data/zone_balance/
  strict_energy_gate_baseline.json` change; no ASHRAE 140 reference
  band change.

- **Sibling framing:** §LIMIT-17 / §LIMIT-24 (Case 950 FF+HVAC sibling
  entries); §LIMIT-16 (Cases 610/630/650 cooling-mode sibling);
  §LIMIT-05 UPDATE (#2453) (900-series annual-energy sibling);
  §FREE-01 / §FREE-02 / §FREE-03 (pre-#1323 FF cohort tracking); the
  §"Aggressive-baseline cohort tracking (Issue #3072)" table does NOT
  include the FF cohort (per Issue #3802 acceptance: "FF cases are
  tracked separately, not in the §LIMIT-05 / #3072 cohort list" —
  the FF cohort is the structural primary that the
  `known_failures` exclusion is *predicated on*, so adding the FF
  cases to the §LIMIT-05 / #3072 cohort would conflate the meta-issue
  with its own justification).
