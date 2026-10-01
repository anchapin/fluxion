# LIMIT-18 — investigation history

Narrative history for **LIMIT-18**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-18` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### LIMIT-18: Case 960 Blind heating_max 2.45 MWh > 1.0 MWh (AC4) — pre-existing test failure (Issue #3104)

- **Description:** The companion integration test
  `tests/ashrae_140_blind_validation.rs::test_blind_mode_case_960_infrastructure`
  fails on unmodified `develop` HEAD against the AC4 reference band
  upper bound: `Case 960 Blind heating_max 2.45 MWh > 1.0 MWh (AC4)`
  (assertion at `tests/ashrae_140_blind_validation.rs:1194`). The
  sibling `cooling_min >= 8.0 MWh (AC4)` assertion is also unreachable
  in the current solver topology. The failure was first observed during
  the wave-orchestration run that produced #3071 (LIMIT-09); the
  #3071 sub-agent explicitly noted: *"A separate
  `test_blind_mode_case_960_infrastructure` failure (Case 960 Blind
  `heating_max 2.45 > 1.0 MWh`) exists on unmodified `develop` HEAD
  and is unrelated to #3071. Test counts unchanged at 17 passed /
  1 failed / 6 ignored before and after this change."*

  The Case 960 Blind `heating_max = 2.45 MWh` over the 1.0 MWh AC4
  upper bound is the Case 960 Blind-mode manifestation of the same
  structural 5R1C + 9R4C single-lumped-mass-node limitation already
  tracked by §LIMIT-12 / #3062 (Case 940 CTF setback overshoot),
  §LIMIT-13 / #3063 (`h_tr_em` time-invariance in 5R1C path),
  §LIMIT-14 / #3061 (Case 960 sunspace annual cooling + peak heating),
  §LIMIT-16 / #3059 (Cases 610 / 630 / 650 peak cooling OVER), and
  §LIMIT-17 / #3058 (Case 950FF night-vent mass coupling). Every member
  of that cohort routes its structural fix to the GaugeSolver
  production-path work (#1465 / #1462), which treats solar and envelope
  heat transfer as geometric curvature rather than per-timestep energy
  injection. The cohort-level tracking stub is
  `docs/adr/0007-gauge-solver-structural-work.md`, and the
  aggressive-baseline cohort (Cases 195 / 600 / 620 / 940 / 960) is
  owned by Issue #3072.

- **Affected Tests:**
  `tests/ashrae_140_blind_validation.rs::test_blind_mode_case_960_infrastructure`
  (the integration test; now `#[ignore]`-quarantined with the reason
  `"Case 960 Blind heating_max 2.45 MWh > 1.0 MWh (AC4) — LIMIT-18
  (structural 5R1C single-lumped-mass-node limitation, unblocked by
  GaugeSolver rework #1465/#1462)"`). The assertion body (both
  `heating_max <= 1.0` and `cooling_min >= 8.0`) is retained below
  the `#[ignore]` marker for documentation; per AGENTS.md / RULES.md /
  ADR-0001, no parameter tuning is permitted on `heating_max` or
  `cooling_min` to absorb the OVER or BELOW.

- **Affected Metrics:** Case 960 Blind `heating_max` (MWh) — directly
  gated against the ASHRAE 140-2023 Annex B Table 8-15 AC4 reference
  band upper bound. The sibling `cooling_min >= 8.0 MWh` AC4 lower
  bound is also currently unreachable in the 5R1C + 9R4C
  single-lumped-mass-node topology. Both metrics are stable across
  unmodified `develop` HEAD runs.

- **Severity:** Low for the strict-energy-gate (#1333) (Case 960 is
  not in `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`
  per `release_gates.yaml` known structural failures). Medium for the
  integration suite `cargo test --test all_tests ashrae_140_blind_validation::`
  — this test is the singular `1 failed` row in the 17 passed /
  1 failed / 6 ignored count reported by the orchestrator. High for
  the AC4 reference-band acceptance check per Issue #1332 AC1 + AC4
  clauses (Cases 600 / 900 / 950 + Case 960 all must hit the Annex B
  Table 8-15 envelope).

- **GitHub Issue:** [#3104](https://github.com/anchapin/fluxion/issues/3104)
  (this entry); sibling issues are **#3071 / LIMIT-09** (Case 950
  5R1C night-vent override — same wave cohort),
  **#3059 / LIMIT-16** (Cases 610 / 630 / 650 peak cooling OVER — 5R1C +
  9R4C air-mass distribution), **#3058 / LIMIT-17** (Case 950FF
  night-vent mass coupling), **#3061 / LIMIT-14** (Case 960 sunspace
  annual cooling + peak heating), **#3062 / LIMIT-12** (Case 940
  CTF setback overshoot), **#3063 / LIMIT-13** (`h_tr_em`
  time-invariance in 5R1C path). Long-term fix routed to GaugeSolver
  rework **#1465 / #1462**. Cohort-level tracking owned by
  Issue #3072 (aggressive-baseline cohort — Cases 195 / 600 / 620 /
  940 / 960). Per AGENTS.md / RULES.md "fix the underlying math";
  per-case parameter tuning to close this gap is explicitly out of
  scope.

- **Status:** 🔄 **Known pre-existing failure, quarantined pending
  GaugeSolver.** Re-enable once #1465 (or equivalent structural fix)
  lands and Case 960 Blind `heating_max` moves to ≤ 1.0 MWh on the
  standard `cargo test --test all_tests ashrae_140_blind_validation:: -- --ignored`
  run. The re-enable acceptance is dual: (a) `heating_max <= 1.0 MWh`,
  (b) `cooling_min >= 8.0 MWh` — both clauses of
  `test_blind_mode_case_960_infrastructure` must hold without any
  solver constant, band, or assertion change.
