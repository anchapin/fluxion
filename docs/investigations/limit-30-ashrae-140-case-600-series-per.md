# LIMIT-30 — investigation history

Narrative history for **LIMIT-30**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-30` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 2. No wording was changed, softened or deleted.

---

#### LIMIT-30: ASHRAE 140 Case 600 series per-tilt / per-azimuth solar distribution audit — Phase B1a PHYSICS-01 (Issue #3797)

- **Description:** Phase B1a (Issue #3797) — diagnostic / audit
  of the ASHRAE 140 Case 600 series (600/610/620/630/640/650,
  the §LIMIT-05 / #3072 aggressive-baseline low-mass cohort)
  per-tilt / per-azimuth solar incident-energy distribution
  deviation against the ASHRAE 140 reference programs (EnergyPlus
  / ESP-r / TRNSYS) on the 2026-09-17 develop HEAD with
  `ThermalSelector::default()` resolving to `ZoneSolverKind::Gauge`
  but **falling through to legacy 5R1C / 9R4C** in the **default
  build** (cargo feature `gauge-solver` is OFF per the AGENTS.md
  Phase A8 note — `Cargo.toml:208-225`). Per the Issue #3797
  acceptance criteria ("diagnostic, no code changes") and
  **AGENTS.md / RULES.md / ADR-0001** ("no parameter tuning",
  "fix the underlying math"), the B1a deliverable is
  **measurement + ranking of failing distribution metrics +
  initial mechanism hypotheses only** — the structural closure is
  routed to B1b / Issue #3798 / PR #3847 at the release-gates
  layer (already merged; the B1b PR registers the cohort at
  `release_gates.yaml → validation.individual.known_failures` per
  the `release_gates.yaml` comment block at lines 83–110, which
  already names B1a as the precursor) and the GaugeSolver
  production-path work coordinated by **#1465 / #1462**
  (production-path switchover staged via **#3291 / PR #3482** for
  Phase A8 default flip, gated on **§LIMIT-21** β-soak closure).

- **Per-tilt / per-azimuth incident-energy calculation PASSES
  (the failing axis is NOT in the per-tilt / per-azimuth
  arithmetic):** Fluxion's `calculate_surface_irradiance`
  implementation agrees with the analytical cos(θᵢ) reference
  (ASHRAE Fundamentals Ch. 14 / Duffie–Beckman Eq. 1.6.3, the
  exact formulation EnergyPlus itself uses for beam-on-tilt)
  within 1 % on the full per-tilt sweep at az = 180° (south) on
  the Denver TMY3 weather year
  (`tests/all_tests/solar_isolation.rs::test_per_tilt_sweep`):
  5 tilts {0°, 30°, 60°, 90°, 180°} all in the [0.99, 1.01]
  annual ratio band, with annual beam {1041.332, 1281.590,
  1184.360, 786.890, 0.000} kWh/m²/year, max abs deviation
  1.4211×10⁻¹³ W/m² on the tilt=0° row (fluxion vs analytical);
  the horizontal-beam test
  (`test_horizontal_incident_solar`) reports 3637 hours with sun
  + DNI > 0, 3573 hours compared (> 1 W/m²), 0 hours exceed 1 %
  per-hour tolerance, annual fluxion 1041.332 kWh/m²/year vs
  analytical 1041.332 kWh/m²/year (annual ratio 1.000000, max abs
  deviation 0.0000 W/m²). The Mock-vs-Physics per-(tilt, az,
  hour) parity test
  (`tests/all_tests/surface_flux_parity.rs::test_parity_combined_tilt_azimuth_matrix`)
  passes within the 1 % ARCHITECTURE.md Module 2 acceptance
  criterion on the full 16 × 8760 = 140,160 (tilt × az × hour)
  assertion grid (the `test_parity_roof_zero_followup_1323`
  `#[ignore]`-quarantined test stays in place until #1323
  closes — this B1a audit does **not** un-ignore it).

- **Per-surface solar distribution parameters that FAIL (ranked by
  deviation — the issue acceptance criterion "the specific failing
  distribution metrics named and ranked by deviation"):**

  | Parameter | Fluxion value | ASHRAE 140 expectation | Deviation | Verdict |
  |---|---|---|---|---|
  | **Case 600 `solar_distribution_to_air`** (LowMass) | **0.30** | 0.0 (100 % to opaque surfaces) | **+0.30** | ❌ FAIL (worst) |
  | **Case 600 `solar_beam_to_mass_fraction`** (LowMass) | **0.30** | 1.0 (100 % to mass) | **−0.70** | ❌ FAIL |
  | **Case 600 fractions sum** (to_air + to_mass) | **0.60** | 1.0 (must sum to one) | **−0.40** | ❌ FAIL |
  | Case 900 `solar_beam_to_mass_fraction` (HighMass) | 0.30 | 1.0 (100 % to mass) | −0.70 | ❌ FAIL |
  | Case 900 `solar_distribution_to_air` (HighMass) | 0.00 | 0.0 (100 % to opaque surfaces) | 0.00 | ✅ PASS |

  Source: `tests/all_tests/solar_distribution_validation.rs`
  (printed via `cargo test --test all_tests solar_distribution_validation::
  -- --nocapture` on the 2026-09-17 develop HEAD). The Case 600
  `solar_distribution_to_air = 0.30` deviation (+0.30 absolute,
  100 % relative deviation from the ASHRAE 140 reference
  expectation of 0.0) is the **worst offender** — this is the
  structural reason the Case 600 series over-deposits solar into
  the air node and under-predicts peak cooling (the per-air-node
  distribution deviation then cascades through the 5R1C
  lumped-mass coupling and the multi-step τ-shortening axis
  measured in B2a / §LIMIT-29).

- **Per-surface Case 600 5R1C conductances vs hand-calc
  (per-tilt / per-azimuth transmission deviation,
  `tests/all_tests/test_case_600_htotal_verification.rs::test_case_600_htotal_hand_verification`,
  2026-09-17 develop HEAD on the canonical Case 600 envelope —
  Floor 48 m², Wall 75.6 m², Window 12 m², Opaque 63.6 m², Roof
  48 m²):**

  | Parameter | Hand-calc | Model | Δ (%) | Verdict |
  |---|---|---|---|---|
  | **h_tr_is** | 1251.32 | **165.60** | **−86.8 %** | worst |
  | **h_tr_ms** | 1092.00 | **240.00** | **−78.0 %** | 2nd worst |
  | h_tr_em | 50.17 | 59.26 | +18.1 % | out of band |
  | h_tr_w | 25.20 | 27.20 | +7.9 % | in band |
  | h_ve | 21.71 | 21.71 | 0.0 % | exact |
  | h_tr_floor | 8.82 | 8.82 | 0.0 % | exact |
  | **Cm** (J/K) | 3,028,278 | **2,162,443** | **−28.6 %** | out of band |
  | h_opaque (5R1C series) | 46.19 | 36.93 | −20.1 % | out of band |
  | H_total (5R1C series + window + ve) | 93.10 | 85.83 | −7.8 % | in band (small) |
  | H_total (simple Σ U·A) | 103.63 | (n/a) | — | reference |
  | 5R1C / simple UA ratio | — | **0.828** | −17.2 % under | structural |

  The h_tr_is / h_tr_ms per-surface conductances are the
  dominant deviations (−86.8 % and −78.0 %); the model produces
  an h_opaque (5R1C series path) of 36.93 W/K vs the
  hand-calculated 46.19 W/K (−20.1 %), which feeds the 0.828
  ratio of the 5R1C series H_total (85.83 W/K) to the simple Σ
  U·A H_total (103.63 W/K) — the structural signature that the
  per-tilt / per-azimuth transmission is systematically
  under-counted at the Case 600 envelope.

- **Case 600 annual / peak metrics vs ASHRAE 140 reference (the
  cascading failures):**

  | Case 600 metric | Engine value | Reference band | Deviation | Verdict |
  |---|---|---|---|---|
  | Annual heating | 4604.57 kWh | [4360.00, 5790.00] kWh | in-band (+5.6 %) | PASS |
  | Annual cooling | 3299.30 kWh | [3920.00, 6140.00] kWh | **−16 % UNDER** | ❌ FAIL |
  | Peak heating | 4.38 kW | [2.80, 3.80] kW | **+15.3 % OVER** | ❌ FAIL |
  | Peak cooling | 3.72 kW | [4.80, 6.20] kW | **−22.5 % UNDER** | ❌ FAIL |
  | `case_600_cooling` strict ±15 % gate | 2.546 MWh | [4.275, 5.784] MWh | **−34.38 % UNDER** | KNOWN-FAIL |

  3/4 Case 600 metrics fail the ±15 % strict-energy gate per
  `python3 scripts/check_strict_energy_gate_regression.py`
  (Issues #2506 / #3572); only Annual Heating is in-band. The
  integration test
  (`tests/all_tests/ashrae_140_case_600.rs::test_case_600_baseline_ashrae_140_reference`)
  prints Annual H 4.64 (ref 4.30–5.71, in-band) / Annual C 2.86
  (ref 6.14–8.45, −57 % UNDER) / Peak H 2.05 (ref 5.20–6.60, −62 %
  UNDER) / Peak C 3.60 (ref 6.80–8.50, −50 % UNDER) — the per-axis
  deviations converge with the production-validator detailed-results
  row above. The comprehensive validator
  (`tests/all_tests/ashrae_140_validation.rs`) emits the matching
  "ATTENTION: Potential regression" lines for Case 600 Annual
  Cooling (3.299 MWh vs ref 7–10), Peak Heating (4.38 kW vs ref
  2.6–4), and Peak Cooling (3.72 kW vs ref 4.6–6) — same 3/4-failures.

- **Three mechanism hypotheses (B1a does not pick a winner — per
  AGENTS.md / RULES.md / ADR-0001):**

  1. **`solar_distribution_to_air` / `solar_beam_to_mass_fraction`
     parameter routing hypothesis (most likely).** The 0.30 / 0.30
     per-surface routing split is the direct cause of the Case
     600 cooling −34.4 % UNDER strict-energy-gate signature:
     0.30 of the per-timestep beam gain that should go to the
     mass node (per ASHRAE 140 expectation
     `solar_beam_to_mass_fraction = 1.0`) is being routed to the
     air node instead, where the HVAC controller reads the
     cooling setpoint signal and trips the cooling plant on an
     over-estimated load. The structural fix is **path
     re-routing** (split the parameter between the LowMass and
     HighMass constructions or re-derive the per-tilt /
     per-azimuth fraction from the energy-balance identity
     `f_beam_to_mass + f_beam_to_air = 1.0`), **not** a numerical
     tuning of the 0.30 / 0.30 values themselves — per
     AGENTS.md / RULES.md / ADR-0001, raising
     `solar_distribution_to_air` to absorb the structural
     cooling gap is forbidden. **Mechanism: per-surface
     distribution routing mismatch; fix is path re-routing, not
     tuning.**

  2. **Per-surface 5R1C h_tr_is / h_tr_ms conductance hypothesis.**
     The hand-calc / model per-surface conductance deltas (−86.8 %
     on h_tr_is and −78.0 % on h_tr_ms) are the structural reason
     the H_total 5R1C path is 0.828× the simple Σ U·A reference
     (−17.2 %). The 5R1C series path is systematically
     under-counted on the Case 600 envelope. The structural fix
     is **5R1C parameter re-derivation** (e.g. via ISO 13790
     §12.2.3 + Annex C, the same convention used by the
     §LIMIT-29 B2a Cm derivation), **not** numerical tuning.
     **Mechanism: per-surface 5R1C parameter drift; fix is
     parameter re-derivation, not tuning.**

  3. **Case 600 series low-mass + 5R1C lumped-mass-node
     hypothesis (the §LIMIT-16 / §LIMIT-05 cousin).** The Case
     600 series shares the same 5R1C + 9R4C single-lumped-mass-
     node pathology as Cases 610 / 630 / 650 (§LIMIT-16 / Issue
     #3059) and the 900-series (B2a / §LIMIT-29); the
     solar-distribution deviation is the LowMass end of the same
     family. The `MAX_CONVECTIVE_TO_AIR_MULTIPLIER = 2.0×` cap
     (PR #3041) closed Cases 620 / 640 but did NOT transfer to
     Case 600 — because Case 600's per-tilt / per-azimuth
     solar-distribution deviation is upstream of the
     `MAX_CONVECTIVE_TO_AIR_MULTIPLIER` axis (the cap is on the
     convective path, not on the solar-distribution routing).
     **Mechanism: same family as §LIMIT-05 / §LIMIT-16 /
     §LIMIT-29; fix is routed to the GaugeSolver production-path
     work, not to per-Case 600 tuning.**

  The three mechanisms are **not mutually exclusive** — the
  B1b structural fix likely needs to address the per-surface
  distribution routing (hypothesis 1) and the per-surface 5R1C
  h_tr_is / h_tr_ms conductance re-derivation (hypothesis 2) and
  the family-level 5R1C lumped-mass-node damping (hypothesis 3)
  to close the Case 600 cooling cascade AND preserve the ±15 %
  strict-energy gate per `release_gates.yaml →
  validation.individual.known_failures` Case 600 series
  annotation (B1b / Issue #3798 / PR #3847).

- **Cross-references:**

  - **B1b / Issue #3798 / PR #3847** — the release-gates
    registration
    (`release_gates.yaml → validation.individual.known_failures`
    Case 600 series annotation, "B1b solar-distribution cohort,
    Issue #3798") that this B1a audit is the **precursor
    measurement for**. The `release_gates.yaml` comment block at
    lines 83–110 already names B1a as the precursor: *"Once
    Issue #3797 (B1a audit) closes with the per-tilt /
    per-azimuth deviation table, a follow-up §LIMIT entry +
    `docs/ASHRAE140_RESULTS.md` structural-failure-mechanism note
    will be filed with the named mechanisms."* The per-surface
    distribution / conductance / metric tables above are the
    input the B1b cohort registration cites.

  - **§LIMIT-29 / Issue #3799** — the B2a thermal-mass τ cousin
    (same `TimeConstantAnalyzer` / ISO 13790 §12.2.3 + Annex C
    family). The B2a Cm = +33.9 % above-envelope and τ = −40 %
    below-envelope deviations are consistent with the B1a
    family-level 5R1C lumped-mass-node damping (hypothesis 3
    above).

  - **§LIMIT-13 / Issue #3063 / ADR-0009** — `h_tr_em`
    time-invariance regression fence; the 18.3 W/m²K canonical
    exterior film coefficient is **unchanged** through the B1a
    measurement window — guarded by
    `tests/regression_exterior_film_unification.rs`; the AGENTS.md
    "no regression on the canonical film coefficient" guard holds
    throughout. The B1a measurements above do not regress this
    fence (the h_tr_em hand-calc 50.17 vs model 59.26 = +18.1 %
    is on the 5R1C series path, not on the 18.3 W/m²K canonical
    exterior film coefficient).

  - **§LIMIT-05 / §LIMIT-05 UPDATE (#2453)** — 900-series
    bidirectional annual-energy over-prediction (the air-mass
    distribution pathology is the same as the B1a
    solar-distribution deviation; the per-tilt / per-azimuth
    distribution deviation on Case 600 series is the LowMass end
    of the same family).

  - **§LIMIT-16 / Issue #3059** — Cases 610 / 630 / 650 peak
    cooling OVER (same Case 600-series family; the
    `MAX_CONVECTIVE_TO_AIR_MULTIPLIER = 2.0×` cap from PR #3041
    closed Cases 620 / 640 but did not transfer to Case 600 —
    the B1a per-tilt / per-azimuth solar-distribution deviation
    is upstream of the `MAX_CONVECTIVE_TO_AIR_MULTIPLIER` axis).

  - **§LIMIT-17 / Issue #3058 / ADR-0011** — Case 950FF
    night-vent mass coupling gap (structurally similar to the
    Case 600 series 5R1C + 9R4C single-lumped-mass-node
    pathology; the night-vent ACH = 13.14 → h_ve ≈ 570.8 W/K
    coupling is the FF-mode cousin of the Case 600 HVAC-mode
    solar-distribution routing mismatch).

  - **§LIMIT-22** — the gauge-build-only
    `test_case_950_mass_temperature_precooled_issue_1422`
    quarantine (Issue #3297); not directly applicable to B1a but
    the gauge τ_mass ≈ 61 h measurement source for the B2a
    cousin.

  - **§LIMIT-21** — Gauge β-path pre-existing air-trajectory
    failure cohort + β-soak 30-night production-path gate
    (Issue #3286, `#3286 β-soak` convention in CI comment
    threads; currently 0/30 nights green). The Case 600 series
    residuals above persist on the production path until the
    `gauge-solver` cargo feature is enabled.

  - **#3072** — the aggressive-baseline cohort tracking (Cases
    195 / 600 / 620 / 940 / 960) that owns the
    `release_gates.yaml → validation.individual.known_failures`
    membership. Case 600 is in this cohort; the B1a measurements
    are the per-axis attribution for the Case 600 entry.

  - **SOLAR-01 / Issue #274** — the pre-existing SOLAR-01 entry
    that documents the 600-series peak-cooling signature; the
    B1a measurements identify the per-tilt / per-azimuth
    distribution axis as the structural mechanism behind the
    SOLAR-01 partial-resolution status (low-mass peak cooling now
    in-band per #1362 / #1328 verification, but the per-axis
    distribution / conductance deviations persist). The
    structural fix routed to GaugeSolver #1465 / #1462 in
    SOLAR-01 is the same unblocker the B1a mechanism hypotheses
    above route to.

  - **Issue #1323 / #1325 / #1330 / #1337** — the per-tilt /
    per-azimuth fixture-data lineage. Issue #1323 (corrected
    constants in roof-solar) is the pre-existing dependency for
    the `test_parity_roof_zero_followup_1323` `#[ignore]` test;
    #1325 / #1330 / #1337 are the analytical / ASHRAE-140-reference
    fixture-data lineage that grounds the B1a per-tilt /
    per-azimuth calculation PASSES above. **All four are
    upstream closed issues** that the B1a audit cites without
    modification.

- **Unblockers:**

  - **GaugeSolver production-path switchover** (Issues
    **#1465 / #1462**, both closed individually; production-path
    staged via **#3291 / PR #3482** for Phase A8 default flip,
    gated on §LIMIT-21 β-soak closure).

  - **PR #3041 / Issue #3059** — the
    `MAX_CONVECTIVE_TO_AIR_MULTIPLIER = 2.0×` cap sibling
    partial-fix on the Cases 610 / 630 / 650 cooling OVER axis;
    no equivalent solar-distribution cap exists for the Case
    600 axis — per ADR-0001 / AGENTS.md / RULES.md, raising
    `solar_distribution_to_air` to absorb the structural cooling
    gap is forbidden.

- **Affected Cases / Metrics (post-#1323 baseline):**

  - Case 600 — Annual heating 4.60 MWh (Ref: 4.36–5.79 MWh,
    in-band +5.6 %); Annual cooling 3.30 MWh (Ref: 3.92–6.14
    MWh, −16 % UNDER); Peak heating 4.38 kW (Ref: 2.80–3.80
    kW, +15.3 % OVER); Peak cooling 3.72 kW (Ref: 4.80–6.20
    kW, −22.5 % UNDER). 3/4 ASHRAE 140 reference-band metrics
    are FAIL-rows on the 84-metric scorecard
    (`docs/ASHRAE140_RESULTS.md` §"Detailed Results / Baseline
    Cases (600 Series)"). The Case 600 series cohort is in
    `release_gates.yaml → validation.individual.known_failures`
    (annotated by B1b / Issue #3798 / PR #3847 — see the
    `release_gates.yaml` comment block at lines 83–110 for the
    B1a → B1b provenance).
  - The Case 600 series (600 / 610 / 620 / 630 / 640 / 650)
    shares the same per-tilt / per-azimuth distribution /
    conductance signature; the B1a audit's Case 600 measurements
    are the per-axis attribution for the whole family.

- **Severity:** **High** — the Case 600 series per-tilt /
  per-azimuth solar-distribution cohort is one of the two
  fundamental ASHRAE 140 validation axes (the other being the
  Case 900 thermal-mass cohort tracked by §LIMIT-05 /
  §LIMIT-29). The Case 600 cooling −34.4 % UNDER on the
  strict ±15 % annual-energy gate per
  `release_gates.yaml → validation.individual.known_failures`
  is the per-axis structural signature that the B1b cohort
  registration cites. The cohort gap blocks the strict ±15 %
  annual-energy gate per the AGENTS.md / RULES.md / ADR-0001
  prohibition on closing it by raising
  `solar_distribution_to_air` / `solar_beam_to_mass_fraction`.

- **GitHub Issue:** [#3797](https://github.com/anchapin/fluxion/issues/3797)
  (Phase B1a PHYSICS-01 solar distribution audit, precursor
  measurement for B1b / Issue #3798 / PR #3847 release-gates
  cohort registration of the Case 600 solar-distribution
  cohort).

- **Status:** 🟡 **Docs + characterization shipped; structural
  fix routed to B1b release-gates registration (Issue #3798 /
  PR #3847, already merged) + GaugeSolver #1465 / #1462
  production-path switchover.** No physics-code change; no
  `solar_distribution_to_air` / `solar_beam_to_mass_fraction` /
  `h_tr_w` / `h_tr_is` / `h_tr_ms` / `h_tr_em` / `h_ve` /
  `h_tr_floor` / `MAX_CONVECTIVE_TO_AIR_MULTIPLIER` /
  `h_ve_night` / `h_tr_em_wall` / `derived_h_tr_3` /
  `h_ms_coeff` change; no
  `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`
  change; no ASHRAE 140 reference-band change; no `#[ignore]`
  quarantine change (the `test_parity_roof_zero_followup_1323`
  `#[ignore]` from #1323 stays in place until #1323 closes;
  this B1a audit does **not** un-ignore it); no reference-data
  CSV / sha256 change; no
  `tests/all_tests/per_tilt_per_azimuth_fixture_data.rs` regen
  (the fixture is auto-generated from
  `tests/reference_data/solar/ashrae_140_surface_incident_solar.csv`
  — Issue #1330; B1a cites the existing fixture without
  modification); no `regression_exterior_film_unification.rs`
  change (the canonical 18.3 W/m²K film coefficient is
  preserved — LIMIT-13 regression fence stays green).

- **Sibling framing:** §LIMIT-13 (h_tr_em time-invariance
  regression fence; the 18.3 W/m²K canonical exterior film
  coefficient is unchanged, guarded by
  `tests/regression_exterior_film_unification.rs`); §LIMIT-05 /
  §LIMIT-05 UPDATE (#2453) (900-series annual-energy + peak
  cooling siblings — the bidirectional OVER / UNDER signature
  on the high-mass end of the 5R1C + 9R4C single-lumped-mass-node
  pathology); §LIMIT-16 / Issue #3059 (Cases 610 / 630 / 650
  peak cooling OVER — the same Case 600-series family, the
  LowMass end of the 5R1C + 9R4C single-lumped-mass-node
  pathology); §LIMIT-17 / Issue #3058 / ADR-0011 (Case 950FF
  night-vent mass coupling gap — structurally similar to the
  Case 600 series solar-distribution deviation); §LIMIT-22
  (gauge τ_mass measurement source for the B2a cousin);
  §LIMIT-21 (β-soak 30-night production-path gate);
  §LIMIT-29 / Issue #3799 (the B2a thermal-mass τ cousin —
  same `TimeConstantAnalyzer` / ISO 13790 §12.2.3 + Annex C
  family); SOLAR-01 / Issue #274 (the pre-existing SOLAR-01
  entry that documents the 600-series peak-cooling signature).
  **Cross-PHYSICS-01**: B1b / Issue #3798 / PR #3847 (the
  release-gates registration that cites this B1a audit as the
  precursor measurement — `release_gates.yaml` lines 83–110
  already name B1a as the precursor); **cross-PHYSICS-01
  (companion fixtures)**: Issue #1323 (corrected roof-solar
  constants; the `test_parity_roof_zero_followup_1323` `#[ignore]`
  stays in place until #1323 closes — this B1a audit does
  **not** un-ignore it), #1325 / #1330 / #1337 (per-tilt /
  per-azimuth fixture-data lineage grounding the calculation
  PASSES — all upstream closed issues cited without
  modification).

#### LIMIT-30 UPDATE (#3961): defaults-only retirement of the 0.30/0.30 solar split measured and refuted

*Added 2026-09-26 · Issue #3961 · zero production-physics change shipped*

Issue #3961 (reopened 2026-09-25) asked whether the zone-solver solar-coupling deficit
for Case 600FF (peak T_air 47.29 °C vs the 64.9–75.1 °C reference band) could be closed
by retiring the `phi_st_window` heuristic (`remaining_sol × st_sol_frac`) and the fixed
0.30/0.30 solar split (`solar_distribution_to_air` / `solar_beam_to_mass_fraction`,
the latter massiveness-blended as `0.30·(1−w)` for the air fraction) to the referee
convention (`to_air = 0.0`, `beam_to_mass = 1.0`;
`tests/all_tests/solar_distribution_validation.rs`).

**Premise re-verified.** Upstream solar delivery is exact: `calculate_zone_solar_gain`
(thermal_model_iterative.rs) computes the exact window-area × incident-irradiance
product (8,437 W peak for Case 600FF), stored as `solar_gains[zone] = Φ_sol/A_floor`
and round-tripped losslessly by the zone solvers. The defect, if any, is purely in the
split mechanism.

**Experiment.** Both retirement endpoints were implemented and measured against the
strict ±15 % annual-energy gate (`scripts/check_strict_energy_gate_regression.py`,
release build, `--include-ignored` capture):

| Configuration | Case 600 cooling gap | Case 800 cooling gap | Case 900 heating | Case 960 cooling | 600FF peak |
|---|---|---|---|---|---|
| baseline (develop) | 34.38 pp (KNOWN-FAIL) | 50.10 pp (KNOWN-FAIL) | 1.633 MWh (in band) | 0.144 MWh (KNOWN-FAIL) | 47.29 °C |
| `to_air=0, beam=1.0` | **44.08 pp REGRESSION** | (810_C 78.59 pp REGRESSION) | **3.289 MWh REGRESSION** | **0.008 MWh REGRESSION** | 47.99 °C |
| `to_air=0, beam=0.30` | **55.08 pp REGRESSION** | **69.04 pp REGRESSION** | 1.633 MWh | 0.144 MWh | **46.01 °C** |

Full-convention regressions: 9 strict-gate violations (600_C, 810_C, 900_H, 900_C,
920_C, 950_C, 960_H, 960_C, 970_C).

**Diagnosis.**

1. `to_air = 0` alone moves the 600FF peak _away_ from the reference band: the 0.30
   low-mass direct-to-air fraction is load-bearing compensation for the missing
   air-node capacitance (#1152-class work), not an arbitrary constant.
2. `beam_to_mass = 1.0` collapses high-mass cooling: the convention routes 100 % of
   window solar through the capacitance-weighted mass pool of
   `step_interior_surface_network`, which concentrates absorption on the ground-coupled
   slab where it is stored and lost outward instead of re-released to the zone
   (Case 960 cooling 0.144 → 0.008 MWh). The referee comment "100 % beam solar to
   opaque surfaces/mass" is a statement about the _air split_ (none to air), not a
   mandate for the internal mass-vs-surface pool weighting; the film-conductance
   (`w_surface`) and capacitance (`w_mass`) weights are the pool semantics in the
   current implementation, and neither is beam-incident geometry.
3. The structural closure therefore requires beam-incident absorption geometry (which
   surfaces the transmitted beam actually strikes) and/or the 5R1C air-node capacitance
   rewrite (#1152), routed per ADR-0017 through the DAE-teacher path (#3979–#3986) —
   not constants.

**Shipped in this update (zero production-physics change):**

- In-code experiment record at the defaults site
  (`thermal_model_core::from_spec_with_selector`) warning against zeroing the
  constants without the structural work.
- Spec pins in `tests/all_tests/solar_split_convention.rs`:
  - `solar_split_defaults_are_load_bearing_issue_3961` — pins the massiveness-blend
    endpoints (Case 600 → 0.30/0.30, Case 900 → 0.0/0.30) so the load-bearing
    calibration cannot drift silently;
  - `gauge_bc_default_semantics_are_pinned` — documents `ZoneBoundaryConditions`
    default semantics (0.0 to air, surface-pool-dominant remainder);
  - `gauge_{coupled,primary}_path_absorbs_full_window_solar_under_convention` — the
    mechanism invariants any structural rewrite must preserve: with convention
    fractions the full window product is absorbed and the telemetry closure
    Σ_i `solar_absorbed_w` + Φ_sol·to_air = Φ_sol holds exactly;
  - `gauge_split_mechanism_fractions_still_partition` — legacy 30/70 partition pin.
- The failing `solar_distribution_validation` referee (3 tests) remains the tracking
  gate for the structural closure.
