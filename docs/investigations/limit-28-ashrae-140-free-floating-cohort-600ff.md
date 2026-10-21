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

## Round 79 (2026-10-21) — sub-hour FD assessment: the gauge arm measured at the post-coupled-N operating point

**Scope question (round-76/§63 handoff).** §63 established no sub-step count N reaches the free-float Tmax bands under the coupled 5R1C scheme (650FF/600FF Tmax "unreachable under any converged N — missing network dynamics, not time resolution"), and named the GaugeSolver route (hourly per-surface FD network) as where sub-hour spatial resolution exists. This section records the assessment's three questions and the first measurement increment. **No engine code changed; no default-build output moved.**

### Measurement increment (smallest honest one — no code, no env gate beyond the existing cargo feature)

The gauge build (`cargo test --features gauge-solver --test all_tests free_float_strict_gate -- --include-ignored --nocapture`) was run at develop `dc1ad186` (engine identical at the round-79 merge tip `ca5b9ecf`; the intervening commit touched only two PyO3 test files outside this harness). Canonical DRYCOLD weather. Raw results, old (default build, §64 `_doc_round78` operating point) → new (gauge arm, same tip):

| metric | default build (legacy routing) | gauge arm (per-surface FD) | band |
|---|---|---|---|
| 600FF Tmin | −3.78 | **−11.99 (IN BAND)** | [−13.8, −9.9] |
| 600FF Tmax | 55.26-class floor | 45.19 (17.2 below floor) | [62.4, 68.4] |
| 650FF Tmin | −17.40 (in band) | −17.88 (0.08 K below floor) | [−17.8, −16.7] |
| 650FF Tmax | 41.74 (34.7 pp) | 42.82 (18.3 below floor) | [61.1, 66.8] |
| 900FF / 950FF Tmin+Tmax | byte-identical | byte-identical | (HighMass 9R4C promotion shared; inert) |

The strict-band tests panic on out-of-band metrics by design (`--include-ignored` transparent-measurement pattern); the printed lines above are the record. Nothing tuned.

### Assessment answer 1 — what per-surface FD conduction on the FF cases requires

It already exists as a measurement route: the gauge arm's per-surface FD stacks (12+ nodes/layer, converged per §25; §5.2.6 wind-dependent exterior film post-#4371; §B3.3 ground boundary post-round-10) route the low-mass FF family with zero new code. What a *production* per-surface FD route would additionally require is exactly what §LIMIT-40 records: resolving the FD stacks' day/night storage-release split (the ~4 K leading night-warming term, §52) and the star-node film coupling (~1.2–1.3 K, §26) — both documented structural model differences of the gauge network, not wiring defects.

### Assessment answer 2 — the gauge arm's 600-series topology residual: separate class, and it does not close Tmax

The residual is a **separate class**, not a blocker: it is the gauge star's own FD storage-release + film-coupling structure (§23–§27, §LIMIT-40), distinct from the legacy path's §53 floors. The fresh measurement sharpens the §63 verdict: at the post-coupled-N operating point the gauge arm *improves* the FF Tmin row (600FF enters band; 650FF within 0.08 K of the floor) but *lowers* the Tmax floor further (45.19/42.82 vs the legacy 55.26-class/41.74 against the same 62.4/61.1 band floors). Both networks undershoot the reference diurnal peak by 17–23 K on the low-mass FF cases. The missing term is therefore not time resolution (§63), not the lumped-vs-per-surface mass split alone (this measurement), and not the FD stacks' exterior coupling (post-#4371 measured) — the remaining candidates are the §LIMIT-40 storage-release split itself and the interior surface/air coupling under free-float swing, consistent with the round-70 finding that the interior network carries the dynamic order.

### Smallest next increment (documented, not yet built)

An env-gated scratch arm (rounds-66–77 pattern) that disables the gauge FD stack's day-side solar storage split — the §52 `disable_fd_for_test`-class discriminator — on the low-mass FF cases only, to bound how much of the 17–23 K Tmax shortfall the storage-release split owns at the new operating point. If that bound comes back small, the honest conclusion is that the reference's FF Tmax carries an exterior-gain magnitude term neither network models (solar-absorption-to-surface partitioning), and the cohort floor moves to that class. Default build stays byte-identical either way; any future production routing decision remains Alex's.

## Round 80 (2026-10-21) — the storage-release-split discriminator at the post-coupled-N operating point

The round-79 scoped experiment, executed. Env-gated scratch harness
(never-commit `FLX_R80_NOFD` / `FLX_R80_NOSTORE` in
`src/physics/gauge_zone_solver.rs`, scratch branch, engine default untouched),
gauge build (`--features gauge-solver`), canonical DRYCOLD, engine
`2e41ae7a`. Harness cross-validation: flag-off reproduces the round-79 gauge
arm bit-for-bit (600FF −11.99/45.19, 650FF −17.88/42.82; 900FF/950FF
byte-identical). Arms:

- `FLX_R80_NOFD` — drop the per-surface FD stacks entirely
  (`disable_fd_for_test` semantics; the §52 discriminator, re-run at the new
  operating point).
- `FLX_R80_NOSTORE` — bypass only the day-side storage-release split: the
  mass-pool share is injected at the interior face instead of as a
  capacity-weighted volume source, so the FD stack still conducts
  through-thickness but does not store/release the day's beam share.

| Tmin/Tmax (°C) | 600FF | 650FF | 900FF | 950FF |
|---|---|---|---|---|
| gauge flag-off (baseline) | −11.99 / 45.19 | −17.88 / 42.82 | −0.26 / 43.28 | −15.67 / 38.78 |
| gauge FD off (NOFD) | −11.61 / 50.86 | −16.70 / 48.39 | byte-identical | byte-identical |
| gauge storage split bypassed (NOSTORE) | −11.95 / 46.73 | −17.87 / 43.55 | byte-identical | byte-identical |
| band | [−13.8,−9.9] / [62.4,68.4] | [−17.8,−16.7] / [61.1,66.8] | [0.6,2.2] / [43.3,46.0] | [−13.4,−12.5] / [36.1,37.1] |

**Attribution.** At the post-coupled-N operating point the storage-release
split owns **+1.54 K of 600FF Tmax and +0.73 K of 650FF Tmax** (NOSTORE vs
baseline) — a small bound against the 17.2 / 18.3 K shortfalls. Dropping the
FD stacks entirely recovers **+5.67 K / +5.57 K** (NOFD) — so roughly a
quarter of the shortfall is FD through-thickness damping in aggregate, of
which the day/night storage split is only ~1/4 to 1/3. The high-mass family
is byte-identical under both flags at this operating point (the 9R4C
HighMass promotion is shared and inert, as round-79 recorded).

Conditioned-family spillover (gauge build, annual tolerance battery):
NOSTORE moves the low-mass conditioned family a few percent (650 C 4.008 →
4.414 MWh, 610 C 4.715 → 5.279, 640 C 6.819 → 7.720, 620 C 8.067 → 8.850 —
mixed direction, no band entries change class) and leaves Case 195
bit-identical; NOFD reproduces the round-70 degenerate-`H_tr,3` signature
(195 H 18.919 → 100.623 MWh) and is ruled out as a production arm, as
before.

**Verdict.** The bounded experiment answers the round-79 scope question: the
missing 17–23 K low-mass FF diurnal peak is **not** owned by the
storage-release split (≤ 1.5 K of 17–23 K), nor by FD through-thickness
damping in aggregate (~5.6 K), nor by time resolution (§63), nor by the
mass-split alone (round 79). Consistent with §52/§53, no measured state
reaches band; the cohort floor moves to the **exterior-gain
partitioning class** (solar-absorption-to-surface partitioning / gain
magnitude the reference programs model and neither network does), with the
interior surface/air coupling under free-float swing (round-70 finding) as
the remaining in-network candidate. Nothing tuned; no engine code landed;
scratch reverted before commit; default build untouched; no re-record
battery owed.

## §Round-81 — exterior-gain partitioning discriminator (2026-10-21, engine `b6827776`)

The round-80 verdict named the **exterior-gain partitioning class**
(solar-absorption-to-surface partitioning) as the cohort floor. This section
executes that class's discriminating experiment on the **default build**
(coupled-N N=12, D5 per-surface, sub-hour air-driver ramp), where the strict
free-float cohort actually runs.

**Premise verified in code and primary source.**

- Engine: at spec init (`thermal_model_core/mod.rs`) the transmitted-solar
  split is `to_air = 0.30·(1−w_solar)` (lightweight → 0.30, Issue #2359's
  ASHRAE 140 Table B-12 F_m; heavyweight → 0.0, ADR-002/#1175) and
  `beam_to_mass = 0.30`; `step_physics_5r1c` applies it as
  `phi_ia += Φ_sol·to_air`, `phi_st += Φ_sol·(1−to_air)·(1−beam)`,
  `phi_m += Φ_sol·(1−to_air)·beam`. The gauge solver threads the same
  fractions through the identical three-way split and distributes the
  surface pool by area-share of `h_tr_is` (`step_interior_surface_network`).
- ASHRAE 140-2017 §5.2.1.12 + Table 5-138 (Case 600): the fractional
  distribution is **among interior surfaces only** — Floor 0.642, Ceiling
  0.1678, East 0.038, West 0.038, North 0.053, South wall 0.0276, 0.035
  retransmitted out the windows — and **nothing directly to zone air**.
  (Programs computing interior distribution internally use interior solar
  absorptance; §5.2.1.12.2 permits other values only with a mathematical,
  physical, or logical basis, applied consistently.)
- ISO 13790 §C.3 capacity-based split was already audited (#4369): formulas
  correct, mass-pool share misplaced at the interior face — fixed (R11 PR-b).

**Arms (env-gated scratch at the spec-init split site; one release build;
flag-off reproduced the strict gate bit-for-bit before any arm ran — 42/42
output-diff metrics +0.0000%, checker 10 PASS / 32 KNOWN-FAIL / 0
REGRESSION on `b6827776`).**

- Arm A `FLX_R81_TOAIR0`: `to_air = 0.0` for all masses (the §5.2.1.12
  air-term endpoint; mass share unchanged 0.30).
- Arm B `FLX_R81_SOL2ST`: `to_air = 0.0, beam_to_mass = 0.0` — 100 % of
  transmitted solar absorbed at the (lumped) surface node, the closest
  lumped representative of Table 5-138. The table's 3.5 % window
  retransmission is NOT modeled (disclosed: arms B slightly over-absorbs).

| strict (canonical DRYCOLD) | flag-off | Arm A to_air=0 | Arm B sol→st | band |
|---|---|---|---|---|
| 600FF Tmin / Tmax | −3.78 / 47.42 | −3.61 / **38.80** | −6.35 / **39.02** | [−13.8,−9.9] / [62.4,68.4] |
| 650FF Tmin / Tmax | −17.40 / 41.74 | −17.39 / **34.64** | −17.49 / **35.30** | [−17.8,−16.7] / [61.1,66.8] |
| 900FF Tmin / Tmax | −0.26 / 43.28 | byte-identical | byte-identical | [0.6,2.2] / [43.3,46.0] |
| 950FF Tmin / Tmax | −15.67 / 38.78 | byte-identical | byte-identical | [−13.4,−12.5] / [36.1,37.1] |

**Conditioned spillover (low-mass family; disclosed, nothing tuned).**
Arm A: 600 H 5.353 → 5.876 MWh, 600 C 2.966 → 1.478; 650 C 1.186 → 0.318;
610/620/630/640 move the same direction (H up, C down ~35–50 %). Arm B:
600 H 7.497 / C 1.196; 650 C 0.457. High-mass rows (195, 810, 900, 920–970)
byte-identical under both arms (`w_solar = 1` → `to_air` already 0 there;
the arms never touch the 9R4C family).

**Verdict.** The exterior-gain partitioning class is **REFUTED in direction**
as the owner of the missing low-mass free-float diurnal peak: both honest
§5.2.1.12 partitionings move 600FF/650FF Tmax **further below** band
(−8.6 / −7.1 K and −7.1 / −6.4 K respectively), while the incumbent
low-mass to_air = 0.30 term is load-bearing in the upward direction —
consistent with the #3961 record (#4079) and #2359's Table B-12 premise.
The floor is therefore NOT owned by solar partitioning among air/surface/
mass; with §Round-80 (storage split ≤1.5 K, FD aggregate ~5.6 K) and §63
(time resolution ruled out at any N), the remaining in-network candidate is
the **magnitude/timing of the absorbed exterior gain itself** (per-tilt
incidence on the real geometry — the LIMIT-30 B1b re-derivation) or an
exterior-driver term neither network carries. A floor-dominant per-surface
Table 5-138 weighting is not representable in the lumped 5R1C (single s-node)
and is deferred to the per-surface interior network where it belongs. Nothing
tuned; no engine code landed; scratch reverted before commit; default build
untouched; no re-record battery owed.

## §Round-82 (2026-10-21) — the exterior-driver class measured: both terms present, correctly signed, no missing driver

Round 81 named two remaining candidates for the low-mass free-float floor:
(a) absorbed-gain magnitude/timing (per-tilt incidence, the LIMIT-30 B1b
line) and (b) an exterior-driver term neither network carries.

**Premise audit.** (a) is already primary-source verified as implemented:
the LIMIT-30 B1a audit measured per-tilt/per-azimuth incident energy
PASS (all tilts in [0.99, 1.01] vs analytical; 0/3573 sun hours over 1 %
per-hour tolerance; 16x8760 tilt-azimuth parity grid), and the window
transmission path applies the incidence-dependent SHGC ratio lookup
(`ashrae_140_window_shgc_ratio`, ASHRAE Handbook Ch. 15 table), diffuse
x0.9 / ground-reflected x0.85 factors, the mid-hour sun convention
(adopted 2026-10-09), and the roof/wall absorptances 0.7/0.6 per Annex B1
(#1140/#4166, #1323 closed). No identified primary-source magnitude or
timing error remains on (a). (b) was therefore measured.

**Discriminator (env-gated scratch at `ExteriorBoundarySurface::
sol_air_temperature`, the single 5R1C exterior-driver construction site;
flag-off reproduced the strict gate bit-for-bit first — 10 PASS / 32
KNOWN-FAIL / 0 REGRESSION, output-diff 42/42 +0.0000 %, fabric harness
PASS on `cb029e56`).** Two honest term-removal arms, no parameter changes:

- Arm `FLX_R82_NOOPAQUE`: the `alpha*I/h_c` absorbed-opaque-solar term of
  sol-air dropped (t_sol_air = T_out − F_sky*eps*dR/h_c).
- Arm `FLX_R82_NOSKY`: the `−F_sky*eps*dR/h_c` sky-longwave term dropped.

| strict (canonical DRYCOLD) | flag-off | NOOPAQUE | NOSKY | band |
|---|---|---|---|---|
| 600FF Tmin / Tmax | −3.78 / 47.42 | **−6.22 / 42.93** | **−1.71 / 49.62** | [−13.8,−9.9] / [62.4,68.4] |
| 650FF Tmin / Tmax | −17.40 / 41.74 | −17.48 / 40.70 | −17.34 / 42.36 | [−17.8,−16.7] / [61.1,66.8] |
| 900FF / 950FF family | — | byte-identical | byte-identical | (9R4C sol-air path, not this site) |

Both-arms-removed (`NOOPAQUE+NOSKY`): 600FF −4.15 / 45.13 — strictly
intermediate, confirming the two terms act independently in opposite
directions.

**Conditioned spillover (disclosed, nothing tuned).** NOOPAQUE degrades
the whole low-mass family: 600 H 5.353 -> 7.299 MWh, 620 H 6.894 -> 9.319,
630 H 7.579 -> 10.188, 195 H 3.933 -> 8.377, with cooling collapsing
(600 C 2.966 -> 1.455); 14 strict-gate regressions. NOSKY: 600 H
5.353 -> 4.211, 600 C 2.966 -> 4.093, 195 C 1.560 -> 1.730; 3 regressions.

**Verdict.** The exterior-driver class is **REFUTED as a missing term**:
both sol-air drivers are present, correctly signed, and load-bearing in
the directions the standard's physics requires — the absorbed-opaque-solar
term carries ~+4.5 K of 600FF Tmax upward (removal pushes the peak further
below band), and the sky-longwave term carries ~−2.1 K of nighttime
cooling (removal pushes Tmin further above band). Neither removal is a fix
candidate; their magnitudes are the Annex B1 / ASHRAE-convention values,
so raising them would be tuning. With (a) premise-verified passing and
(b) now measured present-and-correctly-signed, the round-81 candidate
list is exhausted: the low-mass free-float diurnal-swing compression is
NOT owned by exterior gain magnitude, timing, partitioning, or exterior
drivers. What remains in-network: the per-surface interior
storage-release ordering already bounded at ~1.5 K (§Round-80 NOSTORE),
the floor-dominant Table 5-138 per-surface routing deferred (§Round-81)
to the per-surface interior network, and the 3.5 % window retransmission
the lumped arms do not model. Nothing tuned; scratch reverted before
commit; no engine code landed; no re-record battery owed.
