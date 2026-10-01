# LIMIT-29 — investigation history

Narrative history for **LIMIT-29**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-29` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### LIMIT-29: ASHRAE 140 Case 900 thermal-mass time-constant characterization (post-#3770) — Phase B2a PHYSICS-02 (Issue #3799)

- **Description:** Phase B2a (Issue #3799) — diagnostic / audit of
  the thermal-mass time constant for ASHRAE 140 Case 900 (high-mass
  concrete construction, 9R4C default path on the production
  validator). Per the Issue #3799 acceptance criteria ("diagnostic,
  no code changes") and AGENTS.md / RULES.md / ADR-0001
  ("no parameter tuning", "fix the underlying math"), the B2a
  deliverable is **measurement + mechanism hypothesis documentation
  only** — the structural closure work is routed to the
  GaugeSolver production-path work coordinated by **Issues #1465 /
  #1462** (production-path switchover staged via **#3291 / PR
  #3482** for Phase A8 default flip, gated on §LIMIT-21 β-soak
  closure) plus the **#3770 mass-node 141°C regression fix** (the
  underlying Session-84 physics regression that this B2a audit is
  blocked by — the inline comment in the quarantined
  `tests/validation/thermal_mass.rs::test_thermal_mass_temperature_damping`
  records: *"The thermal mass temperature reaches 141°C due to low
  target_tau_hours (2.0)"*, the Session-84 commit is `8408efb`,
  2026-03-31, and the `#[ignore = "awaiting #3770"]` quarantine stays
  in place — this B2a audit does **not** un-quarantine it). The
  measurements below were taken on the **2026-09-17 pre-#3770-fix
  validator snapshot** (develop HEAD `41c1c84`) with
  `ThermalSelector::default()` resolving to `ZoneSolverKind::Gauge`
  but falling through to legacy 5R1C / 9R4C in the **default build**
  (cargo feature `gauge-solver` is OFF per the AGENTS.md Phase A8
  note — `Cargo.toml:208-225`).

- **Four τ measurements bracket the ASHRAE 140 reference envelope
  midpoint (companion: `docs/ASHRAE140_RESULTS.md` §"Phase B2a
  (PHYSICS-02) thermal-mass time-constant characterization (Case 900,
  post-#3770)"):**

  | τ definition | Value | Source | Deviation vs ASHRAE 140 reference envelope midpoint |
  |---|---|---|---|
  | **Wall lumped R·C τ** | **203.4 h (732,064 s)** | `ctf_coefficient_validation::test_case_900_wall_properties` (R_wall = 1.5618 m²·K/W × C_wall = 468.72 kJ/m²·K) | **+1.7% (in-band)** — heavyweight concrete envelope midpoint ≈ 200 h |
  | **ISO 13790 air-coupled τ (active)** | **3.30 h** | `TimeConstantAnalyzer::for_physics(Cm, Σ h_tr_ms)` with measured Cm ≈ 2.0e7 J/K, h_tr_ms ≈ 1687 W/K | **−40% (below envelope midpoint ≈ 5–8 h)** — the structural signature of the Session-84 2-hour `target_tau_hours` over-driving the mass node |
  | **5R1C air-trajectory τ** | **≈ 1.23 h** | inline note in `ashrae_140_case_900::test_case_900_peak_cooling_within_reference_range` | **Above the explicit-Euler stability band** at dt/τ ≈ 0.81 on a 1 h timestep; structural reason for the −69% UNDER on Case 900 peak cooling (0.89 kW vs [1.20, 3.50] kW per §LIMIT-05) |
  | **FiveR1C vs Gauge τ** | FiveR1C **≈ 25.6 h**; gauge τ_mass **≈ 61 h** | `gauge_validation_case_900::test_case_900_gauge_fiver1c_diurnal_parity` (`#[ignore]` per Issue #1669 Option A) | FiveR1C **−60% UNDER**; gauge **+6% in-band** — the bidirectional asymmetry that Issue #1669 captures |

  Per the **deviation analysis** in the B2a ASHRAE140_RESULTS.md
  section, the **total envelope Cm = 20,084.41 kJ/K** (Wall
  123.10 + Roof 126.53 + Floor 98.02 kJ/m²·K, per
  `thermal_mass_coupling_tests::test_total_thermal_capacitance_calculation`)
  is **+33.9% above the ASHRAE 140 envelope midpoint** — consistent
  with the §LIMIT-05 UPDATE (#2453) bidirectional annual-energy
  OVER signature (more mass accumulates more solar injection). The
  h_tr_ms and h_tr_em measurements (1687.14 W/K and 226.90 W/K
  respectively, per `test_thermal_mass_dynamics::test_case_900_conductance_values`)
  are in-band.

- **Three mechanism hypotheses (B2a does not pick a winner — per
  AGENTS.md / RULES.md / ADR-0001):**

  1. **PR #821 ISO 13790 τ-shortening hypothesis (most likely).**
     The active `h_ms = 9.1 × A_m` formulation raised `h_tr_ms` from
     the lookup-table value of ~650 W/K to the measured ~1687 W/K
     (2.6× higher), shortening the ISO 13790 τ from 5.13 h
     (deprecated `TimeConstantAnalyzer::for_case("900")` lookup) to
     3.30 h (active `TimeConstantAnalyzer::for_physics`). On the 24 h
     ASHRAE 140 weather schedule with a 2 h `target_tau_hours`, the
     1.55× shorter τ on the air-coupled path leaves the mass node
     under-damped against the 24 h solar injection envelope, over-
     driving T_mass to 141°C. **Fix is parameter removal** (drop the
     separate `target_tau_hours` parameter; use
     `TimeConstantAnalyzer::for_physics` directly). **Mechanism:
     physics parameter drift; fix is parameter removal, not
     tuning.**

  2. **Wall lumped R·C τ dominance hypothesis.** The wall material τ
     (203.4 h) is ~60× larger than the air-coupled τ (3.30 h). If
     the Session-84 change routed the mass-node forcing through the
     wall R·C path instead of the air-coupled path, the effective τ
     would be ~200 h (very slow) and the mass node would integrate
     solar injection over many hours without sufficient damping.
     **Fix is path re-routing**, not tuning.

  3. **5R1C air-trajectory τ under-stability hypothesis.** The
     5R1C implicit-Euler formulation has a stability band of dt/τ <
     1; the Case 900 τ at 1.23 h and dt = 1 h sits at dt/τ ≈ 0.81 —
     at the edge of stability. On a 24 h schedule with 6 h solar
     peak (Case 900 noon-peak solar flux ≈ 800 W/m² on the south
     wall + roof), the integrator could oscillate and deposit solar
     energy into the mass node without damping. **Fix is sub-hour
     sub-stepping** (already proposed and blocked by #2300 / §LIMIT-05
     UPDATE), or **GaugeSolver's continuous-time formulation**.

  The three mechanisms are **not mutually exclusive** — the #3770
  fix likely needs to address all three to close the mass-node
  runaway AND restore the ASHRAE 140 Case 900 annual-energy
  bidirectional OVER signature tracked under §LIMIT-05 UPDATE
  (#2453). The B2a audit does **not** propose a fix; it documents
  the measurements.

- **Cross-references:**

  - **B2b / Issue #3800 / PR #3841** — the `release_gates.yaml →
    validation.individual.known_failures` cohort registration that
    this B2a audit is the **precursor measurement for** (the
    `release_gates.yaml` comment block at lines 64–81 already cites
    *"The B2a audit (Issue #3799) characterised the per-step
    thermal-mass response; B2b (this issue) registers the cohort at
    the release-gates layer without retuning any number, baseline,
    or gate logic per AGENTS.md / RULES.md / ADR-0001"*). **No B2a
    code change** — the B2a measurements above are the input the
    B2b cohort registration cites.

  - **#3770** — the underlying mass-node 141°C regression that this
    B2a audit was blocked by (Issue #3799 body: *"Blocked by #3770:
    Case 900 baselines are untrustworthy until the mass-node runaway
    is fixed and `test_thermal_mass_temperature_damping` is
    un-quarantined. Do not start before #3770 closes"*). **RESOLVED
    via PR #3770**: the Session-84 physics regression was eliminated
    by PR #2717 (removed empirical tuning factor per v1.3 no-tuning
    rule) followed by the Phase B2a τ-characterization audit (this
    PR #3845 / Issue #3799) which documented the physical τ
    derivation path. Live measurement post-fix: initial 20.00 °C →
    final 20.75 °C (well within the −50..100 °C physical
    plausibility band). The `test_thermal_mass_temperature_damping`
    `#[ignore]` was removed and the original assertions restored and
    passing; the `tests/QUARANTINE.md` row at #284 was moved to
    `closed (resolve #3770)`. The un-ignore criterion for
    `test_thermal_mass_temperature_damping` is mechanical (the
    −50..100 °C physical plausibility band is a physical bound, not
    a tuned baseline — see RULES.md). The B2a measurements above do
    not affect the #3770 fix path.

  - **§LIMIT-13 / Issue #3063 / ADR-0009** — `h_tr_em`
    time-invariance regression fence (the 18.3 W/m²K canonical
    exterior film coefficient is **unchanged** through the B2a
    measurement window — guarded by
    `tests/regression_exterior_film_unification.rs`; the AGENTS.md
    "no regression on the canonical film coefficient" guard holds
    throughout). The B2a measurements above do not regress this
    fence.

  - **§LIMIT-05 UPDATE (#2453)** — 900-series bidirectional
    annual-energy over-prediction (the air-mass distribution
    pathology is the same as the 5R1C over-damping signature; the
    Cm = +33.9% above-envelope and the τ = −40% below-envelope
    deviations here are consistent with the bidirectional OVER).

  - **§LIMIT-05** — high-mass peak cooling UNDER (the 5R1C τ ≈ 1.23
    h under-stability signature; the dt/τ ≈ 0.81 measurement is
    the structural reason for the −69% UNDER on Case 900 peak
    cooling).

  - **§LIMIT-16 / Issue #3059** — Cases 610 / 630 / 650 peak
    cooling OVER (the low-mass cousin of the Case 900 high-mass
    under-stability; same discrete-node solar-injection pathology).

  - **§LIMIT-17 / Issue #3058 / ADR-0011** — Case 950FF night-vent
    mass coupling gap (the night-vent ACH = 13.14 → h_ve ≈ 570.8
    W/K coupling is structurally similar to the τ-shortening
    mechanism hypothesis above).

  - **§LIMIT-22** — the gauge-build-only
    `test_case_950_mass_temperature_precooled_issue_1422`
    quarantine (Issue #3297) provides the gauge τ_mass ≈ 61 h
    measurement used in the gauge-parity table above. The
    §LIMIT-22 entry's exact Crank-Nicolson proxy is what makes the
    gauge τ_mass measurement meaningful (the legacy trivial proxy
    writes `t_mass = t_air` and cannot resolve the mass time
    constant).

  - **§LIMIT-21** — Gauge β-path pre-existing air-trajectory
    failure cohort + **β-soak 30-night production-path gate**
    (Issue #3286, `#3286 β-soak` convention in CI comment threads;
    currently 0/30 nights green). The B2a measurements above will
    persist on the production path until the `gauge-solver` cargo
    feature is enabled.

  - **PR #3041 / Issue #3059** — the
    `MAX_CONVECTIVE_TO_AIR_MULTIPLIER = 2.0×` cap is the sibling
    partial-fix on the low-mass Case 610 / 630 / 650 cooling OVER
    axis; no equivalent mass-cap exists for the high-mass Case 900
    τ-shortening axis (the §LIMIT-29 entry does NOT propose
    introducing one — per ADR-0001 / AGENTS.md / RULES.md).

- **Unblockers:**

  - **GaugeSolver production-path switchover** (Issues **#1465 /
    #1462**, both closed individually; production-path staged via
    **#3291 / PR #3482** for Phase A8 default flip, gated on
    §LIMIT-21 β-soak closure).

  - **#3770 fix** — once the mass-node 141°C regression is closed
    and `test_thermal_mass_temperature_damping` is un-quarantined,
    the B2a measurements can be re-run on the post-#3770 ground
    truth to confirm that the τ deviations vs the ASHRAE 140
    reference envelope are structural (not a Session-84 regression
    artefact). The three mechanism hypotheses above are designed
    to be disambiguated by the #3770 fix.

- **Affected Cases / Metrics (post-#1323 baseline):**

  - Case 900 — Annual heating 5.13 MWh (Ref: 1.17–2.04 MWh, ~+200% OVER);
    Annual cooling 7.75 MWh (Ref: 2.13–3.67 MWh, ~+150% OVER); Peak
    heating 3.93 kW (Ref: 1.80–2.40 kW, ~+85% OVER); Peak cooling
    3.36 kW (Ref: 1.60–2.10 kW, ~+95% OVER). All 4 ASHRAE 140
    reference-band metrics are FAIL-rows on the 84-metric scorecard
    (`docs/ASHRAE140_RESULTS.md` §"Detailed Results / High-Mass
    Cases (900 Series)"). The Case 900 cohort is in
    `release_gates.yaml → validation.individual.known_failures`
    (added by B2b / Issue #3800 / PR #3841 — see the
    `release_gates.yaml` comment block at lines 64–81 for the
    B2a → B2b provenance).

- **Severity:** **High** — the Case 900 thermal-mass cohort is one
  of the two fundamental ASHRAE 140 validation axes (the other being
  the 600-series and FF cohort tracked by §LIMIT-05 / §LIMIT-16 /
  §LIMIT-28). The Case 900 mass-node 141°C runaway is the **#3770
  physical-plausibility violation** that gates `test_thermal_mass_temperature_damping`
  on the QUARANTINE.md registry (`tests/QUARANTINE.md` wildcard row
  `src/validation/thermal_mass.rs | test_thermal_mass_* | #3770`).
  The cohort gap blocks the strict ±15% annual-energy gate per
  `release_gates.yaml → validation.individual.known_failures`
  (Case 900 is excluded from the gate's `extreme_count` because the
  cohort gap is structural — the FF cohort, not Case 900, is the
  structural primary that the `known_failures` exclusion is
  *predicated on*).

- **GitHub Issue:** [#3799](https://github.com/anchapin/fluxion/issues/3799)
  (Phase B2a PHYSICS-02 thermal-mass time-constant characterization,
  post-#3770 precursor to B2b / Issue #3800).

- **Status:** 🟡 **Docs + characterization shipped; structural fix
  routed to GaugeSolver #1465 / #1462 + #3770 fix.** No
  physics-code change; no
  `MAX_CONVECTIVE_TO_AIR_MULTIPLIER` / `h_ve_night` / `h_tr_em_wall`
  / `solar_distribution_to_air` / `derived_h_tr_3` /
  `target_tau_hours` change; no
  `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`
  change; no ASHRAE 140 reference-band change; no `#[ignore]`
  quarantine change; no reference-data CSV / sha256 change; no
  construction τ parameter / `TimeConstantAnalyzer` change; no
  `regression_exterior_film_unification.rs` change (the canonical
  18.3 W/m²K film coefficient is preserved — LIMIT-13 regression
  fence stays green).

- **Sibling framing:** §LIMIT-13 (h_tr_em time-invariance
  regression fence); §LIMIT-05 / §LIMIT-05 UPDATE (#2453)
  (900-series annual-energy + peak cooling siblings); §LIMIT-16
  (low-mass peak cooling OVER cousin); §LIMIT-17 / §LIMIT-24
  (Case 950 FF+HVAC sibling entries); §LIMIT-22 (gauge τ_mass
  measurement source); §LIMIT-21 (β-soak 30-night production-path
  gate). **Cross-PHYSICS-02**: B2b / Issue #3800 / PR #3841 (the
  release-gates cohort registration that cites this B2a audit as
  the precursor measurement); **cross-PHYSICS-02 (blocked-by)**:
  #3770 (the mass-node 141°C regression whose fix unblocks the
  B2a audit's `test_thermal_mass_temperature_damping` quarantine).
