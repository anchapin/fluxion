# LIMIT-21 — investigation history

Narrative history for **LIMIT-21**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-21` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 5. No wording was changed, softened or deleted.

---

#### LIMIT-21 UPDATE (2026-09-25): ADR-0017 — DAE teacher adopted; flip authority moves to #3986; two-solver limbo closed

**ADR-0017 (Issue #3978, `docs/adr/0017-equation-based-dae-teacher-architecture.md`)** supersedes the β-soak-gated production flip this LIMIT previously described:

- **Flip authority:** the unconditional default flip to the **equation-based DAE teacher** is gated on the **#3986 teacher validation suite** passing — no longer on §LIMIT-21 closure / β-soak tripping (#3297). The β-soak (#3286, still 0/30) continues as the nightly authority on the `gauge-solver` prototype arm.
- **Two-solver limbo closed:** `ThermalSelector::default()` is cfg-dependent and explicit in every build — `Gauge` in `gauge-solver` builds; **explicit legacy `FiveROneC`** (HighMass ⇒ 9R4C auto-promotion preserved) in default builds. An explicit `Gauge` selector in a default build **panics loudly** at `from_spec_with_selector` construction; the silent cfg fall-through to declared-dead 5R1C (#1522) was removed. Zero default-build physics drift was verified (ASHRAE failure sets byte-identical to `develop` in both feature states).
- **GaugeSolver fate:** designated the **DAE-path prototype** (D3) — #3982 must demonstrate a genuine thermal-mass node or the ADR directs retirement. The LIMIT-21 air-trajectory residuals above are exactly the #3982 work list; the 2026-09-20 "no tunable fix" conclusion below stands unchanged.

#### LIMIT-21 UPDATE (2026-09-20): Structural root cause confirmed — no tunable fix; fix routed to GaugeSolver #1465 / #1462

**Context:** PRs #3906 (LIMIT-17 night-vent double-count fix, `c1f28e1` + `604c623`) and #3908 (gauge H·dt sub-stepping for stability, `460bfad`) merged to develop on 2026-09-20. The β-soak nightly workflow (`.github/workflows/nightly-ashrae-140-gauge.yml`) has recorded **0 consecutive green runs** since 2026-09-01 — the streak is pinned at 0/30 (per Issue #3286 CI comment thread; all runs are ❌). This UPDATE documents the current LIMIT-21 status and confirms the structural fix path.

**Root cause — `dt/τ ≈ 3.6` air-node pathology (LIMIT-05 UPDATE #1522 confirmed):** The GaugeSolver air trajectory fails on the same single-lumped thermal-mass node pathology documented in §LIMIT-05 UPDATE (#1522) and §LIMIT-16. The air-node time constant for ASHRAE 140 cases is `τ_air = C_air / h ≈ 0.28 h`, giving `dt/τ ≈ 3.6` at the 1-hour simulation timestep. The air node equilibrates ~98% per step, leaving almost no thermal memory. Winter night-sky radiative forcing pulls T_air to **−42.91 °C** on Case 600FF / 650FF free-float (documented at develop `71fbb31`, 2026-09-20).

Three integration methods were tested in LIMIT-05 UPDATE (#1522) for the same `dt/τ ≈ 3.6` signature:

| Method | Carry-over weight | Peak_cooling | Peak_heating | Net result |
|--------|-------------------|--------------|--------------|------------|
| Legacy (no C_air) | 0 % | OVER +48 % | UNDER −24 % | 13/14 fail |
| Exact exponential | 1.6 % | OVER +18 % | UNDER −27 % | 12/15 fail |
| Implicit Euler | 22 % | OVER +18 % | UNDER −40 % | 9/18 fail |

**Bidirectional failure is the blocker:** `peak_cooling OVER` and `peak_heating UNDER` point in opposite directions — no single air-node damping reduces one without worsening the other. The LIMIT-05 UPDATE (#1522) air-node capacitance option was marked **INFEASIBLE at 1 h timestep**.

**No tunable fix available:** Per **RULES.md**, **AGENTS.md**, and **ADR-0001** ("no parameter tuning", "must-never hardcode results", "fix the underlying math"), the `−42.91°C` free-float minimum and the associated air-trajectory failures cannot be closed by changing any physics constant (`h_ve_night`, `solar_distribution_to_air`, `MAX_CONVECTIVE_TO_AIR_MULTIPLIER`, `h_ms_coeff`, `derived_h_tr_3`, etc.). The `tests/reference_data/zone_balance/strict_energy_gate_baseline.json` baseline must NEVER be raised to hide a regression.

**Confirmed LIMIT-21 cohort (develop `71fbb31`, 2026-09-20):**
- `tests/zone_balance_eplus_isolation.rs` — 2 failures: `test_physics_thermal_model_eplus_case_600_reference_csv` (T_zone mean −12.59 °C, max step jump 34.007 °C); `test_free_floating_case_900ff_isolation` (divergence to non-finite)
- `tests/ashrae_140_case_600_series.rs` — 21/27 failing (nightly criterion 4)
- `tests/known_issues_regression.rs::issue_1457_case_600_series_tracking` — 13 Case 600-series metrics out-of-band (nightly criterion 5; `Case600FF/min_free_float = −42.91°C`)
- `tests/ashrae_140_case_960_sunspace.rs` — 3 failures (sunspace instability, ±140 °C step-level ΔT spikes)
- `tests/ashrae_140_blind_validation.rs` — 1 failure (`test_case_950_night_flush_zone_cooling_in_july`)

**Architectural fix only:** The only viable fix is the GaugeSolver air-trajectory fidelity program (#1465 / #1462) — treating solar as geometric curvature rather than per-timestep energy injection, natively preventing the over-injection that drives the `dt/τ ≈ 3.6` air-node pathology. This is the same architectural route as §LIMIT-05, §LIMIT-16, §LIMIT-17, and the entire §LIMIT-05 UPDATE family.

**Default-build path unaffected:** The **default build (cargo feature `gauge-solver` OFF)** continues to route the `Gauge` selector through the legacy 5R1C/9R4C fallthrough. The LIMIT-21 failures are **gauge-build-only** (`--features gauge-solver`) and do not affect the production default path. Phase A8 (PR #3482) staged the `Gauge` selector as unconditional default but retained the cargo feature as the production-path gate pending §LIMIT-21 β-soak closure.

**What this UPDATE does NOT do:** No physics code changed; no `strict_energy_gate_baseline.json` raised; no ASHRAE 140 tolerance band relaxed; no parameter tuned. This is a documentation-only status update confirming the structural fix path and the 0/30 β-soak streak.

#### LIMIT-21 UPDATE (2026-09-21): Phase 6 — window solar splitting + sky-radiation boundary (PR #3912, Issue #3916)

**Context:** PR #3912 merged to develop on 2026-09-21, completing Phase 6 of the LIMIT-21 air-trajectory fidelity program. This phase adds per-surface sky-radiation boundary inputs and a window solar distribution parameter to the GaugeZoneSolver.

**Phase 6 changes (PR #3912):**
- Added `solar_distribution_to_air` parameter to GaugeZoneSolver — 30 % of window-transmitted solar enters zone air directly; 70 % enters through window conduction path
- Added sky-radiation boundary inputs (`t_sky`, `h_rad_sky`) to `GaugeZoneSolver` for explicit night-sky radiative coupling
- Reverted a bug where unconditional `calc_analytical_loads` call in the gauge path was breaking the Surrogate Drift Tolerance Gate
- Case 640 annual cooling improved: 1.31 → 1.62 MWh (still ~75 % below the 5.95–8.10 MWh reference band)

**Remaining gap — Issue #3916:** The ~1 MWh annual cooling gap for Case 640 (1.62 MWh actual vs 5.95–8.10 MWh expected) is a known architectural limitation. The gauge solver lacks two mechanisms that 5R1C has: (1) **Solar lag correction** — 5R1C filters high-frequency solar transients through a geometric-mean time constant (τ_lag = √(τ_air × τ_mass)); gauge has no equivalent; (2) **Exact exponential vs Implicit Euler** for the air ODE — gauge uses implicit Euler (no analytical solution), giving ~84 %/hr equilibration vs 5R1C's ~78 %/hr. The air-trajectory fidelity investigation is tracked in `.planning/issue-gauge-air-trajectory-fidelity.md` and GitHub Issue #3916. Per **RULES.md** / **AGENTS.md** / **ADR-0001** ("no parameter tuning", "fix the underlying math"), no physics constant is adjusted to close this gap — the structural fix is routed to the GaugeSolver rework **#1465 / #1462**.

**Phase 6 is complete.** The β-soak nightly criteria remain red (0/30); closure of the remaining air-trajectory gap is Issue #3916.

#### LIMIT-21 UPDATE (2026-09-21): Phase 9 — Issue #3916 formally closed; new issue #3918 opened for GaugeZoneSolver interior surface node network

**Context:** Issue #3916 formally closed as "requires new issue for GaugeZoneSolver interior surface node network." New issue **#3918** opened to track the specific architectural work: per-surface T_s tracking, h_tr_is computation, and lag_input = h_tr_is × φ_st / term_rest_1 ≈ 1.9% of window solar.

**Diagnostic written:** `tests/diagnostics/diag_air_node_equilibration.rs` — run with `cargo test --test all_tests diag_air_node_equilibration:: -- --ignored --nocapture` (requires `--features gauge-solver`). Compares 5R1C vs GaugeZoneSolver air-node dynamics under identical weather (Denver TMY3) and initial conditions for Case 640:

| Model | Heating | Cooling | H/C Ratio | vs Reference |
|-------|---------|---------|-----------|-------------|
| Gauge | 6.99 MWh | 1.47 MWh | 4.74 | H +84%, C −75% |
| 5R1C | 4.07 MWh | 3.84 MWh | 1.06 | H +7%, C −35% |
| Reference | 2.75–3.80 | 5.95–8.10 | ~0.45 | — |

Gauge winter heating gap: +49% vs 5R1C (2.52 → 3.74 MWh Dec-Feb). Gauge summer cooling gap: −35% vs 5R1C (1.63 → 1.06 MWh Jun-Aug).

**Root cause confirmed:** Gauge's faster air equilibration (~84%/hr vs 5R1C's ~78%/hr) dissipates solar gains before they can drive cooling demand, and drives faster overnight heat loss in the setback regime requiring more heating recovery. The 5R1C's τ_lag = √(τ_air × τ_mass) solar lag filter requires computing lag_input = h_tr_is × φ_st / term_rest_1 ≈ 1.9% of window solar — GaugeZoneSolver cannot compute this without an interior surface node. The fix requires the GaugeSolver rework tracked in **Issue #3918** (not covered by #1465/#1462 which shipped the framework but not this specific zone-level coupling). Phase 9 is complete.

**LIMIT-22 added (Issue #3297):** Gauge-build-only test failures exposed by replacing the PR2.5 trivial mass-state proxy (`t_mass = (h_tr_em·T_air + h_tr_3·T_air)/(h_tr_em + h_tr_3)`, ~50–170 kWh non-zero strict-gate residual per the #3297 issue body) with the exact Crank-Nicolson mirror of the strict gate (`write_gauge_mass_state_proxy`, `fd7ef13`). Three tests that passed before `fd7ef13` fail on the gauge build — each was passing for a physically-wrong reason: (1) `test_case_950_mass_temperature_precooled_issue_1422` — the > 2 °C overnight mass pre-cool band was satisfied by the trivial proxy writing `t_mass = t_air` (air swing, no mass time constant); the exact CN node at Case 950's τ_mass ≈ 61 h attenuates a 12-h overnight air swing by 1/√(1+(2π·61/12)²) ≈ 0.031 and swings +1.09 °C on the gauge air trajectory (legacy 5R1C: +2.41 °C at T_mass ≈ +41 °C July vs gauge ≈ −27.6 °C); (2) `test_case_960_inter_zone_heat_transfer_analysis` — passed pre-#3297 on the pure-legacy fall-through; with the multi-zone arm re-enabled the gauge integration is oscillatory-unstable for the Case 960 sunspace (±140 °C step-level ΔT spikes around the #3297 fail-closed [−50, 100] °C guard; annual means in-band at ≈ 13.8 / 19.8 °C); (3) `test_different_zones_respond_differently_to_targeted_gain` — the checker's 5R1C residual routes an artificial load gain through φm·m_air_frac only, so with `m_air_frac = 0` the gain leverage is structurally zero and the exact-CN proxy makes every zone imbalance exactly 0 (the pre-#3297 pass was vacuous on the trivial proxy's non-zero baseline residual; sibling of §LIMIT-19 / #3103). All three are quarantined gauge-build-only via `#[cfg_attr(feature = "gauge-solver", ignore = "...")]` — the default-build assertions remain fully live and pass (zone_balance 19/0/2; all three green). No threshold, baseline, or checker formula was changed; no production code was changed. Unblockers: Issue **#3291** (Phase A8 default flip — merged via PR #3482; `ThermalSelector::default() = ZoneSolverKind::Gauge`, gated on the `gauge-solver` cargo feature and §LIMIT-21 β-soak closure) plus #1465 / #1462 (air-trajectory fidelity + multi-zone stability) and the §LIMIT-19 #1344 artificial-gain investigation. See §LIMIT-22.

**LIMIT-24 added (Issue #3551):** Case 950 HVAC-mode annual cooling measures **33.08 kWh vs the ASHRAE 140 reference band 390–920 kWh** (~14× UNDER, ~91 % below the lower bound); peak cooling 0.39 kW vs [0.70, 0.90] kW band (~44 % UNDER). Both metrics are fail-rows on the 84-metric scorecard. This entry is the docs-only structural companion to §LIMIT-17 / #3058 (Case 950FF night-vent free-floating min −23.92 °C vs [−20.20, −17.80] °C band, 3.72 °C outside). §LIMIT-17 records a **regression-avoidance clause** requiring any future solver change to "preserve Case 950 (HVAC mode) annual cooling in the 390–920 kWh band" — but the current HVAC-mode value (33.08 kWh) is already ~14× outside that band, so the preserved-HVAC target is far from the actual HVAC state, and the failure has no separate structural LIMIT entry to track it. The `MAX_CONVECTIVE_TO_AIR_MULTIPLIER = 2.0×` cap from PR #3041 closed Case 650 cooling OVER but did not transfer to Case 950 (HVAC) because Case 950's `h_ve_night ≈ 570.8 W/K` (18:00–07:00) deposits the cooling load into the **mass node (multi-node path)** rather than the **air node (5R1C path)** where the HVAC controller reads the setpoint signal; the derived-`h_tr_3` path that limits Case 950FF winter-min over-prediction (§LIMIT-17 root-cause) also deflects the summer peak away from the air node. The two signatures (HVAC cooling UP + FF min DOWN) are **bidirectionally coupled** — no parameter adjustment to `h_ve_night`, `MAX_CONVECTIVE_TO_AIR_MULTIPLIER`, or `solar_distribution_to_air` can close both at once without violating AGENTS.md / RULES.md / ADR-0001 ("no parameter tuning", "fix the underlying math"); per-case parameter tuning is explicitly out of scope. The architectural fix is routed to the GaugeSolver rework **#1465 / #1462** (both closed individually; production-path switchover staged via #3291 / PR #3482 — Phase A8 default flip, gated on the `gauge-solver` cargo feature and §LIMIT-21 β-soak closure). Sibling entries: §LIMIT-17 / #3058 (Case 950FF companion + regression-avoidance clause), §LIMIT-16 / #3059 (Cases 610/630/650 peak cooling OVER cohort), §LIMIT-05 UPDATE (#2453) (900-series bidirectional annual-energy cohort). Per-month attribution diagnostic `tests/diagnostics/case_950_hvac_mode_seasonal_attribution.rs` is wired into CI (`#[ignore]`-quarantined, runs `--ignored --nocapture`) per the Issue #3551 acceptance criterion; full implementation is a follow-up PR. See §LIMIT-24 for the per-metric engine-vs-reference table, the bidirectional-signature analysis, the §LIMIT-17 regression-avoidance-clause cross-reference, the four-test affected-tests list, and the closure-criterion statement that the same single solver change must close **both** Case 950 (HVAC) annual cooling (390–920 kWh) **and** Case 950FF min free-floating temperature ([−20.20, −17.80] °C) — i.e. the bidirectional fix requires GaugeSolver-style path splitting, not a per-parameter tuning.
**LIMIT-23 added (Issue #3552):** Case 970 (5-zone multi-zone cross-coupling per ASHRAE 140-2017 §B6.7 / 140-2023 Annex B8-3, `sim::multi_zone_network::MultiZoneAirflowNetwork` with 5×5 symmetric inter-zone conductance matrix) reports 4/4 ASHRAE 140 reference-band metrics failing on the 2026-08-16 snapshot — annual heating 18.58 MWh vs reference band [10.54, 14.26] MWh (+30 % to +76 % OVER the band), annual cooling 21.07 MWh vs [7.39, 10.00] MWh (+110 % to +185 % OVER), peak heating 3.80 kW vs [4.00, 8.00] kW (5 % UNDER the low edge), peak cooling 2.58 kW vs [2.50, 5.50] kW (at the low edge). The bidirectional annual OVER signature (heating AND cooling simultaneously >+30 % above the reference band on a 5-zone topology) is consistent with the §LIMIT-05 UPDATE (#2453) 900-series bidirectional over-prediction mechanism — the 5R1C/9R4C air-mass distribution in a 5-zone coupling matrix amplifies the same solar mass-node over-charge that drives the 900-series and §LIMIT-14 / LIMIT-16 / LIMIT-17 cohorts. Reference bands are maintained in `validation::benchmark` and summarised in `docs/ASHRAE140_MULTI_ZONE_RESULTS.md` §"Case 970 Reference Data"; the per-metric engine-output table below is the full strict-gate view. **Documentation/tracking only entry; no physics-code change, no `inter_zone_conductance` / `solar_distribution_to_air` / `h_ms_coeff` change, no `tests/reference_data/zone_balance/case_970_energy_reference.csv` raise.** Case 970 is intentionally NOT added to the §LIMIT-05 / #3072 aggressive-baseline cohort table (per Issue #3552 acceptance criteria: "tracked separately; the cohort list is #3072's purview"); cohort-level tracking remains owned by Issue #3072 (Cases 195 / 600 / 620 / 940 / 960). See §LIMIT-23 for the per-metric engine-output vs reference-band table, the per-zone attribution test stub, and the cross-references to Issue #1446 (Case 970 multi-zone implementation, closed via #1467), §LIMIT-05 / LIMIT-14 / LIMIT-16 cohort, and #1465 / #1462 (GaugeSolver architectural unblocker).

**LIMIT-27 added (Issue #3647):** `tests/test_energy_conservation.rs` cited "FREE-04 in docs/KNOWN_ISSUES.md" as the documented rationale for the 200 W `ENERGY_BALANCE_RESIDUAL_THRESHOLD` (the ~191 W systematic residual of the 5R1C `InvariantChecker` algebraic formulation, introduced by #2225 / PR #2230, commit `12425d8`) — but no `FREE-04` heading ever existed (the FREE-\* family covers free-floating temperature findings only). This is a doc-vs-code drift: someone assuming the citation was stale could delete the test and silently de-facto-relax the strict-energy guard. Fixed docs-side: the residual is now documented as **§LIMIT-27** (25/26 are reserved for the Cases 800 / 810 structural entries per the `strict_energy_gate_baseline.json` `_doc_issue3572` note), the test comment is repointed, and `scripts/check_known_issues_links.py` now validates every `FREE-NN` / `LIMIT-NN` token in `tests/**/*.rs` against actual `### TOKEN` headings. No physics-code change; no threshold change.

**LIMIT-29 added (Issue #3799):** Phase B2a (PHYSICS-02) thermal-mass time-constant characterization of ASHRAE 140 Case 900 (high-mass concrete construction, 9R4C default path per `ThermalSelector::default()` falling through to legacy 5R1C / 9R4C in the default build per the AGENTS.md Phase A8 note — cargo feature `gauge-solver` OFF, `Cargo.toml:208-225`) on the 2026-09-17 pre-#3770-fix validator snapshot (develop HEAD `41c1c84`); the B2a audit extracted four τ values that bracket the ASHRAE 140 reference envelope midpoint and named three competing mechanism hypotheses for the **#3770 mass-node 141°C regression** (Session-84 physics commit `8408efb`, 2026-03-31, recorded in the quarantined-test inline comment as *"The thermal mass temperature reaches 141°C due to low target_tau_hours (2.0)"*). Per the Issue #3799 acceptance criteria ("diagnostic, no code changes") and AGENTS.md / RULES.md / ADR-0001 ("no parameter tuning", "fix the underlying math"), the B2a deliverable is **measurement + mechanism hypothesis documentation only** — the structural closure is routed to the GaugeSolver production-path work coordinated by #1465 / #1462 and the #3770 fix. The four measured τ values are: (1) **Wall lumped R·C τ = 203.4 hours (732,064 s)** from R_wall × C_wall = 1.5618 m²·K/W × 468.72 kJ/m²·K (`tests/all_tests/ctf_coefficient_validation.rs::test_case_900_wall_properties`); this is the **wall-material-only** time constant and is **in-band** with the ≈150–250 h ASHRAE 140 reference envelope for heavyweight concrete construction (deviation +1.7%); (2) **ISO 13790 air-coupled τ = 3.30 hours** (active, post-PR #821) from Cm / Σ h_tr_ms / 3600 = 2.0e7 J/K / 1687.14 W/K / 3600 (`TimeConstantAnalyzer::for_physics`); this is **below envelope** at ~−40% deviation vs the ≈5–8 h reference midpoint — the structural signature that the Session-84 2-hour `target_tau_hours` over-drove the mass node against the ISO 13790 physics τ; (3) **5R1C air-trajectory τ ≈ 1.23 hours** (inline note in `ashrae_140_case_900::test_case_900_peak_cooling_within_reference_range`); this is **above the explicit-Euler stability band** at dt/τ ≈ 0.81 on a 1 h timestep, the structural reason for the −69% UNDER on Case 900 peak cooling (0.89 kW vs [1.20, 3.50] kW reference band per §LIMIT-05); (4) **FiveR1C vs Gauge air-trajectory τ**: FiveR1C ≈ 25.6 h, gauge τ_mass ≈ 61 h (`gauge_validation_case_900::test_case_900_gauge_fiver1c_diurnal_parity` `#[ignore]` per Issue #1669 Option A); the FiveR1C τ is **−60% UNDER** the ≈50–80 h reference midpoint, the gauge τ is **+6% in-band** — the bidirectional asymmetry is the structural signature Issue #1669 captures. The **three mechanism hypotheses** for the #3770 mass-node 141°C runaway that are consistent with the τ measurements above (the B2a audit does not pick a winner — per AGENTS.md / RULES.md / ADR-0001): (a) **PR #821 ISO 13790 τ-shortening hypothesis (most likely)** — the `h_ms = 9.1 × A_m` reformulation raised `h_tr_ms` from ~650 W/K (lookup) to ~1687 W/K (active), shortening the ISO 13790 τ from 5.13 h to 3.30 h (1.55× shorter); on the 24 h ASHRAE 140 weather schedule with a 2 h target_tau_hours, the 1.55× shorter τ on the air-coupled path leaves the mass node under-damped, over-driving T_mass to 141°C; fix is **parameter removal** (drop the separate `target_tau_hours` parameter, use `TimeConstantAnalyzer::for_physics` directly); (b) **Wall lumped R·C τ dominance hypothesis** — the wall material τ (203.4 h) is ~60× larger than the air-coupled τ (3.30 h); if the Session-84 change routed the mass-node forcing through the wall R·C path instead of the air-coupled path, the effective τ would be ~200 h (very slow) and the mass node would integrate solar injection without sufficient damping; fix is **path re-routing**, not tuning; (c) **5R1C air-trajectory τ under-stability hypothesis** — the Case 900 τ at 1.23 h on a 1 h timestep sits at dt/τ ≈ 0.81, at the edge of the explicit-Euler stability band (dt/τ < 1); on a 24 h schedule with 6 h solar peak the integrator could oscillate and deposit solar energy without damping; fix is **sub-hour sub-stepping** (already proposed and blocked by #2300 / §LIMIT-05 UPDATE), or **GaugeSolver's continuous-time formulation**. The three mechanisms are **not mutually exclusive** — the #3770 fix likely needs to address all three to close the mass-node runaway AND restore the ASHRAE 140 Case 900 annual-energy bidirectional OVER signature tracked under §LIMIT-05 UPDATE (#2453). Tracked as a documentation-only entry; the structural fix is routed to the GaugeSolver production-path work coordinated by #1465 / #1462 plus the #3770 fix. Sibling entries: §LIMIT-13 (h_tr_em time-invariance regression fence — 18.3 W/m²K canonical exterior film coefficient unchanged, guarded by `tests/regression_exterior_film_unification.rs`); §LIMIT-05 / §LIMIT-05 UPDATE (#2453) (900-series bidirectional annual-energy over-prediction; the Cm = +33.9% above-envelope and τ = −40% below-envelope deviations here are consistent with the bidirectional OVER signature); §LIMIT-16 / Issue #3059 (Cases 610 / 630 / 650 peak cooling OVER — the low-mass cousin of the Case 900 high-mass under-stability); §LIMIT-17 / Issue #3058 / ADR-0011 (Case 950FF night-vent mass coupling gap — structurally similar to the τ-shortening mechanism hypothesis); §LIMIT-22 (the gauge τ_mass ≈ 61 h measurement used in the gauge-parity table is from #3297's exact Crank-Nicolson proxy). Phase A8 default flip (Issue #3291, merged via PR #3482) wires `ThermalSelector::default() = ZoneSolverKind::Gauge`; the `gauge-solver` cargo feature is intentionally retained as the production-path gate pending §LIMIT-21 β-soak closure — the B2a measurements above will persist on the production path until the feature is enabled. See §LIMIT-29 for the per-metric measurement table, the deviation analysis vs the ASHRAE 140 reference envelope, the three mechanism hypotheses, the cross-references to B2b / Issue #3800 / PR #3841 (the release-gates cohort registration that this B2a audit is the precursor measurement for — `release_gates.yaml` lines 64–81 already cite B2a as the precursor), the #3770 mass-node 141°C regression that B2a is blocked by, and the module-isolation suites (read-only acceptance — all green; the `test_thermal_mass_temperature_damping` quarantine from #3770 stays in place until #3770 closes; this B2a audit does **not** un-quarantine it).

**LIMIT-30 added (Issue #3797):** Phase B1a (PHYSICS-01) solar distribution audit of the ASHRAE 140 Case 600 series (600/610/620/630/640/650, the §LIMIT-05 / #3072 aggressive-baseline low-mass cohort) on the 2026-09-17 develop HEAD with `ThermalSelector::default()` resolving to `ZoneSolverKind::Gauge` but **falling through to legacy 5R1C / 9R4C** in the **default build** (cargo feature `gauge-solver` is OFF per the AGENTS.md Phase A8 note — `Cargo.toml:208-225`). Per the Issue #3797 acceptance criteria ("diagnostic, no code changes") and **AGENTS.md / RULES.md / ADR-0001** ("no parameter tuning", "fix the underlying math"), the B1a deliverable is **per-tilt / per-azimuth incident-energy deviation table + ranking of the failing distribution metrics + initial mechanism hypotheses only** — the structural closure is owned by B1b / Issue #3798 / PR #3847 at the release-gates layer (already merged) and the GaugeSolver production-path work coordinated by #1465 / #1462. The **per-tilt / per-azimuth fluxion vs analytical incident-energy calculation PASSES** (all 5 tilts {0°, 30°, 60°, 90°, 180°} at az=180° (south) in the [0.99, 1.01] annual ratio band on the Denver TMY3 year; horizontal-beam annual fluxion 1041.332 kWh/m²/year vs analytical 1041.332 kWh/m²/year, max abs deviation 0.0000 W/m²; 0 of 3573 hours-with-sun exceed 1 % per-hour tolerance; Mock-vs-Physics 1 % parity on the full 16 × 8760 (tilt × az × hour) assertion grid per `tests/all_tests/surface_flux_parity.rs::test_parity_combined_tilt_azimuth_matrix`) — **the failing axis is NOT in the per-tilt / per-azimuth arithmetic**, it is downstream in the per-surface distribution routing parameters and the 5R1C coupling. The **per-surface distribution metrics that FAIL** (printed by `tests/all_tests/solar_distribution_validation.rs`): (a) **Case 600 (LowMass) `solar_distribution_to_air = 0.30` vs ASHRAE 140 expectation 0.0, Δ +0.30 (worst offender)**; (b) **Case 600 `solar_beam_to_mass_fraction = 0.30` vs ASHRAE 140 expectation 1.0, Δ −0.70**; (c) **Case 600 fractions sum to_air + to_mass = 0.60 vs requirement 1.0, Δ −0.40** (the missing 0.40 fraction is the structural gap); (d) Case 900 `solar_beam_to_mass_fraction = 0.30` vs 1.0, Δ −0.70 (same axis on the HighMass construction); Case 900 `solar_distribution_to_air = 0.00` PASSES. The **per-surface Case 600 5R1C conductances vs hand-calc** (printed by `tests/all_tests/test_case_600_htotal_verification.rs::test_case_600_htotal_hand_verification`) are out of band on the dominant axes: h_tr_is model 165.60 vs hand 1251.32, **Δ −86.8 %**; h_tr_ms model 240.00 vs hand 1092.00, **Δ −78.0 %**; Cm model 2,162,443 vs hand 3,028,278 J/K, **Δ −28.6 %**; H_total 5R1C 85.83 vs simple Σ U·A 103.63 W/K, ratio 0.828 (−17.2 % under); the cumulative effect cascades into the production-validator Case 600 detailed-results row (3/4 metrics fail: Annual Cooling 3299.30 kWh vs ref [3920, 6140] = −16 % UNDER, Peak Heating 4.38 kW vs [2.80, 3.80] = +15.3 % OVER, Peak Cooling 3.72 kW vs [4.80, 6.20] = −22.5 % UNDER) and the strict-energy-gate `case_600_cooling` 2.546 MWh vs [4.275, 5.784] = **−34.38 % UNDER** (KNOWN-FAIL tracked under this §LIMIT-30 cohort). The **three competing mechanism hypotheses** for the Case 600 series per-tilt / per-azimuth deviation signature (the B1a audit does not pick a winner — per AGENTS.md / RULES.md / ADR-0001): (1) **`solar_distribution_to_air` / `solar_beam_to_mass_fraction` parameter routing hypothesis (most likely)** — the 0.30 / 0.30 per-surface routing split is the direct cause of the Case 600 cooling −34.4 % UNDER strict-energy-gate signature; 0.30 of the per-timestep beam gain that should go to the mass node (per ASHRAE 140 expectation `solar_beam_to_mass_fraction = 1.0`) is being routed to the air node instead, where the HVAC controller reads the cooling setpoint signal and trips the cooling plant on an over-estimated load; the structural fix is **path re-routing** (split the parameter between LowMass and HighMass constructions or re-derive the per-tilt / per-azimuth fraction from the energy-balance identity `f_beam_to_mass + f_beam_to_air = 1.0`), **not** a numerical tuning of the 0.30 / 0.30 values themselves — per AGENTS.md / RULES.md / ADR-0001, raising `solar_distribution_to_air` to absorb the structural cooling gap is forbidden; (2) **per-surface 5R1C h_tr_is / h_tr_ms conductance hypothesis** — the hand-calc / model per-surface conductance deltas (−86.8 % on h_tr_is and −78.0 % on h_tr_ms) are the structural reason the H_total 5R1C path is 0.828× the simple Σ U·A reference (−17.2 %); the structural fix is **5R1C parameter re-derivation** (e.g. via ISO 13790 §12.2.3 + Annex C, the same convention used by the §LIMIT-29 B2a Cm derivation), not numerical tuning; (3) **Case 600 series low-mass + 5R1C lumped-mass-node hypothesis (the §LIMIT-16 / §LIMIT-05 cousin)** — the Case 600 series shares the same 5R1C + 9R4C single-lumped-mass-node pathology as Cases 610 / 630 / 650 (§LIMIT-16 / Issue #3059) and the 900-series (B2a / §LIMIT-29); the `MAX_CONVECTIVE_TO_AIR_MULTIPLIER = 2.0×` cap (PR #3041) closed Cases 620 / 640 but did NOT transfer to Case 600 — because Case 600's per-tilt / per-azimuth solar-distribution deviation is upstream of the `MAX_CONVECTIVE_TO_AIR_MULTIPLIER` axis (the cap is on the convective path, not on the solar-distribution routing); the structural fix is the GaugeSolver production-path work coordinated by #1465 / #1462, not per-Case 600 tuning. The three mechanisms are **not mutually exclusive** — the B1b structural fix likely needs to address all three. **Module-isolation suites green (read-only acceptance)** per the B1a audit table: `tests/all_tests/solar_isolation::test_horizontal_incident_solar` PASS, `tests/all_tests/solar_isolation::test_per_tilt_sweep` PASS (5/5 tilts in [0.99, 1.01]), `tests/all_tests/surface_flux_parity::test_parity_combined_tilt_azimuth_matrix` PASS (16 × 8760 assertions), `tests/all_tests/solar_isolation::` 11 passed, `tests/all_tests/ashrae_140_case_600::test_case_600_baseline_ashrae_140_reference` PASS (integration prints the 3/4-failure the detailed-results row already shows — NOT introduced by this audit), `tests/all_tests/ashrae_140_validation::` 3 passed, `tests/regression_exterior_film_unification.rs` (LIMIT-13 regression fence) PASS — the 18.3 W/m²K canonical exterior film coefficient is **unchanged** through the B1a measurement window (the AGENTS.md "no regression on the canonical film coefficient" guard holds), `python3 scripts/check_strict_energy_gate_regression.py` 4 PASS / 12 KNOWN-FAIL / 0 REGRESSION (Issues #2506 / #3572 strict ±15 % gate holds; Case 600 cooling KNOWN-FAIL tracked under this §LIMIT-30 cohort). The **11 pre-existing solar test failures** (3 in `solar_distribution_validation::` — the per-surface distribution metrics the B1a audit characterizes; 5 in `solar_longwave_boundary_traces::`; 1 each in `solar_horizontal_isolation::test_roof_surface_irradiance_matches_energyplus`, `issue_1860_5r1c_time_constant_aware::test_case_650_solar_lag_improves_annual_cooling`, `solar_distribution_tests::test_conductance_mass_dependence`) are **NOT regressions** introduced by this audit — they are pre-existing structural failures consistent with the §LIMIT-30 mechanism hypotheses; per AGENTS.md / RULES.md / ADR-0001, no test assertion is loosened to absorb them. Tracked as a documentation-only entry; the structural fix is routed to the GaugeSolver production-path work coordinated by **#1465 / #1462** (production-path switchover staged via **#3291 / PR #3482** for Phase A8 default flip, gated on **§LIMIT-21** β-soak closure). Sibling entries: §LIMIT-13 (h_tr_em time-invariance regression fence — 18.3 W/m²K canonical exterior film coefficient unchanged, guarded by `tests/regression_exterior_film_unification.rs`); §LIMIT-05 / §LIMIT-05 UPDATE (#2453) (900-series bidirectional annual-energy over-prediction); §LIMIT-16 / Issue #3059 (Cases 610 / 630 / 650 peak cooling OVER — same Case 600-series family); §LIMIT-17 / Issue #3058 / ADR-0011 (Case 950FF night-vent mass coupling gap); §LIMIT-22 (gauge τ_mass measurement source for the B2a cousin); §LIMIT-21 (β-soak 30-night production-path gate); §LIMIT-29 / Issue #3799 (the B2a thermal-mass τ cousin — same `TimeConstantAnalyzer` / ISO 13790 §12.2.3 + Annex C family); **#3072** (the aggressive-baseline cohort tracking — Cases 195 / 600 / 620 / 940 / 960); SOLAR-01 / Issue #274 (the pre-existing SOLAR-01 entry that documents the 600-series peak-cooling signature; the B1a measurements identify the per-tilt / per-azimuth distribution axis as the structural mechanism behind the SOLAR-01 partial-resolution status). **Cross-PHYSICS-01**: B1b / Issue #3798 / PR #3847 (the release-gates registration that cites this B1a audit as the precursor measurement — `release_gates.yaml` lines 83–110 already name B1a as the precursor); **cross-PHYSICS-01 (companion fixtures)**: Issue #1323 (corrected roof-solar constants; the `test_parity_roof_zero_followup_1323` `#[ignore]` stays in place until #1323 closes — this B1a audit does **not** un-ignore it), #1325 / #1330 / #1337 (per-tilt / per-azimuth fixture-data lineage grounding the calculation PASSES — all upstream closed issues cited without modification). See §LIMIT-30 below for the per-axis measurement tables, the three mechanism hypotheses, the cross-references to B1b / Issue #3798 / PR #3847, the module-isolation suites (read-only acceptance — all green), the pre-existing solar test failures (NOT regressions), and the sibling framing to §LIMIT-05 / §LIMIT-13 / §LIMIT-16 / §LIMIT-17 / §LIMIT-21 / §LIMIT-22 / §LIMIT-29 / #3072 / SOLAR-01 / Issue #1323-#1337.

**LIMIT-12 added (Issue #3062):** Case 940 annual heating is 5,158 kWh on the CTF validator path versus 1,289.9 kWh on the blind diagnostic path (per the §LIMIT-05 UPDATE #2452 measurement table, post-PR #3042); the remaining setback-recovery overshoot is structural and tracked without a production-physics change. (Historical: 7,487.81 kWh was the pre-§LIMIT-05-UPDATE snapshot; the latest measured value is 5.158 MWh. See §LIMIT-05 UPDATE (#2452) for the canonical per-path table.)

**LIMIT-15 added (Issue #3060):** Case 195 weather data source methodology — repo's Denver TMY3 annual min −12.47 °C vs ASHRAE 140-2023 DRYCOLD.TM2 annual min −24.4 °C; ~0.6 MWh annual-heating residual gap is a weather-file artefact (NOT a solver bug). Three implementation options (switch test weather file / widen reference band / re-derive reference band from EnergyPlus DRYCOLD.TM2 runs) are documented with risk / cost / benefit analysis; per AGENTS.md / RULES.md / ADR-0001 ("no parameter tuning", "must-never hardcode results"), the decision is routed back to Issue #3060 for maintainer action. No physics-code change; no solver-code change; no reference-band change. Companion deliverables: `docs/investigations/issue-3060-case-195-weather-source.md` (standalone investigation) and `tests/diagnostics/case_195_weather_source_diagnostic.rs` (`#[ignore]`-quarantined per #2536, on-demand weather-source comparison runner).

(LIMIT-08 + LIMIT-09 retained. **LIMIT-10 added (Issue #3065):** Case 960 sunspace inter-zone + full_validation tests (`tests/ashrae_140_case_960_sunspace.rs`) re-asserted against post-#1456 ground truth — sunspace annual mean is ≈ 0 °C under the default 5R1C/9R4C path (was ≈ 15 °C under the pre-#1456 6R2C override that #1456 removed). The failing assertion (`sunspace_mean > back_mean - 15.0`) was calibrated to the pre-#1456 6R2C solver and is no longer reachable under current energy balance. Replacement assertion is a documented physical band `sunspace_mean ∈ (-10, 50) °C` that holds both for the post-#1456 ground truth and once the GaugeSolver structural fix lands. Unblocker is Issue #3059 (GaugeSolver #1465/#1462); per AGENTS.md / RULES.md / ADR-0001, parameter tuning to force the prior 15 °C value is explicitly out of scope. **MULTI-03 added (Issue #3066):** documented the ~88.7 W residual in `test_two_zone_balanced_stub_passes` as a structural artefact of the 9R4C `InvariantChecker` BE-implicit identity evaluated against a hand-balanced stub (T_air = T_mass = T_outdoor with φ_st = 0 → T_s < T_air when h_tr_me > 0); resolved test-only by removing the over-strict `InvariantChecker` assertion and keeping only the `EnergyBalanceValidator` check (which IS zero by the integrated-flux form on the balanced stub and is the Issue #1344 product surface). 23-line test-only change in `tests/cli_multi_zone_energy_conservation.rs`; no solver code modified. **LIMIT-11 added (Issue #3064):** Case 195 high-mass walls (`tests/ashrae_140_solid_conduction_variants.rs::test_case_195_high_mass_walls`) is `#[ignore]`-quarantined with the same template as LIMIT-09 / LIMIT-10; pre-existing zero-energy assertion failure (`high_mass_energy.abs() > 0.0` fails because high-mass returns `0.00 kWh` while baseline Case 195 returns `-18.21 kWh`) tracked through #2868 → #3044 → #3059 with the long-term structural fix routed to GaugeSolver #1465/#1462. No physics-code change; per AGENTS.md / RULES.md / ADR-0001, parameter tuning to force non-zero energy is explicitly out of scope. LIMIT-09 (Issue #3071), LIMIT-10 (Issue #3065), MULTI-03 (Issue #3066), the §"Aggressive-baseline cohort tracking (Issue #3072)" section, and **LIMIT-13 added (Issue #3063)** retained unchanged.)

> **Post-#1323 baseline changes (read first)** — Between the prior "Last Updated" header
> (2026-03-30) and this revision, ~100 days and 30+ validation-affecting PRs landed.
> Per **ARCHITECTURE.md §Current Module Status** ("Anything pre-#1323 numbers is
> obsolete"), every numeric claim in the rows below has been regenerated against the
> post-#1323 surrogate v3.1 + strict ±15% CI gate (#1367, #1368) + Case 900 peak
> cooling verification (#1362, #1328). The 2026-03-30 numbers — most prominently the
> "peak cooling 40–80 % under-predicted" claim in §SOLAR-01 and the "0.86 kW vs
> 2.10–3.50 kW" row in §LIMIT-05 — pre-date #1323 and are superseded. The latest
> per-case engine output lives in `docs/ASHRAE140_RESULTS.md` (Phase 7B snapshot,
> 18.8 % pass rate) and the multi-zone Case 960/970 numbers live in
> `docs/ASHRAE140_MULTI_ZONE_RESULTS.md` (post-#1407 / #1446 / #1456). When this
> document and those result docs disagree, the **post-#1407 `validate_case_960`
> real-physics validator output** is authoritative for Case 960/970 and the
> Phase 7B snapshot in `ASHRAE140_RESULTS.md` is authoritative for Cases 600/900.

This document catalogs all known systematic issues affecting ASHRAE 140 validation
compliance. Issues are categorized by domain and include severity, affected
cases/metrics, GitHub issue links, and resolution status.

#### LIMIT-21: Gauge β-path pre-existing air-trajectory failure cohort (Case 600 / 900FF / 600-series / 960 / 950FF) — β-soak blockers / production-path gate (Issue #3297 + Phase A8 default flip — Issue #3291, PR #3482)

- **Description:** With `--features gauge-solver`, a fixed set of tests
  fails on the gauge solver's **air trajectory** across five test
  binaries. The set was verified **identical** at `fd7ef13^`
  (`832b0fe`) and at HEAD (`0b54606`) on 2026-09-03 — i.e. entirely
  pre-existing, NOT caused by the #3297 mass-state proxy (`fd7ef13`);
  the strict-energy-balance residuals the proxy targets are exactly 0
  (Case 600: 168 violations / max 2.62e4 W → 0 / 0.0 W; Case 900:
  165 / 2.16e5 W → 0 / 0.0 W; Case 960 gauge multi-zone: 0 / 0.0 W):
  - `tests/zone_balance_eplus_isolation.rs` — 2 of 21:
    `test_physics_thermal_model_eplus_case_600_reference_csv` (gauge
    single-zone trajectory vs the EnergyPlus reference CSV: T_zone mean
    −12.59 °C over the 168-hour NREL window, |mean − 20| = 32.594 °C,
    max step jump 34.007 °C) and `test_free_floating_case_900ff_isolation`
    (free-float divergence to non-finite min/max).
  - `tests/ashrae_140_case_600_series.rs` — 21 of 27 (nightly
    criterion 4).
   - `tests/known_issues_regression.rs::issue_1457_case_600_series_tracking`
     (nightly criterion 5) — 13 Case 600-series metrics out-of-band
     pending GaugeSolver #1465 (e.g. `Case650/annual_cooling=85.57MWh`,
     `Case600FF/min_free_float=-42.91C`, `Case650FF/min_free_float=-42.91C`,
     `Case610/peak_heating=3.20kW`; re-verified 2026-09-19 on post-#3890
     develop `021df06`, feature build).
  - `tests/ashrae_140_case_960_sunspace.rs` — 3:
    `test_case_960_comprehensive_energy_validation`,
    `test_case_960_full_validation`,
    `test_case_960_validator_no_longer_6r2c_override_issue_1456`.
  - `tests/ashrae_140_blind_validation.rs` — 1:
    `test_case_950_night_flush_zone_cooling_in_july`.
  These are the blocking residuals for the Issue #3286 β-soak streak —
  nightly criteria 2 (zone_balance), 4 (case_600_series), and 5
  (issue_1457 tracking) stay red; criteria 1 (`ashrae_140_validation`
  3/0), 3 (`integration-cli` 20/0), and 6 (`gauge_validation_case_900`
  9/0 + 1 ignored) are green — and therefore gate the Phase A8
  default-flip (#3291, merged via PR #3482).
- **Production-path gate (Phase A8 / Issue #3291).** The Phase A8 PR
  (#3482) declared `ZoneSolverKind::Gauge` the unconditional default
  of `ThermalSelector::default()` and made the dispatcher's
  `step_physics` (`src/sim/thermal_model_physics/step_dispatcher.rs`)
  panic on a missing gauge backend rather than fall through to legacy
  5R1C/9R4C. The `gauge-solver` cargo feature is intentionally retained
  as the production-path gate pending closure of this LIMIT — **the
  default build (feature OFF) routes the `Gauge` selector to 5R1C/9R4C
  via the `match` arm at the bottom of `step_physics`**, so the
  β-soak-blocking residuals above are not on the production path until
  the β-soak streak trips and the feature is enabled. β-soak gate is
  currently at **0/30 nights green**; #3291's acceptance criterion
  ("34/30") will trip on the 34th straight green nightly criterion set.
  See `AGENTS.md` §Read Before Changing Boundaries — Phase A8 note,
  and `ARCHITECTURE.md` Module 5.
- **Affected Tests:** the set above. Deliberately **NOT**
  `#[ignore]`-quarantined: they are the β-soak gate signal, and
  quarantining them would let the nightly soak go green with the
  underlying physics gap hidden (per AGENTS.md, known failing
  validation gates are structural gaps to fix, not to silence).
- **Affected Metrics:** gauge build only (`--features gauge-solver`).
  Default build passes the zone_balance binary 19/0/2 (2026-09-03
  verification).
- **Severity:** High for the β-soak program (blocks #3286 / #3291);
  zero production impact while the default build ships the legacy
  5R1C/9R4C path.
- **Root-cause direction:** per AGENTS.md, diagnose bottom-up
  (Weather → Solar → Conduction → Ventilation → Zone Balance). The
  gauge integration (quasi-steady per-surface fluxes + implicit-Euler
  air node with per-surface BDF mass states) diverges from the E+
  trajectory on low-mass conditioned (600), free-float high-mass
  (900FF / 600FF), and conditioned series (610–650, 960, 950)
  configurations — the air-trajectory fidelity program of
  #1465 / #1462 / #3059. NOT proxy-addressable: the mass-state proxy
  is telemetry-only and never feeds back into the gauge integration
  (no `.mass` reads in the gauge step path; grep-verified in the #3297
  prior-agent report, and the failure set is unchanged by `fd7ef13`).
- **GitHub Issue:** #3297 (the "zone_balance 21/21" acceptance
  criterion stays open until this entry closes), #3286 (β-soak),
  #3291 (Phase A8 default flip — merged via PR #3482), #1457
  (600-series tracking), #1465 / #1462 (architectural unblocker).
- **Status:** 🔄 **Known structural gaps; routed to the GaugeSolver
  air-trajectory program (#3291 / #1465 / #1462).** Phase A8 (#3291
  / PR #3482) wires gauge as the unconditional default, gated on the
  `gauge-solver` feature and the β-soak program — *not* on a
  per-case tolerance change. No constant, baseline, or threshold
  change is permitted to absorb these failures (AGENTS.md / RULES.md
  / ADR-0001).
- **Beyond-envelope zone-count policy (Issue #3731):** When the
  `gauge-solver` feature flips unconditionally after this β-soak
  closes, a spec routed through `ZoneSolverKind::Gauge` (the
  unconditional `ThermalSelector::default()` per the Phase A8 note
  above) with `spec.num_zones > MAX_ZONES = 100` (the Phase-1a
  gauge envelope documented in `ARCHITECTURE.md` Module 6 / point 7)
  would otherwise reach the deep `assert!` in the shadow-mode
  `ThermalManifold::new(num_zones)` (`src/physics/gauge_solver.rs:28`)
  and panic instead of surfacing the typed `PhysicsError::initialization`
  every other gauge pre-check uses. The canonical typed seam is
  `physics::geometry_tensor::ZoneCountPolicy`
  (`src/physics/geometry_tensor.rs`), wired into both gauge
  initialization entry points — `enable_gauge_solver` and
  `enable_gauge_solver_multi_zone` (`src/sim/thermal_model_core/mod.rs`)
  — **outside** the `#[cfg(feature = "gauge-solver")]` gate so the
  rejection surface is identical in both feature states. Beyond-
  envelope counts surface as `Empty` (num_zones == 0) / `BeyondEnvelope`
  (num_zones > 100) tiers with a typed diagnostic naming the offending
  count, the gauge envelope constant, and the Phase-1b geometry rework
  that owns growing it; the boundary tests
  (`enable_gauge_solver_multi_zone_rejects_beyond_envelope_*` and
  `*_accepts_at_capacity` in `src/sim/thermal_model_core/tests.rs`,
  `test_zone_count_policy_*` in `src/physics/geometry_tensor.rs::tests`)
  pin the typed behavior for `= 100`, `= 101`, and `= 200` across
  both feature states. The deep `ThermalManifold::new` assert
  remains as the secondary, deep-defense check (intentionally NOT
  deleted — the wrapper is the primary, the assert is the safety
  net). Future policy extensions (explicit `NineRFourC` fallback
  for beyond-envelope, partitioned gauge solve across two
  `ThermalManifold`s, etc.) MUST hook into `ZoneCountPolicy::for_count`
  + tier switch — bypassing the wrapper is forbidden while this
   entry remains open.

- **UPDATE (Issue #3878, PR #3884, 2026-09-18):** The `test_free_floating_case_900ff_isolation`
  divergence (non-finite min/max) and the Case 600FF/650FF max-free-float ceiling at
  ~39.6°C are fixed. The root cause was `GaugeZoneSolver::step()` using `h_eff * T_ext`
  as a proxy for all surface heat flows in the T_air update formula — solar-driven heat
  flows were invisible to it. `net_power_watts` already captured all surface heat flows
  including solar; it is now used directly in the T_air update (numerator) with `h_total`
  (infiltration-only conductance) as the denominator coefficient. Result: Case 600FF max
  free-float moved from ~39.6°C to ~52.8°C (+13°C); Case 650FF from ~39.6°C to ~51.1°C
  (+11°C); Case 900FF from non-finite divergence to ~39.4°C. **Single-zone path is now
  green on all gauge-soak criteria.** Multi-zone path (`MultiZoneGaugeSolver::step_with_coupling`)
  remains out of scope and is a separate follow-up. MAE improved from 51.77% to 49.82%,
  passing the 50% gate for the first time. See PR #3884.

- **UPDATE (Issue #3890, 2026-09-19):** The multi-zone follow-up landed:
  `MultiZoneGaugeSolver::step_with_coupling` now uses the solar-aware
  surface-flux sum in its T_air update (mirroring the #3878 single-zone
  fix — the ~39.6°C multi-zone free-float ceiling is gone), and the
  single-zone `step` T_air update no longer double-counts
  `Q_internal_w` (the #3878 formula carried it in both
  `net_power_watts` and the explicit numerator term, settling
  free-floating zones at `T_ext + 2Q/(H+h_inf)` instead of
  `T_ext + Q/(H+h_inf)`). **Free-float table re-measured post-merge on
  develop `021df06`, feature build** (`cargo test --features
  gauge-solver --test all_tests ashrae_140_free_floating`): Case 600FF
  max 52.78 → **40.64°C**, Case 650FF max 51.07 → **40.64°C** — the
  #3878-era figures above were measured *with* the double-count; Case
  900FF max **39.36°C** (min −1.09°C) and Case 950FF max **35.34°C**
  (min −18.65°C) unchanged; low-mass winter minima **−42.91°C** on both
  600FF and 650FF. The stale `Case600FF/min_free_float=-36.91C`
  citation earlier in this entry is corrected to **−42.91°C**: the drift
  came from the T1-era fixes, NOT from #3890 — all #1457 tracking
  values verified byte-identical pre/post (2026-09-19, feature build),
  incl. `Case650/annual_cooling=85.57MWh`. Feature-build cohort unmoved
  by #3890: Case 960 16✓/2 quarantine-ignored,
  `zone_balance_eplus_isolation` 19✓/8 ignored, `issue_1457` 3✓/1
  expected LIMIT-21 fail (the β-soak gate signal). See PR #3890;
     docs-only refresh tracked in #3891.

 - **UPDATE (Issue #3297 Phase 5, commit `91c7238`, 2026-09-21):**
   `h_rad_sky = 0.0` in gauge boundary-condition construction has been
   replaced with proper per-surface sky radiative conductance via
   `air_sky_conductance()`. The implementation adds
   `t_sky` / `h_rad_sky` fields to `ZoneBoundaryConditions`,
   a `sky_view_factor()` helper (ISO 13790 §12.3.2 from tilt angle),
   and an `h_rad_sky_for_gauge()` helper that converts the W/K total
   from `air_sky_conductance` to W/m²K for the gauge formula's
   `h_rad_sky / h_exterior` dimensionless ratio. Sky temperature is
   fetched from weather (fallback outdoor − 15 K). Case 640 annual
   cooling improved from **0.000 MWh** (completely broken gauge path)
   to **1.314 MWh** — partial functionality restored.

   **This Phase 5 does NOT close the remaining gauge-solver accuracy
   gap.** With `--features gauge-solver`, Case 640 annual cooling
   measures **1.314 MWh** vs the ASHRAE 140 reference band
   **5.95–8.10 MWh** — ~75 % below the lower bound. The root cause
   is **NOT** in the per-surface sky radiation / per-tilt /
   per-azimuth arithmetic (confirmed passing per LIMIT-30 / Issue #3797
   B1a audit: all 5 tilts at az=180° in the [0.99, 1.01] annual
   ratio band; 0 of 3573 hours-with-sun exceed 1 % per-hour tolerance).
   The failing axis is **downstream in the per-surface distribution
   routing parameters and the 5R1C coupling** — owned separately
   by the GaugeSolver structural-work program (#1465 / #1462) and
   tracked in §LIMIT-30 / Issue #3797 B1b follow-up (PR #3847).
   Phase 5 is therefore complete: the sky radiative implementation
   is correct and the remaining gap is a separate structural issue.
