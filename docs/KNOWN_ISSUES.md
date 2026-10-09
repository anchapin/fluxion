# Known Systematic Issues - ASHRAE 140 Validation

## Summary

| Category | Total Issues | Fixed | Open | Partial | Won't Fix |
|----------|-------------:|------:|-----:|--------:|----------:|
| Foundation (BASE) | 5 | 0 | 5 | 0 | 0 |
| Solar (SOLAR) | 4 | 0 | 4 | 0 | 0 |
| Free-Float (FREE) | 3 | 0 | 3 | 0 | 0 |
| Temperature (TEMP) | 1 | 0 | 1 | 0 | 0 |
| Multi-Zone (MULTI) | 4 | 0 | 4 | 0 | 0 |
| Model Limits (LIMIT) | 36 | 5 | 29 | 2 | 0 |
| Reporting (REPORT) | 4 | 0 | 4 | 0 | 0 |
| CI/Infrastructure (CI) | 4 | 2 | 2 | 0 | 0 |
| fluxion-fluid (FLUID) | 2 | 0 | 2 | 0 | 0 |
| FFD/CFD (FFD) | 2 | 1 | 1 | 0 | 0 |
| Reference data (REF) | 1 | 0 | 1 | 0 | 0 |
| **Total** | **66** | **8** | **56** | **2** | **0** |

*Counts derived from the per-row catalog tables under each category section (`| **CATEGORY-NN** | ... |`) via `scripts/check_known_issues_summary.py`. Edit a row in place (or add a new row) and the table updates on the next regen. Status columns (`Fixed` / `Open` / `Partial` / `Won't Fix`) derive from each row's Status cell: `resolved` -> Fixed, `open` -> Open, `tracking only` -> Partial, `Won't Fix` -> Won't Fix. Rows without a recognized status are counted in the Total column but contribute 0 to the status columns. To regenerate: `python3 scripts/check_known_issues_summary.py --regen --in-place`.*

*Last Updated: 2026-10-09*
## How to read this document

Each limitation is **one row** in the tables below: its current measured value against
the reference band, the issue that owns it, and what would close it. The row is the
current state. It is not a log.

The narrative history of each entry — every dated UPDATE, every superseded measurement,
every mechanism hypothesis — lives in `docs/investigations/<token>-*.md`, linked from the
row. Nothing was deleted when this document was restructured under Issue #4278; 4,883
lines of history moved into 59 files, verbatim.

**Before you change physics or validation code**: read the row, then read its
investigation file. The row tells you where things stand; the file tells you what has
already been tried and refuted.

**When you learn something new**: update the row *in place* so it still states the
current value, and append the detail to the investigation file. Do not append a new
dated block to this document. That append-log habit is what Issue #4278 was filed to
end: §LIMIT-17 carried a regression-avoidance clause requiring Case 950 HVAC annual
cooling to stay in the 390–920 kWh band while §LIMIT-24 recorded that it already
measured 33.08 kWh, roughly 14× outside it. The document asserted a precondition its own
later entry disproved. `scripts/check_known_issues_links.py` now fails on that pattern.

**Quarantine registry**: all `#[ignore]`-quarantined tests are catalogued in
`tests/QUARANTINE.md` with their blocking issues and un-ignore criteria (Issue #3211).
That registry, not this document, is canonical for quarantine state.

> **Post-#1323 baseline (read first)** — per **ARCHITECTURE.md §Current Module Status**,
> any pre-#1323 number is obsolete. The latest per-case engine output lives in
> `docs/ASHRAE140_RESULTS.md`, and the multi-zone Case 960/970 numbers in
> `docs/ASHRAE140_MULTI_ZONE_RESULTS.md`. Where this document and those disagree, the
> post-#1407 `validate_case_960` validator output is authoritative for Case 960/970 and
> the `ASHRAE140_RESULTS.md` snapshot for Cases 600/900.

## Foundation (BASE)

| ID | Symptom | Current value vs band | Issue | Closes when | Status | History |
|---|---|---|---|---|---|---|
| **BASE-01** | Incorrect Window U-Value Application to h_tr_em | — | — | — | open | [history](investigations/base-01-incorrect-window-u-value-application-to.md) |
| **BASE-02** | HVAC Load Calculation Using Ti Instead of Ti_free | — | — | — | open | [history](investigations/base-02-hvac-load-calculation-using-ti-instead.md) |
| **BASE-03** | Thermal Mass Capacitance Incorrect | — | #2706 | — | open | [history](investigations/base-03-thermal-mass-capacitance-incorrect.md) |
| **BASE-04** | Denver TMY Weather Data Confirmation | — | #2429 | — | open | [history](investigations/base-04-denver-tmy-weather-data-confirmation.md) |
| **BASE-05** | Incorrect h_tr_em Heat Transfer Coefficient | — | — | — | open | [history](investigations/base-05-incorrect-htrem-heat-transfer-coefficient.md) |

## Solar (SOLAR)

| ID | Symptom | Current value vs band | Issue | Closes when | Status | History |
|---|---|---|---|---|---|---|
| **SOLAR-01** | Peak Cooling Load Under-Prediction | — | — | — | open | [history](investigations/solar-01-peak-cooling-load-under-prediction.md) |
| **SOLAR-02** | Annual Cooling Energy Under-Prediction (High-Mass) | validator Case 900 annual cooling 1,792.02 kWh vs [2,130, 3,670] (−15.9% under the lower bound; was 552.83 before 2026-10-09). UPDATE 2026-10-09, E+ Perez coefficient-table fix (doc §13): the under-prediction was largely the Perez Fij coefficient mismatch — all high-mass cooling numbers move strongly toward band (900 C 552.83 → 1,792.02; Case 950 C re-enters band on both validator and strict-gate bases; Case 960 C 1.776 MWh re-enters band; 970 C 1.937 → 2.277 MWh); residual tracked in the strict-gate baseline | #2239 | residual GaugeSolver air-mass distribution | open | [history](investigations/solar-02-annual-cooling-energy-under-prediction-high.md) |
| **SOLAR-03** | Solar Shading Cases Not Sensitive to Shading Changes | — | — | — | open | [history](investigations/solar-03-solar-shading-cases-not-sensitive-to.md) |
| **SOLAR-04** | Night Ventilation Cooling Ineffective | — | — | — | open | [history](investigations/solar-04-night-ventilation-cooling-ineffective.md) |

## Free-floating temperature (FREE)

| ID | Symptom | Current value vs band | Issue | Closes when | Status | History |
|---|---|---|---|---|---|---|
| **FREE-01** | Maximum Free-Floating Temperature Under-Prediction (Low-Mass) | — | — | — | open | [history](investigations/free-01-maximum-free-floating-temperature-under-prediction.md) |
| **FREE-02** | Minimum Free-Floating Temperature Over-Prediction (High-Mass) | — | — | — | open | [history](investigations/free-02-minimum-free-floating-temperature-over-prediction.md) |
| **FREE-03** | Free-Floating Temperature Swings Reduced Compared to Reference | — | #2339 | — | open | [history](investigations/free-03-free-floating-temperature-swings-reduced-compared.md) |

## Temperature (TEMP)

| ID | Symptom | Current value vs band | Issue | Closes when | Status | History |
|---|---|---|---|---|---|---|
| **TEMP-01** | Thermal Lag Timing Incorrect | — | — | — | open | [history](investigations/temp-01-thermal-lag-timing-incorrect.md) |

## Multi-zone (MULTI)

| ID | Symptom | Current value vs band | Issue | Closes when | Status | History |
|---|---|---|---|---|---|---|
| **MULTI-01** | Case 960 Peak Heating Anomaly (100 kW) | — | #1407 | — | open | [history](investigations/multi-01-case-960-peak-heating-anomaly-100.md) |
| **MULTI-02** | Validation Energy Accounting Missing COP Conversion | — | — | — | open | [history](investigations/multi-02-validation-energy-accounting-missing-cop-conversion.md) |
| **MULTI-03** | `InvariantChecker` Pre-Step Hand-Balanced Stub Residual | ≈88.7 W residual on the hand-balanced two-zone stub | #3066 | resolved test-side; EnergyBalanceValidator is the surface | open | [history](investigations/multi-03-invariantchecker-pre-step-hand-balanced-stub.md) |
| **PeakHeatingLimit-01** | Case 960 Peak Heating < 2 kW (5R1C architectural) | — | — | — | open | [history](investigations/peakheatinglimit-01-case-960-peak-heating-2-kw.md) |

## 5R1C / structural limitations (LIMIT)

| ID | Symptom | Current value vs band | Issue | Closes when | Status | History |
|---|---|---|---|---|---|---|
| **LIMIT-01** | High-Mass Annual Energy Discrepancy | — | — | — | open | [history](investigations/limit-01-high-mass-annual-energy-discrepancy.md) |
| **LIMIT-02** | Free-Floating Temperature Range for Low-Mass | — | — | — | open | [history](investigations/limit-02-free-floating-temperature-range-for-low.md) |
| **LIMIT-03** | Hardcoded HVAC Capacity Masking Design Errors | — | — | — | open | [history](investigations/limit-03-hardcoded-hvac-capacity-masking-design-errors.md) |
| **LIMIT-04** | Case 960 Peak Heating Overprediction (Multi-Zone) | — | — | — | open | [history](investigations/limit-04-case-960-peak-heating-overprediction-multi.md) |
| **LIMIT-05** | High-Mass Peak Cooling — direction **inverted** since Phase 7B (see #1280) | Case 900 peak cooling 1.46 kW vs [1.20, 3.50] kW (validator). UPDATE 2026-10-09, E+ Perez coefficient-table fix (doc §13): peak cooling 0.89 → 1.46 kW — IN BAND on this row's band table; the narrower ASHRAE 140 published band [1.60, 2.10] is still slightly under. Direction-inverted symptom closed by the corrected reference-engine solar stack | #1280 | GaugeSolver multi-node air trajectory (#1465 / #1462) | resolved | [history](investigations/limit-05-high-mass-peak-cooling-direction-inverted.md) |
| **LIMIT-06** | 600-Series Annual Heating Correction (Empirical) | — | #522 | — | open | [history](investigations/limit-06-600-series-annual-heating-correction-empirical.md) |
| **LIMIT-07** | Default-schema `/v1/simulate` diverges at timestep 91 (un-initialized model physics) — RESOLVED | divergence at timestep 91 fixed by model initialisation | #2747 | RESOLVED | resolved | [history](investigations/limit-07-default-schema-v1-simulate-diverges-at.md) |
| **LIMIT-08** | ASHRAE 140 Case 195 (no-loads) peak heating below reference band on the repo's Denver TMY | Case 195 peak heating below band on repo Denver TMY | #2868 | weather-source decision in #3060 / LIMIT-15 | open | [history](investigations/limit-08-ashrae-140-case-195-no-loads.md) |
| **LIMIT-09** | Case 950 5R1C free-float night-vent override — pre-existing test failure | Case 950 5R1C free-float night-vent override fails | #3071 | GaugeSolver production path | open | [history](investigations/limit-09-case-950-5r1c-free-float-night.md) |
| **LIMIT-10** | Case 960 sunspace winter mean 0 °C vs pre-#1456 15 °C — assertion aligned with post-#1456 ground truth | Case 960 sunspace winter mean ≈ 0 °C (was ≈ 15 °C pre-#1456) | #3065 | band assertion (−10, 50) °C holds post-fix | open | [history](investigations/limit-10-case-960-sunspace-winter-mean-0.md) |
| **LIMIT-11** | Case 195 high-mass walls — pre-existing zero-energy assertion | Case 195 high-mass 0.00 kWh vs baseline −18.21 kWh | #3064 | non-zero high-mass energy without tuning | open | [history](investigations/limit-11-case-195-high-mass-walls-pre.md) |
| **LIMIT-12** | Case 940 annual heating CTF-validator vs blind-diagnostic path divergence — setback-recovery overshoot | Case 940 annual heating 5,158 kWh (CTF) vs 1,289.9 kWh (blind) | #3062 | the two paths agree | open | [history](investigations/limit-12-case-940-annual-heating-ctf-validator.md) |
| **LIMIT-13** | `h_tr_em` (envelope-to-mass conductance) remains time-invariant in 5R1C path — tracking stub | h_tr_em time-invariant in 5R1C; 18.3 W/m²K film coefficient fenced | #3063 | time-varying h_tr_em in the production path | tracking only | [history](investigations/limit-13-htrem-envelope-to-mass-conductance-remains.md) |
| **LIMIT-14** | Case 960 sunspace annual cooling and peak heating below band — GaugeSolver-blocked air-mass distribution gap | Case 960 annual cooling 1.776 MWh vs [1.55, 2.78]; peak heating 2.58 kW vs [2.0, 8.0] — both IN BAND. UPDATE 2026-10-09, E+ Perez coefficient-table fix (doc §13): annual cooling 0.63 → 1.776 MWh re-enters band; peak heating 1.17 → 2.58 kW re-enters band. RESOLVED by the corrected reference-engine solar stack | #3061 | GaugeSolver air-mass distribution | resolved | [history](investigations/limit-14-case-960-sunspace-annual-cooling-and.md) |
| **LIMIT-15** | ASHRAE 140 Case 195 — Denver TMY min −12.47 °C vs DRYCOLD.TM2 min −24.4 °C weather data source mismatch | Denver TMY3 min −12.47 °C vs DRYCOLD.TM2 −24.4 °C; ≈0.6 MWh residual | #3060 | maintainer decision on #3060 (3 options documented) | open | [history](investigations/limit-15-ashrae-140-case-195-denver-tmy.md) |
| **LIMIT-16** | Cases 610/630/650 peak cooling OVER — 5R1C + 9R4C air-mass distribution structural gap | Cases 610/630/650 peak cooling +48 %, +39 %, +92 % OVER | #3059 | GaugeSolver multi-node (#1465 / #1462) | open | [history](investigations/limit-16-cases-610-630-650-peak-cooling.md) |
| **LIMIT-17** | Case 950FF night-vent mass coupling overwhelms F_sky correction — tracking stub | Case 950FF min free-float −21.62 °C vs [−20.20, −17.80] °C (max free-float 37.84 °C in [35.50, 38.50]). UPDATE 2026-10-09, E+ Perez coefficient-table fix: min −23.92 → −21.62 °C — closer, still below | #3058 | one solver change closes this **and** LIMIT-24 | tracking only | [history](investigations/limit-17-case-950ff-night-vent-mass-coupling.md) |
| **LIMIT-18** | Case 960 Blind heating_max 2.45 MWh > 1.0 MWh (AC4) — pre-existing test failure | Case 960 blind heating_max 2.45 MWh vs AC4 ≤ 1.0 MWh | #3104 | GaugeSolver production path | open | [history](investigations/limit-18-case-960-blind-heatingmax-2-45.md) |
| **LIMIT-19** | `test_one_watt_artificial_gain_increases_imbalance` — InvariantChecker post-step algebraic-invariant confusion | 1 W artificial gain shrinks the residual instead of growing it | #3103 | EnergyBalanceValidator follow-up (#1344) | open | [history](investigations/limit-19-testonewattartificialgainincreasesimbalance-invariantchecker-post-step-algebraic-invariant.md) |
| **LIMIT-20** | `test_solid_conduction_variants_integration` — 75% pass-rate threshold, HighMass variant structural failure | solid-conduction variants 3/4 = 75.0 % (HighMass returns 0.00 kWh) | #3218 | HighMass variant non-zero; threshold never lowered | open | [history](investigations/limit-20-testsolidconductionvariantsintegration-75-pass-rate-threshold-highmass.md) |
| **LIMIT-21** | ADR-0017 — DAE teacher adopted; flip authority moves to #3986; two-solver limbo closed | — | #3986 | — | open | [history](investigations/limit-21-adr-0017-dae-teacher-adopted-flip.md) |
| **LIMIT-22** | Gauge-build-only test failures exposed by the exact Crank-Nicolson mass-state proxy | gauge-build-only failures under the Crank-Nicolson mass-state proxy | #3297 | gauge arm green under the proxy | open | [history](investigations/limit-22-gauge-build-only-test-failures-exposed.md) |
| **LIMIT-23** | Case 970 5-zone multi-zone cross-coupling annual heating + cooling OVER and peak heating/cooling UNDER band — GaugeSolver-blocked air-mass distribution gap | Case 970 4/4 metrics out: heating 18.58 MWh vs [10.54, 14.26], cooling 21.07 vs [7.39, 10.00] | #3552 | GaugeSolver air-mass distribution. UPDATE 2026-10-08, PR #4336 (issue #4332 9R4C mass/HVAC coupling, Alex-approved physics): on the strict ±15% blind path Case 970 heating moved FURTHER below its band 8.822 → 6.081 MWh (gap 13.85 → 35.96% of band midpoint) and cooling 1.465 → 1.109 MWh — the corrected mass coupling removes spurious over-requested load; recorded, not tuned. UPDATE 2026-10-08, PR #4347 (envelope-reroute phi_m): heating drops further below — 6.081 → 4.807 MWh (gap 35.96 → 46.23% of band midpoint), the one gap that WIDENS; cooling 1.109 → 1.937 MWh (below-band gap narrows). Recorded, not tuned. UPDATE 2026-10-09, E+ Perez coefficient-table fix (doc §13): heating 4.807 → 4.620 MWh (further below; gap 46.23 → 47.74% of band midpoint), cooling 1.937 → 2.277 MWh (below-band gap narrows again). Recorded, not tuned | open | [history](investigations/limit-23-case-970-5-zone-multi-zone.md) |
| **LIMIT-24** | Case 950 HVAC-mode annual cooling ~14× UNDER band — docs-only structural LIMIT entry, companion to §LIMIT-17 | Case 950 HVAC annual cooling 671.17 kWh vs [390, 920] kWh — IN BAND. UPDATE 2026-10-09, E+ Perez coefficient-table fix (doc §13): 33.08 → 671.17 kWh re-enters the published band (the same fix re-enters Case 950 strict-gate cooling, 0.436 → 0.564 MWh). RESOLVED | #3551 | one solver change closes this **and** LIMIT-17 | resolved | [history](investigations/limit-24-case-950-hvac-mode-annual-cooling.md) |
| **LIMIT-27** | 5R1C algebraic residual ~191 W — InvariantChecker evaluates gains at T_old while heat flows use T_new | 5R1C algebraic residual ≈191 W (threshold 200 W) | #3647 | InvariantChecker evaluates gains at T_new | open | [history](investigations/limit-27-5r1c-algebraic-residual-191-w-invariantchecker.md) |
| **LIMIT-28** | ASHRAE 140 Free-Floating cohort (600FF/650FF/900FF/950FF) — 5R1C + 9R4C + multi-node-night-vent structural gap, Phase B3b PHYSICS-03 | 600FF/650FF/900FF/950FF free-float cohort out of band | #3802 | GaugeSolver + multi-node night vent | open | [history](investigations/limit-28-ashrae-140-free-floating-cohort-600ff.md) |
| **LIMIT-29** | ASHRAE 140 Case 900 thermal-mass time-constant characterization (post-#3770) — Phase B2a PHYSICS-02 | Case 900 ISO 13790 τ = 3.30 h vs ≈5–8 h envelope (−40 %) | #3799 | #3770 mass-node fix + GaugeSolver | open | [history](investigations/limit-29-ashrae-140-case-900-thermal-mass.md) |
| **LIMIT-30** | ASHRAE 140 Case 600 series per-tilt / per-azimuth solar distribution audit — Phase B1a PHYSICS-01 | Case 600 cooling 2.546 MWh vs [4.275, 5.784] (−34.4 %); to_air 0.30 vs 0.0 | #3797 | per-surface solar routing re-derivation (B1b / #3798) | open | [history](investigations/limit-30-ashrae-140-case-600-series-per.md) |
| **LIMIT-31** | CTF↔zone coupling defect — conduction backends bypassed for conditioned high-mass cases | conditioned high-mass bypassed the CTF backend; Case 900 heating 5.05 → ≈1.165 MWh | #3979 | resolved in #3979; CTF stays the 5R1C cross-check | open | [history](investigations/limit-31-ctf-zone-coupling-defect-conduction-backends.md) |
| **LIMIT-32** | CTF steady-state flux for low-mass walls has the wrong sign | 80 mm EPS steady-state flux −5.53 W/m² (physical +3.69 W/m²) | #4062 | y-coefficient clamping fixed for τ ≪ dt walls | open | [history](investigations/limit-32-ctf-steady-state-flux-for-low.md) |
| **LIMIT-33** | #4241 ideal-HVAC discrete storage residual inflates Case 600 annual heating ~50% out of the published band — blocks 2 CI gates | FIXED in physics (merged as `9ef9c8f4`, PR #4316): Option A state feedback — the residual load now drives the 5R1C air-node state to the active setpoint on unclamped conditioned hours. NAPI/WD600 Case 600: heating 7181.51 → 5590.48 kWh (in [4314, 5836]), cooling 6026.39 → 4675.48 kWh (in [4275, 5784]). Case 900 unchanged (9R4C-insensitive). Recorded-value baselines re-recorded from the corrected engine (surrogate fallback, strict-energy, fabric parity, Case 600 January grid, npm cooling 4010.89 → 4675.48). Known side effects flagged for review: strict-energy blind-path 600 C 5.187 → 4.023 MWh (just below band, gap 5.01%) and 800 C 4.270 → 3.314 MWh | #4314 | merge of the state-feedback PR closes the NAPI + drift-gate symptom; exact-exponential hold metering (`Q_exact`) left as follow-up study (dt/τ ≈ 7.5 for Case 600 converges toward steady-state) | resolved | [history](investigations/limit-33-ideal-hvac-storage-residual-inflates-case-600.md) |
| **LIMIT-35** | Case 900 annual heating out of band (+70.6%) — sub-hourly conditioned-hour inner loop empirically ruled out: the production 9R4C path is outer-dt-invariant and the loop regresses Case 600 cooling out of band; solar-delivery path (§LIMIT-05-family roof-solar under-counting) is the open suspect | validator Case 900 heating 3,477.88 kWh vs [1,170, 2,040] (cooling 923.56); 9R4C path 3,143.8 (N=1) → 3,148.2 kWh (N=6) — no convergence; measured on DO-NOT-MERGE branch: Case 600 NAPI/WD600 under N=4 heating 5,590.48 → 4,450.67 kWh (in), cooling 4,675.48 → 9,777.69 kWh (out, +109%) | #4332 | Alex chose option (3) 2026-10-08: sub-hourly loop dropped (PR #4333 closed unmerged, develop keeps hourly stepping). Solar-delivery investigation COMPLETE (doc §8): roof-solar under-counting falsified — roof irradiance is OVER-counted +19% (Issue #1326 pinned horizontal ground-reflected to ρ·GHI; E+ reference records 0.0); only effective roof route (sol-air BC) has ~0.26–0.32 kWh heating per kWh/m²·yr lever, so no physical roof-solar correction can close the gap. Alex approved the physics 2026-10-08: the ground-reflection fix MERGED via PR #4335 (view factor continuous at the endpoints) and baselines re-recorded from the corrected engine — strict-gate Case 900 H 3.723 → 3.823 MWh (gap 116.95 → 123.18% of band midpoint), C 0.788 → 0.689; Case 600 H 6.000 → 6.072, C 4.023 → 3.877; fabric parity case_600 ratio_H 1.1218 → 1.1246, ratio_C 0.7057 → 0.6926; NAPI Case 600 H 5,590.48 → 5,654.86, C 4,675.48 → 4,509.99 kWh (both still in published band); hotloop golden EUIs, grid thermal and surrogate-drift baselines reproduce unchanged. Case 900 heating further out of band. Follow-up investigation (doc §9, 2026-10-08): the 9R4C mass network is decoupled from the conditioned zone (multi-node mass steps against free-float air; HVAC heat never charges it) — measured 3–9 K winter mass deficit and split-experiment coupling recovers 1,265 kWh/yr (Case 900 H 3,588.28 → 2,322.94 kWh validator); seasonal shape matches E+ within ~1.5 pp/month, so it is a magnitude defect. Alex approved the coupling fix 2026-10-08 ("I approve of the changes in 4336"): PR #4336 MERGED after full re-record from the corrected engine (branch rebased onto 968e629d). Variant recorded: previous-step committed t_act coupling, per-zone — in HVAC mode t_act equals the active setpoint, so the mass steps against the same setpoint-held air as a within-step coupling would; the within-step variant differs only in metering basis and is left as a documented refinement. Numbers moved: validator Case 900 H 3,588.28 → 2,322.94 kWh (band [1,170, 2,040], still out by ≈283 kWh ≈ 14% of midpoint), C 801.09 → 552.83; strict-gate Case 900 H 3.823 → 2.455 MWh (gap 123.18 → 37.94%), C 0.689 → 0.478; Case 920 heating RE-ENTERS band 5.252 → 3.302 MWh (now pass); Case 960 heating 6.426 → 4.055 MWh (still out); Case 950 C 0.198 → 0.173 MWh; Case 600/800 (5R1C path) unchanged. Side effects recorded honestly: Case 810 heating falls BELOW its band 3.823 → 2.455 MWh (strict-gate now known_fail) and Case 970 heating drops further below 8.822 → 6.081 MWh — both now tracked in their rows/baseline rather than hidden. Fabric harness case_900 H/C 3.823/0.689 → 2.455/0.478 MWh; surrogate fallback baseline re-recorded (4,594.61/9.01 → 3,160.02/95.02 kWh); hotloop golden EUIs, npm suite (54/54), grid thermal reproduce unchanged. Case 900 heating remains out of band (+13.8% over upper bound); residual gap documented here, not tuned. RESOLVED 2026-10-08 by PR #4347 (envelope-reroute phi_m delivery, doc §12; Alex verdict 2026-10-08 20:50 ET: "go with merge option"): the previously computed-but-dropped phi_m gains now reach the envelope mass nodes. Strict-gate Case 900 H 2.455 → 1.649 MWh — IN BAND [1.364, 1.846] (gap 37.94 → 0% of midpoint); validator 2,322.94 → 1,359.28 kWh (in [1,170, 2,040]). Baselines re-recorded from the corrected engine, old→new with dated provenance: fabric case_900 H/C 2.455/0.478 → 1.649/1.207 MWh and case_950 C 0.173 → 0.436 MWh (parity ratios unchanged at 1.0); strict-gate source_command corrected to the actual `--test all_tests` invocation; Case 600/800 and the whole 5R1C family bit-identical on all three paths (validator Case 600 6,238.59/4,498.34 kWh unchanged); hotloop golden EUIs, grid thermal, npm suite and surrogate fallback (3,160.02/95.02/3,255.04 kWh) reproduce unchanged. Case 900 cooling 0.478 → 1.207 MWh remains UNDER its band [2.465, 3.335] — tracked in the strict-gate baseline, not hidden. Downstream honest movements recorded in §LIMIT-36 (810), §LIMIT-37 (920 heating leaves band) and §LIMIT-23 (970). UPDATE 2026-10-09, E+ Perez coefficient-table fix (doc §13; the in-repo E+ 25.2 per-surface reference CSVs show the engine used the 1990-journal Perez Fij set where the reference engine uses the Perez 1999 private-communication set): strict-gate Case 900 H 1.649 → 1.575 MWh STAYS IN BAND [1.364, 1.846]; validator H 1,359.28 → 1,122.65 kWh — now 4.0% BELOW the lower bound 1,170, so the validator-basis metric leaves band again (recorded, not tuned; status re-opened); validator C 552.83 → 1,792.02 kWh (band [2,130, 3,670], far closer). Case 600 validator H 6,238.59 → 5,542.87 kWh RE-ENTERS its published band [4,360, 5,790] and C 4,498.34 → 5,240.11 kWh stays in [3,920, 6,140]. Strict-gate cooling family: Case 600 C 3.877 → 4.616 MWh RE-ENTERS BAND [4.275, 5.784]; Case 950 C 0.436 → 0.564 MWh RE-ENTERS [0.557, 0.753]; every remaining strict-gate cooling gap narrows; Case 810/920/970 heating move further below (tracked in their rows). Full re-record from the corrected engine with dated provenance in the strict-gate and fabric baselines | open |
| **LIMIT-36** | Case 810 annual heating fell BELOW its published band after the #4336 9R4C mass coupling — in band before (3.823 MWh), below after (2.455 MWh); honest movement, recorded not hidden | strict-gate Case 810 heating 3.823 → 2.455 MWh vs [3.357, 4.543] (−26.9% under the lower bound; #1147 band table). 810's CaseSpec is `case_900_baseline()` with only the HVAC equipment swapped, so its ideal-load heating metric equals Case 900's by construction (both 3.823 pre-coupling, both 2.455 after; earlier both 1.633) — the row tracks the same 9R4C correction as §LIMIT-35, not an independent meter | #4332 | a physics decision on the 9R4C coupling restores 810 inside [3.357, 4.543] without re-inflating Case 900 out of [1,170, 2,040]; §LIMIT-35 §10 (2026-10-08) falsified the §9.1 metering basis as the residual driver (≤5.11 kWh/yr exposure, zero clamped hours), so 810's drop is the coupling's genuine magnitude effect — remaining levers there: window-transmitted solar (+15.2% closes the 900 residual) and the inert `phi_m` gain route. UPDATE 2026-10-08, PR #4347 (envelope-reroute phi_m, resolves §LIMIT-35): 810 falls FURTHER below — 2.455 → 1.649 MWh (gap 22.84 → 43.24% of band midpoint), moving in lockstep with Case 900 as its spec implies (both 1.649 again); the phi_m gains that brought 900 in band deepen 810’s below-band gap. Recorded, not tuned. UPDATE 2026-10-09, E+ Perez coefficient-table fix: 1.649 → 1.575 MWh (gap 43.24 → 45.11% of band midpoint), moving in lockstep with Case 900 as its spec implies (both 1.575). Recorded, not tuned | open | [history](investigations/limit-36-case-810-heating-below-band.md) |
| **LIMIT-37** | Case 920 annual heating LEFT its published band after the envelope-reroute phi_m delivery (PR #4347) — it was in band (3.302 MWh) only because the dead phi_m path under-delivered solar to the mass; honest movement, recorded not hidden | strict-gate Case 920 heating 3.302 → 2.488 MWh vs [3.213, 4.347] (gap 19.18% of band midpoint, under the lower bound). Cooling 1.683 MWh remains below band [2.189, 2.961] with the gap narrowing; validator 2,389.91 kWh H (still below [3,260, 4,300]). History: 3.302 → 2.615 at PR #4347; UPDATE 2026-10-09, E+ Perez coefficient-table fix (doc §13): 2.615 → 2.488 MWh (gap 15.82 → 19.18% — the corrected solar stack deepens the below-band heating slightly while narrowing the cooling gap 1.395 → 1.683 MWh). Recorded, not tuned | #4332 | a physics decision restores Case 920 heating inside [3.213, 4.347] without pushing Case 900 heating out of [1.364, 1.846] | open | [history](investigations/limit-37-case-920-heating-leaves-band.md) |
| **LIMIT-38** | `solar_beam_to_mass_fraction` is INERT on the 9R4C per-surface path after PR #4347 — phi_st and phi_m are delivered through the same envelope mass nodes with identical orientation weights, and phi_st + phi_m is independent of the split, so the calibration value (0.6) feeds no output on this path | measured on develop 04a4e031 (issue #4339 sweep, Denver TMY3, Case 900FF free-float): fractions 0.2/0.4/0.6/0.8 all produce Max = 40.12 °C (bit-identical), in the widened bounds [36.00, 46.40] — the parameter's monotonic peak-dampening premise (live at c5b3231c: 36.13/34.13/32.50/31.13 °C) is structurally dead post-#4347. 5R1C path unaffected (st/mass nodes remain distinct there). No validation number moves: this changes what the sweep test asserts (fraction-invariance), not any band or output | #4339 | a physics design decision restores a live surface-vs-mass beam split on the 9R4C path (distinct delivery channel or weights for phi_st vs phi_m); routed alongside the GaugeSolver rework (#1465/#1462) | open | [history](investigations/limit-38-beam-to-mass-fraction-inert.md) |
| **LIMIT-39** | Case 630 annual cooling moved OVER its published band after the E+ Perez coefficient-table fix — in band before (on the 1990-journal Fij set), +9% over after; honest movement, recorded not hidden | Case 630 annual cooling 4.03 MWh vs [2.13, 3.70] (validator path `run_annual_simulation`); test `ashrae_140_case_600_series::case_630::test_annual_cooling` quarantined `#[ignore]` (see tests/QUARANTINE.md). Same fix moves 630 heating TOWARD band (6.95 → 6.65 MWh, still over [5.05, 6.47]) | #4332 | a physics decision brings Case 630 cooling inside [2.13, 3.70]; the residual E/W irradiance asymmetry (hour-convention thread, limit-35 doc §13) is the open suspect | open | [history](investigations/limit-39-case-630-cooling-over-band.md) |

## Reference data (REF)

| ID | Symptom | Current value vs band | Issue | Closes when | Status | History |
|---|---|---|---|---|---|---|
| **REF-01** | Blind-validation monthly reference data — recast as v1.3 documented-shape reference (issues #2677 → #2748) | — | #2677 | — | open | [history](investigations/ref-01-blind-validation-monthly-reference-data-recast.md) |

## Reporting (REPORT)

| ID | Symptom | Current value vs band | Issue | Closes when | Status | History |
|---|---|---|---|---|---|---|
| **REPORT-01** | Systematic Issues Classification Heuristic | — | — | — | open | [history](investigations/report-01-systematic-issues-classification-heuristic.md) |
| **REPORT-02** | Quality Metrics Not Automatically Tracked | — | — | — | open | [history](investigations/report-02-quality-metrics-not-automatically-tracked.md) |
| **REPORT-03** | Missing Issue Traceability to GitHub | — | — | — | open | [history](investigations/report-03-missing-issue-traceability-to-github.md) |
| **REPORT-04** | No "What's Fixed in This Phase" Section | — | — | — | open | [history](investigations/report-04-no-what-s-fixed-in-this.md) |

## CI

| ID | Symptom | Current value vs band | Issue | Closes when | Status | History |
|---|---|---|---|---|---|---|
| **CI-01** | Code coverage gate (issue #1932) — RESOLVED (min_branch_floor hard floors enforced) | min_branch_floor hard floors enforced | #1932 | RESOLVED | resolved | [history](investigations/ci-01-code-coverage-gate-issue-1932-resolved.md) |
| **CI-02** | Debug build linking crashes with rust-lld segfault (issue #2297) | — | #2297 | — | open | [history](investigations/ci-02-debug-build-linking-crashes-with-rust.md) |
| **CI-03** | `ort` pinned to a release candidate (issue #2691) — no stable 2.0 on crates.io | — | #2691 | — | open | [history](investigations/ci-03-ort-pinned-to-a-release-candidate.md) |
| **CI-04** | Node/NAPI Bindings (ubuntu-24.04 + windows-latest) and Surrogate Drift Tolerance Gate (#1784) red on develop — blocked on §LIMIT-33 | RESOLVED in physics by §LIMIT-33 fix (merged as `9ef9c8f4`, PR #4316): NAPI Case 600 heating 5590.48 kWh in [4314, 5836] (was 7181.51), cooling re-pointed 4010.89 → 4675.48 (in published band); surrogate drift baseline re-recorded 2665.03 → 4603.62 kWh (value reproduces identically on develop and the fix branch — Case 900 is 9R4C-insensitive; clears the pre-#4241 drift). npm suite 54/54 pass locally | #4314 | merged as `9ef9c8f4` (PR #4316) | resolved | [history](investigations/ci-04-napi-and-surrogate-drift-gates-red-limit-33.md) |

## fluxion-fluid autodiff (FLUID)

| ID | Symptom | Current value vs band | Issue | Closes when | Status | History |
|---|---|---|---|---|---|---|
| **FLUID-01** | Analytical Jacobian Saturation/Clamping Errors | — | — | — | open | [history](investigations/fluid-01-analytical-jacobian-saturation-clamping-errors.md) |
| **FLUID-02** | VAV Box Gradient Descent Test Input Size Mismatch | — | — | — | open | [history](investigations/fluid-02-vav-box-gradient-descent-test-input.md) |

## FFD/CFD co-simulation (FFD)

| ID | Symptom | Current value vs band | Issue | Closes when | Status | History |
|---|---|---|---|---|---|---|
| **FFD-01** | buoyancy-driven CHTC analytical comparison — RESOLVED (test-side Ra miscalculation) | test-side Ra miscalculation corrected | — | RESOLVED | resolved | [history](investigations/ffd-01-buoyancy-driven-chtc-analytical-comparison-resolved.md) |
| **FFD-02** | peak cooling load tolerance — STRUCTURAL (stub lacks zone air energy balance) | — | — | — | open | [history](investigations/ffd-02-peak-cooling-load-tolerance-structural-stub.md) |

## Entries with no current measurement in the row

A `—` in the *Current value vs band* column means the measurement was not restated when
this document was restructured, not that the entry is stale or closed. The number is in
the linked investigation file. Filling these in is follow-on work; no value was invented
to populate a cell.

## See also

- `docs/ASHRAE140_RESULTS.md` — per-case engine output (authoritative for Cases 600/900)
- `docs/ASHRAE140_MULTI_ZONE_RESULTS.md` — Case 960/970 multi-zone output
- `tests/QUARANTINE.md` — quarantined tests, blocking issues, un-ignore criteria
- `docs/adr/0017-equation-based-dae-teacher-architecture.md` — the DAE teacher decision
- `ARCHITECTURE.md` §Current Module Status — the post-#1323 baseline rule
