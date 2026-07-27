# ASHRAE 140 60% Pass Rate Roadmap

- **What**: Technical roadmap for achieving the 60% ASHRAE 140 validation pass rate required by the release gate in `release_gates.yaml`
- **Current**: 17.2% pass rate (11/64 metrics) — gap of 43 percentage points from the 60% release gate minimum
- **Scope**: Six engineering epics across four parallel execution tracks; each epic maps to a specific GitHub child issue
- **Risk**: Track 1 (GaugeSolver) is the critical path — highest effort, highest single-payoff; Tracks 2–4 are independent and run in parallel
- **Expected**: 12–18 months with 2 engineers to reach 60% pass rate; first meaningful increment (Track 2 night-vent) in 1–2 months
- **Out of scope**: Parameter tuning to force-pass system tests, global 9R4C replacement of 5R1C, surrogate re-calibration before physics lands

*Generated: 2026-07-26 — branch `fix/issue-1907-wave4`*

---

## 1. Current State

### 1.1 Pass Rate Summary

| Metric | Value |
|--------|-------|
| Total ASHRAE 140 metrics | 64 |
| Currently passing | 11 (17.2%) |
| In warning band | 9 |
| Failing | 44 |
| Mean Absolute Error | 109.6% |
| Max single deviation | 1417.9% |
| Release gate minimum | **60.0%** |
| Gap to close | **42.8 pp** |

### 1.2 Per-Series Breakdown

**600 Series (Low-Mass Baseline) — 6 cases × 4 metrics = 24 metrics**

| Case | Annual H | Annual C | Peak H | Peak C | Notes |
|------|----------|----------|--------|--------|-------|
| 600 | 5165 kWh ⚠️ WARN | 5309 kWh ❌ | 4.37 kW ❌ | 5.13 kW ❌ | |
| 610 | 5643 kWh ❌ | 4408 kWh ❌ | 4.63 kW ❌ | 4.42 kW ❌ | shading + discrete-node pathology |
| 620 | 6347 kWh ❌ | 3208 kWh ⚠️ | 4.45 kW ❌ | 3.56 kW ❌ | |
| 630 | 6616 kWh ❌ | 2346 kWh ❌ | 4.46 kW ❌ | 2.96 kW ❌ | shading + discrete-node pathology |
| 640 | 3149 kWh ❌ | 5302 kWh ❌ | 4.38 kW ❌ | 5.13 kW ❌ | setback |
| 650 | 0.0 kWh ✅ | 4401 kWh ❌ | 0.0 kW ✅ | 4.88 kW ❌ | night vent + discrete-node |

**900 Series (High-Mass) — 6 cases × 4 metrics = 24 metrics**

| Case | Annual H | Annual C | Peak H | Peak C | Notes |
|------|----------|----------|--------|--------|-------|
| 900 | 6643 kWh ❌ | 9519 kWh ❌ | 4.65 kW ❌ | 5.01 kW ❌ | |
| 910 | 7689 kWh ❌ | 8868 kWh ❌ | 4.70 kW ❌ | 4.85 kW ❌ | shading |
| 920 | 8484 kWh ❌ | 8436 kWh ❌ | 5.20 kW ❌ | 4.74 kW ❌ | roof-solar + shading |
| 930 | 8889 kWh ❌ | 7575 kWh ❌ | 5.23 kW ❌ | 4.71 kW ❌ | shading |
| 940 | 8814 kWh ❌ | 14014 kWh ❌ | 11.20 kW ❌ | 9.62 kW ❌ | setback |
| 950 | 18727 kWh ❌ | 86138 kWh ❌ | 26.62 kW ❌ | 73.11 kW ❌ | night vent catastrophically broken |

**Free-Floating — 4 cases × 2 metrics = 8 metrics**

| Case | Min Temp | Max Temp | Notes |
|------|----------|----------|-------|
| 600FF | −17.40°C ✅ | 67.24°C ❌ | max free-float too warm |
| 650FF | −23.72°C ❌ | 65.57°C ❌ | night vent, too warm |
| 900FF | −10.17°C ❌ | 56.01°C ❌ | |
| 950FF | −15.08°C ❌ | 53.61°C ❌ | night vent catastrophically broken |

**Special — 2 cases × 4 metrics = 8 metrics**

| Case | Annual H | Annual C | Peak H | Peak C | Notes |
|------|----------|----------|--------|--------|-------|
| 960 | 6688 kWh ❌ | 9574 kWh ❌ | 4.70 kW ❌ | 5.01 kW ❌ | multi-zone 5R1C architectural |
| 195 | 7365 kWh ❌ | 0.0 kWh ✅ | 3.70 kW ❌ | 0.0 kW ✅ | |

### 1.3 Systematic Failure Patterns

| Pattern | Affected Metrics | Root Cause |
|---------|-----------------|------------|
| peak_cooling OVER + peak_heating UNDER | Cases 610–650 (5/5 OVER cooling, 3/3 UNDER heating) | Discrete-node solar-injection pathology — GaugeSolver (#1462) |
| High-mass peak cooling UNDER | Cases 900/910/920/930/940/950 | Roof-solar under-counting (~3×) — KNOWN_ISSUES LIMIT-05 |
| Night ventilation ineffective | Cases 650, 950 | h_ve_night / ACH routing — #1898 |
| Shading sensitivity weak | Cases 610, 630, 910, 930 | Shading coefficient not propagating — SOLAR-03 |
| Case 960 peak heating below floor | Case 960 | 5R1C architectural — PeakHeatingLimit-01 |
| Free-float max too warm | 600FF, 650FF, 900FF, 950FF | Thermal mass damping + solar — FREE-01/FREE-03 |

---

## 2. Root Cause Analysis

### 2.1 The Discrete-Node Solar-Injection Pathology (Track 1)

The simultaneous `peak_cooling OVER + peak_heating UNDER` signature on Cases 610–650 (5/5 OVER on cooling, 3/3 UNDER on heating) is the **textbook signature of a single lumped thermal node on a 1-hour timestep**: per-step solar energy is over-injected into the cooling peak while the winter-night heating peak is smeared/under-captured.

No adjustment to thermal-mass constants or solar-distribution parameters can resolve this within the 5R1C framework — per `AGENTS.md`, this is "parameter tuning to pass system tests" and is explicitly forbidden. The correct fix is **GaugeSolver**, which treats solar as geometric curvature rather than per-timestep energy injection.

See `tests/known_issues_regression.rs::test_issue1457_remaining_600_series_metrics` (currently `#[ignore]`-quarantined) for the machine-traceable signal that flips green when GaugeSolver lands.

### 2.2 Roof-Solar Under-Counting (~3×, Track 3)

Post-#1281 investigation confirms: the `h_ms_total` additive model is over-conservative but does **not** explain the cooling gap. The actual root cause is **upstream roof-solar under-counting** (~3×) documented in `docs/investigations/issue-1280-ctf-peak-load.md` §4. The HVAC demand is correctly proportional to `(T_free − T_set)` but `T_free` itself is too low because the driving solar load is too small.

### 2.3 Night Ventilation Ineffective (#1898, Track 2)

Cases 650/950 show near-identical cooling to non-ventilated cases 600/900, indicating that night ventilation `h_ve_night` is not routing into the zone heat balance. The issue is in `src/sim/ventilation.rs` `ScheduledVentilation::is_active_at_hour` + the per-zone routing in `physics_impl.rs`.

### 2.4 Shading Coefficient Propagation Gap (SOLAR-03, Track 3)

Reference programs show 30–60% cooling reduction with shading devices. Fluxion shows a weaker effect. Benchmarks per `KNOWN_ISSUES.md §SOLAR-03` suggest a **propagation gap** (coefficient is correct, not reaching the thermal network) rather than a coefficient error.

### 2.5 Case 960 Multi-Zone 5R1C Architectural (PeakHeatingLimit-01, Track 4)

Fluxion's 5R1C/9R4C Norton-equivalent `h_coeff` under-predicts Case 960 peak heating because the single lumped-mass node buffers the air-side free-floating temperature. EnergyPlus reports ~3.9 kW at hour 8000 (T_out = −9°C); Fluxion gives ~0.9 kW at T_out = −12°C. This is a **known architectural limitation** and is the documented reason Case 960 is in `known_failures` in `release_gates.yaml`.

---

## 3. Roadmap: Four-Track Execution Plan

### Track 1 — GaugeSolver Production Wiring (#1462) — 3–4 months

**Critical path. Biggest single win. Closes 14 of the 14 remaining Case 600 metrics.**

**Goal:** Replace per-timestep discrete-node solar injection in the 5R1C lumped-mass path with the gauge-theory formulation, eliminating the discrete-node pathology documented in `KNOWN_ISSUES.md §LIMIT-05 UPDATE`.

**Touch points:**
- `src/physics/method_selector.rs` — Add `Gauge` variant to `ThermalMethod` enum; extend `select_method` so 5R1C stays low-mass fast path and Gauge becomes high-mass / Case-600 default
- `src/physics/gauge_solver.rs` — Promote `energy_storage_rate` from the 0.0 stub to the actual scalar-field derivative
- `src/physics/gauge_zone_solver.rs` — Air-node ODE (lines 36–40) must consume the post-#1522 `air_thermal_capacitance` field on `ThermalModelData`
- `src/validation/ashrae_140_validator.rs::enable_advanced_solver` (line ~1471) — Extend solver-selection branch to opt Case-600 metrics into Gauge (mirror existing CTF/FD dispatch)
- `tests/known_issues_regression.rs::test_issue1457_remaining_600_series_metrics` — Remove `#[ignore]` when Track 1 lands

**Why first:** Per the #1522 investigation, no 5R1C-side air-node ODE can resolve the simultaneous `peak_cooling OVER / peak_heating UNDER` signature. GaugeSolver is the only path consistent with `AGENTS.md` ("no parameter tuning").

**Closes:** Cases 610–650 (14 metrics per `§LIMIT-05 UPDATE` table) plus 600FF/650FF free-float min temp.

---

### Track 2 — Night Ventilation Fix (#1898) — 1–2 months

**Highest ROI per unit risk. Independent of Track 1. Can run in parallel.**

**Goal:** Verify `Q_night_vent` end-to-end for Case 950 at the spec boundary (T_out, T_zone, ACH=13.14), compare cumulative kWh to EnergyPlus hourly reference; align with the `h_tr_is` boost already applied during active hours.

**Touch points:**
- `src/sim/ventilation.rs` — `ScheduledVentilation::night_ventilation` + `is_active_at_hour` (~lines 242/512–545)
- `src/sim/thermal_model_physics/physics_impl.rs` — Verify per-zone routing of `h_ve_night`, `h_vent_mass_zone` covers all Case 950/650 surfaces and not only zone 0 (lines 519, 1533, 2485, 2851)
- `tests/ashrae_140_setback_ventilation.rs` — Ventilation-specific regression suite
- `KNOWN_ISSUES.md §SOLAR-04 / §LIMIT-05` row for Case 950 (#1422 follow-up)

**Approach:** Trace `Q_night_vent` end-to-end for Case 950 at spec boundary (T_out, T_zone, ACH=13.14), compare cumulative kWh to EnergyPlus hourly reference; align with the `h_tr_is` boost already applied during active hours.

**Closes:** Cases 650, 950 (peak_cooling + annual_cooling band, both currently OVER by 92–352% per `§LIMIT-05 UPDATE` row for Case 950 peak_cooling).

---

### Track 3 — Solar Path: Roof Under-Counting + Shading Sensitivity — 3–5 months combined

**Sub-tasks must be sequenced Track 1 → Track 3 (solar injection is affected by Gauge topology changes).**

#### Track 3a — Roof Solar Under-Counting (~2–3 months)

**Goal:** Verify the post-#1323 roof_irr.total_wm2 routing into the air/mass split is consistent across Cases 900/910/920/930/940/950/960. The remaining Cases 910/920/930/940/950 peak_cooling gaps per `§LIMIT-05 UPDATE` table map to this.

**Touch points:**
- `src/sim/solar_gain_distribution.rs` — Verify roof irradiance routing
- `src/sim/invariant_checker.rs` (line 416 `sol_air.for_roof`) — Invariant enforcement for roof
- `src/sim/solar.rs` — Solar position and irradiance calculation
- `docs/investigations/issue-1280-ctf-peak-load.md §4` — Existing roof-solar investigation (must re-verify after Track 1)

**Closes:** Cases 910/920/930/940/950 peak_cooling (59–87% UNDER → within band).

#### Track 3b — Shading Sensitivity Fix (SOLAR-03) (~1–2 months)

**Goal:** Confirm shading coefficient propagates into the solar-gain distribution. Benchmarks per `KNOWN_ISSUES.md §SOLAR-03` show 30–60% cooling reduction in reference programs vs. weaker effect in Fluxion — propagation gap, not coefficient gap.

**Touch points:**
- `src/sim/shading.rs` — Shading device application
- `fluxion_core::ashrae_cases::ShadingDevice` consumers — Coefficient consumption

**Closes:** Cases 610/630 (low-mass shading) and 910/930 (high-mass shading) annual_cooling and peak_cooling.

---

### Track 4 — Architecture: Case 960 Multi-Zone + 9R4C Sky-Radiative — 2–3 months

**Independent of Tracks 1–3 if treated as an additional solver method. Shares 9R4C transport code with the surrogate path.**

#### Track 4a — Case 960 Multi-Zone (PeakHeatingLimit-01)

**Goal:** Switch Case 960 to the 9R4C solver via `from_spec` + `MassAirCouplingMode::ParallelResistance`, then wire `sky_temp` from `WeatherSource::sky_temperature()` into `prepare_solvers_and_sol_air` on the 9R4C path.

**Touch points:**
- `src/physics/multi_node_solver.rs` — 9R4C solver
- `src/sim/multi_zone_network.rs` — Multi-zone network
- `src/sim/interzone_radiation.rs` — Inter-zone heat transfer
- `src/sim/sky_radiation.rs` (`SkyRadiationExchange::for_roof`, line 376)
- `src/sim/thermal_model_physics/physics_impl.rs` (lines 330/1639/2456) — Sky temp wiring
- `tests/ashrae_140_case_960_sunspace.rs` — Case 960 integration suite (currently 15/15 per #1456; remaining gap is peak heating architectural)

**Note:** 5R1C cannot close Case 960 peak heating by construction (1.4 kW vs 2 kW reference floor — documented as a known architectural gap in `KNOWN_ISSUES.md §PeakHeatingLimit-01`). `test_peak_load_validation` allows a documented 5R1C under-prediction tolerance (< 85% error from the 5 kW reference midpoint).

**Closes:** Case 960 peak_heating ≥ 2 kW (architectural improvement; may remain in known_failures).

#### Track 4b — 9R4C Sky-Radiative Path (#1858)

**Goal:** Close the high-mass free-float night min residual (~0.6°C warm) documented in `docs/investigations/ISSUE_1168_ROOT_CAUSE.md`.

**Touch points:**
- `src/sim/sky_radiation.rs` — Sky radiation exchange
- `src/sim/thermal_model_physics/physics_impl.rs` — Physics implementation

**Closes:** 900FF/950FF minimum free-floating temperature (2–4°C too warm).

---

## 4. Parallel Track: HVAC BESTEST (RP-865)

The acceptance criteria in `release_gates.yaml` require **both** ASHRAE 140 and HVAC BESTEST ≥ 60%. Currently only ASHRAE 140 is wired into the required checks.

**Goal:** Wire `tests/hvac_bestest_validation.rs` and `src/validation/hvac_bestest/{cases,runner}.rs` into the same release-gate CI workflow (`.github/workflows/ashrae_140_strict_energy_gate.yml`) that enforces the ASHRAE 140 60% gate, so both become a single blocking check.

**Touch points:**
- `.github/workflows/ashrae_140_strict_energy_gate.yml` — Add HVAC BESTEST job
- `tests/hvac_bestest_validation.rs` — RP-865 cases AE101–AE445
- `tests/validation/hvac_bestest/README.md` — Case documentation

---

## 5. Acceptance Criteria Alignment

| Gate | Threshold | Primary Closer | Notes |
|------|-----------|---------------|-------|
| `validation.min_pass_rate` | ≥ 60% | Tracks 1+3 (bulk), Track 2 (Case 650/950), Track 4 (Case 960) | |
| `validation.max_mae` | ≤ 50% | Track 1 (closes simultaneous OVER/UNDER signature on Case 600 series) | MAE cannot fall below ~80% without Track 1 |
| `validation.individual.max_deviation` | ≤ 100% | Tracks 1+2 (Case 600/650/950 within band) | 2 cases may exceed per `extreme_deviation_limit` |
| `validation.by_series.baseline` | ≥ 50% | Track 1 | |
| `validation.by_series.high_mass` | ≥ 50% | Tracks 3+4 | |
| `validation.by_series.free_floating` | ≥ 60% | Track 4b | |
| `validation.by_series.special` | ≥ 50% | Track 4a | |

**Known failures:** `release_gates.yaml:59–62` lists `"900"` and `"600"` as `known_failures`. Post-Track 1: remove Case 600 from `known_failures`. Post-Track 3: remove Case 900 from `known_failures`. This is a follow-up after the epics land — not in scope for this roadmap.

---

## 6. Dependency Ordering

```
Track 1 (GaugeSolver) ──────────────────────────────┐
  ↓ (solar injection changes affect Track 3)         │
Track 3a (Roof Solar) ───────────────────────────────│── All run in parallel
Track 2 (Night Vent) ────────────────────────────────┤     with 2 engineers
Track 4a (Case 960 Multi-Zone) ─────────────────────┤
Track 4b (9R4C Sky-Radiative) ──────────────────────┘

Track 3b (Shading) — independent, run after 3a

HVAC BESTEST CI — independent, run in parallel with all
```

**Critical path:** Track 1 → Track 3a. Tracks 2 and 4 are fully independent and should run in parallel with Track 1.

---

## 7. Out of Scope (Explicit)

- `tests/known_issues_regression.rs::test_issue1457_remaining_600_series_metrics` `#[ignore]` quarantine stays until Track 1 lands. Any mechanism to force Case 600 series into band via parameter tuning is **explicitly anti-pattern** per `AGENTS.md`.
- Replacing 5R1C globally with 9R4C — the low-mass default stays on 5R1C to preserve ≥150 configs/sec throughput gate (`release_gates.yaml:72–74`).
- Surrogate v3.1 re-calibration — runs **after** Tracks 1 and 3, not before, because retraining against a soon-to-be-superseded solver is throwaway work.
- Parameter tuning of any kind to make system tests pass — per `AGENTS.md` "fix the underlying math."

---

## 8. GitHub Issue Chain

| Epic | Issue | Track | Parent |
|------|-------|-------|--------|
| GaugeSolver production wiring | #1462 | Track 1 | #1907 |
| Night ventilation fix | #1898 | Track 2 | #1907 |
| Roof-solar under-counting | KNOWN_ISSUES LIMIT-05 | Track 3a | #1907 |
| Shading sensitivity | SOLAR-03 | Track 3b | #1907 |
| Case 960 multi-zone | PeakHeatingLimit-01 | Track 4a | #1907 |
| 9R4C sky-radiative | #1858 | Track 4b | #1907 |
| HVAC BESTEST CI wiring | *(new issue)* | Parallel | #1907 |

---

## 9. Success Metrics

| Milestone | Target Pass Rate | Key Closures |
|----------|-----------------|--------------|
| Post-Track 2 (Night Vent) | ~25% | Case 650/950 annual + peak cooling |
| Post-Track 1 (GaugeSolver) | ~40–45% | 14 Case 600 metrics, Cases 610–650 peak |
| Post-Track 3a (Roof Solar) | ~50–55% | Cases 900/910/920/930/940/950 peak cooling |
| Post-Track 4 (Multi-Zone + Sky-Rad) | ~60%+ | Case 960, 900FF/950FF free-float |
| Post-Track 3b (Shading) | ~60%+ | Cases 610/630/910/930 annual + peak |

---

## 10. CI Gate Wiring

When each epic lands, the following `#[ignore]` quarantined tests should be flipped to active:

| Epic | Test to Unquarantine |
|------|----------------------|
| Track 1 (GaugeSolver) | `tests/known_issues_regression.rs::test_issue1457_remaining_600_series_metrics` |
| Track 2 (Night Vent) | `tests/ashrae_140_setback_ventilation.rs` (existing, verify all pass) |
| Track 3a (Roof Solar) | `tests/ashrae_140_case_900.rs` (peak cooling assertions) |
| Track 4a (Case 960) | `tests/ashrae_140_case_960_sunspace.rs::test_peak_load_validation` |
| Track 4b (Sky-Radiative) | `tests/ashrae_140_free_floating.rs` (900FF/950FF min temp) |

`KNOWN_ISSUES.md` "Last Updated" line must be refreshed on every epic merge (CI-gated at 60 days per issue #1723).
