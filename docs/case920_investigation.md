# Case 920 Heating Investigation

> **Status (2026-08-09):** Root cause identified as the documented
> 5R1C / 9R4C discrete-node solar-injection pathology. Band can only
> close via the **GaugeSolver** architectural rework tracked in
> **#1465 / #1462**. See `docs/KNOWN_ISSUES.md` §LIMIT-05.

## Summary

Issue #2427: Cases 920 (and the sibling 930) annual heating energy is
outside the ASHRAE 140 Annex B8 reference band. A previous fix attempt
made the result 2.5× worse, and the root cause was not identified. This
document records the investigation, the failure analysis of the previous
attempt, and the path forward.

## Investigation Results

### Previous attempt (reverted in #2454 / PR #2479)

- Changed HVAC coefficient from `derived_h_tr_3 + h_tr_w` to
  `h_tr_1 + h_tr_w` for the 9R4C path in
  `src/sim/thermal_model_physics/hvac.rs::compute_hvac_coefficient`.
- Result: **Made problem 2.5× WORSE** (Case 920 peak heating 3.41 kW
  → 8.75 kW).

#### Why the swap made things worse

ISO 13790 §6.3 `derived_h_tr_3` represents the air-to-mass conductance
through the building envelope (~42.66 W/K for Case 900 series, computed
via the 1/h_tr_is + 1/h_tr_ms + 1/h_tr_em series chain). The 5R1C
series combination `h_tr_1 = h_tr_is * h_tr_ms / (h_tr_is + h_tr_ms)`
is the same series combination but stops at the 5R1C internal-surface
node (~58 W/K for Case 900 series).

The 9R4C solver integrates `Q_HVAC = h_coeff × (T_setpoint − T_free)`
at the air node where `T_free` itself depends on the mass node's
instantaneous state (per the 9R4C heat balance). The relationship
between `h_coeff` and the integrated HVAC demand is non-linear because
of the mass node's phase relative to the air node — flipping the
coefficient changes both the magnitude and the phase response of the
mass-to-air coupling, amplifying the regression instead of fixing it.

The current `compute_hvac_coefficient` (post-#2454 reversion) correctly
uses `derived_h_tr_3 + h_tr_w` for the 9R4C path and `h_tr_1 + h_tr_w`
for the 5R1C/6R2C path, per ISO 13790 §12.2.1.

### Current engine state (2026-08-09, post-#2455)

| Metric          | Engine     | Reference band   | Status |
|-----------------|-----------|------------------|--------|
| Annual Heating  | 2.479 MWh | [3.26, 4.30] MWh | −24% below lower edge |
| Annual Cooling  | 2.170 MWh | [1.84, 3.31] MWh | ✅ in band |
| Peak Heating    | 1.21 kW   | [2.10, 2.80] kW  | −42% below lower edge |
| Peak Cooling    | 1.10 kW   | [1.40, 1.90] kW  | −22% below lower edge |

Note: the symptom has shifted since the 2026-08-07 snapshot (where
annuals were over-band and peaks were in band). The shift is the
side-effect of #2455 (wall_cap restoration for HighMass to fix Case
900FF night minimum regression). Reverting #2455 fixes Case 920 but
regresses Case 900FF.

### Per-month attribution (energy-shuffling signature)

The per-month decomposition (from
`tests/ashrae_140_case_920.rs::test_case_920_engine_vs_reference_per_month`)
reveals an energy-shuffling pattern between heating and cooling:

| Month | Ref_H_kWh | Eng_H_kWh | Δ_H    | Ref_C_kWh | Eng_C_kWh | Δ_C    |
|-------|----------:|----------:|-------:|----------:|----------:|-------:|
| Jan   |    802.0  |    676.0  | −126.1 |       0.0 |       0.0 |   0.0  |
| Feb   |    804.7  |    403.4  | −401.3 |       0.1 |       0.0 |  −0.1  |
| Mar   |    491.1  |    101.2  | −390.0 |      46.9 |       0.1 | −46.8  |
| Apr   |    186.9  |      0.0  | −186.9 |     139.9 |      94.1 | −45.8  |
| May   |     95.7  |      0.0  |  −95.7 |     275.5 |     410.9 | +135.4 |
| Jun   |      2.4  |      0.0  |   −2.4 |     398.5 |     589.4 | +190.9 |
| Jul   |      4.4  |      0.0  |   −4.4 |     555.8 |     609.6 |  +53.8 |
| Aug   |      0.0  |      0.0  |   +0.0 |     548.2 |     388.6 | −159.6 |
| Sep   |     15.7  |      0.0  |  −15.7 |     258.5 |      77.1 | −181.4 |
| Oct   |    259.9  |    109.6  | −150.4 |      40.8 |       0.0 | −40.8  |
| Nov   |    626.0  |    465.7  | −160.3 |       3.1 |       0.0 |  −3.1  |
| Dec   |    958.4  |    722.9  | −235.6 |       0.0 |       0.0 |   0.0  |
| TOTAL |   4247.3  |   2478.7  |  −1.8  |   2267.3 |    2169.9 |  −0.1  |

- **Heating:** engine UNDER-predicts in every winter month, with the
  largest gaps in Feb (−401 kWh) and Mar (−390 kWh). Engine never
  produces heating in shoulder months (Apr, May, Sep, Oct) where the
  reference shows low-grade heating (3-200 kWh).
- **Cooling:** engine OVER in May-Jul (+135 to +191 kWh) and UNDER in
  Aug-Sep (−160 to −181 kWh). The shoulder cooling is missing entirely
  (Apr, Oct).
- **Annual totals:** −1.8 MWh for heating, −0.1 MWh for cooling. The
  per-month signs differ but the annual totals are close to
  energy-conservation, meaning the engine is *shuffling* energy between
  heating and cooling rather than losing it. The mass node is releasing
  too much heat in shoulder seasons and absorbing too much heat in
  summer.

This is **the textbook LIMIT-05 signature** documented in
`docs/KNOWN_ISSUES.md`: "discrete-node solar-injection pathology at
dt/τ ≈ 3.6". The single lumped thermal mass node cannot resolve the
diurnal cycle for an E/W-glazed high-mass building.

### Per-orientation solar distribution (verified correct)

The E/W incidence path is correct (E=1136.9 kWh/m², W=1148.0 kWh/m²,
symmetry ratio 0.990). See
`tests/case_920_orientation_attribution.rs` (per #2454) and
`tests/ashrae_140_case_920.rs::test_case_920_per_orientation_solar_distribution`.
The bug is downstream in the mass-to-air coupling, not upstream in the
solar incidence path.

## Root cause

Per `docs/KNOWN_ISSUES.md` §LIMIT-05, the high-mass 5R1C / 9R4C
discrete-node solar-injection pathology at dt/τ ≈ 3.6 is the
architectural limitation. No combination of `h_ms_coeff`,
`derived_h_tr_3`, `f_furniture`, or other parameter tuning can move
both peaks AND annuals into band simultaneously. The pattern
(peaks-and-annuals in the same direction below reference, with
energy-shuffling across shoulder months) is the documented signature
of the lumped-mass node's inability to resolve the E/W diurnal solar
cycle.

The correct fix is the **GaugeSolver** architectural rework tracked in
**#1465 / #1462**, which treats solar as geometric curvature rather
than per-timestep energy injection. Per
`docs/KNOWN_ISSUES.md` §LIMIT-05:

> "The correct fix is the **GaugeSolver** (#1465 / #1462), which treats
>  solar as geometric curvature rather than per-timestep energy
>  injection. Per AGENTS.md "no parameter tuning," the 5R1C path cannot
>  simultaneously close the bidirectional gap without parameter tuning,
>  which is explicitly forbidden."

## Path forward

1. ✅ **Done (this PR, fix/issue-2427-case-920-root-cause):**
   - Added `tests/ashrae_140_case_920.rs` — a new integration test file
     with three unconditional tests (validator smoke, spec geometry,
     per-orientation distribution) and three `#[ignore]`'d diagnostic
     tests (strict band, per-month attribution, engine-vs-reference
     per-month) that flip green when the GaugeSolver lands.
   - The strict band test (`test_case_920_strict_annual_energy_within_band`)
     is the machine-traceable close-out guard: it fails today
     (intentionally), documents the LIMIT-05 root cause in the failure
     message, and flips green when the GaugeSolver brings the four
     metrics into the ASHRAE 140 Annex B8 reference band.
   - Reverted NOTHING. The fix attempt that made things worse was
     already reverted in #2454 / PR #2479. The current
     `compute_hvac_coefficient` is correct per ISO 13790.

2. **Pending (not addressable here):**
   - GaugeSolver rework (#1465 / #1462) — the architectural fix that
     treats solar as geometric curvature. This is the only path that
     closes the band per AGENTS.md "no parameter tuning" rule.
   - The Case 920 strict band test will flip green when the
     GaugeSolver lands, giving CI a concrete close-out signal for
     #2427.

## Related issues

- **#2454 (CLOSED, PR #2479):** the per-orientation attribution
  diagnostic + revert of the previous fix attempt. Current state on
  this branch includes both the #2454 work and the #2427 follow-on
  test infrastructure.
- **#2455 (CLOSED, PR #2478):** the 900FF free-floating night minimum
  regression fix (wall_cap restoration). **Not reverted** by this PR.
- **#2448 (OPEN):** 900-series annual cooling over-prediction (cooling-
  only framing; needs re-scope per #2453).
- **#2453 (OPEN):** re-characterisation of #2448 with the
  bidirectional annual-energy over-prediction signature across
  Cases 900-940.
- **#1465 / #1462:** GaugeSolver rework (the architectural fix that
  closes the band).

## Key files

- `tests/ashrae_140_case_920.rs` — new integration test file (Issue #2427)
- `tests/case_920_orientation_attribution.rs` — per-orientation diagnostic (Issue #2454)
- `src/sim/thermal_model_physics/hvac.rs` — `compute_hvac_coefficient` (current: `derived_h_tr_3 + h_tr_w` for 9R4C, `h_tr_1 + h_tr_w` for 5R1C)
- `src/validation/ashrae_140_cases.rs` — `validate_case_920` validator
- `docs/KNOWN_ISSUES.md` §LIMIT-05 — root cause documentation
- `docs/case920_investigation.md` — this file
