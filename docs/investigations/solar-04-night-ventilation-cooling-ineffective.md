# SOLAR-04 — investigation history

Narrative history for **SOLAR-04**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `SOLAR-04` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### SOLAR-04: Night Ventilation Cooling Ineffective

- **Description:** Case 650 (low-mass night ventilation) and Case 950 (high-mass night ventilation) test the effectiveness of nighttime natural ventilation for reducing daytime cooling. Reference shows significant cooling reduction. Fluxion shows minimal effect - cooling energy nearly identical to non-ventilated cases (600/900). This suggests either ventilation air exchange not implemented correctly, or thermal mass interaction not modeled properly.
- **Affected Cases:** 650, 950
- **Affected Metrics:** Annual Cooling, Peak Cooling
- **Severity:** Medium
- **GitHub Issue:** #276
- **Status:** 🔄 Open
- **Phase Addressed:** Phase 3
- **Resolution Notes:** Night ventilation parameter exists in case specs but may not be correctly applied in ventilation heat transfer calculations. Infiltration/ventilation rate multiplication during night hours needs verification.

---

## Round 26 update (2026-10-13): fan-capacity spec-pairing defect FIXED; residual effectiveness gap re-confirmed as model-form

Round 26's falsification pass checked the night-vent implementation against ASHRAE
140-2017 §5.2.2.1.5.2 / Table 5-160 line by line:

- **Schedule hours:** spec 18:00–07:00 (Hour 19 through Hour 7); engine
  `operating_hours (18, 7)` with `is_active_at_hour` wrapping midnight — correct.
- **Ventilation model form:** the 140 spec has NO operable-vent-area, wind, or stack
  coefficient model — night ventilation is a fixed fan capacity added to the specified
  infiltration. The engine implements exactly that (h_ve_night = capacity × ρ × cp / 3600
  on the zone air node, plus the #1279 ACH-based forced-convection interior-film
  multiplier under §5.2.1.10's "programs that compute time-varying coefficients" branch).
- **Fan capacity vs density route (DEFECT, fixed):** Table 5-160 gives two sanctioned
  inputs — 1703.16 m³/h (13.14 ach) for programs that automatically correct vent fan mass
  flow for altitude, and **1400 m³/h (10.80 ach) for programs with a fixed sea-level
  assumption**. This engine applies a fixed ρ = 1.2 kg/m³ everywhere, so it is the second
  kind of program, but it was built with the first kind's input: mass flow
  1703.16 × 1.2 = 2043.8 kg/h instead of 1680 (sea-level route) or 1730 (altitude route,
  ρ = 1.0156 at the table's 1650.9 m) — 18–22 % over-ventilation. Corrected to
  `fan_capacity: 1400.0` in `NightVentilation::case_650()` (round 26 PR).

Measured after the fix (Golden-NREL strict path, develop a0c78f6e → branch): 650 C
3.508 → 3.498 MWh; 950 C 0.640 → 0.654 MWh (stays in band); 650FF Tmin −24.45 → −24.33 °C,
Tmax 54.37 → 54.30; 950FF Tmin −22.44 → −21.97 °C, Tmax 37.98 → 38.05 (stays in band).
All other annual and free-float observables bit-identical; checker 9 PASS / 33 KNOWN-FAIL /
0 REGRESSION after the in-PR baseline re-record.

**The §28 effectiveness observation stands:** with the capacity now spec-conformant, the
engine's 900→950 cooling drop (1.684 → 0.654, −1.03 MWh) is still roughly half the
reference band-midpoint drop (−2.245 MWh). The residual is the free-temperature/mass
trajectory carrier (§19/§27), not a ventilation-term defect — the vent term's magnitude is
now exactly the spec's, so the remaining gap cannot be closed by any ventilation-input
correction. SOLAR-04's remaining scope is therefore the same accepted model-form carrier
tracked by §LIMIT-05/§LIMIT-17; the input-side question this row was opened for
("ventilation rate multiplication during night hours needs verification") is now settled
and fixed.
