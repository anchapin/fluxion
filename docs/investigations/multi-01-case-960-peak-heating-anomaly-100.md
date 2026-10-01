# MULTI-01 — investigation history

Narrative history for **MULTI-01**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `MULTI-01` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 2. No wording was changed, softened or deleted.

---

#### MULTI-01: Case 960 Peak Heating Anomaly (100 kW)

- **Description:** Case 960 peak heating was showing 100 kW (reference: 2.0-8.0 kW) due to two bugs:
  1. **Validator unit bug**: step_physics() returns kWh, but validator was multiplying by 1000 and treating as Watts
  2. **5R1C broadcasting bug**: Equipment path was using `from_scalar()` to broadcast same thermal_demand to all zones instead of using per-zone values from `hvac_power_demand()`
- **Affected Cases:** 960 (multi-zone sunspace building)
- **Affected Metrics:** Peak Heating (kW)
- **Severity:** High
- **GitHub Issue:** N/A (found during Phase 7A)
- **Status:** ✅ Fixed (Phase 7A)
- **Phase Addressed:** Phase 7A
- **Resolution Notes:**
  - Fix 1: Changed validator to use `model.get_peak_heating_power_kw() * 1000.0` instead of `hvac_kwh * 1000.0`
  - Fix 2: Changed equipment path to use `hvac_power_demand()` for per-zone values, added proper peak tracking from per-zone demand sum
  - Removed duplicate peak tracking code that was using undefined variable
  - Result: Peak heating now 8.90 kW (was 100 kW), just 0.9 kW above reference max of 8.0 kW
  - The small remaining deviation (11% above max) is acceptable given 5R1C model simplifications for 2-zone coupling

  **Follow-up (post-#1407 / #1456):** The Phase 7A fix above was correct for the
  kW-vs-kWh broadcasting bug, but the Case 960 engine output itself was still
  unreliable until the multi-zone validator was rewritten against the real 8760-
  step physics simulation (not the 12.4-vs-12.5 MWh self-referential stub that
  fabricated PASS — see `docs/ASHRAE140_MULTI_ZONE_RESULTS.md` §"Removed stub
  (issue #1407)"). The post-#1407 result for Case 960 peak heating is
  ~1.4 kW (below the 2 kW reference min), classified under **PeakHeatingLimit-01**
  below as an architectural 5R1C under-prediction rather than the MULTI-01 bug.

  **Validation-side fixes that close MULTI-01's accounting chain:**
  - **#1396 (`fix(validation): correct MultiZoneValidator energy-conservation
    accounting`)** — `MultiZoneValidator` no longer zeroes actual outputs on
    FAIL, so `validate_case_960` reports real engine kWh/kW instead of 0.0.
  - **#1399 (`fix(#1397): correct mass-node energy balance + unblock 3 of 4
    pre-existing validator tests`)** — fixed the mass-node balance sign error
    that caused the back-zone HVAC demand to be attributed to the sunspace
    instead of the conditioned zone, which is the underlying reason peak
    heating was 100 kW in the validator path even after the unit fix.
  - **#1402 + #1403 (`fix(invariant-checker): extend 9R4C branch to mirror
    BE-implicit lumped-mass`)** — extended the `invariant_checker` to enforce
    mass-node energy balance on the 9R4C path used by Case 960. Without
    #1402/#1403 the 9R4C mass-node drift was invisible to CI even though
    `MultiZoneValidator` was reporting numbers.

#### MULTI-01b: Case 960 — 6R2C Override Regression (#1456)

- **Description:** The `validate_case_960` path and `enable_advanced_solver`
  (Case-960 specific branch) were forcing `model.configure_6r2c_model(0.75, 100.0, None)`
  on top of the default 5R1C/9R4C selection. The 6R2C configuration pushed the
  back-zone to ~16°C (below the 20°C heating setpoint) and over-predicted annual
  heating by 264% (7.47 MWh vs 1.65-2.45 MWh reference) while driving annual
  cooling to 0.00 MWh (vs 1.55-2.78 reference).
- **Affected Cases:** 960 (multi-zone sunspace building)
- **Affected Metrics:** Annual Heating, Annual Cooling, Peak Heating, Peak Cooling
- **Severity:** High
- **GitHub Issue:** [#1456](https://github.com/anchapin/fluxion/issues/1456)
- **Status:** ✅ Fixed (#1456)
- **Phase Addressed:** Phase Wave 6
- **Resolution Notes:** Removed the broken `configure_6r2c_model` calls in
  `validate_case_960` (line 2503) and `enable_advanced_solver` (line 1458).
  The default 5R1C/9R4C path now produces:
  - Annual Heating ≈ 1.6 MWh (after COP/0.9), within 30% of reference midpoint
  - Annual Cooling ≈ 0.5 MWh (after COP/3.0)
  - Peak Heating ≈ 1.4 kW (below 2 kW reference minimum — see PeakHeatingLimit-01)
  - Peak Cooling ≈ 1.4 kW (within 0-4 kW reference band)
  The 14-test integration suite at `tests/ashrae_140_case_960_sunspace.rs`
  was 10/14 before the fix and is 15/15 after (added 1 regression test).
