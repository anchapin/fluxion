# LIMIT-08 — investigation history

Narrative history for **LIMIT-08**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-08` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### LIMIT-08: ASHRAE 140 Case 195 (no-loads) peak heating below reference band on the repo's Denver TMY (Issue #2868 — partially resolved)

- **Description:** Issue #2868 reported Case 195 annual heating ~6552 kWh vs the ASHRAE 140-2023 inter-program range [3951, 4217] kWh — a ~+82 % over-prediction. Root cause was that the zone with neither windows nor ventilation (`H_ve = H_tr,w = 0`) collapsed `H_tr,3 = 1/(1/H_tr,2 + 1/H_ms)` to zero in ISO 13790's supply-air elimination, decoupling the mass node from the air node and pinning the controlled zone air ~10 K BELOW its 20 °C setpoint. With `t_i_act = t_i_free + Q_hvac / h_tr_is` (a divisor that ignores the series path through mass→envelope), the ideal HVAC injected ~1534 W to hold a zone at ~10 °C, against an envelope loss of ~143 W, producing a ~10× energy-balance violation and the headline annual over-prediction. A second bug applied the hard-coded `SolAirTemperature::ashrae_140_default()` exterior IR emittance (ε = 0.9) to every case — Case 195 specifies ε_ext = 0.1 to suppress sky radiative exchange and isolate solid conduction.

  The Issue #2868 fix lands annual heating in the ASHRAE 140-2023 band [3.951, 4.217] MWh and brings the energy-balance violation to ~1× (Q ≈ 700 W injected against envelope loss ≈ 700 W). Annual cooling is ~0 kWh on the no-weather path because the spec's `opaque_absorptance = 0.0` zeroes the solar contribution; with `weather_data` set (the `tests/issue_2891_outdoor_convection.rs` path), `sky_temperature()` uses the horizontal-infrared channel and a small cooling load appears from the sol-air term.

  Peak heating on the post-fix model is **~1.0 kW**, **below** the ASHRAE 140-2023 reference band [1.791, 1.802] kW. The gap is a weather-file artifact, not a physics bug: the repo's synthetic Denver TMY3 has an annual minimum of −12.47 °C, while ASHRAE 140-2023 uses DRYCOLD.TM2 (min −24.4 °C, max 35.0 °C). The peak hour in DRYCOLD.TM2 sits at a colder extreme so the band centres on `UA × (20 − T_min) = 40.5 × 44.4 ≈ 1.80 kW`; the repo's weather file caps the demand at `40.5 × 32.5 ≈ 1.32 kW` at best. The benchmark (`src/validation/benchmark.rs` Case 195) was updated to the ASHRAE 140-2023 inter-program ranges so the strict gate and the validator pick up future weather-file changes correctly.

- **Affected Tests:** `tests/ashrae_140_case_195_solid_conduction.rs` (assertions relaxed to `ANNUAL_HEATING ∈ [3.50, 4.40]` MWh and `PEAK_HEATING ≤ 1.20 kW` to absorb the weather-file peak gap), `tests/issue_2891_outdoor_convection.rs` (already had a permissive ceiling `<= 6.30 MWh`; unchanged), `src/validation/benchmark.rs` Case 195 entries (corrected to ASHRAE 140-2023 ranges).
- **Affected Metrics:** Case 195 Peak Heating (kW) — bounded by weather, not engine.
- **Severity:** Low (engineering complete; gap is documented and tracked).
- **GitHub Issue:** #2868
- **Status:** ✅ **Fixed** for annual heating + annual cooling energy conservation; peak heating gap is a known weather-file limitation, tracked for the v1.3 release alongside the Case 600/900 strict-energy gate (#2506). The methodology follow-up (Issue #3060) is tracked as **§LIMIT-15** with three implementation options (switch test weather file / widen reference band / re-derive reference band from EnergyPlus DRYCOLD.TM2 runs) routed back to Issue #3060 for maintainer decision; per AGENTS.md / RULES.md / ADR-0001, none of the three options is auto-implementable in a single sub-agent's documentation PR.
