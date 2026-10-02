# BASE-04 — investigation history

Narrative history for **BASE-04**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `BASE-04` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### BASE-04: Denver TMY Weather Data Confirmation

- **Description:** ASHRAE 140 requires Denver TMY (Typical Meteorological Year) weather data for all cases. Initial implementation used generic weather data, causing discrepancies.
- **Affected Cases:** All cases
- **Affected Metrics:** All metrics (weather drives all simulations)
- **Severity:** High
- **Status:** ✅ Fixed (Phase 1)
- **Phase Addressed:** Phase 1
- **Resolution Notes:** Integrated Denver TMY weather file from ASHRAE 140 reference data. All simulations now use correct year-one weather sequence.
- **Weather Station Note (Issue #2429):** Fluxion's embedded `DenverTmyWeather` is a synthetic approximation of Denver climate (per `src/weather/denver.rs`), while ASHRAE 140 reference data (`tests/reference_data/`) was generated with the actual `USA_CO_Golden-NREL.724666_TMY3.epw` file. Denver-Stapleton TMY3 (39.83°N, 104.65°W, 1655m) and Golden-NREL TMY3 (39.74°N, 105.18°W, 1829m) are ~45 km apart at different elevations, giving slightly different summer peak conditions. For most cases this minor difference is insignificant. Case 950 (night ventilation, high-mass) is most sensitive: the weather station difference explains a ~0.3 kW variation in peak_cooling. Current result (0.859 kW) is within the 0.70–0.90 kW reference band, so this is a documented minor effect, not a blocking issue.
