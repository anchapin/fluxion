# LIMIT-03 — investigation history

Narrative history for **LIMIT-03**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-03` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### LIMIT-03: Hardcoded HVAC Capacity Masking Design Errors

- **Description:** HVAC capacity (hvac_heating_capacity, hvac_cooling_capacity) was hardcoded to 100 kW for all cases. This unrealistically high value masked bugs and caused validation results to show peak loads hitting artificial capacity limits instead of actual demand.

- **Affected Cases:** All cases with HVAC (but most noticeable for Case 960 showing Peak H=100 kW vs expected 2-8 kW)

- **Affected Metrics:** Peak Heating, Peak Cooling

- **Severity:** High

- **Status:** ✅ Fixed

- **Phase Addressed:** Phase 7A (HVAC capacity fix)

- **Resolution Notes:** Changed from hardcoded 100 kW to floor-area-based calculation:
  - Heating: 500 W/m² × total_floor_area
  - Cooling: 600 W/m² × total_floor_area

  Examples:
  - Case 600 (96 m²): heating = 48 kW
  - Case 900 (96 m²): heating = 48 kW
  - Case 960 (64 m²): heating = 32 kW

  Case 960 now shows Peak H=32 kW (still too high but improved from 100 kW).

- **TODO:** Implement design day load calculation to determine HVAC capacity from actual peak loads at design temperatures (e.g., -5°C heating, 35°C cooling) with 1.1-1.2x safety margin.
