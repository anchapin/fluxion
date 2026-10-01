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
