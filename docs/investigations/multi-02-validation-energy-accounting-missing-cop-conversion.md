# MULTI-02 — investigation history

Narrative history for **MULTI-02**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `MULTI-02` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### MULTI-02: Validation Energy Accounting Missing COP Conversion

- **Description:** Case 960 annual cooling was 353% above reference because Fluxion's validation compared thermal HVAC energy directly to ASHRAE reference values, which are electrical. The missing COP/efficiency conversion caused apparent over-prediction. Solar gains and inter-zone heat transfer were correctly modeled; the issue was purely in the validation accounting.
- **Affected Cases:** 960
- **Affected Metrics:** Annual Cooling, Annual Heating
- **Severity:** High
- **GitHub Issue:** #273
- **Status:** ✅ Fixed (Phase 8)
- **Phase Addressed:** Phase 8
- **Resolution Notes:** Added COP correction (cooling COP=3.0, heating efficiency=0.9) to validation paths: `validate_case_960` and `validate_analytical_engine`. Core engine unchanged (thermal loads preserved). Case 960 now passes validation after correction.
