# LIMIT-01 — investigation history

Narrative history for **LIMIT-01**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-01` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### LIMIT-01: High-Mass Annual Energy Discrepancy

- **Description:** High-mass buildings (900 series) show annual heating 30-200% above reference and cooling 20-80% above reference. The 5R1C model's single thermal mass node and simplified radiation/convection assumptions cannot capture the dynamic response of extremely high thermal mass buildings (Cm ≈ 1,000,000 J/K). This is a known limitation - yearly totals drift from reference due to accumulated phase errors in implicit integration.
- **Affected Cases:** 900, 910, 920, 930, 940, 950, 900FF, 950FF
- **Affected Metrics:** Annual Heating, Annual Cooling
- **Severity:** Medium (accepted limitation)
- **Status:** ✅ Won't Fix (by design)
- **Phase Addressed:** N/A (known from start)
- **Resolution Notes:** The 5R1C model is a simplified representation intended for quick load estimation, not detailed simulation. For high-mass cases, we accept larger tolerances. Reference ranges in `benchmark.rs` are calibrated for 5R1C to reflect this limitation.
