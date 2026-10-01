# LIMIT-02 — investigation history

Narrative history for **LIMIT-02**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-02` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### LIMIT-02: Free-Floating Temperature Range for Low-Mass

- **Description:** Low-mass free-floating temperatures (600FF, 650FF) show max temperatures ~15°C below reference. The 5R1C model may underrepresent the rapid heating from solar gains due to lumped capacitance smoothing. This is an accepted trade-off for computational efficiency.
- **Affected Cases:** 600FF, 650FF
- **Affected Metrics:** Max Free-Float Temp
- **Severity:** Low (acceptable for annual energy)
- **Status:** ✅ Won't Fix (by design)
- **Phase Addressed:** N/A
- **Resolution Notes:** Model calibrated to match annual energy, not hourly free-floating extremes. Free-floating cases are diagnostic only - primary metrics are HVAC energy.
