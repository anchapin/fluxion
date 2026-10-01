# FREE-02 — investigation history

Narrative history for **FREE-02**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `FREE-02` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### FREE-02: Minimum Free-Floating Temperature Over-Prediction (High-Mass)

- **Description:** Free-floating minimum temperatures (winter nadir) for high-mass cases (900FF, 950FF) are 2-4°C above reference, and 950FF specifically fails by >3°C. High thermal mass should provide temperature stability and prevent excessive cooling. Over-prediction suggests inadequate heat loss or insufficient thermal mass responsiveness in cold conditions.
- **Affected Cases:** 900FF (borderline), 950FF (fail)
- **Affected Metrics:** Min Free-Float Temp (°C)
- **Severity:** Medium
- **Status:** ⚠️ Partial - 900FF now passes, 950FF still fails
- **Phase Addressed:** Phase 2
- **Resolution Notes:** Thermal mass integration corrected (implicit solver for Cm > 500 J/K). 900FF now within reference. 950FF min temperature still high - possibly due to ground coupling or night ventilation effects in free-float mode.
