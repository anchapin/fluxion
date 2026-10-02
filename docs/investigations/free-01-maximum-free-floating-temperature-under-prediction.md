# FREE-01 — investigation history

Narrative history for **FREE-01**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `FREE-01` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### FREE-01: Maximum Free-Floating Temperature Under-Prediction (Low-Mass)

- **Description:** Free-floating maximum temperatures (summer peak) for low-mass cases (600FF, 650FF) are 15-25°C below reference ranges. Low-mass buildings should experience higher temperature swings due to less thermal inertia. The under-prediction suggests either excessive heat loss or insufficient solar gain absorption in free-floating mode.
- **Affected Cases:** 600FF, 650FF
- **Affected Metrics:** Max Free-Float Temp (°C)
- **Severity:** High
- **Status:** 🔄 Open (partially addressed)
- **Phase Addressed:** Phase 2 (partial), Phase 3 (remaining)
- **Resolution Notes:** Thermal mass corrections (Phase 2) worsened this - T_max decreased further. Root cause likely in solar gain distribution or heat loss coefficients. Without HVAC, any error in gains/losses directly shows in temperature trajectory.
