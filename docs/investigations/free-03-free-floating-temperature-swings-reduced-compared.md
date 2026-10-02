# FREE-03 — investigation history

Narrative history for **FREE-03**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `FREE-03` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### FREE-03: Free-Floating Temperature Swings Reduced Compared to Reference

- **Description:** All free-floating cases show damped temperature swings compared to reference programs. This was expected initially (thermal mass was under-predicted), but even after correcting thermal mass capacitance, swings remain smaller than reference. This indicates either the thermal mass time constant is still too long or heat transfer coefficients are too high, damping diurnal cycles excessively.
- **Affected Cases:** All free-floating cases (600FF, 650FF, 900FF, 950FF)
- **Affected Metrics:** Min Free-Float, Max Free-Float (both show reduced amplitude)
- **Severity:** Medium
- **Status:** ✅ Resolved
- **Phase Addressed:** Phase 2
- **Resolution Notes:** Resolved via Issue #2339 sub-hour air-node sub-stepping (commit 645116d). All FF cases now pass acceptance criteria: 600FF swing=86.3°C (≥80°C, ref 80.5°C), 650FF=91.1°C (≥82°C, ref 86.5°C), 900FF=62.8°C (≥50°C, ref 48.2°C), 950FF=66.8°C (≥58°C, ref 58.7°C).
