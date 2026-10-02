# REPORT-01 — investigation history

Narrative history for **REPORT-01**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `REPORT-01` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### REPORT-01: Systematic Issues Classification Heuristic

- **Description:** Current `classify_systematic_issues()` in `reporter.rs` uses simple heuristics based on case ID and metric type. This crude approach misses many nuanced failure patterns and misclassifies some valid failures. For example, it classifies all 900 series annual energy as `ModelLimitation` even though some cases should be `SolarGains` or `ThermalMass` depending on the specific metric.
- **Affected:** Validation report accuracy
- **Severity:** Medium
- **Status:** 🔄 Open (improved in 05-04)
- **Phase Addressed:** Phase 5
- **Resolution Notes:** Plan 05-04 includes improved analyzer module with data-driven classification.
