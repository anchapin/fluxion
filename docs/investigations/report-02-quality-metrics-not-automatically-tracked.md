# REPORT-02 — investigation history

Narrative history for **REPORT-02**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `REPORT-02` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### REPORT-02: Quality Metrics Not Automatically Tracked

- **Description:** No automatic computation of quality metrics (pass rate, MAE, max deviation) and historical tracking across phases. Currently manual extraction from reports.
- **Affected:** Progress monitoring
- **Severity:** Low
- **Status:** 🔄 Open (implementing in 05-04)
- **Phase Addressed:** Phase 5
- **Resolution Notes:** Creating `analyzer.rs` with `QualityMetrics` struct and phase comparison.
