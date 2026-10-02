# BASE-05 — investigation history

Narrative history for **BASE-05**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `BASE-05` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### BASE-05: Incorrect h_tr_em Heat Transfer Coefficient

- **Description:** The opaque exterior-to-mass conductance (h_tr_em) was calculated using an incorrect physics-based formula with fixed parameters (k=0.7 W/mK, d=0.1m for low-mass; k=1.4 W/mK, d=0.2m for high-mass) instead of using actual construction U-values from assembly layers. The variable `h_tr_op` was correctly calculated using actual U-values but never used. This caused h_tr_em to be 7.5x too high, propagating to sensitivity calculations and causing massive heating overprediction.
- **Affected Cases:** All cases (600, 900, and variant series)
- **Affected Metrics:** Annual Heating (2.5-3.5x overpredicted), Peak Heating (2.5x overpredicted)
- **Severity:** Critical
- **GitHub Issue:** #N/A (found during Phase 7A investigation)
- **Status:** ✅ Fixed (Phase 7A)
- **Phase Addressed:** Phase 7A
- **Resolution Notes:** Fixed by using `h_tr_op` (calculated from actual construction U-values) instead of `h_tr_em_physics` (calculated from fixed k and d parameters). Annual heating reduced from 3.5x overpredicted to 1.2-1.6x. Peak heating now within reference range. Low-mass peak cooling now within reference range.
