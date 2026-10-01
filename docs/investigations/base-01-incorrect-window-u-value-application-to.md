# BASE-01 — investigation history

Narrative history for **BASE-01**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `BASE-01` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### BASE-01: Incorrect Window U-Value Application to h_tr_em

- **Description:** Window U-value was incorrectly applied to h_tr_em (transmission: exterior → mass). The window's U-value should only affect h_tr_w (window conductance) and not the overall exterior-to-mass transmission coefficient. This caused incorrect heat flow from exterior to thermal mass.
- **Affected Cases:** All cases with windows (600, 610, 620, 630, 640, 650, 900, 910, 920, 930, 940, 950, 600FF, 650FF, 900FF, 950FF)
- **Affected Metrics:** Annual Heating, Annual Cooling, Peak Heating, Peak Cooling
- **Severity:** Critical
- **GitHub Issue:** (referenced in initial architecture issues)
- **Status:** ✅ Fixed (Phase 1)
- **Phase Addressed:** Phase 1
- **Resolution Notes:** Fixed by correcting `apply_parameters()` to separate window U-value (affects only `h_tr_w`) from overall envelope conductance. Window area calculations now properly accounted for in h_tr_w only.
