# BASE-02 — investigation history

Narrative history for **BASE-02**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `BASE-02` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### BASE-02: HVAC Load Calculation Using Ti Instead of Ti_free

- **Description:** HVAC demand calculation used current zone air temperature (Ti) instead of free-floating temperature (Ti_free). This violated ISO 13790's requirement that HVAC mode determination and load calculation should consider what the temperature would be without HVAC input, accounting for thermal mass buffering. The error caused systematic heating load over-prediction and incorrect HVAC energy allocation.
- **Affected Cases:** All cases with HVAC (all except free-floating)
- **Affected Metrics:** Annual Heating, Annual Cooling, Peak Heating, Peak Cooling
- **Severity:** Critical
- **Status:** ✅ Fixed (Phase 1)
- **Phase Addressed:** Phase 1
- **Resolution Notes:** Implemented correct Ti_free calculation per ISO 13790 equation: `Ti_free = (num_tm + num_phi_st + num_rest) / den`. HVAC mode (heating/cooling/off) determined from Ti_free, and load magnitude calculated as `|Ti_free - setpoint| * sensitivity`.
