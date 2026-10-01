# LIMIT-04 — investigation history

Narrative history for **LIMIT-04**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-04` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### LIMIT-04: Case 960 Peak Heating Overprediction (Multi-Zone)

- **Description:** Case 960 (sunspace + back-zone) shows peak heating of 32 kW (hitting capacity limit) while reference is 2-8 kW. The root cause is the free-floating sunspace (zone 1) overheating to unrealistic temperatures (235°C), which transfers extreme heat through the common wall to zone 0.

- **Affected Cases:** 960

- **Affected Metrics:** Peak Heating

- **Severity:** High

- **Status:** ⚠️ **Known Limitation** (5R1C model limitation for multi-zone sunspaces)

- **Phase Addressed:** Phase 7B (investigated, root cause identified)

- **Resolution Notes:** Investigation showed that the sunspace (zone 1) is free-floating and accumulates solar heat without effective cooling:
  1. Sunspace has 6 m² south-facing windows + high-mass construction
  2. Denver TMY summer weather provides high solar radiation
  3. Sunspace is free-floating with only 0.5 ACH infiltration
  4. Door opening ventilation (stack effect) provides insufficient cooling (~0.1-0.2 ACH)
  5. Solar gains accumulate, sunspace heats to 235°C over 17 hours
  6. Heat transfers through 21.6 m² common wall to back-zone
  7. Back-zone temperature crashes to -26°C, HVAC demand hits 32 kW capacity limit

This is a known limitation of the 5R1C model for multi-zone buildings with free-floating zones. The simplified model doesn't capture complex thermal dynamics of sunspaces. The inter-zone heat transfer works correctly, but the lack of effective sunspace ventilation causes unrealistic temperatures.

- **Potential Solutions:**
  1. Increase minimum ventilation for free-floating zones (currently 0.5 ACH may be insufficient)
  2. Adjust solar gain distribution for sunspace configurations
  3. Separate sunspace from thermal model (decouple from conditioned zone)
  4. Accept as model limitation and document sunspace validation as out-of-scope
