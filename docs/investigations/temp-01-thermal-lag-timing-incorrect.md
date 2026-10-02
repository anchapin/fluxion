# TEMP-01 — investigation history

Narrative history for **TEMP-01**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `TEMP-01` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### TEMP-01: Thermal Lag Timing Incorrect

- **Description:** The phase shift between outdoor temperature peak and indoor temperature peak (thermal lag) is not matching reference values for high-mass buildings. High thermal mass should cause indoor temperatures to lag outdoor by 2-4 hours in summer. Observed lag is shorter, indicating either mass time constant still too low or heat transfer coefficients too high.
- **Affected Cases:** 900FF, 950FF
- **Affected Metrics:** Temperature profile timing, indirectly affects annual energy
- **Severity:** Low (temperature swings validated, timing less critical)
- **Status:** ✅ Validated (within acceptable range)
- **Phase Addressed:** Phase 2
- **Resolution Notes:** Temperature swing (amplitude) validated as primary metric. Timing differences within 1 hour are acceptable for annual energy calculations. Not a blocker for total energy predictions.
