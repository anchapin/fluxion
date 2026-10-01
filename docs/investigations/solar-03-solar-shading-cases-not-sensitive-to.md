# SOLAR-03 — investigation history

Narrative history for **SOLAR-03**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `SOLAR-03` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### SOLAR-03: Solar Shading Cases Not Sensitive to Shading Changes

- **Description:** Cases 610, 630 (low-mass) and 910, 930 (high-mass) test the effect of south-facing and east/west shading devices. Reference programs show significant cooling reduction (30-60%) with shading. Fluxion shows smaller shading effects, indicating either incorrect shading coefficient application or insufficient solar gain to begin with (shading reduces already-low simulated gains).
- **Affected Cases:** 610, 630, 910, 930
- **Affected Metrics:** Annual Cooling, Peak Cooling
- **Severity:** Medium
- **Status:** 🔄 Open
- **Phase Addressed:** Phase 3
- **Resolution Notes:** Shading device configuration appears correct in case definitions, but solar radiation reduction not propagating correctly through thermal network. Possibly related to solar distribution to mass vs glass.
