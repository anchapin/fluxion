# ASHRAE 140 Standards Version Declaration

**Product:** fluxion v0.8.0+
**Document Type:** Standards Version Declaration
**Maintained by:** Building Standards Engineer
**Last Updated:** 2026-06-16
**Related Issue:** #750

---

## 1. Declaration

fluxion targets **ASHRAE Standard 140-2023** (Building Thermal Envelope and Fabric Load Tests) for compliance validation.

This document declares the specific edition of ASHRAE 140 used for all testing, the rationale for version selection, and the migration path.

---

## 2. Applicable Standard

| Field | Value |
|-------|-------|
| Standard Name | ASHRAE Standard 140 — Standard Method of Test for the Evaluation of Building Energy Analysis Computer Program Capabilities |
| Edition Declared | **ASHRAE 140-2023** (published 2023) |
| Previous Edition | ASHRAE 140-2017 |
| Normative Sections | All normative sections of 140-2023 including Annexes A through G |
| Compliance Target | Full compliance with all applicable sections of ASHRAE 140-2023 |

---

## 3. Version-Specific Compliance Notes

### 3.1 ASHRAE 140-2023 vs 140-2017

ASHRAE 140-2023 introduced the following changes relevant to fluxion:

| Change Area | ASHRAE 140-2017 | ASHRAE 140-2023 | fluxion Status |
|------------|-----------------|-----------------|----------------|
| Section 5.2.2 (Solar Distribution) | Implicit — assumed 100% to interior surfaces | Explicit: 100% transmitted solar distributed by area × absorptance | Implementation uses ISO 13790 (DEV-010) |
| Section 7 Free-float cases | Cases 600FF/650FF/900FF/950FF | Unchanged | HVAC not disabled — DEV-001 |
| Annex B material tables | Tables B1-1 through B1-6 | Tables updated for some material properties | See DEV-002, DEV-006, DEV-007 |
| Annex C Weather Data | TMY2 for Denver | TMY for Denver (same file, updated metadata) | Current: synthetic approximation — DEV-004 |
| Section 8 output reporting | Basic metrics | Expanded metrics including incident solar per orientation | Partially implemented — DEV-005 |

### 3.2 Section 5.2.2 Solar Distribution (Critical — DEV-010)

**ASHRAE 140-2023 Section 5.2.2** specifies:

> "The total transmitted solar gain transmitted by the glazing shall be distributed to the opaque surfaces of the building interior (walls, floor, ceiling) in proportion to their areas."

The transmitted solar fraction to the air node shall be zero. The distribution to opaque surfaces follows:

```
φᵢ = Aᵢ / ΣAⱼ  (for uniform interior surface absorptance α = 0.6)
```

**Current fluxion implementation:** References ISO 13790 Table C.2 monthly fractions (European RC-network method). This is **not applicable** to ASHRAE 140 compliance and constitutes DEV-010.

### 3.3 Annex C Weather Data

ASHRAE 140-2023 Annex C specifies the **Denver Stapleton TMY** file for all standard cases. The weather file location is:

```
USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw
```

**Current fluxion issue:** Code generates synthetic sine/cosine approximations of temperature and solar data rather than using the actual Annex C TMY file. This is DEV-004 — a **compliance blocker**.

---

## 4. Version Governance

### 4.1 Version Tracking in Code

The ASHRAE edition is declared in the `ReportHeader` struct (`src/validation/report.rs`):

```rust
pub struct ReportHeader {
    pub ashrae_edition: String,  // Set to "ASHRAE 140-2023"
}
```

### 4.2 Migration Policy

| Event | Action |
|-------|--------|
| ASHRAE 140-202x published | Evaluate changes; create migration issue; update this document |
| New ASHRAE edition becomes normative for compliance | Create new version of this document; maintain both editions during transition |
| ASHRAE 140-2023 withdrawn | Announcement-style update to this document with rationale |

### 4.3 Related Standards

For standards roadmap and relationship to ASHRAE 90.1, RESNET HERS, Title 24, and ISO 52016, see `standards-roadmap.md`.

---

## 5. Compliance Implications

**IMPORTANT:** fluxion has **not yet achieved formal ASHRAE 140-2023 compliance**. The following conditions must be met before any compliance claim:

1. All P1 open issues resolved (DEV-001, DEV-004, DEV-008 — see `deviations-register.md`)
2. All 64 ASHRAE 140-2023 validation metrics pass within acceptance tolerance
3. SQT Report (this repository) completed and signed
4. Inter-program comparison charts generated and reviewed
5. Weather file provenance documented with SHA256 hash
6. Reproducible test execution log generated from CI

Until these conditions are met, no ASHRAE 140 compliance claim shall be made in marketing materials, user documentation, or certification submissions.

---

## 6. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-06-16 | Building Standards Engineer | Initial version for Issue #750 |

---

*End of Standards Version Declaration*
