# fluxion Standards Certification Roadmap

**Product:** fluxion v0.8.0+
**Maintained by:** Building Standards Engineer
**Last Updated:** 2026-06-16
**Related Issue:** #751
**Scope:** Standards landscape for future certification planning. Current target: ASHRAE 140-2023.

---

## Overview

fluxion currently targets ASHRAE 140 (building thermal envelope). Several adjacent and market-relevant standards share significant physics overlap with fluxion. This document tracks prioritization and known gaps for each standard.

---

## Status Summary

| Standard | Market | Priority | fluxion Status | Version |
|----------|--------|----------|----------------|---------|
| ASHRAE 140-2023 | US/International (validation baseline) | 1 (current) | Active development | 140-2023 |
| ASHRAE 90.1 Appendix G | US Commercial | 2 | Planned | 90.1-2022 → 90.1-2025 |
| RESNET HERS | US Residential | 3 | Planned | 301-2022 → 301-202x (draft) |
| California Title 24 ACM | California | 4 | Planned | 2025 (eff. Jan 2026) |
| ISO 52016 / EN 13790 | EU | 5 | Planned | 52016-1:2017, 52016-3:2024 |

---

## 1. ASHRAE 140-2023 — Current Target

**Standard:** ASHRAE Standard 140-2023 (Building Thermal Envelope and Fabric Load Tests)
**Status:** Active development (see `deviations-register.md` for open issues)
**Prerequisite:** Yes — required prerequisite for all other standards listed below

---

## 2. ASHRAE 90.1 Appendix G — Commercial Market

**Standard:** ASHRAE 90.1-2022 (Energy Standard for Commercial Buildings) Appendix G Performance Rating Method
**Market:** Required for commercial building permits in most US jurisdictions; used for LEED, green building codes
**Version Note:** ASHRAE 90.1-2025 is currently under development. The 2022 version remains the normative baseline for compliance. fluxion will target 90.1-2022 Appendix G for initial certification, with a migration path to 90.1-2025 upon final publication.
**Priority:** 2
**fluxion Status:** Not started

### Key Gaps vs. Current fluxion

| Gap | Description |
|-----|-------------|
| Multi-zone HVAC | VAV, heat pump, chiller/boiler plant simulation not in ASHRAE 140 validation suite |
| HVAC system simulation | Real equipment curves vs. ideal loads |
| Lighting power density scheduling | Not currently modeled |
| Exterior lighting | Not currently modeled |
| Plug loads | Not currently modeled |

### Validation Path
Requires successful ASHRAE 140 as prerequisite → then 90.1 Appendix G acceptance tests.

---

## 3. RESNET HERS — Residential Market

**Standard:** ANSI/RESNET/ICC 301-2022 (current); RESNET/ICC 301-202x draft (2025 update in progress)
**Market:** Required for US new residential energy ratings; required for EPA Energy Star, DOE Zero Energy Ready Home
**Version Note:** Draft PDS-01 (RESNET/ICC 301-202x) amends ANSI/RESNET/ICC 301-2022 for the 2025 edition. Key changes expected to include updated HVAC sizing methodology, electric readiness scoring, and revised Solar/Photovoltaic treatment. fluxion will target 301-2022 for initial certification, with migration to 301-202x upon finalization and adoption.
**Priority:** 3
**fluxion Status:** Not started

### Key Gaps vs. Current fluxion

| Gap | Description |
|-----|-------------|
| Infiltration model | Current: fixed 0.5 ACH. Required: coupled to indoor-outdoor temperature difference (ASHRAE 136 method) |
| Mechanical ventilation | Per ASHRAE 62.2, not currently modeled |
| Duct leakage | Distribution efficiency not modeled |
| DHW | Domestic hot water energy not modeled |
| Weather files | TMY3 (different from ASHRAE 140 Annex C) |

### Validation Path
Separate engine approval process through RESNET (independent of ASHRAE 140).

---

## 4. California Title 24 ACM — California Market

**Standard:** California Title 24, Part 6 (Energy Code); 2025 ACM Reference Manual
**Market:** California only (but largest single US market for building energy compliance)
**Version Note:** The 2025 Title 24 update takes effect January 1, 2026. Key changes include expanded heat pump requirements for new residential construction, electric-readiness mandates, revised ventilation standards (per ASHRAE 62.1/62.2), and a new Peak Cooling Energy compliance metric in the single-family ACM. The multi-family and commercial ACM updates are documented in the 2025 ACM Reference Manual. fluxion will target the 2025 ACM for initial certification (effective 2026).
**Priority:** 4
**fluxion Status:** Not started

### Key Gaps vs. Current fluxion

| Gap | Description |
|-----|-------------|
| Climate zones | California CZ1–CZ16, not currently supported |
| Engine approval | CEC-registered engines must pass DOE-2 comparison tests |
| Reference engine | CBECC-Com comparison required |

### Validation Path
CEC-registered engine approval via DOE-2 comparison tests.

---

## 5. ISO 52016 / EN 13790 — European Market

**Standard:** ISO 52016-1:2017, ISO 52016-3:2024 (new); EN ISO 13790:2008 (legacy)
**Market:** Required for EU building energy certificates; EPBD compliance
**Version Note:** ISO 52016-1:2017 remains the normative base for energy needs, internal temperatures, and heating/cooling loads. ISO 52016-3:2024 (CEN ISO 52016-3:2023) introduces new calculation methods for buildings with adaptive envelope elements. The EPBD (Energy Performance of Buildings Directive) 2024 revision drives adoption across EU member states. fluxion will target ISO 52016-1:2017 as the initial compliance path, with 52016-3 coverage for adaptive envelope scenarios.
**Priority:** 5
**fluxion Status:** Not started

### Key Gaps vs. Current fluxion

| Gap | Description |
|-----|-------------|
| Zone model | Different RC-network model (2-node RC vs. full FD/CTF) |
| Weather files | EPW format from EnergyPlus for EU locations |
| Output metrics | Primary energy (not delivered energy) |

### Validation Path
ISO 52016 compliance testing and EN 13790 alignment.

---

## IECC Relationship

**Standard:** IECC 2021 / 2024 (International Energy Conservation Code)
**Relationship:** IECC references ASHRAE 90.1 for commercial buildings; residential has its own prescriptive path
**Note:** No separate simulation engine approval required; 90.1 compliance covers commercial IECC

---

## Recommended Roadmap

```
Phase 1: ASHRAE 140-2023 (current)
  └─ Complete deviations register
  └─ Achieve full 140 compliance

Phase 2: ASHRAE 90.1 Appendix G
  └─ Multi-zone HVAC modeling
  └─ Equipment curves
  └─ Lighting/plug load scheduling
  └─ Target: 90.1-2022 (migrate to 90.1-2025 post-publication)

Phase 3: RESNET HERS
  └─ Infiltration model upgrade (ASHRAE 136 method)
  └─ Ventilation (ASHRAE 62.2)
  └─ DHW energy modeling
  └─ Duct leakage distribution efficiency
  └─ Target: 301-2022 (migrate to 301-202x post-finalization)

Phase 4: Title 24 ACM
  └─ California climate zones CZ1–CZ16
  └─ CBECC-Com comparison testing
  └─ Target: 2025 ACM (effective Jan 2026)

Phase 5: ISO 52016
  └─ 2-node RC model alignment
  └─ EPW weather support for EU locations
  └─ Primary energy metrics
  └─ Adaptive envelope coverage (ISO 52016-3:2024)
```

---

## Notes

- ASHRAE 140 is the **prerequisite** for all other standards — completion of 140 compliance is blocking for subsequent phases.
- RESNET has a **separate approval process** from ASHRAE; early coordination with RESNET recommended before Phase 3 begins. Draft PDS-01 (301-202x) is under development — monitor RESNET for final publication to avoid duplicating early modeling work.
- Title 24 2025 takes effect **January 1, 2026** — fluxion should target the 2025 ACM Reference Manual for initial certification.
- ASHRAE 90.1-2025 is under development — monitor for final publication; fluxion's initial 90.1-2022 Appendix G work will migrate to the new version.
- ISO 52016-3:2024 introduces adaptive envelope methods — consider early alignment for EU market readiness.
- Title 24 requires **DOE-2 comparison** — understanding DOE-2's underlying physics will be important for Phase 4.
