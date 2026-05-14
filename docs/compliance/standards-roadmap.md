# fluxion Standards Certification Roadmap

**Product:** fluxion v0.8.0+
**Maintained by:** Building Standards Engineer
**Last Updated:** 2026-05-14
**Related Issue:** #751
**Scope:** Standards landscape for future certification planning. Current target: ASHRAE 140-2023.

---

## Overview

fluxion currently targets ASHRAE 140 (building thermal envelope). Several adjacent and market-relevant standards share significant physics overlap with fluxion. This document tracks prioritization and known gaps for each standard.

---

## Status Summary

| Standard | Market | Priority | fluxion Status |
|----------|--------|----------|----------------|
| ASHRAE 140-2023 | US/International (validation baseline) | 1 (current) | Active development |
| ASHRAE 90.1-2022 Appendix G | US Commercial | 2 | Planned |
| RESNET HERS (ANSI/RESNET/ICC 301-2022) | US Residential | 3 | Planned |
| California Title 24 ACM | California | 4 | Planned |
| ISO 52016-2017 / EN 13790 | EU | 5 | Planned |

---

## 1. ASHRAE 140-2023 — Current Target

**Standard:** ASHRAE Standard 140-2023 (Building Thermal Envelope and Fabric Load Tests)
**Status:** Active development (see `deviations-register.md` for open issues)
**Prerequisite:** Yes — required prerequisite for all other standards listed below

---

## 2. ASHRAE 90.1-2022 Appendix G — Commercial Market

**Standard:** ASHRAE 90.1-2022 (Energy Standard for Commercial Buildings) Appendix G Performance Rating Method
**Market:** Required for commercial building permits in most US jurisdictions; used for LEED, green building codes
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

**Standard:** ANSI/RESNET/ICC 301-2022
**Market:** Required for US new residential energy ratings; required for EPA Energy Star, DOE Zero Energy Ready Home
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

**Standard:** California Title 24 Alternative Calculation Method (ACM)
**Market:** California only (but largest single US market for building energy compliance)
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

**Standard:** ISO 52016-2017 (supersedes EN ISO 13790:2008)
**Market:** Required for EU building energy certificates; EPBD compliance
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

Phase 3: RESNET HERS
  └─ Infiltration model upgrade
  └─ Ventilation (62.2)
  └─ DHW energy
  └─ Duct leakage

Phase 4: Title 24 ACM
  └─ California climate zones
  └─ CBECC-Com comparison testing

Phase 5: ISO 52016
  └─ 2-node RC model alignment
  └─ EPW weather support
  └─ Primary energy metrics
```

---

## Notes

- ASHRAE 140 is the **prerequisite** for all other standards — completion of 140 compliance is blocking for subsequent phases.
- RESNET has a **separate approval process** from ASHRAE; early coordination with RESNET recommended before Phase 3 begins.
- Title 24 requires **DOE-2 comparison** — understanding DOE-2's underlying physics will be important for Phase 4.
