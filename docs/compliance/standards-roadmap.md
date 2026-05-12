# Standards Roadmap

**Contributes to:** [#751](https://github.com/anchapin/fluxion/issues/751)  
**Status:** Initial draft — for planning only; no code changes implied  
**Last updated:** 2026-05-12

---

## Purpose

This document tracks which building energy standards fluxion targets for validation or certification, in what order, and what the key physics/feature gaps are for each. It is a living planning document — update it as priorities shift.

---

## Prioritization

| Priority | Standard | Market | Status | Prerequisites |
|----------|----------|--------|--------|---------------|
| 1 | **ASHRAE 140-2023** | All US simulations | 🔴 In progress (~36% pass rate) | — |
| 2 | **ASHRAE 90.1-2022 App. G** | US commercial buildings | 🔴 Not started | ASHRAE 140 completion |
| 3 | **RESNET HERS (ANSI/RESNET/ICC 301-2022)** | US residential ratings | 🔴 Not started | ASHRAE 140 + infiltration model |
| 4 | **California Title 24 ACM** | California (all sectors) | 🔴 Not started | 90.1 path + CEC review |
| 5 | **ISO 52016 / EN 13790** | European market | 🔴 Not started | Multi-zone, different RC model |

---

## Detailed Standard Profiles

### 1. ASHRAE Standard 140-2023 (BESTEST)

**What it is:** The foundational building thermal envelope test procedure. Tests a simulation engine's ability to correctly model heat transfer through walls, windows, infiltration, and HVAC ideal loads.

**Market relevance:** Required as prerequisite for most other US standards (90.1, RESNET). Used by national labs, DOE, ASHRAE technical committees to evaluate new engines.

**Current status:** ~36% pass rate. Key blockers: synthetic weather data (#732), peak load CTF solver limitations (DEV-001), Section 8 output gaps (#749).

**Target:** v1.0 (finite volume solver + Waves 1–5 physics fixes + output reporting compliance)

**Key physics needed (beyond current):**
- ✅ ISO 13790 5R1C thermal network (implemented, accuracy improving)
- ❌ Peak load accuracy — requires v1.0 FV solver
- ❌ Normative weather file (#732)
- ❌ Section 8 output format (#749)

---

### 2. ASHRAE Standard 90.1-2022 Appendix G — Performance Rating Method

**What it is:** The energy standard for commercial buildings. Appendix G defines the simulation-based performance rating method used for LEED, green building codes, above-code incentives.

**Market relevance:** Required for commercial building permits in most US jurisdictions. Pathway for LEED v4.1 energy credits. Very large market.

**Current status:** Not started. Requires ASHRAE 140 as prerequisite.

**Key physics gaps vs. current fluxion:**
- Multi-zone HVAC (VAV, heat pump, chiller/boiler plant with real equipment performance curves)
- Lighting power density scheduling
- Plug loads, process loads
- Service water heating
- Fenestration model with shading schedule

**Validation:** Requires successful ASHRAE 140, then 90.1 Appendix G acceptance tests (ASHRAE has a formal tool approval process).

---

### 3. RESNET HERS — ANSI/RESNET/ICC 301-2022

**What it is:** The US residential energy rating standard. Powers EPA Energy Star, DOE Zero Energy Ready Home, and most state residential energy codes.

**Market relevance:** Required for new residential energy ratings and many federal/state incentive programs. Every new home built to code needs a HERS rating.

**Current status:** Not started.

**Key physics gaps vs. current fluxion:**
- Infiltration: ASHRAE 136 method (coupled to indoor/outdoor ΔT) — current fixed 0.5 ACH is not sufficient
- Mechanical ventilation: ASHRAE 62.2 (balanced, exhaust, supply)
- Duct leakage and distribution efficiency model
- Domestic hot water (DHW) energy
- Weather: TMY3 files for US locations (different from ASHRAE 140 Annex C Denver file)

**Validation:** Separate RESNET engine approval process (independent of ASHRAE 140). Requires RESNET to test and certify the engine.

---

### 4. California Title 24 — Alternative Calculation Method (ACM)

**What it is:** California's building energy code compliance path via simulation.

**Market relevance:** California is the single largest US building market. ACM approval required for energy compliance certification on all new California buildings.

**Current status:** Not started.

**Key physics gaps:**
- California climate zones (CZ1–CZ16): California-specific weather files
- CBECC-Com reference engine comparison: must demonstrate equivalence or superiority to the state reference engine
- CEC registration: formal state government approval process

---

### 5. ISO 52016-1:2017 / EN ISO 13790 — European Market

**What it is:** European standard for building energy calculations. Successor to EN 13790. Required for EU Building Energy Performance Certificates (EPC) under EPBD.

**Market relevance:** Required for EU energy certificates. Applicable across all EU member states and UK.

**Key differences from ASHRAE 140:**
- Zone model: 2-node RC network (different from fluxion's 5R1C, but related)
- Weather: European EPW files (EnergyPlus Weather) for EU locations
- Output metrics: Primary energy (not delivered energy); different conversion factors
- Monthly or hourly calculation modes

**Note:** fluxion's ISO 13790 5R1C base is partially aligned with this standard. More investigation needed to quantify the delta.

---

## Gap vs. Current fluxion Physics — Summary

| Feature | ASHRAE 140 | 90.1 App G | RESNET HERS | Title 24 | ISO 52016 |
|---------|-----------|-----------|------------|---------|---------|
| Single-zone thermal model | ✅ | ❌ need multi | ✅ (residential) | ❌ | ✅ |
| Multi-zone HVAC | ❌ | ❌ critical | ✅ simple | ❌ | ❌ |
| Equipment curves (real HVAC) | N/A (ideal loads) | ❌ required | partial | ❌ | N/A |
| Wind/stack infiltration | ❌ (not needed for 140) | partial | ❌ required | ❌ | ❌ |
| Mechanical ventilation | ❌ | ❌ | ❌ required | ❌ | ❌ |
| DHW energy | ❌ | ❌ | ❌ required | ❌ | ❌ |
| Normative weather files | ❌ (#732) | ❌ | ❌ | ❌ | ❌ |
| Peak load accuracy | ❌ (DEV-001) | ❌ | ❌ | ❌ | ❌ |

---

## Recommended Next Decision Point

After ASHRAE 140 v1.0 completion: decide whether to pursue 90.1 App G (commercial, larger market) or RESNET HERS (residential, different physics set) first. These require largely non-overlapping physics additions and should probably be sequential, not parallel.
