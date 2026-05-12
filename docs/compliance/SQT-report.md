# Software Qualification Test (SQT) Report — fluxion

**Status:** 🟡 IN PROGRESS — Wave 1 / Wave 1.5 physics corrections applied; output reporting gaps (#749) still block full completion  
**ASHRAE 140 edition:** 2023  
**fluxion version:** v0.8.0 (in progress; target for submission is v1.0)  
**Document owner:** Building Standards Engineer + Building Scientist and Rust Developer  
**Last updated:** 2026-05-12

---

## 1. Program Description

**Program name:** fluxion  
**Version:** v0.8.0  
**Developer:** [STUB — full name and contact]  
**Organization:** National Laboratory of the Rockies  
**Language / runtime:** Rust (compiled, release mode); Python bindings via pyo3  
**Repository:** https://github.com/anchapin/fluxion  
**License:** [STUB — confirm license]

---

## 2. Simulation Approach

### 2.1 Thermal Zone Model

fluxion uses an **ISO 13790 5-resistance / 1-capacitance (5R1C) thermal network** for zone-level heat balance calculations.

Key parameters:

| Parameter | Value | Source / Notes |
|-----------|-------|----------------|
| Number of zones | 1 | Multi-zone support in development |
| Timestep | 1 hour | Fixed |
| Exterior convection coefficient (h_ext) | **29.3 W/m²K** | ASHRAE 140 Sec. 5.2; corrected in Wave 1 (PR #765, was 18.3) |
| Interior convection coefficient (h_int) | **8.29 W/m²K** | CTF + method_selector; corrected in Wave 1 (was 8.0) |
| Exterior surface thermal resistance (R_ext) | **0.03413 m²K/W** | Derived from h_ext = 29.3; corrected in Wave 1 (was 0.04) |
| Interior surface thermal resistance (R_int) | **0.12063 m²K/W** | Derived from h_int = 8.29; corrected in Wave 1 (was 0.125) |
| Infiltration model | Fixed 0.5 ACH | No wind/stack coupling in v0.8.0 (see DEV-003) |

### 2.2 Conduction Model

fluxion uses a **Conductive Transfer Function (CTF) solver** for wall/roof conduction.

[STUB — describe CTF solver implementation: order of CTF coefficients, wall construction layering approach]

Known limitations in v0.8.0:
- CTF solver uses simplified parameters that underestimate thermal storage in high-mass constructions
- Peak load prediction accuracy significantly affected; see DEV-001 in deviations register
- Planned fix: v1.0 finite-difference/volume solver (Wave 2 routing in progress)

**Wave 1 correction applied (PR #765):** FD h coefficients in the CTF path were also corrected from 8.0/25.0 → 8.29/29.3 in Wave 1.5 (PR #772).

### 2.3 Solar Radiation Model

[STUB — document solar model: direct/diffuse separation, surface tilt handling, shading]

**Wave 1.5 corrections applied (PR #772):**
- Sol-air temperature for roof now wires `SolAirTemperature::for_roof()` — applies the ASHRAE long-wave radiation correction, resulting in approximately **−1.9°C reduction in peak roof sol-air temperature** at peak cooling conditions
- Solar Heat Gain Coefficient (SHGC): corrected from **0.789 → 0.787** (ASHRAE 140 Table B1-5)

### 2.4 Infiltration Model

Current implementation: Fixed 0.5 ACH, independent of indoor/outdoor temperature difference and wind.

Deviation from ASHRAE 140: ASHRAE 140 test cases specify fixed infiltration; this implementation is compliant for the standard test suite but will not be suitable for RESNET HERS certification (which requires infiltration coupled to temperature difference).

### 2.5 HVAC Model

[STUB — document HVAC ideal loads model. Describe how heating/cooling setpoints are applied.]

### 2.6 Surrogate Layer (AI Acceleration)

fluxion optionally replaces physics computations with trained ONNX neural network surrogates for high-throughput optimization workloads.

**Important:** ASHRAE 140 validation runs use `use_surrogates=False` (analytical physics mode only). Surrogate outputs are not used in compliance testing.

---

## 3. Input Data

### 3.1 Weather Data

[STUB — see `weather-file.md` for provenance documentation. This section should describe format, TMY/AMY distinction, and any preprocessing applied.]

### 3.2 Building Geometry

**Wave 1 correction applied (PR #765):** `case_900` opaque wall surface area corrected from **97.2 → 63.6 m²** per ASHRAE 140 Table B1-1. New constant `OPAQUE_WALL_AREA = 63.6 m²` introduced. This is a **−35% correction** that directly scales envelope heat fluxes; it is one of the most significant geometry-level fixes in the 900-series.

[STUB — describe how zone geometry (area, volume, surface areas) maps to the 5R1C model parameters. Provide the complete mapping table for ASHRAE 140 Case 600 and Case 900.]

| Parameter | Case 600 | Case 900 | Source |
|-----------|----------|----------|--------|
| Zone floor area | [STUB] m² | [STUB] m² | ASHRAE 140 Table B1-1 |
| Zone volume | [STUB] m³ | [STUB] m³ | ASHRAE 140 Table B1-1 |
| Opaque wall area | [STUB] m² | **63.6 m²** | ASHRAE 140 Table B1-1; Wave 1 corrected |
| Window area | [STUB] m² | [STUB] m² | ASHRAE 140 Table B1-5 |
| Roof area | [STUB] m² | [STUB] m² | ASHRAE 140 Table B1-1 |

### 3.3 Material Properties

**Wave 1 validated values (PR #765).** All values cross-referenced against ASHRAE 140-2023 Annex B.

#### Envelope Constructions

| Construction | Layer | Thermal Conductivity k (W/mK) | Density ρ (kg/m³) | Specific Heat Cp (J/kgK) | Source |
|---|---|---|---|---|---|
| Heavyweight wall | Concrete | **0.51** | **1400** | **840** | ASHRAE 140 Annex B; Wave 1 `ashrae_140_heavyweight()` |
| Insulated wall | Foam board | **0.040** | **10** | **1400** | ASHRAE 140 Annex B; Wave 1 `ashrae_140_foam_board()` |
| Interior finish | Gypsum board | **0.16** | **784** | **840** | ASHRAE 140 Annex B; Wave 1 `ashrae_140()` gypsum |
| [STUB] | Roof construction | [STUB] | [STUB] | [STUB] | ASHRAE 140 Annex B |
| [STUB] | Floor construction | [STUB] | [STUB] | [STUB] | ASHRAE 140 Annex B |

#### Convection Coefficients (Surface Boundaries)

| Coefficient | Value | Thermal Resistance | ASHRAE 140 Reference |
|---|---|---|---|
| h_ext (exterior) | **29.3 W/m²K** | R_ext = **0.03413 m²K/W** | Sec. 5.2 |
| h_int (interior) | **8.29 W/m²K** | R_int = **0.12063 m²K/W** | Sec. 5.2 |

#### Solar Parameters

| Parameter | Value | ASHRAE 140 Reference |
|---|---|---|
| SHGC (window) | **0.787** | Table B1-5; Wave 1.5 corrected (was 0.789) |
| [STUB — absorptance, emittance values] | | |

---

## 4. Numerical Methods

### 4.1 Time Integration

- Timestep: 1 hour (fixed)
- Integration scheme: [STUB — Euler, Crank-Nicolson, or other?]
- Convergence criteria: [STUB — if iterative, document tolerance and max iterations]

### 4.2 Annual Simulation

- Hours simulated: 8,760 (one year)
- Warmup period: [STUB — document warmup/pre-conditioning approach if any]

---

## 5. Test Environment

| Item | Value |
|------|-------|
| Operating System | [STUB — document CI OS; e.g., Ubuntu 22.04] |
| Rust version | [STUB — document via `rustc --version`] |
| Compiler flags | `--release` (optimization level 3) |
| Hardware | [STUB — CPU model used for CI runs] |
| Git commit | [STUB — pin to specific commit hash for submission] |

---

## 6. Known Deviations

See `deviations-register.md` for the full register. Summary of primary deviations as of v0.8.0 (post-Wave 1 / Wave 1.5):

**Resolved by Wave 1 (PR #765) and Wave 1.5 (PR #772):**
- ~~Incorrect h_ext: 18.3 W/m²K~~ → corrected to 29.3 W/m²K (DEV-006)
- ~~Incorrect h_int: 8.0 W/m²K~~ → corrected to 8.29 W/m²K (DEV-007)
- ~~Incorrect R_ext/R_int~~ → corrected to 0.03413 / 0.12063 m²K/W (DEV-008)
- ~~Incorrect case_900 surface area: 97.2 m²~~ → corrected to 63.6 m² (DEV-009)
- ~~Sol-air LW correction not wired for roof~~ → wired `for_roof()` (DEV-010)
- ~~SHGC 0.789~~ → corrected to 0.787 (DEV-011)

**Remaining open deviations:**

1. Peak load overestimation (76–100%+) — CTF solver limitation with instantaneous peak conditions (DEV-001; targeted in Wave 2)
2. High-mass annual energy: ±15–30% deviation from reference range (DEV-002; partially reduced by Wave 1 geometry + material corrections)
3. Fixed infiltration rate — no wind/stack coupling (DEV-003; compliant for ASHRAE 140)
4. **COMPLIANCE BLOCKER** — Synthetic weather data (DEV-004; Issue #732)
5. **COMPLIANCE BLOCKER** — Missing Section 8 output fields (DEV-005; Issue #749)

---

## 7. Test Results Summary

See `docs/ASHRAE140_RESULTS_v0.8.0.md` for complete validation results (when available).

**Overall pass rate as of v0.8.0:** ~**36%** of test cases within reference ranges

**Expected improvement post-Wave 1 + Wave 1.5 (before CI confirmation):**

| Correction | Expected Effect |
|---|---|
| h_ext: 18.3 → 29.3 W/m²K | Reduces exterior surface resistance ~17%; moves 900-series heating/cooling toward reference band |
| case_900 surface area: 97.2 → 63.6 m² (−35%) | Directly scales envelope heat fluxes by ~0.65; significant correction to peak cooling loads |
| Material property corrections (k, ρ, Cp) | Increases thermal mass correctly; dampens 900-series peak loads |
| Sol-air LW fix (roof) | ~−1.9°C peak cooling sol-air correction |
| SHGC: 0.789 → 0.787 | Minor; reduces solar gain slightly |
| **Combined (Wave 1 + Wave 1.5 + Wave 2 FD routing)** | **+8–12 passing metrics projected on 900-series** |

> ⚠️ These projections are based on engineering analysis. Actual CI-confirmed pass rate will be updated here once PR #765 and PR #772 are merged and CI results are available.

[STUB — update this section with per-case pass/fail table once output reporting gaps (#749) are resolved]

---

## 8. Physics Corrections Log

This section tracks post-v0.8.0 physics parameter corrections as they are applied.

### Wave 1 — PR #765 (`fix/wave1-material-properties-and-h-coefficients`)

**Merged:** [STUB — date]  
**Issues addressed:** #734, #735, #736, #737, #755

| Parameter | Before | After | ASHRAE 140 Reference |
|---|---|---|---|
| h_ext | 18.3 W/m²K | **29.3 W/m²K** | Sec. 5.2 |
| h_ext default | 25.0 W/m²K | **29.3 W/m²K** | Sec. 5.2 |
| h_int | 8.0 W/m²K | **8.29 W/m²K** | Sec. 5.2 |
| R_ext | 0.04 m²K/W | **0.03413 m²K/W** | Derived from h_ext |
| R_int | 0.125 m²K/W | **0.12063 m²K/W** | Derived from h_int |
| case_900 surface area | 97.2 m² | **63.6 m²** | Table B1-1 |

New material constructors added:
- `ashrae_140_heavyweight()` — k=0.51 W/mK, ρ=1400 kg/m³, Cp=840 J/kgK
- `ashrae_140_foam_board()` — k=0.040 W/mK, ρ=10 kg/m³, Cp=1400 J/kgK
- `ashrae_140()` gypsum — k=0.16 W/mK, ρ=784 kg/m³, Cp=840 J/kgK

New constant: `OPAQUE_WALL_AREA = 63.6 m²`

### Wave 1.5 — PR #772 (`fix/741-sol-air-lw-shgc-fd-h`)

**Merged:** [STUB — date]  
**Issues addressed:** #741

| Parameter | Before | After | Notes |
|---|---|---|---|
| Sol-air (roof) | Not wired | **`SolAirTemperature::for_roof()` wired** | ~−1.9°C peak cooling reduction |
| SHGC | 0.789 | **0.787** | ASHRAE 140 Table B1-5 |
| FD h_int | 8.0 W/m²K | **8.29 W/m²K** | Missed in Wave 1; corrected here |
| FD h_ext | 25.0 W/m²K | **29.3 W/m²K** | Missed in Wave 1; corrected here |

---

## 9. Submitter Attestation

[STUB — signature block for formal submission]

_I attest that the results reported herein were generated by the program version described above, using the inputs and methods documented here, without modification._

Name: _____________________________  
Title: _____________________________  
Date: _____________________________  
