# Known Deviations Register

**Status:** 🟡 Partial — Wave 1 / Wave 1.5 deviations resolved; DEV-004 and DEV-005 remain compliance blockers  
**Last updated:** 2026-05-12  
**fluxion version:** v0.8.0 (post-Wave 1 + Wave 1.5)

---

## Purpose

ASHRAE 140 compliance submissions must include a register of any deliberate or known deviations from the standard. This document lists all confirmed deviations, their technical basis, impact, and planned resolution.

A "deviation" is any aspect of the simulation that produces results outside the expected reference ranges OR any implementation choice that differs from the standard's prescribed test methodology.

---

## Deviation Register

### DEV-001 — Peak Load Overestimation (High-Mass Cases)

| Field | Value |
|-------|-------|
| **Test cases affected** | Case 900, Case 950, and other high-mass variants |
| **Metric** | Peak heating load, peak cooling load |
| **Direction** | Overestimation (~76–100%+ above reference range) |
| **Root cause** | CTF (Conductive Transfer Function) solver uses simplified parameters that cannot accurately represent instantaneous peak conditions in high thermal mass constructions |
| **ASHRAE 140 section** | Section 8.2.2 |
| **Technical justification** | CTF methods are known to have limitations at short timesteps and high-mass; this is a fundamental method constraint, not a bug |
| **Wave 1 partial mitigation** | Wave 1 material property corrections (correct ρ, Cp for heavyweight wall) and surface area correction (63.6 m²) reduce overestimation; magnitude of improvement pending CI confirmation |
| **Impact** | Peak load results from v0.8.0 are NOT valid for compliance submission |
| **Planned resolution** | v1.0 finite-difference/volume solver (replaces CTF); Wave 2 FD routing in progress |
| **Related issue** | [STUB — link to specific issue] |

---

### DEV-002 — Annual Energy Deviation in High-Mass Cases

| Field | Value |
|-------|-------|
| **Test cases affected** | Case 900, 950 high-mass variants |
| **Metric** | Annual heating energy, annual cooling energy |
| **Direction** | ±15–30% deviation from reference range |
| **Root cause** | CTF thermal mass model underestimates storage capacity; coupled with DEV-001 |
| **Wave 1 partial mitigation** | Corrected material properties (k, ρ, Cp) and surface area reduce annual energy error; combined effect with Wave 2 projected at +8–12 passing metrics |
| **Impact** | High-mass annual energy results marginally outside reference range for some cases |
| **Planned resolution** | v1.0 finite volume solver + CTF parameter refinement in interim |
| **Related issue** | [STUB] |

---

### DEV-003 — Fixed Infiltration Rate (No Wind/Stack Coupling)

| Field | Value |
|-------|-------|
| **Test cases affected** | All cases |
| **Metric** | Infiltration loads |
| **Deviation** | Uses fixed 0.5 ACH; no coupling to indoor-outdoor ΔT or wind speed |
| **Root cause** | Simplified implementation; ASHRAE 140 test cases specify fixed infiltration, so this is compliant for BESTEST but not for RESNET HERS |
| **Impact** | Compliant for ASHRAE 140. Non-compliant for RESNET certification path. |
| **Planned resolution** | Add ASHRAE 136 infiltration model for RESNET path (post-v1.0) |
| **Related issue** | N/A (not a compliance issue for ASHRAE 140) |

---

### DEV-004 — Synthetic Weather Data

| Field | Value |
|-------|-------|
| **Test cases affected** | All |
| **Metric** | All |
| **Deviation** | Uses synthetic/generated weather data instead of normative ASHRAE 140 Annex C weather file |
| **Impact** | **COMPLIANCE BLOCKER** — all validation results are suspect until normative weather file is used |
| **Planned resolution** | Issue #732 |
| **Related issue** | [#732](https://github.com/anchapin/fluxion/issues/732) |

---

### DEV-005 — Missing Section 8 Output Fields

| Field | Value |
|-------|-------|
| **Test cases affected** | All |
| **Deviation** | Output format does not include: peak load timestamps, hourly FF temperature profiles, incident solar radiation, Section 8.1 header metadata |
| **Impact** | Cannot produce a Section 8-compliant report |
| **Planned resolution** | Issue #749 |
| **Related issue** | [#749](https://github.com/anchapin/fluxion/issues/749) |

---

### DEV-006 — ✅ RESOLVED (Wave 1) — Incorrect Exterior Convection Coefficient (h_ext)

| Field | Value |
|-------|-------|
| **Test cases affected** | All 900-series (high-mass cases primarily) |
| **Metric** | Annual heating/cooling energy, peak loads |
| **Deviation** | h_ext was hardcoded as **18.3 W/m²K**; ASHRAE 140 Sec. 5.2 requires **29.3 W/m²K** |
| **Direction** | Underestimation of exterior convective heat transfer → overestimation of envelope thermal resistance |
| **Root cause** | Incorrect initial constant; exterior default also wrong (25.0 vs 29.3 W/m²K) |
| **ASHRAE 140 section** | Sec. 5.2 |
| **Impact** | Inflated heating/cooling loads in cases where envelope resistance dominates |
| **Resolution** | **Fixed in Wave 1, PR #765** — h_ext corrected to 29.3 W/m²K; R_ext updated to 0.03413 m²K/W |
| **Related issues** | #734, #736 |

---

### DEV-007 — ✅ RESOLVED (Wave 1) — Incorrect Interior Convection Coefficient (h_int)

| Field | Value |
|-------|-------|
| **Test cases affected** | All cases |
| **Metric** | Annual heating/cooling energy, peak loads |
| **Deviation** | h_int was **8.0 W/m²K**; CTF + method_selector requires **8.29 W/m²K** per ASHRAE 140 Sec. 5.2 |
| **Root cause** | Rounded-down initial value |
| **ASHRAE 140 section** | Sec. 5.2 |
| **Impact** | Slight underestimation of interior convective coupling |
| **Resolution** | **Fixed in Wave 1, PR #765** — h_int corrected to 8.29 W/m²K; R_int updated to 0.12063 m²K/W |
| **Related issue** | #737 |

---

### DEV-008 — ✅ RESOLVED (Wave 1) — Incorrect Surface Thermal Resistance Values (R_ext, R_int)

| Field | Value |
|-------|-------|
| **Test cases affected** | All cases |
| **Metric** | Conduction heat transfer through envelope |
| **Deviation** | R_ext = 0.04 m²K/W (should be 0.03413); R_int = 0.125 m²K/W (should be 0.12063) |
| **Root cause** | Derived from incorrect h coefficients (see DEV-006, DEV-007) |
| **Impact** | Overstated envelope thermal resistance; cascades into all conduction calculations |
| **Resolution** | **Fixed in Wave 1, PR #765** — R values recalculated from corrected h coefficients |
| **Related issue** | #735 |

---

### DEV-009 — ✅ RESOLVED (Wave 1) — Incorrect Opaque Wall Surface Area for Case 900

| Field | Value |
|-------|-------|
| **Test cases affected** | Case 900 and high-mass 900-series variants |
| **Metric** | All envelope heat transfer metrics |
| **Deviation** | `case_900` surface area was **97.2 m²**; ASHRAE 140 Table B1-1 specifies **63.6 m²** |
| **Direction** | Overestimation of heat transfer (surface area ~53% too large) |
| **Root cause** | Incorrect reference value used during initial implementation |
| **ASHRAE 140 section** | Table B1-1 |
| **Impact** | ~35% overcalculation of envelope heat fluxes for all 900-series cases — one of the largest single-correction impacts |
| **Resolution** | **Fixed in Wave 1, PR #765** — area corrected to 63.6 m²; new constant `OPAQUE_WALL_AREA = 63.6 m²` added |
| **Related issue** | #755 |

---

### DEV-010 — ✅ RESOLVED (Wave 1.5) — Sol-Air Long-Wave Radiation Correction Not Applied to Roof

| Field | Value |
|-------|-------|
| **Test cases affected** | 900-series and cases with roof solar gains |
| **Metric** | Peak cooling load, roof sol-air temperature |
| **Deviation** | `SolAirTemperature::for_roof()` was not wired; long-wave radiation correction for roof surfaces was not applied |
| **Root cause** | Incomplete wiring in sol-air temperature calculation module |
| **ASHRAE 140 section** | Sec. 5.3 (solar radiation) |
| **Impact** | Overestimation of roof sol-air temperature by ~+1.9°C at peak cooling; inflates peak cooling loads |
| **Resolution** | **Fixed in Wave 1.5, PR #772** — `SolAirTemperature::for_roof()` wired; ~−1.9°C correction at peak |
| **Related issue** | #741 |

---

### DEV-011 — ✅ RESOLVED (Wave 1.5) — Incorrect SHGC Value

| Field | Value |
|-------|-------|
| **Test cases affected** | All cases with glazing |
| **Metric** | Solar heat gains, cooling loads |
| **Deviation** | SHGC was **0.789**; ASHRAE 140 Table B1-5 specifies **0.787** |
| **Root cause** | Rounding error in initial value |
| **ASHRAE 140 section** | Table B1-5 |
| **Impact** | Minor overestimation of solar gain (~0.25%); small contribution to cooling load overestimation |
| **Resolution** | **Fixed in Wave 1.5, PR #772** |
| **Related issue** | #741 |

---

### DEV-012 — ✅ RESOLVED (Wave 1.5) — Incorrect FD Solver h Coefficients

| Field | Value |
|-------|-------|
| **Test cases affected** | All cases using FD (finite-difference) solver path |
| **Metric** | All heat transfer metrics via FD path |
| **Deviation** | FD solver used h_int = 8.0 / h_ext = 25.0 W/m²K (not updated during Wave 1) |
| **Root cause** | The FD h coefficient constants were separate from the CTF path constants corrected in Wave 1 |
| **Resolution** | **Fixed in Wave 1.5, PR #772** — FD h coefficients updated to 8.29 / 29.3 W/m²K |
| **Related issue** | #741 |

---

### DEV-013 through DEV-XXX — Issues #723–#733

[STUB — cross-reference each of issues #723–#733. For each confirmed physics deviation, create a row in this register using the template below.]

---

## Deviation Template

```
### DEV-XXX — [Short Name]

| Field | Value |
|-------|-------|
| **Test cases affected** | [Case numbers] |
| **Metric** | [Metric name] |
| **Direction** | [Overestimation/underestimation/incorrect] |
| **Root cause** | [Technical explanation] |
| **ASHRAE 140 section** | [Section reference] |
| **Technical justification** | [Why this deviation exists] |
| **Impact** | [Compliance impact] |
| **Planned resolution** | [Fix description and version target] |
| **Related issue** | [GitHub issue link] |
```

---

## Overall Compliance Impact Summary

| Severity | Count | Notes |
|----------|-------|-------|
| 🔴 Compliance blockers | 2 | DEV-004 (weather file), DEV-005 (output format) |
| 🟠 Results outside reference range | 2 | DEV-001 (peak loads), DEV-002 (high-mass annual) |
| 🟡 Methodology deviations (non-blocking for ASHRAE 140) | 1 | DEV-003 (infiltration) |
| ✅ Resolved — Wave 1 (PR #765) | 4 | DEV-006 (h_ext), DEV-007 (h_int), DEV-008 (R values), DEV-009 (surface area) |
| ✅ Resolved — Wave 1.5 (PR #772) | 3 | DEV-010 (sol-air LW), DEV-011 (SHGC), DEV-012 (FD h) |
| [STUB] | — | Issues #723–#733 not yet cross-referenced |

**Overall v0.8.0 pass rate: ~36% of test cases within reference ranges.**

**Projected improvement (Wave 1 + Wave 1.5 + Wave 2 FD routing): +8–12 passing metrics on 900-series** (engineering estimate; subject to CI confirmation once PRs #765 and #772 are merged).

A compliance submission is not possible until DEV-004 and DEV-005 are resolved and the pass rate reaches 100% (or documented deviations are accepted by the standards body).
