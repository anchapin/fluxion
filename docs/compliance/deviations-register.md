# Known Deviations Register

**Status:** 🟡 Partial — primary deviations documented; impact assessments and issues #723–#733 need cross-referencing  
**Last updated:** 2026-05-12  
**fluxion version:** v0.8.0

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
| **Impact** | Peak load results from v0.8.0 are NOT valid for compliance submission |
| **Planned resolution** | v1.0 finite-difference/volume solver (replaces CTF) |
| **Related issue** | [STUB — link to specific issue] |

---

### DEV-002 — Annual Energy Deviation in High-Mass Cases

| Field | Value |
|-------|-------|
| **Test cases affected** | Case 900, 950 high-mass variants |
| **Metric** | Annual heating energy, annual cooling energy |
| **Direction** | ±15–30% deviation from reference range |
| **Root cause** | CTF thermal mass model underestimates storage capacity; coupled with DEV-001 |
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

### DEV-006 through DEV-013 — Issues #723–#733

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
| [STUB] | — | Issues #723–#733 not yet cross-referenced |

**Overall v0.8.0 pass rate: ~36% of test cases within reference ranges.**

A compliance submission is not possible until DEV-004 and DEV-005 are resolved and the pass rate reaches 100% (or documented deviations are accepted by the standards body).
