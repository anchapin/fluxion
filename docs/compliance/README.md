# ASHRAE 140 Compliance Documentation

**Product:** fluxion v0.8.0+
**Standard:** ASHRAE Standard 140-2023
**Scope:** Building Thermal Envelope and Fabric Load Tests
**Issue:** #750
**Last Updated:** 2026-06-16

---

## Overview

This directory contains the documentation required for a formal ASHRAE 140 compliance submission. The documentation is organized according to the requirements specified in ASHRAE 140-2023 Section 1.5 (Compliance) and Section 6 (Software Quality Testing).

**Current Compliance Status:** `NOT SUBMISSION-READY`

Multiple P1 open issues prevent formal compliance certification. See `deviations-register.md` for the full list.

---

## Required Documents

| # | Document | File | Status | Notes |
|---|----------|------|--------|-------|
| 1 | Software Qualification Test (SQT) Report | `SQT-report.md` | **DRAFT** | Describes numerical methods, timestep, convergence criteria, limitations |
| 2 | Standards Version Declaration | `ashrae140-version.md` | **DRAFT** | Declares ASHRAE 140-2023 as the target standard |
| 3 | Weather File Provenance Statement | `weather-file-provenance.md` | **DRAFT** | Documents exact file, source URL, SHA256 hash |
| 4 | Inter-Program Comparison Charts | `inter-program-comparison.md` | **DRAFT** | Bar charts showing fluxion vs reference programs |
| 5 | Known Deviations Register | `deviations-register.md` | **ACTIVE** | Table of deliberate deviations with justification |
| 6 | Reproducible Test Execution Log | **TODO** | CI-generated versioned log for compliance audit |

---

## Submission Checklist

### Pre-Submission Requirements

Before making any ASHRAE 140 compliance claim, verify:

#### P1 Blockers — Must Resolve Before Submission

- [ ] **[DEV-001]** Free-float HVAC not disabled — HVAC produces unphysical zone temperatures (~125°C)
  - Issue: #725
  - Status: OPEN

- [ ] **[DEV-004]** Synthetic weather instead of Annex C TMY — all 64 validation metrics affected
  - Issue: #732
  - Status: OPEN
  - Weather file exists in repo: `assets/weather/USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw`
  - SHA256: `ce78fcda675cdde480025cda2fde5444b7346e81c1d521d05189dcbf70794224`

- [ ] **[DEV-008]** No single source of truth for material properties — root cause of multiple deviations
  - Issue: #755
  - Status: OPEN

#### P1 Issues — Spec Complete, Awaiting Implementation

- [ ] **[DEV-002]** 900-series concrete wrong thermal properties in CTF solver (k/ρ/Cp)
  - Issue: #730, #735, #752
  - Status: SPEC ONLY
  - Fix in: `v1.3-physics-spec.md`

- [ ] **[DEV-006]** Surface film coefficients wrong in all code locations (h_int=8.29, h_ext=29.3)
  - Issue: #733, #753
  - Status: SPEC ONLY

- [ ] **[DEV-009]** CTF surface area = 97.2 m² (should be 63.6 m²) for 900-series
  - Issue: #756
  - Status: SPEC ONLY

### Documentation Requirements

- [ ] All 6 required documents created and reviewed
- [ ] SQT Report signed by qualified software engineer
- [ ] Standards Version Declaration confirmed
- [ ] Weather file SHA256 hash verified against actual file
- [ ] Inter-program comparison charts generated from actual validation runs
- [ ] Reproducible test execution log captured from CI

### Validation Requirements

- [ ] All 64 ASHRAE 140-2023 validation metrics pass within acceptance tolerance
- [ ] Pass rate ≥ 95% (target: 18/18 cases × multiple metrics)
- [ ] Mean Absolute Error (MAE) < 5% across all metrics
- [ ] No systematic biases identified in delta analysis

### Code Quality Requirements

- [ ] No P1 open issues
- [ ] No SPEC-ONLY items awaiting implementation
- [ ] All P2 items triaged and scheduled
- [ ] Code review completed for all deviations
- [ ] ARCHITECTURE.md updated to reflect as-built implementation

---

## Document Descriptions

### 1. SQT-report.md — Software Qualification Test Report

Describes fluxion's numerical methods for ASHRAE 140 compliance:

- Heat conduction solvers (CTF, FD, 5R1C RC-network)
- Solar radiation calculation method
- Ventilation and infiltration modeling
- Zone energy balance approach
- Timestep selection (1-hour external, variable sub-timestep internal)
- Convergence criteria
- Known limitations

### 2. ashrae140-version.md — Standards Version Declaration

Formally declares the ASHRAE 140 edition:

- Edition: ASHRAE 140-2023
- Version-specific notes (vs 140-2017)
- Section 5.2.2 solar distribution change notes
- Version governance policy
- Migration path for future editions

### 3. weather-file-provenance.md — Weather File Provenance Statement

Documents the required weather file for ASHRAE 140:

- Required file: Denver Stapleton TMY (WMO# 724690)
- SHA256 hash: `ce78fcda675cdde480025cda2fde5444b7346e81c1d521d05189dcbf70794224`
- Current status: Synthetic weather in use (DEV-004)
- Required code changes to use actual TMY file

### 4. inter-program-comparison.md — Inter-Program Comparison Charts

Describes the chart generation infrastructure:

- Reference programs: EnergyPlus, ESP-r, TRNSYS, DOE-2
- Chart types required for compliance
- Implementation status in `src/validation/report.rs`
- Known gaps: reference program bars not plotted, y-axis hardcoded

### 5. deviations-register.md — Known Deviations Register

Active register of all known deviations from ASHRAE 140-2023:

- 17 deviations documented (DEV-001 through DEV-017)
- Status: OPEN, IN PROGRESS, SPEC ONLY, RESOLVED
- Priority: P1 (blocker) and P2
- Wave assignment for phased resolution

### 6. Reproducible Test Execution Log — TODO

**Not yet created.** Requires CI pipeline enhancement:

- GitHub Actions workflow: `ashrae_140_validation.yml`
- Artifacts: `validation_output.txt`, `validation_results.json`, `ASHRAE_140_VALIDATION_REPORT.md`
- Required: versioned, timestamped execution log with git SHA

---

## Directory Structure

```
docs/compliance/
├── README.md                          # This file — submission checklist
├── SQT-report.md                      # Software Qualification Test Report
├── ashrae140-version.md               # Standards Version Declaration
├── weather-file-provenance.md         # Weather File Provenance Statement
├── inter-program-comparison.md        # Inter-Program Comparison Charts
├── deviations-register.md              # Known Deviations Register
├── standards-roadmap.md               # Broader standards roadmap (supplemental)
└── section8-output-gaps.md           # Section 8 output gaps (supplemental)
```

---

## Related Issues

| Issue | Title | Status |
|-------|-------|--------|
| #750 | Missing compliance documentation for ASHRAE 140 | IN PROGRESS (this PR) |
| #751 | Standards certification roadmap | IN PROGRESS |
| #732 | Synthetic weather instead of Annex C TMY | OPEN |
| #725 | Free-float HVAC not disabled | OPEN |
| #755 | No single source of truth for material properties | OPEN |
| #667 | ASHRAE 140 reference data | IN PROGRESS |

---

## How to Verify Weather File Hash

```bash
# Verify the Denver TMY file hash
cd /path/to/fluxion
sha256sum assets/weather/USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw
# Expected: ce78fcda675cdde480025cda2fde5444b7346e81c1d521d05189dcbf70794224
```

---

## CI Pipeline

The ASHRAE 140 validation runs in CI via:

- **Workflow:** `.github/workflows/ashrae_140_validation.yml`
- **Schedule:** Nightly at 2 AM UTC + on every push to main/develop + PR
- **Artifacts:** Uploaded for 30 days
  - `validation_output.txt` — raw test output
  - `validation_results.json` — structured results
  - `ASHRAE_140_VALIDATION_REPORT.md` — generated report
  - `gate_status.json` — release gate status

---

## Contact

For questions about this documentation, contact the Building Standards Engineer or open a GitHub issue tagged with `ashrae-140` and `documentation`.

---

*Last reviewed: 2026-06-16*
