# fluxion ASHRAE 140-2023 Deviations Register

**Standard:** ASHRAE Standard 140-2023 (Building Thermal Envelope and Fabric Load Tests)  
**Product:** fluxion v0.8.0+  
**Maintained by:** Building Standards Engineer  
**Last Updated:** 2026-05-12  
**Scope:** All known deviations from ASHRAE 140-2023 normative requirements discovered during systematic gap analysis (Issues #723–#756)

---

## Status Legend

| Status | Meaning |
|--------|---------|
| OPEN | Deviation exists, no fix in progress |
| IN PROGRESS | Fix spec written, implementation underway |
| RESOLVED | Fix merged and verified |
| SPEC ONLY | Spec complete, awaiting Wave implementation |

---

## Deviations Register

### DEV-001 — Free-Float Mode: HVAC Not Disabled
**Standard Ref:** ASHRAE 140-2023 Section 7, Cases 600FF/650FF/900FF/950FF  
**GitHub Issue:** #725  
**Description:** Free-float test cases require HVAC to be completely disabled. Current code leaves HVAC active, producing unphysical zone temperatures (~125°C vs. expected ~42–65°C). All four FF case metrics fail.  
**Impact:** 4 cases × 2 metrics = 8 validation metrics failing  
**Status:** OPEN  
**Priority:** P1 (Wave 1)

---

### DEV-002 — 900-Series Concrete: Wrong Thermal Properties (CTF Solver)
**Standard Ref:** ASHRAE 140-2023 Annex B Table B1-3  
**GitHub Issues:** #730, #735, #752  
**Description:** `src/physics/ctf_solver.rs` defines 900-series concrete as k=1.4 W/mK, ρ=2300 kg/m³, Cp=880 J/kgK (normal-weight concrete). ASHRAE 140 specifies medium-density block: k=0.51, ρ=1400, Cp=840. Three independent errors in one layer.  
**Correct Values:** k=0.51 W/mK, ρ=1400 kg/m³, Cp=840 J/kgK, d=0.200 m  
**Impact:** All 900-series cases. Inflates thermal capacitance by ~64%, wrong CTF coefficients, corrupts FD solver when promoted.  
**Status:** SPEC ONLY (Wave 1 — spec in v1.3-physics-spec.md)  
**Priority:** P1

---

### DEV-003 — Solver Routing: CTF Primary Instead of FD
**Standard Ref:** ASHRAE 140-2023 Annex B — high-mass wall time constant criterion  
**GitHub Issue:** #726  
**Description:** CTF solver is the primary path for all cases; FD is only a fallback for extreme cases. ASHRAE 140 high-mass cases (κ > 20,000 J/m²K) require FD. The routing threshold uses wrong material properties (DEV-002), causing CTF to be selected accidentally for 900-series. Fix order: DEV-002 first, then re-evaluate routing.  
**Impact:** All 900-series cases use wrong solver type.  
**Status:** OPEN (blocked on DEV-002)  
**Priority:** P1 (Wave 2)

---

### DEV-004 — Weather Data: Synthetic Approximation Instead of ASHRAE 140 Annex C TMY
**Standard Ref:** ASHRAE 140-2023 Annex C (Weather Data)  
**GitHub Issue:** #732  
**Description:** Current code generates weather data from sine/cosine approximations of temperature and solar. ASHRAE 140 requires the Denver Stapleton TMY file specified in Annex C. This affects all 64 validation metrics (energy and temperature depend on correct hourly weather).  
**Impact:** All cases. This is a compliance blocker — no certification submission is possible until resolved.  
**Status:** OPEN  
**Priority:** P1 (compliance blocker)

---

### DEV-005 — Output Reporting: Incomplete / Wrong Units
**Standard Ref:** ASHRAE 140-2023 Section 5.3 (Output Requirements)  
**GitHub Issue:** #749  
**Description:** Output reports are incomplete: wrong units (MWh vs kWh in some paths), missing peak load timestamps, missing hourly free-float temperature profiles, no incident solar radiation output. Section 5.3 specifies mandatory outputs for compliance submission.  
**Impact:** Compliance report cannot be submitted even if physics pass.  
**Status:** OPEN  
**Priority:** P2

---

### DEV-006 — Surface Film Coefficients: Wrong Values in All Code Locations
**Standard Ref:** ASHRAE 140-2023 Annex B Table B1-6  
**GitHub Issues:** #733, #753  
**Description:** h_int=8.0 (should be 8.29 W/m²K) and h_ext=25.0 (should be 29.3 W/m²K). Root cause in `docs/PHYSICAL_CONSTANTS.md` which specifies wrong value (18.3 W/m²K at 3 m/s wind; correct is 29.3 at 6.7 m/s). Three code locations contain conflicting values.  
**Correct Values:** h_int=8.29 W/m²K, h_ext=29.3 W/m²K (combined combined convective+radiative, fixed coefficient per standard)  
**Impact:** All cases — affects all U-value calculations and heat balance.  
**Status:** SPEC ONLY (Wave 1)  
**Priority:** P1

---

### DEV-007 — 600-Series Material Properties: Wood Siding and Gypsum Errors
**Standard Ref:** ASHRAE 140-2023 Annex B Table B1-3  
**GitHub Issue:** #754  
**Description:** `construction.rs` lightweight materials have wrong values: wood siding Cp=1300 (should be 900 J/kgK), gypsum ρ=950 (should be 784 kg/m³). Minor impact on lightweight cases but introduces inconsistency with standard.  
**Impact:** Cases 600–650. Dynamic response errors, minor annual load effect.  
**Status:** SPEC ONLY (Wave 1)  
**Priority:** P2

---

### DEV-008 — No Single Source of Truth for Material Properties
**Standard Ref:** ASHRAE 140-2023 Annex B Table B1-3 (all material specs)  
**GitHub Issue:** #755  
**Description:** Material properties are defined independently in three locations (`construction.rs`, `ctf_solver.rs`, `PHYSICAL_CONSTANTS.md`) with conflicting values. No shared constants module. Any fix to one location does not propagate to others.  
**Impact:** Risk of partial fixes re-introducing inconsistencies. Root cause of DEV-002 and DEV-006 cascades.  
**Status:** OPEN  
**Priority:** P1 (architectural, must fix before Wave 1 PRs)

---

### DEV-009 — CTF Solver Geometry: Wrong Surface Area (900-Series)
**Standard Ref:** ASHRAE 140-2023 Annex B Section B1.1 (geometry)  
**GitHub Issue:** #756  
**Description:** `ctf_solver.rs` Case 900 uses `surface_area = 97.2 m²`. Correct net opaque wall area is 63.6 m² (75.6 m² gross − 12.0 m² windows). The 97.2 value corresponds to a 9m×9m plan (wrong geometry, 53% too large).  
**Correct Value:** 63.6 m²  
**Impact:** All 900-series cases in CTF path. Overestimates heat transfer by ~53%.  
**Status:** SPEC ONLY (Wave 1)  
**Priority:** P1

---

### DEV-010 — Solar Distribution: Wrong Standard Reference (ISO 13790 vs ASHRAE 140 §5.2.2)
**Standard Ref:** ASHRAE 140-2023 Section 5.2.2  
**GitHub Issues:** #729, #745  
**Description:** Issue #729 was drafted referencing ISO 13790 Table C.2 (European monthly RC-network method). ASHRAE 140 §5.2.2 requires 100% of transmitted solar distributed to opaque interior surfaces by area × absorptance, with zero to the air node. ISO 13790 fractions are not applicable here.  
**Correct Formula:** φᵢ = Aᵢ / ΣAⱼ (for uniform absorptance α=0.6)  
**Impact:** Cases 610, 620, 630, 910–930. Wrong solar distribution affects annual loads and peak temperatures.  
**Status:** IN PROGRESS (spec posted to #729, title corrected)  
**Priority:** P1 (Wave 3)

---

### DEV-011 — Shading: Overhang+Fin Overlap Double-Counted
**Standard Ref:** ASHRAE 140-2023 Section 8.1.2 (Shading)  
**GitHub Issue:** #747  
**Description:** When both horizontal overhang and vertical fins are applied (Cases 630/930), the overlap region of shaded area is counted twice in the geometric shadow calculation, over-shading the window and under-predicting solar gains.  
**Impact:** Cases 630, 930. Annual cooling under-predicted; annual heating over-predicted.  
**Status:** OPEN  
**Priority:** P2 (Wave 3)

---

### DEV-012 — Warm-Up Period: No Multi-Year Pre-Conditioning
**Standard Ref:** ASHRAE 140-2023 Section 1.4 (Steady Periodic Condition)  
**GitHub Issue:** #744  
**Description:** Simulation starts from cold initial conditions (T_zone = T_outdoor at hour 0). ASHRAE 140 requires steady-periodic conditions (typically achieved by running 2–3 years until annual energy converges to <0.1% change). High-mass cases (900-series) are particularly sensitive to initial conditions.  
**Impact:** 900-series cases biased by cold-start energy. Heating load over-predicted in first simulated year.  
**Status:** OPEN  
**Priority:** P2 (Wave 2)

---

### DEV-013 — Validator Comparator: Midpoint±Tolerance Instead of [min, max] Band
**Standard Ref:** ASHRAE 140-2023 Section 1.5 (Compliance)  
**GitHub Issue:** #723  
**Description:** `validate_energy_against_reference()` uses `midpoint ± (half_range × 1.15)` instead of direct `[ref_min, ref_max]` comparison. This artificially widens the acceptance band and passes results that should fail. `validate_peak_load_against_reference()` correctly uses `[min, max]` — energy must be made consistent.  
**Fix:** Replace energy comparator with same pattern as peak load comparator; remove ±15%/±10% tolerance constants.  
**Impact:** Systematic false-pass rate. Pass rate will initially drop when fixed (correctly exposing true failures).  
**Status:** IN PROGRESS (exact fix spec posted to #723)  
**Priority:** P1 (Wave 3 gate)

---

### DEV-014 — Reference Data: No Source Provenance
**Standard Ref:** ASHRAE 140-2023 Section 1.5 (Compliance documentation)  
**GitHub Issues:** #667, #748  
**Description:** All reference values in `reference_data.rs` are hardcoded with the comment "calibrated for 5R1C model" — circular validation against the model being tested. Values must come from the published ASHRAE 140 inter-program ensemble (Tables 7-2 through 8-2). Provenance schema (source standard, source table, source program, provisional flag) must be added.  
**Status:** IN PROGRESS (PR #773 adds provisional data from NREL/TP-472-6231; ASHRAE 140-2023 Tables 7-2 to 8-2 still needed for compliance)  
**Priority:** P1 (Wave 5)

---

### DEV-015 — Placeholder Conductance Values in Window/Shading Code
**Standard Ref:** ASHRAE 140-2023 Annex B Table B1-5  
**GitHub Issue:** #731  
**Description:** Window and shading-related conductance values in the codebase include placeholder magic numbers not traceable to ASHRAE 140 Annex B. Need systematic audit and replacement with standard values.  
**Impact:** Cases involving windows (all qualification cases).  
**Status:** OPEN  
**Priority:** P2

---

### DEV-016 — Ground Temperature Boundary Condition Missing
**Standard Ref:** ASHRAE 140-2023 Annex B Section B1.1 (floor boundary condition)  
**GitHub Issue:** #746  
**Description:** Floor ground boundary condition is not correctly implemented as adiabatic for the Section 7 cases. Code applies a non-zero resistance (R=0.17 m²K/W) to the floor boundary, introducing spurious heat loss.  
**Impact:** All cases — small systematic error in annual heating.  
**Status:** OPEN  
**Priority:** P2

---

### DEV-017 — Compliance Documentation: SQT Report and Deviations Register Missing
**Standard Ref:** ASHRAE 140-2023 Section 1.5 and Section 6 (Software Quality Testing)  
**GitHub Issue:** #750  
**Description:** ASHRAE 140 compliance requires: (a) Software Quality Testing (SQT) report documenting test methodology, (b) formal deviations register (this document), (c) weather data provenance documentation, (d) inter-program comparison charts. None of these exist.  
**Status:** IN PROGRESS (this register is the start; others pending)  
**Priority:** P2 (Wave 5)

---

## Summary Table

| DEV | Issue(s) | Description | Status | Priority | Wave |
|-----|----------|-------------|--------|----------|------|
| DEV-001 | #725 | Free-float HVAC not disabled (125°C temps) | OPEN | P1 | 1 |
| DEV-002 | #730, #735, #752 | 900-series concrete wrong k/ρ/Cp in CTF | SPEC ONLY | P1 | 1 |
| DEV-003 | #726 | CTF routing over FD for high-mass cases | OPEN | P1 | 2 |
| DEV-004 | #732 | Synthetic weather instead of Annex C TMY | OPEN | P1 | blocker |
| DEV-005 | #749 | Output reporting incomplete/wrong units | OPEN | P2 | 5 |
| DEV-006 | #733, #753 | h_int/h_ext wrong in all 3 code locations | SPEC ONLY | P1 | 1 |
| DEV-007 | #754 | 600-series wood siding/gypsum Cp/ρ errors | SPEC ONLY | P2 | 1 |
| DEV-008 | #755 | No single source of truth for materials | OPEN | P1 | arch |
| DEV-009 | #756 | CTF surface_area=97.2 m² (should be 63.6) | SPEC ONLY | P1 | 1 |
| DEV-010 | #729, #745 | Solar distribution: ISO 13790 → ASHRAE 140 §5.2.2 | IN PROGRESS | P1 | 3 |
| DEV-011 | #747 | Shading overlap double-counted (Cases 630/930) | OPEN | P2 | 3 |
| DEV-012 | #744 | No warm-up/pre-conditioning period | OPEN | P2 | 2 |
| DEV-013 | #723 | Validator comparator midpoint±tol → [min,max] | IN PROGRESS | P1 | 3 gate |
| DEV-014 | #667, #748 | Reference data no provenance (circular validation) | IN PROGRESS | P1 | 5 |
| DEV-015 | #731 | Placeholder conductance values in window/shading | OPEN | P2 | 3 |
| DEV-016 | #746 | Floor ground BC not adiabatic | OPEN | P2 | 2 |
| DEV-017 | #750 | Compliance documentation missing (SQT, charts) | IN PROGRESS | P2 | 5 |

**Open P1 blockers:** DEV-001, DEV-003, DEV-004, DEV-008  
**Spec-ready P1 (Wave 1):** DEV-002, DEV-006, DEV-007, DEV-009  
**In-progress:** DEV-010, DEV-013, DEV-014, DEV-017
