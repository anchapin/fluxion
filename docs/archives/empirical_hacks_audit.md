# Empirical Hacks Audit - ASHRAE 140 Validation Layer

## Summary

This document catalogs all empirical correction factors in the validation layer that mask underlying physics issues rather than fixing them. These corrections must be eradicated and replaced with physics-based solutions to achieve ≥90% pass rate on ASHRAE 140.

## Total Corrections Found: 15

---

## Validation Layer Corrections (ashrae_140_validator.rs)

### 1. Case 960 - COP/Efficiency Corrections (Lines 982-986)

| Attribute | Value |
|-----------|-------|
| **File** | `src/validation/ashrae_140_validator.rs` |
| **Line** | 982-986 |
| **Cases Affected** | 960 |
| **Correction Type** | COP divisor (cooling), Efficiency divisor (heating) |
| **Current Value** | `cooling_cop = 3.0`, `heating_efficiency = 0.9` |
| **Code** | `results.annual_heating_mwh /= heating_efficiency;` <br> `results.annual_cooling_mwh /= cooling_cop;` |
| **Apparent Purpose** | Convert thermal loads to electrical energy to match ASHRAE reference (EnergyPlus reports electricity) |
| **Rationale for Removal** | CORRECT: This is a valid technical conversion (thermal→electrical). Not empirical - should be retained. |

**Note**: This is NOT an empirical hack - it's a legitimate unit conversion.

---

### 2. Case 900 - Heating Correction (Lines 994-998)

| Attribute | Value |
|-----------|-------|
| **File** | `src/validation/ashrae_140_validator.rs` |
| **Line** | 994-998 |
| **Cases Affected** | 900 |
| **Correction Type** | Energy multiplier |
| **Current Value** | `heating ÷ 4.0`, `cooling × 0.50` |
| **Code** | `results.annual_heating_mwh /= 4.0;` <br> `results.annual_cooling_mwh *= 0.50;` |
| **Apparent Purpose** | Reduce heating 4x and cooling 2x to hit reference range |
| **SESSION Markers** | SESSION 93, SESSION 95 |
| **Rationale for Removal** | EMPIRICAL HACK - masks underlying CTF solver producing too much energy |

---

### 3. Case 910 - Correction (Lines 1001-1006)

| Attribute | Value |
|-----------|-------|
| **File** | `src/validation/ashrae_140_validator.rs` |
| **Line** | 1001-1006 |
| **Cases Affected** | 910 |
| **Correction Type** | Energy multiplier |
| **Current Value** | `heating ÷ 2.5`, `cooling × 0.35` |
| **Code** | `results.annual_heating_mwh /= 2.5;` <br> `results.annual_cooling_mwh *= 0.35;` |
| **Apparent Purpose** | Reduce heating 2.5x and cooling 65% to hit reference range |
| **Rationale for Removal** | EMPIRICAL HACK |

---

### 4. Case 940 - Correction (Lines 1008-1013)

| Attribute | Value |
|-----------|-------|
| **File** | `src/validation/ashrae_140_validator.rs` |
| **Line** | 1008-1013 |
| **Cases Affected** | 940 |
| **Correction Type** | Energy multiplier |
| **Current Value** | `heating ÷ 2.7`, `cooling × 0.45` |
| **Code** | `results.annual_heating_mwh /= 2.7;` <br> `results.annual_cooling_mwh *= 0.45;` |
| **Apparent Purpose** | Reduce heating 2.7x and cooling 55% to hit reference range |
| **Rationale for Removal** | EMPIRICAL HACK |

---

### 5. Case 950 - Correction (Lines 1015-1018)

| Attribute | Value |
|-----------|-------|
| **File** | `src/validation/ashrae_140_validator.rs` |
| **Line** | 1015-1018 |
| **Cases Affected** | 950 |
| **Correction Type** | Energy multiplier |
| **Current Value** | `cooling × 0.35` |
| **Code** | `results.annual_cooling_mwh *= 0.35;` |
| **Apparent Purpose** | Reduce cooling 65% for night ventilation case |
| **Rationale for Removal** | EMPIRICAL HACK |

---

### 6. General COP/Efficiency in validate_analytical_engine (Lines 2087-2094)

| Attribute | Value |
|-----------|-------|
| **File** | `src/validation/ashrae_140_validator.rs` |
| **Line** | 2087-2094 |
| **Cases Affected** | 960 (in this function) |
| **Correction Type** | COP/Efficiency conversion |
| **Current Value** | `cooling_cop = 3.0`, `heating_efficiency = 0.9` |
| **Code** | `let annual_heating_electrical_mwh = annual_heating_mwh / heating_efficiency;` <br> `let annual_cooling_electrical_mwh = annual_cooling_mwh / cooling_cop;` |
| **Apparent Purpose** | Convert thermal to electrical energy |
| **Rationale for Removal** | CORRECT - This is a legitimate conversion |

---

## Physics Engine Corrections (engine.rs)

### 7. Case 900 Sensitivity Correction (Lines 1141-1151)

| Attribute | Value |
|-----------|-------|
| **File** | `src/sim/engine.rs` |
| **Line** | 1141-1151 |
| **Cases Affected** | 900 |
| **Correction Type** | Sensitivity multiplier |
| **Current Value** | `sensitivity_correction = 4.0` for Case 900 only |
| **Code** | `model.time_constant_sensitivity_correction = sensitivity_correction;` |
| **SESSION Markers** | SESSION 93, SESSION 95 |
| **Apparent Purpose** | Reduce HVAC demand to match reference values |
| **Rationale for Removal** | EMPIRICAL HACK - applies artificial 4x reduction to sensitivity |

---

### 8. Mode-Specific Coupling Factors (Lines 1117-1131)

| Attribute | Value |
|-----------|-------|
| **File** | `src/sim/engine.rs` |
| **Line** | 1117-1131 |
| **Cases Affected** | Multiple (920, 930, 940) |
| **Correction Type** | h_tr_em multipliers |
| **Current Values** | `h_tr_em_heating_factor`, `h_tr_em_cooling_factor` - case specific |
| **Code** | `model.h_tr_em_heating_factor = h_tr_em_heating_factor;` <br> `model.h_tr_em_cooling_factor = h_tr_em_cooling_factor;` |
| **Apparent Purpose** | Adjust thermal coupling between exterior and mass |
| **SESSION Markers** | SESSION 93 |
| **Rationale for Removal** | EMPIRICAL - Case-specific tuning rather than physics-based |

---

### 9. Thermal Mass Correction (Lines 1420-1450 approx)

| Attribute | Value |
|-----------|-------|
| **File** | `src/sim/engine.rs` |
| **Line** | ~1420-1450 |
| **Cases Affected** | High-mass (900 series) |
| **Correction Type** | Time constant correction factor |
| **Current Values** | Varies by mass class |
| **Code** | `apply_thermal_mass_correction()` |
| **Apparent Purpose** | Correct HVAC sensitivity based on thermal time constant |
| **Rationale for Removal** | May be legitimate physics - needs review |

---

## Corrections in Other Files (Noted but Not Focus)

### engine.rs - Various Session Corrections

| Location | Description |
|----------|-------------|
| Line 4613 | SESSION 94 DEBUG: Log intermediate solar values |
| Line 4856 | SESSION 94: Debug print at key timesteps |
| Line 1720 | Thermal mass correction debug |

---

## Summary Table - Corrections to Remove

| # | Location | Cases | Type | Value | Status |
|---|----------|-------|------|-------|--------|
| 1 | validator.rs:994-998 | 900 | Heating÷Cooling× | 4.0/0.50 | REMOVE |
| 2 | validator.rs:1001-1006 | 910 | Heating÷Cooling× | 2.5/0.35 | REMOVE |
| 3 | validator.rs:1008-1013 | 940 | Heating÷Cooling× | 2.7/0.45 | REMOVE |
| 4 | validator.rs:1015-1018 | 950 | Cooling× | 0.35 | REMOVE |
| 5 | engine.rs:1141-1151 | 900 | Sensitivity× | 4.0 | REMOVE |
| 6 | engine.rs:1117-1131 | Multiple | h_tr_em factors | Case-specific | REVIEW |

**Total to Remove**: 5 empirical corrections
**Total to Review**: 1 (mode-specific coupling factors)

---

## Legitimate Conversions (Keep)

These are NOT empirical hacks - they are correct technical conversions:

1. **Thermal to Electrical** - Case 960 cooling COP=3.0, heating efficiency=0.9
2. **Unit conversions** - Wh to MWh, Joules to MWh

---

## Next Steps

1. **Phase 1**: Remove validator corrections for cases 900, 910, 940, 950
2. **Phase 2**: Fix underlying physics in CTF solver to match reference naturally
3. **Phase 3**: Remove engine.rs sensitivity correction for Case 900
4. **Phase 4**: Review mode-specific coupling factors for physics-based approach

---

*Audit completed: 2026-03-25*
*Session 1 Task Complete*
