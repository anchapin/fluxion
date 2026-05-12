# ASHRAE 140 Version Declaration

**Status:** 🟡 Partial — version identified, formal statement not yet in codebase  
**Last updated:** 2026-05-12

---

## Target Standard

fluxion targets **ASHRAE Standard 140-2023** (*Standard Method of Test for the Evaluation of Building Energy Analysis Computer Programs*).

## Version History in Codebase

| fluxion version | ASHRAE 140 edition tested against | Notes |
|-----------------|-----------------------------------|-------|
| v0.8.0 | ASHRAE 140-2023 | Reference ranges updated to 2023 multi-program ranges (EnergyPlus, ESP-r, TRNSYS) |
| < v0.8.0 | ASHRAE 140-2017 or unspecified | [STUB — verify historical editions if needed] |

## Key 2023 vs. 2017 Differences Relevant to fluxion

| Area | 2017 | 2023 |
|------|------|------|
| Energy output units | GJ | kWh |
| Reference programs | EnergyPlus, DOE-2, BLAST | EnergyPlus, ESP-r, TRNSYS (updated) |
| High-mass test cases | Present | Present, revised reference ranges |
| Free-float cases | Present | Present |
| Section 8 report format | [check] | Requires peak load date+hour, hourly profiles |

## Required Actions

- [ ] Hard-code `ASHRAE_140_EDITION = "2023"` in validation runner code and report headers
- [ ] Confirm that reference data loaded from `refdata/` matches 2023 multi-program ranges (not 2017)
- [ ] Add edition declaration to Section 8.1 header output (tracked in #749)
