# ASHRAE 140 Multi-Zone Validation Results

> **DRAFT — pending Wave 6 / issue #1446** (multi-zone inter-zone coupling).
> Numbers below are the engine's *current* output through the rewritten,
> real physics validator (issue #1407). The strict ±15% CI gate (#1368)
> is now wired to actual engine results instead of the previous
> 12.4-vs-12.5 MWh self-referential placeholder comparison.
>
> The previous version of this document recorded fabricated PASS
> verdicts from a stub that compared two hardcoded numbers against
> themselves — see the "Removed stub" section below.

## Overview

This document presents the validation of Fluxion's multi-zone solver
against ASHRAE 140 Case 960 (two-zone sunspace building), as specified
in ASHRAE 140-2017 §6.4 / 140-2023 Annex B8. The validator now
exercises the full 8760-step physics simulation through
`ASHRAE140Validator::validate_case_960` and compares against the
canonical inter-program envelope published in
`crate::validation::benchmark::CASE_960_*`.

Cases 970 and 980 remain framework stubs (tracked separately from
#1407).

## Validation Environment

### Hardware / Software Configuration
- **Processor**: Developer-machine dependent (CI matrix)
- **Fluxion Version**: post-#1407
- **Weather file**: `assets/weather/USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw`
- **ASHRAE 140 Reference**: EnergyPlus, ESP-r, TRNSYS, DOE2, BSIMAC, CSE,
  DeST (Std140_TF_Results.pdf, TESS 19-Aug-2024) — sourced via
  `crate::validation::benchmark::get_benchmark_data("960")` and
  `tests/reference_data/zone_balance/case_960_energy_reference.csv`.

### Tolerance Configuration
- **Annual Energy**: ±15% (ASHRAE 140 compliance band; #1368)
- **Peak Loads**: ±10% (ASHRAE 140 compliance band)
- **Temperature Profiles**: ±1.0 °C (Fluxion internal standard)

## Case 960 Reference Data

The validator uses the ASHRAE 140-2023 inter-program envelope published
in `data/ashrae140_reference.json` and re-exported through
`validation::benchmark`:

| Metric                | Reference Min | Reference Max | Midpoint |
|-----------------------|--------------:|--------------:|---------:|
| Annual Heating (MWh)  | 1.65          | 2.45          | 2.050    |
| Annual Cooling (MWh)  | 1.55          | 2.78          | 2.165    |
| Peak Heating (kW)     | 2.00          | 8.00          | 5.000    |
| Peak Cooling (kW)     | 0.00          | 4.00          | 2.000    |

The ±15% / ±10% tolerance gates from issue #1368 are applied on top of
this envelope. For example, the annual heating PASS band is
[1.402, 2.817] MWh.

## Case 960 Results — Current Engine Output

The numbers below come from the rewritten validator running the full
8760-step physics simulation against the Case 960 spec.

| Metric                | Fluxion (actual) | Reference band | Status    |
|-----------------------|-----------------:|----------------|:---------:|
| Annual Heating (MWh)  | 7.47             | [1.65, 2.45]   | ❌ FAIL   |
| Annual Cooling (MWh)  | 0.00             | [1.55, 2.78]   | ❌ FAIL   |
| Peak Heating (kW)     | 1.07             | [2.00, 8.00]   | ❌ FAIL   |
| Peak Cooling (kW)     | 0.00             | [0.00, 4.00]   | ❌ FAIL   |

**Verdict**: 0 of 4 metrics within tolerance. The strict ±15% CI gate
(#1368) now produces a meaningful FAIL for Case 960 — exactly the
behavior the previous stub masked.

### Known gaps (Wave 6 / issue #1446)

The 7.47 MWh heating result is ~265% above the canonical midpoint
2.05 MWh. This is consistent with the known inter-zone coupling gap
documented in issue #1446 / Wave 6: the current `h_tr_iz = 1.5 W/K`
(door-only) under-resolves heat transfer between the conditioned
back-zone and the free-floating sunspace, so the conditioned zone
loses more heat than it should and the heating load is over-predicted.
Cooling reads 0.00 MWh because the same under-resolution prevents the
sunspace from ever getting warm enough to drive a sensible cooling
load on the back-zone.

Until #1446 lands, **Case 960 cannot satisfy the strict ±15% CI gate**
and the validator correctly reports FAIL on every metric. This is the
intended state — the validator is now trustworthy, even though the
underlying multi-zone model is not yet accurate enough to pass.

## Removed stub (issue #1407)

Prior to issue #1407, `MultiZoneValidator::validate_case_960` and its
sibling `validate_case_960_with_validator` did the following:

1. Hardcoded `Case960Reference { annual_heating: 12.4, annual_cooling: 8.7,
   peak_heating: 5.2, peak_cooling: 4.8 }` in
   `ashrae_140_multi_zone.rs::load_case_960_reference_data`.
2. Hardcoded `actual_heating = 12.5` / `actual_cooling = 8.5` /
   `actual_peak_heating = 5.1` / `actual_peak_cooling = 4.9` inside the
   stub.
3. Compared the two hardcoded sets and reported PASS.

This is a self-referential tautology: `|12.5 - 12.4| / 12.4 = 0.81%`
always passed the ±15% gate, even though the underlying engine
produced 7.47 MWh. The 12.4 MWh "reference" was 4-7× larger than every
reference program (EnergyPlus, ESP-r, TRNSYS, DOE2, BSIMAC, CSE, DeST).

The fix replaces:

- `MultiZoneValidator::validate_case_960` — now delegates to
  `ASHRAE140Validator::validate_case_960`, which runs the full
  8760-step simulation through `ThermalModel::step_physics`.
- `MultiZoneValidator::run_multi_zone_validation` — emits actual
  engine outputs (`vrep.annual_heating_mwh`, etc.) into the
  `BenchmarkReport` instead of zeroing them on FAIL.
- `MultiZoneValidator::validate_case_960_with_validator` — same.
- `MultiZoneValidator::run_comprehensive_validation` — same.
- `MultiZoneValidator::export_results_to_csv` — same.
- `Case960Reference::load_case_960_reference_data` — now sources the
  reference midpoints from `benchmark::CASE_960_*` constants.
- `tests/ashrae_140_case_960_sunspace.rs::test_case_960_reference_loading`
  — now asserts the canonical midpoints (2.05 / 2.165 / 5.0 / 2.0)
  instead of the bogus stub values (12.4 / 8.7 / 5.2 / 4.8).
- `src/validation/ashrae_140_multi_zone.rs::test_case_960_validator_runs_real_model_not_stub`
  — new regression test that:
  - Asserts the validator takes > 100 ms (proving real physics is being
    stepped, not the < 1 ms stub).
  - Asserts the validator does **not** fabricate PASS for the current
    (broken) engine output.
  - Asserts `compare_against_reference` returns PASS for the canonical
    midpoint inputs (proving the comparison logic itself is sound).
  - Asserts `compare_against_reference` returns FAIL for the previous
    stub's hardcoded placeholder values (12.4 / 8.7 / 100 / 100).

## Validation Command Reference

```bash
# Run the rewritten multi-zone validator
cargo test -p fluxion --lib ashrae_140_multi_zone

# Run the new regression test specifically
cargo test -p fluxion --lib test_case_960_validator_runs_real_model_not_stub

# Run the full Case 960 integration suite
cargo test -p fluxion --test ashrae_140_case_960_sunspace

# Run the standalone multi-zone validation binary
cargo run -p fluxion --bin run_multi_zone_validation -- case960
```

## Acceptance Criteria (issue #1407)

- [x] `MultiZoneValidator::validate_case_960` rewritten to delegate to
      `ASHRAE140Validator::validate_case_960` (real 8760-step physics).
- [x] Reference data sourced from `validation::benchmark::CASE_960_*`,
      which derives from `data/ashrae140_reference.json` (canonical
      ASHRAE 140-2023 inter-program envelope).
- [x] New regression test `test_case_960_validator_runs_real_model_not_stub`
      enforces timing > 100ms, no fabricated PASS, comparator logic
      correctness.
- [x] `docs/ASHRAE140_MULTI_ZONE_RESULTS.md` regenerated to reflect
      the new validator and the current (FAIL) engine output.
- [ ] Case 960 satisfies the strict ±15% CI gate — deferred to
      Wave 6 / issue #1446 (inter-zone coupling).

## References

- **Code**:
  - `src/validation/ashrae_140_multi_zone.rs` — rewritten
  - `src/validation/ashrae_140_validator.rs` — `validate_case_960` (real)
  - `src/validation/benchmark.rs` — `CASE_960_*` reference constants
  - `tests/ashrae_140_case_960_sunspace.rs` — integration tests
- **Standard**: ASHRAE 140-2017 §6.4 / ASHRAE 140-2023 Annex B8
- **Prior issues**: #1292, #1368, #1396, #1399, #1446

## Change Log

**v1.1.0 (post-#1407)**:
- Replaced the 12.4-vs-12.5 MWh self-referential stub with real
  physics simulation through `ASHRAE140Validator::validate_case_960`.
- Source reference values from `validation::benchmark::CASE_960_*`
  (canonical ASHRAE 140-2023 inter-program envelope).
- Added `Case960InterProgramBounds` and `Case960CompareOutcome` types
  for downstream consumers (CLI / docs).
- Added regression test `test_case_960_validator_runs_real_model_not_stub`.
- Documented current engine output and Wave 6 / #1446 dependency.

**v1.0.0 (pre-#1407)**:
- Initial multi-zone validation scaffolding.
- Stub validator that fabricated PASS by comparing two hardcoded
  numbers.
- Documented as "✅ PASSED" but the verdict was not connected to any
  real engine output.

## License

© 2026 Fluxion Energy Modeling Collective. Licensed under the Apache
License, Version 2.0.