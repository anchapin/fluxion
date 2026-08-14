# ASHRAE 140 Annual Re-Validation Schedule

## Overview

This document describes the annual re-validation process for ASHRAE Standard 140
building energy simulation validation. The re-validation ensures that Fluxion's
ASHRAE 140 blind validation pass rate is maintained above the 60% threshold
(matching `release_gates.yaml → validation.min_pass_rate` and `SCORECARD.md`)
as code evolves.

## Schedule

**Re-validation Window:** Every January (January 1 - January 31)
**Last Day of Validation:** January 31
**Responsible Team:** Fluxion Validation Team

## Prerequisites

Before running re-validation:

1. Access to latest ASHRAE 140 reference data from official sources
2. Valid EnergyPlus installation (version 23.2 or later)
3. Access to ESP-r and TRNSYS reference implementations (optional)
4. All CI workflows passing on main branch

## Process

### Phase 1: Reference Data Update

1. **Source latest ASHRAE 140 reference data:**
   - EnergyPlus: https://energyplus.net/downloads (NREL)
   - ESP-r: University of Strathclyde distribution
   - TRNSYS: Thermal Energy System Specialists

2. **Update data directory:**
   ```bash
   # Update EnergyPlus reference data
   cp -r /path/to/latest/energyplus/reference/data \
     tests/reference_data/energyplus/

   # Update ESP-r reference data
   cp -r /path/to/latest/esp-r/reference \
     tests/reference_data/esp-r/

   # Update TRNSYS reference data
   cp -r /path/to/latest/trnsys/reference \
     tests/reference_data/trnsys/
   ```

3. **Document reference data version in `tests/reference_data/ashrae140/versions.json`**
   (the closest equivalent to `docs/ashrae_140/reference_data_versions.md`; the
   latter path does not exist in this repository — see Issue #2864.)

### Phase 2: Run Full Validation Suite

1. **Execute validation:**
   ```bash
   cargo test --test ashrae_140_validation --release
   ```

2. **Collect results:**
   - Pass rate
   - Failed cases list
   - Mean Absolute Error (MAE)
   - Per-case heating/cooling energy values

3. **Generate validation report:**
   ```bash
   ./scripts/annual_ashrae_revalidation.sh --year YYYY --report-only
   ```
   (the actual script is `scripts/annual_ashrae_revalidation.sh` — see Phase 4
   below. The previously-cited `./scripts/generate_ashrae_report.sh` does not
   exist in this repository — see Issue #2864.)

### Phase 3: Analysis and Sign-off

1. **Review failed cases:**
   - Document root cause for each failure
   - Determine if failure is due to:
     - Code regression (needs fixing)
     - Reference data update (acceptable drift)
     - Known limitation (document and accept)

2. **Calculate metrics:**
   - Overall pass rate (must be >= 60% per `release_gates.yaml → validation.min_pass_rate`)
   - Mean Absolute Error (must be ≤ 50% per `release_gates.yaml → validation.max_mae`)
   - Comparison with previous year's results

3. **Sign-off requirements:**
   - [ ] All critical regressions fixed or accepted
   - [ ] Pass rate >= 60%
   - [ ] Mean Absolute Error ≤ 50%
   - [ ] Validation report generated and archived
   - [ ] Reference data versions documented

### Phase 4: Milestone Completion

1. **Create milestone:**
   ```bash
   gh milestone create "ASHRAE-140-YYYY-Annual-Revalidation" \
     --description "Annual ASHRAE 140 re-validation for YYYY" \
     --due-date YYYY-02-15
   ```

2. **Close issues and update documentation:**
   - Update this document with any process changes
- Archive validation report under `tests/reference_data/ashrae140/annual_reports/`
      (the previously-cited `docs/ashrae_140/annual_reports/` directory does not
      exist in this repository — see Issue #2864.)
    - Update `ARCHITECTURE.md` if module interfaces changed

## Automated Script

The annual re-validation process can be automated using:

```bash
./scripts/annual_ashrae_revalidation.sh --year YYYY --dry-run
```

See `scripts/annual_ashrae_revalidation.sh` for full documentation.

## Success Criteria

| Criterion | Target | Acceptable Range | Source of Truth |
|-----------|--------|------------------|-----------------|
| Pass Rate | >= 60% | 60-100% | `release_gates.yaml` `validation.min_pass_rate: 0.60` |
| Mean Absolute Error | <= 50% | 0-50% | `release_gates.yaml` `validation.max_mae: 50` |
| Failed Cases | within `extreme_deviation_limit: 2` | 0-2 cases | `release_gates.yaml` `validation.individual.known_failures: ["600","900"]` |

## Failed Case Handling

### If Pass Rate < 60%

1. **Immediate action:**
   - Block all PRs to main until resolved
   - Create issue for each failed case
   - Prioritize fixes by severity

2. **Root cause categories:**
   - **Regression:** Code change caused failure → revert or fix
   - **Reference Drift:** Reference data changed → update tolerance or document
   - **Physics Limitation:** Model limitation → document as known issue

### If Pass Rate >= 60%

1. **Archive results**
2. **Update status dashboard**
3. **Proceed with normal development**

## Reference Sources

| Source | Organization | URL |
|--------|--------------|-----|
| EnergyPlus | NREL | https://energyplus.net |
| ESP-r | University of Strathclyde | https://www.esru.strath.ac.uk/ESP-r |
| TRNSYS | TESS | https://www.trnsys.com |

## Change Log

| Date | Change | Author |
|------|--------|--------|
| YYYY-MM-DD | Initial version | - |

## See Also

- [ASHRAE 140 Mathematical Model](ashrae_140/mathematical_model.md)
- [ASHRAE 140 Test Cases](../src/validation/ashrae140/cases/)
- [CI Gate Workflow](../.github/workflows/ashrae_validation.yml)
- [Validation Results Dashboard](https://github.com/anchapin/fluxion/actions/workflows/ashrae_validation.yml)