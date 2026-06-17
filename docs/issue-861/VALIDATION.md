# Issue #861: ASHRAE 140 Case 900 Validation with Multi-Node HVAC

## Summary

This document records the validation results for ASHRAE 140 Case 900 (high-mass building with HVAC) using the multi-node thermal model (9R4C thermal network).

**Date**: 2026-06-16
**Status**: Validation reveals significant deviations from ASHRAE 140 reference ranges

## Reference Ranges (ASHRAE 140-2023)

| Metric | Reference Range |
|--------|----------------|
| Annual Heating | 1.17 - 2.04 MWh |
| Annual Cooling | 2.13 - 3.67 MWh |
| Peak Heating | 1.10 - 2.10 kW |
| Peak Cooling | 2.10 - 3.50 kW |
| 900FF Min Temp | -6.40 to -1.60°C |
| 900FF Max Temp | 41.80 - 46.40°C |

## Validation Results

### Case 900 (High-Mass with HVAC)

| Metric | Calculated | Reference Range | Status |
|--------|-----------|----------------|--------|
| Annual Heating | 2.94 MWh | 1.17 - 2.04 MWh | FAIL |
| Annual Cooling | 0.23 MWh | 2.13 - 3.67 MWh | FAIL |
| Peak Heating | 1.02 kW | 1.10 - 2.10 kW | PASS |
| Peak Cooling | 0.29 kW | 2.10 - 3.50 kW | FAIL |

**Key Issues:**
- Annual cooling is severely under-predicted (0.23 MWh vs 2.13-3.67 MWh expected)
- Annual heating is over-predicted (2.94 MWh vs 1.17-2.04 MWh expected)
- Peak cooling is under-predicted (0.29 kW vs 2.10-3.50 kW expected)

### Case 900FF (Free-Floating)

| Metric | Calculated | Reference Range | Status |
|--------|-----------|----------------|--------|
| Min Temperature | -3.54°C | -6.40 to -1.60°C | MARGINAL |
| Max Temperature | 35.01°C | 41.80 - 46.40°C | FAIL |

**Key Issues:**
- Max temperature is under-predicted (35.01°C vs 41.80-46.40°C expected)
- The building thermal mass is damping temperature swings too aggressively

## Diagnostic Observations

1. **Zone Temperature Range**: 20.00°C - 27.11°C (setpoint range)
   - The zone stays within setpoints too easily due to excessive thermal mass damping

2. **Heating/Cooling Hours**:
   - Heating hours: 4868 (55.6% of year)
   - Cooling hours: 1749 (20.0% of year)
   - This suggests the building heats more than expected

3. **HVAC Capacity Settings**:
   - Heating capacity: 2100 W
   - Cooling capacity: 100000 W (effectively unlimited)

## Related Issues

- **Issue #862**: Multi-node free-floating temperature validation
- **Issue #863**: Sol-air temperatures for per-surface gains
- **Issue #864**: Per-surface solar gain distribution
- **Issue #865**: Warm-up period for multi-node HVAC

## Test Files

- `tests/case_900_multinode_validation.rs` - Main multi-node validation tests
- `tests/case_900ff_multinode_validation.rs` - Free-floating validation tests
- `tests/ashrae_140_case_900.rs` - Comprehensive Case 900 tests

## Next Steps

1. Investigate why cooling demand is severely under-predicted
2. Review thermal mass calibration for high-mass construction
3. Check HVAC control logic integration
4. Validate solar gain distribution to mass nodes
5. Compare single-node (5R1C) vs multi-node (9R4C) results

## Notes

- This is a **validation task** - documenting current behavior, not modifying physics
- The test infrastructure uses 14-day warm-up per ASHRAE 140 §B2
- Denver TMY weather data is used for the simulation
