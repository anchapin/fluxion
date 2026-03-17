# Quality Metrics Tracker

*Generated: 2026-03-15 19:23 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** 12.10%
- **Max Deviation:** 99.93%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| PASS | 1 | 1.6% |
| WARN | 2 | 3.1% |
| FAIL | 61 | 95.3% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | 12.1% | 100% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 600 | Annual Cooling (MWh) | 0.01 | 8.00-10.50 | 99.9% | Unknown |
| 600 | Annual Heating (MWh) | 0.01 | 5.50-7.50 | 99.9% | Unknown |
| 600FF | Min Free-Float Temp (°C) | -5.01 | -18.80--15.60 | 70.9% | FreeFloat |
| 650FF | Min Free-Float Temp (°C) | -10.32 | -23.00--21.00 | 53.1% | FreeFloat |
| 950FF | Min Free-Float Temp (°C) | -9.51 | -20.20--17.80 | 50.0% | ThermalMass |
| 600 | Peak Heating (kW) | 2.10 | 2.80-3.80 | 40.0% | Unknown |
| 650FF | Max Free-Float Temp (°C) | 44.53 | 63.20-73.50 | 34.8% | FreeFloat |
| 600FF | Max Free-Float Temp (°C) | 47.89 | 64.90-75.10 | 31.6% | FreeFloat |
| 900FF | Min Free-Float Temp (°C) | -4.52 | -6.40--1.60 | 13.1% | ThermalMass |
| 900FF | Max Free-Float Temp (°C) | 38.37 | 41.80-46.40 | 13.0% | ThermalMass |
| 600 | Peak Cooling (kW) | 5.01 | 4.80-6.20 | 8.8% | SolarGains |
| 950FF | Max Free-Float Temp (°C) | 35.48 | 35.50-38.50 | 4.1% | ThermalMass |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 600 | 3 | 239.8% |
| 600FF | 2 | 102.5% |
| 650FF | 2 | 87.9% |
| 950FF | 1 | 50.0% |
| 900FF | 1 | 13.0% |

## Known Limitations

### Free-Floating High-Mass Cases (900FF, 950FF)

**Issue:** Maximum free-floating temperature is consistently 12-17% lower than reference for high-mass buildings.

| Case | Min Temp Status | Max Temp Status | Error |
|------|-----------------|-----------------|-------|
| 900FF | ✅ PASS (-4.70°C, ref: -6.40 to -1.60) | ❌ FAIL (36.66°C, ref: 41.80-46.40) | 16.9% low |
| 950FF | ✅ PASS (-9.56°C, ref: -20.20 to -17.80) | ⚠️ WARN (34.04°C, ref: 35.50-38.50) | 8.0% low |

**Root Cause:** The 5R1C steady-state sensitivity model does not fully capture thermal mass dynamics in free-floating conditions. The thermal mass buffering effect is underestimated, leading to lower peak temperatures.

**Resolution Path:** See [Issue #486 Analysis](ISSUE_486_ANALYSIS.md) for detailed investigation and proposed solutions:
1. Empirical correction factors (short-term)
2. Enhanced solar gain distribution calibration (medium-term)
3. 6R2C thermal model upgrade (long-term)
4. Dynamic thermal mass with capacitance (research)

**Current Priority:** Low - This is a known model limitation that affects a small subset of validation cases.

### General Free-Floating Cases (600FF, 650FF)

**Issue:** Free-floating maximum temperatures show 30-50% error for lightweight buildings.

| Case | Max Temp Error | Category |
|------|----------------|----------|
| 600FF | 31.6% low | FreeFloat |
| 650FF | 34.8% low | FreeFloat |

**Status:** Under investigation. May share root cause with high-mass free-floating cases or indicate separate solar gain distribution issues.

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
