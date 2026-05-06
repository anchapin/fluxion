## Problem Description

ASHRAE 140 Appendix C specifies **synthetic clear-day solar profiles** for validation, NOT actual TMY weather data. Currently, Fluxion uses real Denver TMY weather (`tests/data/v0.5/weather/denver.epw`) which contains variable cloud conditions rather than the prescribed ASHRAE clear-day values.

## Investigation Findings

### Code Locations
- Weather data: `src/weather/denver.rs` - DenverStapleton TMY (WMO#724690)
- Solar calculation: `src/sim/solar.rs:234-272` - `calculate_surface_irradiance()`
- EPW parsing: `src/weather/epw.rs:393-394` - DNI/DHI extraction

### Critical Discrepancy: Two Solar Constants
| Location | Value | File |
|----------|-------|------|
| `physics::constants::solar::ashrae_140::SOLAR_CONSTANT` | 1361.0 W/m² | Correct for ASHRAE |
| `sim::sky_radiation::SOLAR_CONSTANT` | 1366.1 W/m² | Used in extraterrestrial_irradiance() |

### Weather vs. ASHRAE Prescribed Values

| Metric | ASHRAE 140 Appendix C | Denver TMY (actual) |
|--------|----------------------|---------------------|
| Data type | Synthetic clear-day | Real hourly TMY |
| Peak DNI (summer noon) | ~1000 W/m² | 1257 W/m² (extreme clear day) |
| Peak GHI | ~900-1000 W/m² | Variable |

### Impact on Issue #666

The free-floating cases use TMY weather with extreme solar values. On a clear summer day in Denver TMY, DNI can reach 1257 W/m² which is 26% higher than typical ASHRAE clear-day values. This could contribute to excessive heating.

## Proposed Fix Options

1. **Option A**: Use ASHRAE 140 synthetic weather data for validation test cases
2. **Option B**: Normalize TMY data to ASHRAE 140 design day values
3. **Option C**: Add scaling factor to match ASHRAE 140 summer design day irradiance

## Files to Investigate
- `src/weather/denver.rs` - Weather data source
- `src/sim/solar.rs:253` - extraterrestrial_irradiance() uses wrong solar constant
- `src/sim/sky_radiation.rs:37` - Uses 1366.1 instead of 1361.0

## Verification
- Compare free-floating temperatures using TMY vs. ASHRAE prescribed irradiance
- Verify which weather data source is used for blind validation tests