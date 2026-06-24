## Issue Description

`WeatherDependentVentilation::get_ach()` in `src/sim/ventilation.rs:323-329` ignores all weather parameters (outdoor temp, indoor temp, wind speed, zone volume) and always returns `base_ach`.

**Current code (broken):**
```rust
fn get_ach(&self, _hour: usize) -> f64 { self.base_ach }
```

The weather-dependent calculation exists as `get_ach_weather()` but is **not part of the trait**.

## ARCHITECTURE.md Contract (violated)

Module 4 contract explicitly requires weather inputs:
- Outdoor temperature → `f64` [C]
- Indoor temperature → `f64` [C]
- Wind speed → `f64` [m/s]
- Zone volume → `f64` [m3]

## Fix Options

1. **Option A**: Add weather parameters to `get_ach` signature:
   ```rust
   fn get_ach(&self, hour: usize, T_outdoor: f64, T_indoor: f64, wind_speed: f64, volume: f64) -> f64;
   ```

2. **Option B**: Rename `get_ach` to `get_base_ach()` and promote `get_ach_weather()` to trait level.

## Files Affected

- `src/sim/ventilation.rs:323-329`
- `src/sim/ventilation.rs` trait definition

## Acceptance Criteria

- [ ] `WeatherDependentVentilation::get_ach(12)` returns different values for different weather conditions
- [ ] Trait contract matches ARCHITECTURE.md Module 4 specification
- [ ] All implementations (`ConstantVentilation`, `ScheduledVentilation`, `WeatherDependentVentilation`) have consistent trait signatures

## References

- ARCHITECTURE.md Module 4 (lines 231-263)