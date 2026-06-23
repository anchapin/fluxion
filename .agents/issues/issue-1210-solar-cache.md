## Issue Description

Solar position is recalculated **per surface per timestep** despite being deterministic for fixed lat/lon:

```rust
let sun_pos = calculate_solar_position(lat, lon, year, month, day, hour);
```

For a building at fixed lat/lon:
- 5 surfaces × 8760 timesteps/year = **43,800 calls/year**
- Only **8,760** unique datetime values

Each call involves ~15 trig functions. This is 5× redundant computation.

## Additional Optimization Opportunity

Per-call redundancy in solar_position.rs:
- `gamma.cos()`, `gamma.sin()`, `(2*gamma).cos()`, etc.
- `lat_rad.sin()`, `lat_rad.cos()` (constant for fixed location)
- `declination` (function of day-of-year only)

These could be cached by day-of-year (365 values max).

## Recommended Fix

Cache solar position by `(year, month, day, hour)` in simulation state:

```rust
sun_pos_cache: Vec<Option<SolarPosition>>,  // indexed by hour_of_year

// At timestep start:
let hour_idx = (day_of_year - 1) * 24 + hour as usize;
if let Some(cached) = self.sun_pos_cache.get(hour_idx) {
    sun_pos = cached.clone();
} else {
    sun_pos = calculate_solar_position(...);
    self.sun_pos_cache[hour_idx] = Some(sun_pos);
}
```

## Estimated Impact

- Solar position cache: **10-15% faster** overall simulation
- Declination by day-of-year cache: **3-5% additional**
- Total: **~20% speedup** on solar-related computation

## Files Affected

- `src/sim/thermal_model_physics/physics_impl.rs`
- `src/sim/thermal_model_core.rs` (add cache to state)
- `src/solar/solar_position.rs` (day-of-year caching)

## Acceptance Criteria

- [ ] Solar position computed once per timestep, reused across surfaces
- [ ] Benchmark shows measurable speedup (target: 10-15%)
- [ ] Cache memory overhead is bounded (8760 × ~48 bytes ≈ 400KB)