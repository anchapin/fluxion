# Bottom-Up Testing Module Audit

**Created**: 2026-08-28
**Status**: IN_PROGRESS
**Author**: Claude Code session

---

## 1. Current Coverage Baseline

Source: `validation/coverage_baseline.json` (dated 2026-08-10)

### Overall Metrics
| Metric | Value |
|--------|-------|
| Line Coverage | 79.8% |
| Branch Coverage | 63.8% |
| Lines Hit | 84,545 / 105,944 |
| Branches Hit | 4,959 / 7,775 |

### Per-Path Coverage

| Path | Line % | Branch % | Min Branch Floor | v1.3 Target |
|------|--------|----------|------------------|-------------|
| weather_solar | 97.1% | **61.1%** | 60.0% | 75.0% |
| weather_ventilation | 92.7% | 88.5% | 85.0% | 75.0% |
| conduction_zone | 87.7% | 64.7% | 63.0% | 75.0% |
| hvac_zone | 88.8% | 68.3% | 66.0% | 75.0% |
| sim | 81.6% | 62.7% | — | — |
| validation | 89.9% | 67.6% | — | — |

---

## 2. Per-Module Coverage Status

### Weather Module (`src/weather/`)

**Files**: `mod.rs`, `denver.rs`, `epw.rs`

**Status**: Well-tested
- EPW parsing: 97%+ line coverage
- TMY3 embedded data: Fully covered
- **Branch gap**: 61.1% vs 75.0% target

**Key Tested Functions**:
- `HourlyWeatherData::new()` — constructor with all weather parameters
- `EpwWeatherSource::from_file()` — EPW file parsing
- `DenverTmyWeather::get_hour()` — TMY3 data retrieval
- Weather interpolation between hourly timesteps

**Potential Untested Edge Cases**:
- EPW v2/v3/AMY/IWEC version detection edge cases
- Invalid EPW header handling
- Hour-of-year boundary wrapping (hour 8759)
- Negative/invalid weather values (DNI, DHI, GHI < 0)

---

### Solar Module (`src/solar/`)

**Files**: `mod.rs`, `pv.rs`, `solar_position.rs`, `surface_irradiance.rs`

**Status**: Well-tested for calculations, gap in wiring
- Line coverage: High (per-surface irradiance calculations)
- **Branch gap**: 61.1% in weather_solar path

**Key Tested Functions**:
- `solar_position::calculate_solar_position()` — zenith, azimuth
- `surface_irradiance::calculate_surface_irradiance()` — per-tilt/azimuth
- Perez model decomposition (beam/diffuse split)

**Potential Untested Functions**:
- `solar_position::sunrise_hour_angle()` — rarely called directly
- `pv::calculate_pv_output()` — not integrated in ASHRAE path
- Beam-to-mass distribution in thermal context

---

### Physics Module (`src/physics/`)

**Status**: Moderate coverage with known gaps
- `gauge_zone_solver.rs` line 621 fix recently applied
- **Branch coverage**: 64.7% (conduction_zone), 68.3% (hvac_zone)

**Key Tested Functions**:
- `step_physics_5r1c` — main transient coupling (recently fixed)
- Conduction CTF calculations
- HVAC equipment curves

**Wire-Edge Gaps (HIGH PRIORITY)**:

| Edge | Description | Test Coverage |
|-------|-------------|---------------|
| Weather → Solar | TMY3 DNI/DHI/GHI to `solar_position` input | **GAP**: No direct wiring test |
| Solar → Conduction | Surface irradiance to per-surface heat flux | **GAP**: No integration test |

---

## 3. Key Untested Functions Identified

### Weather → Solar Wiring

```rust
// src/weather/mod.rs
pub fn get_hourly_solar(&self, hour: usize) -> SolarData {
// NOT directly tested — only indirectly via ASHRAE validator
}
```

### Solar → Conduction Wiring

```rust
// src/solar/surface_irradiance.rs
pub fn calculate_surface_heat_flux(&self, surface: &Surface) -> HeatFlux {
// NOT tested in isolation — only via system tests
}
```

### Gauge Zone Solver

```rust
// src/physics/gauge_zone_solver.rs
fn step_physics_5r1c(&mut self, dt: f64) -> Result<()> {
// Line 621: recently fixed coupling factor
// Need regression test for this specific fix
}
```

---

## 4. Wire-Edge Gaps (Critical Path)

### Gap 1: Weather TMY3 → Solar Irradiance

**Description**: The diagnostic chain starts with Weather data, but there's no integration test that:
1. Loads TMY3 weather data for Denver
2. Extracts DNI/DHI/GHI for a specific hour
3. Passes those values to `solar_position::calculate_solar_position()`
4. Verifies the resulting irradiance values

**Impact**: ASHRAE Case 900/960 solar gain validation depends on this wiring

**Recommended Test**: `tests/weather_solar_integration.rs`

---

### Gap 2: Solar Irradiance → Per-Surface Heat Flux

**Description**: Even when solar position is correct, there's no test that:
1. Computes surface irradiance for tilted surfaces
2. Distributes beam/diffuse radiation
3. Converts to heat flux for the 5R1C thermal network

**Impact**: Case 600 peak cooling validation failures (LIMIT-16)

**Recommended Test**: `tests/solar_conduction_integration.rs`

---

## 5. Test Infrastructure Requirements

### Required Files (from Handover)

| File | Purpose | Status |
|------|---------|--------|
| `coverage.toml` | Coverage target configuration | ✅ Created |
| `tests/weather_solar_integration.rs` | Weather→Solar wiring test | 🔲 Pending |
| `tests/solar_conduction_integration.rs` | Solar→Conduction wiring test | 🔲 Pending |

### Environment Variables

```bash
export FLUXION_EPW_DIR=/home/alex/Projects/fluxion/assets/weather
```

Without this, ~40 tests fail with "EPW file not found".

---

## 6. Known Structural Limitations (LIMIT-*)

These require **GaugeSolver rework** and won't be fixed by unit testing alone:

| Issue | Cases | Description |
|-------|-------|-------------|
| LIMIT-14 | Case 960 | Inter-zone transfer |
| LIMIT-16 | 610/630/650 | Peak cooling |
| LIMIT-17 | Case 950FF | Night-vent |
| LIMIT-18 | Case 960 Blind | Heating max |
| LIMIT-19 | InvariantChecker | Artificial gain |
| LIMIT-20 | Case 195 HighMass | Returns 0 kWh |

---

## 7. Next Actions

1. **Create `tests/weather_solar_integration.rs`** — wire Weather TMY3 → solar irradiance
2. **Create `tests/solar_conduction_integration.rs`** — wire solar irradiance → per-surface flux
3. **Run `cargo clippy --all-targets -- -D dead_code`** — verify no dead code introduced
4. **Run `FLUXION_EPW_DIR=$(pwd)/assets/weather cargo test -p fluxion --lib`** — validate tests pass
5. **Update coverage baseline** with `python scripts/coverage_baseline.py --update`

---

## 8. Tolerance Policy (from PRD Section 8)

| Domain | Tolerance |
|--------|-----------|
| Temperature | ±0.5°C |
| Relative Energy | ±1% |
| Relative Power | ±1% |

Property-based testing via `proptest` should be limited to **critical numeric functions only**, not applied blanket.
