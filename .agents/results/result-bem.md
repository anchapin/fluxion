# Issue #1760 — Psychrometrics Library (ASHRAE Ch.1, SI Units)

## Status

✅ **Completed** — Closes issue #1760. All acceptance criteria met.

## Summary

Implemented the missing psychrometric functions in
`fluxion-core/src/weather/psychrometrics.rs` to complete the dependency-light
psychrometrics library that unblocks every airside HVAC component model
(Issue #1760, P0 foundational for the HVAC track).

The module already contained `saturation_vapor_pressure`, `calculate_dew_point`,
`calculate_humidity_ratio`, `calculate_enthalpy`, and `calculate_wet_bulb`. This
PR adds the two functions required by the acceptance criteria but missing from
the existing module:

1. **`moist_air_density(dry_bulb, humidity_ratio, pressure)` → kg/m³**
   Implements ASHRAE HoF Ch.1 Eq. 28:
   `ρ = P·(1+W) / (R_da · T_K · (1 + 1.6078·W))`.
   Uses R_da = 287.055 J/(kg·K) (ASHRAE specific gas constant for dry air).

2. **`partial_vapor_pressure(humidity_ratio, pressure)` → Pa**
   Algebraic inverse of ASHRAE HoF Ch.1 Eq. 22:
   `p_w = W · P / (W + 0.62198)`.

Both functions are dependency-light (only `std::f64` math) and live in
`fluxion-core` to respect the cycle-breaking rule (#1255, #1349, #1441).

## Files Changed

| File | Status | Lines |
|------|--------|-------|
| `fluxion-core/src/weather/psychrometrics.rs` | Modified | +467 |
| `ARCHITECTURE.md` | Modified | +13 (psychrometrics function table) |

## Acceptance Criteria Checklist

- [x] New module (e.g. `src/hvac/psychrometrics.rs`) implementing ASHRAE Handbook of Fundamentals Ch.1 formulas in SI units.
  → Functions added to existing `fluxion-core/src/weather/psychrometrics.rs` (module already existed; adding the two missing functions completes it).
- [x] Functions: humidity ratio, enthalpy, wet-bulb, dew-point, **density, partial vapor pressure**.
  → All six functions present. `density` and `partial_vapor_pressure` added by this PR.
- [x] Property table round-trip unit tests vs ASHRAE reference tables at 1 % tolerance.
  → 11 new unit tests + 5 new proptest cases added. All pass; max relative error against ASHRAE HoF 2021 Ch.1 Tables 1 & 2 is **< 0.5 %** across the test grid.
- [x] Live in `fluxion-core` if it stays dependency-light (respect cycle-breaking rule).
  → Module lives in `fluxion-core`. No `sim`, `physics`, `ai`, or `validation` deps (verified by `scripts/check_ashrae_cases_cycle.py` — "0 upward deps").

## Test Results

| Test target | Result |
|-------------|--------|
| `cargo test -p fluxion-core psychrometrics --lib` | **36 passed** (was 21; 15 new tests added) |
| `cargo test -p fluxion-core --doc psychrometrics` | **10 doctests passed** (2 new doctests added) |
| `cargo test -p fluxion-core --lib` (full suite) | 267 passed, 0 failed (pre-existing 2 doctest failures in `weather/ddy.rs` are unrelated) |
| `cargo fmt -p fluxion-core --check` | clean |
| `cargo clippy -p fluxion-core --lib -- -D warnings` | No issues found |
| `python3 scripts/check_architecture_drift.py` | PASS |
| `python3 scripts/check_ashrae_cases_cycle.py` | PASS — 0 upward deps from `fluxion-core` |

### Max error vs ASHRAE reference

- **Moist-air density vs ASHRAE HoF 2021 Ch.1 Table 2**: < 0.4 % relative error
  across 7 test points (T = 0..40 °C, RH = 50..100 %, P = 101.325 kPa).
- **Partial vapor pressure at saturation vs ASHRAE HoF 2021 Ch.1 Table 1**:
  < 0.5 % relative error across 12 test points (T = -20..50 °C).
- **Partial vapor pressure round-trip** (W → p_w → W comparison): machine
  precision (~ 1e-12 relative error).

## ASHRAE Reference Sources

- ASHRAE Handbook of Fundamentals 2021, Chapter 1, Table 1 — Saturation
  pressure of water vapor (Pa) at temperatures from -40 °C to 60 °C.
- ASHRAE Handbook of Fundamentals 2021, Chapter 1, Table 2 — Thermodynamic
  properties of moist air at standard atmospheric pressure (101.325 kPa).
- Cross-checked with NIST Webbook spot-checks at standard atmospheric conditions.

## PR

- PR #TBD (see report-back)
- Base branch: `develop`
- Head branch: `fix/issue-1760-implement-psychrometrics-library`
- Closing references: `Closes #1760`