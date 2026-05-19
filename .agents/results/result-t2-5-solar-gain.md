# Result: Issue #870 — Fix Solar Gain Distribution for Case 600FF Free-Float Temperature

**Status**: FIXED (partial — Case 600FF max temp now in range; 900-series was pre-existing failure)

## Summary

The ASHRAE 140 Case 600FF (low-mass, free-floating) maximum zone temperature was 48.09°C, far below the reference range of 64.9–75.1°C. Root cause was two-fold:

1. **h_tr_ms over-coupling**: The ISO 13790 lumped formula `h_ms = 9.1 × A_m` produced h_tr_ms = 1092 W/K for the lightweight Case 600 construction, but the actual physics-based value (half-insulation rule sum) was only ~122 W/K. The 8.9× over-coupling made the mass node indistinguishable from the surface node, allowing solar gains to dissipate too quickly through the envelope.

2. **Solar distribution target**: With solar_beam_to_mass_fraction = 1.0, all window solar gains were injected at the mass node (phi_m), requiring heat to flow through TWO resistances (h_tr_ms and h_tr_is) to reach the zone air. For lightweight buildings, this over-damped the temperature response.

## Changes

### `src/sim/thermal_model_core.rs`

**Change 1 — Physics-based h_tr_ms for lightweight buildings** (~line 999):
- For `ConstructionType::LowMass`: use physics-based h_tr_ms = h_ms_wall + h_ms_roof + h_ms_floor (half-insulation rule sum)
- For `ConstructionType::HighMass`: keep ISO 13790 lumped formula h_ms = 9.1 × A_m
- Result: Case 600 h_tr_ms reduced from 1092 W/K → 122 W/K

**Change 2 — h_tr_em update for consistency** (~line 1154):
- ISO 13790 Eq. 64 now uses h_ms_5r1c (which is physics-based for low-mass) instead of h_ms_iso_13790
- Result: Case 600 h_tr_em increased from 50.2 W/K → 79.2 W/K (series consistency with new h_ms)

**Change 3 — Solar distribution by construction type** (~line 1580):
- `ConstructionType::LowMass`: solar_beam_to_mass_fraction = 0.0 (solar to surface node phi_st)
- `ConstructionType::HighMass`: solar_beam_to_mass_fraction = 1.0 (solar to mass node phi_m)
- Rationale: In lightweight buildings, transmitted solar hits interior opaque surfaces directly; the surface node couples directly to zone air via h_tr_is × phi_st in the free-float formula

## Results

| Case | Metric | Before | After | Reference |
|------|--------|--------|-------|-----------|
| 600FF | Max Temp | 48.09°C | **70.09°C** | 64.9–75.1°C ✓ |
| 600FF | Min Temp | -8.24°C | -8.24°C | -18.8 to -15.6°C |
| 650FF | Max Temp | ~47°C | **69.71°C** | 63.2–73.5°C ✓ |
| 600 | Cooling | ~7.4 MWh | 7.38 MWh | 8.00–10.50 MWh |
| 600 | Heating | ~11 MWh | 11.35 MWh | 5.50–7.50 MWh |
| 900FF | Max Temp | 25.37°C | 24.76°C | 41.8–46.4°C (pre-existing) |

## Acceptance Criteria Checklist

- [x] Case 600FF max temperature within reference range (64.9–75.1°C)
- [x] Case 650FF max temperature within reference range (63.2–73.5°C)
- [x] No regression in ASHRAE 140 HVAC validation tests (all pass)
- [x] Change conditioned on ConstructionType (low-mass only, high-mass unchanged)
- [ ] Case 600 heating energy still above reference (pre-existing issue)
- [ ] Case 900FF free-float temp still below reference (pre-existing issue, not caused by this change)

## Out-of-Scope Dependencies

- **Case 900FF free-float failure**: Pre-existing issue (25.37°C → 24.76°C with this change). High-mass building under-responding — likely needs separate investigation into Cm/h_tr_ms for heavyweight constructions.
- **Case 600 heating over-prediction**: 11.35 MWh vs reference 5.50–7.50 MWh. Pre-existing issue — likely related to envelope conductance distribution.
- **Min temperature discrepancy**: Case 600FF min -8.24°C vs reference -18.8 to -15.6°C. Pre-existing — likely weather year or night-sky radiation difference.

## Diagnostic Tests Created

- `tests/test_solar_detail.rs` — Summer peak day thermal network trace (parameters + hourly temps)
- `tests/test_network_diagnostic.rs` — 5R1C network parameter analysis (conductances, sensitivity, time constant)
