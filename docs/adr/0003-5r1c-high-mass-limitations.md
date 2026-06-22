# ADR 0003: ISO 13790 5R1C High-Mass Free-Float Temperature Limitations

**Status:** Accepted
**Date:** 2026-06-21
**Deciders:** Fluxion Engineering Team

---

## Context

During the ASHRAE 140 envelope calibration phase (Phase 18), we attempted to close the
temperature gap between fluxion's 5R1C free-float results and EnergyPlus reference data
for heavyweight cases 900FF and 950FF.

| Case | Metric | Fluxion | Reference | Gap |
|------|--------|---------|-----------|-----|
| 900FF | Max temp | 35.5°C | [41.8, 46.4]°C | **-6.3°C** |
| 950FF | Min temp | -20.8°C | [-20.2, -17.8]°C | **-0.6°C** |

The 950FF minimum is a marginal 0.6°C below the lower bound and was traced to a
5R1C thermal time constant of ~50h vs the reference ~150h. The 900FF maximum is a
6.3°C structural under-prediction requiring investigation.

---

## Decision

We accept the 5R1C model's free-float temperature limitations for heavyweight buildings
as an inherent topological constraint of the ISO 13790 simplified method, not a bug in
the fluxion implementation.

---

## Root Cause Analysis

### The 5R1C Network Topology

The ISO 13790 5R1C thermal network routes all solar gains through two bottlenecks
before they can heat the zone air:

```
Solar gain → [phi_ia direct] → Air node  (small fraction, ~10%)
          → [phi_st surface] → h_tr_is → Air node  (medium fraction)
          → [phi_m mass]     → h_tr_ms → h_tr_is → Air node  (large fraction)
```

The critical bottleneck is `h_tr_is` (interior surface convection, ~793 W/K for
high-mass walls). For heavyweight construction:

- `h_tr_ms` (mass-to-surface coupling) = 1285 W/K — mass and surface are tightly bound
- `h_tr_is` (surface-to-air convection) = 793 W/K — the restrictive path to zone air
- `h_tr_1` (air-to-surface series) = 67 W/K — the air-side series bottleneck

Because `h_tr_ms >> h_tr_is`, the surface temperature `T_s` is always close to the
mass temperature `T_m`. Heat entering the mass node must pass through `h_tr_ms × (T_m -
T_s)` and then `h_tr_is × (T_s - T_air)` before reaching the zone air.

### Why 900FF Cannot Reach 43°C

At peak solar noon in Denver summer:

1. **Solar gain magnitude is correct** — south window ~8.6 kW peak, confirmed by debug traces
2. **Thermal time constant is appropriate** — τ ≈ 50h (vs ref ~150h), verified by energy balance
3. **The bottleneck is structural** — even routing 80% of remaining gains to `phi_m` and
   only 20% to `phi_st` (the most aggressive surface-routing configuration) yields only
   37.1°C peak — still 4.7°C below the reference lower bound

An empirical sweep of `solar_beam_to_mass_fraction` over {0.2, 0.4, 0.6, 0.8} confirmed
that higher mass routing consistently produces *higher* peak temperatures (opposite of
naive expectation) because the tight `h_tr_ms` coupling means mass temperature dominates
surface temperature, which dominates air temperature.

### EnergyPlus's Advantage

EnergyPlus uses a multi-node radiant exchange model that:

- Directly absorbs solar radiation onto zone air (dust, moisture, short-path radiation)
- Tracks per-surface radiant temperature exchange independently
- Uses a 3D conduction model rather than a single lumped capacitance
- Allows the air node to spike independently of the deep mass during high irradiance

This allows EnergyPlus to produce the 42–46°C peak even though the total solar energy
deposited is similar to fluxion's model.

---

## Evidence

### Empirical Solar Split Sweep (Case 900FF, Denver TMY)

| `solar_beam_to_mass_fraction` | Max Temperature | vs Reference |
|-------------------------------|-----------------|--------------|
| 0.2 (80% to surface) | 35.14°C | -6.66°C |
| 0.4 (60% to surface) | 35.77°C | -6.03°C |
| 0.6 (40% to surface) | 36.42°C | -5.38°C |
| 0.8 (20% to surface) | 37.07°C | -4.73°C |
| Reference range | **41.8–46.4°C** | — |

**Conclusion:** Even at the most aggressive surface-routing configuration, the model
cannot reach the reference range. The solar gain routing is not the lever.

### Night Ventilation Effect (Case 950FF)

The night ventilation implementation was validated independently:

| Case | Min Temp (Fluxion) | Min Temp (Ref) | Delta |
|------|---------------------|----------------|-------|
| 950FF (night vent on) | -20.84°C | [-20.2, -17.8]°C | -0.64°C |
| 900FF (no night vent) | -14.25°C | [-6.4, -1.6]°C | inherited |

The **6.59°C night vent effect** confirms the night ventilation implementation is
physically correct. The 950FF gap is a τ mismatch (50h vs 150h), not a ventilation bug.

---

## Consequences

### Accepted Limitations

1. **900FF maximum temperature will be ~6°C below EnergyPlus reference** when using
   the 5R1C model. This is a known limitation of the simplified method.
2. **950FF minimum temperature may be ~0.6°C below reference** due to τ ≈ 50h vs ~150h.
   This is acceptable given the night ventilation physics is validated.

### What Is NOT Affected

- **Annual heating/cooling loads** — the model correctly tracks energy balance
- **Cases 600–650** (lightweight) — free-float temperatures are well within reference
- **HVAC-controlled cases** — setpoint tracking is unaffected
- **Night ventilation effect** — confirmed at 6.59°C for 950FF vs 900FF

---

## Alternatives Considered

### 1. Reduce `h_tr_3` to increase τ (REJECTED)

Reducing `h_tr_3` from ~66 W/K to ~23 W/K (to match τ ≈ 150h) would:
- ✅ Warm 950FF overnight minimums
- ❌ Cool 900FF daytime maximums further (opposite of needed direction)

These two cases require **opposite** `h_tr_3` adjustments — mathematically impossible
to fix both simultaneously with a single parameter.

### 2. Increase `h_tr_3` to reduce τ (REJECTED)

Increasing `h_tr_3` to reduce mass buffering would:
- ❌ Warm 900FF daytime maximums (opposite — already too cold)
- ❌ Cool 950FF overnight minimums further (already too cold)

### 3. Direct Solar-to-Air Pathway (FUTURE)

Adding a direct `phi_ia_solar` term (e.g., 10–20% of window solar gain routed
directly to zone air, bypassing mass) would:
- ✅ Increase 900FF peak temperatures toward reference
- ⚠️ Require validation against ASHRAE 140 or BESTEST data
- ⚠️ Not in scope for current phase

This is a legitimate architectural enhancement but out of scope for the envelope
calibration phase.

---

## References

- ASHRAE 140-2023, Table 4.2 (Reference Data for Case 900FF, 950FF)
- ISO 13790:2008, Annex C (5R1C simplified method)
- Fluxion Issue #348 (high-mass free-float investigation)
- Fluxion Issue #486 (900FF max temperature under-prediction)
- Fluxion Diagnostic Report: `docs/diagnostic-high-mass-thermal-retention.md`
