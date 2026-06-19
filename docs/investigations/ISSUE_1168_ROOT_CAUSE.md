# Issue #1168 — Free-Float Over-Damping: Root Cause Analysis

> **Status:** Root cause identified — **escalated to #1152** (5R1C mass-coupling
> restructure). No physics code changed in this investigation; this document
> plus the diagnostic harness (`tests/issue_1168_free_float_diagnostic.rs`)
> are the deliverables.
>
> **Investigator:** backend agent (wave: fix/issue-1168-free-float-temperature)

## TL;DR

The free-floating air temperature is over-damped because the **5R1C
single-mass-node steady-state topology algebraically pins the air node to the
slow mass node**. It is **neither the thermal capacitance (Cm) nor the solar
distribution**. The fix requires the #1152 mass-coupling restructure (or a
dynamic / multi-node solver); it is not achievable by tuning a coefficient.

## Acceptance-criteria answer

> **"Root cause identified: is it heat transfer rate or thermal capacitance?"**

**Neither in isolation — it is the steady-state coupling STRUCTURE.**

The air temperature is computed as a quasi-steady-state weighted average of the
mass temperature and the outdoor dry-bulb:

```
T_air = (H_air_mass · T_mass + h_ext · T_outdoor + Φ_solar) / (H_air_mass + h_ext)
```

where `H_air_mass = h_tr_ms ∥ h_tr_is` (series). For Case 600FF the weights are
**68 % mass / 32 % outdoor**, so the air follows the sluggish mass rather than
the forcing. A weighted average is **bounded** by its inputs, so the air can
never reach the reference extremes.

## Measured network parameters (from the diagnostic harness)

| Parameter | 600FF | 900FF | Meaning |
|---|---|---|---|
| `h_tr_ms` (mass↔surface) | 240.0 | 1608.0 W/K | mass-surface coupling |
| `h_tr_is` (surface↔air) | 165.6 | 165.6 W/K | surface-air coupling |
| `h_tr_em` (ext↔mass) | 59.26 | 48.88 W/K | opaque envelope |
| `h_ext = h_w + h_ve` | 46.91 | 91.31 W/K | **air↔outdoor** |
| `Cm` | 2.32e6 | 1.07e7 J/K | mass capacitance |
| **`H_air_mass` (series)** | **97.99** | **150.14 W/K** | air↔mass coupling |
| **`H_air_mass / h_ext`** | **2.09** | **1.64** | air pinned to mass if > 1 |
| `τ_mass = Cm/(h_em+h_tr_3)` | 6.66 h | 22.33 h | mass time constant |

### Free-float results (annual, after 2 warm-up years)

| Case | AIR min | AIR max | Δ AIR | MASS min | MASS max | Ref max range |
|---|---|---|---|---|---|---|
| 600FF | −5.6 | 54.6 | 60.1 | −5.4 | 35.1 | [64.9, 75.1] ❌ |
| 650FF | −11.1 | 54.0 | 65.2 | −7.2 | 34.8 | ❌ |
| 900FF | −2.4 | 35.5 | 37.9 | 1.6 | 31.6 | [41.8, 46.4] ❌ |
| 950FF | −9.6 | 35.5 | 45.0 | 0.0 | 31.1 | ❌ |

Outdoor dry-bulb swing recorded by the harness: **−12.5 °C to 32.5 °C**.

## Three controlled experiments (Python, full year, linearized 5R1C)

The linear model reproduces the engine's behaviour (600FF ≈ [−7.4, 61.0] vs
engine [−5.6, 54.6]).

### Experiment 1 — sweep solar-to-air fraction `air_frac` (600FF)

| air_frac | AIR min | AIR max |
|---|---|---|
| 0.00 | −7.2 | 46.0 |
| 0.40 | −7.3 | 53.4 |
| 0.80 (code) | −7.4 | 61.0 |
| 0.95 | −7.4 | 63.8 |

**Solar routing moves the daytime MAX a little but has ZERO effect on the
night MIN** (no sun at night). The night-min gap (ref −17.2) is therefore
**not** a solar-distribution problem. The current `air_frac = 0.80` is a
calibration constant that is already stale (the source comment at
`thermal_model_core.rs:1675` claims `0.95 → 72.89 °C`, set when
`h_tr_is ≈ 1422 W/K`; it is now 165.6 W/K). Retuning it is forbidden by
AGENTS.md and cannot fix the min anyway.

### Experiment 2 — sweep thermal capacitance `Cm` (600FF)

| Cm scale | τ | AIR min | AIR max |
|---|---|---|---|
| 1.0 | 6.66 h | −7.4 | 61.0 |
| 0.1 | 0.67 h | −12.8 | 67.4 |
| 0.02 | 0.13 h | −13.6 | 69.3 |

**Even with the mass slashed to 2 % (τ = 8 min), the min only reaches −13.6.**
Capacitance is **not** the cause. The default `Cm` (τ ≈ 6.7 h) is physically
reasonable for a low-mass building.

### Experiment 3 — sweep the mass-air COUPLING `h_tr_ms` (600FF)

| h_ms scale | H_air_mass/h_ext | AIR min | AIR max |
|---|---|---|---|
| 1.0 | 2.09 | −7.4 | 61.0 |
| 0.5 | 1.48 | −8.2 | 70.7 ← max in range |
| 0.25 | 0.94 | −9.5 | 83.8 ← overshoots |
| 0.10 | 0.45 | −11.8 | 103.4 |

The coupling is the dominant control on the **daytime max**, but reducing it
overshoots and the **night min still never reaches −17.2**. No single value of
`h_tr_ms` puts all four FF cases in range — confirming it is structural, not a
tunable coefficient. (Per AGENTS.md, `h_tr_ms` must not be tuned; that is
exactly what #1152 restructures.)

## Why the air cannot reach the reference extremes (the structural proof)

In the 5R1C simple-hourly formulation **only the mass node carries
capacitance**; the air and surface nodes are solved algebraically (instantaneous
steady state given `T_mass`). Therefore:

```
T_air = weighted_average(T_mass, T_outdoor)        # + solar term
```

A weighted average is bounded by its arguments, so:

1. **Night min** — `T_air ≥ min(T_mass, T_outdoor_drybulb)`. The outdoor dry-bulb
   floor is −12.5 °C, and the air is further held up by the 68 % mass weight.
   The ASHRAE reference min of **−18.8 °C is below the dry-bulb** — it is
   produced by **direct longwave radiative cooling to the cold sky**. In 5R1C,
   sky radiation reaches the zone only via the *sol-air temperature driving the
   mass node* (`h_tr_em`); the **air node has no direct radiative path**, so a
   sub-dry-bulb air temperature is **unreachable**. This is why 900FF (heavy
   mass, warm reference min −4 °C) passes the min while 600FF (light mass, cold
   reference min −17.2 °C) cannot.

2. **Day max** — solar heats the air, but the 68 % mass weight drags it back
   toward the sluggish mass, under-shooting the peak.

Both are consequences of the **single-lumped-mass-node, quasi-steady-state air
topology** — the limitation that #1152 exists to remove.

## Why this is the same family as PR #1172 (and is still separate)

PR #1172 fixed the **HVAC-mode** cooling gap by replacing an asymmetric
controller formula. Its root-cause note observed the identical underlying
phenomenon — *"solar heats the zone AIR faster than the thermal MASS; the mass
lags the air"* — and explicitly attributed the residual gap to *"the 5R1C
steady-state floor documented in ARCHITECTURE.md — Issue #1152."* Free-float
mode has no HVAC controller, so #1172's fix does not touch it; the over-damping
persists for the structural reason above.

## Recommended fix (out of scope here)

1. **Primary:** #1152 — restructure the 5R1C mass coupling so the air node is
   not algebraically pinned to a single lumped mass (e.g. explicit air-node
   capacitance, or the 9R4C multi-node model extended to free-float).
2. **Required for the night min specifically:** give the air/interior-surface
   node a direct longwave-to-sky path so the free-floating air can drop below
   dry-bulb under clear-sky radiative cooling.
3. Remove the now-stale `solar_distribution_to_air` calibration (0.80 / 0.40 /
   0.10) once the structural fix lets the physics use the ASHRAE-140-correct
   value (solar → opaque surfaces, none directly to air).

## Evidence artifacts

- `tests/issue_1168_free_float_diagnostic.rs` — prints network parameters,
  time constants, and annual air/mass min-max for 600FF/650FF/900FF/950FF.
  Run: `cargo test --test issue_1168_free_float_diagnostic -- --nocapture --ignored`
- This document: `docs/investigations/ISSUE_1168_ROOT_CAUSE.md`
- Baseline tests (no production code changed): `weather_isolation` 19✓,
  `solar_isolation` 7✓, `ashrae_140_blind_validation` ✓.

## Conclusion

**Do not merge a coefficient change for #1168.** The acceptance criteria
(600FF/900FF in range) cannot be met by any defensible single-parameter change
without the #1152 restructure. This issue is unblocked by #1152.
