# #4238 — ACH-dependent interior surface convection for night ventilation: assessment

**Date:** 2026-10-16 (round 44). **Verdict: no engine change.** The mechanism the
issue asks about is already implemented; the paper's remaining gap is measured
at 0.29 K on the recorded floor and sits inside the structural-difference
attribution of §52/§53. Implementation beyond this would be tuning risk, not
demonstrated physics.

## What fluxion already does

- `h_tr_is_ach_multiplier` (`src/sim/ventilation.rs`, Issue #1279): the
  ASHRAE/EnergyPlus forced-convection correlation
  `h_c = h_c_still + 0.84·ACH^0.8` (h_c_still = 3.45 W/m²K), applied to the
  air-surface coupling `h_tr_is` whenever night ventilation is active, in both
  the mass-node step and the zone-air computation
  (`step_9r4c.rs` "Issue #1279" sites). This is exactly the class of
  ACH-dependent interior HTC the referenced paper (Buildings 16(17):3555,
  2026) argues fixed-correlation models lack.
- The multiplier is capped at 2.0 (`MAX_CONVECTIVE_TO_AIR_MULTIPLIER`,
  Issue #2871) to prevent mass-node pulsed-charging dump on the morning ramp —
  a stability constraint, not a physics claim. At the ASHRAE 140 night-vent
  ACH (~10.8 for the 1400 m³/h Table 5-160 fan in the 129.6 m³ zone) the
  uncapped ratio is ~2.9×, so the cap binds.

## What the paper adds beyond this

The paper's claim is that surface-level (position- and mass-distribution-
dependent) convection matters beyond zone-average ACH correlations, and that
existing correlations underpredict night-vent mass cooling. That is a real
fidelity direction, but it is a different mechanism than the one this issue's
concrete investigation steps ask about (steps 1–2: audit ACH dependence,
evaluate an ACH-dependent HTC). Both steps are already satisfied by #1279.

## Measured materiality on the recorded floor (§53, round 43)

The film/convection boost was measured directly on 950FF (the legacy 9R4C
night-vent case): removing it entirely moves Tmax 38.78 → 39.01 °C — the whole
ACH-dependent convection mechanism owns **0.29 K** of the residual, on a floor
whose dominant measured term is night-ventilation coupling strength (halving
the primary-source 1400 m³/h fan flow reaches the published band; §53 verdict:
structural, recorded, not tuned). 650FF is bit-identical (low-mass 5R1C
routing).

## Why not implement now

- The only unexplored arm this issue could motivate is raising/removing the
  #2871 cap. That arm reaches for the same floor the fan-flow arm reaches, and
  the fan flow is a primary-source value: uncapping to move a validator number
  is the §53 tuning pattern, reversed.
- The paper's surface-level convection (position-dependent HTC) is a genuine
  long-term fidelity item, but it is a structural model change with its own
  validation burden — out of scope for this research pointer and not
  demonstrably material at the current 0.29 K scale.

**Closes when:** either the cap's stability premise (#2871) is superseded by a
stable formulation that admits the uncapped ACH curve, or a surface-level HTC
model is scoped with evidence that it moves the night-vent floor beyond the
flow-coupling attribution.

**Sources:** #4238; Buildings 16(17):3555 (2026); §52/§53 in
docs/KNOWN_ISSUES.md + docs/investigations/ (rounds 42–43); Issue #1279;
Issue #2871.
