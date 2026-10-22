# Free-float energy-balance audit — 5R1C per-class path (loop round 86)

Date: 2026-10-22 · Branch: `physics/r86-ff-audit` · Engine: `87fb7375` (develop, post-#4533/#4535)

## Scope

Audit where the energy goes in the free-float cases on the now-default per-class
5R1C path, via an env-gated read-only dump (`FLX_FF_EB_AUDIT=1`) in
`step_physics_5r1c` and an offline harness (`scripts/audit_free_float_balance.py`).

**Routing split (finding zero).** The four FF cases do not share a solver:
600FF/650FF are `ThermalModelType` 5R1C models and run the per-class sub-step
mass advance (#4533); 900FF/950FF are `NineRFourC` models and never enter
`step_physics_5r1c` — this dump emits nothing for them, and their night-vent
routing is structurally different (`step_9r4c.rs:97-130` adds `h_ve_night` as
an air exchange **and** an `ach_night_vent` scaling of `h_tr_is` — the same
double-coupling shape). This audit covers the 5R1C pair; a 9R4C-side dump is
the named follow-up for the 900FF Tmax gap and the 950FF Tmin overshoot.

## Method

Per step, the dump emits (all values are the engine's own applied quantities,
read at the end of the step):

- `FFAUDIT_P` — the solar partition `phi_ia` / `phi_st` / `phi_m` per zone;
- `FFAUDIT_Z` — zone row: driving temps (`t_out`, `t_sky`, `t_g`, `t_sol_air`),
  conductances (`h_tr_em`, `h_air_mass`, `h_ext` incl. the night-vent
  addition, `h_ve_night`, `h_vent_mass`, `g_coeff`), states (`t_air` committed,
  `t_m_new`/`tm_start`, `c_air`, `term_rest_1`), the applied gains and the
  night-vent active flag;
- `FFAUDIT_C` — per-class coefficients at the last hour-sub-step (`w`, `cm_c`,
  `a_c`, `b_c`, `h_am_c`, `h_em_c`, `h3_c`, `t_ext_c`, `phi_c`);
- `FFAUDIT_D` — per-class storage delta `cm_c·(tm_c_new − tm_prev)` every
  sub-step (exact telescoping material).

Basis note: `derived_h_ext`/`derived_h_tr_3` are physical W/K (the step
multiplies the air num/den by `term_rest_1` after them);
`derived_ground_coeff` is added to the already-scaled numerator and carries
the factor.

Flux reconstruction is an hourly trapezoid integral over the committed
end-of-hour states — an approximation against the sub-step Crank–Nicolson
integrator, so the script reports closure residuals per node (absolute and
relative to the hour's mean flux). Two flux paths are **not** reconstructed
and show up as residual: the R67 surface-pool layer (`phi_st` enters surface
nodes that exchange with both air and mass; round-71 default) and the exact
per-class sub-step structure. The mass-side night-vent term is reconstructed
with the zone-air drive the CN call actually uses (pinned by the slice-2
contract tests, PR #4536); the BE/FE fallback branches of `advance_node`
would drive it to outdoor but are not taken for realistic masses.

Exact (not reconstructed) quantities: all dumped conductances, the applied
gain partition, the committed state trajectories, mass storage
`cm·(t_m_new − tm_start)`, and the per-class storage split.

## Findings — 600FF (Tmax 47.24 vs band [62.4, 68.4])

Annual integrals (MJ): gains into air 11,570 · into mass 8,099 ·
envelope (sol-air) −15,220 · ground −2,899 · air↔mass +413 · net mass
storage −29.5. The mass node is a pass-through, not a store: it turns over
intra-day and ends the year within 30 MJ of where it started.

Peak day (day 254 of the weather year — the gate's Tmax day, reproduced
exactly at 47.24 °C):

| bucket (MJ) | air node | mass node |
|---|---|---|
| gains (phi_ia / phi_m) | 39.26 | 27.48 |
| envelope (sol-air/dry-bulb) | −44.33 | −18.58 |
| air↔mass exchange | +1.70 | −1.70 |
| ground | −17.80 | — |
| storage change | −0.07 | +0.53 |

- The mass node contributes **1.70 MJ of 39.26 MJ** to the peak day's air
  balance and its temperature moves 31.9 → 34.6 °C (2.6 K) against the air's
  24.0 → 47.2 K swing. It absorbs and returns its 27 MJ intra-day
  (stor_m +826 kJ/h at noon, −420 kJ/h at night) with no phase shift deep
  enough to matter at the peak.
- The afternoon peak is an **instantaneous air-node balance**: at 12:00 the
  air sits 17.2 K above outdoor with phi_ia = 1,642 W, implying a ~95 W/K
  total loss conductance (envelope + series mass link ~22 W/K + surface-pool
  path). There is no stored-energy release stacking into late afternoon —
  which is what EnergyPlus's CTF walls do and why its peaks reach 62–68 °C.
- The dominant sink the 5R1C network does have is the surface-pool path:
  phi_st = 17.8 MJ on the peak day (46% of the day's 39 MJ phi_ia), routed
  through the R67 layer. Its release timing is the natural lever for the
  missing afternoon stack (see experiments below).

## Findings — 650FF (Tmin −25.2 vs band [−9.2, −7.4])

Same case with night ventilation (1,400 m³/h, 18:00–07:00, ASHRAE 140
Table 5-160 sea-level route; h_ve = 469 W/K at ρ=1.2):

Annual integrals (MJ): vent via the **air side** −6,509 · via the **mass
side −20,469** · envelope (air) −5,074 · mass-sol-air +6,013 · air↔mass
−3,876. The sign flips matter: the vent-chilled zone runs so cold that the
mass sits BELOW its sol-air driving for much of the year (the envelope heats
the mass, +6,013, where 600FF's mass loses −7,836), and the air→mass flow
reverses (the vent-chilled air strips the mass all year).

- The mass-side night-vent coupling (`h_vent_mass_zone`, Issues
  #1898/#1693 lineage) adds 469 W/K of mass↔air coupling during 13 vent
  hours — against a ~22 W/K series base coupling. During vent hours the
  mass node loses thermal autonomy and follows the vent-chilled air toward
  outdoor.
- Measured directly on the January 18 night: the 650FF mass falls to
  −1.7 → −5.3 °C overnight (600FF, same weather, holds +9 to +12 °C), with
  stor_m −1,604 kJ/h at midnight and the air floor at −10.8 °C (annual min
  −25.2). The mass-side term removes ~3.1× the energy the ISO air-side
  exchange removes (−20,469 vs −6,509 MJ/yr).
- This quantifies the concern already recorded in the engine's own history
  (Issue #821's note that "setting `h_vent_mass_zone = 0` removes the
  double-counting"): on the live per-class path the term is not a 30%
  boost; it is the dominant ventilation cooling mechanism, and it is the
  measured driver of the 650FF Tmin miss (and, on the 9R4C side with the
  same double-coupling shape, of the 950FF signature).

## Validation of the harness

- Class-storage cross-check: Σ_c FFAUDIT_D vs `cm·ΔT` — max gap 1,309 kJ
  (600FF) / 4,331 kJ (650FF) over 8,758 hours, consistent with the R67
  surface layer taking a share of the composite advance (the pure
  class-CN telescoping identity assumes no t_s-first layer).
- Closure residuals (median, as % of the hour's mean |flux|): air 134%
  (600FF) / 158% (650FF); mass 363% / 364%. Night hours dominate the
  relative numbers (small flux scale); the exact quantities above do not
  depend on the reconstruction.

## Candidate experiments (not executed here; each has a measured premise)

1. **Vent air-only arm** (`FLX_R86_VENT_AIR_ONLY`, engine change pending
   Alex's default-change gate): set `h_vent_mass_zone = 0` on the 5R1C
   sub-step paths. Measured premise: −20.5 GJ/yr of mass-side extraction on
   650FF. Expect the 650FF Tmin to recover toward the band; the full battery
   + honest re-record applies, and Case 950 (HVAC) cooling must be
   re-checked since the term was introduced for its mass pre-cooling.
2. **Surface-pool release timing** (blitz slice candidate): dump the R67
   `t_s` pool states (the missing closure leg, ~18–21 MJ/day at the 600FF
   peak) and measure the pool→air release lag; the afternoon stack EnergyPlus
   produces would come from delaying that return, not from more gain.
3. **9R4C audit row** for 900FF/950FF (routing split above), before any
   950FF vent verdict on that family.

## Reproducing

```
FLX_FF_EB_AUDIT=1 cargo test --release --test all_tests free_float_strict_gate \
  -- --include-ignored --nocapture --test-threads=1 2> ff_audit.log
python3 scripts/audit_free_float_balance.py ff_audit.log
```

The flag is read per step through `util::env_bool` and gates two `eprintln!`
sites plus two class-loop sites; the flag-off path adds no floating-point
operations, so engine outputs are bit-identical with the audit off (the
strict battery and output-diff pass unchanged on this branch).

## Round-87 execution record (2026-10-22, same day)

Candidate experiment 1 above was executed the same day, Alex-approved at
2026-10-22 06:41 ET ("I approve the vent arm"). Branch
`physics/r87-vent-air-default` on develop a77be343.

**Implementation discovered a sharper shape than the audit proposed.** The
audit targeted `h_vent_mass_zone`, but the first arm at that site was
byte-neutral on the strict battery: on the live R76/R85 sub-step path the
mass-side coupling enters as `h_vent_i` folded into the h3 air↔mass
coupling (step_5r1c, per-class site h3_c = h_am_c + h_vent_i·w and lumped
site h_tr3_arg_i = h_air_mass_i + h_vent_i), while `h_vent_mass_zone`
lives only in the legacy hourly advance that the coupled-N default
discards (the round-83 finding, restated). The flip therefore zeroes BOTH
sites behind `FLX_R86_VENT_AIR_ONLY` (falsy token restores the pre-87
coupling; default air-only). The invariant checker needed no change: its
receipt field `ConductionState::h_vent_mass_zone` is never written by the
engine (always 0), so the checker identity already assumed air-only — the
flip makes the receipt truthful rather than widening anything.

**Measured old → new** (canonical weather, strict battery, branch vs
a77be343):

| row | old | new | published band | verdict move |
|---|---|---|---|---|
| 650 C | 1.226 MWh | 2.270 MWh | [3.881, 5.250] | gap 58.15 → 35.29 pp (improves, still under-band) |
| 650FF Tmin | −17.41 °C | −16.20 °C | [−17.8, −16.7] | leaves band upward; honest known_fail 6.09 pp (§LIMIT-43) |
| 650FF Tmax | 41.89 °C | 46.11 °C | [61.1, 66.8] | gap 34.50 → 27.90 pp (improves; the missing-afternoon-stack problem remains) |

Every other strict row, all four free-float observables for
600FF/900FF/950FF, and Case 950 (0.918 MWh) are byte-identical: 950 is
high-mass and auto-promotes to 9R4C, where the mass-side term was already
removed by LIMIT-17 — the strict-path 5R1C night-vent family is exactly
{650, 650FF}, as the routing split predicted. Run B
(`FLX_R86_VENT_AIR_ONLY=0`) reproduces the pre-flip record bit-for-bit
(strict + output-diff both green against the old baselines), so the
opt-out is a faithful A/B restore of the audited coupling.

The audit's 3.1× mass-side estimate compares to a measured
sensitivity: removing the term recovered 1.21 K of the −17.41 °C
over-chill (7.0% of the recorded value) and lifted 650 C by +1.044 MWh
(+85%), consistent with the term having acted through the mass most of
the year while the air-side exchange does the ISO-correct work.

Re-records in the same PR: strict JSON (3 rows + dated _doc), output-diff
CSV (650/C, 650FF/Tmin, 650FF/Tmax), §7 headline pins + re-emitted
workflow_results.csv; §6 (WD100–600 cases), fabric (600/900/950), hotloop
goldens (Case 600), grid-adapter baseline (Case 600), and the #4536 vent
contracts all verified unchanged. Checker recount: 9 PASS / 33 KNOWN-FAIL
/ 0 REGRESSION (ff_650ff_min pass → known_fail).
