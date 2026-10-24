# Free-float energy-balance audit, round-92 refresh (r86 → r92)

**Date:** 2026-10-23 (loop round 92)
**Build:** `ae66d514` (develop tip) + the round-92 contract-pin branch at `48c6c378` — engine identical to develop; round-91 flips 1–3 all default (strict gate reproduces the round-91 record: 16 PASS / 26 KNOWN-FAIL / 0 REGRESSION, output-diff 42/42 at +0.0000%).
**Method:** `FLX_FF_EB_AUDIT=1` dump on the free-float strict-gate leg, offline harness `scripts/audit_free_float_balance.py` per case. One method fix vs earlier runs: the dump must run `--test-threads=1` — the four FF cases emit FFAUDIT lines concurrently otherwise and the stderr lines interleave mid-token (the parallel run of this refresh produced a torn `lag=10.test` token and an unparseable log).

## What changed since the r86 audit (2026-10-22)

The r86 audit ran before round-91 flips 2–3. Since then, on the default build:

- **Flip 2** retired the direct solar splits on the spec path: `solar_distribution_to_air = 0.0` and `solar_beam_to_mass_fraction = 0.0` (`thermal_model_core/mod.rs:2530-2545`). The r86 annual table's "gains into air 11,570 MJ / into mass 8,099 MJ" buckets are structurally **zero** now — the harness reports `gains_air = 0.0`, `gains_m = 0.0` for both cases.
- **Flip 3** made the Table 5-138 per-class routing the default surface-pool mass route (`FLX_R91_T5138`, class fractions renormalized over the 0.9664 surface-row sum). Transmitted solar now reaches the network **exclusively through the surface-pool flux `phi_st`**: the class drives on the mass side, plus the §C.3 lag-site air-share term on the air side.

## 600FF — annual integral, r86 → r92 (MJ)

| bucket | r86 | r92 |
|---|---|---|
| gains into air (phi_ia) | 11,570 | 0.0 |
| gains into mass (phi_m) | 8,099 | 0.0 |
| vent exchange | 0.0 | 0.0 |
| envelope (sol-air), air node | −15,220 | −20,098.9 (ext_air) |
| envelope (sol-air), mass node | (in r86's envelope row) | −22,668.8 (ext_m) |
| ground, air node | −2,899 | −3,695.5 |
| air↔mass exchange | +413 | +9,715.1 (to_mass) |
| net mass storage | −29.5 | −26.3 |
| net air storage | (small) | −1.2 |

The envelope and air↔mass magnitudes grew substantially: with no direct-to-air solar, the zone runs on the surface-pool route alone, and the swing-day storage swings carry the solar (below). The annual storage residuals still close (−26.3 MJ mass / −1.2 MJ air over a 8,760 h year), as they must for a closed loop.

## 650FF — annual integral, r86 → r92 (MJ)

| bucket | r86 | r92 |
|---|---|---|
| vent via air node | −6,509 | −20,599.7 (vent_air) |
| vent via mass node | −20,469 | 0.0 (vent_m) |
| envelope (air) | −5,074 | −11,477.1 (ext_air) |
| mass sol-air | +6,013 | −18,127.8 (ext_m) |
| air↔mass | −3,876 | +12,512.1 (to_mass) |
| ground | (small) | −2,286.9 |
| net storage | (r86: pass-through) | −0.6 air / −30.0 mass |

650FF coverage note: both 5R1C cases dump complete years (8,760 FFAUDIT_Z rows each, last hour 8759 present in both; the cases' KNOWN-FAIL Tmax band asserts — 600FF 55.06 °C vs [62.4, 68.4], 650FF 54.04 °C vs [61.1, 66.8] — fire after the simulation loop, so the dumps are unaffected). The 900/950FF pair runs the 9R4C family and emits no FFAUDIT lines, as at r86.

## Swing-day structure (600FF, day 18 of year, T_air range 52.78 K)

Hourly `phi_st` (the transmitted pool flux) rises 0 → 8,308 W between 07:00 and 12:00 and the mass-node storage absorbs it nearly one-to-one through the class drives (stor_m peaks +26.4 MJ/h at 11:00). The air node's books show a separate `phi_st` bucket of **+56.22 MJ on that day** — the §C.3 lag-site air-share coupling — riding on top of the class drives. 650FF's swing day 17 shows the same structure (phi_st air bucket +56.22 MJ; the two cases share window geometry, so the transmitted flux matches).

## Structural finding — candidate solar over-routing under the T5138 default (recorded, not acted on)

Reading the two route sites on the default build:

- The class drives carry `Σ_c (f_c/0.9664)·phi_st = 1.0·phi_st` — the renormalized fractions sum to exactly 1 (`step_5r1c.rs`, T5138 branch at the `phi_c` composition site).
- The §C.3 lag-site air-share term is **still active** in the air ODE under the same default: `lag_input_r77` is scaled by `(1 − r88_share_zone)` (`step_5r1c.rs:1640-1645`, `FLX_R88_PHIST_MASS_SHARE` default on), and that scaled term enters the air numerator every sub-step.

Under the retained §C.3 arm (the falsy-token route) those two shares partition `phi_st` exactly ("air + mass = phi_st", the site's own comment). Under the T5138 default the partition is gone: the classes receive the full pool flux **and** the air receives the `(1 − s_share)` share, so the composed network receives more than the transmitted pool flux by the air term. The flip-3 "energy-conserving by construction" comment covers only the class drives. The ASHRAE 140 Table 5-138 reference semantics route the post-sink solar to surface destinations plus a window-loss row — nothing to air — which makes the §C.3-style air term the odd one out.

This is **recorded, not repaired**: the standing direction routes ambiguous engine-semantics through Alex, and the fix (either deriving the class drives from the post-air-share remainder or zeroing the lag-site air term under T5138) changes default-build behavior and requires the full flip discipline (battery + re-record). The precise annual excess needs one more dump field (per-zone `h_tr_is` or the hour's `s_share` in FFAUDIT_Z) — a one-line emitter addition proposed as the next slice's first item; the swing-day `phi_st` air bucket (+56.22 MJ/day) is the current best measured bound.

## Harness validation, r86 → r92

| metric | r86 | r92 |
|---|---|---|
| class-storage cross-check max gap (600FF) | 1,309 kJ | 26,938.6 kJ |
| class-storage cross-check max gap (650FF) | 4,331 kJ | 27,344.9 kJ |
| closure residual med, air (600FF / 650FF) | 134% / 158% | 167.8% / 199.2% |
| closure residual med, mass (600FF / 650FF) | 363% / 364% | 40.2% / 75.7% |

The ~20× cross-check growth tracks flip 3: the identity `Σ_c FFAUDIT_D ≈ cm·ΔT` assumes no t_s-first layer **and no class-direct drive**; with the classes now charging directly from the pool at fixed fractions, the per-class vs composite divergence is the new physics, not an instrumentation fault. The mass-node relative residual collapsed because the mass flux scale grew with the class drives.

## Open items carried from r86

1. The §6.2.1 Table 6-1 day-list debt stays with Alex (unchanged).
2. Surface-pool release timing (r86's item 2): the harness now exposes the `phi_st` air bucket per swing day, but the pool-state dump (`t_s` per class) is still absent.
3. The 9R4C audit row for 900FF/950FF (r86's item 3) is still open; those cases run the 9R4C family and emit no FFAUDIT lines.
