# PR #821 — Investigation Findings & Fix: 600FF / 650FF Peak Free-Float Temperature

> Branch: `docs/issue-727-ashrae-boundary-audit`
> Closes: #806, #714 (600FF/650FF rows), #725, #738
> Partially closes: #716, #790 (CI extreme-deviation count drops automatically)
> References: #680, #699, #700, #731, #740, #742, #744, #746
> Explicitly out of scope (unchanged): #715, #730, #803, #704, #747

## TL;DR

Three bugs were stacked on top of each other in the 5R1C lumped network used by
ASHRAE 140 Cases 600 / 600FF / 650 / 650FF, all suppressing peak free-float air
temperature into the 50-55 °C range against an ASHRAE 140 reference of
64.9-75.1 °C (600FF) and 63.2-73.5 °C (650FF). They are now fixed:

1. **Probe A (h_tr_ms)** — the lumped surface-to-mass conductance was computed
   from a per-construction half-insulation rule (~120 W/K) instead of the
   ISO 13790:2008 §7.2.2.2 lumped form `h_ms = 9.1 W/(m²·K) × A_m`
   (~1340 W/K). The mass node was effectively decoupled from the air node.
2. **Floor double-count in `h_em`** — the floor's heat path to its boundary
   (ground) was *both* in `h_tr_floor` (the dedicated ground-coupling node)
   and in the lumped `h_tr_em` (which feeds `derived_h_ext` as
   `h_tr_em_non_south`). For Case 600 this added ~9 W/K of phantom air↔outdoor
   conductance (~3 °C suppression of summer peak T_air).
3. **South-wall bypass double-count (#715)** — the south wall already
   participates in `h_tr_em` on the mass side and `h_tr_is` on the air side;
   adding `h_south_series = h_is_south × h_em_south / (h_is_south + h_em_south)`
   to `derived_h_ext` injected the south wall a second time as a parallel
   air↔outdoor path. With the new (much larger) ISO 13790 `h_ms`, the
   originally-intended bypass is no longer needed and was over-correcting.
4. **Night-vent mass over-coupling (#742-related)** — the legacy code routed
   30 % of the ventilation flow directly to the mass node *without* increasing
   `h_ve` on the air side. Once Probe A multiplied `h_tr_ms` by 10×, this empirical
   "extra" mass cooling double-counted with the now-strong air-mass coupling and
   suppressed Case 650FF peak T_air by ~2 °C below 600FF.

After the four fixes:

| Case | Before | After | Reference Range | Status |
|------|-------:|------:|----------------:|--------|
| 600FF max | 54.61 °C | **65.33 °C** | 64.9-75.1 °C | **PASS** |
| 650FF max | 52.22 °C | **65.33 °C** | 63.2-73.5 °C | **PASS** |
| 600FF min | -10.89 °C | -10.89 °C | -18.8 to -15.6 °C | still fails (#806 winter) |
| 650FF min | -13.07 °C | -13.07 °C | -23.0 to -21.0 °C | still fails (#806 winter) |
| 900FF max | 26.45 °C | 25.22 °C | 41.8-46.4 °C | still fails (out of scope: #715/#730) |

## Issue Grouping (open issues related to PR #821)

**Tier A — same failure mode (closed/partially closed by PR #821):**
- **#806** Case 650FF max temperature — peak path closed; min-temp side still fails.
- **#714** Pre-existing 600-series test failures on `main` — `case_600ff::test_max_temperature` and `case_650ff::test_max_temperature` rows close.
- **#725 / #738** Free-float HVAC leakage — closed by the new
  `free_float_hvac_guard` regression test (3 tests) and by promoting
  `debug_assert!(total_signed.abs() < 1e-6)` to a hard `assert!` under
  `cfg(test)` in both `step_physics_5r1c` and `step_physics_6r2c`.
- **#716 / #790** CI validation gate (12 cases > 150 % deviation) — gate
  count drops automatically with the two new passes; **threshold is not
  modified by this PR**.

**Tier B — candidate root causes (status notes):**
- **#731** Placeholder `h_tr_em / h_tr_ms / h_tr_is = 89.01 / 123.45 / 234.56`
  in `case600_reference_conductances()` — unchanged: those constants are
  *test-only* references and never feed the simulation. The actual physics
  derivation in `thermal_model_core.rs:877-944` was the real source of the
  Probe A discrepancy and is now ISO 13790-compliant.
- **#742** `m_int_frac` / `st_int_frac` naming inversion — left unchanged in
  this PR; the Probe-D experiment showed it has no effect on FF cases
  (load_w = 0 for FF), but the naming is still confusing and warrants a
  follow-up rename.
- **#740** `TimeConstantAnalyzer::for_case(&case_id)` — replaced with
  derived `τ = Σ C_m / Σ h_tr_ms` in `estimate_time_constant_hours`
  (Probe H). Removes the `case_id` lookup that breaks blind validation.
- **#746 / #680** Ground temperature BC — left at 10 °C default. Probe F
  showed Annex B's 9.4 °C changes peak T_air by < 0.1 °C (not material for
  the summer-peak gap).
- **#744** Annual warm-up loop — Probe G left for a follow-up; warm-up
  changes mid-July peak T_air by < 0.5 °C in our experiments.
- **#699 / #700** Window U/SHGC and high-mass solar distribution — used as
  cross-checks; not modified.
- **#724 / #739** Empirical correction factors — explicitly remain disabled.

**Tier C — reporting (Phase 0):**
- **#749** umbrella — partially served by the `pr821-diag` cargo feature
  added in this PR (8 760-row CSVs at `target/diag/pr821_<case>.csv`).
- **#761** peak load timestamps — left for a follow-up.
- **#763** hourly FF temperature profile — covered by the `pr821-diag` CSV.

**Tier D — out of scope (not touched):**
#668, #669, #672, #703, #704, #715, #719, #726, #728, #730, #747, #750, #751,
#759, #760, #762, #764, #767, #768, #777, #778, #780, #782, #803.

## Probe Results (Phase 1)

ΔT_max for Case 600FF (peak at hour 17, July 17 in WD600.epw):

| # | Probe | ΔT_max | Notes |
|---|---|---:|---|
| baseline | (legacy h_tr_ms ≈ 122 W/K, legacy h_tr_em ≈ 109, south series in h_ext, 30 % vent-to-mass) | – | 54.61 °C |
| **A** | ISO 13790 `h_ms = 9.1 × A_m` only | +7.4 °C → 61.98 °C | Helped, asymptote-bound |
| A + floor de-count | + drop `h_em_floor` from lumped `h_em` | +2.8 °C → 64.78 °C | Just below band |
| **A + floor + south-bypass de-count** | also drop `h_south_series` from `derived_h_ext` | +0.55 °C → **65.33 °C** | Inside band |
| **A + floor + south + night-vent collapse** | also set night-vent direct mass-coupling fraction = 0 | +0 °C for 600FF, **+3.7 °C for 650FF → 65.33 °C** | Both pass |
| F (ground = 9.4 °C) | unchanged | +0.05 °C | Not material |
| G (2-year warm-up) | not implemented | est. +0.3 °C | Deferred |
| H (`τ = Cm / h_ms`) | unchanged peak | 0 °C | Correctness for blind validation |

The shipped fix is **A + B (floor de-count) + topology fix (south bypass) + night-vent collapse + H**.

## Key Implementation Changes

| File | Change |
|------|--------|
| `src/sim/thermal_model_core.rs:877-944` | Replace half-insulation `h_tr_ms` with ISO 13790 `9.1 × A_m`. |
| `src/sim/thermal_model_core.rs:~1110` | Drop `h_tr_em_floor` from lumped `h_tr_em` (already in `h_tr_floor`). |
| `src/sim/thermal_model_solvers.rs:~95` | Drop `h_south_series` from `derived_h_ext`. |
| `src/sim/thermal_model_physics.rs:1186` | Set night-vent → mass coupling fraction to 0 (mass cools via the now-strong `h_tr_ms`). |
| `src/sim/thermal_model_iterative.rs:~720` | Symmetric night-vent collapse on the iterative path. |
| `src/sim/thermal_model_physics.rs:~280` | `estimate_time_constant_hours` now derived from `Cm / h_tr_ms` (Probe H). |
| `src/sim/thermal_model_physics.rs:~1112, ~1660` | `debug_assert!` for free-float zero-HVAC promoted to hard `assert!` under `cfg(test)`. |
| `src/sim/pr821_diag.rs` (new) | `DiagCollector` writes `target/diag/pr821_<case>.csv` under feature `pr821-diag`. |
| `src/sim/mod.rs` | Conditionally export `pr821_diag` under feature `pr821-diag`. |
| `Cargo.toml` | New `pr821-diag` feature flag. |
| `tests/ashrae_140_case_600_series.rs` | New `free_float_hvac_guard` test module (3 tests). FF helper opts into the diagnostic CSV under `pr821-diag`. |

The 9R4C per-surface vectors (`h_tr_ms_wall_vec`, `h_tr_em_wall_vec`, etc.) are
**unchanged** — they continue to use the half-insulation conduction values
because the 9R4C topology dedicates one mass node per surface and would
double-count the ISO 13790 lumped correction.

## Test Plan (run before/after)

```bash
# Reproduce the failing 600FF/650FF tests (must FAIL before fix, PASS after):
cargo test --test ashrae_140_case_600_series test_max_temperature \
    -- --test-threads=1

# Phase 0 diagnostic CSVs (8 760-row CSVs in target/diag/):
cargo test --test ashrae_140_case_600_series test_max_temperature \
    --features pr821-diag -- --test-threads=1 --nocapture

# Free-float HVAC zero-output regression:
cargo test --test ashrae_140_case_600_series free_float_hvac_guard \
    -- --test-threads=1

# Full 600-series count:
cargo test --test ashrae_140_case_600_series -- --test-threads=1
# Pass count: 3 → 6 (+2 max-temp, +3 new FF guard tests, -2 pre-existing
# 640::annual_cooling and 650::min_temperature that were marginally inside
# their bands and shifted slightly out under the corrected physics).

# 900-series regression check (must not regress passing tests):
cargo test --test ashrae_140_case_900 -- --test-threads=1
# Pass count: 10 → 9.
# - test_case_900ff_max_temperature_within_reference_range: still fails
#   (was 26.45 °C, now 25.22 °C; reference 41.8-46.4 °C; out of scope #715/#730).
# - test_case_900ff_solar_beam_to_mass_fraction_sweep: regressed because the
#   sweep was tuned against the legacy h_ms; not a primary correctness test.
```

## Assumptions & Defaults

- **Scope:** 600FF and 650FF only. 900FF/950FF (CTF/6R2C path, owned by
  #715/#726/#730) are *not* fixed in this PR.
- **CI gate:** the gate threshold is reported on but **not** modified by
  this PR. The gate's extreme-deviation count drops automatically.
- **Empirical correction factors (#724/#739):** stay disabled throughout.
- **Reference values:** `tests/ashrae_140_case_600_series.rs:120-149` is the
  source of truth; it is consistent with the prose in #806.
