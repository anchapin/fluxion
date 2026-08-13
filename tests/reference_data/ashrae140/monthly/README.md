# ASHRAE 140 — Monthly Energy Reference Data

Monthly heating/cooling reference data for the ASHRAE 140 blind validation
suite, added in **issue #1165** (Phase D measurement infrastructure). Consumed
by `tests/ashrae_140_blind_validation.rs`.

| File | Case | Mass | Notes |
|------|------|------|-------|
| `case_600_monthly_reference.csv` | 600 | Low-mass | Baseline, 12 m² south window |
| `case_900_monthly_reference.csv` | 900 | High-mass (200 mm concrete) | Heavy mass variant |

---

## STATUS: v1.3 Reference — documented derivation (issue #2748)

The values in these CSVs are the **v1.3 Phase D monthly reference** for the
blind validation gate `test_monthly_energy_validation_baseline` in
`tests/ashrae_140_blind_validation.rs`. They are **not** direct EnergyPlus
monthly outputs — ASHRAE Standard 140-2023 Annex B publishes annual + peak
figures only, and no monthly breakdown exists in any published source (the
NREL/TP-472-6333 BESTEST report carries monthly figures as plots only, not
citeable tabulated values).

The reference is the **authoritative annual midpoint** (from
`tests/reference_data/zone_balance/case_{600,900}_energy_reference.csv` /
ASHRAE 140-2023 Annex B / NREL/TP-472-6231 Table 3-2) **redistributed across
calendar months by the heating/cooling degree-day share** of the repository's
own hourly Denver TMY3 weather file (balance point 18.3 °C / 65 °F, ASHRAE
Fundamentals Ch. 19 "Degree-Day Method"). This is the only path that:

1. Sums to the authoritative annual midpoint **exactly**
   (`Σ(monthly_mwh) == annual midpoint` by construction).
2. Has a documented physical method (ASHRAE Fundamentals Ch. 19).
3. Does not require new EnergyPlus physics work or a new published monthly
   source.

**Why not direct EnergyPlus output (issue #2748 investigation):**
EnergyPlus 25.2.0 runs on the in-repo Case 600/900 IDFs
(`tests/reference_data/energyplus_models/ashrae_140_case_{600,900}.idf`) but
the runs reproduce cooling ~50× below (Case 600) and ~5× below (Case 900) the
ASHRAE band; Case 900 heating is 8.6× above the band and inverted in
direction from Case 600 — a sign that the IDFs need additional
insulation/glazing/concrete-mass fixes before E+ output can serve as the
monthly reference. Using those numbers would itself be a different shape of
fabrication. The IDF physics fix is tracked under
`docs/KNOWN_ISSUES.md` §SOLAR-02 UPDATE (Issue #2239) / §LIMIT-05 / the
post-#1323 / #1213 / #1328 cooling-load chain.

**Gate status (post-#2748):** `test_monthly_energy_validation_baseline` was
un-`#[ignore]`'d and now runs against this documented-shape reference in CI.
The gate is **reporting-only** (no assert) — the underlying cooling physics
gap means the monthly pass/fail rate will report low, which is the
correct signal. The gate is no longer a "v1.3 DoD blocker" because CI no
longer reports false-confidence pass/fail against fabricated data — every
monthly PASS/FAIL is now against a documented-shape reference derived from
the authoritative annual midpoint.

When the IDF physics is fixed and E+ reproduces the ASHRAE band per Issue
#2239 closure, regenerate with:

```bash
python3 scripts/generate_monthly_aggregate.py --case 600
python3 scripts/generate_monthly_aggregate.py --case 900
```

The aggregator (`scripts/generate_monthly_aggregate.py`) consumes
`tests/reference_data/zone_balance/case_<id>_energy_hourly.csv` (produced
by `tests/reference_data/zone_balance/generate_case_600_900_energy.py` from
the in-repo IDFs) and emits these CSVs in the exact same schema. The
hourly-to-monthly reduction is unit-tested in
`scripts/ci/test_generate_monthly_aggregate.py` (41 tests: boundary-hour
mapping, ±10% acceptance window, multi-zone support, negative-value
clamping, schema round-trip, end-to-end Case 920 sanity check against the
real E+ hourly export).

---

## Methodology (fully reproducible)

For each case and month:

```
monthly_mid (MWh) = ANNUAL_AUTHORITY_MID × MONTHLY_FRACTION
accept_min         = monthly_mid × 0.90      # Phase D ±10% (issue #668)
accept_max         = monthly_mid × 1.10
```

### `ANNUAL_AUTHORITY_MID`
Midpoint of the **ASHRAE 140-2023 Annex B** reference band across BEM programs
— the same authoritative source already used for the annual tolerance tests in
`tests/reference_data/zone_balance/case_{600,900}_energy_reference.csv` and
`src/validation/benchmark.rs`:

| Case | Annual heating (MWh) | Annual cooling (MWh) |
|------|----------------------|----------------------|
| 600  | 4.36 .. 5.79 → 5.075 | 3.92 .. 6.14 → 5.030 |
| 900  | 1.17 .. 2.04 → 1.605 | 2.13 .. 3.67 → 2.900 |

### `MONTHLY_FRACTION`
Heating/cooling degree-day share of each month, computed from the repository's
**own** hourly weather file
`tests/reference_data/weather/denver_tmy3_reference.csv` (the same TMY3 that
drives the simulation: `USA_CO_Golden-NREL.724666_TMY3.epw`):

- Balance-point temperature **18.3 °C (≈65 °F)** — ASHRAE Fundamentals,
  *Degree-Day Method*, Ch. 19.
- Heating degree-hours per month: `Σ max(0, 18.3 − T_hour)`
- Cooling degree-hours per month: `Σ max(0, T_hour − 18.3)`
- Fraction = `month_degree_hours / annual_degree_hours`. Fractions sum to 1.0, so
  monthly values sum back to the authoritative annual midpoint (verified:
  Case 600 → 5.075 / 5.030 MWh; Case 900 → 1.605 / 2.900 MWh).

Computed with hourly data and Python (no mental arithmetic). The degree-day
fractions for Denver are:

```
Month  HDD-frac  CDD-frac
Jan    0.1496    0.0000
Feb    0.1542    0.0000
Mar    0.1328    0.0065
Apr    0.0860    0.0200
May    0.0553    0.0813
Jun    0.0188    0.1878
Jul    0.0110    0.2642
Aug    0.0068    0.2809
Sep    0.0278    0.1357
Oct    0.0806    0.0216
Nov    0.1329    0.0019
Dec    0.1442    0.0001
```

---

## Caveats / known limitations of the v1.3 reference

1. **Temperature-only shape.** Degree days capture the temperature-driven share
   of load but **not** solar gains. Case 600 cooling is strongly solar-driven
   (12 m² south glazing), so the pure CDD shape under-weights shoulder-season
   cooling. A direct E+ run will shift more cooling into spring/autumn.
2. **No thermal-lag modelling.** Case 900's 200 mm concrete stores daytime solar
   gains and releases them overnight/into shoulder hours, flattening and shifting
   the cooling shape vs. the pure CDD distribution.
3. **Per-month PASS/FAIL against this v1.3 reference is reported but not asserted.**
   The test `test_monthly_energy_validation_baseline` runs and prints the
   monthly breakdown and pass rate, but the engine's cooling under-prediction
   (per `docs/KNOWN_ISSUES.md` §SOLAR-02 UPDATE / Issue #2239) means the
   pass rate will be low until the cooling physics is fixed. Once Issue #2239
   closes, the test can be hardened to assert a Phase D pass-rate target.

---

## <a id="regen"></a>Regeneration path when direct E+ output lands

The `scripts/generate_monthly_aggregate.py` aggregator is the in-place
replacement for these CSVs. Once the IDF physics is fixed and E+ reproduces
the ASHRAE band, run:

1. The hourly regenerator (`tests/reference_data/zone_balance/generate_case_600_900_energy.py`)
   against the corrected IDFs — produces
   `tests/reference_data/zone_balance/case_<id>_energy_hourly.csv`.
2. The monthly aggregator (`scripts/generate_monthly_aggregate.py --case 600`)
   — consumes the hourly CSV and writes the same-schema
   `case_<id>_monthly_reference.csv` in this directory. The aggregator's
   `--validate` subcommand re-runs the reduction and asserts Σ(monthly) is
   inside the annual band (catches E+ regenerator regressions before they
   reach the monthly CSVs).
3. `cargo test --test ashrae_140_blind_validation -- --nocapture` to
   re-print the Phase D monthly breakdown against the now-authoritative
   E+ monthly totals.
4. Refresh `docs/archive/planning/BLIND_VALIDATION_RESULTS.md` with the new
   pass rate.
5. (Optional) Drop the §Caveats block above once the engine cooling gap
   closes (Issue #2239) and the gate can be hardened from reporting-only
   to a Phase D ±10% assert.

---

## Sources / citations

- **ASHRAE Standard 140-2023**, *Standard Method of Test for the Evaluation of
  Building Energy Analysis Computer Programs*, Annex B (annual + peak reference
  bands across BEM programs). → annual authoritative bands.
- **ASHRAE Handbook — Fundamentals**, *Energy Estimating and Modeling Methods*
  (Degree-Day Method, balance-point 18.3 °C / 65 °F). → monthly fraction method.
- **NREL/TP-472-6333** — Judkoff & Neymark, *Home Energy Rating System Building
  Energy Simulation Test (BESTEST)*, IEA SHC Task 12. → BESTEST case definitions.
- **TMY3 weather**: `USA_CO_Golden-NREL.724666_TMY3.epw` (39.74°N, 105.18°W),
  bundled with EnergyPlus; repo copy at
  `tests/reference_data/weather/denver_tmy3_reference.csv`. → degree-day source.
- Repo conventions: `ARCHITECTURE.md` §Reference Data; `AGENTS.md`.
