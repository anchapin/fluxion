# ASHRAE 140 — Monthly Energy Reference Data

Monthly heating/cooling reference data for the ASHRAE 140 blind validation
suite, added in **issue #1165** (Phase D measurement infrastructure). Consumed
by `tests/ashrae_140_blind_validation.rs`.

| File | Case | Mass | Notes |
|------|------|------|-------|
| `case_600_monthly_reference.csv` | 600 | Low-mass | Baseline, 12 m² south window |
| `case_900_monthly_reference.csv` | 900 | High-mass (200 mm concrete) | Heavy mass variant |

---

## ⚠️ STATUS: PLACEHOLDER — v1.3 DoD BLOCKER (issue #2677)

> **DO NOT use these CSVs for pass/fail gating.** They are **placeholder**
> values — a degree-day-derived *shape* applied to the authoritative annual
> midpoint, **not** direct EnergyPlus monthly outputs. The dependent Phase D
> ±10% monthly gate (`test_monthly_energy_validation_baseline` in
> `tests/ashrae_140_blind_validation.rs`) is **`#[ignore]`'d as of #2677** so
> CI does not report a pass rate against fabricated data. Treat any per-month
> PASS/FAIL against these values as **not signal**.

>>> **REPLACE when**: a direct EnergyPlus ≥25.2.0 run produces monthly rollups
>>> of `Zone Air System Sensible Heating/Cooling Energy` for Cases 600 and 900.
>>> The Case 900 IDF is currently "pending" in
>>> `tests/reference_data/energyplus_models/` (per
>>> `case_900_energy_reference.csv`), and the regenerator
>>> (`generate_case_600_900_energy.py`) has not been run for either case — see
>>> the [TODO](#todo) section below.

**Why no authoritative monthly data exists in-repo (issue #2677
investigation, 2026-08-12):**

1. **ASHRAE Standard 140-2023 Annex B** publishes **annual** and **peak**
   results across reference programs (EnergyPlus, ESP-r, TRNSYS, DOE-2, …).
   The standard does **not** publish a **monthly** breakdown for the
   qualification cases.
2. The original IEA SHC Task 12 / BESTEST report (Judkoff & Neymark,
   NREL/TP-472-6333) and the EnergyPlus BESTEST validation reports contain some
   monthly figures **as plots**, but no verified, citeable tabulated monthly
   values for Cases 600/900 could be extracted through the available tooling.
3. The repo's `tests/reference_data/zone_balance/` directory holds **annual**
   bands for Cases 600/900 (sourced from NREL/TP-472-6231 Table 3-2 / ASHRAE
   140-2023 Annex B) and **hourly** EnergyPlus CSVs for Cases 920/950/960 — but
   **no hourly or monthly EnergyPlus output for Cases 600/900**. Without hourly
   E+ data for 600/900, monthly totals cannot be derived.

Per the #1165 fallback clause ("do NOT fabricate data… leave a placeholder
file with a TODO, and document what's needed") and the #2677 v1.3 DoD
resolution, this directory ships a **reproducible placeholder reference** plus
a clear replacement path, **and the dependent CI gate is `#[ignore]`'d** so the
placeholder cannot produce false-confidence pass/fail signals.

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

## Caveats / known limitations of the placeholder reference

1. **Temperature-only shape.** Degree days capture the temperature-driven share
   of load but **not** solar gains. Case 600 cooling is strongly solar-driven
   (12 m² south glazing), so the pure CDD shape under-weights shoulder-season
   cooling. A direct E+ run will shift more cooling into spring/autumn.
2. **No thermal-lag modelling.** Case 900's 200 mm concrete stores daytime solar
   gains and releases them overnight/into shoulder hours, flattening and shifting
   the cooling shape vs. the pure CDD distribution.
3. **Per-month PASS/FAIL against this placeholder carries no signal.** The
   ±10% window is applied around a placeholder midpoint whose *shape* is
   approximate; month-level pass/fail is therefore indicative only. As of #2677,
   the dependent test (`test_monthly_energy_validation_baseline`) is
   `#[ignore]`'d so CI does **not** report a pass rate against these values.

---

## <a id="todo"></a>TODO — Replace placeholder monthly reference with direct E+ output

Target: before Phase D acceptance (#668); tracked as v1.3 DoD blocker in
issue #2677. Steps:

1. Stand up an EnergyPlus ≥ 25.2.0 environment (path expected at
   `/usr/local/EnergyPlus-25-2-0/energyplus`, per
   `tests/reference_data/zone_balance/generate_case_600_900_energy.py`).
2. Add the **Case 900 IDF** to `tests/reference_data/energyplus_models/`
   (currently noted as "pending" in the zone_balance generator) and register it
   in `generate_case_600_900_energy.py`.
3. Extend the generator to also emit a **monthly** rollup (sum hourly
   `Zone Air System Sensible Heating/Cooling Energy` by month) for Cases 600 and
   900, writing the two CSVs in this directory **with the same column schema**
   (`month, *_mid_mwh, *_accept_min_mwh, *_accept_max_mwh`). Convert the ±10%
   Phase D window around the published E+ monthly midpoint.
4. **Remove the `#[ignore]`** on `test_monthly_energy_validation_baseline` in
   `tests/ashrae_140_blind_validation.rs` (added in #2677) so the Phase D
   monthly gate runs in CI against the now-authoritative values.
5. Re-run `cargo test --test ashrae_140_blind_validation -- --ignored --nocapture`
   and refresh `docs/archive/planning/BLIND_VALIDATION_RESULTS.md`.
6. Remove the "PLACEHOLDER" banner from both CSV headers and the §STATUS block
   at the top of this README once authoritative values land.

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
