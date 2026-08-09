# ASHRAE 140 — Blind Validation Results

Baseline measurement for the ASHRAE 140 blind validation suite
(`tests/ashrae_140_blind_validation.rs`) when all corrections are disabled
(`ValidationMode::Blind`). Refreshed by running:

```bash
cargo test --test ashrae_140_blind_validation -- --nocapture
```

This file is **living output** — the numbers below are a snapshot from the run
that landed issue #1165. Re-run the test and update the tables when the physics
or reference data change.

---

## §7 Acceptance criteria status

| Tolerance | Criterion | This Run | Met? |
|-----------|-----------|----------|:----:|
| Annual energy | ±15% (annual heating & cooling) | see annual run | ❌ |
| **Monthly energy** | **±10% (Phase D, #668/#1165)** | **11.36% pass (5/39)** | ❌ |
| Peak loads | ±15% (peak heating & cooling) | see annual run | ❌ |
| Free-floating temp | ±1.0 °C (min/max) | see annual run | ❌ |

The monthly row is **newly measurable** as of #1165 — previously the suite did
not measure monthly energy at all. The measurement infrastructure now exists;
the physics that would raise the pass rate is tracked in #1163 and #1168.

> **Reporting-only.** The blind validation suite never fails the build. It
> prints pass/fail to stdout and updates this file. Regressions are caught by
> the unit-test suite, not by this reporting test.

---

## Monthly pass rate (tracked separately — issue #1165)

Phase D criterion: each calendar month within **±10%** of the reference
midpoint, reported per case × month × (heating|cooling).

| Scope | Total metrics | Passed | Failed | Pass rate | MAE |
|-------|--------------:|-------:|-------:|----------:|----:|
| Cases 600 + 900, monthly | 39 | 5 | 34 | **11.36%** | 149.0% |

### Snapshot — Case 600 Monthly Heating (MWh)

| Month | Sim | Ref mid | ±10% window | Error | Status |
|-------|------:|--------:|-------------|------:|:------:|
| Jan | 0.8436 | 0.7592 | 0.6833 .. 0.8351 | 11.1% | FAIL |
| Feb | 0.6459 | 0.7826 | 0.7043 .. 0.8608 | 17.5% | FAIL |
| Mar | 0.4526 | 0.6740 | 0.6066 .. 0.7414 | 32.8% | FAIL |
| Apr | 0.1831 | 0.4365 | 0.3928 .. 0.4801 | 58.1% | FAIL |
| May | 0.0369 | 0.2806 | 0.2526 .. 0.3087 | 86.8% | FAIL |
| Jun | 0.0069 | 0.0954 | 0.0859 .. 0.1050 | 92.8% | FAIL |
| Jul | 0.0065 | 0.0558 | 0.0502 .. 0.0614 | 88.4% | FAIL |
| Aug | 0.0279 | 0.0345 | 0.0311 .. 0.0380 | 19.2% | FAIL |
| Sep | 0.1450 | 0.1411 | 0.1270 .. 0.1552 | 2.8% | **PASS** |
| Oct | 0.4065 | 0.4090 | 0.3681 .. 0.4499 | 0.6% | **PASS** |
| Nov | 0.6882 | 0.6745 | 0.6070 .. 0.7419 | 2.0% | **PASS** |
| Dec | 0.8638 | 0.7318 | 0.6586 .. 0.8050 | 18.0% | FAIL |

### Snapshot — Case 900 Monthly Heating (MWh)

| Month | Sim | Ref mid | ±10% window | Error | Status |
|-------|------:|--------:|-------------|------:|:------:|
| Jan | 0.3440 | 0.2401 | 0.2161 .. 0.2641 | 43.3% | FAIL |
| Feb | 0.2340 | 0.2475 | 0.2227 .. 0.2722 | 5.5% | **PASS** |
| Oct | 0.1314 | 0.1294 | 0.1164 .. 0.1423 | 1.6% | **PASS** |
| Dec | 0.3504 | 0.2314 | 0.2083 .. 0.2546 | 51.4% | FAIL |

(Case 600/900 monthly cooling is currently far below the reference band across
all months — the cooling-energy physics is the dominant open issue tracked in
#1168. Full per-month cooling rows are printed by the test run.)

### Interpretation

- Heating in shoulder/summer months is over-suppressed (the model under-heats
  once solar gains appear), while winter heating is in the right ballpark for
  Case 600. This is consistent with the annual-heating profile.
- Cooling is structurally low everywhere — known physics gap (#1168).
- A handful of months pass (Case 600 heating Sep/Oct/Nov; Case 900 heating
  Feb/Oct), which is expected to be coincidental against the **interim**
  reference rather than evidence the physics is correct.

---

## ⚠️ Monthly reference status: INTERIM

The monthly reference values are **not** direct EnergyPlus outputs. They are a
reproducible, degree-day-derived interim reference used to stand up the
measurement infrastructure:

- **Annual authority**: ASHRAE Standard 140-2023 Annex B reference bands (the
  same source used for the annual tolerance tests in
  `tests/reference_data/zone_balance/`).
- **Monthly shape**: heating/cooling degree-day share computed from the
  repository's own `tests/reference_data/weather/denver_tmy3_reference.csv`
  (balance point 18.3 °C, ASHRAE Fundamentals degree-day method).
- Monthly values sum back to the authoritative annual midpoint (verified).
- Per-month acceptance window is ±10% around the interim midpoint.

**Before Phase D acceptance closes**, the interim reference must be replaced
with direct EnergyPlus monthly totals. See
`tests/reference_data/ashrae140/monthly/README.md` → "TODO — Replace interim
monthly reference" for the exact regeneration steps.

---

## How monthly energy is computed

The simulation accumulates heating/cooling energy per timestep (in kWh, via
`ThermalModel::get_heating_energy_kwh()` / `get_cooling_energy_kwh()`). The
test snapshots the cumulative value before/after each hourly `step_physics`
call and buckets the delta into the current calendar month (hour-of-year →
month via the non-leap TMY calendar). kWh ÷ 1000 → MWh. By construction
Σ(monthly) == annual for the same run (the test asserts this internally and
warns on any aggregation drift > 1e-6 MWh).
