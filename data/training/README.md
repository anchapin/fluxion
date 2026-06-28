# `data/training/` — Physics-Extracted Surrogate Training Samples

This directory holds the **physics-extracted** training samples consumed by
`tools/train_surrogate.py` (Issue #553 / Issue #1338).

> **Physics is the only legitimate training source.** Per Issue #1286
> `SurrogateInputs::from_synthetic` was closed on the Rust side; per Issue
> #1338 the Python training pipeline now refuses to run without physics
> samples here, unless `--allow-synthetic-for-benchmark-only` is passed.

---

## Required EnergyPlus run inputs

The samples are produced by a parametric EnergyPlus sweep driven from the
existing fluxion reference pipeline:

| Input                       | Source                                                                                  |
|-----------------------------|------------------------------------------------------------------------------------------|
| Parametric IDFs (Cases 600–960) | `tools/generate_case_900_idf.py` and friends (ASHRAE 140 reference suite)             |
| EPW weather files           | `data/weather_locations.json` (canonical TMY3 locations)                                  |
| Output extraction           | `tools/ep_oracle.py` (per-surface energy deltas → heating / cooling loads)                |
| Orchestration               | `scripts/cloud_campaign_manager.py` / `scripts/autonomous_parameter_sweep.py` (cloud runs) |

Each combination of (building case × weather location × parametric
variation) yields one row of `(X, y)` features. See
`tools/ep_oracle.py::extract_per_zone_loads` for the canonical extraction
shape.

---

## Expected schema

`tools/train_surrogate.py::load_training_data()` consumes one or more files
matching the glob `samples_*.csv` (newest by mtime wins). Each CSV must
expose exactly these columns:

| Column            | dtype   | Units | Notes                                                     |
|-------------------|---------|-------|-----------------------------------------------------------|
| `outdoor_temp`    | float32 | °C    | Clipped to `[-20, 45]`                                     |
| `heating_setpoint`| float32 | °C    | Clipped to `[15, 25]`                                      |
| `cooling_setpoint`| float32 | °C    | Clipped to `[20, 30]`                                      |
| `hour_of_day`     | int     | 0–23  | Hour-of-year index within the simulation                   |
| `day_of_year`     | int     | 1–366 | Day index                                                  |
| `month`           | int     | 1–12  | Derived from `day_of_year`                                 |
| `u_value`         | float32 | W/m²K | Clipped to `[0.1, 2.0]`                                    |
| `wwr`             | float32 | –     | Window-to-wall ratio, `[0.1, 0.9]`                         |
| `heating_load`    | float32 | W     | **Target 1** — derived from EP `Zone Ideal Loads Supply Air` |
| `cooling_load`    | float32 | W     | **Target 2** — derived from EP `Zone Ideal Loads Supply Air` |

Numeric columns are cast to `float32` on load (matches the input contract
of the MLP/XGBoost/RF trainers).

---

## Minimum sample count

| Use case                     | Min samples | Rationale                                                |
|------------------------------|-------------|----------------------------------------------------------|
| Smoke / sanity               | 1 000       | Just enough to exercise the training loop                |
| **Production surrogate v3.1**| **10 000**  | Matches Wave 5 retrain (PR #1334) — see Surrogate v3.1   |
| Hardened CI gate             | 50 000      | Recommended for stable regression on the ±15% gate       |

Smaller sample counts will trigger the `train_test_split` warning inside
MLP fit; the script will still run but the trained weights are not
considered production-ready.

---

## Regenerator command

The canonical regeneration pipeline is:

```bash
# 1. (One-time) Author or refresh the parametric sweep spec.
python scripts/autonomous_parameter_sweep.py --spec configs/ashrae140_v3_1.yaml

# 2. Run the EnergyPlus cloud campaign.
python scripts/cloud_campaign_manager.py --spec configs/ashrae140_v3_1.yaml \
    --output-dir data/training

# 3. Concatenate per-run outputs into a single samples_*.csv.
python tools/ep_oracle.py collect --src data/training --out data/training/samples_$(date -u +%Y%m%dT%H%M%SZ).csv

# 4. Train the surrogate (production path; no synthetic fallback).
python tools/train_surrogate.py --data-dir data/training --output-dir models
```

For benchmark/CI harnesses only:

```bash
python tools/train_surrogate.py --allow-synthetic-for-benchmark-only \
    --run-benchmark --shap-analysis --ensemble
```

The benchmark path uses `tools.train_surrogate.generate_synthetic_thermal_data`
and **MUST NOT** be used to produce a model that is committed back to
`models/`.

---

## Why this contract exists

* **Issue #1286** — Closed `SurrogateInputs::from_synthetic` in the Rust
  core so the ONNX runtime cannot silently consume random tensors.
* **Issue #1139** — Identified that ad-hoc EnergyPlus reference runs were
  being shadowed by in-memory generators.
* **Issue #719**  — Tracked the broader "ground truth = EnergyPlus" rule
  that this contract enforces at the training boundary.
* **Issue #1338** — Closes the Python-side gap: prior to this audit
  `tools/train_surrogate.py:414-415` silently fell back to
  `generate_synthetic_thermal_data(10000)` whenever `data/training/` was
  empty, bypassing the #1286 fix.