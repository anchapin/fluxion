# Phase 4a MLP Surrogate Architecture — Input/Output Design & Training Strategy

| Field | Value |
|---|---|
| Status | Draft for review |
| Phase | v3.0 / Phase 4a (MLP baseline) |
| Owner | Surrogate / AI Lead |
| Issue | [#764](https://github.com/...) |
| Parent epic | [#718](https://github.com/...) (v3.0 Surrogate Training & ONNX Export) |
| Data source | [#719](https://github.com/...) (v2.1 Synthetic Data Generation) |
| Follow-on | [#708](https://github.com/...) (Phase 4b MLP+GP ensemble for UQ) |
| Competes with | [#977](https://github.com/...) (XGBoost/xDT export) |
| Architecture | `ARCHITECTURE.md` § Zone Balance → `SurrogateThermalModel` |
| Governance | `docs/SURROGATE_GOVERNANCE.md` (binding) |
| ONNX I/O baseline | `docs/ONNX_INFERENCE_PIPELINE.md` (binding) |
| Holdout spec | `data/ashrae140_holdout.json` (binding) |

---

## 1. Overview

This document defines the **canonical Phase 4a MLP surrogate architecture** for
Fluxion: the 23-feature input schema, the network topology, the training
strategy, the ONNX export contract, and the corresponding Rust `SurrogateInputs`
struct extension. It is the single source of truth for two deliverables in
[#718](https://github.com/...):

1. `scripts/train_surrogate.py` — Python training + export pipeline
2. The Rust `SurrogateInputs` struct extension in `src/ai/surrogate.rs`

**Architecture decision (locked by issue body):** MLP-first (this spec) →
MLP+GP ensemble as the Phase 4b follow-on (unlocks [#708](https://github.com/...)).
XGBoost ([#977](https://github.com/...)) is a parallel research path, not a
replacement.

The Rust inference layer (`SurrogateManager` + `ort` integration in
`src/ai/surrogate.rs`) is already production-ready; this spec defines what
gets loaded into it.

---

## 2. Architecture Alignment

The MLP implements the `SurrogateThermalModel` leaf in the
`ThermalModelTrait` hierarchy documented in `ARCHITECTURE.md` § Zone Balance
(§ "Thermal Model Trait Hierarchy"):

```text
ThermalModelTrait  (sim/thermal_model.rs)
├── PhysicsThermalModel       (analytical 5R1C)
├── SurrogateThermalModel     ← THE MLP SPECIFIED HERE
├── UnifiedThermalModel       (runtime physics/surrogate switch)
└── MockThermalModel          (test placeholder)
```

The single-timestep data flow per `ARCHITECTURE.md` § "Data Flow: Single
Timestep" supplies every input feature in Section 3 from the upstream
physics modules. The MLP replaces only the `step_physics` inner loop; it does
not change the module boundary.

**Governance constraints** (from `docs/SURROGATE_GOVERNANCE.md`, all binding):

| Rule | Section | Implication for this spec |
|------|---------|---------------------------|
| Domain of validity declaration | §1 | Per-feature bounds in §3 must match `SurrogateDomain` |
| Fallback to analytical on OOD | §2 | `SurrogateDomain::is_valid` validates the 17 new fields |
| Versioning schema | §3 | Model file naming `surrogate_1.0.0+onnx1.17.0.onnx` |
| Validation artifacts required | §4 | MAE/RMSE/R²/Max-Error + ASHRAE 140 sweep |
| Composite rules | §5 | Single-zone MLP is a `ComponentSurrogate` |

**ONNX I/O baseline** (from `docs/ONNX_INFERENCE_PIPELINE.md`):

- Input tensor: `X: float32[batch, N]`
- Output tensor: `Y: float32[batch, M]`
- This spec pins `N=23` and `M=2` and names them `inputs` / `loads`.

---

## 3. Input Feature Set (23 features, 17 continuous + 6 one-hot)

**Design rationale.** Fluxion's CTF and FD conduction solvers, ventilation
module, and zone heat balance together depend on the same physical drivers:
outdoor boundary conditions, interior setpoints, building envelope
properties, and temporal thermal lag. The 23-feature set is the minimum
schema that lets a single MLP generalize across ASHRAE 140 Section 7
buildings.

The schema below is the **same** schema declared in
`data/ashrae140_holdout.json` (23-feature MLP input schema) and used by
`scripts/ashrae_benchmark_harness.py` to flag in-domain samples during the
[#719](https://github.com/...) LHC sweep. Any change to the schema must update
all three of: this spec, the holdout JSON, and the benchmark harness.

### 3a. Boundary conditions — 4 features (per-timestep, vary with weather)

| # | Feature | Range | Units | Source module | Notes |
|---|---------|-------|-------|---------------|-------|
| 0 | `exterior_temp` | −50 to 60 | °C | `weather::HourlyRecord.dry_bulb` | Already in current struct |
| 1 | `solar_rad_global` | 0 to 1200 | W/m² | `sim::solar::SurfaceIrradiance.beam + diffuse + ground` | Rename of `solar_rad` |
| 2 | `humidity` | 0 to 100 | % RH | `weather::psychrometrics` | Already in current struct |
| 3 | `wind_speed` | 0 to 20 | m/s | `weather::HourlyRecord.wind_speed` | **New** — drives infiltration ACH and exterior film coefficient |

### 3b. Interior conditions — 3 features (constant per building config)

| # | Feature | Range | Units | Source | Notes |
|---|---------|-------|-------|--------|-------|
| 4 | `zone_temp_setpoint` | 10 to 40 | °C | HVAC schedule | Rename of `zone_temp` |
| 5 | `occupancy_density` | 0 to 10 | ppl/100 m² | Schedule | Already in struct (renamed) |
| 6 | `internal_gains` | 0 to 50 | W/m² | Lights + equipment schedules | **New** |

### 3c. Temporal features — 4 features (cyclical encoding)

Critical for capturing thermal lag and diurnal/seasonal variation in CTF
outputs. Raw hour-of-day and day-of-year would force the network to learn
the modulo boundary; sin/cos pair is the standard surrogate convention
([IBPSA 2024 protocol](https://www.ibpsa.org/) §3.2).

| # | Feature | Encoding | Notes |
|---|---------|----------|-------|
| 7 | `hour_sin` | `sin(2π · hour / 24)` | **New** |
| 8 | `hour_cos` | `cos(2π · hour / 24)` | **New** |
| 9 | `doy_sin` | `sin(2π · doy / 365)` | **New** — day-of-year |
| 10 | `doy_cos` | `cos(2π · doy / 365)` | **New** |

### 3d. Building envelope — 6 features (constant per building config, varied in LHC)

These are held constant for a given building config and varied during the
[#719](https://github.com/...) LHC sweep.

| # | Feature | Range | Units | Notes |
|---|---------|-------|-------|-------|
| 11 | `wall_u_value` | 0.10 to 3.00 | W/m²K | Overall opaque envelope U-value |
| 12 | `window_to_wall_ratio` | 0.05 to 0.80 | fraction | All facades combined |
| 13 | `window_shgc` | 0.20 to 0.87 | fraction | Solar heat gain coefficient |
| 14 | `floor_area` | 20 to 5000 | m² | log-normalized before training (see §5) |
| 15 | `thermal_mass_index` | 0.10 to 1.00 | fraction | Normalized from CTF `Cm` coefficient |
| 16 | `infiltration_ach` | 0.10 to 3.00 | ACH | At 50 Pa reference, or natural ACH |

### 3e. Climate zone — 6 one-hot features (constant per config)

One-hot encoding for the six ASHRAE 169-2021 climate zones represented in
`data/weather_locations.json` (current snapshot: 4A, 5A, 5B, 6A, 6B, 7A).

| # | Feature | Value |
|---|---------|-------|
| 17 | `cz_4A` | 1 iff zone == 4A |
| 18 | `cz_5A` | 1 iff zone == 5A |
| 19 | `cz_5B` | 1 iff zone == 5B |
| 20 | `cz_6A` | 1 iff zone == 6A |
| 21 | `cz_6B` | 1 iff zone == 6B |
| 22 | `cz_7A` | 1 iff zone == 7A |

**Total input dimensionality: 23 features (17 continuous + 6 one-hot).**

---

## 4. Output Feature Set (2 targets)

| # | Output | Range | Units | Notes |
|---|--------|-------|-------|-------|
| 0 | `heating_load` | 0 to 500 | kW | Non-negative magnitude; `max(·, 0)` post-inference |
| 1 | `cooling_load` | 0 to 500 | kW | Non-negative magnitude; `max(·, 0)` post-inference |

**Sign convention.** Both outputs are non-negative magnitudes. The dominant
mode (heating vs. cooling) is implicit in the input setpoint, exterior
temperature, and solar gain. This matches the ASHRAE 140 convention of
reporting heating and cooling loads separately and avoids forcing the
network to learn a discontinuous sign change at the setpoint crossover.

> **Future extension (Phase 4b).** Adding a third output `zone_temp_delta`
> (K, signed) and a fourth `prediction_std` (K, positive, for GP ensemble UQ)
> is non-breaking: the ONNX graph just grows in the last dimension. The
> spec is written to make this addition a single-line change.

---

## 5. Network Topology

```text
Input (23) → Linear(23→64) → BatchNorm1d(64) → ReLU
          → Linear(64→64) → BatchNorm1d(64) → ReLU
          → Linear(64→32) → BatchNorm1d(32) → ReLU
          → Linear(32→ 2)                          [heating_load, cooling_load]
```

**Design choices** (with rationale):

- **3 hidden layers, 64/64/32 neurons** — depth-first search starting
  point per IBPSA 2024 protocol; matches the ORNL large-scale EnergyPlus
  surrogate study topology. Width 64 is the L1-cache sweet spot for
  32-byte aligned FP32 weights.
- **BatchNorm1d after each Linear** — required for training stability
  because the 17 continuous features have heterogeneous scales
  (W/m²K, m/s, °C, fraction, m²). BN is the standard remedy and
  ONNX-opset-17 compatible.
- **ReLU** — standard regression surrogate activation. No dropout because
  inference must be deterministic (governance §2).
- **No output activation** — raw regression. The model can produce
  negative logits during training; the Rust side clamps to `≥ 0`.
- **Single MLP per (component × building_type)** — no ensembling in 4a;
  ensembles come in 4b via the existing `CompositeSurrogate` machinery
  (`src/ai/modular_surrogate.rs`).

**Parameter count (verified, FP32):**

| Layer | Weights | Biases | BN (γ+β) | Total |
|-------|---------|--------|----------|-------|
| Linear 23→64 | 1,472 | 64 | 128 | 1,664 |
| Linear 64→64 | 4,096 | 64 | 128 | 4,288 |
| Linear 64→32 | 2,048 | 32 | 64 | 2,144 |
| Linear 32→2  | 64 | 2 | 0 | 66 |
| **Total** | **7,680** | **162** | **320** | **8,162** |

≈ 8.2 K parameters (issue body rounded to ~9 K). On-disk sizes:

| Precision | Bytes/param | Model size |
|-----------|-------------|------------|
| FP32 | 4 | 31.9 KB |
| FP16 | 2 | 15.9 KB |
| INT8 | 1 | 8.0 KB |

The model fits comfortably in L1 cache (32 KB typical) for inference and is
~2,000× smaller than typical EnergyPlus weather files. ONNX export
quantization is therefore optional for size, not required for performance.

**Alternative architectures to benchmark in 4a (do not ship in 4a):**

- Shallow: `Linear(23→128) → ReLU → Linear(128→64) → ReLU → Linear(64→2)`
  (fewer non-linearities, faster export)
- Deep: `Linear(23→128) → Linear(128→128) → Linear(128→64) → Linear(64→32) → Linear(32→2)`
  (more capacity for nonlinear interactions)

The 64/64/32 choice is the depth-3 baseline; the alternatives are tracked
in the benchmark report delivered with the trained model.

---

## 6. Training Strategy

### 6.1 Data generation

| Aspect | Value | Source |
|--------|-------|--------|
| Data source | LHC sweep output from Phase 3 | [#719](https://github.com/...) |
| Sample volume | **50 K – 200 K** per `(component × building_type)` | IBPSA 2024 protocol: ANNs need ≥ 10 K; 50 K is a robust floor |
| Parameter space | 17 continuous + 1 categorical from §3d/§3e | `SurrogateDomain::temp_bounds` etc. |
| Sampling | Latin Hypercube | `pyDOE2` or `scipy.stats.qmc.LatinHypercube` |
| Target | `(heating_load, cooling_load)` from physics engine | `src/sim/thermal_model_core.rs` |
| Storage format | Parquet (primary), HDF5 (optional) | `pyarrow` |

### 6.2 Train / validation / test split

| Split | Fraction | Reproducible seed |
|-------|----------|-------------------|
| Train | 70 % | seed=42 |
| Validation | 15 % | seed=42 |
| Test | 15 % | seed=42 |

For a 100 K-sample dataset, that yields 70 K / 15 K / 15 K samples. The
`data/ashrae140_holdout.json` 18 cases are **additionally** held out from
all three splits (they test the generalization gap on ASHRAE 140 Section 7
reference buildings specifically).

### 6.3 Normalization

| Target | Method | Fitted on |
|--------|--------|-----------|
| Continuous inputs (indices 0–16) | `StandardScaler` (zero mean, unit variance) | Train split only |
| `floor_area` (index 14) | `log1p` first, then `StandardScaler` | Train split only |
| One-hot inputs (17–22) | None | n/a (already 0/1) |
| Outputs (heating, cooling) | `StandardScaler` per output | Train split only |

**Critical constraint:** the fitted scaler `mean_` and `scale_` vectors
**must be embedded as ONNX graph initializers** (constant nodes) so that the
Rust `SurrogateManager` receives raw physical units. This eliminates an
entire class of normalization-drift bugs and is the only supported export
shape (see §7).

### 6.4 Optimizer & schedule

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Optimizer | Adam | Standard for ANN surrogates |
| Learning rate | 1e-3 | Adam default |
| Betas | (0.9, 0.999) | Adam default |
| LR schedule | `CosineAnnealingLR(T_max=100)` | Smooth decay, no manual tuning |
| Early stopping | patience=50 epochs on validation MSE | Stops ~epoch 60–80 in practice |
| Batch size | 512 | Fits in 8 GB GPU; good gradient signal |
| Max epochs | 200 (hard cap) | Early stopping almost always fires first |

### 6.5 Loss function

| Loss | Use |
|------|-----|
| Primary | MSE on **normalized** outputs |
| Monitored | RMSE% = normalized RMSE as a percentage of output range |
| Acceptance | **RMSE% < 2 %** on test split |

The 2 % threshold matches `ARCHITECTURE.md` § Phase 3 ("Surrogates must
match physics within 2 % on held-out data") and is the only ship gate.

**Per-output thresholds (informational, not gates):** heating and cooling
loads are reported separately so a model that learns cooling well but
mishandles heating is flagged even if the combined RMSE% is within bound.

### 6.6 Reproducibility

- `torch.manual_seed(42)`, `numpy.random.seed(42)`, `random.seed(42)`
- Pin `torch==2.x.y` and `onnx==1.x.y` in `requirements.txt`
- Embed the seed and dependency versions in `metadata.json` per
  `SURROGATE_GOVERNANCE.md` § 3.2

---

## 7. ONNX Export Contract

```python
# Pseudocode — actual implementation in scripts/export_onnx.py
torch.onnx.export(
    model,                                  # nn.Module
    dummy_input,                            # shape: (1, 23), float32
    "models/surrogate_zone_v1.0.0.onnx",
    opset_version=17,                       # ort 2.0.0-rc.10 compatible
    input_names=["inputs"],
    output_names=["loads"],
    dynamic_axes={
        "inputs": {0: "batch"},
        "loads":  {0: "batch"},
    },
    do_constant_folding=True,
)
```

**Rust-side contract** (what `SurrogateManager::load_onnx` consumes):

| Aspect | Value |
|--------|-------|
| Input tensor | `inputs: float32[batch, 23]` — **raw physical units** |
| Output tensor | `loads: float32[batch, 2]` — `[heating_load_kW, cooling_load_kW]` — **raw physical units** |
| Scaler | Embedded as ONNX initializers (no Rust-side normalization) |
| Opset | 17 (matches `ort = 2.0.0-rc.10` per `Cargo.toml`) |
| Quantization | Optional FP16/INT8 (governance §2, `QuantizationConfig`) |
| Versioning | `surrogate_<major>.<minor>.<patch>+onnx<opset>.onnx` per governance §3 |

**Why embed the scaler in the graph:** decouples the Rust inference
hot-path from Python training artifacts. Eliminates the need to ship
`scaler.pkl` (or equivalent) alongside the model. Matches the
`ONNX_INFERENCE_PIPELINE.md` "Float32 input, Float32 output, raw units"
contract.

**Why the existing I/O names `X`/`Y` (in `ONNX_INFERENCE_PIPELINE.md`) become
`inputs`/`loads` in this spec:** the new names are domain-specific and
self-documenting for thermal-load inference; `X`/`Y` are placeholders. This
is a backward-compatible *additive* change to the pipeline doc — old models
with `X`/`Y` continue to work because the existing inference code
auto-extracts `outputs[0]` regardless of name.

---

## 8. `SurrogateInputs` Struct Extension (Rust)

The current `src/ai/surrogate.rs` struct has 6 fields. The Phase 4a
schema needs 17 continuous + 1 categorical. The extension is
backwards-compatible via `Default::default()` and a `from_temps` legacy
constructor that defaults the new fields to mid-range values (preserving
existing tests).

```rust
/// Phase 4a MLP input — 17 continuous features + climate_zone string.
/// The 6 one-hot climate features are computed in the feature pipeline,
/// not stored on the struct.
#[derive(Clone, Debug)]
pub struct SurrogateInputs {
    // --- Boundary conditions (4) ---
    pub exterior_temp: f64,        // °C, [-50, 60]
    pub solar_rad_global: f64,     // W/m², [0, 1200]   (was: solar_rad)
    pub humidity: f64,             // % RH, [0, 100]
    pub wind_speed: f64,           // m/s, [0, 20]      (new)

    // --- Interior conditions (3) ---
    pub zone_temp_setpoint: f64,   // °C, [10, 40]      (was: zone_temp)
    pub occupancy_density: f64,    // ppl/100m², [0, 10] (was: occupancy)
    pub internal_gains: f64,       // W/m², [0, 50]     (new)

    // --- Temporal features (4, cyclical) ---
    pub hour_sin: f64,             // sin(2π·h/24)
    pub hour_cos: f64,             // cos(2π·h/24)
    pub doy_sin: f64,              // sin(2π·doy/365)
    pub doy_cos: f64,              // cos(2π·doy/365)

    // --- Building envelope (6, constant per config) ---
    pub wall_u_value: f64,         // W/m²K, [0.10, 3.00]
    pub window_to_wall_ratio: f64, // fraction, [0.05, 0.80]
    pub window_shgc: f64,          // fraction, [0.20, 0.87]
    pub floor_area: f64,           // m², [20, 5000]   (log-normalized pre-train)
    pub thermal_mass_index: f64,   // fraction, [0.10, 1.00]
    pub infiltration_ach: f64,     // ACH, [0.10, 3.00]

    // --- Climate (categorical) ---
    pub climate_zone: String,      // ASHRAE 169, e.g. "4A", "5A", ..., "7A"
}

impl SurrogateInputs {
    /// Total feature count after one-hot expansion: 17 + 6 = 23.
    pub const FEATURE_DIM: usize = 23;

    /// Convert to ONNX-ready flat feature vector (length 23).
    /// One-hot expansion of `climate_zone` happens here.
    pub fn to_feature_vector(&self) -> [f64; 23] { /* ... */ }

    /// Legacy 6-field constructor — defaults new fields to mid-range
    /// for backwards compatibility with existing tests.
    pub fn from_temps(temps: &[f64]) -> Self { /* ... */ }
}
```

**`SurrogateDomain` updates required:**

- Add `wind_speed_bounds: (f64, f64)`
- Add `internal_gains_bounds: (f64, f64)`
- Add `wall_u_value_bounds`, `window_to_wall_ratio_bounds`, `window_shgc_bounds`,
  `floor_area_bounds`, `thermal_mass_index_bounds`, `infiltration_ach_bounds`
- Update `is_valid(&SurrogateInputs)` to validate the 17 continuous fields
  (governance §1.2)
- Expand `climate_zones: Vec<String>` from 3 zones to the full 6-zone set
  declared in §3e

**Migration note for existing callers:** the old 6-field struct
constructor `SurrogateInputs::from_temps(&[20.0, 22.0])` is preserved.
`SurrogateManager::predict_loads(&[f64; 2])` continues to work because the
mock fallback (constant 1.2) is dimension-agnostic. The moment a real ONNX
model is loaded, **all 23 features must be present** or
`predict_loads_onnx` returns the dimension error already in
`src/ai/surrogate.rs:868`.

---

## 9. Validation & Acceptance Criteria

### 9.1 Model-quality gates

Per `SURROGATE_GOVERNANCE.md` § 4.1.2 and `ARCHITECTURE.md` § Phase 3:

| Metric | Threshold | Test data |
|--------|-----------|-----------|
| MAE | < 0.10 (relative) | Held-out 15 % test split |
| RMSE | < 0.15 (relative) | Held-out 15 % test split |
| R² | > 0.95 | Held-out 15 % test split |
| Max error | < 0.50 (relative) | Held-out 15 % test split |
| RMSE% (the ship gate) | **< 2 %** | Held-out 15 % test split |

### 9.2 ASHRAE 140 generalization check

Per `SURROGATE_GOVERNANCE.md` § 4.1.3 and the
`data/ashrae140_holdout.json` 18-case holdout:

| Metric | Threshold | Test data |
|--------|-----------|-----------|
| ASHRAE 140 case pass rate | ≥ 95 % within domain | 18 BESTEST holdout cases |
| Annual heating MWh | Within `[min, max]` from `data/ashrae140_reference.json` | Cases 600, 610, 620, ..., 995 |
| Annual cooling MWh | Within `[min, max]` | Same |
| Peak heating/cooling kW | Within `[min, max]` | Same |
| Free-float max zone temp | Within `[min, max]` | FF cases (600FF, 650FF, 900FF, 950FF, 980FF) |

### 9.3 Out-of-domain behaviour

Per `SURROGATE_GOVERNANCE.md` § 2:

- `SurrogateDomain::is_valid` rejects inputs outside §3a/§3b/§3d bounds
  → `predict_loads_governed` falls back to `analytical_loads` (WARN logged)
- Climate zone not in §3e set → fallback
- `metadata.json` declares all bounds, climate zones, and training period
  per `SURROGATE_GOVERNANCE.md` § 3.2

### 9.4 ONNX integration smoke test

```rust
#[test]
fn mlp_surrogate_loads_and_runs() {
    let manager = SurrogateManager::load_onnx("models/surrogate_zone_v1.0.0+onnx1.17.0.onnx")
        .expect("model must load");
    assert!(!manager.is_mock(), "real ONNX model must be active");
    let inputs = SurrogateInputs::default_residential_4a_jan15();
    let feats = inputs.to_feature_vector();
    let loads = manager.predict_loads_onnx(&feats).expect("inference must succeed");
    assert_eq!(loads.len(), 2, "MLP outputs [heating, cooling]");
    assert!(loads[0] >= 0.0 && loads[1] >= 0.0, "loads are non-negative magnitudes");
}
```

### 9.5 Deliverable checklist (ship gates)

- [ ] Spec reviewed and merged (this document)
- [ ] `SurrogateInputs` extended per §8 (Rust PR, separate issue)
- [ ] `SurrogateDomain` bounds expanded per §8 (Rust PR, same)
- [ ] `scripts/generate_training_data.py` produces dataset matching §3 schema
- [ ] `scripts/train_surrogate.py` trains to < 2 % RMSE on test split
- [ ] `scripts/export_onnx.py` exports with opset 17, input/output shapes per §7
- [ ] `models/surrogate_zone_v1.0.0+onnx1.17.0.onnx` produced
- [ ] `validation_artifacts/surrogate_zone_v1.0.0/` populated per governance §4.2
- [ ] Rust integration test §9.4 passes
- [ ] ASHRAE 140 §9.2 pass rate ≥ 95 % on the 18-case holdout

---

## 10. Risks and Open Questions

| Risk | Severity | Mitigation |
|------|----------|------------|
| Physics validation incomplete (v1.3 blind pass rate < 80 %) | **Blocker** | Per [#719](https://github.com/...) prerequisite, training data would embed physics errors. Hold training until prerequisite met. |
| 50 K samples insufficient for 4A/5A/6A/7A generalization | Medium | Start with 50 K, scale to 200 K if §9.1/§9.2 fail. Track scaling curve. |
| `floor_area` log-normalization leakage between train/test | Medium | Refit scaler per fold; document in `metadata.json` |
| Opset 17 vs. newer `ort` versions | Low | Pin `ort = 2.0.0-rc.10`; bump jointly with `opset_version` |
| MLP undershoots nonlinear interactions (radiation, PCM, etc.) | Low | The 4b GP ensemble ([#708](https://github.com/...)) adds UQ; deep variant in §5 is fallback |
| Climate zone set drift in `data/weather_locations.json` | Low | One-hot position is fixed (§3e); add new zones as additive index 23+, not in-place |

**Open questions to resolve before merging this spec:**

1. **Climate zone set.** §3e commits to 6 zones based on the current
   `data/weather_locations.json`. Should we add 2A/3A for cold-climate
   coverage, or hold for Phase 5?
2. **Internal gains units.** §3b uses W/m² (flux density) consistent with
   EnergyPlus. Should the struct store W (absolute) and derive density
   from `floor_area`? Current proposal keeps density because it makes
   the LSTM-friendly "per area" normalization work without coupling to
   `floor_area`.
3. **Phase 4b wiring.** The `loads` output name and 2-D shape must be
   forward-compatible with adding `zone_temp_delta` and `prediction_std`
   in 4b. Confirmed non-breaking (last-dim grow) but a separate ADR
   should be filed when 4b starts.

---

## 11. Related Documents

| Document | Relationship |
|----------|--------------|
| `ARCHITECTURE.md` | Module boundaries; § Zone Balance, § Phase 3 validation |
| `docs/SURROGATE_GOVERNANCE.md` | Binding governance policy; this spec implements § 1–5 |
| `docs/ONNX_INFERENCE_PIPELINE.md` | Binding I/O baseline; this spec refines it for MLP |
| `data/ashrae140_holdout.json` | 23-feature schema reference; LHC holdout specification |
| `data/ashrae140_reference.json` | Acceptance thresholds (min/max/mean per case) |
| `data/weather_locations.json` | Climate zone enumeration source |
| `src/ai/surrogate.rs` | Rust inference layer; `SurrogateManager::predict_loads_onnx` |
| `src/ai/modular_surrogate.rs` | `CompositeSurrogate` for future 4b ensemble |
| `src/sim/thermal_model.rs` | `ThermalModelTrait`; `SurrogateThermalModel` is the MLP host |
| `scripts/ashrae_benchmark_harness.py` | ASHRAE 140 § 9.2 runner |
| `docs/architecture/multi_zone.md` | Multi-zone extension (out of scope for 4a) |

| Issue | Relationship |
|-------|--------------|
| [#718](https://github.com/...) | Parent epic — v3.0 Surrogate Training & ONNX Export |
| [#719](https://github.com/...) | v2.1 Synthetic Data Generation — produces the training set |
| [#708](https://github.com/...) | Phase 4b — MLP+GP ensemble (UQ), downstream of this spec |
| [#977](https://github.com/...) | XGBoost/xDT export — parallel research path |
| #976 | Active-learning recommender — not a prerequisite, but synergies |

---

## 12. Versioning

| Field | Value |
|-------|-------|
| Spec version | 0.1 (draft) |
| First MLP model version expected | 1.0.0+onnx1.17.0 |
| ONNX opset | 17 |
| `ort` version | 2.0.0-rc.10 |
| Authoring date | 2026-06-14 |
