# Surrogate Model Card (v3.0)

## Overview

| Property | Value |
|----------|-------|
| **Version** | 3.0.0 |
| **Date** | 2026-06-17 |
| **Status** | Phase 4a: MLP Baseline |
| **Framework** | scikit-learn MLPRegressor → ONNX |
| **ONNX Opset** | 17 |
| **Inference Runtime** | onnxruntime 1.24+ / ort 2.0+ |

## Architecture

### Model Type
Single hidden-layer MLP (Multi-Layer Perceptron) per physics component.

### Component Models

| Component | Input Dim | Output Dim | Hidden Layers | Parameters |
|-----------|-----------|------------|---------------|------------|
| `zone_thermal` | 7 | 1 | (64, 32) | ~2,500 |
| `solar_gain` | 8 | 1 | (64, 32) | ~2,700 |
| `conduction` | 6 | 1 | (64, 32) | ~2,100 |
| `ventilation` | 6 | 1 | (64, 32) | ~2,100 |

### Training Configuration

```python
MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate='adaptive',
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
)
```

### ONNX I/O Specification

All models conform to the standard ONNX I/O specification for Rust `SurrogateManager`:

```
Input:  Tensor name="X", shape=[batch_size, n_features], type=float32
Output: Tensor name="Y", shape=[batch_size, n_outputs], type=float32
```

### Model Files

```
models/
├── surrogate_zone_thermal.onnx     # Zone thermal load prediction
├── surrogate_solar_gain.onnx       # Solar gain calculation
├── surrogate_conduction.onnx        # Conduction heat transfer
├── surrogate_ventilation.onnx       # Ventilation load calculation
├── surrogate_zone_thermal_metrics.json
├── surrogate_solar_gain_metrics.json
├── surrogate_conduction_metrics.json
└── surrogate_ventilation_metrics.json
```

## Input Features

### zone_thermal
| Feature | Units | Range | Description |
|---------|-------|-------|-------------|
| `exterior_temp` | °C | -10 to 40 | Outdoor air temperature |
| `zone_temp` | °C | 18 to 26 | Indoor zone temperature |
| `solar_rad` | W/m² | 0 to 800 | Solar radiation on surface |
| `humidity` | % | 20 to 80 | Relative humidity |
| `occupancy` | persons | 0 to 5 | Number of occupants |
| `hvac_mode` | enum | 0,1,2 | 0=off, 1=heating, 2=cooling |
| `climate_zone_encoded` | - | 0 to 8 | ASHRAE climate zone index |

### solar_gain
| Feature | Units | Range | Description |
|---------|-------|-------|-------------|
| `latitude` | ° | 25 to 50 | Site latitude |
| `longitude` | ° | -120 to -70 | Site longitude |
| `day_of_year` | - | 1 to 366 | Day of year |
| `hour_of_day` | - | 0 to 23 | Hour of day |
| `surface_tilt` | ° | 0,30,45,60,90 | Surface tilt angle |
| `surface_azimuth` | ° | 0 to 360 | Surface azimuth angle |
| `direct_normal_irradiance` | W/m² | 0 to 1000 | DNI |
| `diffuse_horizontal_irradiance` | W/m² | 0 to 300 | DHI |

### conduction
| Feature | Units | Range | Description |
|---------|-------|-------|-------------|
| `exterior_temp` | °C | -10 to 40 | Outdoor air temperature |
| `interior_temp` | °C | 18 to 26 | Indoor temperature |
| `wall_u_value` | W/m²K | 0.1 to 2.5 | Wall U-value |
| `wall_area` | m² | 10 to 100 | Wall surface area |
| `wall_mass` | kg | 50 to 500 | Wall thermal mass |
| `surface_emissivity` | - | 0.7 to 0.95 | Surface emissivity |

### ventilation
| Feature | Units | Range | Description |
|---------|-------|-------|-------------|
| `exterior_temp` | °C | -10 to 40 | Outdoor air temperature |
| `interior_temp` | °C | 18 to 26 | Indoor temperature |
| `wind_speed` | m/s | 0 to 10 | Wind speed |
| `indoor_pressure` | Pa | 99000 to 101500 | Indoor air pressure |
| `ventilation_rate` | CFM | 0 to 500 | Ventilation flow rate |
| `ach` | 1/hr | 0 to 10 | Air changes per hour |

## Output Targets

| Component | Output | Units |
|-----------|--------|-------|
| `zone_thermal` | `thermal_load` | kW |
| `solar_gain` | `solar_gain_W` | W |
| `conduction` | `conduction_flux_W` | W/m² |
| `ventilation` | `ventilation_load_W` | W |

## Performance Metrics

### Acceptance Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| RMSE vs Physics | < 2% | Normalized by output range |
| Inference Time | < 1 ms | P95 single prediction |
| ONNX Opset | ≥ 17 | Compatible with ort 2.0+ |

### Phase 4a Targets
- [ ] RMSE < 2% for all components
- [ ] Inference time < 1ms per prediction
- [ ] ONNX opset 17+ compatible
- [ ] `cargo test ai::surrogate --lib` passes
- [ ] `SurrogateMode::NeuralWithFallback` uses neural path

### Phase 4b (Follow-on)
- [ ] MLP+GP ensemble for uncertainty quantification
- [ ] Enable MIRAI temporal forecasting (#708)
- [ ] Training data reduction to 5K-15K samples via PINN physics loss

## Training Data

### Synthetic Data Generation
Generated using `scripts/generate_training_data.py`:

```bash
python scripts/generate_training_data.py \
    --n-scenarios 5000 \
    --timesteps-per-scenario 8760 \
    --components zone_thermal,solar_gain,conduction,ventilation \
    --climate-zones 1A,2A,3A,4A,5A,6A,7,8 \
    --output-dir data/synthetic/v2.1
```

### Data Format
```
data/synthetic/v2.1/
├── zone_thermal/
│   ├── train.parquet   (~4M records)
│   ├── val.parquet
│   ├── test.parquet
│   └── metadata.json
├── solar_gain/
├── conduction/
└── ventilation/
```

## Usage

### Training
```bash
# Train all components
python scripts/train_surrogate.py --all-components --data-dir data/synthetic/v2.1

# Train specific component
python scripts/train_surrogate.py --component zone_thermal --data-dir data/synthetic/v2.1
```

### Export & Validate ONNX
```bash
# Export trained models to ONNX
python scripts/export_onnx.py --all-models --input-dir models/

# Validate models
python scripts/export_onnx.py --all-models --benchmark
```

### Benchmark vs Physics
```bash
# Validate against physics baseline
python scripts/validate_surrogate.py --all-models --n-samples 1000

# Check RMSE and inference time
python scripts/validate_surrogate.py --component zone_thermal --model models/surrogate_zone_thermal.onnx
```

### Rust Inference
```rust
use fluxion::ai::surrogate::{SurrogateManager, InferenceBackend};

let manager = SurrogateManager::load_onnx("models/surrogate_zone_thermal.onnx")?;
let loads = manager.predict_loads_with_fallback(&[20.0, 22.0, 800.0, 50.0, 2.0, 1.0, 4.0])?;
```

## Literature

### Phase 4a: Pure MLP (Current)
- Fast, auditable, easy ONNX export
- 50K-200K training samples
- Target: RMSE < 2%, inference < 1ms

### Phase 4b: MLP+GP Ensemble (Follow-on)
See Issue #718 Comment: Architecture Decision (2026-05-12)

- GP layer on MLP embeddings for uncertainty quantification
- Enables probabilistic multi-horizon forecasting
- Reduces training data to 1,766-15,000 samples via PINN physics loss
- Optimal λ_physics weight: 0.1-0.3 (from IBPSA 2025)

### Key References
1. IBPSA 2025 — Inverse PINN for Building Thermal Parameter Identification
2. ScienceDirect — PINN for Cooling Electricity Prediction (Historic Buildings)
3. Academia.edu — PINN on ASHRAE RP-1312 / BOPTEST Benchmarks

## Related Issues

| Issue | Relationship |
|-------|--------------|
| #718 | This epic |
| #719 | v2.1 Synthetic Data Generation (prerequisite) |
| #708 | MIRAI temporal forecasting (feeds into) |
| #726 | FD solver for PINN training (future) |

## Changelog

### v3.0.0 (2026-06-17)
- Initial MLP baseline implementation
- scikit-learn MLPRegressor → ONNX export
- ONNX opset 17 for ort 2.0+ compatibility
- Per-component models: zone_thermal, solar_gain, conduction, ventilation
