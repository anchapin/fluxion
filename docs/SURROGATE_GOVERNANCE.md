# Surrogate Governance Policy

## Overview

This document establishes governance rules for AI surrogate models within Fluxion. Surrogates are **bounded services** that replace expensive CFD/ray-tracing with pre-trained neural networks, not hidden substitutes for physics.

**Core Principle:** Surrogates must always be transparently bounded—they operate within defined domains and fall back to analytical physics when out-of-domain.

---

## 1. Domain of Validity

### 1.1 Required Validity Bounds

Every surrogate model **must** declare its domain of validity before deployment:

| Parameter | Required Range | Notes |
|-----------|---------------|-------|
| Temperature | -50°C to 60°C | Exterior temperature bounds |
| Zone Temperature | 10°C to 40°C | Interior zone air temperature |
| Solar Radiation | 0 to 1200 W/m² | Clear-sky + cloud cover |
| Humidity | 0% to 100% RH | Relative humidity |
| Occupancy | 0 to 10 persons/m² | occupant density |
| HVAC Capacity | Model-specific | Defined at training time |

### 1.2 Validity Boundary Enforcement

```rust
/// Surrogate validity domain - must be declared for each model
#[derive(Clone, Debug)]
pub struct SurrogateDomain {
    /// Temperature bounds (exterior) in Celsius
    pub temp_bounds: (f64, f64),     // e.g., (-50.0, 60.0)
    /// Zone temperature bounds in Celsius
    pub zone_temp_bounds: (f64, f64), // e.g., (10.0, 40.0)
    /// Solar radiation bounds in W/m^2
    pub solar_bounds: (f64, f64),     // e.g., (0.0, 1200.0)
    /// Training climate zones (ASHRAE zones)
    pub climate_zones: Vec<String>,   // e.g., ["4A", "5A", "6A"]
    /// Building types trained on
    pub building_types: Vec<String>,  // e.g., ["residential", "commercial"]
    /// Date range of training data (ISO 8601)
    pub training_period: (String, String),
}

impl SurrogateDomain {
    /// Check if inputs are within domain bounds
    pub fn is_valid(&self, inputs: &SurrogateInputs) -> bool {
        let temp_valid = inputs.exterior_temp >= self.temp_bounds.0
                      && inputs.exterior_temp <= self.temp_bounds.1;
        let zone_valid = inputs.zone_temp >= self.zone_temp_bounds.0
                      && inputs.zone_temp <= self.zone_temp_bounds.1;
        let solar_valid = inputs.solar_rad >= self.solar_bounds.0
                       && inputs.solar_rad <= self.solar_bounds.1;
        let climate_valid = self.climate_zones.contains(&inputs.climate_zone);

        temp_valid && zone_valid && solar_valid && climate_valid
    }
}
```

### 1.3 Multi-Zone Composites

For `CompositeSurrogate` (modular surrogates combining solar, HVAC, infiltration, thermal mass):

- Each `ComponentSurrogate` must declare its individual domain
- The composite domain is the **intersection** of all component domains
- Domain validation must occur at composite level and component level

---

## 2. Fallback Behavior

### 2.1 Fallback Triggers

Surrogates **must** fall back to analytical models when:

| Trigger Condition | Fallback Action |
|------------------|-----------------|
| Input outside domain bounds | Use `analytical_loads()` |
| ONNX inference returns empty | Use `analytical_loads()` |
| Model file not found | Use `analytical_loads()` with warning |
| GPU/backend failure | Fall back to `fallback_to_cpu: true` config |
| Session pool exhausted (max_retries) | Use `analytical_loads()` with error logged |

### 2.2 Fallback Hierarchy

```rust
pub enum SurrogateMode {
    /// Full neural surrogate inference
    NeuralOnly,
    /// Neural with automatic fallback to analytical
    NeuralWithFallback,
    /// Analytical only (no surrogate)
    AnalyticalOnly,
}

impl SurrogateManager {
    /// Predict with enforced fallback policy
    pub fn predict_loads_governed(
        &self,
        temps: &[f64],
        domain: &SurrogateDomain,
        mode: SurrogateMode,
    ) -> Result<Vec<f64>, String> {
        let inputs = SurrogateInputs::from_temps(temps);

        // Check domain validity
        if !domain.is_valid(&inputs) {
            warn!(
                "Inputs out of domain bounds for surrogate. \
                 Temp: {:.1}, Zone: {:.1}, Solar: {:.1}, Climate: {}. \
                 Falling back to analytical model.",
                inputs.exterior_temp, inputs.zone_temp,
                inputs.solar_rad, inputs.climate_zone
            );
            return self.analytical_loads(temps);
        }

        match mode {
            SurrogateMode::NeuralOnly => self.predict_loads_batched(&[temps.to_vec()]),
            SurrogateMode::NeuralWithFallback => self.predict_loads_with_fallback(temps),
            SurrogateMode::AnalyticalOnly => self.analytical_loads(temps),
        }
    }
}
```

### 2.3 Required Fallback Methods

Every `SurrogateManager` must implement:

| Method | Purpose |
|--------|---------|
| `analytical_loads()` | Physics-based fallback calculations |
| `predict_loads_with_fallback()` | Automatic neural → analytical transition |
| `gpu_supported()` | Runtime check for GPU availability |

### 2.4 Fallback Logging

All fallback events **must** be logged at `WARN` level with:
- Timestamp
- Reason for fallback
- Input values that triggered fallback
- Output from analytical model

---

## 3. Versioning

### 3.1 Version Schema

Surrogate models follow **Semantic Versioning** with model-specific suffix:

```
major.minor.patch+onnx_version
example: 1.2.0+onnx1.14.0
```

| Component | Increment Rule |
|-----------|---------------|
| Major | Breaking domain changes (bounds widened/narrowed) |
| Minor | New component added, improved accuracy |
| Patch | Bug fixes, weight updates without domain change |

### 3.2 Model Metadata

Every ONNX model **must** include metadata:

```json
{
  "model_version": "1.2.0+onnx1.14.0",
  "domain": {
    "temp_bounds": [-50.0, 60.0],
    "zone_temp_bounds": [10.0, 40.0],
    "solar_bounds": [0.0, 1200.0],
    "climate_zones": ["4A", "5A", "6A"],
    "building_types": ["residential"]
  },
  "training": {
    "period": ["2020-01-01", "2023-12-31"],
    "samples": 1000000,
    "framework": "PyTorch 2.0",
    "export_tool": "torch.onnx 1.14.0"
  },
  "validation": {
    "test_mae": 0.023,
    "test_rmse": 0.041,
    "test_r2": 0.998,
    "validation_date": "2024-01-15"
  }
}
```

### 3.3 Version Compatibility

| Surrogate Version | Fluxion Version | Compatible |
|------------------|-----------------|------------|
| 1.x.x | 1.x.x | ✅ Yes |
| 2.x.x | 1.x.x | ❌ No (major version mismatch) |
| 1.x.x | 2.x.x | ⚠️ Requires re-validation |

### 3.4 Version Storage

```
models/
├── rl_policy/
│   ├── policy.onnx
│   └── policy.json          # Model metadata
├── surrogate_v1/
│   ├── surrogate_1.0.0+onnx1.14.0.onnx
│   └── metadata.json
└── surrogate_v2/
    ├── surrogate_2.0.0+onnx1.14.0.onnx
    └── metadata.json
```

---

## 4. Required Validation Artifacts

### 4.1 Pre-Deployment Validation

Before any surrogate model can be deployed, the following artifacts **must** be generated and reviewed:

#### 4.1.1 Domain Coverage Report

- Heatmap of training data density across domain parameters
- Identification of extrapolation regions
- Boundary stability analysis

#### 4.1.2 Accuracy Metrics

| Metric | Required Threshold | Test Data |
|--------|-------------------|-----------|
| MAE | < 0.1 (relative) | Held-out 20% of training |
| RMSE | < 0.15 (relative) | Held-out 20% of training |
| R² | > 0.95 | Held-out 20% of training |
| Max Error | < 0.5 (relative) | Held-out 20% of training |

#### 4.1.3 ASHRAE 140 Comparison

For building thermal models:
- Run ASHRAE 140 test cases with surrogate
- Compare against analytical Fluxion results
- Pass rate must be ≥ 95% within domain

#### 4.1.4 Uncertainty Quantification

- Predictions must include uncertainty bounds (`PredictionWithUncertainty`)
- Out-of-domain inputs must show increasing uncertainty
- Calibration against known physical limits

### 4.2 Validation Artifact Structure

```
validation_artifacts/
└── <model_name>/
    ├── metadata.json              # Model metadata (Section 3.2)
    ├── domain_coverage.html       # Visualization of training domain
    ├── accuracy_metrics.json      # MAE, RMSE, R2, Max Error
    ├── ashrae140_results.csv     # ASHRAE 140 test case results
    ├── uncertainty_calibration.json
    └── <model_file>.onnx         # The model itself
```

### 4.3 Continuous Validation

- **Pre-commit**: Domain bounds check on new models
- **CI/CD**: Run ASHRAE 140 suite with surrogate enabled
- **Release**: Re-validate surrogate when Fluxion physics changes

### 4.4 Validation Failure Policy

If validation metrics fall below thresholds:

1. **Warning** (one metric below): Log warning, allow deployment with disclosure
2. **Failure** (multiple metrics below): Block deployment, require model retraining
3. **Critical** (domain extrapolation detected): Hard block, reject model

---

## 5. Composite Surrogate Rules

### 5.1 Component Requirements

For `CompositeSurrogate` (aggregating multiple `ComponentSurrogate`):

1. Each component must have its own domain declaration
2. Each component must have its own validation artifacts
3. Composite domain = intersection of component domains

### 5.2 Compositor Validation

- Test composite on ASHRAE 140 cases
- Verify energy conservation: sum of components ≈ total physics
- Validate that removing any component shows appropriate accuracy degradation

---

## 6. Governance Compliance

### 6.1 Policy Adoption

| Date | Milestone |
|------|-----------|
| YYYY-MM-DD | Policy created (this document) |
| YYYY-MM-DD | Existing surrogates documented with domains |
| YYYY-MM-DD | Fallback enforcement implemented |
| YYYY-MM-DD | Versioning schema adopted |
| YYYY-MM-DD | Validation artifacts required for deployment |

### 6.2 Exceptions

Exceptions to this policy require:
1. Written approval from AI/ML Lead + Physics Lead
2. Documented risk assessment
3. Time-bounded exception (max 90 days)

---

## 7. Related Documents

- [RULES.md](../RULES.md) - Core physics and code rules
- [validation_reporting.md](./validation_reporting.md) - Validation reporting system
- [PHYSICAL_CONSTANTS.md](./PHYSICAL_CONSTANTS.md) - Physical bounds reference
- [ASHRAE 140 Validation](./ASHRAE140_VALIDATION.md) - ASHRAE compliance
