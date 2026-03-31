# Phase 5 Quick Start Guide

This guide helps developers begin Phase 5 (Production Validation & Calibration) work immediately after Phase 4 completes.

## Pre-Phase 5 Checklist

Before starting Phase 5, confirm:

- [ ] Phase 4 complete: `cargo test` passes (16/16 tests)
- [ ] Models generated: `assets/thermal_surrogate.onnx` exists
- [ ] Python environment ready: `pip install torch numpy onnx`
- [ ] Git branch created: `git checkout -b phase-5-validation`
- [ ] Dataset access arranged (see Data Sources section below)

---

## Directory Structure for Phase 5

```
fluxion/
├── tools/
│   ├── data_collection.py          [NEW] Phase 5 data pipeline
│   ├── retrain_on_real_data.py     [NEW] Retraining with real data
│   └── train_surrogate.py          [EXISTING] From Phase 4
├── src/
│   ├── validation/                 [NEW] Phase 5 validation modules
│   │   ├── mod.rs
│   │   ├── cross_validator.rs      [NEW]
│   │   ├── ashrae_140.rs           [NEW]
│   │   └── physics_validator.rs    [NEW]
│   └── ... (existing)
├── assets/
│   ├── thermal_surrogate.onnx      [EXISTING] Phase 4 model
│   ├── training_data.npz           [EXISTING]
│   ├── real_building_data.npz      [NEW] Phase 5 data
│   └── model_metrics_v2.json       [NEW] Phase 5 results
└── docs/
    ├── ROADMAP.md                  [NEW] This document
    └── VALIDATION_RESULTS.md       [NEW] Phase 5 results
```

---

## Data Sources (Priority Order)

### 1. ASHRAE 140 Reference Buildings (RECOMMENDED)
- **Source**: https://www.ashrae.org/technical-resources/ashrae-140
- **Format**: EnergyPlus IDF files
- **Advantage**: Standardized, validated, benchmarked
- **Size**: 6 buildings, hourly data
- **License**: Open / Educational use OK

**Getting started**:
```bash
# Download manually or programmatically
wget https://archive.nrel.gov/ashrae140/ASHRAE140-buildings.zip
unzip ASHRAE140-buildings.zip -d data/ashrae140/
```

### 2. NREL Commercial Building Metadata
- **Source**: https://data.openei.org/datasets/dataset/commercial-buildings-data
- **Format**: CSV + weather data (TMY)
- **Advantage**: Large dataset (>20K buildings)
- **Size**: ~1GB
- **License**: CC-BY-4.0

**Getting started**:
```python
# tools/download_nrel_data.py
import openei

buildings = openei.fetch_commercial_buildings()
buildings.to_csv('data/nrel_commercial.csv')
```

### 3. Building Energy Efficiency Data (BEED)
- **Source**: https://data.nrel.gov/submissions/13
- **Format**: Mixed (CSV, JSON)
- **Advantage**: Retrofit data (before/after)
- **Size**: ~10K buildings
- **License**: Open Data

### 4. Local/Custom Data (If Available)
- Real utility bills (contact local utilities)
- Building automation system logs
- HVAC monitoring data
- Retrofit projects you have access to

---

## Phase 5 Implementation Steps

### Step 1: Data Collection & Preprocessing (Week 1)

**Create**: `tools/data_collection.py`

```python
#!/usr/bin/env python3
"""Phase 5: Data collection and preprocessing."""

import numpy as np
import pandas as pd
from pathlib import Path

class DataCollector:
    """Unified interface for multiple data sources."""

    def load_ashrae_140(self, base_path: str) -> dict:
        """Load ASHRAE 140 reference buildings."""
        buildings = {}

        for building_path in Path(base_path).glob("*.idf"):
            # Parse EnergyPlus IDF
            building_id = building_path.stem

            # Extract metadata and simulation results
            # This requires EnergyPlus or a parser

            buildings[building_id] = {
                'params': {...},      # U-value, etc.
                'temps': [...],       # Hourly zone temperatures
                'energy': [...],      # Hourly energy consumption
                'weather': [...]      # Weather data
            }

        return buildings

    def combine_datasets(self, *datasets) -> np.ndarray:
        """Combine multiple data sources."""
        X = []  # Parameters
        y = []  # Outputs (temperatures)

        for dataset in datasets:
            for building_id, data in dataset.items():
                X.append(data['params'])
                y.append(data['temps'])

        return np.array(X), np.array(y)

if __name__ == "__main__":
    collector = DataCollector()

    # Load ASHRAE 140
    ashrae_data = collector.load_ashrae_140("data/ashrae140/")

    # Load NREL (if available)
    # nrel_data = collector.load_nrel_data("data/nrel_commercial.csv")

    # Combine
    X, y = collector.combine_datasets(ashrae_data)

    # Save
    np.savez('assets/real_building_data.npz', X=X, y=y)
    print(f"✓ Collected {X.shape[0]} samples")
```

**Success criteria**:
- Load ASHRAE 140 buildings
- Extract parameters and results
- Save to `assets/real_building_data.npz`
- Verify shape: X=(N, 2), y=(N, 10)

### Step 2: Retraining Pipeline (Week 1-2)

**Create**: `tools/retrain_on_real_data.py`

```python
#!/usr/bin/env python3
"""Phase 5: Retrain model on real data."""

import torch
import numpy as np
from pathlib import Path

def retrain_with_real_data(
    real_data_path: str = "assets/real_building_data.npz",
    synthetic_data_path: str = "assets/training_data.npz",
    mixing_ratio: float = 0.7,  # 70% real, 30% synthetic
    epochs: int = 100
):
    """Retrain surrogate on mix of real and synthetic data."""

    # Load data
    real = np.load(real_data_path)
    synthetic = np.load(synthetic_data_path)

    X_real, y_real = real['X_train'], real['y_train']
    X_synthetic, y_synthetic = synthetic['X_train'], synthetic['y_train']

    # Combine with weighting
    n_real = int(len(X_real) * mixing_ratio)
    n_synthetic = len(X_synthetic) - n_real

    X_train = np.vstack([X_real, X_synthetic[:n_synthetic]])
    y_train = np.vstack([y_real, y_synthetic[:n_synthetic]])

    print(f"Training data: {X_train.shape[0]} samples")
    print(f"  Real: {n_real} ({100*n_real/len(X_train):.0f}%)")
    print(f"  Synthetic: {n_synthetic} ({100*n_synthetic/len(X_train):.0f}%)")

    # Train model (same as Phase 4, but with new data)
    model = train_model(X_train, y_train, epochs=epochs)

    # Export
    torch.onnx.export(
        model,
        torch.randn(1, 2),
        "assets/thermal_surrogate_v2_calibrated.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=12
    )

    print("✓ Saved to assets/thermal_surrogate_v2_calibrated.onnx")

if __name__ == "__main__":
    retrain_with_real_data()
```

**Success criteria**:
- Load both real and synthetic datasets
- Combine with proper weighting
- Train model without errors
- Export to ONNX format

### Step 3: ASHRAE 140 Validation (Week 2)

**Create**: `src/validation/ashrae_140.rs`

```rust
// src/validation/ashrae_140.rs
use std::collections::HashMap;

pub struct ASHRAE140Validator {
    buildings: HashMap<String, ASHRAE140Building>,
}

pub struct ASHRAE140Building {
    pub name: String,
    pub baseline_energy: f64,
    pub baseline_temps: Vec<f64>,
}

impl ASHRAE140Validator {
    pub fn new() -> Self {
        // Load ASHRAE 140 reference values
        Self {
            buildings: HashMap::new(),
        }
    }

    pub fn validate_analytical_engine(&mut self) -> ValidationReport {
        // Test analytical engine against ASHRAE 140 values
        // This is the "ground truth" for Phase 5

        let mut report = ValidationReport::default();

        for (building_id, building) in &self.buildings {
            // Run Fluxion analytical simulation
            // Compare to ASHRAE 140 baseline

            let mae = (building.baseline_energy - analytical_energy).abs();
            report.add_result(building_id, mae);
        }

        report
    }

    pub fn validate_surrogate(
        &self,
        surrogate_path: &str
    ) -> ValidationReport {
        // Test surrogate model

        let manager = SurrogateManager::load_onnx(surrogate_path).unwrap();
        let mut report = ValidationReport::default();

        for (building_id, building) in &self.buildings {
            // Predict using surrogate
            // Compare to ASHRAE 140 baseline

            let mae = (building.baseline_energy - surrogate_energy).abs();
            report.add_result(building_id, mae);
        }

        report
    }
}

#[derive(Default)]
pub struct ValidationReport {
    results: Vec<(String, f64)>,  // (building_id, error)
}

impl ValidationReport {
    pub fn add_result(&mut self, building_id: &str, error: f64) {
        self.results.push((building_id.to_string(), error));
    }

    pub fn mae(&self) -> f64 {
        self.results.iter().map(|(_, e)| e).sum::<f64>() / self.results.len() as f64
    }

    pub fn print_summary(&self) {
        println!("Validation Report:");
        println!("  MAE: {:.4}", self.mae());
        for (building_id, error) in &self.results {
            println!("  {}: {:.4}", building_id, error);
        }
    }
}
```

**Add test**: `src/validation/mod.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ashrae_140_validation() {
        let mut validator = ASHRAE140Validator::new();

        let report_analytical = validator.validate_analytical_engine();
        println!("Analytical MAE: {:.4}", report_analytical.mae());
        assert!(report_analytical.mae() < 0.10);  // Within 10%

        let report_surrogate = validator
            .validate_surrogate("assets/thermal_surrogate_v2_calibrated.onnx");
        println!("Surrogate MAE: {:.4}", report_surrogate.mae());
        assert!(report_surrogate.mae() < 0.05);  // Within 5%
    }
}
```

### Step 4: Generate Validation Report (Week 2-3)

**Create**: `docs/VALIDATION_RESULTS.md`

```markdown
# Phase 5 Validation Results

## Summary
- Analytical MAE: X.XX%
- Surrogate MAE (vs ASHRAE 140): Y.YY%
- Surrogate vs Analytical difference: Z.ZZ%

## Per-Building Results
| Building | Analytical MAE | Surrogate MAE | Improvement |
|----------|---|---|---|
| ... | | | |

## Conclusion
✓ Model meets validation criteria
```

---

## Phase 5 Rust Tests

Add to `src/sim/engine.rs`:

```rust
#[cfg(test)]
mod phase5_tests {
    use crate::ai::surrogate::SurrogateManager;
    use crate::sim::engine::ThermalModel;

    #[test]
    fn test_phase5_real_data_integration() {
        // Load real data
        let real_data = load_real_building_data().unwrap();

        // Run analytical
        let analytical_results = run_analytical_on_real_data(&real_data);

        // Run surrogate (Phase 4 model)
        let surrogate = SurrogateManager::load_onnx(
            "assets/thermal_surrogate.onnx"
        ).unwrap();
        let surrogate_results = run_surrogate_on_real_data(&real_data, &surrogate);

        // Compare
        let mae = compute_mae(&analytical_results, &surrogate_results);
        println!("Phase 5 MAE: {:.4}", mae);

        // Phase 4 success criteria
        assert!(mae < 0.10);  // Within 10%
    }

    #[test]
    fn test_phase5_calibrated_model() {
        // Load calibrated model (Phase 5 output)
        let surrogate = SurrogateManager::load_onnx(
            "assets/thermal_surrogate_v2_calibrated.onnx"
        );

        if let Err(_) = surrogate {
            // Model not yet generated, skip
            return;
        }

        let surrogate = surrogate.unwrap();
        let real_data = load_real_building_data().unwrap();

        let mae = compute_mae_on_real_data(&surrogate, &real_data);

        // Phase 5 success criteria
        assert!(mae < 0.05);  // Within 5%
    }
}
```

---

## Timeline & Milestones

### Week 1: Data & Retraining
- [ ] Monday: Set up data collection
- [ ] Wednesday: Load ASHRAE 140 or NREL data
- [ ] Friday: Retrain model with real data

### Week 2: Validation
- [ ] Monday: Implement ASHRAE 140 validator
- [ ] Wednesday: Run validation tests
- [ ] Friday: Compile results

### Week 3: Documentation & Review
- [ ] Monday: Generate validation report
- [ ] Wednesday: Performance analysis
- [ ] Friday: PR review & merge

---

## Common Issues & Solutions

### Issue: Real data not available
**Solution**:
1. Use ASHRAE 140 reference buildings (publicly available)
2. Generate synthetic data closer to reality (higher complexity)
3. Partner with energy company for anonymized data

### Issue: ONNX model loading fails
**Solution**:
1. Verify libonnxruntime installed: `brew install onnxruntime`
2. Check model file exists: `ls -lh assets/thermal_surrogate.onnx`
3. Validate model: `python -c "import onnx; onnx.load(...)"`

### Issue: Retraining takes too long
**Solution**:
1. Reduce dataset size (sample random subset)
2. Use GPU: `torch.cuda.is_available()`
3. Reduce epochs: start with 20 instead of 100

---

## Success Criteria Checklist

- [ ] Real building dataset loaded
- [ ] Retraining completes without errors
- [ ] Calibrated model exported to ONNX
- [ ] Analytical engine MAE < 10% vs ASHRAE 140
- [ ] Surrogate MAE < 5% vs ASHRAE 140
- [ ] Validation report generated
- [ ] Tests passing: `cargo test --lib`
- [ ] Phase 5 branch merged to main

---

## Next: Phase 6

After Phase 5 complete:
1. Calibrated model ready for production
2. Accuracy validated against standards
3. Ready for GPU acceleration (Phase 6)

**Start Phase 6**:
- Implement GPU inference backends
- Benchmark on multi-device setup
- Quantize model for faster inference

---

**Version**: 1.0 | **Last Updated**: Nov 21, 2024
