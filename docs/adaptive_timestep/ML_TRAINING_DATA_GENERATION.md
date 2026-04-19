# ML Training Data Generation for Timestep Controller

## Overview

Training a surrogate model to control variable timestep requires diverse, representative synthetic data covering:
- Different building thermal masses (τ = 0.5h to 12h)
- Various occupancy/equipment schedules
- Weather patterns (calm, transient, extreme)
- HVAC operational modes (heating, cooling, off, auto)

## Data Generation Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Training Data Generation Pipeline                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Building   │    │    Weather   │    │   Internal   │          │
│  │  Parameters  │    │   Profiles   │    │     Loads    │          │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘          │
│         │                   │                   │                    │
│         ▼                   ▼                   ▼                    │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │              Scenario Generator                          │        │
│  │  - Combine parameters into valid building configs      │        │
│  │  - Ensure diverse coverage (Latin hypercube sampling)    │        │
│  └────────────────────────────┬────────────────────────────┘        │
│                               │                                     │
│                               ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │              Simulation Executor                          │        │
│  │  - Run simulation with multiple timestep configurations   │        │
│  │  - Record metrics per timestep                            │        │
│  └────────────────────────────┬────────────────────────────┘        │
│                               │                                     │
│                               ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │              Feature/Target Extraction                   │        │
│  │  - Compute features per simulation step                  │        │
│  │  - Compute "optimal" dt_multiplier as target             │        │
│  └────────────────────────────┬────────────────────────────┘        │
│                               │                                     │
│                               ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │              Dataset Assembly                            │        │
│  │  - Shuffle and split (train/val/test)                     │        │
│  │  - Normalize features                                    │        │
│  │  - Export to ONNX or numpy format                        │        │
│  └─────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Scenario Generation

### Building Parameter Space

```rust
pub struct BuildingConfig {
    // Thermal characteristics
    time_constant_hours: f32,  // τ range: 0.5 - 12.0 hours

    // Geometry
    zone_area_m2: f32,          // 20 - 500 m²
    window_ratio: f32,          // 0.1 - 0.5
    ceiling_height_m: f32,      // 2.4 - 4.0

    // Construction (for U-values)
    wall_u_value: f32,          // 0.2 - 1.5 W/m²K
    roof_u_value: f32,          // 0.15 - 0.8 W/m²K

    // Internal loads
    lighting_density: f32,     // 5 - 20 W/m²
    equipment_density: f32,     // 5 - 25 W/m²
    occupancy_density: f32,     // 0.05 - 0.2 persons/m²

    // Setpoints
    heating_setpoint: f32,      // 15 - 22 °C
    cooling_setpoint: f32,      // 22 - 30 °C
    deadband: f32,              // 2 - 5 K
}

impl BuildingConfig {
    pub fn sample_random(rng: &mut impl Rng) -> Self {
        BuildingConfig {
            time_constant_hours: rng.gen_range(0.5..12.0),
            zone_area_m2: rng.gen_range(20.0..500.0),
            window_ratio: rng.gen_range(0.1..0.5),
            ceiling_height_m: rng.gen_range(2.4..4.0),
            wall_u_value: rng.gen_range(0.2..1.5),
            roof_u_value: rng.gen_range(0.15..0.8),
            lighting_density: rng.gen_range(5.0..20.0),
            equipment_density: rng.gen_range(5.0..25.0),
            occupancy_density: rng.gen_range(0.05..0.2),
            heating_setpoint: rng.gen_range(15.0..22.0),
            cooling_setpoint: rng.gen_range(22.0..30.0),
            deadband: rng.gen_range(2.0..5.0),
        }
    }
}
```

### Occupancy Schedule Space

```rust
pub enum OccupancyScheduleType {
    Office,      // 8am-6pm Mon-Fri
    Retail,      // 10am-9pm daily
    Residential, // 6pm-8am daily + weekends
    Hospital,    // 24/7
    Warehouse,   // 6am-10pm Mon-Sat
    School,      // 8am-4pm Mon-Fri
}

pub struct OccupancySchedule {
    schedule_type: OccupancyScheduleType,
    weekend_offset_hours: f32,  // 0-3 hours random variation
    holiday_factor: f32,       // 0.0-0.3 fraction of days as holidays
}
```

### Weather Profile Space

```rust
pub enum WeatherProfileType {
    Temperate,      // 5-25°C, low variability
    Continental,    // -10-35°C, high seasonal variability
    Tropical,       // 20-35°C, high humidity, monsoon
    Desert,         // 10-45°C, extreme diurnal swing
    Coastal,       // 10-30°C, moderate marine influence
}

pub struct WeatherProfile {
    profile_type: WeatherProfileType,
    season_factor: f32,         // 0.0 = summer, 1.0 = winter
    transient_probability: f32, // 0.0-0.3 probability of weather front
}
```

### Latin Hypercube Sampling

To ensure diverse coverage of the parameter space, use Latin Hypercube Sampling (LHS):

```rust
pub fn generate_scenarios(n_scenarios: usize) -> Vec<SimulationScenario> {
    let mut rng = StdRng::seed_from_u64(42);

    // Define parameter bounds
    let building_params = vec![
        ("time_constant_hours", 0.5, 12.0),
        ("zone_area_m2", 20.0, 500.0),
        ("window_ratio", 0.1, 0.5),
        // ... etc
    ];

    // Generate LHS samples
    let lhs_samples = lhc_sample(&building_params, n_scenarios, &mut rng);

    lhs_samples.into_iter().enumerate().map(|(i, params)| {
        SimulationScenario {
            scenario_id: format!("scenario_{:04}", i),
            building: BuildingConfig::from_lhs(params),
            occupancy: OccupancySchedule::sample_random(&mut rng),
            weather: WeatherProfile::sample_random(&mut rng),
        }
    }).collect()
}
```

**Target: 500-2000 scenarios** covering diverse building types and climates.

---

## Simulation Execution

### Multi-Configuration Simulation

For each scenario, run simulation with multiple timestep configurations:

```rust
pub struct TimestepConfig {
    name: &'static str,
    dt_seconds: f32,
}

static TIMESTEP_CONFIGS: &[TimestepConfig] = &[
    TimestepConfig { name: "baseline_60min", dt_seconds: 3600.0 },
    TimestepConfig { name: "fine_15min", dt_seconds: 900.0 },
    TimestepConfig { name: "very_fine_5min", dt_seconds: 300.0 },
];
```

### Simulation Output Per Timestep

```rust
pub struct TimestepRecord {
    // Identifier
    scenario_id: String,
    simulation_day: u16,
    hour_of_day: u8,

    // Input features
    features: TimestepFeatures,

    // Simulation metrics
    zone_temperature: f32,
    thermal_mass_charge: f32,
    hvac_power: f32,
    energy_delta: f32,

    // Multiple dt results for comparison
    dt_60min_temperature: f32,
    dt_15min_temperature: f32,
    dt_5min_temperature: f32,

    // Convergence info
    subcycling_detected: bool,
    convergence_iterations: u8,
}
```

### Target Computation

The target `dt_multiplier` is computed by comparing solution quality across timestep configurations:

```rust
pub fn compute_optimal_multiplier(
    record: &TimestepRecord,
    base_dt: f32,
) -> f32 {
    let temp_60 = record.dt_60min_temperature;
    let temp_15 = record.dt_15min_temperature;
    let temp_5 = record.dt_5min_temperature;

    // Temperature difference between 15min and 60min baseline
    let error_60_vs_15 = (temp_60 - temp_15).abs();

    // Temperature difference between 5min and 15min (refinement margin)
    let error_15_vs_5 = (temp_15 - temp_5).abs();

    // If 60min solution is very close to 15min, we can use larger dt
    if error_60_vs_15 < 0.05 {
        // Very stable - can use 2x base dt
        return 2.0;
    } else if error_60_vs_15 < 0.2 {
        // Moderately stable - use standard dt
        return 1.0;
    } else if error_60_vs_15 < 0.5 {
        // Some error - use smaller dt
        return 0.5;
    } else {
        // High error - need fine dt
        return 0.25;
    }
}
```

### Feature Extraction Per Timestep

```rust
impl TimestepFeatures {
    pub fn extract(
        context: &SimulationContext,
        hour: usize,
        day: usize,
    ) -> Self {
        let thermal_mass = context.thermal_capacitance_jk / 1e6; // MJ/K

        TimestepFeatures {
            hour_of_day: (hour as f32) / 24.0,
            day_of_week: ((day % 7) as f32) / 7.0,
            day_of_year: ((day % 365) as f32) / 365.0,

            time_constant_hours: thermal_mass / context.conductance_wk,
            current_zone_temp: context.zone_temperature,
            thermal_mass_charge: context.mass_temperature - context.zone_temperature,

            expected_occupancy: context.occupancy_fraction(hour),
            expected_lighting: context.lighting_wm2(hour),
            expected_equipment: context.equipment_wm2(hour),

            temp_swing_next_6h: context.weather.max_temp_swing_6h(hour),
            solar_avg_next_6h: context.weather.solar_avg_6h(hour),

            hvac_mode: context.hvac_mode as u8,
            hvac_capacity_available: context.hvac_capacity_ratio,

            recent_convergence_rate: context.convergence_rate_6h,
            recent_subcycling_count: context.subcycling_count_6h,

            building_mass_class: classify_mass(thermal_mass),
            window_ratio: context.window_ratio,
        }
    }
}
```

---

## Dataset Assembly

### Data Volume Targets

| Dataset Size | Scenarios | Days/Sim | Steps/Scenario | Total Records |
|-------------|-----------|----------|----------------|---------------|
| Small | 200 | 30 | 720 | 144,000 |
| Medium | 500 | 30 | 720 | 360,000 |
| Large | 1000 | 30 | 720 | 720,000 |

### Feature Normalization

```rust
pub struct NormalizationParams {
    mean: Tensor,
    std: Tensor,
}

impl TimestepDataset {
    pub fn normalize(&mut self, params: &NormalizationParams) {
        for record in &mut self.records {
            record.features = params.normalize(record.features);
        }
    }
}
```

### Train/Val/Test Split

```rust
pub fn split_dataset(
    dataset: TimestepDataset,
    train_frac: f32,
    val_frac: f32,
) -> (TimestepDataset, TimestepDataset, TimestepDataset) {
    let mut shuffled = dataset.shuffle(seed);

    let n_total = shuffled.records.len();
    let n_train = (n_total as f32 * train_frac) as usize;
    let n_val = (n_total as f32 * val_frac) as usize;

    let train = shuffled.records[..n_train].to_vec();
    let val = shuffled.records[n_train..n_train+n_val].to_vec();
    let test = shuffled.records[n_train+n_val..].to_vec();

    (TimestepDataset::new(train), TimestepDataset::new(val), TimestepDataset::new(test))
}
```

---

## Model Training

### Architecture (Target: <1MB, <1ms inference)

```rust
pub struct TimestepControllerModel {
    // 4-layer MLP with 64 hidden units
    layers: vec![
        Linear(28, 64),   // input features -> hidden
        Linear(64, 64),    // hidden -> hidden
        Linear(64, 32),    // hidden -> hidden
        Linear(32, 1),     // hidden -> output
    ],
    activation: LeakyReLU(0.1),
    dropout: 0.1,
    output_activation: Sigmoid,  // dt_multiplier in [0, 1]
}
```

### Training Configuration

```rust
pub struct TrainingConfig {
    learning_rate: 1e-3,
    batch_size: 256,
    epochs: 100,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    weight_decay: 1e-4,
    early_stopping_patience: 10,
}

pub struct LossFunction {
    // Primary: Mean squared error on dt_multiplier
    primary: MSE,

    // Auxiliary: Penalize underestimation of error
    auxiliary: |pred, target, error_15min| {
        if pred < target && error_15min > 0.5 {
            2.0 * (pred - target).powi(2)
        } else {
            0.0
        }
    },
}
```

---

## Output Format

### ONNX Model Export

```rust
pub fn export_to_onnx(model: &TimestepControllerModel, path: &Path) -> Result<()> {
    // Export to ONNX for cross-platform inference
    // Include normalization parameters in model metadata
}
```

### Dataset Export

```rust
pub fn export_dataset(dataset: &TimestepDataset, path: &Path) -> Result<()> {
    // Export as numpy .npz for Python training
    // Or as JSON for Rust training
}
```

---

## Quality Assurance

### Data Validation

```rust
pub fn validate_dataset(dataset: &TimestepDataset) -> ValidationResult {
    let mut issues = Vec::new();

    // Check for NaN/Inf
    for (i, record) in dataset.records.iter().enumerate() {
        if record.features.contains_nan() {
            issues.push(format!("NaN in features at record {}", i));
        }
        if !record.dt_multiplier.is_finite() {
            issues.push(format!("Invalid target at record {}", i));
        }
    }

    // Check target distribution
    let targets: Vec<f32> = dataset.records.iter().map(|r| r.dt_multiplier).collect();
    let target_mean = mean(&targets);
    let target_std = std_dev(&targets);

    if target_std < 0.1 {
        issues.push("Target distribution too narrow - may not generalize".to_string());
    }

    ValidationResult { issues }
}
```

### Model Validation

```rust
pub fn validate_model(model: &TimestepControllerModel, test_set: &TimestepDataset) -> ModelMetrics {
    let mut predictions = Vec::new();
    let mut errors = Vec::new();

    for record in test_set.records.iter() {
        let pred = model.predict(&record.features);
        predictions.push(pred);

        let error = (pred - record.dt_multiplier).abs();
        errors.push(error);
    }

    ModelMetrics {
        mae: mean(errors),
        rmse: sqrt(mean(errors.iter().map(|e| e.powi(2)))),
        r2: compute_r2(&predictions, &test_set.targets),
        # within_5%: percentage where |pred - target| < 0.05,
        # within_10%: percentage where |pred - target| < 0.10,
    }
}
```

---

## Implementation Roadmap

### Phase A: Data Generation Infrastructure
1. [ ] Define `BuildingConfig`, `OccupancySchedule`, `WeatherProfile` structs
2. [ ] Implement Latin Hypercube sampling for scenario generation
3. [ ] Create simulation runner with multiple dt configurations
4. [ ] Build feature extraction and target computation

### Phase B: Dataset Generation
1. [ ] Generate 500+ scenarios covering parameter space
2. [ ] Run simulations and collect timestep records
3. [ ] Compute targets and assemble dataset
4. [ ] Validate dataset quality

### Phase C: Model Training
1. [ ] Implement model architecture in PyTorch or burn (Rust)
2. [ ] Train with early stopping and validation monitoring
3. [ ] Export to ONNX for inference

### Phase D: Integration
1. [ ] Add `TimestepControllerModel` to Fluxion
2. [ ] Implement runtime inference pipeline
3. [ ] Add safety bounds and fallback logic

---

## Synthetic Data Example

### Sample Scenario

```json
{
  "scenario_id": "scenario_0042",
  "building": {
    "time_constant_hours": 5.3,
    "zone_area_m2": 150.0,
    "window_ratio": 0.25,
    "wall_u_value": 0.5,
    "lighting_density": 10.0,
    "equipment_density": 12.0,
    "occupancy_density": 0.1
  },
  "occupancy": {
    "schedule_type": "Office",
    "weekend_offset_hours": 0.5,
    "holiday_factor": 0.1
  },
  "weather": {
    "profile_type": "Continental",
    "season_factor": 0.5
  },
  "records": [
    {
      "day": 1,
      "hour": 9,
      "features": {
        "hour_of_day": 0.375,
        "day_of_week": 0.143,
        "time_constant_hours": 5.3,
        "thermal_mass_charge": 0.65,
        "expected_occupancy": 0.9,
        "hvac_mode": 2,
        "temp_swing_next_6h": 4.2
      },
      "dt_multiplier": 0.5
    }
  ]
}
```

---

## References

- "Learning-Based Timestep Reduction for Building Energy Simulation" (similar work)
- LHS methodology: "Latin Hypercube Sampling: A Tool for Sampling a Finite Universe"
- ONNX runtime for cross-platform inference
