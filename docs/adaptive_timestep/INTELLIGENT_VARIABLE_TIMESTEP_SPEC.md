# Intelligent Variable Timestep Simulation System

## Context

Fluxion's current adaptive timestep implementation (`adaptive_timestep.rs`) uses a static threshold-based approach (τ ≥ 2 hours → 6-minute timesteps). This is a good baseline but leaves significant optimization potential on the table.

Real building energy simulations exhibit complex, predictable patterns:
- Occupancy schedules create predictable thermal mass charging/discharging cycles
- Weather forecasts enable anticipatory timestep selection
- HVAC availability windows affect thermal response urgency
- Weekly patterns repeat with minor variations

This spec proposes a multi-phase intelligent system that uses schedule analysis, runtime feedback, weather integration, and ML surrogate control to optimize timestep selection in real-time.

---

## Problem Statement

### Current Limitations
1. **Static threshold**: Only considers building thermal mass, ignores operational patterns
2. **No look-ahead**: Doesn't use weather forecast or occupancy anticipation
3. **No feedback**: Doesn't adapt based on observed simulation behavior
4. **Uniform sub-cycling**: Applies same timestep across entire simulation, even when unnecessary

### Opportunity
- 8760-hour annual simulations spend ~60-70% of time in predictable low-activity periods
- Weather-driven thermal transients are often short (< 4 hours)
- Occupancy-driven loads follow ~168-hour weekly cycles

### Goal
Reduce computational cost by 2-4x for typical simulations while maintaining ±1% accuracy vs. fine-timestep baseline.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Intelligent Timestep Controller                      │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐   │
│  │  Pre-Simulation  │   │   Weekly Review  │   │  Daily Weather   │   │
│  │   Schedule       │   │    & Adaptation │   │    Forecast      │   │
│  │    Analysis      │   │                 │   │     Engine       │   │
│  └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘   │
│           │                     │                     │              │
│           ▼                     ▼                     ▼              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Timestep Profile Manager                      │   │
│  │  - Weekly profiles (168-hour blocks)                            │   │
│  │  - Daily overrides based on weather                             │   │
│  │  - Real-time adjustment signals                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   ML Surrogate Controller                        │   │
│  │  - Feature extraction                                           │   │
│  │  - Inference (optional, configurable)                            │   │
│  │  - Fallback to rule-based if ML unavailable                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Pre-Simulation Schedule Analysis

### Objective
Analyze building operational schedules and pre-compute an optimal timestep profile.

### Inputs
```rust
pub struct ScheduleAnalysis {
    occupancy_profile: OccupancyProfile,
    lighting_schedule: LightingSchedule,
    equipment_schedule: Vec<Equipment>,
    hvac_availability: Vec<TimeAvailability>,
}

pub struct TimeAvailability {
    day_of_week: u8,        // 0-6 (Mon-Sun)
    hour_start: u8,         // 0-23
    hour_end: u8,           // 0-23
    is_controllable: bool,  // Can HVAC modulate freely?
}
```

### Thermal Activity Classification

```rust
pub enum ThermalActivity {
    /// High internal gains + HVAC available → needs fine resolution
    HighActivity,
    /// Moderate activity, HVAC compensating → standard resolution
    ModerateActivity,
    /// Low/zero occupancy, free-float possible → coarse resolution acceptable
    LowActivity,
    /// Quick thermal transient expected → sub-hourly recommended
    Transient,
}
```

### Pre-Simulation Analysis Algorithm

```
1. For each hour h in typical week (168 hours):
   a. Calculate occupancy_density(h) = occupants / zone_area
   b. Calculate internal_gain_density(h) = (lighting + equipment + occupancy) / zone_area
   c. Classify HVAC_availability(h) = available / limited / unavailable
   d. Compute thermal_stress(h) = internal_gain_density * HVAC_availability_factor
   e. Assign ThermalActivity(h)

2. Build weekly profile:
   for day in 0..7:
     for hour in 0..23:
       profile[day * 24 + hour] = classify(hour, day)

3. Identify transient windows (consecutive HighActivity < 4 hours)

4. Generate baseline timestep sequence:
   - HighActivity: dt = 15 minutes
   - ModerateActivity: dt = 30 minutes
   - LowActivity: dt = 60 minutes
   - Transient: dt = 5 minutes
```

### Output
```rust
pub struct TimestepProfile {
    /// Weekly pattern (168 entries)
    weekly_pattern: Vec<ThermalActivity>,
    /// Timestep sequence for typical week
    weekly_timesteps: Vec<Duration>,
    /// Override windows for transient detection
    transient_windows: Vec<TimeWindow>,
}

pub struct TimeWindow {
    start_hour: usize,
    end_hour: usize,
    recommended_dt: Duration,
}
```

---

## Phase 2: Weekly Review & Adaptation

### Objective
At the end of each simulated week, analyze performance and adjust next week's profile.

### Performance Metrics

```rust
pub struct WeeklyMetrics {
    /// Total sub-cycling events (timestep where solution oscillated)
    sub_cycling_count: usize,
    /// Average convergence rate per timestep family
    convergence_rate: f64,
    /// Energy smoothness (std dev of hourly energy changes)
    energy_smoothness: f64,
    /// Actual vs predicted occupancy adherence
    occupancy_adherence: f64,
    /// Weather forecast accuracy (if using forecast)
    forecast_accuracy: f64,
}
```

### Adaptation Rules

| Metric Condition | Adjustment |
|-----------------|------------|
| `sub_cycling_count > threshold` | Decrease dt by 25% for that activity class |
| `convergence_rate < 0.95` | Split transient window into smaller dt |
| `energy_smoothness > expected` | Increase dt by 15% for stable periods |
| `forecast_accuracy < 0.8` | Reduce reliance on forecast, use conservative dt |

### Implementation
```rust
pub fn adapt_weekly_profile(
    current: &TimestepProfile,
    metrics: &WeeklyMetrics,
) -> TimestepProfile {
    // Create adjusted profile based on metrics
    let mut adjusted = current.clone();

    if metrics.sub_cycling_count > 50 {
        // Aggressive reduction for oscillatory behavior
        adjusted.reduce_dt_for_activity(ThermalActivity::ModerateActivity, 0.75);
    }
    if metrics.energy_smoothness > 0.1 {
        // Increase efficiency for stable periods
        adjusted.increase_dt_for_activity(ThermalActivity::LowActivity, 1.15);
    }

    adjusted
}
```

---

## Phase 3: Daily Weather Forecast Integration

### Objective
At the start of each simulated day, check weather forecast and adjust timestep profile.

### Weather Features

```rust
pub struct WeatherForecast {
    /// Hourly temperatures for next 24 hours
    temperatures: [f64; 24],
    /// Hourly solar radiation (W/m²)
    solar_radiation: [f64; 24],
    /// Cloud cover fraction (0-1)
    cloud_cover: [f64; 24],
}

pub struct WeatherAnalysis {
    /// Maximum temperature swing in next 24h
    max_temp_swing: f64,
    /// Solar variability index (std dev of radiation)
    solar_variability: f64,
    /// Hours until next significant weather change
    time_to_transient: usize,
    /// Confidence in forecast (0-1)
    forecast_confidence: f64,
}
```

### Daily Override Logic

```python
def compute_daily_overrides(profile: TimestepProfile, forecast: WeatherForecast) -> TimestepProfile:
    analysis = analyze_weather(forecast)

    overrides = []

    if analysis.max_temp_swing > 10.0:  # > 10°C swing
        # High thermal stress expected - use finer dt
        for h in find_transient_hours(forecast):
            overrides.append((h, Duration::minutes(10)))

    if analysis.solar_variability > 0.3:  # High variability
        # Solar-driven transients likely
        for h in range(analysis.sunrise_hour, analysis.sunset_hour):
            if forecast.solar_radiation[h] > 400:  # High solar
                overrides.append((h, Duration::minutes(15)))

    if analysis.forecast_confidence < 0.7:
        # Uncertain forecast - use conservative dt
        for h in range(24):
            overrides.append((h, min(profile.weekly_timesteps[h], Duration::minutes(30))))

    return apply_overrides(profile, overrides)
```

---

## Phase 4: ML Surrogate Controller

### Objective
Train a model to predict optimal timestep multiplier given current context.

### Feature Engineering

```rust
pub struct TimestepFeatures {
    // Time features (normalized)
    hour_of_day: f32,        // 0-1
    day_of_week: f32,        // 0-1
    day_of_year: f32,        // 0-1

    // Thermal state
    time_constant_hours: f32,  // τ from building params
    current_zone_temp: f32,    // °C
    thermal_mass_charge: f32, // 0-1 (mass temp - zone temp)

    // Internal gains prediction
    expected_occupancy: f32,   // 0-1
    expected_lighting: f32,    // W/m²
    expected_equipment: f32,   // W/m²

    // Weather features (look-ahead)
    temp_swing_next_6h: f32,    // °C
    solar_avg_next_6h: f32,    // W/m²

    // HVAC state
    hvac_mode: u8,             // 0=off, 1=heating, 2=cooling, 3=auto
    hvac_capacity_available: f32,  // 0-1

    // History features
    recent_convergence_rate: f32,  // 0-1
    recent_subcycling_count: u8,   // count in last 6h

    // Building characteristics
    building_mass_class: u8,   // 0=light, 1=medium, 2=heavy
    window_ratio: f32,          // fraction
}

pub struct TimestepTarget {
    /// Recommended timestep multiplier vs baseline
    dt_multiplier: f32,  // 0.25 = 4x faster, 2.0 = 2x slower
}
```

### Training Data Generation

See "ML Training Data Generation Approach" section below.

### Model Architecture

```rust
pub struct TimestepControllerModel {
    // Lightweight model for real-time inference
    // Target: < 1ms inference time

    layers: Vec<LinearLayer>,
    activation: ReLU,
    dropout: 0.1,

    // Latency target: ~0.5ms on CPU
    // Memory target: < 1MB model size
}

impl TimestepControllerModel {
    pub fn predict(&self, features: &TimestepFeatures) -> f32 {
        // Forward pass
        let x = features.to_tensor();
        let normalized = self.normalize(x);
        let hidden = self.layers.iter().fold(normalized, |x, l| l.relu(l(x)));
        let output = self.output_layer(hidden);
        output[0]  // dt_multiplier
    }
}
```

### Runtime Inference Pipeline

```
Simulation Loop:
  1. Every timestep:
     a. Extract features from current state
     b. If use_ml_controller:
        - Predict dt_multiplier via model
        - Apply business rules (safety bounds)
     c. Else:
        - Use rule-based fallback
     d. Update dt = base_dt * dt_multiplier
     e. Clamp dt to [min_dt, max_dt]

  2. Every 6 hours:
     a. Collect runtime metrics
     b. Update convergence rate estimate
```

### Safety Bounds

```rust
pub fn apply_dt_multiplier_rules(multiplier: f32, context: &TimestepFeatures) -> f32 {
    let mut dt = multiplier;

    // Hard bounds
    dt = dt.clamp(0.25, 4.0);  // Never more than 4x faster/slower than base

    // Safety overrides
    if context.recent_subcycling_count > 3 {
        dt = dt.min(1.0);  // Reduce if oscillating
    }

    if context.hvac_mode == 0 && context.thermal_mass_charge > 0.8 {
        dt = dt.min(2.0);  // Allow faster when free-float and mass charged
    }

    // Weather override
    if context.temp_swing_next_6h > 15.0 {
        dt = dt.min(0.5);  // Finer dt for extreme weather
    }

    dt
}
```

---

## Implementation Phases

### Phase 1: Schedule Analysis (Core Prerequisite)
- [ ] Define `ThermalActivity` enum
- [ ] Implement `ScheduleAnalyzer` struct
- [ ] Build `TimestepProfile` from schedules
- [ ] Integrate with `ThermalModel::solve_timesteps()`

### Phase 2: Weekly Adaptation (Optional Enhancement)
- [ ] Define `WeeklyMetrics` struct
- [ ] Implement profile adaptation logic
- [ ] Add weekly review hook in simulation loop

### Phase 3: Weather Integration (Optional Enhancement)
- [ ] Add `WeatherForecast` struct
- [ ] Implement forecast parsing (EPW or custom)
- [ ] Build daily override logic

### Phase 4: ML Controller (Future Work)
- [ ] Generate synthetic training data
- [ ] Train and validate model
- [ ] Implement runtime inference
- [ ] Add fallback to rule-based system

---

## Configuration

```toml
[adaptive_timestep]
# Enable intelligent timestep system
enabled = true

# Base timestep (seconds) - 3600 = 1 hour
base_dt = 3600

# Min/max bounds
min_dt = 300      # 5 minutes minimum
max_dt = 7200      # 2 hours maximum

# Phase toggles
use_schedule_analysis = true
use_weekly_adaptation = false
use_weather_forecast = false
use_ml_controller = false

# ML controller options (when use_ml_controller = true)
ml_model_path = "models/timestep_controller.onnx"
ml_fallback_to_rules = true

# Safety thresholds
subcycling_threshold = 50
max_consecutive_small_dt = 10
```

---

## Backward Compatibility

- `TimestepMode::Fixed` and `TimestepMode::Adaptive` continue to work as before
- `ThermalModel::solve_timesteps()` behavior unchanged when `use_*` flags are false
- New configuration only activates when explicitly enabled

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Computational speedup | 2-4x vs 6-min constant | benchmark with/without |
| Accuracy vs fine baseline | ±1% annual EUI | compare to 15-min reference |
| ML controller latency | < 1ms per inference | profiling |
| Coverage (typical buildings) | > 90% of ASHRAE 140 cases | validation suite |

---

## Files to Modify/Create

```
src/sim/
  ├── adaptive_timestep.rs      # Extend with new structs
  ├── timestep_profile.rs        # NEW: TimestepProfile and ScheduleAnalyzer
  ├── weather_forecast.rs       # NEW: Weather forecast integration
  ├── ml_timestep_controller.rs # NEW: ML surrogate (optional)
  └── engine.rs                 # Modify: integrate with solve_timesteps()

models/
  └── timestep_controller.onnx   # NEW: trained ML model

tests/
  ├── test_timestep_profile.rs   # NEW
  └── test_schedule_analyzer.rs   # NEW
```

---

## References

- Gaffer on Games: "Fix Your Timestep" - foundational variable dt principles
- EnergyPlus documentation on variable timestep strategies
- ASHRAE 140 test cases for validation
