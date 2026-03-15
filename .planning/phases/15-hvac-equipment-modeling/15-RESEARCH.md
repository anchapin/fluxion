# Phase 15: HVAC Equipment Modeling - Research

**Researched:** 2026-03-13
**Domain:** HVAC equipment modeling, efficiency curves, control strategies
**Confidence:** MEDIUM

## Summary

Phase 15 requires implementing realistic HVAC equipment models with polynomial efficiency curves, variable capacity control, and cycling loss modeling. The research identifies a clear path forward using the existing hvac.rs infrastructure, building on Phase 14 thermal mass corrections, and leveraging ASHRAE 140 validation framework. Key implementation decisions from CONTEXT.md provide strong guidance on approach: cubic polynomial efficiency curves, unified VariableCapacityEquipment trait, predictive control with thermal inertia, and combined cycling loss modeling.

**Primary recommendation:** Implement the VariableCapacityEquipment trait first, then enhance existing VAV, CAV, and HeatPump structures with polynomial efficiency curves, add new Chiller and Boiler models, integrate predictive control into ThermalModel::step_physics, and validate against ASHRAE Cases 800-810.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Efficiency Curve Approach:**
- Curve type: Polynomial curves (not lookup tables or simple linear degradation)
- Curve inputs: PLR + temperature (2D polynomial/surface)
- Polynomial degree: Cubic (degree 3)
- Coefficient source: AHRI reference data (validated against ASHRAE Cases 800-810)

**Equipment Depth:**
- Model detail: Variable capacity (continuous modulation) - supports continuous 0-100% modulation
- Integration approach: Unified trait - Create `VariableCapacityEquipment` trait for all equipment types
- Trait methods: Capacity, efficiency, power, and PLR tracking - Core methods: calculate_capacity(), calculate_efficiency(), calculate_power(), plus PLR tracking and runtime hours
- Variable capacity limits: AHRI + ASHRAE validation - Use both AHRI reference data and ASHRAE 140 Cases 800-810

**Control Strategies:**
- Control type: Variable capacity modulation - HVAC control continuously modulates capacity based on conditions
- Control logic: Predictive with thermal inertia - Control signal considers thermal inertia, not just current temperature
- Predictive factors: Temperature, rate of change, and thermal mass state - Current zone temp, dT/dt, thermal mass temp from 5R1C
- Inertia tuning: ASHRAE + Guideline 14 stability criteria - Tune thermal inertia gain to match ASHRAE Cases 800-810

**Cycling & Losses:**
- Cycling model: Combined approach (startup penalty + minimum runtime) - Both fixed energy penalty for startup AND minimum runtime constraints
- Startup penalty calculation: Combined penalty model - Part-load ratio degradation + separate startup energy penalty
- Minimum runtime enforcement: Combined tracking - Per-timestep state tracking + cumulative hours tracking
- Penalty values and limits: AHRI reference data - Use AHRI Standard data for startup energy penalties and minimum runtime limits

### Claude's Discretion

- Exact polynomial coefficients (researcher will determine from AHRI data)
- Thermal inertia gain factor value (tune against ASHRAE + Guideline 14)
- Minimum runtime duration (5-15 minutes range, AHRI provides guidance)
- PLR degradation curve shape (researcher fits to AHRI data)

### Deferred Ideas (OUT OF SCOPE)

None - discussion stayed within phase scope. All decisions relate to efficiency curves, equipment depth, control strategies, and cycling losses as defined in Phase 15 requirements.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| HVAC-01 | Implement VAV (Variable Air Volume) system modeling | Existing VAVTerminal struct in hvac.rs; enhance with VariableCapacityEquipment trait implementation |
| HVAC-02 | Implement CAV (Constant Air Volume) system modeling | Existing CAVSystem struct in hvac.rs; enhance with VariableCapacityEquipment trait implementation |
| HVAC-03 | Implement heat pump equipment modeling (with efficiency curves) | Existing HeatPump struct in hvac.rs; replace linear degradation with cubic polynomial curves |
| HVAC-04 | Implement chiller equipment modeling (with part-load ratios) | New Chiller struct implementing VariableCapacityEquipment trait |
| HVAC-05 | Implement boiler equipment modeling (with part-load ratios) | New Boiler struct implementing VariableCapacityEquipment trait |
| HVAC-06 | Implement economizer mode (free cooling) | Add economizer method to HVACSystemType; requires Phase 16 psychrometrics for enthalpy calculations |
| HVAC-07 | Implement equipment efficiency curves and part-load degradation | Polynomial curve module with cubic degree; 2D surface for PLR + temperature inputs |
| HVAC-08 | Implement cycling loss modeling | Track equipment state (on/off), runtime hours, startup count; apply penalties in energy calculation |
| HVAC-09 | Support configurable HVAC control strategies (setpoints, deadbands, schedules) | Enhance IdealHVACController with predictive control using thermal inertia factors |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Rust std | 1.70+ | Core language, f64 arithmetic, trait system | Built-in, no dependencies |
| serde | 1.0 | Serialization of equipment structs | Already used in hvac.rs for VAVTerminal, CAVSystem, HeatPump |
| serde_json | 1.0 | Configuration loading for efficiency curve coefficients | Already a dependency, enables JSON config files |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| approx | 0.5 | Floating point comparisons in tests | Dev-dependency already present; use for COP/power comparisons |
| rstest | 0.18 | Parameterized testing for efficiency curves | Dev-dependency already present; test multiple PLR/temperature combos |
| rayon | 1.10 | Parallel evaluation (respecting BatchOracle pattern) | Already a dependency; do NOT use for nested parallelism in equipment calculations |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Polynomial curves | Lookup tables | Polynomials are more flexible and maintainable; lookup tables require discrete interpolation |
| Cubic polynomial | Quadratic/linear | Cubic captures S-shaped efficiency degradation patterns typical in HVAC equipment |
| Unified trait | Separate implementations per equipment type | Trait enables code reuse and consistent testing; separates concern of interface vs implementation |
| AHRI coefficients | ASHRAE reference only | AHRI provides manufacturer reference data; ASHRAE 140 validates but doesn't specify curves |

**Installation:**
```bash
# All dependencies already present in Cargo.toml
# No new crates required
cargo build --release
```

## Architecture Patterns

### Recommended Project Structure
```
src/sim/
├── hvac/
│   ├── mod.rs                    # Module export, public API
│   ├── equipment.rs              # VariableCapacityEquipment trait, Chiller, Boiler
│   ├── efficiency_curves.rs      # Polynomial curve evaluation, coefficient structs
│   ├── control.rs               # Predictive control logic with thermal inertia
│   ├── cycling.rs               # Startup penalties, minimum runtime tracking
│   └── economizer.rs            # Economizer mode (depends on Phase 16 psychrometrics)
├── engine.rs                    # ThermalModel with enhanced HVAC integration
└── hvac.rs                     # Existing VAVTerminal, CAVSystem, HeatPump (enhance)
```

### Pattern 1: VariableCapacityEquipment Trait

**What:** Unified trait for all variable-capacity HVAC equipment types

**When to use:** All equipment that supports continuous modulation (VAV, CAV, HeatPump, Chiller, Boiler)

**Example:**
```rust
/// Trait for variable-capacity HVAC equipment with efficiency curves
pub trait VariableCapacityEquipment: Send + Sync + Clone {
    /// Calculate actual capacity at current conditions (W)
    fn calculate_capacity(&self, plr: f64, outdoor_temp: f64) -> f64;

    /// Calculate efficiency (COP or EER) at current conditions
    fn calculate_efficiency(&self, plr: f64, outdoor_temp: f64, mode: HVACMode) -> f64;

    /// Calculate power consumption at current conditions (W)
    fn calculate_power(&self, load: f64, outdoor_temp: f64, mode: HVACMode) -> f64;

    /// Get rated capacity (W)
    fn rated_capacity(&self) -> f64;

    /// Get rated efficiency at design conditions
    fn rated_efficiency(&self, mode: HVACMode) -> f64;

    /// Get current part-load ratio (0-1)
    fn current_plr(&self) -> f64;

    /// Update equipment state for next timestep
    fn update_state(&mut self, current_load: f64, outdoor_temp: f64, mode: HVACMode);
}
```

### Pattern 2: Polynomial Efficiency Curves

**What:** Cubic polynomial curves for COP/efficiency as function of PLR and temperature

**When to use:** All equipment efficiency calculations (heating, cooling, part-load degradation)

**Example:**
```rust
/// Polynomial efficiency curve coefficients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyCurve {
    /// Coefficients for COP = a + b*PLR + c*PLR^2 + d*PLR^3
    pub plr_coefficients: [f64; 4],
    /// Temperature coefficient (COP degrades per degree from design)
    pub temp_coefficient: f64,
    /// Design outdoor temperature (°C)
    pub design_temp: f64,
}

impl EfficiencyCurve {
    /// Calculate COP at given PLR and outdoor temperature
    pub fn cop_at(&self, plr: f64, outdoor_temp: f64) -> f64 {
        // PLR contribution: cubic polynomial
        let plr_cop = self.plr_coefficients[0]
            + self.plr_coefficients[1] * plr
            + self.plr_coefficients[2] * plr.powi(2)
            + self.plr_coefficients[3] * plr.powi(3);

        // Temperature degradation: linear from design temp
        let temp_diff = (self.design_temp - outdoor_temp).abs();
        let temp_factor = 1.0 - self.temp_coefficient * temp_diff;

        plr_cop * temp_factor.max(0.3) // Minimum 30% of rated COP
    }
}
```

### Pattern 3: Predictive Control with Thermal Inertia

**What:** Control signal considers thermal mass state, not just current zone temperature

**When to use:** Variable capacity modulation control in solve_timesteps loop

**Example:**
```rust
/// Predictive HVAC control using thermal inertia
pub struct PredictiveController {
    heating_setpoint: f64,
    cooling_setpoint: f64,
    thermal_inertia_gain: f64,  // Tuning parameter (α)
    temp_rate_gain: f64,         // Tuning parameter (β)
}

impl PredictiveController {
    /// Calculate control signal (0-1 modulation factor)
    /// Uses thermal mass temperature and rate of change to smooth response
    pub fn calculate_modulation(
        &self,
        zone_temp: f64,
        mass_temp: f64,
        temp_rate: f64,  // dT/dt
    ) -> (HVACMode, f64) {
        // Inertia factor based on mass temperature offset
        let inertia_factor = self.thermal_inertia_gain * (zone_temp - mass_temp);

        // Predictive factor based on temperature rate
        let predictive_factor = self.temp_rate_gain * temp_rate;

        // Effective setpoint adjusted by inertia and prediction
        let effective_heating_sp = self.heating_setpoint + inertia_factor - predictive_factor;
        let effective_cooling_sp = self.cooling_setpoint + inertia_factor - predictive_factor;

        let mode = if zone_temp < effective_heating_sp {
            HVACMode::Heating
        } else if zone_temp > effective_cooling_sp {
            HVACMode::Cooling
        } else {
            HVACMode::Off
        };

        // Modulation factor based on temperature error
        let temp_error = match mode {
            HVACMode::Heating => effective_heating_sp - zone_temp,
            HVACMode::Cooling => zone_temp - effective_cooling_sp,
            HVACMode::Off => 0.0,
        };

        let modulation = (temp_error * 10.0).clamp(0.0, 1.0); // Tunable sensitivity
        (mode, modulation)
    }
}
```

### Pattern 4: Cycling Loss Tracking

**What:** Track equipment state, runtime hours, startup events, and apply penalties

**When to use:** All equipment models in solve_timesteps energy calculation

**Example:**
```rust
/// Cycling loss tracking for equipment
#[derive(Debug, Clone)]
pub struct CyclingTracker {
    /// Equipment state from previous timestep
    pub was_on: bool,
    /// Cumulative runtime hours (for annual validation)
    pub cumulative_runtime_hours: f64,
    /// Startup count (for penalty calculation)
    pub startup_count: u32,
    /// Minimum runtime in timesteps (e.g., 5 minutes = 5 timesteps)
    pub minimum_runtime_timesteps: u32,
    /// Current runtime since last startup (timesteps)
    pub current_runtime_timesteps: u32,
    /// Energy penalty per startup (kWh)
    pub startup_penalty_kwh: f64,
    /// PLR degradation factor (e.g., 0.2 for +20% at 0% PLR)
    pub plr_degradation_factor: f64,
}

impl CyclingTracker {
    /// Calculate cycling loss for current timestep
    pub fn calculate_cycling_loss(
        &mut self,
        is_on: bool,
        plr: f64,
    ) -> f64 {
        let mut loss_kwh = 0.0;

        // Detect startup event
        if is_on && !self.was_on {
            self.startup_count += 1;
            self.current_runtime_timesteps = 0;
            // Apply startup penalty
            loss_kwh += self.startup_penalty_kwh;
        }

        // Update state
        self.was_on = is_on;
        if is_on {
            self.current_runtime_timesteps += 1;
            self.cumulative_runtime_hours += 1.0 / 3600.0; // 1 timestep = 1 hour
        }

        // Check minimum runtime constraint
        let must_run = is_on && self.current_runtime_timesteps < self.minimum_runtime_timesteps;

        // PLR degradation: efficiency penalty at low PLR
        // Example: At PLR=0.3, degradation=0.2 → multiplier = 1.0 + 0.2 * 0.7 = 1.14
        if is_on && !must_run {
            let plr_penalty = self.plr_degradation_factor * (1.0 - plr);
            // Return efficiency multiplier (not energy directly)
            // Power calculation will multiply by this
            1.0 + plr_penalty
        } else {
            1.0 // No degradation penalty
        }
    }
}
```

### Anti-Patterns to Avoid

- **Nested parallelism in equipment calculations**: Do NOT use `rayon::par_iter()` inside `calculate_power()` or efficiency methods. The BatchOracle pre-commit hook enforces single-level parallelism. Equipment calculations run per-configuration in parallel, not internally.

- **Hardcoded efficiency values**: Use coefficient structs that can be loaded from configuration files. This enables AHRI reference data integration without code changes.

- **Ignoring thermal mass state**: The Phase 14 thermal mass corrections provide `mass_temperatures` in ThermalModel. Use these for predictive control, not just `temperatures` (zone air temps).

- **Simple setpoint hysteresis**: Don't use only zone temperature + deadband for control. Incorporate thermal inertia (mass temp) and temperature rate (dT/dt) for smoother, more realistic control.

- **Separate calculation of startup penalty and PLR degradation**: Combine both in the cycling loss calculation as specified in CONTEXT.md decision.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Polynomial evaluation | Manual coefficient multiplication | Implement generic `evaluate_polynomial()` method in efficiency_curves.rs | Avoids code duplication, enables coefficient array reuse |
| Efficiency curve fitting | Manual least-squares implementation | Use nalgebra or existing linear algebra (faer already in dependencies) | Robust, tested numerical methods |
| 2D surface interpolation | Custom bilinear implementation | Evaluate separate 1D curves for PLR and temperature, multiply results | Simpler, faster, maintains separability of variables |
| Configuration parsing | Manual JSON parsing | Use serde_json (already dependency) | Type-safe, handles errors automatically |
| Floating point comparisons | Manual epsilon checks | Use `approx` crate (already dev-dependency) | Handles relative/absolute tolerance correctly |

**Key insight:** Custom implementations of numerical methods (curve fitting, polynomial evaluation, surface interpolation) are error-prone and hard to maintain. Use existing, tested libraries wherever possible.

## Common Pitfalls

### Pitfall 1: Inefficient Polynomial Evaluation

**What goes wrong:** Repeatedly calculating `plr.powi(2)` and `plr.powi(3)` in hot loops creates unnecessary multiplication operations.

**Why it happens:** Naive implementation evaluates each term separately: `a + b*plr + c*plr*plr + d*plr*plr*plr`

**How to avoid:** Use Horner's method for polynomial evaluation: `((d*plr + c)*plr + b)*plr + a` - only 3 multiplications regardless of degree.

**Warning signs:** Profiling shows high CPU time in efficiency curve calculations.

### Pitfall 2: Ignoring Part-Load Ratio Bounds

**What goes wrong:** PLR calculated as `load / rated_capacity` can exceed 1.0 or go below 0.0 due to load spikes or capacity degradation at extreme temperatures.

**Why it happens:** Equipment capacity degrades with temperature (heat pumps especially), but load calculation doesn't account for this.

**How to avoid:** Clamp PLR to [0.0, 1.0] after calculation: `let plr = (load / actual_capacity).clamp(0.0, 1.0)`

**Warning signs:** COP calculations returning NaN or extreme values (negative, >10).

### Pitfall 3: Incorrect Thermal Inertia Tuning

**What goes wrong:** Control signal oscillates or responds too slowly if thermal inertia gain factors are not tuned properly.

**Why it happens:** Thermal inertia gain (α) and temperature rate gain (β) are equipment-specific and building-specific. Wrong values cause instability.

**How to avoid:** Tune against ASHRAE Cases 800-810 using ASHRAE Guideline 14 stability criteria. Start with α=0.1, β=0.01 and adjust.

**Warning signs:** Control signal oscillates between heating and cooling rapidly, or zone temp overshoots setpoint significantly.

### Pitfall 4: Forgetting Minimum Runtime Enforcement

**What goes wrong:** Equipment cycles on/off every timestep (every hour) creating unrealistic cycling losses.

**Why it happens:** Control signal doesn't track equipment state or enforce minimum runtime after startup.

**How to avoid:** Use CyclingTracker with `minimum_runtime_timesteps` field. After startup, maintain "must_run" flag until runtime threshold satisfied.

**Warning signs:** Annual startup count equals annual cooling/heating hours (should be 10-100x lower).

### Pitfall 5: Mixing Up Heating/Cooling Efficiency Curves

**What goes wrong:** Using heating COP curve for cooling mode or vice versa, resulting in incorrect efficiency values.

**Why it happens:** Heat pumps have separate heating and cooling performance curves with different coefficients. Easy to confuse them.

**How to avoid:** Store curves in separate fields: `heating_curve: EfficiencyCurve`, `cooling_curve: EfficiencyCurve`. Pass `mode: HVACMode` to efficiency calculation.

**Warning signs:** Cooling power higher than heating power in summer, or vice versa in winter.

## Code Examples

### Example 1: VariableCapacityEquipment Implementation for HeatPump

```rust
// Source: Context decision to enhance existing HeatPump struct
impl VariableCapacityEquipment for HeatPump {
    fn calculate_capacity(&self, plr: f64, outdoor_temp: f64) -> f64 {
        let base_capacity = match self.mode {
            HeatPumpMode::Heating => self.heating_capacity,
            HeatPumpMode::Cooling => self.cooling_capacity,
            HeatPumpMode::Off => 0.0,
        };

        // Capacity degrades with temperature
        let temp_diff = match self.mode {
            HeatPumpMode::Heating => (self.design_temp_heating - outdoor_temp).abs(),
            HeatPumpMode::Cooling => (outdoor_temp - self.design_temp_cooling).abs(),
            HeatPumpMode::Off => 0.0,
        };

        let capacity_factor = 1.0 - (temp_diff * 0.01); // 1% per degree
        base_capacity * capacity_factor.max(0.3) * plr
    }

    fn calculate_efficiency(&self, plr: f64, outdoor_temp: f64, mode: HVACMode) -> f64 {
        match mode {
            HVACMode::Heating => self.efficiency_curve_heating.cop_at(plr, outdoor_temp),
            HVACMode::Cooling => self.efficiency_curve_cooling.cop_at(plr, outdoor_temp),
            HVACMode::Off => 0.0,
        }
    }

    fn calculate_power(&self, load: f64, outdoor_temp: f64, mode: HVACMode) -> f64 {
        let capacity = self.rated_capacity();
        let plr = (load / capacity).clamp(0.0, 1.0);
        let efficiency = self.calculate_efficiency(plr, outdoor_temp, mode);
        if efficiency > 0.0 {
            load / efficiency
        } else {
            0.0
        }
    }

    fn rated_capacity(&self) -> f64 {
        self.heating_capacity.max(self.cooling_capacity)
    }

    fn rated_efficiency(&self, mode: HVACMode) -> f64 {
        match mode {
            HVACMode::Heating => self.heating_cop,
            HVACMode::Cooling => self.cooling_cop,
            HVACMode::Off => 0.0,
        }
    }

    fn current_plr(&self) -> f64 {
        // Track internally in struct
        self.current_plr
    }

    fn update_state(&mut self, current_load: f64, outdoor_temp: f64, mode: HVACMode) {
        let capacity = self.calculate_capacity(1.0, outdoor_temp);
        self.current_plr = if capacity > 0.0 {
            (current_load / capacity).clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.mode = mode;
    }
}
```

### Example 2: Integration in ThermalModel::step_physics

```rust
// Source: Existing HVAC demand calculation in engine.rs
// Enhance to use VariableCapacityEquipment and predictive control

fn step_physics_5r1c(&mut self, timestep: usize, outdoor_temp: f64) -> f64 {
    // ... existing code up to Ti_free calculation ...

    // Calculate temperature rate (dT/dt)
    let dt = 3600.0; // 1 hour
    let temp_rate = if timestep > 0 {
        (self.temperatures.as_ref()[0] - self.previous_temperatures.as_ref()[0]) / dt
    } else {
        0.0
    };

    // Predictive control using thermal inertia
    let (hvac_mode, modulation) = self.predictive_controller.calculate_modulation(
        self.temperatures.as_ref()[0],
        self.mass_temperatures.as_ref()[0], // Phase 14: thermal mass state available
        temp_rate,
    );

    // Calculate HVAC demand using variable capacity equipment
    let hvac_power = if let Some(ref equipment) = self.hvac_equipment {
        // Free-floating temp without HVAC
        let ti_free = /* ... existing calculation ... */;

        // Calculate required load
        let required_load = match hvac_mode {
            HVACMode::Heating => {
                let temp_deficit = self.hvac_controller.heating_setpoint - ti_free;
                (temp_deficit / self.sensitivity.as_ref()[0]).max(0.0)
            }
            HVACMode::Cooling => {
                let temp_excess = ti_free - self.hvac_controller.cooling_setpoint;
                (temp_excess / self.sensitivity.as_ref()[0]).max(0.0)
            }
            HVACMode::Off => 0.0,
        };

        // Apply modulation (0-100% capacity)
        let modulated_load = required_load * modulation;

        // Calculate power with efficiency curve
        let power = equipment.calculate_power(modulated_load, outdoor_temp, hvac_mode);

        // Apply cycling losses
        let efficiency_multiplier = self.cycling_tracker.calculate_cycling_loss(
            power > 0.0,
            equipment.current_plr(),
        );

        power * efficiency_multiplier
    } else {
        0.0 // No equipment
    };

    // Apply HVAC to zone temperature
    // ... existing Ti_act calculation ...
}
```

### Example 3: AHRI Coefficient Loading from JSON

```rust
// Source: CONTEXT.md decision to use AHRI reference data
// Load efficiency curve coefficients from configuration file

use serde_json;
use std::fs;

/// Load AHRI efficiency curve coefficients from JSON file
pub fn load_ahri_coefficients(path: &str) -> Result<EfficiencyCurveConfig, anyhow::Error> {
    let content = fs::read_to_string(path)?;
    let config: EfficiencyCurveConfig = serde_json::from_str(&content)?;
    Ok(config)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyCurveConfig {
    /// Heat pump heating coefficients (cubic polynomial)
    pub heatpump_heating: CurveCoefficients,
    /// Heat pump cooling coefficients (cubic polynomial)
    pub heatpump_cooling: CurveCoefficients,
    /// Chiller coefficients (cubic polynomial)
    pub chiller: CurveCoefficients,
    /// Boiler coefficients (cubic polynomial)
    pub boiler: CurveCoefficients,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurveCoefficients {
    /// Cubic polynomial: a + b*PLR + c*PLR^2 + d*PLR^3
    pub plr: [f64; 4],
    /// Temperature coefficient (per degree from design)
    pub temp_coefficient: f64,
    /// Design outdoor temperature (°C)
    pub design_temp: f64,
}

// Example JSON config:
/*
{
  "heatpump_heating": {
    "plr": [3.5, -0.8, 0.5, -0.2],
    "temp_coefficient": 0.02,
    "design_temp": -5.0
  },
  "heatpump_cooling": {
    "plr": [3.0, -0.5, 0.3, -0.1],
    "temp_coefficient": 0.03,
    "design_temp": 35.0
  },
  ...
}
*/
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Simple linear COP degradation | Cubic polynomial efficiency curves | This phase (Phase 15) | More accurate representation of S-shaped efficiency curves, especially at low PLR |
| Fixed capacity equipment | Variable capacity modulation | This phase (Phase 15) | Realistic equipment behavior, enables predictive control strategies |
| Simple setpoint hysteresis | Predictive control with thermal inertia | This phase (Phase 15) | Smoother response, reduced cycling, better matches real HVAC systems |
| No cycling loss modeling | Startup penalty + minimum runtime | This phase (Phase 15) | More accurate annual energy consumption, prevents unrealistic cycling |

**Deprecated/outdated:**
- Linear COP degradation (2%/°C): Replaced by cubic polynomial curves for better accuracy at all PLR levels
- Single-stage on/off control: Replaced by variable capacity modulation for modern equipment (VAV, inverter compressors)
- Hardcoded efficiency values: Replaced by coefficient-based curves loaded from AHRI reference data

## Open Questions

1. **AHRI coefficient data availability**
   - What we know: CONTEXT.md specifies AHRI Standard 550/590 for chillers and 210/240 for heat pumps as coefficient sources
   - What's unclear: Where to obtain actual AHRI coefficient data (may be behind paywall or require manufacturer access)
   - Recommendation: Start with reasonable default coefficients based on typical equipment values; add configuration file loading for AHRI data when available; validate against ASHRAE 140 Cases 800-810 as intermediate target

2. **ASHRAE 140 Cases 800-810 specifications**
   - What we know: These cases specifically test HVAC equipment performance
   - What's unclear: Exact case specifications (load profiles, setpoints, equipment types) not yet in codebase
   - Recommendation: Research ASHRAE 140 standard documentation or EnergyPlus source code for case 800-810 definitions; implement based on reference when available

3. **Thermal inertia tuning values**
   - What we know: Need to tune α (thermal inertia gain) and β (temperature rate gain) against ASHRAE Cases 800-810
   - What's unclear: Starting values for α and β; stability criteria from ASHRAE Guideline 14
   - Recommendation: Start with α=0.1, β=0.01; perform parameter sweep against ASHRAE validation; tune to ±10% annual energy tolerance

4. **Minimum runtime duration**
   - What we know: AHRI provides guidance (5-15 minute range)
   - What's unclear: Specific value for each equipment type (heat pump vs chiller vs boiler)
   - Recommendation: Use 5 timesteps (5 hours in hourly simulation) as starting point; adjust based on AHRI reference data and cycling analysis

5. **Economizer mode enthalpy calculations**
   - What we know: Economizer mode requires Phase 16 psychrometrics for enthalpy calculations
   - What's unclear: Whether to implement economizer before Phase 16 or defer entirely
   - Recommendation: Defer economizer implementation to after Phase 16; create placeholder method that always returns false for "economizer_active" check

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | cargo test (Rust built-in) |
| Config file | None (uses Cargo.toml dev-dependencies) |
| Quick run command | `cargo test --package fluxion --lib sim::hvac -- --nocapture` |
| Full suite command | `cargo test --package fluxion --lib` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| HVAC-01 | VAV system modeling with variable capacity | unit | `cargo test test_vav_variable_capacity -- --nocapture` | ❌ Wave 0 |
| HVAC-02 | CAV system modeling with variable capacity | unit | `cargo test test_cav_variable_capacity -- --nocapture` | ❌ Wave 0 |
| HVAC-03 | Heat pump with polynomial efficiency curves | unit | `cargo test test_heatpump_efficiency_curves -- --nocapture` | ❌ Wave 0 |
| HVAC-04 | Chiller with part-load ratios | unit | `cargo test test_chiller_efficiency_curves -- --nocapture` | ❌ Wave 0 |
| HVAC-05 | Boiler with part-load ratios | unit | `cargo test test_boiler_efficiency_curves -- --nocapture` | ❌ Wave 0 |
| HVAC-06 | Economizer mode free cooling | integration | `cargo test test_economizer_mode -- --nocapture` | ❌ Wave 0 |
| HVAC-07 | Equipment efficiency curves | unit | `cargo test test_polynomial_efficiency_curves -- --nocapture` | ❌ Wave 0 |
| HVAC-08 | Cycling loss modeling | unit | `cargo test test_cycling_losses -- --nocapture` | ❌ Wave 0 |
| HVAC-09 | Configurable control strategies | unit | `cargo test test_predictive_control -- --nocapture` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test --package fluxion --lib sim::hvac -- --nocapture`
- **Per wave merge:** `cargo test --package fluxion --lib` (full HVAC module test suite)
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `src/sim/hvac/equipment.rs` - VariableCapacityEquipment trait, Chiller, Boiler structs
- [ ] `src/sim/hvac/efficiency_curves.rs` - Polynomial curve evaluation, coefficient structs
- [ ] `src/sim/hvac/control.rs` - Predictive control logic with thermal inertia
- [ ] `src/sim/hvac/cycling.rs` - CyclingTracker, startup penalties, minimum runtime
- [ ] `src/sim/hvac/tests/equipment_tests.rs` - Unit tests for VariableCapacityEquipment trait
- [ ] `src/sim/hvac/tests/efficiency_curve_tests.rs` - Unit tests for polynomial curves
- [ ] `src/sim/hvac/tests/control_tests.rs` - Unit tests for predictive control
- [ ] `src/sim/hvac/tests/cycling_tests.rs` - Unit tests for cycling losses
- [ ] `tests/ashrae_140_cases_800_810.rs` - Integration tests for ASHRAE 140 Cases 800-810
- [ ] Framework install: None needed (cargo test is built-in)
- [ ] AHRI coefficient data: Create `src/sim/hvac/ahri_coefficients.json` with default values

## Sources

### Primary (HIGH confidence)
- Existing codebase: `src/sim/hvac.rs` - VAVTerminal, CAVSystem, HeatPump structures with current implementation
- Existing codebase: `src/sim/engine.rs` - ThermalModel, step_physics, HVAC demand calculation
- Phase 14 thermal mass corrections: Available in `mass_temperatures` VectorField
- CONTEXT.md: Locked implementation decisions from /gsd:discuss-phase session

### Secondary (MEDIUM confidence)
- ASHRAE Standard 140: Validation framework and test cases 800-810 (mentioned in CONTEXT.md)
- AHRI Standards 550/590 and 210/240: Reference data for efficiency curve coefficients (mentioned in CONTEXT.md)
- ASHRAE Guideline 14: Stability criteria for control tuning (mentioned in CONTEXT.md)

### Tertiary (LOW confidence)
- Web search attempts: Unable to retrieve AHRI coefficient data or ASHRAE 140 Case 800-810 specifications due to search service limitations
- Specific AHRI coefficient values: Need to obtain from AHRI standards or manufacturer data (unknown access)
- Exact ASHRAE 140 Case 800-810 specifications: Need to obtain from standard documentation (unknown access)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All dependencies already present in Cargo.toml; no new crates needed
- Architecture: HIGH - Clear trait-based pattern established in existing hvac.rs; integration points identified
- Pitfalls: MEDIUM - Identified common HVAC modeling issues, but AHRI coefficient uncertainty remains
- Validation: MEDIUM - Test framework well-established, but ASHRAE Case 800-810 specifications unknown

**Research date:** 2026-03-13
**Valid until:** 2026-04-13 (30 days - stable domain, but AHRI coefficient availability may change)

**Research limitations:**
- Web search service unable to retrieve AHRI standard documents or ASHRAE 140 Case 800-810 specifications
- AHRI coefficient data availability unclear (may be paywalled or require manufacturer access)
- Specific thermal inertia tuning values unknown (will require parameter sweep against validation data)

**Next steps for planner:**
1. Create plans for Wave 0: VariableCapacityEquipment trait and efficiency curve infrastructure
2. Create plans for Wave 1: Enhance existing VAV, CAV, HeatPump with trait implementation
3. Create plans for Wave 2: Add Chiller and Boiler models
4. Create plans for Wave 3: Integrate predictive control and cycling losses into ThermalModel
5. Create plans for Wave 4: Implement ASHRAE 140 Case 800-810 validation tests
6. Plan for economizer mode after Phase 16 (psychrometrics dependency)
