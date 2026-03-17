# Adaptive Timestep Theory for High-Mass Buildings

**Date:** 2026-03-17
**Author:** Fluxion Development Team
**Phase:** 25-02 (Alternative Physics Implementation)

---

## Executive Summary

This document presents the theoretical foundation for adaptive timestep integration in building energy simulation, specifically targeting high-mass buildings where the standard 1-hour timestep leads to numerical accuracy issues. We analyze thermal mass time constants, derive stability criteria, and provide recommendations for timestep selection.

**Key Finding:** High-mass buildings (Case 900 series) have thermal time constants τ ≈ 4-5 hours, requiring Δt ≤ 6 minutes for numerical stability and accuracy. Low-mass buildings (Case 600 series) have τ ≈ 0.5-1 hour and can use Δt = 1 hour safely.

---

## 1. Thermal Mass Time Constant Analysis

### 1.1 Definition

The thermal time constant τ characterizes how quickly a building's thermal mass responds to temperature changes:

```
τ = C / (h_tr_ms + h_tr_em)
```

Where:
- **C** = Thermal capacitance of the building (J/K)
- **h_tr_ms** = Heat transfer coefficient to mass surfaces (W/K)
- **h_tr_em** = Heat transfer coefficient to external mass (W/K)

### 1.2 Physical Interpretation

The time constant represents:
- **63.2% response time:** Time to reach 63.2% of final temperature after a step change
- **Numerical stability limit:** Maximum timestep for stable explicit integration
- **Physical resolution:** Minimum timestep to capture thermal dynamics

### 1.3 Time Constants for ASHRAE 140 Cases

| Case | Building Type | C (J/K) | h_tr_ms + h_tr_em (W/K) | τ (hours) |
|------|--------------|---------|------------------------|-----------|
| 600 | Low-mass office | 2.4×10⁶ | 800 | 0.83 |
| 650 | Low-mass + internal mass | 3.5×10⁶ | 900 | 1.08 |
| 900 | High-mass office | 1.2×10⁷ | 650 | 5.13 |
| 920 | High-mass + increased mass | 1.8×10⁷ | 700 | 7.14 |
| 930 | High-mass + high U-value | 1.2×10⁷ | 1200 | 2.78 |
| 940 | High-mass + low U-value | 1.2×10⁷ | 400 | 8.33 |
| 960 | High-mass + direct gain | 1.2×10⁷ | 800 | 4.17 |

**Note:** Values are approximate, calculated from ISO 13790 5R1C network parameters.

---

## 2. Numerical Stability Analysis

### 2.1 Explicit Euler Stability Criterion

For explicit Euler integration of the RC network:

```
T(t+Δt) = T(t) + (Δt/C) × Σ(Q_in - Q_out)
```

**Stability requires:**

```
Δt < 2C / Σh = 2τ
```

Where Σh is the sum of all conductances connected to the node.

### 2.2 Accuracy Criterion

For accurate transient response (not just stability):

```
Δt < τ / 10
```

This ensures:
- **<5% amplitude error** for sinusoidal inputs
- **<1° phase error** for diurnal cycles
- **Convergence** to analytical solution

### 2.3 CFL-like Condition for RC Networks

The Courant-Friedrichs-Lewy (CFL) condition adapted for thermal RC networks:

```
Fo = α × Δt / Δx² < Fo_critical
```

Where:
- **Fo** = Fourier number (dimensionless timestep)
- **α** = Thermal diffusivity (m²/s)
- **Δx** = Characteristic length (wall thickness / nodes)
- **Fo_critical** ≈ 0.5 for explicit schemes

For implicit schemes (unconditionally stable):
- **Accuracy still requires:** Δt < τ / 10

---

## 3. Adaptive Timestep Strategy

### 3.1 Mode Selection Logic

```
if τ > τ_threshold:
    use_adaptive_timestep = True
    base_dt = min(Δt_min, τ / 10)
else:
    use_adaptive_timestep = False
    base_dt = 1 hour
```

**Recommended threshold:** τ_threshold = 2 hours

### 3.2 Timestep Selection Table

| Time Constant τ | Recommended Δt | Timesteps/Hour | Use Case |
|-----------------|----------------|----------------|----------|
| τ < 1 hour | 60 min | 1 | Low-mass buildings |
| 1 ≤ τ < 2 hours | 30 min | 2 | Medium-mass buildings |
| 2 ≤ τ < 4 hours | 15 min | 4 | Moderate high-mass |
| 4 ≤ τ < 6 hours | 6 min | 10 | High-mass (Case 900) |
| τ ≥ 6 hours | 3 min | 20 | Very high-mass |

### 3.3 Diurnal Adaptation (Optional)

For additional efficiency, adapt timestep based on time of day:

| Time Period | Solar Variability | Recommended Δt |
|-------------|-------------------|----------------|
| Night (22:00-06:00) | Low | 15-30 min |
| Morning (06:00-09:00) | High (sunrise) | 6 min |
| Day (09:00-16:00) | Moderate | 10-15 min |
| Evening (16:00-22:00) | High (sunset) | 6 min |

---

## 4. Expected Accuracy Improvement

### 4.1 Current Baseline (1-hour timestep)

**Case 900 Annual Energy:**
- Heating: 5.35 MWh (262-322% above reference)
- Cooling: 4.75 MWh (29-123% above reference)

### 4.2 Predicted Improvement with Adaptive Timestep

**Hypothesis:** Timestep error contributes ~20-30% of total error; remaining 50-100% is fundamental 5R1C structural limitation.

**Prediction for Case 900 (6-minute timestep):**
- Heating: 2.5-3.5 MWh (70-120% above reference, improved from 262-322%)
- Cooling: 3.0-4.0 MWh (15-50% above reference, improved from 29-123%)

**Rationale:**
1. **Better resolution of thermal lag:** High-mass walls store/release heat over hours
2. **Accurate solar gain integration:** Rapid changes at sunrise/sunset
3. **Stable numerical integration:** Δt/τ ≈ 0.02 < 0.1 threshold

### 4.3 Limitations

Adaptive timestep **will not** achieve ±15% target because:
1. **5R1C structural limitation:** Lumped capacitance cannot capture temperature gradients through mass
2. **Missing physics:** No spatial resolution within wall layers
3. **Approximate coupling:** Surface-to-core heat transfer simplified

**Conclusion:** Adaptive timestep is an **interim improvement**, not a final solution. CTF or Finite Difference methods (Plans 25-03, 25-04) are required for ±15% accuracy.

---

## 5. Performance Impact

### 5.1 Timestep Multiplication Factor

| Base Δt | Adaptive Δt | Multiplication Factor |
|---------|-------------|----------------------|
| 60 min | 6 min | 10× |
| 60 min | 15 min | 4× |
| 60 min | 30 min | 2× |

### 5.2 Throughput Prediction

**Baseline (1-hour timestep):** ~2,575 configs/sec

**Adaptive timestep (6-minute):**
- 10× more timesteps
- Fixed overhead (I/O, setup) amortized
- **Expected throughput:** ~400-600 configs/sec (4-5× slowdown, not 10×)

### 5.3 Memory Impact

- **Minimal increase:** Same state variables, just more frequent updates
- **History buffers:** May need to store more timesteps for output (optional)

---

## 6. Implementation Requirements

### 6.1 Configuration API

```rust
pub enum TimestepMode {
    Fixed { dt: Duration },
    Adaptive {
        base_dt: Duration,      // e.g., 6 minutes
        min_dt: Duration,       // e.g., 1 minute
        threshold_tau: f64,     // e.g., 2.0 hours
    },
}
```

### 6.2 Scheduler Interface

```rust
pub struct AdaptiveTimestepScheduler {
    mode: TimestepMode,
    building_tau: f64,
}

impl AdaptiveTimestepScheduler {
    pub fn schedule_simulation(&self, total_hours: usize) -> Vec<Duration>;
    pub fn get_timestep(&self, hour: usize) -> Duration;
}
```

### 6.3 Thermal Model Changes

- `step_physics(dt: Duration)` — accept variable timestep
- Update mass node equations: `T(t+Δt) = f(T(t), Δt)`
- Ensure numerical stability for any Δt < Δt_max

### 6.4 HVAC Integration

- Energy accumulation: `E = Σ(P × Δt)` with variable Δt
- Cycling logic: adapt minimum runtime to timestep
- Control decisions: maintain same logic, just more frequent evaluation

---

## 7. Validation Plan

### 7.1 Unit Tests

- [ ] Time constant calculation for all ASHRAE 140 cases
- [ ] Scheduler returns correct timestep sequence
- [ ] Stability check: Δt < 2τ
- [ ] Accuracy check: Δt < τ/10

### 7.2 Integration Tests

- [ ] Case 900 with 6-minute timestep
- [ ] Case 600 with 1-hour timestep (no regression)
- [ ] Mixed cases: verify mode selection logic

### 7.3 Validation Metrics

- **Annual energy:** Heating, cooling (MWh)
- **Monthly energy:** 12 months × cases
- **Hourly profiles:** Zone temperature, HVAC power
- **Performance:** configs/sec, memory usage

---

## 8. Recommendations

### 8.1 Immediate Actions (Phase 25-02)

1. **Implement adaptive timestep scheduler** with threshold τ = 2 hours
2. **Use 6-minute timestep** for high-mass buildings (τ > 2 hours)
3. **Maintain 1-hour timestep** for low-mass buildings (τ < 2 hours)
4. **Validate** on all 18 ASHRAE 140 cases

### 8.2 Future Work

1. **Combine with ML correction** (Plan 25-05) for remaining error
2. **Evaluate CTF/Finite Difference** (Plans 25-03, 25-04) for ±15% target
3. **Consider implicit integration** for unconditional stability

### 8.3 Decision Gate

After Phase 25-02 completion:
- **If accuracy improved to <100% error:** Proceed to Phase 25-05 (ML correction)
- **If accuracy still >100% error:** Prioritize Phase 25-03/25-04 (CTF/FD)

---

## 9. References

1. ISO 13790:2008 — Energy performance of buildings — Calculation of energy use for heating and cooling
2. ASHRAE Standard 140-2023 — Standard Method of Test for the Evaluation of Building Energy Analysis Computer Programs
3. Henninger, R.H., & Witte, M.J. (2023). ASHRAE 140-2023 Standard Development Report
4. Trčka, M., & Hensen, J.L.M. (2010). Overview of building energy simulation tools. IEA ECBCS Annex 43
5. Clarke, J.A. (2001). Energy Simulation in Building Design. Butterworth-Heinemann.

---

*Document created: 2026-03-17 for Phase 25 Alternative Physics Implementation*
