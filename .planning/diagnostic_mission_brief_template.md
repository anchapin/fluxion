# Diagnostic Mission Brief

> Structured template for autonomous ASHRAE 140 discrepancy investigation.
> Fill in each section before launching a diagnostic agent sweep.

---

## 1. Test Case Identification

### Target Case
- **Case ID:** `________`
- **ASHRAE 140 Series:** [ ] 600 [ ] 900 [ ] 800 [ ] 195/470 [ ] Other: ___
- **Variant:** [ ] Standard [ ] Free-Floating (FF) [ ] Setback [ ] Other: ___

### Failing Metric(s)
| Metric | Simulated Value | Reference Range | Absolute Error | Relative Error |
|-------|----------------|-----------------|----------------|----------------|
| Heating Energy | | | | |
| Cooling Energy | | | | |
| Peak Heating Load | | | | |
| Peak Cooling Load | | | |
| Free-Float Temperature | | | | |

### Pass/Fail Status
- [ ] **PASS** — All metrics within tolerance
- [ ] **FAIL** — One or more metrics outside tolerance
- [ ] **PARTIAL** — Some metrics pass, others fail

---

## 2. Error Margin Specification

### Tolerance Bands (per ASHRAE 140 / project standard)

| Metric Type | Absolute Tolerance | Relative Tolerance |
|------------|-------------------|-------------------|
| Annual Energy | ±15% of reference mean | ±___% |
| Monthly Energy | ±10% of reference mean | ±___% |
| Peak Loads | ±15% of reference mean | ±___% |
| Free-Floating Temperature | ±1.0°C | ±___°C |
| Hourly Temperature | ±0.5°C | ±___°C |

### Target Error Margin for This Brief
- **Primary metric:** ________________
- **Acceptable error:** ±___% or ±___°C
- **Minimum improvement threshold:** Reduce MAE from ___% to ≤___%

---

## 3. Parameter Manipulation Boundaries

### Parameters the Agent MAY Manipulate

| Parameter | Symbol | Default Value | Search Range | Step Size |
|-----------|--------|---------------|-------------|-----------|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

### Parameters the Agent MAY NOT Manipulate
- [ ] Case ID or case-type hints
- [ ] Reference benchmark values
- [ ] Correction factors calibrated against ASHRAE 140 results
- [ ] Weather data (EPW)
- [ ] Building geometry from ASHRAE 140 specification

### Constraints
1. All parameter changes must be physics-based (first principles)
2. No case-specific tuning — parameters must generalize across cases
3. Changes must be traceable to a specific module in `ARCHITECTURE.md`

---

## 4. Module Under Investigation

- [ ] Weather (`src/weather/`)
- [ ] Solar Position & Irradiance (`src/sim/solar.rs`)
- [ ] Conduction (`src/physics/solver_trait.rs`)
- [ ] Ventilation (`src/sim/ventilation.rs`)
- [ ] Zone Balance (`src/sim/thermal_model.rs`)
- [ ] Surface Heat Flux (`src/sim/surface_flux_provider.rs`)
- [ ] Other: ________________

### Suspected Root Cause
_Describe the suspected physics error in 2-3 sentences._

---

## 5. Sweep Configuration

### Sweep Type
- [ ] **Grid Search** — Full factorial parameter combination
- [ ] **Random Search** — N random samples within bounds
- [ ] **Gradient Descent** — Optimize continuous parameter
- [ ] **Binary Search** — Find boundary of acceptable performance
- [ ] **Latin Hypercube** — Space-filling sampling

### Sweep Parameters
- **Max iterations:** _____
- **Samples per parameter:** _____
- **Concurrent runs:** _____
- **Timeout per run:** _______ seconds

### Overnight Execution
- [ ] **Enabled** — Run continuously until completion or timeout
- [ ] **Disabled** — Run only during business hours

---

## 6. Logging & Output

### Trace Output Directory
```
.sdd/traces/diagnostic/{case_id}_{timestamp}/
```

### Files to Generate
- [ ] `sweep_config.json` — This brief serialized
- [ ] `parameter_sweep_results.jsonl` — One JSON object per parameter combination
- [ ] `convergence_log.csv` — MAE vs. iteration for each metric
- [ ] `best_parameters.json` — Optimal parameter set found
- [ ] `divergence_report.md` — Human-readable summary

### Metrics to Log Per Run
```json
{
  "case_id": "",
  "run_id": "",
  "parameters": {},
  "heating_mae": 0.0,
  "cooling_mae": 0.0,
  "peak_heating_mae": 0.0,
  "peak_cooling_mae": 0.0,
  "temperature_mae": 0.0,
  "overall_pass": true,
  "timestamp": ""
}
```

---

## 7. Success Criteria

### Minimum Success
- [ ] MAE for primary metric reduced to ≤___%
- [ ] At least ___% of metrics pass tolerance bands
- [ ] No regression in previously passing metrics

### Full Success
- [ ] All metrics within tolerance bands
- [ ] Parameters are physically interpretable
- [ ] No case-specific corrections required
- [ ] Results reproducible across multiple runs

### Abandon Criteria
- [ ] MAE does not improve after ___ iterations
- [ ] Best achievable MAE > ___% (indicates wrong parameter space)
- [ ] Sweep exceeds ___ hours without meaningful progress

---

## 8. Pre-Sweep Checklist

### Environment
- [ ] `cargo build --release` succeeds
- [ ] `cargo test --test ashrae_140_validation -- --nocapture` runs without panic
- [ ] Reference data files present in `tests/reference_data/`
- [ ] `.sdd/traces/` directory exists and is writable

### Agent Authorization
- [ ] Agent has read access to all source files
- [ ] Agent has write access to `.sdd/traces/diagnostic/`
- [ ] Agent may execute `cargo test` commands
- [ ] Agent may NOT commit/push changes directly

### Safety
- [ ] Parameter ranges verified to not cause numerical instability
- [ ] Timeout set to prevent infinite loops
- [ ] Crash recovery: log current state before exiting on error

---

## 9. Notes & Observations

_Pre-sweep observations about this case (e.g., which metrics fail, known bugs):_

_______________________________________________________________
_______________________________________________________________
_______________________________________________________________

---

## 10. Post-Sweep Sign-Off

### Results Summary
- **Best MAE achieved:** ___%
- **Best parameters found:** ________________
- **Recommendations:** ________________

### Human Review
- [ ] Reviewed by: ________________
- [ ] Date: ________________
- [ ] Approved for integration: [ ] Yes [ ] No

### Next Steps
- [ ] Implement fix based on best parameters
- [ ] Run full ASHRAE 140 validation suite
- [ ] Update ARCHITECTURE.md if module interfaces changed
- [ ] File new issue if root cause differs from suspected cause
