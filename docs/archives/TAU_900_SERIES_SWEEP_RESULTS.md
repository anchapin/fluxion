# Case 900 (VeryHeavy) τ Sweep Results

**Date:** 2026-03-29
**Task:** Parametric study of thermal time constant (τ) for Case 900
**Status:** Complete

---

## Background

Case 900 uses a 6R2C thermal network model for high-mass construction (concrete). The τ override was extended to support 6R2C models by adjusting `h_tr_em` (envelope-to-exterior conductance) based on the target τ:

```
τ = C_envelope / h_tr_em  →  h_tr_em = C_envelope / τ
```

**Environment Variable:** `FLUXION_TARGET_TAU_VERYHEAVY` (default: 12.0 hours)

---

## Results

| τ (hours) | Heating (MWh) | Reference (MWh) | Heating % Error | Cooling (MWh) | Reference (MWh) | Cooling % Error |
|------------|-----------------|-------------------|------------------|-----------------|-------------------|-----------------|
| 10.0 | 29.06 | 1.17-2.04 (1.61) | +1704% | 0.60 | 2.13-3.67 (2.90) | -79% |
| 15.0 | 27.56 | 1.61 | +1611% | 0.64 | 2.90 | -78% |
| 20.0 | 26.40 | 1.61 | +1539% | 0.72 | 2.90 | -75% |
| 25.0 | 25.46 | 1.61 | +1481% | 0.81 | 2.90 | -72% |
| 30.0 | 24.69 | 1.61 | +1433% | 0.90 | 2.90 | -69% |

**Reference Ranges:**
- Heating: 1.17 - 2.04 MWh (midpoint: 1.61 MWh)
- Cooling: 2.13 - 3.67 MWh (midpoint: 2.90 MWh)

---

## Key Findings

### 1. τ Override is Working

The τ override correctly affects results:
- Higher τ → Lower heating demand (29.06 → 24.69 MWh)
- Higher τ → Higher cooling demand (0.60 → 0.90 MWh)
- Trend follows physics: longer τ = slower response = more heat storage

### 2. No τ Value Achieves Acceptable Accuracy

Even with τ = 30.0 hours (2.5x the default):
- **Heating error:** +1433% (reference: 1.61, actual: 24.69)
- **Cooling error:** -69% (reference: 2.90, actual: 0.90)

Both metrics fail to meet ASHRAE 140 acceptance criteria.

### 3. Fundamental Physics Issue Beyond τ

The pattern suggests a deeper issue with the 6R2C model:
- **Heating massively overpredicted:** Envelope is losing heat too fast
- **Cooling underpredicted:** Envelope is not gaining enough heat or storing too much

This asymmetry suggests the 6R2C thermal network may have incorrect conductance balance:
- `h_tr_em` (envelope-to-exterior) may be too high → excessive heat loss
- `h_tr_me` (envelope-to-internal) may need adjustment
- Internal mass fraction (75% envelope / 25% internal) may not be optimal

---

## Physics Interpretation

### For 6R2C Model

The envelope mass temperature update (simplified):
```
C_env × dT_env/dt = h_tr_em × (T_outdoor - T_env) + h_tr_ms × (T_s - T_env) + ...
```

The dominant heat flow is `h_tr_em × (T_outdoor - T_env)` (exterior to envelope).

Setting `h_tr_em = C_env / τ` controls how fast the envelope responds to outdoor changes:
- **Lower h_tr_em** → Slower response → More heat stored → Lower heating, Higher cooling
- **Higher h_tr_em** → Faster response → Less heat stored → Higher heating, Lower cooling

### Current Issue

With τ = 10-30 hours:
- Calculated `h_tr_em` ranges: 346-115 W/K
- This is still resulting in +1400% heating error

The envelope mass is responding too slowly to retain heat during cooling, but also losing heat too fast during heating - contradictory behavior suggesting network imbalance.

---

## Comparison with 600-Series (5R1C)

| Series | Model | Current τ | Heating Error | Cooling Error | Pass Rate |
|---------|--------|-------------|----------------|--------------|------------|
| 600-Series | 5R1C | 6.0h | +80% (Case 600) | -8% (Case 600) | 83% cooling |
| 900-Series | 6R2C | 12.0h default | +1500% (Case 900) | -72% (Case 900) | 0% |

The 6R2C model for high-mass cases performs significantly worse than 5R1C for low-mass.

---

## Recommendations

### Immediate (High Priority)

1. **Investigate 6R2C conductance balance:**
   - Verify `h_tr_me` (envelope-to-internal) value (currently fixed at 100 W/K)
   - Test different envelope/internal mass split ratios (currently 75%/25%)
   - Check if `h_tr_ms` needs adjustment in 6R2C context

2. **Consider 5R1C for 900-series:**
   - The 600-series (5R1C) achieved 83% cooling pass rate
   - Test if 5R1C with VeryHeavy τ works better for 900-series

3. **Verify other thermal parameters for Case 900:**
   - Internal gain values (200 W total)
   - Solar gain distribution
   - Surface heat transfer coefficients (`h_tr_is`, `h_tr_ms`)

### Medium Priority

4. **Systematic 6R2C parameter sweep:**
   - Test `h_tr_me` values: 10, 25, 50, 100, 200, 400 W/K
   - Test envelope mass fractions: 0.5, 0.6, 0.7, 0.8, 0.9
   - Identify optimal combination

### Long-Term

5. **Investigate ASHRAE 140 reference implementations:**
   - Compare with EnergyPlus or other validated tools
   - Identify differences in thermal network structure
   - Document best practices for 6R2C implementation

---

## Code Changes Made

### 1. Extended τ Override to 6R2C Model

**File:** `src/sim/engine.rs` (lines 888-920)

**Change:** Added τ override for 6R2C high-mass cases (900-series)

```rust
if spec.case_id.starts_with('9') {
    model.configure_6r2c_model(0.75, 100.0);

    // Apply τ override to envelope capacitance
    let target_tau_hours = if is_veryheavy {
        std::env::var("FLUXION_TARGET_TAU_VERYHEAVY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(12.0)
    } else {
        12.0
    };

    let required_h_tr_em = envelope_cap / (target_tau_hours * 3600.0);
    model.h_tr_em = VectorField::from_scalar(required_h_tr_em, num_zones);
}
```

**Note:** Used `case_id.starts_with('9')` instead of mass class detection because wall-based mass classification may not accurately reflect total building mass.

---

## Testing Commands

```bash
# Test different τ values for Case 900
for tau in 10.0 15.0 20.0 25.0 30.0; do
    FLUXION_TARGET_TAU_VERYHEAVY=$tau cargo run --release --bin fluxion validate --case 900
done

# Test default value (12.0 hours)
cargo run --release --bin fluxion validate --case 900
```

---

## Conclusion

The τ override infrastructure is now functional for 6R2C models, enabling systematic parametric studies. However, τ tuning alone cannot resolve the fundamental 6R2C modeling issues for Case 900.

**Next Steps:**
1. Investigate 6R2C conductance balance (`h_tr_me`, envelope/internal mass split)
2. Test 5R1C model for 900-series as alternative approach
3. Verify internal gain and solar parameter values

---

**Report Complete.**
