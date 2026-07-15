# Modular Surrogate vs Monolithic Benchmark
Generated: 2026-07-15T17:15:59.831371

## Summary

| Case | Modular H/C Acc | Monolithic H/C Acc | Disagreement | Recommendation |
|------|----------------|-------------------|--------------|----------------|
| 600 | H:69%/C:0% | H:62%/C:0% | 47.4% | pursue PINN composition |
| 900 | H:54%/C:0% | H:-237%/C:0% | 77.6% | pursue PINN composition |
| 960 | H:-37%/C:27% | H:-271%/C:33% | 43.3% | pursue PINN composition |

## Detailed Results

### Case 600

**Modular (modular):**
- Annual heating accuracy: 69.07%
- Annual cooling accuracy: 0.00%
- Per-timestep MAE heating: 376.62 W
- Per-timestep MAE cooling: 574.18 W
- Within 5% tolerance: False

**Monolithic:**
- Annual heating accuracy: 61.91%
- Annual cooling accuracy: 0.00%
- Per-timestep MAE heating: 385.15 W
- Per-timestep MAE cooling: 574.18 W
- Within 5% tolerance: False

**Disagreement:** 47.40%
**Recommendation:** pursue PINN composition

---

### Case 900

**Modular (modular):**
- Annual heating accuracy: 54.41%
- Annual cooling accuracy: 0.01%
- Per-timestep MAE heating: 124.72 W
- Per-timestep MAE cooling: 331.02 W
- Within 5% tolerance: False

**Monolithic:**
- Annual heating accuracy: -236.63%
- Annual cooling accuracy: 0.00%
- Per-timestep MAE heating: 616.77 W
- Per-timestep MAE cooling: 331.03 W
- Within 5% tolerance: False

**Disagreement:** 77.57%
**Recommendation:** pursue PINN composition

---

### Case 960

**Modular (modular):**
- Annual heating accuracy: -36.80%
- Annual cooling accuracy: 27.15%
- Per-timestep MAE heating: 386.54 W
- Per-timestep MAE cooling: 160.55 W
- Within 5% tolerance: False

**Monolithic:**
- Annual heating accuracy: -271.30%
- Annual cooling accuracy: 33.30%
- Per-timestep MAE heating: 708.47 W
- Per-timestep MAE cooling: 162.13 W
- Within 5% tolerance: False

**Disagreement:** 43.29%
**Recommendation:** pursue PINN composition

---

## Conclusion

Some cases show accuracy outside 5% tolerance. Further investigation needed
to determine if physics-informed (PINN) composition would improve accuracy.